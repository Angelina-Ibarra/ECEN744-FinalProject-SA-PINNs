import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io
import math
import argparse
import os
import sys
import csv
import matplotlib.gridspec as gridspec
from plotting import newfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import layers, activations
from scipy.interpolate import griddata
from eager_lbfgs import lbfgs, Struct

SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    from pyDOE import lhs
except Exception:
    try:
        from pyDOE2 import lhs
    except Exception:
        def lhs(n, samples):
            result = np.empty((samples, n))
            for j in range(n):
                cut = np.linspace(0.0, 1.0, samples + 1)
                u = np.random.rand(samples)
                points = cut[:samples] + u * (cut[1:] - cut[:samples])
                np.random.shuffle(points)
                result[:, j] = points
            return result

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Optimizers.learnable_optimizer import LearnableOptimizer, reshape_to_model
from Optimizers.pinn_quasi_newton import run_quasi_newton_refinement

layer_sizes = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

sizes_w = []
sizes_b = []
for i, width in enumerate(layer_sizes):
    if i != 1:
        sizes_w.append(int(width * layer_sizes[1]))
        sizes_b.append(int(width if i != 0 else layer_sizes[1]))

def set_weights(model, w, sizes_w, sizes_b):
        for i, layer in enumerate(model.layers[0:]):
            start_weights = sum(sizes_w[:i]) + sum(sizes_b[:i])
            end_weights = sum(sizes_w[:i+1]) + sum(sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(sizes_w[i] / sizes_b[i])
            weights = tf.reshape(weights, [w_div, sizes_b[i]])
            biases = w[end_weights:end_weights + sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)



def get_weights(model):
        w = []
        for layer in model.layers[0:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)

        w = tf.convert_to_tensor(w)
        return w


def neural_net(layer_sizes):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))
    return model


def build_results_dir(tf_iter, newton_iter, optimizer_name, qn_method_bfgs=None):
    optimizer_folder = optimizer_name.capitalize()
    if optimizer_name == "quasi-newton" and qn_method_bfgs:
        optimizer_folder = f"Quasi-Newton_{qn_method_bfgs}"
    run_folder = f"{tf_iter}_tf_iter_{newton_iter}_newton_iter"
    results_dir = os.path.join(PROJECT_ROOT, "Results", "Burgers", optimizer_folder, run_folder)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def write_csv(path, header, rows):
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


u_model = neural_net(layer_sizes)

u_model.summary()


def loss(x_f_batch, t_f_batch,
             x0, t0, u0, x_lb,
             t_lb, x_ub, t_ub, col_weights, u_weights):

    f_u_pred = f_model(x_f_batch, t_f_batch)
    u0_pred = u_model(tf.concat([x0, t0],1))
    u_lb_pred, _ = u_x_model(x_lb, t_lb)
    u_ub_pred, _ = u_x_model(x_ub, t_ub)

    mse_0_u = tf.reduce_mean(tf.square(u_weights*(u0 - u0_pred)))

    mse_b_u = tf.reduce_mean(tf.square(u_lb_pred - 0)) + \
            tf.reduce_mean(tf.square(u_ub_pred - 0)) #since ub/lb is 0

    mse_f_u = tf.reduce_mean(tf.square(col_weights*f_u_pred))


    return  mse_0_u + mse_b_u + mse_f_u , mse_0_u, mse_f_u


@tf.function
def f_model(x,t):
    u = u_model(tf.concat([x,t], 1))
    u_x = tf.gradients(u,x)
    u_xx = tf.gradients(u_x, x)
    u_t = tf.gradients(u,t)
    f_u = u_t + u*u_x - (0.01/tf.constant(math.pi))*u_xx

    return f_u



@tf.function
def u_x_model(x,t):
    u = u_model(tf.concat([x,t],1))
    u_x = tf.gradients(u,x)
    return u,u_x

@tf.function
def grad(model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    with tf.GradientTape(persistent=True) as tape:
        #tape.watch(col_weights)
        #tape.watch(u_weights)
        loss_value, mse_0, mse_f = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)
        grads = tape.gradient(loss_value, u_model.trainable_variables)
        #print(grads)
        grads_col = tape.gradient(loss_value, col_weights)
        grads_u = tape.gradient(loss_value, u_weights)

    return loss_value, mse_0, mse_f, grads, grads_col, grads_u

def fit(x_f, t_f, x0, t0, u0, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights, tf_iter, newton_iter, optimizer_name="adam", qn_method="BFGS", qn_method_bfgs="SSBroyden2"):
    # Built in support for mini-batch, set to N_f (i.e. full batch) by default
    batch_sz = N_f
    n_batches =  N_f // batch_sz
    start_time = time.time()
    tf_optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.90)
    tf_optimizer_coll = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.90)
    tf_optimizer_u = tf.keras.optimizers.legacy.Adam(learning_rate=0.005, beta_1=0.90)
    learnable_optimizer = LearnableOptimizer(learning_rate=0.005)
    training_history = []

    phase1_name = "adam" if optimizer_name == "quasi-newton" else optimizer_name
    print(f"starting {phase1_name} training (phase 1)")

    for epoch in range(tf_iter):
        for i in range(n_batches):

            x0_batch = x0#[i*batch_sz:(i*batch_sz + batch_sz),]
            t0_batch = t0#[i*batch_sz:(i*batch_sz + batch_sz),]
            u0_batch = u0#[i*batch_sz:(i*batch_sz + batch_sz),]

            x_f_batch = x_f[i*batch_sz:(i*batch_sz + batch_sz),]
            t_f_batch = t_f[i*batch_sz:(i*batch_sz + batch_sz),]

            loss_value,mse_0, mse_f, grads, grads_col, grads_u = grad(u_model, x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)

            if optimizer_name == "learnable":
                weights_updated = learnable_optimizer.apply_gradients(
                    u_model.trainable_variables, grads, learnable_optimizer.optimizer
                )
                reshape_to_model(weights_updated, u_model)
            else:
                tf_optimizer.apply_gradients(zip(grads, u_model.trainable_variables))
            tf_optimizer_coll.apply_gradients(zip([-grads_col], [col_weights]))
            tf_optimizer_u.apply_gradients(zip([-grads_u], [u_weights]))


        if epoch % 10 == 0:
            elapsed = time.time() - start_time
            print('It: %d, Time: %.2f' % (epoch, elapsed))
            tf.print(f"mse_0: {mse_0}  mse_f: {mse_f}   total loss: {loss_value}")
            start_time = time.time()
        training_history.append(
            [epoch, float(loss_value.numpy()), float(mse_0.numpy()), float(mse_f.numpy())]
        )


    loss_and_flat_grad = get_loss_and_flat_grad(
        x_f_batch,
        t_f_batch,
        x0_batch,
        t0_batch,
        u0_batch,
        x_lb,
        t_lb,
        x_ub,
        t_ub,
        col_weights,
        u_weights,
    )

    if optimizer_name == "quasi-newton":
        print(f"Starting quasi-Newton (SciPy) refinement on network weights "
              f"[qn_method={qn_method}, qn_method_bfgs={qn_method_bfgs}]")
        lbfgs_history = run_quasi_newton_refinement(
            loss_and_flat_grad,
            get_weights(u_model),
            newton_iter,
            qn_method=qn_method,
            qn_method_bfgs=qn_method_bfgs,
        )
    else:
        print("Starting L-BFGS training")
        _, lbfgs_f_hist, _ = lbfgs(
            loss_and_flat_grad,
            get_weights(u_model),
            Struct(),
            maxIter=newton_iter,
            learningRate=0.8,
        )
        lbfgs_history = [
            [step, float(value.numpy())] for step, value in enumerate(lbfgs_f_hist)
        ]
    return training_history, lbfgs_history


# L-BFGS implementation from https://github.com/pierremtb/PINNs-TF2.0
def get_loss_and_flat_grad(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights):
    def loss_and_flat_grad(w):
        with tf.GradientTape() as tape:
            set_weights(u_model, w, sizes_w, sizes_b)
            loss_value, _, _ = loss(x_f_batch, t_f_batch, x0_batch, t0_batch, u0_batch, x_lb, t_lb, x_ub, t_ub, col_weights, u_weights)
        grad = tape.gradient(loss_value, u_model.trainable_variables)
        grad_flat = []
        for g in grad:
            grad_flat.append(tf.reshape(g, [-1]))
        grad_flat = tf.concat(grad_flat, 0)
        #print(loss_value, grad_flat)
        return loss_value, grad_flat

    return loss_and_flat_grad


def predict(X_star):
    X_star = tf.convert_to_tensor(X_star, dtype=tf.float32)
    u_star, _ = u_x_model(X_star[:,0:1],
                     X_star[:,1:2])

    f_u_star = f_model(X_star[:,0:1],
                 X_star[:,1:2])

    return u_star.numpy(), f_u_star.numpy()



lb = np.array([-1.0]) #x upper boundary
ub = np.array([1.0]) #x lower boundary

N0 = 100
N_b = 100 #100 per upper and lower boundary, so 200 total (matches paper's Nb = 200)
N_f = 10000

col_weights = tf.Variable(tf.reshape(tf.repeat(100.0, N_f),(N_f, -1)))
u_weights = tf.Variable(tf.random.uniform([N0, 1]))

#load data, from Raissi et. al
data = scipy.io.loadmat('burgers_shock.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = data['usol']
Exact_u = np.real(Exact)


#grab random points off the initial condition
idx_x = np.random.choice(x.shape[0], N0, replace=False)
x0 = x[idx_x,:]
u0 = tf.cast(Exact_u[idx_x,0:1], dtype = tf.float32)

idx_t = np.random.choice(t.shape[0], N_b, replace=False)
tb = t[idx_t,:]

# Sample collocation points via LHS with correct 2D bounds: x in [-1, 1], t in [0, 1]
lb_f = np.array([-1.0, 0.0])
ub_f = np.array([1.0, 1.0])
X_f = lb_f + (ub_f - lb_f) * lhs(2, N_f)

x_f = tf.convert_to_tensor(X_f[:,0:1], dtype=tf.float32)
t_f = tf.convert_to_tensor(X_f[:,1:2], dtype=tf.float32)


#generate point vectors for training
X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

#seperate point vectors
x0 = tf.cast(X0[:,0:1], dtype = tf.float32)
t0 = tf.cast(X0[:,1:2], dtype = tf.float32)

x_lb = tf.convert_to_tensor(X_lb[:,0:1], dtype=tf.float32)
t_lb = tf.convert_to_tensor(X_lb[:,1:2], dtype=tf.float32)

x_ub = tf.convert_to_tensor(X_ub[:,0:1], dtype=tf.float32)
t_ub = tf.convert_to_tensor(X_ub[:,1:2], dtype=tf.float32)

# Begin training (--tf-iter / --newton-iter control phase lengths)
parser = argparse.ArgumentParser(description="Train Burgers PINN")
parser.add_argument(
    "--optimizer",
    choices=["adam", "learnable", "quasi-newton"],
    default="adam",
    help=(
        "Phase-1 optimizer for network weights (Adam or learnable); "
        "'quasi-newton' uses Adam in phase 1 then SciPy quasi-Newton in phase 2 "
        "(requires modified SciPy minimize as in Quasi-Newton Optimizer Examples)."
    ),
)
parser.add_argument(
    "--tf-iter",
    type=int,
    default=100,
    help="Number of phase-1 (Adam or learnable) training epochs.",
)
parser.add_argument(
    "--newton-iter",
    type=int,
    default=100,
    help="Maximum L-BFGS or quasi-Newton refinement iterations on network weights.",
)
parser.add_argument(
    "--qn-method",
    choices=["BFGS", "bfgsr", "bfgsz"],
    default="BFGS",
    help="Top-level quasi-Newton family (only used when --optimizer quasi-newton).",
)
parser.add_argument(
    "--qn-method-bfgs",
    choices=["BFGS", "BFGS_scipy", "SSBFGS_OL", "SSBFGS_AB",
             "SSBroyden1", "SSBroyden2", "SSBroyden3"],
    default="SSBroyden2",
    help=(
        "BFGS sub-variant when --qn-method=BFGS. "
        "Requires patched SciPy _optimize.py from Optimizers/."
    ),
)
args = parser.parse_args()
tf_iter = args.tf_iter
newton_iter = args.newton_iter
results_dir = build_results_dir(
    tf_iter=tf_iter,
    newton_iter=newton_iter,
    optimizer_name=args.optimizer,
    qn_method_bfgs=args.qn_method_bfgs if args.optimizer == "quasi-newton" else None,
)

training_history, lbfgs_history = fit(
    x_f,
    t_f,
    x0,
    t0,
    u0,
    x_lb,
    t_lb,
    x_ub,
    t_ub,
    col_weights,
    u_weights,
    tf_iter=tf_iter,
    newton_iter=newton_iter,
    optimizer_name=args.optimizer,
    qn_method=args.qn_method,
    qn_method_bfgs=args.qn_method_bfgs,
)

write_csv(
    os.path.join(results_dir, "training_loss.csv"),
    ["epoch", "total_loss", "mse_0", "mse_f"],
    training_history,
)
write_csv(
    os.path.join(results_dir, "lbfgs_loss.csv"),
    ["step", "loss"],
    lbfgs_history,
)

#generate mesh to find U0-pred for the whole domain
X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact_u.T.flatten()[:,None]

lb = np.array([-1.0, 0.0])
ub = np.array([1.0, 1])

# Get preds
u_pred, f_u_pred = predict(X_star)

#find L2 error
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
print('Error u: %e' % (error_u))

final_training_loss = training_history[-1][1] if training_history else float("nan")
final_lbfgs_loss = lbfgs_history[-1][1] if lbfgs_history else float("nan")
qn_method_log = args.qn_method if args.optimizer == "quasi-newton" else ""
qn_method_bfgs_log = args.qn_method_bfgs if args.optimizer == "quasi-newton" else ""
write_csv(
    os.path.join(results_dir, "results_summary.csv"),
    ["optimizer", "qn_method", "qn_method_bfgs", "tf_iter", "newton_iter",
     "error_u", "final_training_loss", "final_lbfgs_loss"],
    [[args.optimizer, qn_method_log, qn_method_bfgs_log, tf_iter, newton_iter,
      float(error_u), final_training_loss, final_lbfgs_loss]],
)


U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

FU_pred = griddata(X_star, f_u_pred.flatten(), (X, T), method='cubic')


#plotting script in the style of Raissi et al

######################################################################
############################# Plotting ###############################
######################################################################

X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])

fig, ax = newfig(1.3, 2)
ax.remove()

####### Row 0: h(t,x) ##################
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(U_pred.T, interpolation='nearest', cmap='YlGnBu',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)


line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[25]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[50]*np.ones((2,1)), line, 'k--', linewidth = 1)
ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.set_title('$u(t,x)$', fontsize = 10)

####### Row 1: h(t,x) slices ##################
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact_u[:,25], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = %.2f$' % (t[25, 0]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact_u[:,50], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = %.2f$' % (t[50, 0]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact_u[:,75], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = %.2f$' % (t[75, 0]), fontsize = 10)

fig.savefig(os.path.join(results_dir, "u_field_and_slices.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

#show u_pred across domain
fig, ax = plt.subplots()

ec = plt.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
            extent=[0.0, 1.0, -1.0, 1.0],
            origin='lower', aspect='auto')

ax.autoscale_view()
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
cbar = plt.colorbar(ec)
cbar.set_label('$u(x,t)$')
plt.title("Predicted $u(x,t)$",fontdict = {'fontsize': 14})
fig.savefig(os.path.join(results_dir, "u_pred.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Show F_U_pred across domain, should be close to 0
fig, ax = plt.subplots()

ec = plt.imshow(FU_pred.T, interpolation='nearest', cmap='rainbow',
            extent=[0.0, math.pi/2, -5.0, 5.0],
            origin='lower', aspect='auto')

ax.autoscale_view()
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
cbar = plt.colorbar(ec)
cbar.set_label('$\overline{f}_u$ prediction')
fig.savefig(os.path.join(results_dir, "f_u_pred.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# collocation point weights
fig, ax = plt.subplots()
ax.scatter(t_f, x_f, c = col_weights.numpy(), s = col_weights.numpy()/5)
fig.savefig(os.path.join(results_dir, "collocation_weights.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
