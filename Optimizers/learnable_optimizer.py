import tensorflow as tf
import numpy as np
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

# Generate the collocation points for the PINN model
def generate_dataset(data, N_pde, N_iv, bs_pdes, bs_inits, k=1, p=0):

    # Extract the parameters of the model
    t_bdry = data['t_bdry']
    x_bdry = data['x_bdry']
    u0 = data['u0']

    # Sample points where to evaluate the PDE
    tx_min = np.array([t_bdry[0], x_bdry[0]])
    tx_max = np.array([t_bdry[1], x_bdry[1]])  
    pde_points = tx_min + (tx_max - tx_min)*lhs(2, N_pde)
    t_pde = pde_points[:,0]
    x_pde = pde_points[:,1]
    pdes = np.column_stack([t_pde, x_pde]).astype(np.float32)

    # Sample points where to evaluate the initial values
    init_points = tx_min[1:] + (tx_max[1:] - tx_min[1:])*lhs(1, N_iv)
    x_init = init_points
    t_init = t_bdry[0]+ 0.0*x_init
    u_init = u0(x_init, k, p)
    inits = np.column_stack([t_init, x_init, u_init]).astype(np.float32)

    ds_pde = tf.data.Dataset.from_tensor_slices(pdes)
    ds_pde = ds_pde.cache().shuffle(N_pde).batch(bs_pdes)

    ds_init = tf.data.Dataset.from_tensor_slices(inits)
    ds_init = ds_init.cache().shuffle(N_iv).batch(bs_inits)

    ds = tf.data.Dataset.zip((ds_pde, ds_init))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds

# Define the network
def build_neural_network(n_layers = 6, n_units = 20, summary = False):
    xleft, xright = -1, 1
    inp1 = tf.keras.layers.Input(shape=(1,))
    b1 = tf.keras.layers.Lambda(lambda x: 2.0*(x - xleft)/(xright-xleft)-1.0)(inp1)

    # Make input periodic
    inp2 = tf.keras.layers.Input(shape=(1,))
    b2 = tf.keras.layers.Lambda(lambda x: tf.concat((tf.cos(np.pi*x), 
                                                    tf.sin(np.pi*x)), axis=1))(inp2)

    b = tf.keras.layers.Concatenate()([b1, b2])

    for i in range(n_layers):
        b = tf.keras.layers.Dense(n_units, activation='tanh')(b)
    out = tf.keras.layers.Dense(1, activation='linear')(b)

    model = tf.keras.models.Model([inp1, inp2], out)

    if summary:
        model.summary()

    return model

# Convert the ragged tensors that make up a model into a one-dimensional vector
def reshape_to_vector(vars):
    vars_vector = tf.concat([tf.reshape(p, [-1]) for p in vars], axis=0)
    return vars_vector

# This updates the weights of a given model as ragged tensor from the weights
# collected in a one-dimensional weight vector
def reshape_to_model(vars, model):

    k = 0

    for (i, curr_layer) in enumerate(model.layers):
        weights_and_biases = curr_layer.get_weights()

    # Check if the layer actually has weights and then set them
    if len(weights_and_biases) > 1:
        new_weights_and_biases = []
        for l in range(len(weights_and_biases)):
            shape_weights = tf.shape(weights_and_biases[l])
            no_weights = tf.reduce_prod(shape_weights)
            new_weights_and_biases.append(tf.reshape(vars[k:k+no_weights], shape_weights))
            k += no_weights 

        # Now set the new weights  
        model.layers[i].set_weights(new_weights_and_biases)

def standardize_tensor(vars, ep = 1e-5):
    return vars/tf.sqrt(tf.reduce_mean(tf.square(vars) + ep, axis=0))

@tf.function
def train_step(pdes, inits, model, c=-0.0025):

    t_pde, x_pde = pdes[:,:1], pdes[:,1:2]
    t_init, x_init, u_init = inits[:,:1], inits[:,1:2], inits[:,2:3]

    # Outer gradient for tuning network parameters
    with tf.GradientTape() as tape:

        # Inner gradient for derivatives of u wrt x and t
        with tf.GradientTape() as tape1:
            tape1.watch(t_pde), tape1.watch(x_pde)
            with tf.GradientTape() as tape2:
                tape2.watch(t_pde), tape2.watch(x_pde)
                with tf.GradientTape() as tape3:
                    tape3.watch(t_pde), tape3.watch(x_pde)
                    u = model([t_pde, x_pde])
                [ut, ux] = tape3.gradient(u, [t_pde, x_pde])  
            uxx = tape2.gradient(ux, x_pde)
        uxxx = tape1.gradient(uxx, x_pde)

        # Solve the KdV equations
        eqn = ut+u*ux-c*uxxx

        # Define the PDE loss  
        PDEloss = tf.reduce_mean(tf.square(eqn))

        # Define the initial value loss
        u_init_pred = model([t_init, x_init])      
        IVloss = tf.reduce_mean(tf.square(u_init-u_init_pred))

        # Global loss
        loss = PDEloss + IVloss

    # Compute the gradient of the global loss wrt the model parameters
    grads = tape.gradient(loss, model.trainable_variables)

    return PDEloss, IVloss, grads

class LearnableOptimizer():
  def __init__(self, learning_rate=5e-4, beta_1=0.9, beta_2=0.999, 
               m_decay = [0.5, 0.9, 0.99, 0.999], 
               v_decay = [0.5, 0.9, 0.99, 0.999],
               ep=1e-5, use_v_moment = True):

    # For the Adam part
    self.learning_rate = learning_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.ep = ep

    # For the learnable part
    self.m_decay = m_decay
    self.v_decay = v_decay    
    self.use_v_moment = use_v_moment

    self.no_m = len(m_decay)
    self.no_v = len(v_decay)

    if self.no_m != self.no_v:
      raise ValueError('Number of second moment decay value needs to equal number of first moment decay values.')

    self.built = False

    # Scaling of learnable parameters
    self.lambda1 = learning_rate
    self.lambda2 = 0.001 
    self.lambda3 = 0.001

    # Build a learnable optimizer
    self.optimizer = self.build_optimizer_model()


  def build_optimizer_model(self, summary = False):
    step_vector = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

    # Build learnable optimizer MLP

    # Used for all optimizers:
    inp_w = tf.keras.layers.Input(shape=(1,))
    inp_g = tf.keras.layers.Input(shape=(1,))
    inp_m = tf.keras.layers.Input(shape=(self.no_m,))

    # If using the second momentum as well:
    if self.use_v_moment:
      inp_v = tf.keras.layers.Input(shape=(self.no_v,))
      inp_rv = tf.keras.layers.Input(shape=(self.no_v,))
      inp_corrected = tf.keras.layers.Input(shape=(self.no_v,))
    else:
      inp_v = tf.keras.layers.Input(shape=(1,))

    inp_timestep = tf.keras.layers.Input(shape=(1,))

    # Generate time embedding
    time_embedding = tf.keras.layers.Lambda(lambda t: [tf.math.tanh(t/x) for x in step_vector])(inp_timestep)

    # Concatenate and standardize the input features
    if self.use_v_moment:
      b = tf.keras.layers.Concatenate()([inp_w, inp_g, inp_m, inp_v, inp_rv, 
                                         inp_corrected])
      b = tf.keras.layers.Lambda(standardize_tensor)(b)
      b = tf.keras.layers.Concatenate()([b, *time_embedding])
    else:
      b = tf.keras.layers.Concatenate()([inp_w, inp_g, inp_m, inp_v])
      b = tf.keras.layers.Lambda(standardize_tensor)(b)
      b = tf.keras.layers.Concatenate()([b, *time_embedding])

    # A simple MLP network
    for i in range(2):
      b = tf.keras.layers.Dense(32, activation='swish')(b)
    
    # Initializer for the weight matrix (to produce close to zero initially)
    initializer = tf.keras.initializers.RandomNormal(0.0, 1e-3, seed=1234)

    out_mag = tf.keras.layers.Dense(1, kernel_initializer=initializer)(b)
    out_dir = tf.keras.layers.Dense(1, kernel_initializer=initializer)(b)
    out_nom = tf.keras.layers.Dense(1, kernel_initializer=initializer)(b)


    if self.use_v_moment:
      optModel = tf.keras.models.Model([inp_w, inp_g, inp_m, inp_v, inp_rv,
                                        inp_corrected, inp_timestep],
                                       [out_mag, out_dir, out_nom])
    else:
      optModel = tf.keras.models.Model([inp_w, inp_g, inp_m, inp_v, inp_timestep],
                                       [out_mag, out_dir, out_nom])
    if summary:
      optModel.summary()

    return optModel


  # This is the learnable optimizer
  def apply_gradients(self, vars, grads, optimizerModel):

    # Reshape the weights and gradients into a vector
    vars_vector = reshape_to_vector(vars)
    grads_vector = reshape_to_vector(grads)

    # Add the momentum and velocity vectors
    if not self.built:

      # Reset the step counter
      self.steps = 0

      # Initialize momenta and second momenta
      self.m = tf.zeros(tf.shape(vars_vector))
      self.v = tf.zeros(tf.shape(vars_vector))

      self.ms = []
      self.vs = []

      for i in range(self.no_m):
        self.ms.append(tf.zeros(tf.shape(vars_vector)))
        
        # Second momenta update
        if self.use_v_moment:
          self.vs.append(tf.zeros(tf.shape(vars_vector)))

      self.built = True

    # ----- This is the Adam update -----

    # Update step counter
    self.steps += 1

    # Adam update rule
    self.m = self.beta_1*self.m + (1-self.beta_1)*grads_vector
    self.v = self.beta_2*self.v + (1-self.beta_2)*tf.square(grads_vector)

    m_corr = self.m/(1-self.beta_1**self.steps) 
    v_corr = self.v/(1-self.beta_2**self.steps) 

    adam_update = self.learning_rate*m_corr/tf.sqrt(v_corr + self.ep)

    # ----- This is the Learnable part -----

    if self.use_v_moment:
      self.vhs = []
      self.rsqrtvs = []

    # Store the current AggMo momenta
    for i in range(self.no_m):
      
      # Momentum with respective decay rate
      self.ms[i] = self.m_decay[i]*self.ms[i] + (1-self.m_decay[i])*grads_vector

      if self.use_v_moment:
        self.vs[i] = self.v_decay[i]*self.vs[i] + (1-self.v_decay[i])*tf.square(grads_vector)
        self.vhs.append(self.ms[i]/tf.sqrt(self.vs[i] + self.ep))
        self.rsqrtvs.append(1./tf.sqrt(self.vs[i] + self.ep))
        
    # Reshape all the input features    
    weights = tf.expand_dims(vars_vector, axis=1)
    gradients = tf.expand_dims(grads_vector, axis=1)
    m = tf.transpose(tf.convert_to_tensor(self.ms), [1,0])
    
    if self.use_v_moment:
      v = tf.transpose(tf.convert_to_tensor(self.vs), [1,0])
      vh = tf.transpose(tf.convert_to_tensor(self.vhs), [1,0])
      rsqrtv = tf.transpose(tf.convert_to_tensor(self.rsqrtvs), [1,0])
    else:
      v = tf.expand_dims(self.v, axis=1)

    steps = tf.expand_dims(self.steps + 0*vars_vector, axis=1)

    # Predict magnitude, precondition and direction of update step
    if self.use_v_moment:
      [mag, dir, mag_nom] = optimizerModel([weights, gradients, m, v, rsqrtv, 
                                            vh, steps])    
    else:
      [mag, dir, mag_nom] = optimizerModel([weights, gradients, m, v, steps])
    
    # Blackbox update
    bb_update = self.lambda1*dir*tf.exp(self.lambda2*mag)/tf.sqrt(v[:,-1] + self.ep)

    # Scaled Adam update
    adam_scaled = tf.exp(self.lambda3*mag_nom[:,0])*adam_update

    # ---- Combined update -----

    # Update is the sum of Adam and learned update
    vars_vector -= (adam_scaled + bb_update[:,0])

    return vars_vector


  # Persistent evolution strategy for training the learnable optimizer
  def train_optimizer(self, data, model, N_pde = 10000, N_iv = 100, tasks = 5,
                      cycles = 5, steps = 100, K = 1, N = 2, sigma = 0.1, 
                      alpha = 1e-3, wd = 0.0, clipping = True, use_adam = False):

    # Shape of the optimizer weights as weight vector
    opt_shape = tf.shape(reshape_to_vector(self.optimizer.trainable_variables))

    # If using Adam for training, initialize first and second momentum
    if use_adam:
      opt_m = tf.zeros(shape = opt_shape)
      opt_v = tf.zeros(shape = opt_shape)

    # Main task loss
    task_loss = []

    self.steps = 0

    # Generate training datasets
    ds, c = [], []
    for task in range(tasks):
      
      # Sample a random IC
      k = np.random.randint(1, 4)
      p = np.random.uniform(-np.pi/2, np.pi/2)

      ds.append(generate_dataset(data, N_pde, N_iv, N_pde//10, N_iv//10, k, p))
      c.append(-0.0025)

    for cycle in range(cycles):

      print(f'Working on cycle {cycle+1} out of {cycles} cycles:')

      # Main task training loop
      for task in range(tasks):

        print(f'Working on task {task+1} out of {tasks} tasks:')

        # Reset the optimizer
        self.built = False

        # Convert optimizer ragged tensor to vector
        opt_vars = reshape_to_vector(self.optimizer.trainable_variables)

        # Initialize perturbed optimizers and their model copies
        models = []
        optimizers = []
        for i in range(N):
          models.append(build_neural_network())
          optimizers.append(self.build_optimizer_model())
          xi = np.zeros(shape = (N, opt_shape[0]), dtype=np.float64)

        # Mean loss over l-th step
        mean_losses = []

        # Run the optimization a total of steps time steps
        for l in range(steps):

          # Initialize the gradient vector
          gradient_pes = tf.zeros(shape = opt_shape)

          # Loss accumulator for N models
          all_losses = []

          print(f'Working on optimization step {l+1} out of {steps} steps:')

          # Train N particles
          for i in range(N):

            print(f'Training model {i+1} out of {N} models.')

            optimizers[i].set_weights(self.optimizer.get_weights())
            optimizer_i_vars = reshape_to_vector(optimizers[i].trainable_variables)

            # Perturbations
            if tf.math.mod(i+1, 2)==0:
              eps = -eps
            else:
              eps = tf.random.normal(shape = opt_shape, stddev=sigma**2)

            # Adjust weights of the temporary optimizer using the perturbations
            optimizer_i_vars += eps

            # Write these weights into the optimizer
            reshape_to_model(optimizer_i_vars, optimizers[i])

            # ---- Unroll the model ----
            losses = np.zeros(K)
            for k in range(K):
              
              # Mini-batch gradient descent
              no_batches = 0

              for (pdes, inits) in ds[task]:

                # Train the model for one batch (using the linear advection equation!)
                PDEloss, IVloss, grads = train_step(pdes, inits, models[i], c[task])
                temp_loss = PDEloss + IVloss

                # Gradient descent step with current perturbed learnable optimizer
                weights_updated = self.apply_gradients(models[i].trainable_variables, 
                                                      grads, optimizers[i])
                
                # Update the current model's weights
                reshape_to_model(weights_updated, models[i])

                # Check if model failed to train
                if tf.math.is_nan(temp_loss):
                  raise ValueError("Loss became NaN.")

                # Update the loss
                losses[k] += temp_loss
                no_batches += 1
              losses[k] /= no_batches

            # Update the perturbation vector
            xi[i,] += eps

            # Update the gradient vector
            gradient_pes += tf.cast(xi[i,]*tf.reduce_sum(losses), tf.float32)

            # Store loss values for i-th model
            all_losses.append(tf.reduce_mean(losses))

          print(f'Accumulated loss over step {l+1}: {tf.reduce_mean(all_losses): 6.4f}.\n')

          # Append the mean loss
          mean_losses.append(tf.reduce_mean(all_losses))

          # Correct the scaling to get the true gradient
          gradient_pes /= (N*sigma**2) 
          
          # Apply gradient clipping to norm 1
          if clipping:
            gradient_pes = tf.clip_by_norm(gradient_pes, clip_norm=1)

          # Update the optimizer weights using AdamW
          if use_adam:

            opt_m = self.beta_1*opt_m + (1-self.beta_1)*gradient_pes
            opt_v = self.beta_2*opt_v + (1-self.beta_2)*tf.square(gradient_pes)

            opt_m_corr = opt_m/(1-self.beta_1**(l+1))
            opt_v_corr = opt_v/(1-self.beta_2**(l+1))

            # AdamW update formula
            opt_vars -= alpha*(opt_m_corr/tf.sqrt(opt_v_corr + self.ep) + 
                              wd*opt_vars)
            
          # Standard SGD with weight decay
          else:

            opt_vars -= alpha*(gradient_pes + wd*opt_vars)

          # Update the optimizer
          reshape_to_model(opt_vars, self.optimizer)

        task_loss.append(mean_losses)

    print('Training done!')

    return np.array(task_loss)

  # ------------------------------------------------------------------
  # ------------------------------------------------------------------

  def train_model(self, data, model, epochs=5000, N_pde=10000, N_iv=100, k=1,
                  p=0):

    # Reset the optimizer
    self.built = False

    # Training dataset
    ds = generate_dataset(data, N_pde, N_iv, N_pde//10, N_iv//10, k, p)
  
    # Epoch loss initialization
    epoch_loss = np.zeros(epochs)

    # Main training loop
    for i in range(epochs):

      n_batches = 0
      for (pdes, inits) in ds:

        # Gradient step    
        PDEloss, IVloss, grads = train_step(pdes, inits, model)
        weights_updated = self.apply_gradients(model.trainable_variables, grads,
                                               self.optimizer)
                                                         
        # Update the current model's weights
        reshape_to_model(weights_updated, model)

        # Accumulate the loss
        epoch_loss[i] += PDEloss + IVloss
        n_batches += 1

      epoch_loss[i] /= n_batches
      n_batches = 0

      if (np.mod(i, 100)==0):
        print(f"PDE loss, IV loss in {i}th epoch: {PDEloss.numpy():6.4f}, {IVloss.numpy():6.4f}")
      
    return epoch_loss