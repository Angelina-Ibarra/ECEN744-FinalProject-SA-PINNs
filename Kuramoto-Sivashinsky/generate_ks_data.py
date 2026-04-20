"""
Generate reference solution for the Kuramoto-Sivashinsky benchmark via ETDRK4
spectral integration. One-shot utility; re-run only if the domain/IC changes.

PDE: u_t + u*u_x + u_xx + u_xxxx = 0
Domain: x in [-1, 1] (periodic), t in [0, 1]
IC: u(x, 0) = -sin(pi*x)

Method: ETDRK4 (Kassam & Trefethen, 2005) with 256 Fourier modes, dt = 1e-4,
saves 201 time frames (dt_save = 0.005). Output layout matches AC.mat /
burgers_shock.mat: keys 't', 'x', 'usol' where usol has shape (N_x, N_t).
"""

import numpy as np
import os
import scipy.io

# ---- problem parameters ----
N = 256                          # Fourier modes
Lx = 2.0                         # domain length (x in [-1, 1])
T = 1.0                          # final time
dt = 1e-4                        # integrator step
n_save = 201                     # saved time frames (incl. t=0 and t=T)
M_contour = 16                   # contour points for ETDRK4 coefficients

# ---- spatial grid (periodic) ----
x = np.linspace(-1.0, 1.0, N, endpoint=False)
k = (2.0 * np.pi / Lx) * np.concatenate([np.arange(0, N // 2),
                                         np.arange(-N // 2, 0)])

# ---- linear operator in Fourier space: u_t = (k^2 - k^4) u_hat + ... ----
Lk = k ** 2 - k ** 4

# ---- ETDRK4 coefficient precomputation via contour integral ----
E = np.exp(dt * Lk)
E2 = np.exp(dt * Lk / 2.0)
r = np.exp(1j * np.pi * (np.arange(1, M_contour + 1) - 0.5) / M_contour)
LR = dt * Lk[:, None] + r[None, :]
Q = dt * np.real(np.mean((np.exp(LR / 2.0) - 1.0) / LR, axis=1))
f1 = dt * np.real(np.mean(
    (-4.0 - LR + np.exp(LR) * (4.0 - 3.0 * LR + LR ** 2)) / LR ** 3, axis=1))
f2 = dt * np.real(np.mean(
    (2.0 + LR + np.exp(LR) * (-2.0 + LR)) / LR ** 3, axis=1))
f3 = dt * np.real(np.mean(
    (-4.0 - 3.0 * LR - LR ** 2 + np.exp(LR) * (4.0 - LR)) / LR ** 3, axis=1))

# ---- nonlinear operator: returns FFT(-u*u_x) = -0.5 * i*k * FFT(u^2) ----
g = -0.5j * k
def nonlinear(v_hat):
    u_phys = np.real(np.fft.ifft(v_hat))
    return g * np.fft.fft(u_phys ** 2)

# ---- initial condition ----
u0 = -np.sin(np.pi * x)
v = np.fft.fft(u0)

# ---- time integration ----
n_steps = int(round(T / dt))
save_every = n_steps // (n_save - 1)
assert save_every * (n_save - 1) == n_steps, "n_steps must be divisible by (n_save - 1)"

u_save = np.zeros((N, n_save))
t_save = np.zeros(n_save)
u_save[:, 0] = u0
t_save[0] = 0.0
next_save_idx = 1

for step in range(1, n_steps + 1):
    Nv = nonlinear(v)
    a = E2 * v + Q * Nv
    Na = nonlinear(a)
    b = E2 * v + Q * Na
    Nb = nonlinear(b)
    c = E2 * a + Q * (2.0 * Nb - Nv)
    Nc = nonlinear(c)
    v = E * v + Nv * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3

    if step % save_every == 0 and next_save_idx < n_save:
        u_save[:, next_save_idx] = np.real(np.fft.ifft(v))
        t_save[next_save_idx] = step * dt
        next_save_idx += 1

# ---- sanity checks before saving ----
assert next_save_idx == n_save, f"saved {next_save_idx} frames, expected {n_save}"
assert not np.any(np.isnan(u_save)), "reference solution contains NaN — reduce dt or check IC"
assert np.max(np.abs(u_save)) < 50.0, (
    f"max |u| = {np.max(np.abs(u_save)):.2f} — spectral blow-up suspected"
)

# ---- write KS.mat matching AC.mat / burgers_shock.mat layout ----
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KS.mat")
scipy.io.savemat(out_path, {
    "t": t_save.reshape(-1, 1),
    "x": x.reshape(-1, 1),
    "usol": u_save,
})

print(f"wrote {out_path}")
print(f"  t shape = {t_save.shape}, x shape = {x.shape}, usol shape = {u_save.shape}")
print(f"  u range: [{u_save.min():.4f}, {u_save.max():.4f}]")
