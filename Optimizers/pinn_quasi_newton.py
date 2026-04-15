"""
SciPy quasi-Newton refinement for PINN weights (network only).

This follows the pattern in ``Quasi-Newton Optimizer Examples`` and expects a
SciPy build whose ``minimize`` supports the extended ``BFGS`` / ``bfgsr`` /
``bfgsz`` options (``hess_inv0``, ``method_bfgs``, ``r_inv0``, ``Z0``, etc.).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.optimize import minimize


def run_quasi_newton_refinement(
    loss_and_flat_grad,
    w0_tf,
    newton_iter,
    *,
    qn_method: str = "BFGS",
    qn_method_bfgs: str = "SSBroyden2",
    print_every: int = 10,
):
    """
    Refine flattened network weights with SciPy quasi-Newton.

    Parameters
    ----------
    loss_and_flat_grad
        Callable ``w_tf -> (loss_tf, grad_flat_tf)`` with ``w_tf`` a 1-D
        TensorFlow tensor (float32) of network weights.
    w0_tf
        Initial weight vector (same shape/dtype as used in training).
    newton_iter
        ``maxiter`` passed to SciPy for each ``minimize`` call.
    qn_method
        One of ``\"BFGS\"``, ``\"bfgsr\"``, ``\"bfgsz\"`` (see project examples).
    qn_method_bfgs
        Sub-method when ``qn_method == \"BFGS\"`` (e.g. ``\"SSBroyden2\"``).
    print_every
        Print callback losses when ``step % print_every == 0``.

    Returns
    -------
    list[list]
        ``[[step, loss], ...]`` compatible with existing ``lbfgs_loss.csv`` writers.
    """
    w0 = np.asarray(w0_tf.numpy(), dtype=np.float64).ravel()
    n = w0.size
    history: list[list[float | int]] = []
    step = {"n": 0}

    def fun_and_jac(weights_np: np.ndarray):
        w_tf = tf.convert_to_tensor(weights_np.astype(np.float32))
        loss_tf, grad_tf = loss_and_flat_grad(w_tf)
        g = np.asarray(grad_tf.numpy(), dtype=np.float64).ravel()
        return float(loss_tf.numpy()), g

    def callback(xk=None, intermediate_result=None, **_kwargs):
        if intermediate_result is not None:
            loss_v = float(intermediate_result.fun)
        elif xk is not None:
            loss_v = fun_and_jac(xk)[0]
        else:
            return
        history.append([step["n"], loss_v])
        if print_every and (step["n"] % print_every == 0 or step["n"] == 0):
            print(f"  QN step {step['n']}, loss: {loss_v}")
        step["n"] += 1

    if newton_iter <= 0:
        return history

    options = _build_options(qn_method, n, newton_iter, qn_method_bfgs)

    result = minimize(
        fun_and_jac,
        w0,
        method=qn_method,
        jac=True,
        options=options,
        tol=0.0,
        callback=callback,
    )

    w_final = tf.convert_to_tensor(np.asarray(result.x, dtype=np.float32))
    loss_tf, _ = loss_and_flat_grad(w_final)
    if not history:
        history.append([0, float(loss_tf.numpy())])
    return history


def _build_options(method: str, n: int, maxiter: int, method_bfgs: str):
    if method == "BFGS":
        initial_scale = False
        h0 = np.eye(n, dtype=np.float64)
        return {
            "maxiter": maxiter,
            "gtol": 0.0,
            "hess_inv0": h0,
            "method_bfgs": method_bfgs,
            "initial_scale": initial_scale,
        }
    if method == "bfgsr":
        r0 = sparse.csr_matrix(np.eye(n))
        return {"maxiter": maxiter, "gtol": 0.0, "r_inv0": r0}
    if method == "bfgsz":
        z0 = np.eye(n, dtype=np.float64)
        return {"maxiter": maxiter, "gtol": 0.0, "Z0": z0}
    raise ValueError(
        f"Unknown qn_method {method!r}; expected 'BFGS', 'bfgsr', or 'bfgsz'."
    )
