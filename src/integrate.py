"""Shared differentiable integrator. Both models are integrated by the EXACT
same code path -- only `model.vector_field` differs."""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp


def rollout(model, y0, ts, *, adaptive: bool = False, max_steps: int = 4096):
    """Integrate dy/dt = model.vector_field(y) and save at times `ts`.

    `adaptive=False` (constant step) is used during training: it makes the
    number of solver steps static, which keeps `jit`/`vmap` fast and the
    backward pass cheap. `adaptive=True` is used for long evaluation rollouts
    where we want the solver -- not the model -- to control accuracy.
    """
    term = diffrax.ODETerm(lambda t, y, args: model.vector_field(y))
    if adaptive:
        controller = diffrax.PIDController(rtol=1e-8, atol=1e-10)
        solver = diffrax.Tsit5()
    else:
        controller = diffrax.ConstantStepSize()
        solver = diffrax.Tsit5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=controller,
        max_steps=max_steps,
    )
    return sol.ys


def batch_rollout(model, y0s, ts, **kw):
    """vmap rollout over a batch of initial conditions sharing one time grid."""
    return jax.vmap(lambda y0: rollout(model, y0, ts, **kw))(y0s)


# --------------------------------------------------------------------------- #
# Fixed-step integrators on a uniform time grid (for the symplectic comparison).
# Hand-rolled with lax.scan so they are transparent, fast under vmap, and fully
# differentiable for training.
# --------------------------------------------------------------------------- #
def leapfrog_rollout(model, y0, ts, n_substeps: int = 1):
    """Stoermer-Verlet (kick-drift-kick) for a SEPARABLE Hamiltonian model.

        p <- p - (h/2) dV/dq(q)      # half kick
        q <- q +  h    dT/dp(p)      # drift
        p <- p - (h/2) dV/dq(q)      # half kick

    This map is symplectic by construction for ANY separable H, so the discrete
    flow conserves a shadow Hamiltonian -> energy error stays bounded with no
    secular growth, however long you integrate.
    """
    d = y0.shape[-1] // 2
    q0, p0 = y0[:d], y0[d:]
    h = (ts[1] - ts[0]) / n_substeps

    def save_step(carry, _):
        def sub(qp, _):
            q, p = qp
            p = p - 0.5 * h * model.dV_dq(q)
            q = q + h * model.dT_dp(p)
            p = p - 0.5 * h * model.dV_dq(q)
            return (q, p), None
        (q, p), _ = jax.lax.scan(sub, carry, None, length=n_substeps)
        return (q, p), jnp.concatenate([q, p])

    _, ys = jax.lax.scan(save_step, (q0, p0), None, length=ts.shape[0] - 1)
    return jnp.concatenate([y0[None], ys], axis=0)


def rk4_rollout(model, y0, ts, n_substeps: int = 1):
    """Classic RK4 at the same fixed step -- accurate but NON-symplectic, so its
    energy error grows secularly. The control arm for the symplectic comparison."""
    h = (ts[1] - ts[0]) / n_substeps
    f = lambda y: model.vector_field(y)

    def save_step(y, _):
        def sub(y, _):
            k1 = f(y)
            k2 = f(y + 0.5 * h * k1)
            k3 = f(y + 0.5 * h * k2)
            k4 = f(y + h * k3)
            return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), None
        y, _ = jax.lax.scan(sub, y, None, length=n_substeps)
        return y, y

    _, ys = jax.lax.scan(save_step, y0, None, length=ts.shape[0] - 1)
    return jnp.concatenate([y0[None], ys], axis=0)


def batch_leapfrog(model, y0s, ts, n_substeps: int = 1):
    return jax.vmap(lambda y0: leapfrog_rollout(model, y0, ts, n_substeps))(y0s)
