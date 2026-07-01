"""Ground-truth two-body (Kepler) dynamics and conserved quantities.

Milestone 1-2 uses the *reduced* two-body problem: by going to the relative
coordinate q = r2 - r1 and the centre-of-mass frame, the gravitational two-body
problem collapses to a single particle of reduced mass mu=1 moving in a central
1/r potential with G*M=1. This is integrable and NON-chaotic -- the right
sandbox to prove the machinery works before facing the chaotic three-body case.

State convention everywhere in this project:
    y = [q_x, q_y, p_x, p_y]   (configuration then momentum)

Conserved along exact trajectories:
    energy        E = 1/2 |p|^2 - 1/|q|
    ang. momentum L = q_x p_y - q_y p_x      (z-component, scalar in 2D)

These two scalars are the yardstick for the whole project: a structure-
preserving model should hold them flat even when its trajectory eventually
decorrelates from truth.
"""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # conserved-quantity drift is tiny; need f64


def hamiltonian(y: jnp.ndarray) -> jnp.ndarray:
    """True Kepler Hamiltonian H(q, p) = 1/2 |p|^2 - 1/|q|  (scalar)."""
    q, p = y[:2], y[2:]
    return 0.5 * jnp.dot(p, p) - 1.0 / jnp.linalg.norm(q)


def angular_momentum(y: jnp.ndarray) -> jnp.ndarray:
    """L_z = q_x p_y - q_y p_x."""
    return y[0] * y[3] - y[1] * y[2]


def true_vector_field(t, y, args=None):
    """Hamilton's equations for the exact Kepler problem.

    dq/dt =  p
    dp/dt = -q / |q|^3
    """
    q, p = y[:2], y[2:]
    r3 = jnp.linalg.norm(q) ** 3
    return jnp.concatenate([p, -q / r3])


def initial_condition(r0: float, eccentricity: float) -> jnp.ndarray:
    """Build a bound orbit IC at perihelion: position (r0, 0), velocity purely
    tangential. For a 1/r potential the perihelion speed is

        v = sqrt((1 + e) / r0),

    which gives a closed ellipse of eccentricity `e` (e=0 -> circle). Energy is
    negative (bound) for all e in [0, 1).
    """
    v = jnp.sqrt((1.0 + eccentricity) / r0)
    return jnp.array([r0, 0.0, 0.0, v])


def integrate_true(y0: jnp.ndarray, ts: jnp.ndarray) -> jnp.ndarray:
    """High-accuracy reference trajectory via an 8th-order solver with tight
    tolerances -- this is our 'data' and our ground-truth baseline."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(true_vector_field),
        diffrax.Dopri8(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=float(ts[1] - ts[0]),
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=1_000_000,
    )
    return sol.ys


def make_dataset(key, n_orbits: int, t_span: float, n_points: int):
    """Generate `n_orbits` reference trajectories with varied size/eccentricity.

    Returns:
        trajs: (n_orbits, n_points, 4) states
        ts:    (n_points,) shared time grid
    """
    ts = jnp.linspace(0.0, t_span, n_points)
    k_r, k_e = jax.random.split(key)
    r0s = jax.random.uniform(k_r, (n_orbits,), minval=0.8, maxval=1.4)
    eccs = jax.random.uniform(k_e, (n_orbits,), minval=0.0, maxval=0.5)
    y0s = jax.vmap(initial_condition)(r0s, eccs)
    trajs = jax.vmap(lambda y0: integrate_true(y0, ts))(y0s)
    return trajs, ts
