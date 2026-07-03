"""Bridge to the three-body physics engine + 3-body conserved quantities.

Data source: the periodic-orbit engine at ~/PycharmProjects/PythonProject1
(`three_body.py` + `orbits.db`). We pull real catalogued orbits, integrate an
ensemble of slightly-perturbed initial conditions around each (to fill a tube of
phase space rather than a single 1-D closed curve), and hand back trajectories in
the SAME [q (6), p (6)] layout the models use.

State layout (identical to the engine's):
    y = [x1,y1, x2,y2, x3,y3,  vx1,vy1, vx2,vy2, vx3,vy3]
With equal masses m=1, momentum p_i = v_i, so the true Hamiltonian
    H = 1/2 |p|^2  -  sum_{i<j} 1/|r_i - r_j|
is separable -> SeparableHamiltonianNODE + leapfrog apply directly at dim=6.

Conserved quantities (the chaos-robust evaluation yardstick): energy E, angular
momentum L_z, and linear momentum (P_x, P_y). For COM-frame orbits P ~ 0.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ENGINE_DIR = Path(os.environ.get("THREEBODY_ENGINE",
                                 Path.home() / "PycharmProjects" / "PythonProject1"))


def _import_engine():
    """Import the physics engine, failing loud if the repo isn't found."""
    if not (ENGINE_DIR / "three_body.py").exists():
        raise FileNotFoundError(
            f"three-body engine not found at {ENGINE_DIR}. Set $THREEBODY_ENGINE "
            f"to the PythonProject1 checkout, or generate data another way.")
    if str(ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(ENGINE_DIR))
    import three_body  # noqa: E402
    return three_body


# --------------------------------------------------------------------------- #
# Conserved quantities (JAX, for evaluation) -- mirror the engine's numpy defs.
# --------------------------------------------------------------------------- #
_PAIRS = ((0, 1), (0, 2), (1, 2))


def energy(y):
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    KE = 0.5 * jnp.sum(v ** 2)
    PE = 0.0
    for i, j in _PAIRS:
        PE = PE - 1.0 / jnp.linalg.norm(r[i] - r[j])
    return KE + PE


def angular_momentum(y):
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    return jnp.sum(r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])


def linear_momentum(y):
    v = y[6:].reshape(3, 2)
    return jnp.sum(v, axis=0)  # (2,)


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #
def reference_trajectory(vx, vy, ts, sigma=0.0, seed=0):
    """Engine ground-truth trajectory from a (possibly perturbed) symmetric IC,
    sampled on times `ts`. Used to build held-out test orbits for evaluation."""
    tb = _import_engine()
    state0 = tb.build_state_symmetric(vx, vy)
    if sigma > 0:
        rng = np.random.default_rng(seed)
        state0 = _recenter(state0 + sigma * rng.standard_normal(12))
    ts = np.asarray(ts)
    sol = tb.integrate_orbit(state0, float(ts[-1]) + 1e-9, max_step=float(ts[1] - ts[0]))
    return jnp.asarray(sol.sol(ts).T)


def perturbed_ics(vx, vy, n, sigma, seed):
    """A batch of `n` initial conditions: the base symmetric orbit plus n-1
    small COM-frame perturbations. Returns (n, 12)."""
    tb = _import_engine()
    base = tb.build_state_symmetric(vx, vy)
    rng = np.random.default_rng(seed)
    out = [base] + [_recenter(base + sigma * rng.standard_normal(12)) for _ in range(n - 1)]
    return jnp.asarray(np.stack(out))


def integrate_state(state0, ts):
    """Engine DOP853 ground truth from a raw state, sampled on `ts` -> (len(ts), 12)."""
    tb = _import_engine()
    ts = np.asarray(ts)
    sol = tb.integrate_orbit(np.asarray(state0), float(ts[-1]) + 1e-9,
                             max_step=float(ts[1] - ts[0]))
    return jnp.asarray(sol.sol(ts).T)


def stable_symmetric_sources(n_max=3):
    """A few stable, symmetric (L=0) orbits from the catalogue: (vx, vy, T, name).
    Stable + symmetric -> clean, non-ejecting trajectories: the right first
    3-body data before tackling genuinely chaotic (unstable) orbits."""
    import sqlite3
    con = sqlite3.connect(ENGINE_DIR / "orbits.db")
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT COALESCE(matched_name, name) AS nm, param1, param2, T "
        "FROM orbits WHERE parametrisation='symmetric' AND is_stable=1 "
        "ORDER BY T LIMIT ?", (n_max,)).fetchall()
    con.close()
    return [(r["param1"], r["param2"], r["T"], r["nm"]) for r in rows]


def _recenter(state):
    """Zero the centre of mass and total momentum (equal masses)."""
    r = state[:6].reshape(3, 2) - state[:6].reshape(3, 2).mean(0)
    v = state[6:].reshape(3, 2) - state[6:].reshape(3, 2).mean(0)
    return np.concatenate([r.ravel(), v.ravel()])


def generate(seed=0, n_perturb=8, sigma=0.02, periods=2.0, pts_per_period=100,
             n_sources=3, cache=True):
    """Build an ensemble of 3-body trajectories around stable catalogued orbits.

    Returns (trajs, dt) with trajs of shape (n_traj, n_points, 12). Every
    trajectory shares dt but has its OWN number of points is fixed (periods and
    pts_per_period are per-orbit so windows share a relative time grid within a
    source; we resample all to a common dt and length)."""
    cache_path = Path(__file__).parent.parent / "datasets" / f"threebody_s{seed}.npz"
    if cache and cache_path.exists():
        d = np.load(cache_path)
        return jnp.asarray(d["trajs"]), float(d["dt"])

    tb = _import_engine()
    rng = np.random.default_rng(seed)
    sources = stable_symmetric_sources(n_sources)
    if not sources:
        raise RuntimeError("no stable symmetric orbits found in catalogue")

    # Common time grid: use the shortest period to set dt, fixed length for all.
    T_ref = min(s[2] for s in sources)
    n_points = int(periods * pts_per_period)
    dt = T_ref / pts_per_period

    trajs = []
    for vx, vy, T, name in sources:
        base = tb.build_state_symmetric(vx, vy)
        ts = np.arange(n_points) * dt
        for k in range(n_perturb):
            state0 = base if k == 0 else _recenter(base + sigma * rng.standard_normal(12))
            try:
                sol = tb.integrate_orbit(state0, float(ts[-1]) + dt, max_step=dt)
                ys = sol.sol(ts).T  # (n_points, 12)
            except RuntimeError:
                continue  # skip an ejecting/collapsing perturbation
            if np.all(np.isfinite(ys)):
                trajs.append(ys)
        print(f"  source '{name}' (T={T:.3f}): ensemble now {len(trajs)} trajectories")

    trajs = np.stack(trajs)
    if cache:
        cache_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(cache_path, trajs=trajs, dt=dt)
    return jnp.asarray(trajs), float(dt)
