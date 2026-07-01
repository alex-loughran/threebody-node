"""Ground-truth dynamics for a toy COARSE-GRAINED peptide (a bead-spring model).

This is the protein analogue of `physics.py`. Where the Kepler sandbox was a
single particle in a 1/r potential (2-D, integrable), here we have N beads in
3-D connected into a chain -- a minimal but genuine coarse-grained protein:

    * backbone      : harmonic bonds between consecutive beads  (chain topology)
    * non-bonded    : softened Lennard-Jones between all |i-j|>=2 pairs
                      -> excluded volume (repulsion) + weak attraction.
      The competition between stiff bonds wanting length r0 and LJ contacts
      wanting length ~2^(1/6)*sigma is exactly the *frustration* that makes a
      chain fold/collapse rather than sit still. That is the protein-relevant
      physics in the cheapest possible package.

State convention (mirrors the 2-D project, generalised to N beads in 3-D):
    y = [ q_0..q_{N-1} , p_0..p_{N-1} ]   flattened, length 6N
        q_i, p_i in R^3   (configuration block first, then momentum block)

The Hamiltonian is defined once; the exact force field is obtained by autodiff
(J . grad H) so there is no hand-derived-gradient bug surface. Because H is a
sum of a per-bead kinetic term and PAIRWISE distance potentials, it is manifestly
invariant under a global rotation+translation of every bead -- i.e. the true
dynamics are SE(3)-equivariant. That symmetry is the thing the equivariant model
gets for free and the plain model must waste capacity (and data) relearning.
"""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)  # energy-drift metric is tiny; need f64

# --- system definition (module-level constants; small enough to be explicit) --
N_BEADS = 5
DIM = 3
K_BOND = 20.0      # backbone stiffness
R0 = 1.0           # equilibrium bond length
LJ_EPS = 0.5       # non-bonded well depth
LJ_SIGMA = 1.0     # non-bonded length scale (min at 2^(1/6) sigma ~= 1.12)
LJ_SOFT = 0.5      # softening: r_eff = sqrt(r^2 + soft^2), caps the 1/r^12 core

STATE_DIM = 2 * N_BEADS * DIM  # = 30

# consecutive-bead index pairs (backbone bonds)
_BOND_I = np.arange(N_BEADS - 1)
_BOND_J = np.arange(1, N_BEADS)
# non-bonded pairs: all i<j with |i-j| >= 2
_nb = [(i, j) for i in range(N_BEADS) for j in range(i + 2, N_BEADS)]
_NB_I = jnp.array([i for i, _ in _nb])
_NB_J = jnp.array([j for _, j in _nb])
_BOND_I, _BOND_J = jnp.array(_BOND_I), jnp.array(_BOND_J)


def _split(y):
    """flat y -> (q, p) each (N_BEADS, DIM)."""
    n = N_BEADS * DIM
    return y[:n].reshape(N_BEADS, DIM), y[n:].reshape(N_BEADS, DIM)


def hamiltonian(y: jnp.ndarray) -> jnp.ndarray:
    """True energy H(q, p) = T(p) + V_bond(q) + V_nonbonded(q)  (scalar).

    Built purely from |p_i|^2 and pairwise distances |q_i - q_j|, hence invariant
    under any global SE(3) transform of the beads."""
    q, p = _split(y)
    # kinetic (unit masses)
    T = 0.5 * jnp.sum(p * p)
    # harmonic backbone bonds
    d_bond = jnp.linalg.norm(q[_BOND_J] - q[_BOND_I], axis=-1)
    V_bond = 0.5 * K_BOND * jnp.sum((d_bond - R0) ** 2)
    # softened Lennard-Jones on non-bonded pairs
    d_nb = jnp.linalg.norm(q[_NB_J] - q[_NB_I], axis=-1)
    r = jnp.sqrt(d_nb ** 2 + LJ_SOFT ** 2)
    sr6 = (LJ_SIGMA / r) ** 6
    V_nb = jnp.sum(4.0 * LJ_EPS * (sr6 ** 2 - sr6))
    return T + V_bond + V_nb


def _symplectic_grad(y):
    """J . grad H : dq/dt = dH/dp, dp/dt = -dH/dq (flattened)."""
    g = jax.grad(hamiltonian)(y)
    n = N_BEADS * DIM
    dH_dq, dH_dp = g[:n], g[n:]
    return jnp.concatenate([dH_dp, -dH_dq])


def true_vector_field(t, y, args=None):
    return _symplectic_grad(y)


def initial_condition(key, jitter: float = 0.15, p_scale: float = 0.4):
    """An extended chain along x (spacing R0) with small transverse jitter and
    small random momenta. Net (centre-of-mass) momentum is removed so the whole
    thing does not translate -- keeps the demo tidy and total momentum a clean
    conserved quantity."""
    k_q, k_p = jax.random.split(key)
    base = jnp.stack([R0 * jnp.arange(N_BEADS),
                      jnp.zeros(N_BEADS),
                      jnp.zeros(N_BEADS)], axis=-1)          # (N, 3)
    q = base + jitter * jax.random.normal(k_q, (N_BEADS, DIM))
    p = p_scale * jax.random.normal(k_p, (N_BEADS, DIM))
    p = p - jnp.mean(p, axis=0, keepdims=True)               # zero COM momentum
    q = q - jnp.mean(q, axis=0, keepdims=True)               # centre at origin
    return jnp.concatenate([q.reshape(-1), p.reshape(-1)])


def integrate_true(y0, ts):
    """High-accuracy reference trajectory (8th-order, tight tol) = our 'data'."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(true_vector_field),
        diffrax.Dopri8(),
        t0=float(ts[0]), t1=float(ts[-1]), dt0=float(ts[1] - ts[0]),
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-12),
        max_steps=1_000_000,
    )
    return sol.ys


def make_dataset(key, n_traj: int, t_span: float, n_points: int):
    """Returns trajs (n_traj, n_points, 6N) and shared time grid ts (n_points,)."""
    ts = jnp.linspace(0.0, t_span, n_points)
    keys = jax.random.split(key, n_traj)
    y0s = jax.vmap(lambda k: initial_condition(k))(keys)
    trajs = jax.vmap(lambda y0: integrate_true(y0, ts))(y0s)
    return trajs, ts


# --------------------------------------------------------------------------- #
# SE(3) helper -- used by the equivariance-generalisation test.
# --------------------------------------------------------------------------- #
def random_rotation(key):
    """A uniform random 3x3 rotation via QR of a Gaussian matrix (det fixed +1)."""
    a = jax.random.normal(key, (3, 3))
    Q, Rm = jnp.linalg.qr(a)
    Q = Q * jnp.sign(jnp.diag(Rm))                # make QR unique
    Q = Q * jnp.sign(jnp.linalg.det(Q))           # ensure det = +1 (rotation, not reflection)
    return Q


def apply_rotation(y, Rm):
    """Rotate every bead's position AND momentum by R (a global SE(3) action)."""
    q, p = _split(y)
    q = q @ Rm.T
    p = p @ Rm.T
    return jnp.concatenate([q.reshape(-1), p.reshape(-1)])
