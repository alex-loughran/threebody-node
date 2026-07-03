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

EVERYTHING here is N-agnostic: the bead count is inferred from the length of the
state vector, and the bond / non-bonded structure is derived from the chain
ordering (bead i bonds to i+1). That is what lets a model trained at one N be
evaluated at another -- the size-generalisation experiment.

The Hamiltonian is defined once; the exact force field is obtained by autodiff
(J . grad H) so there is no hand-derived-gradient bug surface. Because H is a
sum of a per-bead kinetic term and PAIRWISE distance potentials, it is manifestly
invariant under a global rotation+translation of every bead -- i.e. the true
dynamics are SE(3)-equivariant.
"""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # energy-drift metric is tiny; need f64

# --- system definition (a chain of N_BEADS default; all functions accept other N)
N_BEADS = 5
DIM = 3
K_BOND = 20.0      # backbone stiffness
R0 = 1.0           # equilibrium bond length
LJ_EPS = 0.5       # non-bonded well depth
LJ_SIGMA = 1.0     # non-bonded length scale (min at 2^(1/6) sigma ~= 1.12)
LJ_SOFT = 0.5      # softening: r_eff = sqrt(r^2 + soft^2), caps the 1/r^12 core

STATE_DIM = 2 * N_BEADS * DIM  # = 30 for the default chain (fixed-dim models use this)


def _split(y):
    """flat y -> (q, p) each (N, DIM), with N inferred from the vector length."""
    n = y.shape[-1] // 2
    N = n // DIM
    return y[:n].reshape(N, DIM), y[n:].reshape(N, DIM)


def hamiltonian(y: jnp.ndarray) -> jnp.ndarray:
    """True energy H(q, p) = T(p) + V_bond(q) + V_nonbonded(q)  (scalar).

    Built purely from |p_i|^2 and pairwise distances |q_i - q_j|, hence invariant
    under any global SE(3) transform of the beads. N-agnostic: pairs and bond
    structure are derived from the chain ordering."""
    q, p = _split(y)
    N = q.shape[0]
    # kinetic (unit masses)
    T = 0.5 * jnp.sum(p * p)
    # all i<j pairs; a pair is a backbone bond iff the beads are consecutive
    ii, jj = jnp.triu_indices(N, k=1)
    d = jnp.linalg.norm(q[jj] - q[ii], axis=-1)
    is_bond = (jj == ii + 1)
    is_nb = (jj >= ii + 2)
    # harmonic backbone bonds
    V_bond = 0.5 * K_BOND * jnp.sum(jnp.where(is_bond, (d - R0) ** 2, 0.0))
    # softened Lennard-Jones on non-bonded pairs
    r = jnp.sqrt(d ** 2 + LJ_SOFT ** 2)
    sr6 = (LJ_SIGMA / r) ** 6
    lj = 4.0 * LJ_EPS * (sr6 ** 2 - sr6)
    V_nb = jnp.sum(jnp.where(is_nb, lj, 0.0))
    return T + V_bond + V_nb


def linear_momentum(y: jnp.ndarray) -> jnp.ndarray:
    """P = sum_i p_i  (3-vector). Conserved iff H is TRANSLATION invariant --
    which a distance-only potential is (internal pairwise forces cancel by
    Newton's third law), so the equivariant model conserves it structurally."""
    _, p = _split(y)
    return jnp.sum(p, axis=0)


def angular_momentum(y: jnp.ndarray) -> jnp.ndarray:
    """L = sum_i q_i x p_i  (3-vector). Conserved iff H is ROTATION invariant --
    the SO(3) half of the SE(3) symmetry. This is the rotational analogue of the
    machine-precision linear-momentum result on the 3-body figure-eight."""
    q, p = _split(y)
    return jnp.sum(jnp.cross(q, p), axis=0)


def _symplectic_grad(y):
    """J . grad H : dq/dt = dH/dp, dp/dt = -dH/dq (flattened)."""
    g = jax.grad(hamiltonian)(y)
    n = y.shape[-1] // 2
    return jnp.concatenate([g[n:], -g[:n]])


def true_vector_field(t, y, args=None):
    return _symplectic_grad(y)


def initial_condition(key, n_beads: int = N_BEADS, jitter: float = 0.15, p_scale: float = 0.4):
    """An extended chain along x (spacing R0) with small transverse jitter and
    small random momenta. Net (centre-of-mass) momentum is removed so the whole
    thing does not translate, and the chain is centred at the origin."""
    k_q, k_p = jax.random.split(key)
    base = jnp.stack([R0 * jnp.arange(n_beads),
                      jnp.zeros(n_beads),
                      jnp.zeros(n_beads)], axis=-1)          # (N, 3)
    q = base + jitter * jax.random.normal(k_q, (n_beads, DIM))
    p = p_scale * jax.random.normal(k_p, (n_beads, DIM))
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


def make_dataset(key, n_traj: int, t_span: float, n_points: int, n_beads: int = N_BEADS):
    """Returns trajs (n_traj, n_points, 6N) and shared time grid ts (n_points,)."""
    ts = jnp.linspace(0.0, t_span, n_points)
    keys = jax.random.split(key, n_traj)
    y0s = jax.vmap(lambda k: initial_condition(k, n_beads))(keys)
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
