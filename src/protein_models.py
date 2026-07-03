"""Three competing models for the coarse-grained peptide, in ascending order of
physical structure. Each exposes the SAME `.vector_field(y)` so the integrator
and training loop are identical -- only the class changes.

    PlainNODE3D              : MLP: R^6N -> R^6N. No structure. The foil.
    HamiltonianNODE3D        : MLP: R^6N -> scalar H_theta; dynamics = J.grad H.
                               Energy-conserving, but reads RAW coordinates, so it
                               is NOT rotation/translation invariant.
    EquivariantHamiltonianNODE : scalar H_theta built ONLY from invariant features
                               (pairwise distances, per-bead speeds). Its gradient
                               is therefore an equivariant force field -- energy
                               conservation AND SE(3)-equivariance from one autodiff.

The first two bake in the bead count (fixed MLP input dim) and can only ever run
at the N they were built for. The equivariant model has NO N in its weights
(f_kin is per-bead, g_pot is per-pair, H is a SUM), so the SAME parameters run on
any chain length -- and being separable H = T(p)+V(q) it also plugs straight into
the symplectic leapfrog integrator (it implements dT_dp / dV_dq like
SeparableHamiltonianNODE / PairwiseHamiltonianNODE).
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from . import protein as P

_N, _D = P.N_BEADS, P.DIM
_NQ = _N * _D  # length of the position block for the FIXED-dim models


def _split_fixed(y):
    return y[:_NQ].reshape(_N, _D), y[_NQ:].reshape(_N, _D)


class PlainNODE3D(eqx.Module):
    """Vanilla Neural ODE on the raw flattened state. Fixed input dim -> fixed N."""
    mlp: eqx.nn.MLP

    def __init__(self, key, width: int = 128, depth: int = 3):
        self.mlp = eqx.nn.MLP(P.STATE_DIM, P.STATE_DIM, width, depth,
                              activation=jax.nn.softplus, key=key)

    def vector_field(self, y):
        return self.mlp(y)


class HamiltonianNODE3D(eqx.Module):
    """Hamiltonian NODE: scalar energy from the RAW coordinates. Conserves its own
    H_theta, but because the MLP sees absolute positions/momenta it has no built-in
    notion that a rotated peptide has the same energy. Fixed input dim -> fixed N."""
    mlp: eqx.nn.MLP

    def __init__(self, key, width: int = 128, depth: int = 3):
        self.mlp = eqx.nn.MLP(P.STATE_DIM, "scalar", width, depth,
                              activation=jax.nn.softplus, key=key)

    def hamiltonian(self, y):
        return self.mlp(y)

    def vector_field(self, y):
        g = jax.grad(self.hamiltonian)(y)
        return jnp.concatenate([g[_NQ:], -g[:_NQ]])   # J . grad H


class EquivariantHamiltonianNODE(eqx.Module):
    """Scalar Hamiltonian assembled from SE(3)-INVARIANT features only:

        T_theta(p) = sum_i  f_theta(|p_i|^2)                    (per-bead, shared)
        V_theta(q) = sum_{i<j} g_theta(d_ij, is_bonded_ij)      (per-pair, shared)

    Both f and g read scalars that do not change under a global rotation or
    translation, so H_theta is invariant -> its symplectic gradient is equivariant.

    N-AGNOSTIC: the bead count is read from the vector length at call time, so the
    same weights evaluate any chain length (bonds = consecutive beads). SEPARABLE:
    T depends only on p, V only on q, so dT_dp / dV_dq exist and leapfrog applies.
    """
    f_kin: eqx.nn.MLP   # |p_i|^2 -> scalar kinetic contribution
    g_pot: eqx.nn.MLP   # [d_ij, bond_flag] -> scalar potential contribution

    def __init__(self, key, width: int = 64, depth: int = 2):
        kf, kg = jax.random.split(key)
        self.f_kin = eqx.nn.MLP(1, "scalar", width, depth, activation=jax.nn.softplus, key=kf)
        self.g_pot = eqx.nn.MLP(2, "scalar", width, depth, activation=jax.nn.softplus, key=kg)

    def kinetic(self, p):
        """p flat (length D*N) -> scalar. Shared net over each bead's speed^2."""
        pv = p.reshape(-1, _D)
        sp2 = jnp.sum(pv * pv, axis=-1, keepdims=True)        # (N, 1)
        return jnp.sum(jax.vmap(self.f_kin)(sp2))

    def potential(self, q):
        """q flat (length D*N) -> scalar. Shared net over each pair's (d, bond)."""
        qv = q.reshape(-1, _D)
        N = qv.shape[0]
        ii, jj = jnp.triu_indices(N, k=1)
        d = jnp.linalg.norm(qv[jj] - qv[ii], axis=-1, keepdims=True)   # (P, 1)
        bond = (jj == ii + 1).astype(d.dtype)[:, None]                 # (P, 1)
        feat = jnp.concatenate([d, bond], axis=-1)                     # (P, 2)
        return jnp.sum(jax.vmap(self.g_pot)(feat))

    def hamiltonian(self, y):
        n = y.shape[-1] // 2
        return self.kinetic(y[n:]) + self.potential(y[:n])

    def dT_dp(self, p):
        return jax.grad(self.kinetic)(p)

    def dV_dq(self, q):
        return jax.grad(self.potential)(q)

    def vector_field(self, y):
        n = y.shape[-1] // 2
        return jnp.concatenate([self.dT_dp(y[n:]), -self.dV_dq(y[:n])])   # J . grad H
