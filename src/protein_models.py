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

The three-way comparison isolates the two inductive biases: Plain->Hamiltonian
adds energy conservation; Hamiltonian->Equivariant adds the symmetry. That lets us
attribute each plot to a specific piece of physics.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from . import protein as P

_N, _D = P.N_BEADS, P.DIM
_NQ = _N * _D  # length of the position block

# Pair topology is a FIXED property of the system, not a parameter -- keep it at
# module level (as physics.py does) so it never becomes a trainable model leaf.
_PAIRS = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
_BONDS = {(i, i + 1) for i in range(_N - 1)}
_PAIR_I = jnp.array([i for i, _ in _PAIRS])
_PAIR_J = jnp.array([j for _, j in _PAIRS])
_BOND_FLAG = jnp.array([[1.0 if (i, j) in _BONDS else 0.0] for i, j in _PAIRS])  # (P,1)


def _split(y):
    return y[:_NQ].reshape(_N, _D), y[_NQ:].reshape(_N, _D)


class PlainNODE3D(eqx.Module):
    """Vanilla Neural ODE on the raw flattened state."""
    mlp: eqx.nn.MLP

    def __init__(self, key, width: int = 128, depth: int = 3):
        self.mlp = eqx.nn.MLP(P.STATE_DIM, P.STATE_DIM, width, depth,
                              activation=jax.nn.softplus, key=key)

    def vector_field(self, y):
        return self.mlp(y)


class HamiltonianNODE3D(eqx.Module):
    """Hamiltonian NODE: scalar energy from the RAW coordinates. Conserves its own
    H_theta, but because the MLP sees absolute positions/momenta it has no built-in
    notion that a rotated peptide has the same energy."""
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
    The per-bead / per-pair *sums* also make the model permutation-structured and
    trivially size-generalisable (add beads, same weights) -- the seed of a real
    equivariant-GNN force field, minus the machinery.
    """
    f_kin: eqx.nn.MLP   # |p_i|^2 -> scalar kinetic contribution
    g_pot: eqx.nn.MLP   # [d_ij, bond_flag] -> scalar potential contribution
    bond_flag: jnp.ndarray   # (n_pairs,) 1.0 if the pair is a backbone bond
    idx_i: jnp.ndarray
    idx_j: jnp.ndarray

    def __init__(self, key, width: int = 64, depth: int = 2):
        kf, kg = jax.random.split(key)
        self.f_kin = eqx.nn.MLP(1, "scalar", width, depth, activation=jax.nn.softplus, key=kf)
        self.g_pot = eqx.nn.MLP(2, "scalar", width, depth, activation=jax.nn.softplus, key=kg)
        # enumerate ALL i<j pairs once; tag which are backbone bonds
        pairs = [(i, j) for i in range(_N) for j in range(i + 1, _N)]
        bonds = {(int(i), int(i + 1)) for i in range(_N - 1)}
        self.idx_i = jnp.array([i for i, _ in pairs])
        self.idx_j = jnp.array([j for _, j in pairs])
        self.bond_flag = jnp.array([1.0 if (i, j) in bonds else 0.0 for i, j in pairs])

    def hamiltonian(self, y):
        q, p = _split(y)
        # kinetic: shared net over each bead's speed^2
        sp2 = jnp.sum(p * p, axis=-1, keepdims=True)          # (N, 1)
        T = jnp.sum(jax.vmap(self.f_kin)(sp2))
        # potential: shared net over each pair's (distance, bond_flag)
        d = jnp.linalg.norm(q[self.idx_j] - q[self.idx_i], axis=-1, keepdims=True)  # (P,1)
        feat = jnp.concatenate([d, self.bond_flag[:, None]], axis=-1)               # (P,2)
        V = jnp.sum(jax.vmap(self.g_pot)(feat))
        return T + V

    def vector_field(self, y):
        g = jax.grad(self.hamiltonian)(y)
        return jnp.concatenate([g[_NQ:], -g[:_NQ]])   # J . grad H
