"""The competing dynamics models.

PlainNODE               : learns the vector field f_theta(y) -> dy/dt directly.
HamiltonianNODE         : learns a SCALAR H_theta(y); the vector field is its
                          symplectic gradient, so energy conservation is
                          structural, not fitted.
SeparableHamiltonianNODE: learns H_theta = T_theta(p) + V_theta(q) with two nets,
                          the separable form an explicit symplectic integrator needs.

All expose the same `.vector_field(y)` interface so the integrator and training
loop are identical -- the ONLY difference between experiments is which class you
instantiate. That is deliberately the cleanest possible A/B test of "does the
physics inductive bias help?".

`dim` is the HALF-dimension (number of configuration coords): dim=2 for the
2-body relative problem, dim=6 for the planar 3-body problem. The full phase
state is length 2*dim, laid out [q (dim), p (dim)]. The q/p split at call time is
inferred from the input length, so the integrators need no dimension argument.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def _split(y):
    """Split a phase-space vector [q, p] into (q, p) at its midpoint."""
    d = y.shape[-1] // 2
    return y[:d], y[d:]


class PlainNODE(eqx.Module):
    """Vanilla Neural ODE: an MLP that maps state -> time-derivative of state."""

    mlp: eqx.nn.MLP

    def __init__(self, key, dim: int = 2, width: int = 64, depth: int = 3):
        self.mlp = eqx.nn.MLP(
            in_size=2 * dim,
            out_size=2 * dim,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,  # smooth -> well-behaved through the ODE solver
            key=key,
        )

    def vector_field(self, y: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(y)


class HamiltonianNODE(eqx.Module):
    """Hamiltonian Neural ODE: an MLP mapping state -> a scalar 'energy' H_theta,
    with dynamics dq/dt = dH/dp, dp/dt = -dH/dq. The field is a symplectic
    gradient of a single scalar, so the flow conserves H_theta (the model cannot
    represent a non-conservative force even if the data tried to push it there)."""

    mlp: eqx.nn.MLP

    def __init__(self, key, dim: int = 2, width: int = 64, depth: int = 3):
        self.mlp = eqx.nn.MLP(
            in_size=2 * dim,
            out_size="scalar",  # equinox returns a true scalar -> jax.grad gives R^{2 dim}
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            key=key,
        )

    def hamiltonian(self, y: jnp.ndarray) -> jnp.ndarray:
        return self.mlp(y)

    def vector_field(self, y: jnp.ndarray) -> jnp.ndarray:
        grad_H = jax.grad(self.hamiltonian)(y)
        dH_dq, dH_dp = _split(grad_H)
        return jnp.concatenate([dH_dp, -dH_dq])        # symplectic gradient J . grad H


class SeparableHamiltonianNODE(eqx.Module):
    """H_theta(q, p) = T_theta(p) + V_theta(q): kinetic and potential learned by
    SEPARATE networks. Because dq/dt = T'(p) depends only on p and dp/dt = -V'(q)
    depends only on q, the system is separable -- exactly the structure an
    explicit symplectic (leapfrog / Stoermer-Verlet) integrator requires.

    Two structures stack here: a separable model (the physically-correct
    kinetic+potential split, which the true gravitational H obeys) and a
    symplectic integrator. Together they remove *secular* energy drift entirely.
    """

    T_mlp: eqx.nn.MLP  # kinetic, p -> scalar
    V_mlp: eqx.nn.MLP  # potential, q -> scalar

    def __init__(self, key, dim: int = 2, width: int = 64, depth: int = 2):
        kT, kV = jax.random.split(key)
        mk = lambda k: eqx.nn.MLP(dim, "scalar", width, depth, activation=jax.nn.softplus, key=k)
        self.T_mlp, self.V_mlp = mk(kT), mk(kV)

    def kinetic(self, p):
        return self.T_mlp(p)

    def potential(self, q):
        return self.V_mlp(q)

    def hamiltonian(self, y):
        q, p = _split(y)
        return self.kinetic(p) + self.potential(q)

    def dT_dp(self, p):
        return jax.grad(self.kinetic)(p)

    def dV_dq(self, q):
        return jax.grad(self.potential)(q)

    def vector_field(self, y):
        """Provided so the SAME model can also be run through the non-symplectic
        integrator -- that lets us isolate the integrator's effect."""
        q, p = _split(y)
        return jnp.concatenate([self.dT_dp(p), -self.dV_dq(q)])


class PairwiseHamiltonianNODE(eqx.Module):
    """Physics-informed separable Hamiltonian for N gravitating bodies.

        H_theta = T_theta(p)  +  sum_{i<j} g_theta(|r_i - r_j|)

    The potential is a sum of IDENTICAL pairwise terms of the inter-body distance
    -- permutation-invariant, and (fed a 1/d feature) able to represent the sharp
    1/r close-approach force that a plain V_theta(q) over raw coordinates cannot.
    This is the architecture that makes Hamiltonian nets work on gravity. Still
    separable in (q, p), so leapfrog applies unchanged.
    """

    T_mlp: eqx.nn.MLP
    g_mlp: eqx.nn.MLP
    n_bodies: int = eqx.field(static=True)

    def __init__(self, key, n_bodies: int = 3, width: int = 64, depth: int = 2):
        kT, kg = jax.random.split(key)
        self.T_mlp = eqx.nn.MLP(2 * n_bodies, "scalar", width, depth,
                                activation=jax.nn.softplus, key=kT)
        # pair potential g(d): input features [d, 1/d] -> scalar
        self.g_mlp = eqx.nn.MLP(2, "scalar", width, depth,
                                activation=jax.nn.softplus, key=kg)
        self.n_bodies = n_bodies

    def kinetic(self, p):
        return self.T_mlp(p)

    def potential(self, q):
        r = q.reshape(self.n_bodies, 2)
        total = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                d = jnp.linalg.norm(r[i] - r[j])
                total = total + self.g_mlp(jnp.array([d, 1.0 / d]))
        return total

    def hamiltonian(self, y):
        q, p = _split(y)
        return self.kinetic(p) + self.potential(q)

    def dT_dp(self, p):
        return jax.grad(self.kinetic)(p)

    def dV_dq(self, q):
        return jax.grad(self.potential)(q)

    def vector_field(self, y):
        q, p = _split(y)
        return jnp.concatenate([self.dT_dp(p), -self.dV_dq(q)])


class FixedKineticPairwiseNODE(eqx.Module):
    """Same pairwise potential as `PairwiseHamiltonianNODE`, but the kinetic energy
    is FIXED to the exact T(p) = 1/2 |p|^2 (unit masses) instead of learned.

    We *know* the kinetic term exactly, so learning T_theta(p) only wastes
    capacity and injects error. Hard-coding it makes half the Hamiltonian exact,
    focuses all parameters on the hard part (the potential), and halves the
    parameter count -- a stronger, physically-correct inductive bias aimed
    squarely at lowering the model-error floor. Still separable -> leapfrog / logH
    apply unchanged.
    """

    g_mlp: eqx.nn.MLP
    n_bodies: int = eqx.field(static=True)

    def __init__(self, key, n_bodies: int = 3, width: int = 64, depth: int = 2):
        self.g_mlp = eqx.nn.MLP(2, "scalar", width, depth,
                                activation=jax.nn.softplus, key=key)
        self.n_bodies = n_bodies

    def kinetic(self, p):
        return 0.5 * jnp.dot(p, p)               # EXACT, not learned

    def potential(self, q):
        r = q.reshape(self.n_bodies, 2)
        total = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                d = jnp.linalg.norm(r[i] - r[j])
                total = total + self.g_mlp(jnp.array([d, 1.0 / d]))
        return total

    def hamiltonian(self, y):
        q, p = _split(y)
        return self.kinetic(p) + self.potential(q)

    def dT_dp(self, p):
        return p                                  # exact gradient of 1/2 |p|^2

    def dV_dq(self, q):
        return jax.grad(self.potential)(q)

    def vector_field(self, y):
        q, p = _split(y)
        return jnp.concatenate([self.dT_dp(p), -self.dV_dq(q)])
