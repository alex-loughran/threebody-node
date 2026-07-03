"""Milestone 4A: beating the close-approach wall with time-transformed
(logarithmic-Hamiltonian) symplectic integration.

Fixed-step leapfrog uses uniform REAL-time steps, so it under-resolves close
approaches (perihelion / near-collisions) where the dynamics are fast -- the
source of the energy spikes that limited the 3-body rollouts. The logH leapfrog
(Preto & Tremaine 1999; Mikkola & Tanikawa 1999) integrates in a fictitious time
s with dt/ds = 1/(-U(q)), auto-shrinking steps at close approaches while staying
symplectic (it's a coordinate change, not an ad-hoc step controller).

Part 1 -- rigorous validation on TRUE Kepler across eccentricity (no learning):
         fixed-step leapfrog degrades sharply with e; logH stays flat.
Part 2 -- the same integrator applied to the LEARNED 3-body model: close-approach
         energy spikes tamed on a held-out figure-eight.

Run:  .venv/bin/python experiment_adaptive.py
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import physics
from src import threebody as tb
from src.integrate import leapfrog_rollout, logh_leapfrog_rollout
from src.models import PairwiseHamiltonianNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


class TrueKepler:
    """Exact 2-body Kepler, with the separable interface logH needs."""
    def kinetic(self, p): return 0.5 * jnp.dot(p, p)
    def potential(self, q): return -1.0 / jnp.linalg.norm(q)
    def dT_dp(self, p): return p
    def dV_dq(self, q): return q / jnp.linalg.norm(q) ** 3
    def vector_field(self, y):
        q, p = y[:2], y[2:]
        return jnp.concatenate([p, -q / jnp.linalg.norm(q) ** 3])


def _tune_ds(model, y0, n_steps, target_t):
    """Pick the fictitious step so logH reaches ~target real time in n_steps
    (reached time is ~linear in ds); two rescales suffice."""
    ds = target_t / n_steps
    for _ in range(3):
        _, ts = logh_leapfrog_rollout(model, y0, n_steps, ds)
        reached = float(ts[-1])
        if reached <= 0 or not np.isfinite(reached):
            break
        ds *= target_t / reached
    return ds


def emax(traj, E0):
    E = jnp.array([physics.hamiltonian(y) for y in traj])
    return float(jnp.max(jnp.abs(E - E0))) / abs(E0)


def part1_eccentricity_sweep():
    m = TrueKepler()
    eccs = [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    n_steps, n_orbits = 3000, 3
    fixed_drift, logh_drift = [], []
    print("=== Part 1: true Kepler, fixed-step vs logH leapfrog (3 orbits, 3000 steps) ===")
    print("   e     fixed ΔE/|E0|     logH ΔE/|E0|")
    for e in eccs:
        a = 1.0 / (1.0 - e)                 # perihelion r0=1
        T = float(2 * jnp.pi * a ** 1.5)
        y0 = physics.initial_condition(1.0, e)
        E0 = float(physics.hamiltonian(y0))
        # fixed-step leapfrog, uniform real time over n_orbits
        ts = jnp.arange(n_steps) * (n_orbits * T / n_steps)
        lf = leapfrog_rollout(m, y0, ts, n_substeps=1)
        df = emax(lf, E0)
        # logH leapfrog, same step budget, tuned to the same horizon
        ds = _tune_ds(m, y0, n_steps, n_orbits * T)
        ys, _ = logh_leapfrog_rollout(m, y0, n_steps, ds)
        dl = emax(ys, E0)
        fixed_drift.append(df); logh_drift.append(dl)
        print(f"  {e:.2f}   {df:.3e}      {dl:.3e}")
    return eccs, fixed_drift, logh_drift


def load_model(width=64):
    path = f"{RESULTS}/threebody_model.eqx"
    if not os.path.exists(path):
        return None
    skeleton = PairwiseHamiltonianNODE(jax.random.PRNGKey(0), n_bodies=3, width=width, depth=2)
    return eqx.tree_deserialise_leaves(path, skeleton)


def part2_learned_threebody():
    model = load_model()
    if model is None:
        print("\n=== Part 2: skipped (no results/threebody_model.eqx yet) ===")
        return None
    print("\n=== Part 2: learned 3-body model, fixed leapfrog vs logH (held-out figure-eight) ===")
    vx, vy, T, _ = tb.stable_symmetric_sources(1)[0]
    dt = T / 100.0
    n_fixed = 900                                  # ~9 periods of real-time steps
    ts = jnp.arange(n_fixed) * dt
    true = tb.reference_trajectory(vx, vy, ts, sigma=0.015, seed=999)
    y0 = true[0]
    E0 = float(tb.energy(y0))

    lf = leapfrog_rollout(model, y0, ts, n_substeps=1)
    ds = _tune_ds(model, y0, n_fixed, float(ts[-1]))
    ys, tt = logh_leapfrog_rollout(model, y0, n_fixed, ds)

    e3 = lambda tr: np.asarray(jax.vmap(tb.energy)(tr)) - E0
    dfix = float(np.max(np.abs(e3(lf)))) / abs(E0)
    dlog = float(np.max(np.abs(e3(ys)))) / abs(E0)
    print(f"  horizon ~{float(ts[-1])/T:.1f} orbits, {n_fixed} steps each")
    print(f"  fixed leapfrog  max ΔE/|E0| = {dfix:.3e}")
    print(f"  logH  leapfrog  max ΔE/|E0| = {dlog:.3e}  (reached {float(tt[-1])/T:.1f} orbits)")
    return (np.asarray(ts) / T, e3(lf)), (np.asarray(tt) / T, e3(ys))


def main():
    eccs, fixed_drift, logh_drift = part1_eccentricity_sweep()
    p2 = part2_learned_threebody()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].semilogy(eccs, fixed_drift, "o-", color="tab:orange", label="fixed-step leapfrog")
    ax[0].semilogy(eccs, logh_drift, "o-", color="tab:blue", label="logH leapfrog")
    ax[0].set(title="True Kepler: energy drift vs eccentricity\n(3 orbits, equal step budget)",
              xlabel="eccentricity e", ylabel="max ΔE/|E0|")
    ax[0].grid(True, which="both", alpha=0.3); ax[0].legend()

    if p2 is not None:
        (t_lf, e_lf), (t_lh, e_lh) = p2
        ax[1].plot(t_lf, e_lf, "tab:orange", lw=1.0, label="fixed leapfrog")
        ax[1].plot(t_lh, e_lh, "tab:blue", lw=1.0, label="logH leapfrog")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set(title="Learned 3-body: energy drift over rollout\n(held-out figure-eight)",
                  xlabel="orbits", ylabel="ΔE")
        ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "train 3-body model first\n(experiment_threebody.py)",
                   ha="center", va="center")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/adaptive.png", dpi=130)
    print(f"\nsaved {RESULTS}/adaptive.png")


if __name__ == "__main__":
    main()
