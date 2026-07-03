"""NOETHER: symmetry of the learned Hamiltonian -> conserved momenta.

Noether's theorem ties every continuous symmetry of H to a conserved quantity:
    translation invariance  ->  linear  momentum  P = sum_i p_i
    rotation    invariance  ->  angular momentum  L = sum_i q_i x p_i

The equivariant model builds H_theta from pairwise distances and speeds ONLY, so
it is exactly SE(3)-invariant -> it must conserve BOTH P and L along its flow, to
machine precision, by construction. This is the 3-D rotational analogue of the
machine-precision LINEAR-momentum result on the 3-body figure-eight.

The clean foil is HamiltonianNODE3D: it also conserves its own energy (it is a
Hamiltonian), but reads RAW coordinates, so it has NO spatial symmetry. Any gap in
momentum conservation between the two is therefore attributable to the symmetry
alone -- energy conservation is held fixed.

Run:  .venv/bin/python experiment_protein_noether.py
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import protein as P
from src.integrate import rollout
from src.protein_models import EquivariantHamiltonianNODE, HamiltonianNODE3D
from experiment_protein import build_windows, train, DT

N_STEPS = 700
T_LONG = 20.0


def conserved_curves(model, y0):
    """Accurate (adaptive) rollout, then track how far the model's own energy and
    the two momenta wander from their initial values."""
    ts = jnp.arange(0.0, T_LONG, DT)
    traj = rollout(model, y0, ts, adaptive=True, max_steps=300_000)
    H = jax.vmap(model.hamiltonian)(traj)
    Pm = jax.vmap(P.linear_momentum)(traj)         # (T, 3)
    Lm = jax.vmap(P.angular_momentum)(traj)        # (T, 3)
    dE = jnp.abs((H - H[0]) / H[0])
    dP = jnp.linalg.norm(Pm - Pm[0], axis=1)                       # abs: P0 ~= 0
    dL = jnp.linalg.norm(Lm - Lm[0], axis=1) / jnp.linalg.norm(Lm[0])
    return ts, dE, dP, dL


def main():
    key = jax.random.PRNGKey(0)
    k_d, k_e, k_h, k_ic = jax.random.split(key, 4)

    print("Training EquivariantHamiltonianNODE (SE(3)-invariant) ...")
    y0s, tgt, ts_win = build_windows(k_d, n_beads=5)
    equiv = train(EquivariantHamiltonianNODE(k_e), y0s, tgt, ts_win, k_e, "equiv", n_steps=N_STEPS)

    print("Training HamiltonianNODE3D (energy-conserving, NO symmetry) ...")
    ham = train(HamiltonianNODE3D(k_h), y0s, tgt, ts_win, k_h, "ham-raw", n_steps=N_STEPS)

    y0 = P.initial_condition(k_ic, n_beads=5)
    ts, dE_e, dP_e, dL_e = conserved_curves(equiv, y0)
    _,  dE_h, dP_h, dL_h = conserved_curves(ham, y0)

    print(f"\n== conservation over t={T_LONG} (final values) ==")
    print(f"  {'model':>14} | {'energy |dE/E|':>14} {'|dP| (abs)':>12} {'|dL|/|L0|':>12}")
    print(f"  {'Equivariant':>14} | {float(dE_e[-1]):>14.2e} {float(dP_e[-1]):>12.2e} {float(dL_e[-1]):>12.2e}")
    print(f"  {'HamiltonianRaw':>14} | {float(dE_h[-1]):>14.2e} {float(dP_h[-1]):>12.2e} {float(dL_h[-1]):>12.2e}")
    print("  (both conserve their OWN energy; only the symmetric model conserves P and L)")

    # ---- figure -------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    fig, (axP, axL) = plt.subplots(1, 2, figsize=(12, 4.5))
    ge, gh = "tab:green", "tab:orange"

    axP.semilogy(ts, dP_e + 1e-18, color=ge, lw=2, label="Equivariant (SE(3)-invariant)")
    axP.semilogy(ts, dP_h + 1e-18, color=gh, lw=2, label="Hamiltonian (raw coords)")
    axP.set_title("(2) Linear momentum drift  |P(t) − P₀|")
    axP.set_xlabel("time"); axP.set_ylabel("|ΔP|")
    axP.legend(); axP.grid(alpha=0.3)

    axL.semilogy(ts, dL_e + 1e-18, color=ge, lw=2, label="Equivariant (SE(3)-invariant)")
    axL.semilogy(ts, dL_h + 1e-18, color=gh, lw=2, label="Hamiltonian (raw coords)")
    axL.set_title("(2) Angular momentum drift  |L(t) − L₀| / |L₀|")
    axL.set_xlabel("time"); axL.set_ylabel("|ΔL| / |L₀|")
    axL.legend(); axL.grid(alpha=0.3)

    fig.tight_layout()
    out = "results/protein_noether.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
