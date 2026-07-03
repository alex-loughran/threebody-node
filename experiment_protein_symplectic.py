"""SYMPLECTIC INTEGRATION of the learned peptide Hamiltonian.

The equivariant model is separable, H_theta = T_theta(p) + V_theta(q), so it drops
straight into the Stoermer-Verlet leapfrog (kick-drift-kick). This is the exact
milestone-3 story from the Kepler/three-body work, now on the protein model:

  * leapfrog is symplectic -> it conserves a SHADOW Hamiltonian, so the model's
    energy H_theta stays BOUNDED for arbitrarily long rollouts, even at a coarse
    step where an equally-cheap RK4 leaks energy secularly.

Two panels:
  (A) energy drift vs time at a coarse step -- leapfrog bounded, RK4 drifts.
  (B) step-size sweep -- final drift vs h; leapfrog stays flat where RK4 blows up.

Note we track H_theta (the LEARNED energy), because that is the invariant the
symplectic map protects. Whether H_theta approximates the true energy is the
separate modelling question answered by the other experiments.

Run:  .venv/bin/python experiment_protein_symplectic.py
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import protein as P
from src.integrate import leapfrog_rollout, rk4_rollout
from src.protein_models import EquivariantHamiltonianNODE
from experiment_protein import build_windows, train

N_STEPS = 700
COARSE_DT = 0.10       # bond period ~1, so this is ~10 steps/period: coarse but stable
T_LONG = 120.0         # long horizon to expose secular drift
DT_SWEEP = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13]


def energy_curve(model, y0, ts, rollout_fn):
    traj = rollout_fn(model, y0, ts)
    H = jax.vmap(model.hamiltonian)(traj)
    return jnp.abs((H - H[0]) / H[0])


def main():
    key = jax.random.PRNGKey(0)
    k_d, k_tr, k_ic = jax.random.split(key, 3)

    print("Training EquivariantHamiltonianNODE (5-bead) for the symplectic test ...")
    y0s, tgt, ts_win = build_windows(k_d, n_beads=5)
    model = train(EquivariantHamiltonianNODE(k_tr), y0s, tgt, ts_win, k_tr,
                  "equiv", n_steps=N_STEPS)

    y0 = P.initial_condition(k_ic, n_beads=5)

    # ---- (A) energy drift vs time at the coarse step ------------------------
    ts = jnp.arange(0.0, T_LONG, COARSE_DT)
    d_lf = energy_curve(model, y0, ts, leapfrog_rollout)
    d_rk = energy_curve(model, y0, ts, rk4_rollout)
    print(f"\n== (A) drift at h={COARSE_DT} over t={T_LONG} "
          f"({ts.shape[0]} steps) ==")
    print(f"  leapfrog  final |dH/H| = {float(d_lf[-1]):.2e}   max = {float(jnp.max(d_lf)):.2e}")
    print(f"  RK4       final |dH/H| = {float(d_rk[-1]):.2e}   max = {float(jnp.max(d_rk)):.2e}")

    # ---- (B) step-size sweep: final drift vs h ------------------------------
    print("\n== (B) step-size sweep (final |dH/H| over t=60) ==")
    lf_sweep, rk_sweep = [], []
    for h in DT_SWEEP:
        tsh = jnp.arange(0.0, 60.0, h)
        fl = float(energy_curve(model, y0, tsh, leapfrog_rollout)[-1])
        fr = float(energy_curve(model, y0, tsh, rk4_rollout)[-1])
        lf_sweep.append(fl); rk_sweep.append(fr)
        print(f"  h={h:.2f}:  leapfrog {fl:.2e}   RK4 {fr:.2e}")

    # ---- figure -------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))

    axA.semilogy(ts, d_lf + 1e-16, color="tab:green", lw=2, label="leapfrog (symplectic)")
    axA.semilogy(ts, d_rk + 1e-16, color="tab:red", lw=2, label="RK4 (non-symplectic)")
    axA.set_title(f"(A) Learned-energy drift at coarse step h={COARSE_DT}")
    axA.set_xlabel("time"); axA.set_ylabel("|ΔH_θ / H_θ|")
    axA.legend(); axA.grid(alpha=0.3)

    axB.loglog(DT_SWEEP, lf_sweep, "o-", color="tab:green", lw=2, label="leapfrog")
    axB.loglog(DT_SWEEP, rk_sweep, "s-", color="tab:red", lw=2, label="RK4")
    axB.set_title("(B) Final drift vs step size (t=60)")
    axB.set_xlabel("step size h"); axB.set_ylabel("final |ΔH_θ / H_θ|")
    axB.legend(); axB.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = "results/protein_symplectic.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
