"""Milestone 4D: push the model-accuracy floor down.

The recurring bottleneck across M3-M4B is the model-error floor (H_theta != true H):
it caps the 3-body rollout horizon AND caps 4B triage at "prune, can't pick". Two
levers, measured here:

  1. INDUCTIVE BIAS -- fix the kinetic energy to the exact T = 1/2|p|^2 instead of
     learning it (`FixedKineticPairwiseNODE`). Half the Hamiltonian becomes exact.
  2. DATA -- train the same model on growing amounts of engine data and watch the
     held-out floor. Falling floor => data-limited (engine data is cheap, so this
     is the good outcome); flat floor => architecture-limited.

We measure held-out short-horizon state RMSE (the floor), then re-run the 4B
triage on the best model to see whether "prune" moves toward "pick".

Run:  .venv/bin/python experiment_accuracy.py
"""
from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment import make_windows
from experiment_threebody import train
from src import threebody as tb
from src.integrate import batch_leapfrog, leapfrog_rollout
from src.models import FixedKineticPairwiseNODE, PairwiseHamiltonianNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


def held_out_set(vx, vy, T, dt, n_orbits=1, n_test=6):
    """A few unseen perturbed figure-eights + engine ground truth, ~n_orbits each."""
    ts = jnp.arange(int(n_orbits * T / dt)) * dt
    trues = [tb.reference_trajectory(vx, vy, ts, sigma=0.02, seed=5000 + i) for i in range(n_test)]
    return ts, jnp.stack(trues)


def floor_rmse(model, ts, trues):
    """Mean over held-out orbits of the short-horizon state RMSE (the accuracy floor)."""
    preds = batch_leapfrog(model, trues[:, 0], ts, n_substeps=4)
    return float(jnp.sqrt(jnp.mean((preds - trues) ** 2)))


def triage(model, vx, vy, T, M=256):
    """Compact 4B-style triage score on this model: Spearman + median-split AUC +
    pick-best P@10 for return proximity |y(T)-y(0)|."""
    ts = jnp.linspace(0.0, T, 120)
    ics = tb.perturbed_ics(vx, vy, M, sigma=0.03, seed=2024)
    pred = np.asarray(batch_leapfrog(model, ics, ts))
    true = np.stack([np.asarray(tb.integrate_state(ic, jnp.linspace(0.0, T, 60))) for ic in ics])
    ics_np = np.asarray(ics)
    rp_pred = np.linalg.norm(pred[:, -1] - ics_np, axis=1)
    rp_true = np.linalg.norm(true[:, -1] - ics_np, axis=1)
    ra, rb = np.argsort(np.argsort(rp_pred)), np.argsort(np.argsort(rp_true))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    worst = rp_true > np.median(rp_true)
    ranks = np.argsort(np.argsort(rp_pred)) + 1
    npos = int(worst.sum())
    auc = (ranks[worst].sum() - npos * (npos + 1) / 2) / (npos * (M - npos))
    pick10 = len(set(np.argsort(rp_pred)[:10]) & set(np.argsort(rp_true)[:10])) / 10
    return rho, float(auc), pick10, float(rp_pred.min())


def main():
    key = jax.random.PRNGKey(0)
    vx, vy, T, _ = tb.stable_symmetric_sources(1)[0]

    print("generating a large 3-body dataset ...")
    trajs, dt = tb.generate(seed=0, n_perturb=32, sigma=0.025, periods=2.0,
                            pts_per_period=100, n_sources=3)
    print(f"  {trajs.shape[0]} trajectories available")
    ts_win = jnp.arange(6) * dt
    ts_ho, trues = held_out_set(vx, vy, T, dt)

    # Iters scaled to data so every model is comparably trained (the earlier
    # fixed-budget run left the big dataset undertrained -> misleading scaling).
    ITERS = 6000
    runs = [
        ("fixed-kinetic  40", FixedKineticPairwiseNODE(jax.random.fold_in(key, 1), 3, 64), 40),
        ("fixed-kinetic  90", FixedKineticPairwiseNODE(jax.random.fold_in(key, 2), 3, 64), 90),
        ("learned-kinetic 40", PairwiseHamiltonianNODE(jax.random.fold_in(key, 3), 3, 64), 40),
    ]
    results = []
    best = (None, 1e9)
    for i, (name, model0, n) in enumerate(runs):
        y0s, targets = make_windows(trajs[:n], W=6)
        m = train(model0, y0s, targets, ts_win, iters=ITERS, batch=128, lr=2e-3, substeps=4, seed=10 + i)
        f = floor_rmse(m, ts_ho, trues)
        rho, auc, p10, rpf = triage(m, vx, vy, T)
        results.append((name, n, f, rho, auc, p10, rpf))
        print(f"  [{name}] floor RMSE {f:.3e}  | triage ρ={rho:.2f} AUC={auc:.2f} pick@10={p10:.2f}")
        if f < best[1] and "fixed" in name:
            best = (m, f)

    eqx.tree_serialise_leaves(f"{RESULTS}/threebody_model_v2.eqx", best[0])
    print(f"  saved best fixed-kinetic model -> {RESULTS}/threebody_model_v2.eqx  (floor {best[1]:.3e})")

    print("\n=== summary (old learned-kinetic baseline: floor ~0.22, pick@10=0.00) ===")
    print("  model                floor      ρ     AUC   pick@10")
    for name, n, f, rho, auc, p10, rpf in results:
        print(f"  {name:18s}  {f:.3e}  {rho:5.2f}  {auc:.2f}   {p10:.2f}")

    # --- plots: floor bar (fixed vs learned) + triage pick@10 progress ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    names = [r[0] for r in results]
    cols = ["tab:blue" if "fixed" in n else "tab:orange" for n in names]
    ax[0].bar(names, [r[2] for r in results], color=cols)
    ax[0].axhline(0.22, color="gray", ls="--", lw=1, label="old baseline floor")
    ax[0].set(title="Held-out 1-period state RMSE (accuracy floor)", ylabel="RMSE")
    ax[0].tick_params(axis="x", labelrotation=15); ax[0].legend()

    ax[1].bar(names, [r[5] for r in results], color=cols)
    ax[1].axhline(0.0, color="gray")
    ax[1].set(title="Triage pick@10 (0.00 for old baseline → prune-only)", ylabel="precision@10")
    ax[1].tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/accuracy.png", dpi=130)
    print(f"\nsaved {RESULTS}/accuracy.png")


if __name__ == "__main__":
    main()
