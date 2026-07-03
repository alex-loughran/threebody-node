"""Milestone 4B: the learned Hamiltonian NODE as a cheap surrogate integrator
for active-learning orbit search.

Thesis: in orbit hunting you scan many candidate initial conditions and keep the
near-periodic ones (small "return proximity" |y(T) - y(0)|). Calling the exact
DOP853 integrator on every candidate is the cost bottleneck. A learned symplectic
surrogate can integrate a whole BATCH of candidates in parallel (jax.vmap + jit)
and pre-rank them, so DOP853 only runs on the promising few.

This script measures the two things that decide whether that pays off:
  (1) SPEEDUP  -- batched surrogate throughput vs serial DOP853, same horizon.
  (2) TRIAGE FIDELITY -- does the surrogate's predicted return proximity RANK
      candidates the same way the true integrator does? (Spearman correlation +
      precision@k for picking the most-periodic candidates.)

Run:  .venv/bin/python experiment_surrogate.py
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

from src import threebody as tb
from src.integrate import batch_leapfrog
from src.models import PairwiseHamiltonianNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


def load_model(width=64):
    skel = PairwiseHamiltonianNODE(jax.random.PRNGKey(0), n_bodies=3, width=width, depth=2)
    return eqx.tree_deserialise_leaves(f"{RESULTS}/threebody_model.eqx", skel)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    model = load_model()
    vx, vy, T, _ = tb.stable_symmetric_sources(1)[0]
    M = 256                      # candidate initial conditions to triage
    n_pts = 60                   # samples over ~1 period (where the model is accurate)
    ts = jnp.linspace(0.0, T, n_pts)

    print(f"generating {M} candidate ICs (perturbed figure-eight) ...")
    ics = tb.perturbed_ics(vx, vy, M, sigma=0.03, seed=2024)

    # ---- ground truth: DOP853 per candidate (serial), timed --------------
    tb.integrate_state(ics[0], ts)          # warm up numba JIT before timing
    t0 = time.time()
    true = np.stack([np.asarray(tb.integrate_state(ic, ts)) for ic in ics])
    t_true = time.time() - t0

    # ---- surrogate: one batched leapfrog rollout, timed ------------------
    n_steps = 120                            # fixed-step budget for the surrogate
    ts_fine = jnp.linspace(0.0, T, n_steps)
    _ = batch_leapfrog(model, ics[:2], ts_fine).block_until_ready()   # warm up jit
    t0 = time.time()
    pred = batch_leapfrog(model, ics, ts_fine).block_until_ready()
    t_surr = time.time() - t0
    pred = np.asarray(pred)

    # ---- return proximity: |y(T) - y(0)| (small => near-periodic) --------
    ics_np = np.asarray(ics)
    rp_true = np.linalg.norm(true[:, -1, :] - ics_np, axis=1)
    rp_pred = np.linalg.norm(pred[:, -1, :] - ics_np, axis=1)

    rho = spearman(rp_pred, rp_true)

    def pick_best(k):   # find the MOST periodic (smallest rp): the hard task
        return len(set(np.argsort(rp_pred)[:k]) & set(np.argsort(rp_true)[:k])) / k

    # Coarse pruning: can the surrogate tell the least-periodic HALF from the rest?
    # AUC via the rank (Mann-Whitney) identity; then the practical statement:
    # if we drop the predicted-worst half, what fraction of the truly-best half survive?
    median = np.median(rp_true)
    worst = rp_true > median
    ranks = np.argsort(np.argsort(rp_pred)) + 1
    n_pos = int(worst.sum()); n_neg = M - n_pos
    auc = (ranks[worst].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    keep = set(np.argsort(rp_pred)[: M // 2])                 # surrogate keeps predicted-best half
    truly_best = set(np.argsort(rp_true)[: M // 2])
    recall_best = len(keep & truly_best) / len(truly_best)

    floor = float(rp_pred.min())
    print("\n=== surrogate vs DOP853 (256 candidates, ~1 period) ===")
    print(f"  DOP853  (serial)   : {t_true*1e3:7.1f} ms   ({t_true/M*1e3:.3f} ms/candidate)")
    print(f"  surrogate (batched): {t_surr*1e3:7.1f} ms   ({t_surr/M*1e3:.3f} ms/candidate)")
    print(f"  throughput speedup : {t_true/t_surr:6.1f}x   (CPU f64; a GPU would widen this)")
    print(f"  triage Spearman ρ  : {rho:.3f}    median-split AUC : {auc:.3f}")
    print(f"  COARSE prune: keep predicted-best half → retains {recall_best*100:.0f}% of truly-best half")
    print(f"  FINE pick-best P@10/@25: {pick_best(10):.2f} / {pick_best(25):.2f}  "
          f"(blind below rp floor {floor:.2f})")

    # ---- plots -----------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].bar(["DOP853\n(serial)", "surrogate\n(batched)"], [t_true * 1e3, t_surr * 1e3],
              color=["tab:orange", "tab:blue"])
    ax[0].set(title=f"Wall-clock for {M} candidates (~1 period)", ylabel="ms")
    ax[0].text(1, t_surr * 1e3, f"  {t_true/t_surr:.0f}× faster", va="bottom", ha="center")

    ax[1].scatter(rp_true, rp_pred, s=14, alpha=0.6, color="tab:blue")
    lim = max(rp_true.max(), rp_pred.max())
    ax[1].plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax[1].axhspan(0, floor, color="tab:red", alpha=0.08)
    ax[1].text(lim * 0.55, floor * 0.5, "surrogate blind below its accuracy floor",
               color="tab:red", fontsize=8, va="center")
    ax[1].set(title=f"Return-proximity triage  (ρ={rho:.2f}: prunes bad, can't pick best)",
              xlabel="true |y(T)−y(0)|  (DOP853)", ylabel="surrogate |y(T)−y(0)|")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/surrogate.png", dpi=130)
    print(f"\nsaved {RESULTS}/surrogate.png")


if __name__ == "__main__":
    main()
