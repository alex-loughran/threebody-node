"""Milestone 4B, closing the economic case: the surrogate as stage 1 of a
TWO-STAGE cascade for orbit search.

`experiment_surrogate.py` established the pieces: the batched surrogate is ~5x
faster than serial DOP853 and ranks candidates by return-proximity with Spearman
~0.58 -- good enough to PRUNE bad candidates, too coarse to PICK the best (it is
blind below its own ~0.5 accuracy floor). The decision that actually matters for
the active loop is therefore not either metric alone but their product:

    cascade:  surrogate ranks all M  ->  DOP853 refines only the top-K
    question: total wall-clock vs. RECALL of the truly-best orbits, as K varies.

Because stage 2 runs the EXACT integrator on the kept set, the cascade recovers a
truly-best orbit iff stage 1 KEPT it -- so recall@n(K) is just the fraction of the
n most-periodic orbits that fall in the surrogate's top-K. Cost is modelled from
the measured per-candidate DOP853 time and the measured batched-surrogate time:

    cost(K) = t_surrogate_batch  +  K * t_dop853_per_candidate
    baseline (DOP853 on all M)   =  M * t_dop853_per_candidate   (recall = 1)

Sweeping K traces a cost-vs-recall Pareto curve; its knee is the operating point.

Run:  .venv/bin/python experiment_surrogate_cascade.py
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import threebody as tb
from src.integrate import batch_leapfrog
from experiment_surrogate import load_model, spearman

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


def main():
    model = load_model()
    vx, vy, T, _ = tb.stable_symmetric_sources(1)[0]
    M = 256
    ts = jnp.linspace(0.0, T, 60)
    ts_fine = jnp.linspace(0.0, T, 120)
    targets = [5, 10, 25]                     # "the n most-periodic orbits" we want to recover

    print(f"generating {M} candidate ICs (perturbed figure-eight) ...")
    ics = tb.perturbed_ics(vx, vy, M, sigma=0.03, seed=2024)
    ics_np = np.asarray(ics)

    # ---- stage-2 unit cost: DOP853 per candidate (serial, timed) ------------
    tb.integrate_state(ics[0], ts)            # warm numba
    t0 = time.time()
    true = np.stack([np.asarray(tb.integrate_state(ic, ts)) for ic in ics])
    t_dop_total = time.time() - t0
    per = t_dop_total / M
    rp_true = np.linalg.norm(true[:, -1, :] - ics_np, axis=1)

    # ---- stage-1 cost: one batched surrogate rollout (timed) ----------------
    batch_leapfrog(model, ics[:2], ts_fine).block_until_ready()   # warm jit
    t0 = time.time()
    pred = np.asarray(batch_leapfrog(model, ics, ts_fine).block_until_ready())
    t_surr = time.time() - t0
    rp_pred = np.linalg.norm(pred[:, -1, :] - ics_np, axis=1)

    order_pred = np.argsort(rp_pred)          # surrogate ranking (best first)
    rho = spearman(rp_pred, rp_true)
    print(f"stage-1 surrogate: {t_surr*1e3:.0f} ms batch, ρ={rho:.2f}   "
          f"stage-2 DOP853: {per*1e3:.2f} ms/candidate\n")

    # ---- cascade sweep over keep-size K -------------------------------------
    Ks = np.unique(np.clip(np.round(np.linspace(5, M, 40)).astype(int), 1, M))
    cost = t_surr + Ks * per                              # seconds
    baseline = M * per
    recall = {n: [] for n in targets}
    for K in Ks:
        kept = set(order_pred[:K].tolist())
        for n in targets:
            best_n = set(np.argsort(rp_true)[:n].tolist())
            recall[n].append(len(best_n & kept) / n)
    recall = {n: np.array(v) for n, v in recall.items()}

    # ---- headline: cheapest cascade reaching >=90% recall of best-10 --------
    n0 = 10
    ok = recall[n0] >= 0.9
    print("=== two-stage cascade (surrogate prune -> DOP853 refine top-K) ===")
    print(f"  baseline DOP853-on-all-{M}: {baseline*1e3:.0f} ms  (recall 1.00 by definition)")
    if ok.any():
        i = np.argmax(ok)                                # first K hitting 90%
        print(f"  90% recall of best-{n0} at K={Ks[i]}  ->  {cost[i]*1e3:.0f} ms "
              f"= {cost[i]/baseline*100:.0f}% of baseline  ({baseline/cost[i]:.2f}x faster)")
    else:
        print(f"  90% recall of best-{n0} not reached below K=M (surrogate too coarse)")
    for n in targets:
        # recall retained if we keep just half the candidates
        half = np.searchsorted(Ks, M // 2)
        print(f"  keep top-{M//2:>3} (50%): recall of best-{n:<2} = {recall[n][half]:.2f}")
    print(f"  NOTE: on CPU the fixed stage-1 cost ({t_surr*1e3:.0f} ms) dominates; on GPU it "
          f"collapses toward 0, pushing the curve to ~K*{per*1e3:.1f}ms and widening the win.")

    # ---- plot: cost vs recall Pareto ----------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {5: "tab:green", 10: "tab:blue", 25: "tab:purple"}
    for n in targets:
        ax[0].plot(cost * 1e3, recall[n], "o-", ms=3, color=colors[n], label=f"best-{n}")
    ax[0].axvline(baseline * 1e3, color="tab:orange", ls="--", lw=1.5, label=f"DOP853 all-{M}")
    ax[0].axhline(0.9, color="gray", ls=":", lw=1)
    ax[0].set(title="Cascade cost vs recall (sweep keep-size K)",
              xlabel="total wall-clock (ms)", ylabel="recall of truly-best-n")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(Ks / M * 100, recall[10], "o-", ms=3, color="tab:blue", label="recall best-10")
    ax[1].plot(Ks / M * 100, cost / baseline, "s-", ms=3, color="tab:red", label="cost / baseline")
    ax[1].axhline(1.0, color="tab:orange", ls="--", lw=1)
    ax[1].set(title="Recall & relative cost vs keep-fraction",
              xlabel="keep top-K (% of candidates)", ylabel="fraction")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{RESULTS}/surrogate_cascade.png", dpi=130)
    print(f"\nsaved {RESULTS}/surrogate_cascade.png")


if __name__ == "__main__":
    main()
