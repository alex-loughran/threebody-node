"""SIZE GENERALISATION, and the fix for it.

The equivariant Hamiltonian NODE has NO bead count in its weights (f_kin is
per-bead, g_pot is per-pair, H is a SUM), so the SAME parameters run on any chain
length. Its fixed-dim cousins (PlainNODE3D / HamiltonianNODE3D) cannot even be
*called* at a new N -- their MLP input dimension is hard-wired to 6*5 = 30.

But "can be evaluated at any N" is not "is accurate at any N". A model trained on
5-bead chains only ever sees the distance/density distribution of 5-bead chains;
a longer chain folds denser and probes SHORTER pairwise distances, off the
training manifold -> g_pot extrapolates -> long rollouts destabilise.

THE FIX (this script): train on a RANGE of chain lengths (N = 4,5,6,7). Because
states of different N have different lengths, they cannot share one batch -- so we
keep a window bank per N and round-robin training steps across them. This widens
g_pot's feature coverage and should both flatten the transfer curve in-range and
tame extrapolation to N = 8, 10.

We compare single-N (train on 5) vs mixed-N (train on 4-7) head to head.

Run:  .venv/bin/python experiment_protein_scale.py
"""
from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

from src import protein as P
from src.integrate import rollout, rk4_rollout
from src.protein_models import EquivariantHamiltonianNODE, PlainNODE3D
from experiment_protein import build_windows, train, DT, BATCH, LR

MIX_NS = [4, 5, 6, 7]          # chain lengths seen during mixed-N training
TEST_NS = [4, 5, 6, 7, 8, 10]  # 8 and 10 are extrapolation for BOTH models
N_STEPS = 800


def train_mixed(model, banks, key, n_steps=N_STEPS, tag="mixed"):
    """Round-robin training across chain lengths. Each step draws a batch from one
    N's bank; the model is N-agnostic so the step fn just re-traces once per N."""
    opt = optax.adam(optax.exponential_decay(LR, n_steps, 0.1, end_value=LR * 0.05))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    Ns = sorted(banks)

    @eqx.filter_value_and_grad
    def loss_fn(m, yb, tb, ts_win):
        pred = jax.vmap(lambda y0: rk4_rollout(m, y0, ts_win))(yb)
        return jnp.mean((pred - tb) ** 2)

    @eqx.filter_jit
    def step(m, opt_state, yb, tb, ts_win):
        loss, grads = loss_fn(m, yb, tb, ts_win)
        updates, opt_state = opt.update(grads, opt_state, m)
        return eqx.apply_updates(m, updates), opt_state, loss

    for i in range(n_steps):
        N = Ns[i % len(Ns)]                    # round-robin over chain lengths
        y0s, tgt, ts_win = banks[N]
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (BATCH,), 0, y0s.shape[0])
        model, opt_state, loss = step(model, opt_state, y0s[idx], tgt[idx], ts_win)
        if i % 300 == 0 or i == n_steps - 1:
            print(f"  [{tag:>10}] step {i:4d}  (N={N})  loss {float(loss):.3e}")
    return model


def eval_at_N(model, N, key, t_pred=3.0, t_energy=15.0, n_ic=8):
    """Predictive MSE/DOF and energy drift, both averaged over n_ic test ICs so
    the metrics are robust to single-trajectory luck (an early lesson: single-IC
    long-horizon drift is noisy and seed-fragile -- use the median over ICs)."""
    y0s = jax.vmap(lambda k: P.initial_condition(k, N))(jax.random.split(key, n_ic))
    ts = jnp.arange(0.0, t_pred, DT)
    truth = jax.vmap(lambda y0: P.integrate_true(y0, ts))(y0s)
    pred = jax.vmap(lambda y0: rollout(model, y0, ts, adaptive=True, max_steps=100_000))(y0s)
    mse = float(jnp.mean((pred - truth) ** 2))

    te = jnp.arange(0.0, t_energy, DT)
    def drift_one(y0):
        traj = rollout(model, y0, te, adaptive=True, max_steps=300_000)
        H = jax.vmap(P.hamiltonian)(traj)
        return jnp.max(jnp.abs((H - H[0]) / H[0]))
    drifts = jax.vmap(drift_one)(y0s)
    return mse, float(jnp.median(drifts))


def sweep(model, key):
    mses, drifts = [], []
    for N in TEST_NS:
        m, d = eval_at_N(model, N, jax.random.fold_in(key, N))
        mses.append(m); drifts.append(d)
    return mses, drifts


def main():
    key = jax.random.PRNGKey(0)
    k_s, k_strain, k_m, k_ev = jax.random.split(key, 4)

    # --- single-N baseline: train on 5 only ----------------------------------
    print("Training SINGLE-N model (N=5 only) ...")
    y5, t5, tw5 = build_windows(k_s, n_beads=5)
    model_single = train(EquivariantHamiltonianNODE(k_strain), y5, t5, tw5, k_strain,
                         "single-N5", n_steps=N_STEPS)

    # --- mixed-N: one window bank per N, round-robin -------------------------
    print(f"\nTraining MIXED-N model (N={MIX_NS}) ...")
    banks = {N: build_windows(jax.random.fold_in(k_m, N), n_beads=N) for N in MIX_NS}
    model_mixed = train_mixed(EquivariantHamiltonianNODE(k_m), banks, k_m, tag="mixed")

    # --- the foil still can't run at a new N ---------------------------------
    try:
        PlainNODE3D(k_ev).vector_field(P.initial_condition(k_ev, n_beads=8))
        print("\nPlainNODE3D at N=8: (unexpectedly ran)")
    except Exception as e:
        print(f"\nPlainNODE3D at N=8: {type(e).__name__} -- fixed-dim model cannot transfer at all.")

    print("\n== single-N (train 5) vs mixed-N (train 4-7), zero-shot across N ==")
    mse_s, drift_s = sweep(model_single, k_ev)
    mse_m, drift_m = sweep(model_mixed, k_ev)
    print(f"  {'N':>3} | {'single MSE/DOF':>15} {'mixed MSE/DOF':>15} | "
          f"{'single drift':>13} {'mixed drift':>13}")
    for i, N in enumerate(TEST_NS):
        star = "*" if N in MIX_NS else " "
        print(f"  {N:>3}{star}| {mse_s[i]:>15.2e} {mse_m[i]:>15.2e} | "
              f"{drift_s[i]:>13.2e} {drift_m[i]:>13.2e}")
    print("  (* = inside the mixed-N training range)")

    # --- figure ---------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    lo, hi = min(MIX_NS), max(MIX_NS)

    for ax in (axL, axR):
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08, label="mixed-N train range")

    axL.plot(TEST_NS, mse_s, "o--", color="tab:gray", lw=2, label="single-N (train on 5)")
    axL.plot(TEST_NS, mse_m, "o-", color="tab:green", lw=2, label="mixed-N (train on 4-7)")
    axL.set_title("(1) Prediction error vs chain length")
    axL.set_xlabel("number of beads N"); axL.set_ylabel("MSE / DOF")
    axL.set_yscale("log"); axL.legend(); axL.grid(alpha=0.3)

    axR.plot(TEST_NS, drift_s, "s--", color="tab:gray", lw=2, label="single-N (train on 5)")
    axR.plot(TEST_NS, drift_m, "s-", color="tab:purple", lw=2, label="mixed-N (train on 4-7)")
    axR.set_title("(1) Long-horizon energy drift vs chain length")
    axR.set_xlabel("number of beads N"); axR.set_ylabel("|ΔE/E| over t=15")
    axR.set_yscale("log"); axR.legend(); axR.grid(alpha=0.3)

    fig.tight_layout()
    out = "results/protein_scale.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
