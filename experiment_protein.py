"""Coarse-grained peptide prototype: does SE(3)-equivariance (on top of a
Hamiltonian structure) pay off for learning protein-like dynamics?

Three models, identical training, only the class differs:
    PlainNODE3D  <  HamiltonianNODE3D  <  EquivariantHamiltonianNODE
                    (+energy conservation)  (+SE(3) symmetry)

Two questions, two plots:
  (A) ENERGY DRIFT over a long rollout -- does the learned flow keep the TRUE
      energy bounded? (Replicates the Kepler finding in 3-D / N-body.)
  (B) EQUIVARIANCE GENERALISATION -- train on ONE orientation, test on globally
      ROTATED initial conditions. The equivariant model should be identical on
      rotated data (that is the entire point); the others degrade.

Run:  .venv/bin/python experiment_protein.py
"""
from __future__ import annotations

import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax

from src import protein as P
from src.integrate import rollout
from src.protein_models import (
    PlainNODE3D, HamiltonianNODE3D, EquivariantHamiltonianNODE,
)

DT = 0.05
WINDOW = 10          # training rollout horizon (steps)
N_TRAIN_TRAJ = 16
T_TRAIN = 6.0
N_STEPS = 1500
BATCH = 64
LR = 2e-3


# --------------------------------------------------------------------------- #
# Data: reference trajectories -> flat bank of short (y0, target) windows.
# --------------------------------------------------------------------------- #
def build_windows(key):
    n_pts = int(T_TRAIN / DT) + 1
    trajs, _ = P.make_dataset(key, N_TRAIN_TRAJ, T_TRAIN, n_pts)   # (M, n_pts, 6N)
    ts_win = jnp.arange(WINDOW + 1) * DT
    starts = jnp.arange(n_pts - WINDOW)
    # gather every window from every trajectory
    def windows_of(traj):
        return jax.vmap(lambda s: jax.lax.dynamic_slice_in_dim(traj, s, WINDOW + 1))(starts)
    wins = jax.vmap(windows_of)(trajs)                            # (M, n_start, W+1, 6N)
    wins = wins.reshape(-1, WINDOW + 1, P.STATE_DIM)
    return wins[:, 0], wins, ts_win                              # y0, target-seq, local ts


# --------------------------------------------------------------------------- #
# Training (generic over model).
# --------------------------------------------------------------------------- #
def train(model, y0s, targets, ts_win, key, tag):
    opt = optax.adam(optax.exponential_decay(LR, N_STEPS, 0.1, end_value=LR * 0.05))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))
    n = y0s.shape[0]

    @eqx.filter_value_and_grad
    def loss_fn(m, yb, tb):
        pred = jax.vmap(lambda y0: rollout(m, y0, ts_win, max_steps=256))(yb)
        return jnp.mean((pred - tb) ** 2)

    @eqx.filter_jit
    def step(m, opt_state, yb, tb):
        loss, grads = loss_fn(m, yb, tb)
        updates, opt_state = opt.update(grads, opt_state, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    t0 = time.time()
    for i in range(N_STEPS):
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (BATCH,), 0, n)
        model, opt_state, loss = step(model, opt_state, y0s[idx], targets[idx])
        if i % 300 == 0 or i == N_STEPS - 1:
            print(f"  [{tag:>10}] step {i:4d}  loss {float(loss):.3e}")
    print(f"  [{tag:>10}] trained in {time.time() - t0:.1f}s")
    return model


# --------------------------------------------------------------------------- #
# Metric A: true-energy drift along a long model rollout.
# --------------------------------------------------------------------------- #
def energy_drift(model, y0, t_end=20.0):
    ts = jnp.arange(0.0, t_end, DT)
    traj = rollout(model, y0, ts, adaptive=True, max_steps=200_000)
    H = jax.vmap(P.hamiltonian)(traj)
    return ts, jnp.abs((H - H[0]) / H[0])


# --------------------------------------------------------------------------- #
# Metric B: short-horizon prediction error, canonical vs globally-rotated ICs.
# --------------------------------------------------------------------------- #
def rollout_mse(model, y0s, ts_eval):
    truth = jax.vmap(lambda y0: P.integrate_true(y0, ts_eval))(y0s)
    pred = jax.vmap(lambda y0: rollout(model, y0, ts_eval, adaptive=True, max_steps=100_000))(y0s)
    return float(jnp.mean((pred - truth) ** 2))


def equivariance_error(model, y0, Rm, ts_eval):
    """||rollout(R.y0) - R.rollout(y0)||: 0 iff the learned flow commutes with
    the rotation. A pure property test, independent of accuracy vs truth."""
    roll = lambda y: rollout(model, y, ts_eval, adaptive=True, max_steps=100_000)
    a = roll(P.apply_rotation(y0, Rm))
    b = jax.vmap(lambda y: P.apply_rotation(y, Rm))(roll(y0))
    return float(jnp.sqrt(jnp.mean((a - b) ** 2)))


def main():
    key = jax.random.PRNGKey(0)
    k_data, k_p, k_h, k_e, k_test, k_rot = jax.random.split(key, 6)

    print("Generating reference data ...")
    y0s, targets, ts_win = build_windows(k_data)
    print(f"  {y0s.shape[0]} training windows of horizon {WINDOW} steps "
          f"({P.N_BEADS} beads, {P.STATE_DIM}-D state)\n")

    models = {}
    print("Training PlainNODE3D (no structure) ...")
    models["Plain"] = train(PlainNODE3D(k_p), y0s, targets, ts_win, k_p, "Plain")
    print("Training HamiltonianNODE3D (energy-conserving, not equivariant) ...")
    models["Hamiltonian"] = train(HamiltonianNODE3D(k_h), y0s, targets, ts_win, k_h, "Hamiltonian")
    print("Training EquivariantHamiltonianNODE (energy + SE(3)) ...")
    models["Equivariant"] = train(EquivariantHamiltonianNODE(k_e), y0s, targets, ts_win, k_e, "Equivariant")

    # ---- test initial conditions (unseen) and their rotated copies ----------
    test_keys = jax.random.split(k_test, 12)
    test_y0s = jax.vmap(lambda k: P.initial_condition(k))(test_keys)
    Rm = P.random_rotation(k_rot)
    rot_y0s = jax.vmap(lambda y: P.apply_rotation(y, Rm))(test_y0s)
    ts_eval = jnp.arange(0.0, 3.0, DT)

    print("\n== Metric B: prediction MSE (canonical vs rotated test ICs) ==")
    print("== plus equivariance error ||f(R.y) - R.f(y)|| ==")
    bar_canon, bar_rot = {}, {}
    for name, m in models.items():
        mse_c = rollout_mse(m, test_y0s, ts_eval)
        mse_r = rollout_mse(m, rot_y0s, ts_eval)
        eqv = equivariance_error(m, test_y0s[0], Rm, ts_eval)
        bar_canon[name], bar_rot[name] = mse_c, mse_r
        print(f"  {name:>12}:  canonical {mse_c:.2e}   rotated {mse_r:.2e}   "
              f"(x{mse_r/mse_c:5.1f})   equiv-err {eqv:.2e}")

    # ---- Metric A: energy drift on one test IC ------------------------------
    print("\n== Metric A: true-energy drift over a long rollout ==")
    drift = {name: energy_drift(m, test_y0s[0]) for name, m in models.items()}
    for name, (ts, d) in drift.items():
        print(f"  {name:>12}:  final |dE/E| = {float(d[-1]):.2e}")

    # ---- figure -------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"Plain": "tab:red", "Hamiltonian": "tab:orange", "Equivariant": "tab:green"}

    for name, (ts, d) in drift.items():
        axA.semilogy(ts, d + 1e-16, color=colors[name], label=name, lw=2)
    axA.set_title("(A) True-energy drift along learned rollout")
    axA.set_xlabel("time"); axA.set_ylabel("|ΔE / E|")
    axA.legend(); axA.grid(alpha=0.3)

    names = list(models)
    x = jnp.arange(len(names))
    w = 0.35
    axB.bar([i - w/2 for i in x], [bar_canon[n] for n in names], w,
            label="canonical ICs", color="tab:blue")
    axB.bar([i + w/2 for i in x], [bar_rot[n] for n in names], w,
            label="globally ROTATED ICs", color="tab:purple")
    axB.set_yscale("log")
    axB.set_xticks(list(x)); axB.set_xticklabels(names)
    axB.set_title("(B) Prediction MSE: trained on one orientation")
    axB.set_ylabel("rollout MSE (log)")
    axB.legend(); axB.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out = "results/protein.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
