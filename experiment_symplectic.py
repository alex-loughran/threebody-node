"""Milestone 3 (step 1): symplectic integration on the 2-body problem.

We train a SEPARABLE Hamiltonian Neural ODE  H_theta = T_theta(p) + V_theta(q)
using a symplectic leapfrog integrator, then show the two things a symplectic
integrator buys you -- neither of which an accurate non-symplectic solver gives:

  (A) NO SECULAR DRIFT. Over many orbits at a fixed step, leapfrog energy error
      oscillates in a bounded band; RK4 (same step, same model) drifts steadily.
  (B) BIG CHEAP STEPS. A step-size sweep: leapfrog keeps energy bounded as the
      step grows, so you can integrate coarsely -> the basis of a fast surrogate.

Run:  .venv/bin/python experiment_symplectic.py
"""
from __future__ import annotations

import argparse
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax

from experiment import conserved, make_windows
from src import physics
from src.integrate import batch_leapfrog, leapfrog_rollout, rk4_rollout
from src.models import SeparableHamiltonianNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


def train(model, y0s, targets, ts_win, *, iters, batch, lr, substeps, seed):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(m, yb, tb):
        preds = batch_leapfrog(m, yb, ts_win, substeps)
        return jnp.mean((preds - tb) ** 2)

    @eqx.filter_jit
    def step(m, opt_state, yb, tb):
        loss, grads = loss_fn(m, yb, tb)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    M, t0 = y0s.shape[0], time.time()
    for it in range(iters):
        sel = rng.integers(0, M, size=batch)
        model, opt_state, loss = step(model, opt_state, y0s[sel], targets[sel])
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"  [separable] iter {it:5d}  loss {float(loss):.3e}  ({time.time()-t0:5.1f}s)")
    return model


def frac_drift(traj, E0):
    E, _ = conserved(traj)
    return float(jnp.max(jnp.abs(E - E0))) / abs(E0)


class TrueKepler:
    """The EXACT separable Kepler Hamiltonian, exposing the same interface as the
    learned model. Used as a 'no model error' reference so we can attribute the
    energy behaviour to the integrator alone."""

    def dT_dp(self, p):
        return p

    def dV_dq(self, q):
        return q / jnp.linalg.norm(q) ** 3

    def vector_field(self, y):
        q, p = y[:2], y[2:]
        return jnp.concatenate([p, -q / jnp.linalg.norm(q) ** 3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-substeps", type=int, default=2)
    ap.add_argument("--long-orbits", type=int, default=80)
    ap.add_argument("--long-step", type=float, default=0.4)  # coarse fixed step
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)
    k_data, k_model = jax.random.split(key)

    print("generating reference orbits ...")
    trajs, ts = physics.make_dataset(k_data, n_orbits=12, t_span=12.0, n_points=240)
    y0s, targets = make_windows(trajs, W=6)
    ts_win = ts[:6] - ts[0]
    print(f"  {y0s.shape[0]} windows")

    print("training separable Hamiltonian NODE (leapfrog in the loop) ...")
    model = train(SeparableHamiltonianNODE(k_model), y0s, targets, ts_win,
                  iters=args.iters, batch=args.batch, lr=args.lr,
                  substeps=args.train_substeps, seed=3)

    # held-out orbit, near-circular so coarse fixed steps stay stable.
    r0, e = 1.0, 0.1
    a = r0 / (1 - e)
    period = float(2 * jnp.pi * a ** 1.5)
    y0 = physics.initial_condition(r0, e)
    E0 = float(physics.hamiltonian(y0))

    # --- accuracy report (model integrated accurately with its own scheme) ---
    ts_acc = jnp.linspace(0.0, 5 * period, 2000)
    acc = leapfrog_rollout(model, y0, ts_acc, n_substeps=8)
    print(f"\nseparable model, 5-orbit energy drift  dE/|E0| = {frac_drift(acc, E0):.3e}")

    # --- (A) long horizon at a fixed coarse step: leapfrog vs RK4 -------------
    h = args.long_step
    n_long = int(args.long_orbits * period / h)
    ts_long = jnp.arange(n_long) * h
    print(f"\nlong run: {args.long_orbits} orbits, fixed step h={h}, {n_long} steps ...")
    lf = leapfrog_rollout(model, y0, ts_long, n_substeps=1)
    rk = rk4_rollout(model, y0, ts_long, n_substeps=1)
    E_lf, _ = conserved(lf)
    E_rk, _ = conserved(rk)
    print(f"  leapfrog (symplectic) energy drift dE/|E0| = {frac_drift(lf, E0):.3e}")
    print(f"  rk4      (non-sympl.) energy drift dE/|E0| = {frac_drift(rk, E0):.3e}")

    # --- (B) step-size sweep over a fixed horizon ----------------------------
    # Solid = learned model; dashed = exact Kepler H (isolates integrator effect).
    true = TrueKepler()
    horizon = 20 * period
    steps = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    sweep_lf, sweep_rk, true_lf, true_rk = [], [], [], []
    for hs in steps:
        n = int(horizon / hs)
        tss = jnp.arange(n) * hs
        sweep_lf.append(frac_drift(leapfrog_rollout(model, y0, tss, 1), E0))
        sweep_rk.append(frac_drift(rk4_rollout(model, y0, tss, 1), E0))
        true_lf.append(frac_drift(leapfrog_rollout(true, y0, tss, 1), E0))
        true_rk.append(frac_drift(rk4_rollout(true, y0, tss, 1), E0))
    print("\nstep-size sweep (max dE/|E0| over 20 orbits):")
    print("  h      leapfrog   rk4        | true-H leapfrog  true-H rk4")
    for hs, a_lf, a_rk, t_lf, t_rk in zip(steps, sweep_lf, sweep_rk, true_lf, true_rk):
        print(f"  {hs:.2f}   {a_lf:.3e}  {a_rk:.3e}  |   {t_lf:.3e}      {t_rk:.3e}")

    # --- plots ----------------------------------------------------------------
    t_long = np.asarray(ts_long) / period
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(t_long, np.asarray(E_rk) - E0, "tab:orange", lw=1.0, label="RK4 (non-symplectic)")
    ax[0].plot(t_long, np.asarray(E_lf) - E0, "tab:blue", lw=1.0, label="leapfrog (symplectic)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(title=f"Energy drift over {args.long_orbits} orbits  (fixed step h={h})",
              xlabel="orbits", ylabel="ΔE")
    ax[0].legend(loc="upper left")

    ax[1].loglog(steps, sweep_rk, "o-", color="tab:orange", label="RK4, learned")
    ax[1].loglog(steps, sweep_lf, "o-", color="tab:blue", label="leapfrog, learned")
    ax[1].loglog(steps, true_rk, "x--", color="tab:orange", alpha=0.6, label="RK4, exact H")
    ax[1].loglog(steps, true_lf, "x--", color="tab:blue", alpha=0.6, label="leapfrog, exact H")
    ax[1].set(title="Energy error vs step size  (20-orbit horizon)",
              xlabel="step size h", ylabel="max ΔE/|E0|")
    ax[1].grid(True, which="both", alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/symplectic.png", dpi=130)
    print(f"\nsaved {RESULTS}/symplectic.png")


if __name__ == "__main__":
    main()
