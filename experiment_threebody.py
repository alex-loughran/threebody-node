"""Milestone 3 (step 2): a symplectic Hamiltonian Neural ODE on the REAL,
chaos-capable three-body problem, trained on trajectories from the physics engine.

What changes vs. the 2-body case, and why it matters:

  * 12-D phase space (3 bodies x (position, velocity)); the learned separable
    Hamiltonian is H_theta = T_theta(p) + V_theta(q), dim=6.
  * Data are real catalogued orbits (figure-eight & friends) plus small
    perturbations, integrated by the engine's DOP853 (energy conserved to ~1e-11).
  * EVALUATION PHILOSOPHY FLIPS. The three-body flow is chaotic: nearby
    trajectories separate exponentially, so pointwise long-horizon trajectory
    error is physically meaningless (even the *true* dynamics are unpredictable
    long-term). We therefore judge the model on (a) SHORT-horizon trajectory
    accuracy and (b) CONSERVED-QUANTITY drift over long horizons -- energy,
    angular momentum, linear momentum -- which remain well-defined forever.

Run:  .venv/bin/python experiment_threebody.py
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

from experiment import make_windows
from src import threebody as tb
from src.integrate import batch_leapfrog, leapfrog_rollout, rk4_rollout
from src.models import PairwiseHamiltonianNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"

E = jax.vmap(tb.energy)
L = jax.vmap(tb.angular_momentum)
P = jax.vmap(tb.linear_momentum)


def train(model, y0s, targets, ts_win, *, iters, batch, lr, substeps, seed):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(m, yb, tb_):
        preds = batch_leapfrog(m, yb, ts_win, substeps)
        return jnp.mean((preds - tb_) ** 2)

    @eqx.filter_jit
    def step(m, opt_state, yb, tb_):
        loss, grads = loss_fn(m, yb, tb_)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), opt_state, loss

    rng = np.random.default_rng(seed)
    M, t0 = y0s.shape[0], time.time()
    for it in range(iters):
        sel = rng.integers(0, M, size=batch)
        model, opt_state, loss = step(model, opt_state, y0s[sel], targets[sel])
        if it % max(1, iters // 12) == 0 or it == iters - 1:
            print(f"  iter {it:5d}  loss {float(loss):.3e}  ({time.time()-t0:5.1f}s)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--train-substeps", type=int, default=2)
    args = ap.parse_args()

    key = jax.random.PRNGKey(0)

    print("generating 3-body trajectories from the physics engine ...")
    trajs, dt = tb.generate(seed=0, n_perturb=8, sigma=0.02, periods=2.0,
                            pts_per_period=100, n_sources=3)
    print(f"  {trajs.shape[0]} trajectories x {trajs.shape[1]} pts, dt={dt:.4f}")
    y0s, targets = make_windows(trajs, W=args.window)
    ts_win = jnp.arange(args.window) * dt
    print(f"  {y0s.shape[0]} training windows")

    print(f"training separable Hamiltonian NODE (dim=6, width={args.width}) ...")
    model = train(SeparableHamiltonianNODE(key, dim=6, width=args.width, depth=3),
                  y0s, targets, ts_win, iters=args.iters, batch=args.batch,
                  lr=args.lr, substeps=args.train_substeps, seed=7)

    # ---- held-out orbit: a fresh perturbation of the figure-eight ------------
    src = tb.stable_symmetric_sources(1)[0]
    vx, vy, T = src[0], src[1], src[2]
    n_eval = int(6 * 100)                 # ~6 periods
    ts_eval = jnp.arange(n_eval) * dt
    true = tb.reference_trajectory(vx, vy, ts_eval, sigma=0.02, seed=999)
    y0 = true[0]
    E0 = float(tb.energy(y0))

    lf = leapfrog_rollout(model, y0, ts_eval, n_substeps=3)
    rk = rk4_rollout(model, y0, ts_eval, n_substeps=3)

    # (a) short-horizon trajectory accuracy (first ~half period)
    n_short = 50
    rmse_short = float(jnp.sqrt(jnp.mean((lf[:n_short] - true[:n_short]) ** 2)))
    # (b) conserved-quantity drift over the whole horizon
    def drift(traj, f, ref):
        vals = f(traj)
        return float(jnp.max(jnp.abs(vals - ref)))
    print("\n=== held-out figure-eight perturbation ===")
    print(f"  E0 = {E0:.4f}   period T = {T:.3f}   horizon = {float(ts_eval[-1])/T:.1f} orbits")
    print(f"  short-horizon (half-period) state RMSE : {rmse_short:.3e}")
    print(f"  leapfrog  energy drift ΔE/|E0|         : {drift(lf, E, E0)/abs(E0):.3e}")
    print(f"  rk4       energy drift ΔE/|E0|         : {drift(rk, E, E0)/abs(E0):.3e}")
    print(f"  leapfrog  |ΔL|                         : {drift(lf, L, 0.0):.3e}")
    print(f"  leapfrog  |ΔP| (max component)         : "
          f"{float(jnp.max(jnp.abs(P(lf) - P(true)[0]))):.3e}")

    # ---- plots ---------------------------------------------------------------
    t_orb = np.asarray(ts_eval) / T
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) real-space paths of the 3 bodies, short horizon (model vs true)
    ns = 100
    for b, col in enumerate(["tab:blue", "tab:green", "tab:red"]):
        tr = np.asarray(true[:ns]); md = np.asarray(lf[:ns])
        ax[0].plot(tr[:, 2 * b], tr[:, 2 * b + 1], col, lw=2.0, alpha=0.5)
        ax[0].plot(md[:, 2 * b], md[:, 2 * b + 1], col, lw=1.0, ls="--")
    ax[0].set(title="3-body paths, ~1 period\n(solid=true, dashed=model)",
              xlabel="x", ylabel="y"); ax[0].set_aspect("equal")

    # (2) energy drift over many orbits: leapfrog vs rk4
    ax[1].plot(t_orb, np.asarray(E(rk)) - E0, "tab:orange", lw=1.0, label="RK4 (non-sympl.)")
    ax[1].plot(t_orb, np.asarray(E(lf)) - E0, "tab:blue", lw=1.0, label="leapfrog (sympl.)")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set(title="Energy drift over the rollout", xlabel="orbits", ylabel="ΔE")
    ax[1].legend(loc="upper left")

    # (3) trajectory divergence (chaos) vs bounded conserved quantities
    div = np.asarray(jnp.linalg.norm(lf - true, axis=1))
    ax[2].semilogy(t_orb, div + 1e-16, "tab:purple", lw=1.2, label="||model − true|| (state)")
    ax[2].semilogy(t_orb, np.abs(np.asarray(E(lf)) - E0) + 1e-16, "tab:blue", lw=1.2,
                   label="|ΔE| (conserved)")
    ax[2].set(title="Why we score conserved quantities, not trajectories",
              xlabel="orbits", ylabel="error (log)")
    ax[2].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(f"{RESULTS}/threebody.png", dpi=130)
    print(f"\nsaved {RESULTS}/threebody.png")


if __name__ == "__main__":
    main()
