"""Milestone 1-2: plain Neural ODE vs Hamiltonian Neural ODE on the 2-body problem.

Headline experiment: train both models on the same short-horizon orbit data,
then roll each out over many periods and measure how much the conserved
quantities (energy, angular momentum) DRIFT. The claim to verify is that adding
Hamiltonian structure cuts energy drift by orders of magnitude -- with no change
to the data, the solver, or the training loop.

Run:  .venv/bin/python experiment.py            # default ~2 min on CPU
      .venv/bin/python experiment.py --iters 3000   # tighter fit
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

from src import physics
from src.integrate import batch_rollout, rollout
from src.models import HamiltonianNODE, PlainNODE

jax.config.update("jax_enable_x64", True)
RESULTS = "results"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_windows(trajs, W):
    """Slice every length-W contiguous window out of every trajectory.

    Returns y0s (M, 4) start states and targets (M, W, 4). Because the time grid
    is uniform, every window shares the SAME relative time axis, so we only need
    one `ts_win` for the whole batch."""
    n_orbits, n_points, _ = trajs.shape
    idx = np.arange(n_points - W + 1)
    y0s, targets = [], []
    for o in range(n_orbits):
        for s in idx:
            y0s.append(trajs[o, s])
            targets.append(trajs[o, s : s + W])
    return jnp.stack(y0s), jnp.stack(targets)


def train(model, y0s, targets, ts_win, *, iters, batch, lr, seed, tag):
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_value_and_grad
    def loss_fn(m, yb, tb):
        preds = batch_rollout(m, yb, ts_win, adaptive=False, max_steps=64)
        return jnp.mean((preds - tb) ** 2)

    @eqx.filter_jit
    def step(m, opt_state, yb, tb):
        loss, grads = loss_fn(m, yb, tb)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(m, eqx.is_inexact_array))
        m = eqx.apply_updates(m, updates)
        return m, opt_state, loss

    rng = np.random.default_rng(seed)
    M = y0s.shape[0]
    t0 = time.time()
    for it in range(iters):
        sel = rng.integers(0, M, size=batch)
        model, opt_state, loss = step(model, opt_state, y0s[sel], targets[sel])
        if it % max(1, iters // 10) == 0 or it == iters - 1:
            print(f"  [{tag}] iter {it:5d}  loss {float(loss):.3e}  ({time.time()-t0:5.1f}s)")
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def conserved(traj):
    """Energy and angular momentum along a trajectory (T, 4) -> (T,), (T,)."""
    E = jax.vmap(physics.hamiltonian)(traj)
    L = jax.vmap(physics.angular_momentum)(traj)
    return E, L


def evaluate(models, ts_eval, y0_test):
    """Roll out the true dynamics and each model from the same held-out IC."""
    true = physics.integrate_true(y0_test, ts_eval)
    out = {"true": true}
    for name, m in models.items():
        out[name] = rollout(m, y0_test, ts_eval, adaptive=True, max_steps=200_000)
    return out


def report(rollouts, ts_eval, period):
    """Fractional energy drift max(|E(t)-E0|)/|E0| at growing horizons. We use
    the max over each horizon (the worst excursion) and normalise by the orbit's
    own |E0| so the number is interpretable: 1e-3 means '0.1% energy error'."""
    print("\n=== held-out rollout metrics (lower = better) ===")
    t = ts_eval
    E_true, L_true = conserved(rollouts["true"])
    E0, L0 = float(E_true[0]), float(L_true[0])
    horizons = [(1, period), (3, 3 * period), ("all", float(t[-1]))]
    head = "  {:12s}" + "".join(f"  ΔE/|E0| @{n}orb" for n, _ in horizons) + "   ΔL/|L0| @all"
    print(head.format(""))
    for name in ("plain", "hamiltonian"):
        E, L = conserved(rollouts[name])
        cells = []
        for _, tmax in horizons:
            m = t <= tmax
            frac = float(jnp.max(jnp.abs(E[m] - E0))) / abs(E0)
            cells.append(f"   {frac:11.3e}")
        ldrift = float(jnp.max(jnp.abs(L - L0))) / abs(L0)
        print(f"  {name:12s}" + "".join(cells) + f"   {ldrift:11.3e}")
    return E0


def plots(rollouts, ts_eval, E0):
    t = np.asarray(ts_eval)
    styles = {"true": ("k", "true"), "plain": ("tab:red", "plain NODE"),
              "hamiltonian": ("tab:blue", "Hamiltonian NODE")}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (1) orbit in the plane -- clip axes so the plain NODE's blow-up doesn't
    #     squash the real ellipses. The plain curve simply leaves the frame.
    for name, (c, lab) in styles.items():
        xy = np.asarray(rollouts[name])
        ax[0].plot(xy[:, 0], xy[:, 1], c, lw=1.3, alpha=0.85, label=lab)
    ax[0].set(title="Held-out orbit (relative coord.)", xlabel="x", ylabel="y",
              xlim=(-3.5, 1.5), ylim=(-2.5, 2.5))
    ax[0].set_aspect("equal"); ax[0].legend(loc="upper left")

    # (2) energy drift, symlog -> shows BOTH the tiny Hamiltonian drift and the
    #     plain NODE's exponential blow-up on one axis.
    for name, (c, lab) in styles.items():
        E, _ = conserved(rollouts[name])
        ax[1].plot(t, np.asarray(E) - E0, c, lw=1.3, alpha=0.85, label=lab)
    ax[1].set_yscale("symlog", linthresh=1e-3)
    ax[1].set(title="Energy drift  ΔE(t)  (symlog)", xlabel="time", ylabel="ΔE")
    ax[1].legend(loc="upper left")

    # (3) zoom: Hamiltonian vs true energy only, linear -> it stays bounded.
    for name in ("true", "hamiltonian"):
        c, lab = styles[name]
        E, _ = conserved(rollouts[name])
        ax[2].plot(t, np.asarray(E) - E0, c, lw=1.3, label=lab)
    ax[2].set(title="Hamiltonian NODE energy drift (zoom)", xlabel="time", ylabel="ΔE")
    ax[2].legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(f"{RESULTS}/energy_drift.png", dpi=130)
    print(f"saved {RESULTS}/energy_drift.png")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--n-orbits", type=int, default=12)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = jax.random.PRNGKey(args.seed)
    k_data, k_plain, k_ham = jax.random.split(key, 3)

    print("generating reference orbits ...")
    trajs, ts = physics.make_dataset(k_data, args.n_orbits, t_span=12.0, n_points=240)
    y0s, targets = make_windows(trajs, args.window)
    ts_win = ts[: args.window] - ts[0]
    print(f"  {y0s.shape[0]} training windows of length {args.window}")

    print("training plain Neural ODE ...")
    plain = train(PlainNODE(k_plain), y0s, targets, ts_win,
                  iters=args.iters, batch=args.batch, lr=args.lr, seed=1, tag="plain")
    print("training Hamiltonian Neural ODE ...")
    ham = train(HamiltonianNODE(k_ham), y0s, targets, ts_win,
                iters=args.iters, batch=args.batch, lr=args.lr, seed=2, tag="hamiltonian")

    # held-out IC: a size/eccentricity inside the training range but not on the grid
    r0_test, e_test = 1.1, 0.3
    a = r0_test / (1.0 - e_test)              # semi-major axis (perihelion start)
    period = float(2 * jnp.pi * a ** 1.5)     # Kepler's third law (G*M=1)
    y0_test = physics.initial_condition(r0=r0_test, eccentricity=e_test)
    ts_eval = jnp.linspace(0.0, 5 * period, 1500)  # ~5 orbits
    rollouts = evaluate({"plain": plain, "hamiltonian": ham}, ts_eval, y0_test)

    E0 = report(rollouts, ts_eval, period)
    plots(rollouts, ts_eval, E0)


if __name__ == "__main__":
    main()
