# threebody-node

**Structure-preserving Neural ODEs for gravitational dynamics** — the deep-learning
track of the three-body orbit project.

A Neural ODE learns the *vector field* of a dynamical system and integrates it with
a differentiable ODE solver. A **Hamiltonian** Neural ODE instead learns a single
scalar `H_θ(q, p)` and *constructs* the dynamics from its symplectic gradient, so
conservation of energy is built into the model class rather than fitted. This repo
builds that idea up from the integrable two-body problem toward the chaotic
three-body problem.

This is intentionally a **separate repo** from the classical-ML surrogate work
(`threebody-ml`) and the physics engine (`PythonProject1`), matching the existing
separation: physics engine = stable library, classical ML = its own repo, DL = here.

## Milestones

- [x] **1 — baseline.** Vanilla Neural ODE on the (non-chaotic) two-body problem;
      verify the differentiable solver + training loop.
- [x] **2 — Hamiltonian structure.** Learn a scalar `H_θ`; show energy drift drops
      by *orders of magnitude* vs. the plain NODE on the same data.
- [~] **3 — symplectic solver + three-body.**
  - [x] *Symplectic integration on 2-body.* Separable model `H_θ = T_θ(p)+V_θ(q)`
        + leapfrog. Over 80 orbits at a coarse step, leapfrog energy stays
        bounded (~7%) while RK4 drifts to ~65%. Step-size sweep (with an exact-H
        reference) shows leapfrog rides the model-error floor while RK4's error
        explodes past h≈0.35. See `results/symplectic.png`, `experiment_symplectic.py`.
  - [~] *Three-body.* Separable Hamiltonian NODE (dim=6) trained on REAL
        catalogued orbits (figure-eight & friends + perturbations) pulled from the
        physics engine's `orbits.db`. Evaluated on conserved-quantity drift, NOT
        long-horizon trajectory error (the flow is chaotic). See
        `experiment_threebody.py`, `src/threebody.py`, `results/threebody.png`.
- [ ] **4 — research hook.** Surrogate integrator (faster-than-true at fixed
      accuracy, feeding `threebody-ml`'s active loop) / conserved-quantity
      discovery / stability classification from the learned flow's Jacobian.

## Result (milestones 1–2)

Both models are trained on identical short-horizon windows of two-body orbits, then
rolled out for ~5 orbits from a held-out initial condition. **Fractional energy
drift `ΔE/|E0|`:**

| horizon | plain NODE | Hamiltonian NODE |
|---------|-----------:|-----------------:|
| 1 orbit | 4.0e-01    | 3.0e-02 |
| 3 orbits| 2.3e+06    | 3.1e-02 |
| 5 orbits| 1.3e+13    | 3.8e-02 |

The plain NODE has no conservation law: vector-field errors act as a net energy
source, the trajectory leaves the data manifold, and the MLP extrapolates into a
blow-up. The Hamiltonian NODE cannot inject net energy into its own `H_θ`, so its
energy drift stays **bounded and flat across horizons** — the signature of
structure preservation. See `results/energy_drift.png`.

## Layout

```
src/physics.py            true Kepler (2-body) dynamics, conserved quantities, data gen
src/threebody.py          bridge to the physics engine: pull real orbits, 3-body conserved qty
src/models.py             PlainNODE, HamiltonianNODE, SeparableHamiltonianNODE (any dim)
src/integrate.py          diffrax rollout + fixed-step leapfrog / RK4 (symplectic comparison)
experiment.py             M1-2: plain vs Hamiltonian NODE, energy drift (2-body)
experiment_symplectic.py  M3: separable model + leapfrog vs RK4 (secular drift, step-size sweep)
experiment_threebody.py   M3: symplectic Hamiltonian NODE on real 3-body engine data
```

The three-body data bridge imports the engine from `~/PycharmProjects/PythonProject1`
(override with `$THREEBODY_ENGINE`) and caches trajectories to `datasets/`.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiment.py            # ~4 min on CPU; writes results/energy_drift.png
.venv/bin/python experiment.py --iters 5000 --lr 2e-3   # tighter fit
```

## Notes

- `jax_enable_x64` is on everywhere: the headline metric is conserved-quantity drift
  at the 1e-3 level, which would be pure rounding noise in float32.
- Training uses **short windows** (~0.25 time units), never long rollouts — the
  model learns the local flow. This is the design choice that makes the eventual
  jump to *chaotic* three-body dynamics tractable: the vector field / Hamiltonian
  is smooth even where individual trajectories are exponentially unpredictable.
