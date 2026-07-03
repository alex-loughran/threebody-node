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

## Theory

The guiding principle of this repo: **don't penalise physics violations — architect
them out.** A soft conservation loss buys you "approximately, after training"; the
right structure gives you "exactly, at initialisation."

### Why a plain Neural ODE drifts

A plain Neural ODE parameterises the vector field directly, `dz/dt = f_θ(z)` with
`z = (q, p)`. Nothing constrains `f_θ`: a generic learned field has the wrong
divergence/curl and acts as a net energy source or sink. Per-step errors compound,
the trajectory leaves the data manifold, and the MLP extrapolates into a blow-up
(here: `ΔE/|E0|` grows to ~1e13 over 5 orbits). The model isn't badly *trained* —
it's badly *constrained*.

### Hamiltonian structure

Hamiltonian mechanics gives dynamics a rigid geometric form. For a Hamiltonian
`H(q, p)`:

```
dq/dt =  ∂H/∂p
dp/dt = -∂H/∂q          ⇔   dz/dt = J ∇H(z),   J = [ 0  I; -I  0 ]
```

`J` (the symplectic matrix) is antisymmetric, and that alone forces energy
conservation along the flow:

```
dH/dt = ∇H · dz/dt = ∇Hᵀ J ∇H = 0        (since xᵀ J x = 0 for antisymmetric J)
```

A **Hamiltonian Neural ODE** learns a *scalar* `H_θ(q, p)` (an MLP) and constructs
the field from its symplectic gradient, `f_θ = J ∇H_θ` (gradient by autodiff). The
dynamics are then Hamiltonian *by construction*: the same one-liner shows `H_θ` is
conserved along the model's own flow. Energy conservation is removed as a *degree of
freedom*, not added as a penalty — this is what "structural rather than fitted"
means, and why energy drift stays bounded and flat across horizons (~3%) where the
plain NODE explodes, on identical data.

Training here is trajectory-matching: integrate `f_θ` with a differentiable solver
and match the rollout, backpropagating through the solve. Note the nested
differentiation — `f_θ` already contains `∇H_θ`, so training needs the gradient *of*
that gradient, composed through the integrator (a natural fit for JAX's `grad`).

### Separability → symplectic integration

Most physical Hamiltonians are **separable**, `H(q, p) = T(p) + V(q)`
(`SeparableHamiltonianNODE` learns the two heads). Separability is exactly the
condition that permits an *explicit* symplectic integrator — leapfrog /
Störmer–Verlet:

```
p_½ = p₀ − (h/2) ∂V/∂q(q₀)
q₁  = q₀ +   h   ∂T/∂p(p_½)
p₁  = p_½ − (h/2) ∂V/∂q(q₁)
```

(Explicit only because `∂H/∂q` depends on `q` alone and `∂H/∂p` on `p` alone; a
non-separable `H` would need an implicit solve.) By backward error analysis a
symplectic integrator conserves a **modified ("shadow") Hamiltonian**
`H̃ = H + O(h²)` *exactly, for all time* — so its energy error oscillates within a
bounded band with **no secular drift**, even at coarse steps. A non-symplectic
method like RK4 has better *local* accuracy but no conserved shadow, so its energy
drifts secularly over long horizons (here: leapfrog ~7% vs RK4 ~65% over 80 orbits
at `h=0.4`).

The key nuance: symplectic integration is **not** "more accurate per step" — at
small `h`, RK4 wins. Leapfrog wins at *coarse* `h` / *long* horizons by trading
pointwise accuracy for a preserved qualitative invariant. That trade — bounded
energy at ~3–4× larger steps — is the basis of the milestone-4 fast-surrogate thesis.

### Honest caveats

- **Conserving the *wrong* invariant doesn't help.** `H_θ` is conserved along the
  model flow, but need not equal the true energy — it's whatever scalar the network
  settled on. A poorly-estimated `H_θ` gives beautifully conserved but *incorrect*
  dynamics.
- **There is a model-error floor.** The learned `H_θ` carries a few-percent error;
  the symplectic integrator never adds error *above* that floor, so the learned
  energy surface — not the integrator — is the bottleneck.
- **Canonical coordinates are assumed.** The `J ∇H` form presumes genuine `(q, p)`
  canonical data. Dissipative or driven systems break the pure-Hamiltonian
  assumption and need extensions (e.g. a learned dissipation / port-Hamiltonian term).

### Lineage

Hamiltonian Neural Networks (Greydanus et al., 2019); Symplectic ODE-Net / SymODEN
(Zhong et al., 2020, integrator-in-the-loop); Lagrangian Neural Networks (Cranmer
et al., 2020, learn `L(q, q̇)` — no momenta needed); SympNets (Jin et al., 2020,
symplectic-by-construction maps).

## Milestones

- [x] **1 — baseline.** Vanilla Neural ODE on the (non-chaotic) two-body problem;
      verify the differentiable solver + training loop.
- [x] **2 — Hamiltonian structure.** Learn a scalar `H_θ`; show energy drift drops
      by *orders of magnitude* vs. the plain NODE on the same data.
- [x] **3 — symplectic solver + three-body.**
  - [x] *Symplectic integration on 2-body.* Separable model `H_θ = T_θ(p)+V_θ(q)`
        + leapfrog. Over 80 orbits at a coarse step, leapfrog energy stays
        bounded (~7%) while RK4 drifts to ~65%. Step-size sweep (with an exact-H
        reference) shows leapfrog rides the model-error floor while RK4's error
        explodes past h≈0.35. See `results/symplectic.png`, `experiment_symplectic.py`.
  - [x] *Three-body.* Pairwise Hamiltonian NODE (`H_θ = T_θ(p) + Σ_{i<j} g_θ(|rᵢ−rⱼ|)`)
        trained on REAL catalogued orbits (figure-eight & friends + perturbations)
        pulled from the physics engine's `orbits.db`. The model reproduces the
        figure-eight for ~1 period (held-out state RMSE ~0.2), leapfrog keeps
        energy bounded for ~4–5 orbits where RK4 blows up, and linear momentum is
        conserved to **machine precision** (translation-invariant pairwise
        potential → Noether). Evaluated on conserved-quantity drift, NOT
        long-horizon trajectory error (the flow is chaotic). See
        `experiment_threebody.py`, `src/threebody.py`, `results/threebody.png`.
        **Boundaries found:** a raw-coordinate `V_θ(q)` can't represent the 1/r
        close-approach singularity (blows up ~1000× worse); and fixed-step
        symplectic integration fights gravitational close approaches — the same
        reason the physics engine uses adaptive DOP853. Long-horizon 3-body
        stability is the open challenge → milestone 4.
- [~] **4 — research hooks.**
  - [x] *A — beat the close-approach wall.* Time-transformed ("logarithmic
        Hamiltonian") leapfrog (`logh_leapfrog_rollout`): integrate in a fictitious
        time `s` with `dt/ds = 1/(−U)`, so steps auto-shrink at close approaches
        while staying symplectic. On true Kepler the win over fixed-step leapfrog
        grows with eccentricity — 4× at e=0.3 up to **~23,000×** at e=0.99 (where
        fixed-step blows up). On the learned 3-body model it eliminates the
        per-orbit close-approach energy spikes (fixed ΔE spikes ~30/orbit → logH
        flat). See `experiment_adaptive.py`, `results/adaptive.png`.
  - [x] *B — surrogate-integrator thesis.* Batched surrogate rollout (`jax.vmap`)
        vs serial DOP853 over 256 candidate ICs, triaged by predicted return
        proximity `|y(T)−y(0)|`. **~5× throughput** (CPU-bound; GPU would widen it)
        and it works as a **coarse pre-filter** (median-split AUC 0.81; dropping
        the predicted-worst half retains 72% of the truly-best) — but its accuracy
        floor (rp≈0.5) makes it **blind to the *best* candidates** (pick-best
        P@10=0). Honest verdict: prune, don't pick — and the lever is model
        accuracy. See `experiment_surrogate.py`, `results/surrogate.png`.
  - [ ] *C — discover the invariants* rather than assume the Hamiltonian split.

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

## Offshoot — SE(3)-equivariant Hamiltonian NODEs for proteins

The same "architect the physics in, don't fit it" principle carries from gravity
to molecules. A coarse-grained peptide is a chain of `N` beads in 3-D under a
*separable* Hamiltonian — harmonic backbone bonds plus a softened Lennard-Jones
non-bonded term (excluded volume + weak attraction, the frustration that folds a
chain). Two structures stack:

- **Hamiltonian** → energy conservation, lifted to 3-D / N-body.
- **SE(3)-equivariance** — the protein-specific bias. Build the scalar `H_θ` from
  *invariant features only* (pairwise distances `|qᵢ−qⱼ|`, per-bead speeds
  `|pᵢ|²`) and `J ∇H_θ` is automatically an *equivariant* force field — exact to
  machine precision at initialisation, with no E(3)-GNN machinery.

A 3-way A/B (`PlainNODE3D` < `HamiltonianNODE3D` < `EquivariantHamiltonianNODE`)
separates the biases: Hamiltonian structure bounds energy drift; equivariance
gives *identical* predictions on globally-rotated ICs (equiv-error ~1e-9) where
the others degrade — while fitting ~8× better with ¼ the parameters
(`results/protein.png`). Four properties then fall out of the one construction:

- **Size generalisation.** `f_kin` is per-bead and `g_pot` per-pair, so the
  weights carry *no bead count* — the same model runs at any `N` (the fixed-dim
  baselines cannot even be called). Trained on 5-bead chains it transfers
  zero-shot to N=4…10; **mixed-N training (4–7) ~halves prediction error and
  flattens the transfer curve**, incl. extrapolation to N=8/10. Honest boundary:
  this fixes *accuracy* (data coverage), **not** *conservation*
  (`results/protein_scale.png`).
- **Symplectic integration.** Being separable, the model drops straight into
  leapfrog — learned energy bounded (~1e-3) where RK4 drifts secularly, same
  crossover story as the 2-body case (`results/protein_symplectic.png`).
- **Noether momenta.** The SE(3)-invariant `H_θ` conserves linear momentum to
  **machine precision** (1e-14; structural — distance-only forces sum to zero)
  and angular momentum to solver tolerance (1e-9), where a raw-coordinate
  Hamiltonian — energy-conserving but not symmetric — violates both by order 1.
  The rotational analogue of the 3-body linear-momentum result
  (`results/protein_noether.png`).

## Layout

```
src/physics.py            true Kepler (2-body) dynamics, conserved quantities, data gen
src/threebody.py          bridge to the physics engine: pull real orbits, 3-body conserved qty
src/protein.py            coarse-grained peptide: N-agnostic ground truth, momenta, SE(3) helpers
src/models.py             PlainNODE, HamiltonianNODE, SeparableHamiltonianNODE (any dim)
src/protein_models.py     PlainNODE3D < HamiltonianNODE3D < EquivariantHamiltonianNODE
src/integrate.py          diffrax rollout + fixed-step leapfrog / RK4 + logH (time-transformed) leapfrog
experiment.py             M1-2: plain vs Hamiltonian NODE, energy drift (2-body)
experiment_symplectic.py  M3: separable model + leapfrog vs RK4 (secular drift, step-size sweep)
experiment_threebody.py   M3: symplectic Hamiltonian NODE on real 3-body engine data
experiment_adaptive.py    M4A: time-transformed (logH) leapfrog vs fixed-step, close approaches
experiment_surrogate.py   M4B: batched surrogate vs DOP853 — speedup + return-proximity triage
experiment_protein.py            offshoot: 3-way A/B (energy drift + rotation generalization)
experiment_protein_scale.py      offshoot: size generalization + mixed-N fix
experiment_protein_symplectic.py offshoot: leapfrog vs RK4 on the learned peptide H
experiment_protein_noether.py    offshoot: SE(3) symmetry -> conserved linear & angular momentum
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
