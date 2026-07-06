# Structure and conservation in the learned dynamics

A technical writeup of *what* these Neural ODEs conserve, *why* they conserve it,
and *to what precision* — across the gravitational three-body track and the
protein offshoot. The single organising idea:

> **Architect the physics in; don't fit it.** A soft penalty ("stay near the data,
> roughly conserve energy") buys *approximately, after training*. The right model
> class and the right integrator buy *exactly, at initialisation* — before a single
> gradient step.

Everything below is a consequence of choosing (a) the **form** of the learned
vector field, (b) the **arguments** the learned scalar is allowed to read, and (c)
the **integrator** that pushes it forward in time. Each choice buys a specific
conserved quantity.

---

## 1. Why a plain Neural ODE conserves nothing

A Neural ODE parameterises the time-derivative of the state directly:

```
dz/dt = f_θ(z),      z = (q, p)
```

and trains `f_θ` (an MLP) to match trajectories through a differentiable ODE
solver. Nothing constrains `f_θ`. A generic learned field has the wrong
divergence and curl, so along the flow it acts as a **net energy source or sink**.
Per-step errors compound, the trajectory leaves the data manifold, and the MLP —
now extrapolating — runs away.

On the two-body (Kepler) problem this is stark. Fractional energy drift
`ΔE/|E₀|` of a plain NODE:

| horizon | plain NODE | Hamiltonian NODE |
|--------:|-----------:|-----------------:|
| 1 orbit | 4.0e-01    | 3.0e-02 |
| 3 orbits| 2.3e+06    | 3.1e-02 |
| 5 orbits| 1.3e+13    | 3.8e-02 |

The plain model isn't badly *trained* — it is badly *constrained*. It has no
conserved quantity because we never gave it one.

---

## 2. Energy: the Hamiltonian construction

Hamiltonian mechanics gives dynamics a rigid geometric form. For a scalar
`H(q, p)`:

```
dq/dt =  ∂H/∂p
dp/dt = −∂H/∂q          ⇔        dz/dt = J ∇H(z),     J = [ 0  I ; −I  0 ]
```

`J` (the symplectic matrix) is antisymmetric, and that single fact forces energy
conservation along the flow:

```
dH/dt = ∇H · dz/dt = ∇Hᵀ J ∇H = 0        (xᵀ J x = 0 for antisymmetric J)
```

A **Hamiltonian Neural ODE** learns a *scalar* `H_θ(q, p)` (an MLP with scalar
output) and *constructs* the field from its symplectic gradient,
`f_θ = J ∇H_θ`, with `∇H_θ` obtained by autodiff. The dynamics are then
Hamiltonian *by construction*, and the same one-liner shows **`H_θ` is conserved
along the model's own flow**. The network cannot represent a non-conservative
force even if the data tried to push it there — energy conservation is removed as
a *degree of freedom*, not added as a penalty.

This is what "structural, not fitted" means, and it is why the Hamiltonian NODE's
energy drift stays bounded and flat across horizons (~3%) where the plain NODE
explodes, **on identical data** (`experiment.py`, `results/energy_drift.png`).

### Training detail — the nested gradient

Training is trajectory-matching: integrate `f_θ` with a differentiable solver,
match the rollout, backpropagate through the solve. Note the nested
differentiation: `f_θ` already contains `∇H_θ`, so the loss gradient needs the
gradient *of* that gradient, composed through the integrator. JAX's `grad`-of-
`grad` handles this directly.

### The important caveat — *which* energy?

`H_θ` is conserved along the model flow, but **need not equal the true energy**.
It is whatever scalar the network settled on. A poorly-estimated `H_θ` gives
beautifully conserved but *incorrect* dynamics. Conserving the wrong invariant
does not help you. This "model-error floor" — the gap between `H_θ` and the true
`H` — is the recurring bottleneck of the whole project: it caps the rollout
horizon and (later) the surrogate's usefulness. Structure removes the *energy-leak*
failure mode; it does not remove the *approximation* error.

---

## 3. Energy under a numerical integrator: symplectic integration

Conservation of `H_θ` holds for the *continuous* flow. A numerical integrator
introduces its own energy error, and here the choice of integrator matters as much
as the choice of model.

Most physical Hamiltonians are **separable**, `H(q, p) = T(p) + V(q)`. Separability
is exactly the condition for an *explicit* symplectic integrator — leapfrog /
Störmer–Verlet:

```
p_½ = p₀ − (h/2) ∂V/∂q(q₀)      # kick
q₁  = q₀ +   h   ∂T/∂p(p_½)      # drift
p₁  = p_½ − (h/2) ∂V/∂q(q₁)      # kick
```

(Explicit only because `∂H/∂q` depends on `q` alone and `∂H/∂p` on `p` alone.) By
backward-error analysis, a symplectic integrator conserves a **modified ("shadow")
Hamiltonian** `H̃ = H + O(h²)` *exactly, for all time*. So its energy error
oscillates in a **bounded band with no secular drift**, even at coarse steps. A
non-symplectic method like RK4 has better *local* accuracy but no conserved shadow,
so its energy drifts secularly.

The nuance worth stating precisely: symplectic integration is **not** "more
accurate per step." At small `h`, RK4 wins. Leapfrog wins at *coarse* `h` and
*long* horizons by trading pointwise accuracy for a preserved qualitative
invariant. That trade — bounded energy at ~3–4× larger steps — is what makes a
learned model usable as a fast surrogate.

Measured, on the learned models:

- **2-body**, 80 orbits at `h=0.4`: leapfrog energy bounded ~7% vs RK4 ~65%
  (`experiment_symplectic.py`).
- **Peptide** (see §5), `t=120` at `h=0.1`: leapfrog `|ΔH_θ/H_θ|` bounded ~1.5e-3
  vs RK4 secular to 2.8e-2; step-size sweep shows the crossover at `h≈0.04`
  (`experiment_protein_symplectic.py`).

### Close approaches — time transformation

Fixed-step symplectic integration fights gravitational close approaches (the force
spikes between steps). The fix is a **time-transformed / logarithmic Hamiltonian**
leapfrog (`logh_leapfrog_rollout`): integrate an extended separable Hamiltonian in
a fictitious time `s` with `dt/ds = 1/(−U)`, so the physical step auto-shrinks near
close approaches while the map stays symplectic. On true Kepler the win over
fixed-step leapfrog grows with eccentricity — **4× at e=0.3 up to ~23,000× at
e=0.99** — and on the learned three-body model it removes the per-orbit
close-approach energy spikes (`experiment_adaptive.py`).

---

## 4. Momentum: symmetry → Noether

Energy is the conserved quantity of *time*-translation symmetry. The other
conserved quantities come from *spatial* symmetries, via **Noether's theorem**:
every continuous symmetry of `H` yields a conserved quantity. Crucially, in this
framework you get them **by controlling which arguments `H_θ` is allowed to read.**

### 4.1 Translation → linear momentum

If `H` depends on positions only through **differences** `qᵢ − qⱼ`, it is invariant
under a global translation `qᵢ → qᵢ + c`. Noether then conserves the total linear
momentum `P = Σᵢ pᵢ`. Directly:

```
dP/dt = Σᵢ dpᵢ/dt = −Σᵢ ∂V/∂qᵢ = 0
```

the last equality because internal pairwise forces come in equal-and-opposite
pairs (Newton's third law) and sum to exactly zero.

This is **structural and solver-independent**: the force vector sums to zero at
*every* step of *any* consistent integrator, so `P` is held to **machine
precision**. On the real three-body figure-eight, the pairwise model
(`H_θ = T_θ(p) + Σ_{i<j} g_θ(|rᵢ−rⱼ|)`) conserves linear momentum to ~1e-14
(`experiment_threebody.py`). A model that read *absolute* coordinates would have no
such symmetry and would leak momentum.

### 4.2 Rotation → angular momentum

If `H` further depends on positions only through **distances** `|qᵢ − qⱼ|` and on
momenta only through rotational invariants, it is invariant under a global rotation
`qᵢ → R qᵢ`, `pᵢ → R pᵢ` for any `R ∈ SO(3)`. Noether then conserves the total
angular momentum `L = Σᵢ qᵢ × pᵢ`. This is the rotational companion of §4.1 and the
centrepiece of the protein work below.

---

## 5. The protein offshoot: SE(3)-equivariance as the source of *all* the momenta

A coarse-grained peptide is a chain of `N` beads in 3-D under a separable
Hamiltonian — harmonic backbone bonds plus a softened Lennard-Jones non-bonded term
(excluded volume + weak attraction, the frustration that folds a chain). Its true
dynamics are **SE(3)-equivariant**: rotate/translate the whole molecule and the
forces rotate/translate with it; the energy is unchanged.

### The construction — one idea buys everything

Build the scalar `H_θ` from **SE(3)-invariant features only**:

```
T_θ(p) = Σᵢ f_θ(|pᵢ|²)                    # per-bead, shared
V_θ(q) = Σ_{i<j} g_θ(dᵢⱼ, is_bondedᵢⱼ)     # per-pair, shared, dᵢⱼ = |qᵢ − qⱼ|
```

Both `f_θ` and `g_θ` read scalars that do not change under a global rotation or
translation. Therefore `H_θ` is exactly SE(3)-invariant, and its symplectic
gradient `J ∇H_θ` is exactly an **equivariant** force field — `f(R·y) = R·f(y)` to
**machine precision, at random initialisation, with no E(3)-GNN machinery.** The
same autodiff move that gave energy conservation (§2) now also gives equivariance.

Four properties fall out of this one construction:

**(a) Energy + rotation generalisation.** A clean 3-way ablation
(`PlainNODE3D` < `HamiltonianNODE3D` < `EquivariantHamiltonianNODE`) separates the
biases. Adding Hamiltonian structure bounds energy drift; adding equivariance gives
*identical* predictions on globally-rotated initial conditions (equivariance error
~1e-9 *through* an adaptive ODE solver) where the others degrade — while fitting
~8× better with ¼ the parameters (`experiment_protein.py`, `results/protein.png`).

**(b) Size generalisation.** `f_θ` is per-bead and `g_θ` per-pair, and `H` is a
*sum*, so the weights carry **no bead count**. The same model runs at any `N`; the
fixed-input-dim baselines cannot even be *called* at a new `N`. Trained on 5-bead
chains it transfers zero-shot to N=4…10; **mixed-N training (4–7) roughly halves
prediction error and flattens the transfer curve**, including extrapolation
(N=8: 2.2×, N=10: 2.5×). Honest boundary: this fixes *accuracy* (data coverage),
not *conservation* — those are orthogonal levers (`experiment_protein_scale.py`,
`results/protein_scale.png`).

**(c) Symplectic integration.** Being separable, the model drops straight into
leapfrog — see §3 for the bounded-energy result.

**(d) Noether momenta.** This is the sharpest test, and it isolates the mechanism
cleanly. Compare the equivariant model against `HamiltonianNODE3D`, which is *also*
a Hamiltonian (so it conserves its own energy) but reads **raw coordinates** (so it
has *no* spatial symmetry). Energy conservation is thereby held fixed as a control;
the only difference is the symmetry.

| model | energy `|ΔE/E|` | linear `|ΔP|` | angular `|ΔL|/|L₀|` |
|---|---|---|---|
| **Equivariant** (SE(3)-invariant) | 3.2e-9 | **1.3e-14** | **8.5e-10** |
| Hamiltonian (raw coords) | 1.7e-10 | 1.15 | 0.70 |

Both conserve their own energy to ~1e-10. Only the symmetric model conserves
momentum — and it does so by ~14 (linear) and ~9 (angular) orders of magnitude,
*attributable to the symmetry alone* (`experiment_protein_noether.py`,
`results/protein_noether.png`). The result holds even for an **untrained**
equivariant model: it is structural, not learned.

---

## 6. The precision hierarchy — *how exactly* is each quantity conserved?

Not all conservation is equal. The three conserved quantities land at three
different precisions, and *why* is itself instructive:

| quantity | symmetry | how conserved | precision | limited by |
|---|---|---|---|---|
| **linear momentum** `P` | translation | forces sum to **exactly** zero every step | ~1e-14 (machine) | floating point |
| **angular momentum** `L` | rotation | zero net torque in *continuous* time | ~1e-9 | integrator `rtol` |
| **energy** `H_θ` | time translation | shadow-Hamiltonian under symplectic map | ~1e-3, bounded | step size `h` (secular for RK4) |
| **true energy** `H` | — | only as well as `H_θ ≈ H` | few % | model-error floor |

- **Linear `P` is exact regardless of the solver.** The invariance is algebraic —
  distance-only forces cancel — so it holds bit-for-bit at every step. Tightening
  the solver does nothing because there is nothing to tighten.
- **Angular `L` is exact only in continuous time.** The torque-free condition is
  respected to the integrator's accuracy; tighten `rtol` and `|ΔL|` drops further.
- **Energy splits in two.** `H_θ` is protected by the *symplectic integrator*
  (bounded, no drift). But the *true* `H` is protected only insofar as `H_θ`
  approximates it — the model-error floor of §2, and the target of the ongoing
  accuracy work (fixing `T=½|p|²` exactly, and scaling training data).

The clean takeaway: **each conserved quantity traces to a specific design choice** —
the `J ∇H` *form* (energy), the *arguments* of `H_θ` (momenta via symmetry), and the
*integrator* (bounded energy in practice). They compose, and each does a job the
others cannot.

---

## 7. What it's good for — the surrogate cascade

Bounded energy at coarse steps makes the learned model a cheap, batched surrogate
integrator. In orbit search you scan many candidate initial conditions and keep the
near-periodic ones (small return proximity `|y(T) − y(0)|`); calling the exact
DOP853 integrator on every candidate is the bottleneck.

The surrogate is ~5× faster (batched, CPU f64) and ranks candidates with
Spearman ρ≈0.58 — good enough to **prune**, too coarse to **pick** (it is blind
below its own accuracy floor). Framed as a **two-stage cascade** (surrogate ranks
all → DOP853 refines the top-K), that coarse ranking is exactly sufficient:
**keeping the predicted-best 50% recovers 100% of the truly-best-10 orbits at 71%
of full DOP853 cost** on CPU, and the surrogate's fixed cost collapses on GPU,
widening the win (`experiment_surrogate.py`, `experiment_surrogate_cascade.py`).
The floor is again the ceiling on how aggressively you can prune — which closes the
loop back to §2 and motivates the accuracy work.

---

## 8. Lineage

Hamiltonian Neural Networks (Greydanus et al., 2019); Symplectic ODE-Net / SymODEN
(Zhong et al., 2020, integrator-in-the-loop); Lagrangian Neural Networks (Cranmer
et al., 2020); SympNets (Jin et al., 2020). Equivariance via invariant features is
the elementary case of E(n)-equivariant GNNs (Satorras et al., 2021) and
tensor-field networks; time-transformed symplectic integration follows
Preto–Tremaine (1999) and Mikkola–Tanikawa (1999).

---

*Reproduce any figure with `.venv/bin/python experiment_<name>.py`. All metrics use
`jax_enable_x64` — the headline is conserved-quantity drift at the 1e-3…1e-14 level,
which would be pure rounding noise in float32.*
