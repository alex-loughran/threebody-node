# A primer: teaching neural networks to respect physics

*An accessible, self-contained walkthrough of what this project does and why.
It assumes you know a little calculus and have seen a neural network before, but
nothing about Neural ODEs, Hamiltonian mechanics, symplectic integrators, or
equivariance — each is introduced from scratch. If you want the terse technical
version instead, see [`conservation.md`](conservation.md).*

---

## 0. The one idea

We want a neural network to predict how a physical system moves through time — a
planet orbiting, a protein folding. The naive approach works for a few steps and
then falls apart: the predicted motion gains or loses energy that isn't there,
and the whole thing spirals off into nonsense.

The fix running through this whole project is a single principle:

> **Build the physics into the shape of the model, instead of asking the model to
> learn it.**

A physical law you *build in* holds exactly, for free, before training even
starts. A physical law you *hope the network learns* holds only approximately,
only near the training data, and tends to fail exactly when you need it most (far
into a long prediction). This document is the story of which laws we build in, how,
and what each one buys.

---

## 1. Systems that evolve in time

Almost all of classical physics is written as: *given the current state of a
system, here is the rule for how it changes in the next instant.* Mathematically,
if `z` is the state (a list of numbers) and `t` is time:

```
dz/dt = f(z)
```

This is an **ordinary differential equation (ODE)**. The function `f` is a
**vector field**: at every point in the space of possible states, it gives the
direction and speed the state moves next. **Integrating** the ODE means starting
from an initial state and repeatedly taking small steps in the direction `f`
points, tracing out a **trajectory**.

For example, a single planet orbiting a star (the "two-body problem", which
reduces to one moving particle) has a state made of its position and its momentum,
and `f` encodes Newton's law of gravity. Integrate it and you get an ellipse — an
orbit that closes on itself and repeats forever.

**The state we use.** Throughout, the state splits into two halves:

```
z = (q, p)
```

where `q` is **configuration** (positions of everything) and `p` is **momentum**
(mass × velocity). This `(position, momentum)` space is called **phase space**,
and it will matter a lot that the state is organised this way.

---

## 2. Neural ODEs: learning the rule from data

What if we don't know `f`? A **Neural ODE** replaces the physical rule with a
neural network:

```
dz/dt = f_θ(z)
```

Here `f_θ` is a small multilayer perceptron (MLP) with weights `θ`. We show it
example trajectories (from a simulator or from data), and train `θ` so that
integrating `f_θ` reproduces them. The clever part is that the ODE solver is
*differentiable*: we can backpropagate the prediction error all the way through
the integration and adjust the weights. It's an elegant idea — learn the *laws of
motion* directly, then simulate with them.

### Why the naive version fails

Nothing in this setup tells `f_θ` that it is describing *physics*. A generic
learned vector field has no reason to conserve energy. Along the trajectory it
acts as a small, persistent energy leak — sometimes adding energy, sometimes
removing it. Over a few steps this is invisible. Over many steps the errors
accumulate in the same direction, the trajectory drifts off the set of states the
network was trained on, and now the MLP is *extrapolating* — evaluating itself in
a region it never saw. Extrapolating MLPs do wild things, and the prediction blows
up.

Concretely, on the two-body orbit, a plain Neural ODE's energy error grows like
this (as a fraction of the true energy):

| after… | energy error |
|---:|---:|
| 1 orbit | 40% |
| 3 orbits | 2,000,000% |
| 5 orbits | 10,000,000,000,000% |

The model isn't badly *trained* — its short-term predictions are fine. It is badly
*constrained*: we never gave it a reason to conserve anything.

---

## 3. Building in energy conservation

Here is where physics helps. There is a beautiful and rigid mathematical structure
underlying almost all non-dissipative physics, called **Hamiltonian mechanics**,
and it hands us energy conservation on a plate.

### The Hamiltonian

For these systems there exists a single scalar function `H(q, p)` — the
**Hamiltonian** — which is the total energy (kinetic + potential). Remarkably, the
*entire* dynamics is determined by this one energy function through **Hamilton's
equations**:

```
dq/dt =  ∂H/∂p        (positions change according to how energy varies with momentum)
dp/dt = −∂H/∂q        (momenta change according to how energy varies with position)
```

Look at the structure: the derivative of `H` with respect to one half of the state
tells the *other* half how to move, and there's a crucial minus sign. We can write
both equations at once as

```
dz/dt = J ∇H(z),      J = [  0   I ]
                          [ −I   0 ]
```

where `∇H` is the gradient of the energy and `J` is a fixed matrix that swaps the
two halves and flips a sign. `J` is **antisymmetric** (`Jᵀ = −J`).

### Why this conserves energy — the whole argument in one line

How fast does the energy change along the motion? By the chain rule:

```
dH/dt = ∇H · (dz/dt) = ∇H · (J ∇H) = 0
```

The last step is a small piece of linear algebra: for *any* antisymmetric matrix
`J`, the quantity `xᵀ J x` is always zero (it equals its own negative). Since
`dz/dt` is built from `J ∇H`, the energy's rate of change is *exactly zero* — not
approximately, not on average, but identically. **Energy conservation is a free
consequence of the geometric form of the equations**, the antisymmetry of `J`.

> **Intuition.** The vector field `J ∇H` always points *along* surfaces of
> constant energy, never across them — like a ball rolling around the side of a
> valley at constant height, never up or down. The motion is trapped on an
> energy shell.

### The Hamiltonian Neural ODE

This suggests a much better way to learn dynamics. Instead of learning the vector
field `f_θ` directly (which can leak energy), we learn the **scalar energy
function** `H_θ(q, p)` — an MLP that outputs a single number — and *construct* the
vector field from it:

```
f_θ = J ∇H_θ           (∇H_θ computed by automatic differentiation)
```

Now the exact same one-line argument applies: whatever the network's weights, the
dynamics it produces conserve `H_θ`. **The network is physically incapable of
representing an energy leak**, even if the training data tried to push it toward
one. Energy conservation has been removed as a possibility, not added as a
penalty.

The payoff on the same two-body problem: where the plain Neural ODE's energy error
reached 10¹³%, the Hamiltonian Neural ODE stays around **3%, flat, for as long as
you integrate** — on identical data, changing nothing but the *shape* of the model.

### One honest catch

`H_θ` is conserved — but it is whatever energy function the network *learned*, which
is not guaranteed to be the *true* energy. If the learned energy surface is a poor
approximation, you get dynamics that are beautifully, perfectly conservative… and
wrong. **Conserving the wrong quantity doesn't save you.** This gap between the
learned `H_θ` and the true `H` — the "model-error floor" — is the quiet villain of
the whole project, and we come back to it at the end.

---

## 4. A second, sneakier source of drift: the integrator

There's a subtlety. Even with a *perfect* energy function, we still have to
*integrate* — take discrete time steps on a computer — and the stepping scheme
itself can inject or drain energy.

The standard workhorse integrator is RK4 (fourth-order Runge–Kutta). It is very
accurate per step. But it has no notion of energy conservation, so over a long
run its energy error creeps steadily in one direction — a **secular drift**.

There is a special family of **symplectic integrators** that respect the geometric
structure of Hamiltonian systems. The simplest is **leapfrog** (also called
Störmer–Verlet), which works when the energy *separates* into a kinetic part
depending only on momentum and a potential part depending only on position,
`H = T(p) + V(q)` — as it does for gravity and for molecules. Leapfrog alternates:

```
half-kick:   nudge momentum using the force −∂V/∂q
drift:       move positions using the velocity ∂T/∂p
half-kick:   nudge momentum again
```

The magic of symplectic integrators: although they don't conserve the true energy
*exactly*, they *do* exactly conserve a slightly-perturbed **"shadow" energy** that
stays close to the real one forever. The practical consequence is that their energy
error **oscillates within a fixed band and never drifts away**, even with large
time steps and over enormous horizons.

> **The honest nuance.** A symplectic integrator is *not* "more accurate per step"
> — at small step sizes RK4 is actually more accurate. What leapfrog buys is a
> preserved *qualitative* property (bounded energy) that lets you take much bigger
> steps over much longer times without the simulation quietly falling apart. For a
> learned model you want to run cheaply and far, that trade is exactly right.

Measured on the learned models: over 80 orbits at a deliberately coarse step,
leapfrog keeps energy bounded at ~7% while RK4 drifts to ~65%. On the peptide
model over a long run, leapfrog's energy error sits flat at ~0.15% while RK4 climbs
steadily.

### Gravity's nasty corner: close approaches

Gravity has a vicious feature: when two bodies pass very close, the force spikes
enormously, and a fixed-step integrator "jumps over" the spike and corrupts the
energy. The remedy is a **time-transformed (logarithmic) leapfrog**: it stretches
and compresses its own time steps — tiny steps during a close approach, big steps
when everything is far apart — while *staying symplectic*. On highly eccentric
orbits this improves accuracy by a factor that grows from ~4× (mild orbit) to
**~23,000×** (a near-collision orbit that the fixed-step method simply can't
handle).

---

## 5. Symmetry and conservation: Noether's theorem

Energy conservation came from a symmetry in *time* (the laws don't change from one
moment to the next). One of the deepest results in physics, **Noether's theorem**,
says this is completely general:

> **Every continuous symmetry of a system corresponds to a conserved quantity.**

Two symmetries of space give us two more conservation laws:

- **Translation symmetry** (the physics doesn't care *where* the system sits) →
  **linear momentum** `P = Σ pᵢ` is conserved.
- **Rotation symmetry** (the physics doesn't care which way the system is
  *oriented*) → **angular momentum** `L = Σ qᵢ × pᵢ` is conserved.

The wonderful thing in our framework is that **we control these symmetries directly,
by choosing what the energy function `H_θ` is allowed to look at.**

- If `H_θ` only ever sees *relative* positions — differences `qᵢ − qⱼ` between
  bodies, never absolute positions — then shifting the whole system by any amount
  leaves `H_θ` unchanged. It is translation-symmetric *by construction*, so it
  conserves linear momentum. (Physically: the internal forces come in
  equal-and-opposite pairs that cancel, so total momentum can't change.)

- If `H_θ` only ever sees *distances* `|qᵢ − qⱼ|` (which don't change when you
  rotate everything) then it is rotation-symmetric too, and conserves angular
  momentum.

So conservation of momentum isn't something we train for or hope for — it's a
consequence of *feeding the network the right invariant quantities.* This is the
bridge to the protein work.

---

## 6. Putting it together: proteins and equivariance

The three-body gravity work above establishes the machinery. The protein
"offshoot" is where several of these ideas stack into one clean construction.

### The toy system

A real protein is impossibly complex, so we use a **coarse-grained** cartoon of one:
a chain of `N` beads floating in 3-D, connected like beads on a string. The energy
has two parts:

- **Backbone bonds**: neighbouring beads are held near a preferred distance by a
  spring (this is the chain's connectivity).
- **Non-bonded interaction**: every pair of non-neighbour beads feels a soft
  push-apart-when-close, pull-together-when-far force (a "Lennard-Jones"
  interaction — excluded volume plus weak attraction).

The tension between the springs wanting one spacing and the non-bonded forces
wanting another is exactly the *frustration* that makes a real chain fold up rather
than sit limp. It's the simplest system that is recognisably protein-*like*.

### The key new idea: equivariance

A molecule's physics doesn't change if you pick it up, rotate it, and move it
across the room. The energy is identical; the forces just come along for the ride,
rotated the same way. In 3-D, the group of all rotations-and-translations is called
**SE(3)**, and we say the true dynamics are **SE(3)-equivariant**:

> rotate/translate the input → the output (the forces, the future motion) is the
> *same rotation/translation* of what you'd have gotten anyway.

A plain neural network reading raw `xyz` coordinates has no idea about this. It
would have to *learn*, from data, that a rotated protein behaves like the original
— wasting capacity, and only ever getting it approximately right.

**The trick** — and it's the same trick as energy conservation, reused — is to feed
the scalar energy function `H_θ` only quantities that *don't change* under rotation
or translation:

- **distances** between beads, `|qᵢ − qⱼ|` (unchanged by rotating/translating),
- **speeds** of beads, `|pᵢ|²` (unchanged by rotating).

If `H_θ` is built only from these **invariant features**, then `H_θ` is exactly
SE(3)-invariant. And it's a mathematical fact that *the gradient of an invariant
scalar is an equivariant vector field.* So the forces we get by differentiating
`H_θ` are automatically, exactly equivariant — with **no special network
architecture**, just the same "differentiate a well-chosen scalar" move we've used
all along. This holds to machine precision *at random initialisation*, before any
training.

### What this one construction buys — four things at once

We compare three models that differ *only* in structure — a plain Neural ODE
(`PlainNODE3D`), a Hamiltonian one that reads raw coordinates (`HamiltonianNODE3D`),
and the equivariant one built from invariant features (`EquivariantHamiltonianNODE`)
— so every result is attributable to a specific piece of structure.

**(a) Energy conservation + rotation generalisation.** Adding Hamiltonian structure
bounds the energy drift (as in §3). Adding equivariance means the model makes
*identical* predictions on a rotated copy of a molecule it has already seen — its
error on rotated test cases is the same as on un-rotated ones, essentially exactly
(to one part in a billion). The non-equivariant models degrade on rotated inputs.
And the equivariant model *fits the data ~8× better using a quarter of the
parameters*, because it isn't wasting capacity re-learning a symmetry.

**(b) Generalising to different molecule sizes.** Because the energy is a *sum* of
per-bead and per-pair terms, the network's weights don't depend on the number of
beads `N` at all. The very same trained model runs on a chain of any length — while
the raw-coordinate models literally cannot be run on a different-sized input (their
input layer is the wrong shape). A model trained only on 5-bead chains makes
sensible predictions on 4-, 6-, 7-, 8-, even 10-bead chains it never saw. Its
accuracy does gradually degrade as the chain gets longer and denser (probing
configurations outside its training experience), and training on a *range* of sizes
roughly halves that error — an honest reminder that "can run at any size"
(architecture) is not the same as "is accurate at any size" (which still needs
representative data).

**(c) Symplectic integration.** Because the energy separates into kinetic +
potential, the equivariant model plugs straight into leapfrog (§4) and inherits
bounded long-horizon energy.

**(d) Conserved momenta — the cleanest demonstration.** This is the sharpest test
of the whole idea. We compare the equivariant model against the raw-coordinate
Hamiltonian model. *Both* conserve their own energy (both are Hamiltonian) — so
energy is held fixed as a control, and the *only* difference between them is the
spatial symmetry. The result:

| model | energy error | linear momentum error | angular momentum error |
|---|---|---|---|
| **Equivariant** (symmetric) | ~1e-9 | **~1e-14** | **~1e-9** |
| Hamiltonian (raw coordinates) | ~1e-10 | ~1 (100%) | ~0.7 (70%) |

Both keep energy to nine or ten decimal places. But only the symmetric model
conserves momentum — and it does so to *machine precision* for linear momentum, and
to the integrator's accuracy for angular momentum. The non-symmetric model violates
both by ~100%. This ~10¹⁴-fold difference comes purely from *how the model reads its
inputs* — a direct, quantitative confirmation of Noether's theorem, live, in a
learned model. (And it holds even before training: it's structural, not learned.)

### Why linear and angular momentum land at *different* precisions

A nice detail: linear momentum is conserved to ~10⁻¹⁴ (machine precision) but
angular momentum "only" to ~10⁻⁹. Why?

- **Linear momentum** is conserved *algebraically*: the internal forces literally
  sum to zero at every single step, in exact arithmetic, regardless of how you
  integrate. So it's limited only by floating-point round-off.
- **Angular momentum** is conserved *in continuous time*, but the discrete
  integrator respects it only up to its own accuracy. Tighten the solver's
  tolerance and this error shrinks further.

Different symmetries, enforced through different mechanisms, land at different
precisions — a small window into how these guarantees actually work under the hood.

---

## 7. What is all this *for*? A cheap surrogate simulator

Structure-preserving models aren't just prettier — they're *useful*, precisely
because bounded energy lets you integrate cheaply and far. One concrete payoff:
**hunting for special orbits.**

In the three-body problem, most starting conditions give chaotic, non-repeating
motion, but a rare few give beautiful periodic orbits (like the famous figure-eight
where three equal masses chase each other around a fixed loop). To find them, you
scan thousands of candidate starting conditions and keep the ones that nearly
return to where they began after one period. The bottleneck is that checking each
candidate with a high-accuracy simulator is slow.

Idea: use the *learned* model as a fast, rough **pre-filter**. It runs a whole batch
of candidates in parallel and is several times faster. It's too rough to *pick* the
very best orbit — its own model error (that floor again) blurs the fine
distinctions — but it's reliably good at *ranking* candidates coarsely. Set up as a
**two-stage cascade** — the fast model throws away the obviously-bad half, the
accurate simulator carefully checks only what survives — you recover **100% of the
genuinely best candidates while doing far less expensive computation.** The rough
model does the cheap triage; the expensive one does the final judgement.

---

## 8. The theme that ties it together: the model-error floor

Notice a villain recurring in §3, §4, §6, and §7: the gap between the *learned*
energy `H_θ` and the *true* energy `H`. Structure guarantees the model conserves
`H_θ`; it does *not* guarantee `H_θ` is correct. This "floor" limits how far the
model can be trusted to roll out, and how finely the surrogate can rank orbits.

So the most recent thread of work attacks the floor directly, with two levers:

1. **Stronger inductive bias.** We *know* the kinetic energy exactly — it's just
   ½·(speed)² — so why learn it? Hard-coding the kinetic half of `H_θ` and letting
   the network spend all its capacity on the hard part (the potential) makes half
   the energy function exact by construction. Early results are striking: this drops
   the model-error floor by roughly **40×**, and — crucially — pushes the orbit
   surrogate from "can only prune" to "can actually pick the best."
2. **More data.** Train on more simulated trajectories and watch whether the floor
   keeps falling (which would mean the limit is data, which is cheap) or plateaus
   (which would mean the limit is the architecture). This part is still in
   progress — the current data-scaling result is noisy and non-monotonic, which is
   itself a signal that there's an optimisation issue to sort out before the clean
   trend shows up.

The through-line of the whole project: **each guarantee is bought by a specific
design choice, and knowing exactly which choice buys which guarantee — and which
problems remain unbought — is the real skill.** The `J ∇H` form buys energy
conservation. The choice of what `H_θ` reads buys the momenta, via Noether. The
symplectic integrator buys bounded energy in practice. And model accuracy — the one
thing structure *can't* hand you for free — remains the frontier.

---

## Where to go next

- [`conservation.md`](conservation.md) — the same material, condensed and with the
  precise mathematics and measured numbers.
- The top-level `README.md` — project overview, milestones, and how to run each
  experiment.
- The `experiment_*.py` scripts each reproduce one figure in `results/`; run any of
  them with `.venv/bin/python experiment_<name>.py`.

*Vocabulary recap: **phase space** = the space of (position, momentum) states;
**vector field** = the rule saying how the state changes; **Hamiltonian** `H` = the
energy as a function of state; **symplectic** = respecting the geometric structure
that makes energy conserved; **invariant** = unchanged by a transformation;
**equivariant** = transforming in step with the input; **Noether's theorem** =
symmetries produce conservation laws.*
