# Proposed Title

**Transient Freezing Governs Flat-Minimum Selection in Stochastic Gradient Descent**

---

# Revised Outline

## Central Thesis

The paper should argue that stochastic gradient descent does not select flat minima mainly through a late-time steady-state bias. Instead, the decisive event is an early-time nonequilibrium **transient freezing** process: SGD remains mobile across competing valleys during early training, and the final valley is selected when inter-valley motion effectively ceases. Stronger noise promotes flatter minima primarily by delaying this freezing time.

---

## Abstract

### Goal

State the problem, identify the conceptual gap in existing explanations, introduce transient freezing as the missing mechanism, and emphasize its broader significance.

### Points to deliver

- Flat-minimum selection in SGD is widely observed, but its dynamical origin remains unsettled.
- Existing theories often emphasize late-stage, quasi-steady, or fixed-landscape reasoning.
- We show that final basin selection is determined earlier, during a nonequilibrium exploratory phase.
- SGD hops among valleys before becoming committed to one basin.
- The relevant control variable is the **freezing time** at which inter-valley exploration shuts down.
- Larger noise promotes flatter minima mainly by extending this exploratory window.
- This provides a predictive nonequilibrium framework for optimization bias and links SGD selection to a broader class of freezing phenomena.

---

## I. Introduction

### Central idea

Recast the paper around a sharper physical question: **when and how is the final basin selected during the strongly nonequilibrium phase of training?**

### Logic

1. SGD reliably finds solutions with good generalization, and these solutions are often associated with flatter minima.
2. Many existing accounts explain this tendency using effective-potential, entropic, or anisotropic-noise arguments that are most natural in late-stage or fixed-landscape settings.
3. However, real training begins far from such regimes: gradients are large, barriers evolve, and trajectories remain highly mobile.
4. This creates a conceptual gap: what actually determines the final basin before the dynamics become quasi-stationary?
5. Our answer is that SGD undergoes an early nonequilibrium search across valleys, and the eventual outcome is set when this search freezes.
6. In this view, noise changes the outcome not only through instantaneous selectivity, but by controlling how long valley competition remains active.

### Framing note

Introduce the Mpemba-effect analogy cautiously and only as a conceptual parallel: the outcome depends on the nonequilibrium pathway to arrest, not only on static preference.

---

## II. Valley Hopping Precedes Basin Commitment

### Central idea

Open with the clearest empirical phenomenon: early in training, SGD does not remain confined to one basin, but continues to explore multiple valleys before ultimately committing.

### Main points

- Use continuation-training or branch-and-continue experiments to identify which valley the system occupies at different times.
- Show that restarting from early checkpoints can lead to different final valleys under independent stochastic realizations.
- Show that restarting from later checkpoints yields the same final valley with high probability.
- Demonstrate that valley identity becomes stable only after a well-defined crossover.

### Key definition

Define a **freezing time** \(t_f\) as the earliest time after which the final valley identity becomes robust under continuation or independent reruns.

### Takeaway

The decisive event is not final relaxation inside a valley, but the loss of inter-valley mobility.

---

## III. Freezing Time Controls Final Basin Selection

### Central idea

Establish freezing time as the organizing variable linking stochastic dynamics to flat-minimum selection.

### Main points

- Measure how \(t_f\) changes with noise level, batch size, or learning-rate-controlled stochasticity.
- Show that stronger noise delays freezing and keeps trajectories mobile across competing valleys for longer.
- Show that later freezing correlates with larger probability of ending in the flatter valley.
- If available, show that flatter-valley selection collapses better as a function of \(t_f\) than as a function of raw noise strength.
- Connect delayed freezing to flatter solutions and, secondarily, to improved generalization.

### Key takeaway

The relevant control parameter is not just noise amplitude itself, but the duration of the nonequilibrium exploratory window.

---

## IV. Static Bias Alone Cannot Explain the Noise Dependence

### Central idea

Formulate the conceptual tension clearly: fixed-landscape or quasi-steady-state bias is real but insufficient.

### Main points

- In a fixed landscape slice, anisotropic noise can generate an effective preference for flatter valleys.
- But increasing noise at fixed position does not necessarily strengthen this static selectivity in the way needed to explain the training outcome.
- Therefore the observed monotonic enhancement of flat-minimum selection with stronger SGD noise cannot be explained by static bias alone.
- The missing ingredient is that the landscape evolves while inter-valley hopping progressively shuts down.

### Role in the paper

This section creates the paradox that the transient-freezing framework resolves.

---

## V. A Minimal Theory of Transient Freezing

### Central idea

After establishing the phenomenon and the paradox, introduce the simplest theory that captures the mechanism.

### Main points

- Construct a minimal two-valley landscape with unequal flatness.
- Include downhill training progress, evolving barriers, and anisotropic or landscape-dependent stochastic forcing.
- Show that the model reproduces early valley hopping, delayed freezing under larger noise, and enhanced final occupation of the flatter valley.
- Use the model to isolate which ingredients are essential and which are merely implementation details.

### Key takeaway

A minimal nonequilibrium model is sufficient to explain why basin selection is history-dependent and controlled by freezing.

---

## VI. Quantitative Predictions and Experimental Tests

### Central idea

Turn the theory into a predictive framework rather than a post hoc interpretation.

### Main points

- Derive or motivate quantitative predictions for freezing time and final flat-valley occupation probability.
- Compare these predictions with the minimal model, toy simulations, and neural-network experiments.
- Test whether the theory captures the observed dependence on noise level, batch size, and training stage.
- Clarify which trends are robust and which depend on simplifying assumptions.

### Strong version to aim for

Show that the final selection probability is better organized by freezing time than by any static late-time descriptor alone.

---

## VII. Discussion: Flat-Minimum Selection as a Nonequilibrium Freezing Process

### Central idea

Conclude with the conceptual shift: SGD solution selection is a transient nonequilibrium selection problem, not merely a steady-state sampling problem.

### Main messages

- The final solution is selected when inter-valley rearrangements cease, not only by which valley would be preferred in a hypothetical steady state.
- Stronger noise promotes flatter minima mainly by delaying the freezing of inter-valley dynamics.
- Effective-potential and entropic-bias pictures remain useful, but they are incomplete without a stopping-time or arrest mechanism.
- The logic is conceptually reminiscent of Mpemba-like phenomena, in which outcomes depend on the pathway to freezing rather than only on equilibrium preference.
- The framework suggests a new principle for optimizer and schedule design: control the timing of basin commitment by controlling the duration of exploratory mobility.

### Broader-impact note

Emphasize that the mechanism should apply whenever training involves competing valleys, evolving barriers, landscape-coupled noise, and a finite exploration window.

---

## Suggested Figure Restructuring

### Figure 1 — Early valley hopping and freezing

- Combine the current early figures into one compact entry-point figure.
- Include a schematic of valley hopping, continuation-training evidence, and an operational definition of \(t_f\).
- End the figure with the key observation that stronger noise delays freezing.

### Figure 2 — Why static bias is insufficient

- Present the fixed-landscape or quasi-steady intuition.
- Show clearly why it does not explain the full noise dependence of the final outcome.

### Figure 3 — Minimal theory and quantitative predictions

- Introduce the two-valley model.
- Show hopping, freezing, and selection trends.
- Compare predictions with experiments.

### Optional Figure 4 — Generality / stronger architecture

- Use only if needed to support the claim of broader relevance beyond the minimal setup.

---

## One-Sentence Paper Claim

**SGD selects flat minima not simply because they are statically favored, but because noise keeps the dynamics mobile across competing valleys until a late enough freezing time for the flatter valley to win.**
