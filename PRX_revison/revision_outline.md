# Proposed Title

**Transient Freezing Governs Flat-Minimum Selection in Stochastic Gradient Descent**

---

# Revised Outline

## Central Thesis

The paper should argue that stochastic gradient descent does not select flat minima mainly through a late-time steady-state bias. Instead, the decisive event is an early-time nonequilibrium **transient freezing** process: SGD remains mobile across competing valleys during early training, and the final valley is selected when inter-valley motion effectively ceases. Stronger noise promotes flatter minima primarily by delaying this freezing point, so basin selection occurs later along the evolving landscape rather than under a stronger static preference alone.

---

## Abstract

### Goal

State the problem, identify the conceptual gap in existing explanations, introduce transient freezing as the missing mechanism, and emphasize its broader significance.

### Points to deliver

- Flat-minimum selection in SGD is widely observed, but its dynamical origin remains unsettled.
- Existing theories often emphasize late-stage, quasi-steady, or fixed-landscape reasoning.
- We show that final basin selection is determined earlier, during a nonequilibrium exploratory phase.
- SGD hops among valleys before becoming committed to one basin.
- The relevant organizing variable is the **freezing time** at which inter-valley exploration shuts down.
- Larger noise promotes flatter minima mainly by extending this exploratory window and therefore shifting when basin selection is locked in.
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

## III. Basin Selection Is Set at the Freezing Point

### Central idea

Move beyond the operational definition of \(t_f\). This section should show that \(t_f\) is the stopping-time variable through which SGD noise changes the final distribution over basins: stronger noise delays the loss of inter-valley mobility, and the geometry measured at the freezing checkpoint reflects the landscape slice at which commitment occurs.

### Logical role after Section II

Section II establishes that a measurable freezing time exists. Section III should make the nontrivial step: \(t_f\) is not just a post hoc label of commitment, but the stopping-time variable that organizes how stochasticity becomes a final basin choice.

### Main points

- First state the distinction from the previous section: the definition of \(t_f\) identifies when commitment occurs, but does not by itself explain why SGD noise changes the final basin distribution.
- Measure how \(t_f\) changes with noise level, batch size, or learning-rate-controlled stochasticity using the continuation-based basin-stability definition.
- Show that stronger noise delays freezing and keeps trajectories mobile across competing valleys for longer.
- Emphasize that this is not merely slower loss minimization or a different loss threshold; it is a delayed loss of basin-label variability under continuation.
- Because cross-entropy keeps changing margins after accuracy saturates, avoid using final-time sharpness as the main geometry variable in this section.
- Define a freezing-time flatness \(F_f=-\log_{10}S_{\rm nw}(\theta_{t_f})\), using neuron-wise relative sharpness, and interpret it as geometry at basin commitment.
- Show that \(F_f\) collapses better as a function of \(\eta t_f\) than as a function of raw noise strength among continuation-stable trajectories that converge at the final checkpoint, consistent with \(t_f\) acting as a mediator of the noise dependence.
- Use final test accuracy as the late-time performance outcome on the same final-converged stable trajectory set, while noting that final sharpness/test loss can include post-freezing CE relaxation.
- Clarify that "basin selection is set at \(t_f\)" is not only definitional: changing SGD noise shifts \(t_f\), and the basin geometry at commitment follows that shift.
- Emphasize that the final outcome depends on the landscape state and quasi-steady valley bias at the freezing point, not on \(t_f\) in isolation.
- End by motivating the minimal model: the next task is to separate the fixed-slice valley bias from the noise-dependent stopping time at which that bias is inherited.

### Key takeaway

Noise promotes flat-minimum selection primarily by delaying the stopping time at which inter-valley mobility is lost. The freezing time \(t_f\) is therefore the experimentally measurable mediator between SGD stochasticity and geometry at basin commitment, while the actual basin chosen still depends on the landscape and valley bias present at that freezing slice.

---

## IV. A Minimal Theory of Transient Freezing

### Central idea

Introduce the simplest reduced model that captures competing valleys, evolving barriers, and geometry-coupled noise, so the roles of static bias and transient freezing can be separated cleanly.

### Main points

- Construct a minimal two-valley landscape with a downhill training coordinate and a transverse valley-selection coordinate.
- Include unequal flatness, an evolving barrier, and anisotropic or landscape-dependent stochastic forcing.
- Show that the model reproduces early valley hopping, delayed freezing under larger noise, and enhanced final occupation of the flatter valley.
- Use the model to define a small set of interpretable variables---flatness contrast, barrier height, effective noise, and freezing point---that can be compared directly with the experiments.

### Key takeaway

A minimal nonequilibrium model contains the ingredients needed to analyze both the fixed-slice bias and the freezing-controlled final selection.

---

## V. Static Bias Is Real but Insufficient

### Central idea

Use the reduced model to separate the quasi-steady preference at a fixed landscape slice from the final basin chosen when exploration freezes.

### Main points

- In a quasi-steady regime at fixed training coordinate, anisotropic noise generates an effective preference for the flatter valley.
- Derive or motivate the fixed-slice occupation probability and identify the quantities that control it.
- Show that increasing noise at fixed position can weaken instantaneous selectivity by making the occupation probability more mixed.
- Therefore the observed monotonic enhancement of flat-minimum selection with stronger SGD noise cannot be explained by static bias alone.
- The missing ingredient is that the final outcome is inherited at the freezing point of an evolving landscape, not from a single fixed-slice bias viewed in isolation.

### Key takeaway

Fixed-landscape bias is real, but the final training outcome depends on when that bias is frozen into the dynamics.

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
- Stronger noise promotes flatter minima mainly by delaying the freezing of inter-valley dynamics, so basin selection occurs later along training.
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
- End the figure with the key observation that basin identity becomes stable only after a measurable commitment time.

### Figure 2 — Freezing-time organization

- Panel A: show median continuation-based \(\eta t_f\) across batch size and learning rate, with stable-fraction annotations.
- Panel C: use a left-right pair of plots comparing freezing-time flatness \(F_f=-\log_{10}S_{\rm nw}(\theta_{t_f})\) against effective noise and against \(\langle\eta t_f\rangle\), with point colors indicating learning rate and the \(\eta=0.1\) sweep omitted; include continuation-stable trajectories that converge at the final checkpoint.
- Panel D: use the same left-right comparison for final test accuracy on the same final-converged stable trajectory set.
- Explain in the caption/text that geometry is measured at \(t_f\) to avoid conflating basin commitment with post-freezing cross-entropy margin relaxation.

### Figure 3 — Minimal theory and delayed freezing

- Introduce the two-valley model.
- Show hopping, freezing, and selection trends.

### Figure 4 — Why static bias is insufficient

- Present the fixed-landscape or quasi-steady intuition within the toy model.
- Show clearly why it does not explain the full noise dependence of the final outcome.

### Figure 5 — Quantitative predictions and experimental tests

- Compare the transient-selection theory with experiments.
- Organize the data explicitly by \(t_f\) when possible.

### Optional Figure 6 — Generality / stronger architecture

- Use only if needed to support the claim of broader relevance beyond the minimal setup.

---

## One-Sentence Paper Claim

**SGD selects flat minima not simply because they are statically favored, but because noise delays the freezing point at which basin competition is resolved, allowing selection to occur later along the evolving landscape where the flatter valley is more likely to prevail.**
