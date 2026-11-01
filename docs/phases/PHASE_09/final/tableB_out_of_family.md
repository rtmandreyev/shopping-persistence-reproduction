# Table B — Out-of-family recovery on anchored learned-dynamics truths (P9-007)

Truth-arm = truth × heterogeneity arm. 'Evaluated' = CI-1 admissible and not DEGENERATE (pre-registered rules). Max HISTORY-FAVOURED is over every evaluated truth-arm and κ ∈ {0.15, 0.25, 0.35}: it is false attribution for the no-own-history class and attribution to the maintained history family for the own-history class. 0/200 per truth-arm has an exact two-sided 95% Clopper–Pearson upper bound of 0.018. Mercury's excluded arms are those with the strongest learned response.

| Policy | Truth class | Evaluated | Max HISTORY-FAVOURED | Verdicts at κ 0.25 | Δ1 range (pp) | Naive flag range (all truth-arms) |
|---|---|---|---|---|---|---|
| Jev 1.13 | no own-history (A1–A7) | 14/14 | 0.0 | EQUIVALENT 0.986, UNRESOLVED 0.014 | (0.0, 9.6) | (0.0, 1.0) |
| Jev 1.13 | own history (B1–B8) | 16/16 | 0.0 | EQUIVALENT 0.820, UNRESOLVED 0.176, LATENT-FAVOURED 0.004 | (2.1, 9.7) | (0.94, 1.0) |
| Jev 1.14 | no own-history (A1–A7) | 14/14 | 0.0 | EQUIVALENT 0.990, UNRESOLVED 0.010 | (0.0, 8.6) | (0.0, 0.99) |
| Jev 1.14 | own history (B1–B8) | 16/16 | 0.0 | EQUIVALENT 0.727, UNRESOLVED 0.273 | (1.8, 8.6) | (0.59, 1.0) |
| Mercury | no own-history (A1–A7) | 7/14 | 0.0 | EQUIVALENT 0.571, UNRESOLVED 0.260, LATENT-FAVOURED 0.169 | (0.0, 23.6) | (0.0, 1.0) |
| Mercury | own history (B1–B8) | 9/16 | 0.0 | NEITHER 0.792, UNRESOLVED 0.208 | (5.4, 26.6) | (1.0, 1.0) |

Code-path check: the Phase 9 Level-2 wrapper reproduces Phase 8's stored in-family detection rates exactly (0.645, 0.285, 0.470), so HISTORY-FAVOURED is reachable.

> Mercury (P-M) disclosure: production acquisition was stopped at sentinel time point 3 by the original pre-registered drift gate (P9-005). That gate was then shown, from pre-production smoke data and the frozen sentinel composition, to be statistically miscalibrated (estimated false-stop probability 0.37-0.49 under smoke-consistent models). After the stop, it was replaced (P9-006) by a stratified, 1% family-wise rule derived from the same pre-production information only. The first 1,708 responses were retained and are included in every Mercury estimate reported here, and acquisition resumed; P-M passed time point 4 under the amended rule. The amended rule needs a single BOTH-cell sentinel to move about 0.27 to stop, so its sensitivity to drift in that cell is weak; this remains a limitation.
