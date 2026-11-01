<!-- Extract of the study's design records: decisions P8-001 to P8-006, which define the
in-family experiment of Section 6, copied byte for byte. The frozen code verifies their
hashes before it reads the stored in-family results. The remaining records are not part
of this package; PROVENANCE.md summarizes them. -->

### P8-001 — Phase 8 freezes a span-matched ABM sufficiency and detectability experiment (researcher decision)

**Authority.** The researcher accepted the read-only pre-P8-001 independent design
review (overall recommendation **B: family existence is close but needs specific
modifications**), with the researcher modifications recorded in
`docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md`. That document is the
binding Phase 8 protocol (§A–§AR). Its Appendix 1 lists every superseded
provisional handoff statement.

**Scientific object.** Phase 8 is a span-matched model-family sufficiency and
detectability experiment on weekly retailer participation. It does not estimate a
true mechanism, a structural habit parameter, a causal state-dependence
coefficient or an unrestricted "best" ABM. The frozen RQ1/RQ2 wording is
preserved, with an interpretation note (design §C):
- **RQ1** asks which predeclared families contain members that reproduce the five
  primary dynamic moments to a predeclared substantive tolerance.
- **RQ2** asks how often the full frozen procedure distinguishes explicit
  purchase-history dependence from non-history persistent dynamics, and how often
  it falsely favours history when none exists.
- **Claim boundary:** model-family sufficiency ≠ true-mechanism identification
  (design §D, §AM).

**D1 is RESOLVED.** It is family existence, modified into a **span-matched
acceptance-set design** (design §O):
- Only the logit-normal heterogeneity layer (μ, s) is calibrated, to CI-1, and it
  is re-fitted at every dynamic point behind a CI-1 feasibility gate.
- Dynamic parameters are searched over predeclared, rule-based, image-matched
  regions. They are **not point-estimated and are not structural or behavioural
  parameters**.
- The dynamic validation moments necessarily define the acceptance set. RQ1 is
  therefore an over-identifying model-adequacy assessment, not pure held-out
  validation.
- The anchored option D1-(b) is rejected.

**Architectures (design §E–§M).**
- **A0:** homogeneous memoryless; retained.
- **A1:** logit-normal heterogeneous memoryless; the negative control and
  tolerance reference. Its verdict is disclosed as not blind.
- **A1′:** stationary, unit-variance Gaussian AR(1) latent state z with index
  α_i + σ_z·z_it. Purchases never update z. ρ and σ_z are common.
- **A2:** bounded geometric purchase-history stock H_{t+1} = λH_t + (1 − λ)y_t,
  H ∈ [0, 1], with index α_i + δ·H_it. One lag (λ = 0) is a boundary slice, not the
  family.
- **Structural sensitivities, one per family:** a two-state symmetric hidden
  Markov state (A1′) and last-purchase recency (A2).
- **Periodic layer:** shared exploratory only, available identically to A1′ and
  A2.
- **Status of sensitivities:** none can alter primary verdicts.
- **Fairness:** reduced-form image (span) parity, verified nonlinearly in Stage 1,
  with deterministic trimming or STOP. The linear ARMA(1,1) equivalence is a design
  hypothesis, not a theorem.

**Moments (design §P–§R).**
- **Calibration:** CI-1 only.
- **Primary (no compensation):**
  - P1 VI-2 `O-M3`;
  - P2 VI-3 `beta_8`;
  - P3 VI-3 `gamma_2`;
  - P4 VI-6 `delta_log_loss_recency`;
  - P5 VI-4 `active_1`.

  P5 is disclosed as the least fundamental. RQ2 reports its discrimination, and it
  is not replaceable.
- **Secondary:** the remaining VI-1 … VI-6 components, including VI-5.
- **Diagnostic:** S/G versions, K-1/K-2, CI-1 fit, E-1/E-2, the β_2 − β_1 shape,
  continuous distances and acceptance maps.
- No promotion to primary after results.

**Adequacy (design §S–§U).** A project-specific, predeclared three-valued rule
(ADEQUATE / INADEQUATE / INCONCLUSIVE). It is not presented as a standard
econometric test.
- **Equivalence:** a two-one-sided-tests (TOST) check at 0.05 for each moment,
  combined by intersection–union across the five primary moments. There is no
  compensation between moments.
- **Inadequacy:** two-tailed Bonferroni across the five primary moments
  (z* = 2.576).
- **Emulator error:** carried into both rules, through the mean error e_{F,j} and
  the SD factors.
- **Tolerance:** τ_j = max(κ·|m_j^real − μ_j^A1|, 2·SE_j^boot), computed once in
  Stage 1 and used unchanged in RQ2 and the real comparison.
- **κ:** reference 0.25, sensitivity {0.15, 0.25, 0.35}. κ*_F (the minimal
  adequate tolerance) is the primary continuous summary.
- **Labels:** κ-STABLE / κ-DEPENDENT; BOUNDARY-DEPENDENT SUFFICIENCY at design
  caps.
- **Disclosed:** the researcher saw the empirical moments before freezing the
  tolerances.

**Regions (design §V).** Design caps, not estimates:
- half-life ≤ 25 weeks (ρ_max = λ_max = 2^(−1/25));
- σ_z ≤ 2.5;
- Δ_1 ≤ 40 pp for A2.

Identical 12 × 12 grids, half-life log-spaced, with deterministic refinement and
equal budgets.

**Purchase envelope (design §W–§Y).**
- **Terminology:** purchase envelope, envelope interior, purchase-envelope
  analysis operator; "availability window" for the latent concept only.
  "Observation window" is retired going forward. Phase 7 history is not
  rewritten.
- **Primary operator: endogenous purchase envelopes.** Behaviour is simulated over
  the full horizon after burn-in. No assigned empirical pairs and no forced
  endpoints. f and l are each household's own first and last simulated purchases,
  and the identical Phase 7 operator code and ≥ 8-week rule apply.
- **This supersedes the Phase 7 Run B endpoint rule (RB.4) for Phase 8.** That
  includes the provisional P7-002/design §10 mechanics of exogenous, type-
  independent window assignment. The analysis operator O itself is unchanged, and
  the Run B rule remains historical preflight evidence.
- **S-a bracket:** empirical (f, l) as an imposed availability window, with H = 0
  at entry. A conclusion is OPERATOR-ROBUST only if the verdict holds under both
  operators; otherwise it is OPERATOR-DEPENDENT / UNRESOLVED.
- **Common horizon:** weeks −155…102 (156 burn-in weeks, about 6.24 half-lives at
  the cap). A1′ starts stationary; A2 starts at H = 0.
- **Environment:** no calendar component in the primary design (an E-1-fitted
  sensitivity only; real week fixed effects are never copied). No stores,
  products, spending or composition (Module C is not run). N = 2,500 and T = 102.
  Common random numbers follow a deterministic SeedSequence tree, with data and
  truth streams independent of model streams.

**RQ2 (design §AD–§AH).**

| Class | Content |
|---|---|
| T0 | A1 |
| T1 | A1′: ρ ∈ {0.5, 0.8, 0.9, 0.95} × LOW/HIGH amplitude, by a CI-1-only rule |
| T2 | Misspecified non-history: HMM, household-specific ρ, AR plus periodic |
| T3 | A2: Δ_1 ∈ {2.5, 5, 10, 15, 20, 30} pp × λ ∈ {0, 0.5, 0.8} |
| T4 | One masking slice: ρ = 0.9 latent plus history at λ = 0.5 |
| T5 | Late N = 10,000 at four frozen points |

- Q = 200 truth panels per point.
- **Δ_1** (one-week intervention effect) is the primary history scale; Δ_12 is
  secondary.
- **Comparative verdicts:** HISTORY-FAVOURED / LATENT-FAVOURED / EQUIVALENT /
  NEITHER / UNRESOLVED.
- **DET** = P(HISTORY-FAVOURED | A2 truth).
- **FA_PROC** is reported, with max over T2 as the headline. **FA_NAIVE** uses a
  mechanical M3/recency rule.
- **MDE_80(λ)** follows a monotone-envelope rule, subject to max FA_PROC ≤ 0.10.
  Otherwise it is UNDEFINED — PROCEDURE NOT ADMISSIBLE, and "MDE > 30 pp" is a
  valid finding.

**Gates and stages (design §AB, §AC, §AI–§AK).**
- **Perfect-model gate:** true-family ADEQUATE ≥ 0.80, INADEQUATE ≤ 0.05, coverage
  of the not-clearly-inadequate set ≥ 0.90, A1-truth HISTORY-FAVOURED ≤ 0.10.
  Failure means STOP.
- **Emulator:** approved conditionally. Its error must be ≤ 0.1·τ_j(0.15) at
  predeclared validation points and is carried into the rule; otherwise rebuild
  once, then STOP.
- **Stages:**
  - 0: correctness;
  - 1: perfect-model, span check and emulator;
  - 2: a predeclared engineering pilot;
  - 3: full RQ2, archived and hash-recorded;
  - 4: the one-time real comparison;
  - 5: late secondary analyses.

  Stages 1–2 cannot redesign anything. Any substantive change needs a new
  decision, a full RQ2 restart and researcher approval.
- **Compute:** a planning ceiling of about 100–130 core-hours, staged, not
  authorized as one run. Heavy runs (H1, H2, H4, also H5 and H6) require the
  explicit heavy-run warning, checkpointing, deterministic seeds and `caffeinate`.
- **Replication:** R_cal = 20, direct R = 200 (up to 500), Q = 200. A changed
  budget requires researcher approval.

**Stopping rule and outcomes (design §AJ, §AM).** No architecture is modified in
response to Phase 8 results. Failure, equivalence, neither, inconclusive and
operator dependence are each reported as such. Outcome wording for Cases 1–5 is
fixed before simulation. New architectures are exploratory follow-ons only.

**Jev (design §AO).** Last and optional; nothing in P8-001 depends on it. Before
any household-level data reach an external LLM or API, the Complete Journey
licence and applicable university and data policy must be verified to permit it.
Otherwise only synthetic, aggregate or privacy-safe representations are used.

**Updated documents.**
- The Phase 8 input contract is updated only where P8-001 supersedes provisional
  Phase 7 language: D1, the endpoint rule, terminology, the periodic variant and
  the primary set. The historical text is preserved and marked.
- `ROADMAP.md` and `README.md`: status only.
- `RESEARCH_DESIGN_PHASES_7_9.md` and `MOMENT_REGISTER.md`: a superseding
  cross-reference note only.

**Status — what this decision is not.**
- No Phase 8 code exists.
- No household panel has been simulated, and no μ, s, ρ, σ_z, δ or λ value has
  been calibrated or estimated.
- No acceptance set, κ*, verdict, detection rate, false-attribution rate, MDE or
  emulator output exists.
- No ML has been rerun, and no Jev work has begun.
- Phase 1–7 findings, archives and final artifacts are unchanged.
- Phase 8 remains the first active phase. The next step is the Phase 8 Run B
  implementation preflight (design §AR).

### P8-002 — A2 amplitude axis in raw δ; RQ2 history truths on the reachable grid (researcher decision)

**Trigger.** Phase 8 Run B found a contradiction in P8-001 §V.2/§V.3, §AD and §AG,
on synthetic engineering checks only. The normalized geometric stock weights last
week's purchase by (1 − λ), which has two effects:
- the one-week intervention effect Δ_1 is **non-monotone** in δ (saturation);
- its maximum shrinks quickly with memory: about 50–76 pp at λ = 0, 15–28 pp at
  λ = 0.53, 5–10 pp at λ = 0.82 and 0.7–1.6 pp at the 25-week cap.

The frozen A2 axis (Δ_1 = 40·k/12 pp, capped where Δ_1 reaches 40 pp) was therefore
unreachable in most cells. The δ ↔ Δ_1 mapping was ill-posed. Several frozen RQ2
history truths could not exist, and the MDE wording could not be evaluated. The run
stopped and the researcher chose the fix below.

**Decision:**
1. **Search axis.** A2 and A2-recency amplitudes are indexed by raw δ = 5·k/12
   (k = 1…12), with **δ ≤ 5** on the logit scale. The history term's maximum SD (δ/2)
   then equals A1′'s σ_z cap of 2.5. The design-cap face is δ = 5.
2. **Labels.** Δ_1 and Δ_12 become per-point interpretive labels, computed in Stage 1
   with CI-1-calibrated heterogeneity. The P8-001 cap "Δ_1 ≤ 40 pp" is **retired**.
3. **RQ2 truths.** The frozen Δ_1 × λ truth grid is kept. Each truth δ is solved on
   the **ascending branch** of Δ_1(δ), with the peak located on the predeclared scan
   0.25·2^(k/4), k = 0…36. Targets above the reachable maximum are **infeasible truth
   points**, recorded and never replaced. This covers T3, T4, T5 and the
   perfect-model gate subset.
4. **MDE_80(λ).** The monotone-envelope rule applies over the reachable tested Δ_1
   values; infeasible values must form the top of the grid.
   - If 0.80 is never reached, the result is "> X pp", where X is the largest
     reachable tested Δ_1 at that λ.
   - With no reachable point, the result is "NO REACHABLE HISTORY TRUTH".
   - Admissibility (max FA_PROC ≤ 0.10) is unchanged.

**Unchanged.** All other P8-001 content: families, timing, moments, the tolerance and
adequacy rules, κ, operators, stages, gates, budgets and the claim boundary. Fairness
is still established by the Stage 1 nonlinear span check, which may trim or STOP.

**Status.** Design amendment only. No production simulation, calibration to real
data, acceptance set, verdict, detection rate, false-attribution rate or MDE exists.
Recorded in `docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md` ("P8-002
amendment") and implemented in `src/complete_journey/abm/` (Phase 8 Run B).

### P8-003 — Coherent model-adequacy gate (researcher decision)

**Context.** The amendment was made before Stage 1. No production Phase 8 result,
real A1′/A2 comparison or RQ2 result existed. It was triggered by the Phase 8 Run B
synthetic preflight and a read-only statistical gate review.

**Defect.** P8-001's tolerance floor (τ_j ≥ 2·SE_j) combined with the TOST margin
(1.645·σ) counted sampling noise twice. At the true parameter, a floor-bound moment
passed equivalence with probability ≈ 0.274 (direct, R = 200), and exactly 0 under
the emulator SD factor. The 0.80 perfect-model gate was therefore unattainable
whenever any primary moment was floor-bound.

**Decision.** The authoritative text is
`docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md`, "P8-003 amendment".
1. **Object.** Δ_j(θ) = E_θ[m_j] − E_P[m_j], with the same estimator, N/T design,
   operator and pipeline.
   - θ is adequate iff |Δ_j| < τ_j for every primary j.
   - A family is adequate iff some admissible interior point is.
   - This is model-family sufficiency, not mechanism identification.
2. **Tolerance.** τ_j(κ) = max(κ·|E_j|, ρ_j), with E_j = m_j^real − μ_j^A1 and the
   design-resolution floor ρ_j = c·SE_j, where
   c = (z_0.95 + Φ⁻¹((1 + 0.80^{1/J})/2))·√(1 + (1.25·1.2)²/50)/(1 − 0.1).
   - c = 4.1598 for J = 5. The review's "≈ 4.13" was the tighter worst-case figure;
     the approved formula governs.
   - ρ_j is design resolution, not a substantive threshold, a discrepancy estimate or
     a confidence interval.
   - A moment is RESOLUTION-LIMITED at κ when ρ_j > κ·|E_j|. Verdicts relying on such
     moments carry "WITHIN DESIGN RESOLUTION".
3. **Uncertainty.** σ±_d,j = √(SE_j² + (g±·s_j(θ))²/R).
   - SE_j is empirical: the archived bootstrap SE for the real panel, the truth-point
     across-panel SD for RQ2 panels.
   - s_j(θ) is the model-panel SD.
   - g± = (1.25, 1/1.2) on the Monte Carlo term only under the emulator, 1 under
     direct simulation.
   - The emulator mean-error bound e_j is added for adequacy and subtracted for
     inadequacy (0 under direct simulation).
4. **Rules.**
   - EQ_j: |d_j| + e_j + z_0.95·σ⁺ ≤ τ_j.
   - OUT_j: |d_j| − e_j − Φ⁻¹(1 − 0.05/(2J))·σ⁻ > τ_j.
   - ADEQUATE requires all five moments (intersection–union, no correction).
     INADEQUATE requires any one (two-tailed Bonferroni). Otherwise INCONCLUSIVE.
     The conditions cannot overlap.
5. **κ\*.** With U_j = |d_j| + e_j + z_0.95·σ⁺:
   - κ*_j = 0 (resolution-limited) if U_j ≤ ρ_j, otherwise U_j/|E_j| (+∞ if E_j = 0);
   - κ*(θ) = max_j κ*_j, and κ*_F = min over θ;
   - plus the descriptive min over θ of max_j |d_j|/|E_j|.

   κ ∈ {0.15, 0.25, 0.35}, reference 0.25. κ_res,j = ρ_j/|E_j| is reported in Stage 1
   before any family comparison.
6. **Perfect-model hard gates (failure means STOP):**
   - G1: pointwise truth inclusion P(θ_true ∈ Acc_F(κ)) ≥ 0.80 at every κ;
   - G2: family ADEQUATE ≥ 0.80 on the production grid plus refinement (κ = 0.25);
   - G3: true-family INADEQUATE ≤ 0.05;
   - G4: P(θ_true ∈ not-clearly-inadequate set) ≥ 0.90;
   - G5: under A1 truth, P(HISTORY_FAVOURED) ≤ 0.10.

   "Coverage" means pointwise truth inclusion only; the acceptance set is not a
   confidence set.
7. **RQ2** reports operating characteristics and never tunes κ, ρ, α, the gates or
   the thresholds after simulation.

**Unchanged:**
- P8-002;
- architecture definitions;
- parameter caps (δ ≤ 5, σ_z ≤ 2.5, the half-life cap);
- the span-parity gate and trimming rule;
- RQ2 truth classes and grids;
- the MDE definition, which now uses the corrected adequacy rule;
- stages, budgets, operators, the seed design and the claim boundary.

Run B risk 1 is resolved. Risk 2 (A2 long-memory span) remains open for Stage 1.

**Status.** A gate amendment, implemented in `src/complete_journey/abm/adequacy.py`,
`config.py`, `rq2.py` and `evaluate.py` (Phase 8 Run B.1). There is no Stage 1, no
production simulation, no real comparison and no verdict.

### P8-004 — Emulator validation by common-random-number pairing (researcher decision)

**Context.** The amendment was made during Phase 8 Stage 1, before any S1-B emulator
build or validation result existed. Using the Stage 1 tolerances, the frozen S1-B
gate was shown to be statistically unattainable on Monte Carlo noise alone, even
with an exact emulator. Two noisy estimates were compared under a max-over-~250 rule:
- emulator nodes at R = 50 against an independent direct reference at R = 200;
- P(SD criterion passes) ≈ 10⁻¹⁹;
- P(mean criterion passes) ≈ 0.006 to 0.24.

The emulator's own Monte Carlo noise is already in σ_d (the s²/R term, P8-003), so
the gate should measure only approximation (interpolation) error.

**Decision.**
1. **Reference.** The emulator validation reference is direct simulation at each
   validation point using the **same 50 MODEL-replicate streams** as the reference
   table (common random numbers). The comparison then isolates interpolation error.
2. **Thresholds unchanged:** |μ_em − μ_direct| ≤ 0.1·τ_j(0.15), and
   s_em ∈ [0.8, 1.2]·s_direct, for every primary moment at every validation point.
3. **Diagnostic.** The independent R = 200 direct comparison is kept as a reported,
   non-binding diagnostic.
4. **Budgets.** Emulator-mediated adequacy uses the full budgets e_j = 0.1·τ_j(0.15)
   and g± = (1.25, 1/1.2), exactly the worst case in the P8-003 resolution-constant
   derivation.

**Status.** Recorded only. S1-B was not executed, because S1-A (span parity) stopped
under the frozen rule. Implementation is deferred to any authorized Stage 1
continuation, and requires a new design-snapshot version. Unchanged: P8-001/2/3, the
architectures, caps, span rule, gates and budgets.

### P8-005 — Phase 8 replaces span parity with independent CI-feasible family sufficiency and adds a heterogeneity robustness bracket (researcher decision)

**Trigger.** The frozen S1-A span gate stopped Stage 1:
- coverage 0.264 / 0.246 against 0.90 required;
- A2 CI-1-feasible at 69 of 144 points, A1′ at 144 of 144;
- a common span with essentially no long-horizon persistence.

A read-only post-Stage-1 review diagnosed four contributing factors:
- a structural divergence of the nonlinear family images, not predicted by the
  linear approximation;
- equilibrium reshaping of the long-run activity distribution under positive
  long-memory feedback;
- a logit-normal heterogeneity layer that already used most of the CI-1 tolerance for
  memoryless A1 (0.0144 of 0.02);
- a span-parity fairness requirement with no standing in the literature, which by
  construction trims away the regions where the families differ.

**Decision.** The authoritative text is
`docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md`, "P8-005 amendment".
1. **Governance.**
   - Stage 1 under snapshot `e59d101b…` (commits `6b0f1e1`, `9934118`) is a FAILED
     PREREGISTERED DESIGN GATE: immutable, never rewritten or deleted, eventual
     disposition an appendix or governance note.
   - Disclosed lessons: span parity was unsupported by the nonlinear CI-feasible
     images; the logit-normal A1 baseline consumed substantial CI-1 tolerance.
   - No parameter value, moment, tolerance, cap, CI-1 threshold, mixture bound, truth
     point or emulator setting is chosen from Stage 1 outcomes.
2. **Fairness.** Equal reduced-form span is retired as a prerequisite. The new
   principle is COMMON RULES, NOT EQUAL EXPRESSIVE POWER: same data, CI-1 gate,
   heterogeneity arm, operator, moments, P8-003 rule, budgets and stopping rules;
   predeclared regions; no post-result modification. Expressive-power differences
   are measured and reported, not trimmed.
3. **RQ1 (future execution):** within each predeclared family and region, does a
   parameterization satisfy both the frozen CI-1 gate and the frozen P8-003 rule on
   the held-out primary moments? Each family's CI-feasible representational
   restrictions are reported. **RQ2:** the operating characteristics of the frozen
   procedure, decomposed into representational exclusion and diagnostic
   discrimination.
4. **Representational exclusion.** A three-state population-level classification
   (REPRESENTABLE / EXCLUDED / INDETERMINATE) over the CI-feasible set only, using
   the validated emulator (R_em = 50), the P8-003 tolerances, the e_j budget, the
   g± emulator-SD factors and Monte Carlo margins, at each κ (reference 0.25).
   Diagnostic discrimination applies the full finite-sample procedure to
   representable truths.
5. **Reporting:**
   - FA_PROC_OVERALL and FA_PROC_REP, plus representability shares;
   - DET_OVERALL and DET_REP, plus shares;
   - MDE_80 stays PRIMARY, annotated; MDE_80_DIAG is SECONDARY;
   - the P8-003 FA admissibility condition is unchanged.
6. **Span analysis** becomes a descriptive representational map: no 0.90 coverage
   requirement, no 50% retention floor, no trimming, no equalization of regions.
7. **CI-1** is unchanged: hard gate at 0.02 under the frozen metric, not loosened or
   converted. The continuous discrepancy is always reported.
8. **Heterogeneity bracket.**
   - The logit-normal layer stays PRIMARY. One robustness layer is added:
     α_i ~ π·N(μ_1, s²) + (1 − π)·N(μ_2, s²), with μ_1 < μ_2.
   - It is applied identically to A1, A1′, A2 and the relevant sensitivities, and
     replaces the two-point mass-mixture sensitivity.
   - **Frozen specification:**
     - bounds π ∈ [0.10, 0.90], d = μ_2 − μ_1 ∈ [0.25, 6], μ_1, μ_2 ∈ [−5, 5]
       (so μ_1 ≤ 4.75), s ∈ [0.05, 3.0];
     - Nelder–Mead on bounded transforms, three deterministic moment-matched starts
       from the primary solution, projected into the interior;
     - the frozen CI-1 objective and the 0.02 gate;
     - INFEASIBLE points handled as in the primary arm (§S.4); MIXTURE-BOUNDARY as
       a disclosure flag;
     - STOP if A1 fails under the mixture, with HETEROGENEITY BRACKET UNAVAILABLE
       as the default outcome and no change to the frozen specification;
     - τ_j and the amplitudes taken from the primary arm;
     - the mixture parameters are nuisance parameters (a π–s identification ridge
       at small d is disclosed) and are never interpreted substantively.
   - **Labels:** ROBUST TO FROZEN HETEROGENEITY BRACKET, or HETEROGENEITY-DEPENDENT.
     The preferred arm is never chosen.
9. **Calibration frontier** (SECONDARY / DESCRIPTIVE / POST-STAGE-1 PREDECLARED).
   Continuous CI-1 discrepancy against the P1–P3 images over the full grid (including
   CI-infeasible points), per family and arm. It is never a gate, never alters
   verdicts, and never justifies loosening 0.02.
10. **Perfect model, emulator, RQ2.**
    - G1–G5 apply per family and arm, with INFEASIBLE TRUTH recorded and never
      replaced.
    - P8-004 is carried forward unchanged, per arm.
    - The T0–T5 grids are preserved with representability annotations, and RQ2 runs
      under both arms (logit-normal primary).
11. **New Stage 1 required.**
    - Run label `S1-P8005`, seed stage key 11, a new archive and work namespace, and
      a new immutable snapshot plus authorization before any computation.
    - Order: S1A maps → S1B emulator → S1C perfect model → S1D frontier (not a
      gate); then, only if S1B and S1C pass, Stage 2 → RQ2 → the one-time real
      comparison. No real comparison before RQ2.
    - Expected compute exceeds the original ceiling (≈ 120–200 core-hours), which
      needs explicit acceptance at authorization.

**Unchanged:**
- architectures (no centering, no new A2, no timing change);
- δ ≤ 5, σ_z ≤ 2.5, the half-life cap, the grid rules and the P8-002 axis;
- P1–P5 and the moment tiers, κ, P8-003 (c, EQ, OUT, κ*, the verdicts, the
  resolution-limited qualifier);
- operators, the S-a bracket, burn-in, calendar disposition, N and T;
- the seed tree and CRN policy (only the MIXTURE stream code is added);
- the T0–T5 truth grids, P8-002 and P8-004;
- the claim boundary, extended in design §P8-005.13.

**Status.** Design amendment only. No simulation, calibration, mixture fit, map,
emulator, perfect-model, frontier, RQ2 or real-comparison output exists under P8-005.
The next step is Run C, the light P8-005 implementation preflight.

### P8-006 — Phase 8 deadline-constrained direct plug-in completion (researcher decision)

**Context and honesty statement.** This is an MS Economics degree project with a hard
deadline: the remaining Phase 8 computation must finish in ≤ 48 hours (preferred) and
at most 72 hours of wall time on the researcher's M2 Pro Mac. The decision is made
before any RQ2 result or real A1′/A2 verdict exists. Its scope choices are
deadline-driven and are recorded as such, not as methodological discoveries.

**Compute evidence.**
- The S1-P8005 run sustained 107.458 core-hours in 19.027 h with 6 workers.
- Its A2 reference-table emulator failed the P8-004 CRN-paired gate after the one
  predeclared refinement, in both heterogeneity arms (S1B STOP, commit `a621506`).
- Direct A2 evaluation with per-panel recalibration costs about 112 s (logit-normal)
  or about 390 s (mixture) per panel and grid point. Brute force would take
  ≈ 143,500 core-hours. Decision-exact witness certification would still take
  ≈ 21,400 core-hours (about five months). Both are rejected as infeasible.

**Decision.** The authoritative text is
`docs/phases/PHASE_08_ABM_SUFFICIENCY_DETECTABILITY.md`, "P8-006 deadline-constrained
completion".

1. **Direct plug-in procedure.**
   - Synthetic RQ2 and perfect-model panels are evaluated against a fixed model
     table: every family point at its calibration to the real CI-1 population
     target (the S1A calibrations, seed stage key 11).
   - Families are not recalibrated to each synthetic panel's noisy CI-1.
   - RQ2 is interpreted as the operating characteristics of the procedure
     conditional on the empirical population calibration.
   - Calibration-target sampling noise is not integrated into every RQ2 panel. This
     is disclosed and bounded by item 6.
   - The real panel is calibrated to its own CI-1 exactly as before.
2. **No load-bearing emulator.**
   - Every confirmatory model image is direct simulation, so P8-003 runs with
     e = 0 and g± = 1.
   - The A1/A1′ emulator is diagnostic and historical only; the A2 emulator is
     search history only. P8-004 and the H2 check are historical evidence and are
     retired from the confirmatory procedure.
3. **Reuse, with an integrity-checked import.**
   - Imported from the S1-P8005 checkpoints:
     - the S1A calibrations and direct R = 50 images (all 144 coarse points, both
       families, both arms, A1);
     - the A1 resolution and frozen tolerances;
     - the gate-truth configurations;
     - the logit-normal gate panels;
     - the A1′ H2 pairs.
   - The import re-verifies snapshot `08a6a672…`: every simulation-semantic source
     file is byte-identical, and every record and config hash is checked. Imported
     checkpoints stay immutable.
   - Coarse images are extended from R = 50 to R = 200 by continuing MODEL
     replicates 50…199 of the same streams. A bounded test verifies that
     replicates 0…49 reproduce the imported images bit for bit, and that pooling
     equals a clean run.
4. **Refinement and precision.**
   - Confirmatory procedure for RQ2, S1C and the real panel: the coarse 12 × 12 grid
     plus the frozen first refinement level.
   - Exact pruning: a cell whose four corners are all CI-1-infeasible is never
     refined, because it has no ADEQ corner and no finite κ*.
   - R = 200 on the coarse grid and at θ_true; R = 50 at refinement points; R = 200
     real-panel audits at the κ* argmin and up to 10 accepted points.
   - Second-level refinement is a secondary calculation for the real panel only.
5. **Known-truth experiment.**
   - Full T0–T4, every frozen reachable point, Q = 200, both heterogeneity arms,
     Stage 3 DATA keys. T5 (N = 10,000) is deferred.
   - Unreachable or infeasible truths stay labelled and are never replaced.
   - A separate Q_pop = 200 population block (panel index offset 10,000) feeds the
     representability classification, so there is no finite-panel selection
     leakage. The full representational-state × verdict table is reported.
   - FA_NAIVE, FA_PROC_OVERALL/REP and DET_OVERALL/REP carry exact Clopper–Pearson
     intervals.
   - MDE_80 is unchanged. A confidence-aware bracket (one-sided 95% lower bound ≥
     0.80) is added. Censoring is reported as "> X pp", never extrapolated.
     MDE_80_DIAG is secondary.
   - Heterogeneity label for RQ2 summaries: ROBUST iff FA admissibility and MDE_80
     at every λ agree across the arms (predeclared).
6. **Calibration-noise checks** (predeclared before any result; the frozen Stage 2
   pilot truths; θ = {1, 5, 9}²):
   - **Checks:**
     - A2 logit-normal: 3 truths (T1 ρ 0.9 HIGH, T2 HMM q 0.05, T3 λ 0.5 Δ_1 10)
       × 20 panels × 9 θ;
     - A2 mixture: 2 truths (T2 HMM q 0.05, T3 λ 0.5 Δ_1 10) × 10 panels × 9 θ;
     - A1′ mixture: 1 truth (T2 HMM q 0.05) × 10 panels × 9 θ.
   - **Panels:** the first n successful RQ2 panels in q order of each truth.
   - **Comparison:** direct per-panel recalibration at R = 50 against the R = 50
     plug-in image on common streams.
   - **Qualifier:** CALIBRATION-NOISE SENSITIVE if any of these holds (it never stops
     the run):
     - p90 |Δμ_j|/τ_j(0.25) > 0.25 for any moment, over pairs feasible on both sides;
     - pointwise status disagreement > 5% at any κ, over pairs feasible on both sides;
     - the CI-1 feasibility mismatch rate > 5%.
   - **Recorded A1′ logit-normal evidence** (349 S1-P8005 H2 pairs: recalibrated at
     R = 25 against plug-in at R = 50):
     - median |Δμ|/τ(0.25) = 0.0094, 0.0262, 0.0072, 0.0092, 0.1042 (P1–P5);
     - maximum = 0.0448, 0.0523, 0.0367, 0.2782, 0.3523;
     - status agreement 1.000, 0.986, 0.980 at κ 0.15, 0.25, 0.35;
     - zero ADEQ↔INADEQ flips.
7. **Perfect-model gates.**
   - P8-003 G1–G5, thresholds unchanged, Q = 200, both arms. A family with no
     assessable gate point fails.
   - Under plug-in, T0/T1 G1/G2 are primarily design-resolution and pipeline
     checks; item 6 is the complementary evidence.
   - A gate truth outside the search lattice still stops the run before S1A. RQ2
     truths are never inserted, so they are only labelled.
8. **Real reveal order (mandatory).**
   - Stage 1 hard gates pass, then the full RQ2 completes, then RQ2 artifacts are
     written write-once, then the RQ2 freeze record (hash of every RQ2 artifact) is
     written. Only then may the one-time reveal ledger be created.
   - Only through that ledger can the real P1..P5 or the real E-2 windows be read.
     `config.AUTHORIZED_STAGE` stays at 1, so no legacy path can read them.
9. **Real comparison and S-a.**
   - Real comparison: both families, both arms, direct only. It reports the CI-1
     feasible set, adequate and not-clearly-inadequate sets, κ*_F and its argmin,
     boundary dependence, binding and resolution-limited moments, comparative
     verdicts and the frozen heterogeneity label.
   - Logit-normal S-a bracket: secondary. Its calibrator equals the frozen one except
     that an undefined CI-1 objective is +∞, pinned bit-identical where defined.
   - Mixture S-a is deferred.
10. **Reliability.**
    - a `ProcessPoolExecutor` that recovers from worker death and hangs;
    - a detached launcher in its own session under `caffeinate`;
    - a run lock with stale detection;
    - a disk guard that stops below 10 GB free;
    - last-error, status, progress-index and plan-manifest files;
    - sharded checkpoints keyed by a job-semantics hash;
    - a read-only monitor;
    - 7 workers if a 15-minute throughput check beats the 6-worker baseline, else 6
      (a computational choice only).
11. **Deadline governance.** No automatic scope reduction. If projected completion
    exceeds 72 hours, the run pauses at the next safe checkpoint and the researcher
    decides.

**Secondary:** the calibration frontier, span maps, logit-normal S-a, the
calibration-noise checks, the MDE bracket and the mixture diagnostics.

**Deferred** (a publication extension):
- T5 (N = 10,000);
- mixture S-a;
- full second-level refinement;
- exhaustive per-panel recalibration RQ2;
- the unlimited-compute design.

**Unchanged:**
- architectures, caps, grid rules, P1–P5, κ;
- the P8-003 c, EQ, OUT and κ* rules;
- the CI-1 gate and calibrators;
- the T0–T4 truth grid;
- the seed tree;
- P8-001 to P8-005, except where items 1–11 explicitly supersede them.

### End of extract
