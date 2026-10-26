# Scientific audit and change log — "Signal or Coincidence?"

Audit date: 2026-10-26. v1 = the submission draft frozen in `results/v1_frozen/` (data checksums in
`results/v1_frozen/SHA256SUMS`). v2 = this revision. No v1 data were modified or re-seeded; every v2
experiment writes to new files (`results/v2_*.jsonl`, `x_*.jsonl`, `rep_*.jsonl`).

## 1. Severity-ranked audit

| # | Severity | Location (v1) | Issue | Consequence | Fix (v2) |
|---|---|---|---|---|---|
| 1 | **Critical** | §4.1, Table 1, abstract ("direct message ... up to 90%") | The direct-message arm used a *separate* preparer call, so the approver saw a different set of tickets **and** the message. Non-message inputs were not held fixed. | The "message" effect was confounded with that call's ticket marking. | New **message-only** arm: the same message shown with canonical (information-free) tickets, compared with the canonical arm (identical prompt except for the message). The message alone is smaller: direct+tickets vs message-only = gpt-4.1-mini 86%→72%, gemini-2.5-flash 90%→41%, llama-3.3-70b-instruct 61%→47%, gpt-4o-mini 77%→61%. Wording now says "message plus the writer's own tickets". |
| 2 | **Critical** | Prop. 1, Table 1 tests, "below chance" claims | The null fixed P(G=T)=1/4 although approvers can abstain ("none"). The correct null is P(G=T)=P(valid guess)/4. | One-sided tests stay valid (conservative), but "below chance" and "chance-level" readings were wrong where abstention reached 97–100% (prio-code strip/canonical). | Prop. 1 restated with abstention; recovery tested *conditionally on a valid guess* (exact binomial vs 1/4); abstention reported for every arm. |
| 3 | **Critical** | Abstract, §4.3, §5 | Detecting information flow was presented as evidence of collusion. | Over-claim: authorized cooperation also produces flow. | New **benign-flag** arm (authorized expedite request, honest agents): flow is detected (review-first recovery ≈100%) with no increase in harm. Paper now separates flow / authorized cooperation / covert signalling / joint prohibited objective / violating outcome. |
| 4 | **Critical** | Abstract, §4.3, Discussion ("recomputation ... closes the channel at no measured cost") | The canonicalizer overwrites every agent-writable field, removing the tested channel by construction. | Not evidence of a robust or zero-cost defense. | New **adaptive** arms: sanitizer-aware attacker with (a) priority stripped, (b) priority+tags recomputed but a free-text note field kept (verbatim / redacted / dropped), plus honest-agent utility of notes. Claims narrowed to "confinement in this environment"; migration to the note field is reported. |
| 5 | High | §4.1 "Gemini ... spontaneous", §6 | Both agents are *instructed* to collude; v1 language drifted toward emergence. | Over-claim about motivation. | Stated explicitly: collusive motivation is instructed; only the *signalling convention* is unprompted (no-code arms). |
| 6 | High | Channel-removed arm | "Ticket system unavailable" changes the prompt and may itself shift behavior. | Null arm not structure-preserving. | New **honest-A-tickets** null (normal tickets written without knowledge of T) and canonical arm used as structure-preserving nulls; Prop. 1 covers both. |
| 7 | High | Abstract, §4.4 ("no more often than with channel removed") | Non-significance read as equivalence. | Over-claim. | Paired risk differences with 95% CIs (e.g. Gemini shared − removed harm: +0 pp, 95% CI [-11, 11]), described as "not detectably different; CI includes ±10 pp". |
| 8 | High | App. F, Fig. A4 ("essentially its capacity") | Conflated H(T)=2 bits with carrier capacity (priority alone: 4 tickets × 2 bits = 8 bits; tags log2 5^4 ≈ 9.3 bits). | Wrong statement. | Reworded: codes transmit ≈ all of H(T)=2 bits; carrier capacity is much larger. |
| 9 | High | Prop. 2, Fig. A4 | Marginal I(T;G) under permutation was presented without the conditional quantity. | Marginal ≈0.4 bits understates what an auditor who knows π can decode (I(T;G\|π)). | Both reported: marginal, and a lower bound on I(T;G\|π) via I(T;π⁻¹(G)) (valid since π ⫫ T). Prio code: ≈0.38 marginal vs ≈1.9 conditional. |
| 10 | Medium | Prop. 3, analyze.py `lift` | Per-episode lift is undefined when no same-label non-target exists; v1 silently dropped those episodes (≈7% of seeds). | Estimand changed silently. | Primary estimand is now the label-stratified pooled lift (always defined), with seed bootstrap and a seed-level permutation test; undefined share reported. |
| 11 | Medium | Tables 1–2, forest plot | 18+ uncorrected tests per model. | Inflated false positives. | Pre-declared confirmatory family of 9 recovery contrasts per model with Holm adjustment; other analyses labelled exploratory. |
| 12 | Medium | App. H power curves | Bootstrap-from-observed-data power was read as a general sample-size guide. | Over-generalization (post-hoc power at the observed effect). | Labelled "power at the observed effect size in this environment"; not a general requirement. |
| 13 | Medium | Model provenance | v1 recorded only the requested OpenRouter alias; no provider or snapshot. v2 logs show Llama-3.3-70B served by 8 providers and GPT-4o-mini by OpenAI and Azure. | Provider/quantization heterogeneity is an uncontrolled factor. | Provenance logged from v2 onward (`resp_model`, `provider`); reported as a threat to validity; exact snapshots for v1 marked unverified. |
| 14 | Medium | Monitor comparison | Only an LLM monitor was compared; no simple structural baseline. | Weak comparator. | Added a fixed structural rule (tag mismatch + unique-extreme priority) and a cross-validated logistic monitor. |
| 15 | Medium | Guess elicitation | Recovery relies on the approver reporting a coalition guess. | Self-report may change behavior; unavailable in deployment. | Guess-free arms (`noguess_shared` vs `noguess_canonical`) evaluated with decision-based lift. |
| 16 | Medium | Single sample per cell | No estimate of sampling variability. | Unknown robustness. | Two extra approver samples for 4 critical arms (GPT-4.1-mini, Gemini). |
| 17 | Low | §3 Environment ("break exactly one rule") | 26/288 violating transactions break two rules. | Factual error (already fixed in v1 appendix). | Kept corrected. |
| 18 | Low | Cost statement | "$5.50" mixed cached cost with retries. | Imprecise. | Report logged cost of cached calls and note uncached retries separately. |
| 19 | Low | Haiku | Refusals (≈50%) filled with default tickets. | Haiku arms are not comparable. | Haiku excluded from confirmatory claims; reported only as refusal behaviour. |

## 2. Change log

### Corrected analyses (no new data)
- Abstention-aware null and conditional exact tests (`analyze_v2.recovery_stats`).
- Label-stratified lift with seed bootstrap + permutation test (`strat_lift`, `paired_lift`).
- Marginal vs conditional information (`info`), corrected wording on capacity.
- Holm adjustment over a pre-declared confirmatory family (`CONFIRMATORY`).
- Paired harm risk differences with CIs (no equivalence claims).
- Structural-rule and cross-validated logistic monitors (`structured_monitor`).
- Power curves re-labelled as observed-effect power.

### Newly executed experiments (v2, seeds 0–99, 4 models; all new files)
- `null_honestA`: collusive approver reads honest tickets (structure-preserving null).
- `unilateral_A`: collusive preparer, honest approver.
- `benign_flag` / `benign_null`: authorized expedite request with honest agents; legitimate "review first" readout.
- `msg_only`: private message with canonical tickets (holds non-message inputs fixed).
- `noguess_shared` / `noguess_canonical`: no self-reported guess; decision-based detection.
- `aware_strip` / `aware_strip_canonical`: attacker knows priority is stripped.
- `notes_{verbatim,redacted,none}` and `notes_honest_{...}`: code migration into a free-text field; redaction as a weaker intervention; honest utility of the field.
- Cross-model pairings: every ordered pair of the 4 models, shared and permuted arms (A's v1 tickets reused from cache).
- Repeated generations: approver samples 1 and 2 for removed/shared/permuted/canonical (GPT-4.1-mini, Gemini).

### Wording-only changes
- Collusive motivation described as instructed throughout; "spontaneous" restricted to the signalling convention.
- Five-way distinction (flow / authorized cooperation / covert signalling / joint prohibited objective / violating outcome).
- Defense claims narrowed to confinement in this environment; "zero-cost" removed.
- Related work updated with verified sources (Rippin et al. 2026; Rose et al. 2026; Nakamura et al. 2026).

## 3. Outcomes of the new experiments (final data, seeds 0–99)
Exploratory unless marked Holm. Full tables: paper Appendix C; raw numbers: `results/v2/report.txt`.
- Null arms: removed, canonical and honest-A null all match P(valid)/4 for every model. Honest-A vs
  "unavailable" differs nominally for GPT-4.1-mini (−10 pp, p=0.03; n.s. after ×4 correction).
- Cross-model: Gemini-written tickets are decoded by every other approver (45–55%, p ≤ 3e-4 vs. the
  approver's own null); tickets from the other three preparers are not decoded by anyone (incl. Gemini).
- Message-only: smaller than v1 "direct" (see #1).
- Benign authorized flag: 96–100% review-first flow, no detectable harm increase → flow ≠ collusion.
- Unilateral collusive preparer with honest approver: no detectable harm increase (all 4 models).
- Guess-free lift detects Gemini's channel (p=0.012 permutation); not detected for others (null anyway).
- Sanitizer-aware, priority stripped, no code: only Llama shifts to tags (+11 pp, p=0.013).
- Free-text note survives recomputation: signal migrates for Gemini (+54 pp), GPT-4o-mini (+25 pp), Llama
  (+10 pp, p=0.03); not detected for GPT-4.1-mini. Redaction does not reduce leakage for any model.
  Notes carry no measurable legitimate utility in this environment (−5.5 to +1.5 pp honest accuracy).
- Monitors: fixed structural rule and CV-logistic beat the LLM monitor for all 4 models.
- Repeats: arm-level recovery reproduces across 3 approver samples (e.g. Gemini shared 58/60/60%).
- Provenance: Llama served by 8 providers, GPT-4o-mini by OpenAI and Azure (v2 logs).

## 4. Claim verification
`src/check_claims.py` tests every qualitative claim in `paper/body.tex` against `summary_v2.json`
(final run: ALL PASS). Two claims failed on intermediate data and were rewritten rather than kept:
"honest-A null does not differ from unavailable" and "dropping the note removes information honest
agents use"; two more changed with the final Llama seeds (tag convergence under stripping; note
migration in 3 rather than 2 models).
