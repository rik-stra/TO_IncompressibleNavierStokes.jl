# Card: mechanism tests 3 and 4 — does the signed mean push in E[0,6] decide the long runs? opened 2026-11-13

Rik, via the paper session (2026-11-13): "do all of them" (the mechanism package of the two critical reviews,
`meta_files/step5_critical_review_2026-11-13.md` §0–2 and D2, `meta_files/step4_critical_review_2026-11-13.md` § on B).
Tests 1 and 2 are the step-4 session's. This card also carries the reach-k1a variants forward (`step5_reach_k1a.md`; the
HOLD there is lifted in this form).

**The hypothesis under test** (the reviews): B's long-run collapse comes from a bulk offset of its generator mean E_ε g in
E[0,6], about −0.05 ‰ of the level per step (≈ −0.015 target sd) in every level bin up to the 75 % quantile, deepening to
about −0.3 ‰ at the drained lows (where it cancels about half of k1's +0.59 ‰ restoring push). One-step checks weigh it as
δ², the solver integrates it. Context: a noise level 8 % wider at k1a's mean does not collapse (step-4 noise test, D 0.32).

🔒 **Predictions and rules below are fixed before any of these runs. Once any coupled number exists, nothing above "Post
hoc" changes.**

## Test 4: k1a + a constant E[0,6] offset (is the bulk offset alone sufficient?)

- **Closure** `LinReg1_nt_CNN_crps_nt_headcorr_off2m0.015` (`tools/offset_closure.jl`, built 2026-11-13 on the laptop): k1a
  deployed unchanged except `scaling.dq_offset` = −0.015 target sd in E[0,6] (−8.700e-5 physical, so = 5.800e-3), added to
  the deployed correction every step (`ts_lstm_online.jl`; supported for a non-generator closure; no src change). That is
  **−0.044 ‰ of E[0,6]'s reference mean level per step** (level 1.960), constant in physical units, so relatively larger at
  the lows (−0.073 ‰ of the 1 % quantile's level 1.200). sha256 `5e66f353d9e0028e8195df09f26b860d2c8bbfa23902c6c08f87612e51763e1f`.
- **Verified:** weights identical element for element; scaling identical except `dq_offset`; the deployed closure
  (stochastic = false) against k1a at the first predicted step on 20 record windows: dQ differs by the offset in E[0,6]
  (residual 2.8e-19) and by 0.0 in the other QoIs; Σ unchanged (bd, LR identical).
- **Runs:** 5 × 100 TU with k1a's replica seeds (`run_online.sh lstm 2 <r>`), gpu_a100.
- **Rule** (the paper session's, fixed here): B-like collapse (D ≥ 0.6 or b ≤ −6 %) → the bulk offset alone is sufficient;
  D ≤ 0.4 and b ≥ −4 % → B also needs its low-tail push, or something else; in between → graded.
- **Prediction (written 2026-11-13, before the run): graded, not a collapse.** D 0.35–0.55, b −4 to −5.5 % (k1a −3.3 %).
  Why: −0.044 ‰ per step integrated over E[0,6]'s restoring time (100–190 steps) shifts the mean level by about −0.5 to −1 %
  (the reviews' estimate); k1's restoring push at the lows (+0.13 to +0.59 ‰) is untouched, unlike in B, so the slide
  should stop short of the drained states. Against that: in the M4 era a constant offset pushed low excursions into
  collapse (2026-09-27, the reason `offset_ref` exists); here the offset is ~10× smaller than k1's low-tail push.

## Test 3: the drift gate, prospective

- **Measure** (`analysis/drift_gate.jl`, laptop): for W0, a, a-c, b, c (and B as calibration), E_ε g against the frozen k1
  mean (= k1a's) from 32 fresh antithetic pairs per window, on the record (10–100 TU, every 10th step) and on k1a's five long
  runs (every 20th step); the bulk E[0,6] offset (level ≤ 75 % quantile), its low tail (≤ 5 %), and the profile over the
  level for all six QoIs, in ‰ of the reference mean level per step. W0 must give exactly 0.
- **Threshold, prediction per variant, and which two go coupled:** written here after the measurement and before any long
  run (next section). The threshold is calibrated on B and k1a — post hoc for those two (their long runs are known).
- **Runs:** 5 × 100 TU with k1a's replica seeds, for the predicted best variant and one predicted failure. If
  the predicted best is a-c, its centring code (src, `Cv.centred`) needs Rik's commit approval and a Snellius pull.
- **Verdict:** both predictions right → the gate predicts the long runs; otherwise → it does not.

### Measured (2026-11-13, laptop; `drift_gate.jl`, logs in the session scratchpad `drift_<v>.log`)

E[0,6], ‰ of the reference mean level per step; ± = SE over the five runs (k1a's runs) or 10 blocks (record); 32 fresh
antithetic pairs per window; record 3 600 windows, k1a's runs 10 000.

| closure | bulk, record | tail, record | **bulk, k1a's runs** | **tail, k1a's runs** |
|---|---|---|---|---|
| W0 (= k1a) | 0 | 0 | **0** (the check: exact) | **0** |
| a (L frozen) | −0.002 ± 0.005 | −0.023 ± 0.020 | −0.017 ± 0.001 | −0.034 ± 0.007 |
| a-c (L frozen, centred) | +0.090 ± 0.006 | +0.081 ± 0.015 | +0.059 ± 0.002 | +0.025 ± 0.007 |
| b (L frozen, CRPS) | +0.037 ± 0.004 | +0.013 ± 0.008 | +0.017 ± 0.002 | −0.032 ± 0.010 |
| c (L trainable, sd cap) | −0.012 ± 0.005 | −0.030 ± 0.017 | −0.032 ± 0.001 | −0.054 ± 0.004 |
| B (collapsed coupled) | −0.026 ± 0.004 | −0.016 ± 0.013 | −0.050 ± 0.001 | −0.071 ± 0.006 |

B on k1a's states reproduces the reviews' like-for-like bins (bulk −0.05; lowest-1 % bin −0.110 against their −0.112).
Profiles for all six QoIs are in the logs; the other QoIs' pushes are of mixed sign and below |0.03| ‰ in the bulk bins
for every variant.

### 🔒 Threshold and predictions (fixed 2026-11-13, after the measurement, BEFORE any test-3 long run)

**Gate (on k1a's long-run states, where the coupled runs live): PASS iff the bulk E[0,6] push ≥ −0.025 ‰ per step AND the
low-tail push ≥ −0.035 ‰ per step**, i.e. half of B's (−0.050 / −0.071), with k1a (W0) at 0 / 0. Calibrated on B and k1a:
**post hoc for those two** (their long runs are known). One-sided: the hypothesis is an anti-restoring (downward) push
that drains E[0,6]; a positive push moves the level up, away from the drained states (step 5 on LinReg1, +0.10 ‰ per step
in E[0,6], ran closer to the reference than LinReg1, D 0.48 against 0.84). The record values are reported, not gated
(B's record bulk, −0.026, would pass a record gate: the offset grows off the data).

| variant | bulk (k1a runs) | tail (k1a runs) | gate | **predicted long-run verdict (D ≤ 0.4 = pass)** |
|---|---|---|---|---|
| W0 | 0 | 0 | pass | pass (it is k1a: D 0.32) |
| a | −0.017 | −0.034 | pass (tail at the margin) | pass |
| a-c | +0.059 | +0.025 | pass (largest margin) | **pass** |
| b | +0.017 | −0.032 | pass (tail near the margin) | pass |
| c | −0.032 | −0.054 | **fail** (both) | **fail** (D > 0.4) |
| B | −0.050 | −0.071 | fail | fail (known: D 1.03) |

**Coupled (test 3):** the predicted best = **a-c** (the largest margin on both counts) and the predicted failure = **c**
(fails both, but less far than B, so it tests the threshold, not just B). a-c deploys with the centring code (`Cv.centred`,
src/ts_lstm.jl + export, V102), which needs Rik's commit approval and a Snellius pull; c deploys with `dff6b5b0`.
**Verdict rule:** a-c D ≤ 0.4 AND c D > 0.4 → the gate predicts the long runs; otherwise it does not.

## Scoring (both tests)

As before: D (5-run range), b, CRPS₁₀₀, e₁₀₀, SS ratio, guard, the KS split (raw / mean-matched / mean- and sd-matched), the
tail split of the bias, collapse episodes per 100 TU; against k1a and B. Hand-over `mechanism_tests_step5.md` in the
side-session notes.

## Post hoc
- 2026-11-13, **code committed `6bf53ed8`** (Rik: "Yes, commit and submit. But you need to check with the step 4 agent";
  the step-4 session's level gate `565fc3b2` was already pushed, so this commit holds only the step-5 session's hunks;
  that session informed) and pushed. Full suite 4034 pass, 1 fail = the pre-existing V36 (after `offset_closure.jl`'s
  check loop was moved into a `let`). Snellius pulled `565fc3b2` → `6bf53ed8`.
- **Submitted 2026-11-13** (gpu_a100, `run_online.sh lstm 2 <r>`, k1a's replica seeds, `-t 00:30:00`, long runs only;
  closures copied, sha256 equal: a-c `5a6f82fe…`, c `548500a6…`, test 4 `5e66f353…`). Deployment check on Snellius's
  checkout (temporary script, removed): a-c is centred and at ε = 0 equals k1a's mean (0.0); c's frozen mean + Ws z =
  k1a's mean (0.0); the test-4 closure has k1a's weights and dq_offset −8.700e-5 in E[0,6] — PASS. Jobs:
  - test 3, a-c (predicted pass): `29227171`–`29227175`;
  - test 3, c (predicted fail): `29227177`, `29227178`, `29227180`, `29227182`, `29227183`;
  - test 4, k1a + E[0,6] offset (predicted graded): `29227184`–`29227188`.
- 2026-11-13, **scored** (Rik: "check if the long runs are finished", "Yes, copy and score them"). All 15 job records
  COMPLETED; c's replica 5 (`29227183`) blew up at step 11 575 (t ≈ 28.9 TU; "Unreasonable large QoI", then NaNs), so c is
  stable 4 of 5. Copied Snellius → laptop, the 15 replicas' sha256 equal. Scored with `longrun_stats.jl`, `gate_census.jl`
  and copies of the step-4 reviewer's `s4_longrun_decomp.jl`, `tail_bias.jl`, `peak_collapse.jl` (c's splits on its 4
  stable runs). Hand-over: side-session notes `mechanism_tests_step5.md`.

  | closure | prediction | D (single runs) | b | CRPS₁₀₀ | e₁₀₀ | KS raw / mean-m. / mean+sd-m. | low-tail part of b | collapses / 100 TU | guard | stable |
  |---|---|---|---|---|---|---|---|---|---|---|
  | k1a | — | 0.32 (0.17–0.51) | −3.3 | 0.369 | 0.655 | 0.32 / 0.19 / 0.20 | −1.7 | 6.4 | 0.002 % | 5/5 |
  | **a-c** (test 3) | pass | **0.53** (0.43–0.77) | **+1.9** | 0.365 | 0.632 | 0.53 / 0.32 / 0.25 | +0.3 | 2.2 | 0 | 5/5 |
  | **c** (test 3) | fail | **0.73** (0.43–0.88) | −7.6 | 0.381 | 0.734 | 0.73 / 0.49 / 0.41 | −5.7 | 9.5 | 4.75 % | 4/5 |
  | **k1a + offset** (test 4) | graded | **0.49** (0.41–0.66) | −4.5 | 0.370 | 0.681 | 0.49 / 0.19 / 0.19 | −2.2 | 7.6 | 0 | 5/5 |
  | B | (fail) | 1.03 (0.76–1.32) | −11.4 | 0.478 | 0.932 | 1.03 / 0.67 / 0.53 | −9.6 | 9.6 | 2.0 % | 5/5 |

  - **Test 3 verdict (the rule fixed above): a-c predicted pass, D 0.53 → wrong; c predicted fail, D 0.73 → right. Not both
    right → the gate as specified does NOT predict the long runs.** a-c fails HIGH (mean / ref 1.01–1.03, b +1.9 %): the
    one-sided gate's assumption that an upward push is harmless was wrong. Post hoc: the bulk push maps monotonically onto
    the long-run bias (B −0.050 ‰ → −11.4 %, c −0.032 → −7.6, k1a 0 → −3.3, a-c +0.059 → +1.9); a two-sided bound would
    have sorted the four runs.
  - **Test 4 verdict: graded** (D 0.49, b −4.5 %), as predicted (D 0.35–0.55, b −4 to −5.5). The constant offset moves k1a's
    mean by about −1.2 points and leaves its shape (KS matched 0.19 / 0.19) and collapse rate (7.6 / 100 TU, as k1a ×1.08)
    unchanged: **the bulk offset alone is not sufficient for B's collapse**; B also needs its low-tail push, or something
    else.
