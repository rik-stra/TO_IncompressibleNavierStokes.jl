# Card: step 2 — LinReg1 + power-law scale (paper Sec. 6.3), opened 2026-10-15, closed 2026-10-16

WORKFLOW.md Stage 0. Methods text: `closures.tex` §Step 2 (`sec:lever-scale`, Eq. `eq: power law`);
offline evidence `colour.tex` §5.5 (`sec:colour-scale`) and `results_LSTMS.md` §17; implementation
check: `paper/round1_gate01.md` §E (P1–P6) and V83 (`test/test_linreg_scale.jl`, two items added
2026-10-15 after an implementation review).

⚠️ **Opened after the training hindcast** (Post hoc, below), not before it: the closure was added on
2026-10-07 and renumbered from 4p on 2026-10-13 without a card. No full-hindcast number exists yet.

🔒 **Once any full-hindcast number of this closure exists, nothing above "Post hoc" changes.**
Late observations go under "Post hoc", dated.

- [x] Rik has seen this card (date: 2026-10-15; "I've read analysis/cards/step2.md")

## Closure

LinReg path (`D6_CLOSURE=lrs`), the mean of LinReg1 (h = 5, λ = 0, fitted on 1–10 TU), deployed with
the turbulence gate, member seeds `member_seed(k, m)` as LinReg1 (common random numbers: every step
draws the same six standard normals as LinReg1, mapped through the Cholesky factor of Σ_ε instead of
Σ; V83).

| run | closure | the one change | matched partner | artefact (`TO_LRS/`) |
|---|---|---|---|---|
| S2 | LinReg1 + power-law scale | the residual η = μ_η + s(q*) ⊙ ε, s_i = (clip(q*_i)/q_ref,i)^β_i, ε ~ N(0, Σ_ε) | LinReg1 (`D6_LinReg1`, `D6train_LinReg1`, the R2 long runs; reused) | `LinReg1_pl` |

`LinReg1_pl`: β = 0.514 / 0.554 / 1.849 / 1.823 / 1.276 / 1.365 (Z/E[0,6], Z/E[7,15], Z/E[16,32]);
clip / q_ref 0.36–1.56 (Z[16,32]) to 0.56–1.36 (Z[0,6]); fit-window gain 0.489 nats/step, held out
(10–100 TU) 0.477. Built 2026-10-07 on the laptop; `--report` re-checked 2026-10-15; copied to Snellius
2026-10-15 (sha256 of `LinReg.jld2` 1465e409…, equal on both).

**Expected, and why** (Sec. 6.3, Sec. 5.5). Offline, the residual's size grows with the level, almost
all of it in the middle and small scales: the constant Σ's sd is 1.6–2.0 times the residual's in the
lowest-level fifth and 0.64–0.75 times it in the highest. Averaged over the training states the scaled
variance equals LinReg1's (mean s² e^logvar = 1.000–1.005 × diag Σ), so the closure moves noise from
low to high states without changing its total. Expected: little change at short leads, where the
states stay near the IC's; less noise in low states, hence fewer deep downward excursions and fewer
guard firings; narrower ensembles at 0.5–3 TU, where LinReg1's are too wide. A one-step gain is
necessary, not sufficient (memory #75: a learned head gained 0.6 nats/step offline and lost online).
**Expected verdict vs LinReg1:** S unlikely (same mean); C possible through the long-lead cells.
⚠️ Written after the training hindcast, so this expectation is informed by it.

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1–7 apply unchanged)

Hindcast K = 90 ICs, M = 10, 100 + 1200 steps; window CRPS_0.5 (steps 1–200) and CRPS_all (1–1200);
δ₀.₅ (1–200); r, the calibrated count and the rank histograms per lead (36 cells); lead-wise
divergence rule; long runs 5 × 100 TU.

- **G** KS guard against **LinReg7**: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg7's.
- **S** CRPS_0.5 paired difference against LinReg1: 90 % CI upper end < 0.
- **C** calibrated cells (0.8 ≤ r ≤ 1.25, 36 cells) against LinReg1: paired CI of the count
  difference > 0, AND the CRPS_0.5 CI upper end < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

The matched partner is LinReg1 itself, so the mechanism comparison and clauses S and C coincide; the
KS guard is also reported against LinReg1 (the partner replaces both baselines, amendment 6).

Reported for every closure, no clause: CRPS_all, δ₀.₅, the calibrated count's reference, the
rank-histogram figure (slopes), mean bias by lead, long-run bias and spread–skill (per run),
divergences, gate firings (hindcast: % of steps, members, ICs; long runs: % of steps). **Proposed for
step 2 (Rik to confirm):** how often the clip binds, below and above, in the hindcast and the long runs.

Tool: `julia --project=analysis analysis/score_d6.jl --compare <D6_LinReg1_pl> <D6_LinReg1>`.

## Settings and where they come from

- No offset and no noise scale fitted against coupled runs.
- Fixed by rule (Sec. 6.3): the state is the predictor q*^n (chosen offline: the same held-out gain
  as q^{n−1}, 0.483 vs 0.481 nats/step; q* is what the guard reads); the clip is the training range
  (no tuning); q_ref,i the geometric mean of q*_i; β_i and the variance at q_ref,i by maximum
  likelihood with the mean held fixed; Σ_ε the zero-mean second moment of the residual divided by s,
  so its diagonal is the fitted variance. The text says "covariance": the residual's mean on the fit
  rows is 3e-4–1.3e-3 sd, so the two differ by ~1e-6 relative.
- ⚠️ **Disclosure: the fit rows are not LinReg1's training rows.** The builder fits on steps
  400–4000 (`lrs_scale_variant.jl`, `guard(steps_of(1, 10))`, 3601 rows); LinReg1 was trained on
  steps 405–3999 (`5_train_LinReg.jl`, `train_range = (400, 4000)` with `create_history` at h = 5;
  LinReg1's Σ equals the residual covariance on exactly these rows, to 2.6e-15). The six extra rows
  (400–404, 4000) include a 4.7σ residual at step 401 in Z/E[7,15]. Effect: |Δβ| ≤ 0.009 (Z[7,15]
  1.849 against 1.841), |Δlogvar| ≤ 0.004, and the upper clip end of Z[0,6] is set by one of these
  rows. At β = 0 the builder would give a Σ_ε within 0.95 % of LinReg1's Σ in variance (Z[7,15]),
  so "with β = 0 the closure is LinReg1" holds for the model form (V83 sets Σ_ε = Σ), not literally
  for this artefact. **Rik, 2026-10-15: keep `LinReg1_pl` as built and disclose; no refit.** Numbers
  from the implementation review of 2026-10-15 (the row ranges re-read in both scripts).
  The round-1 AR fits read the same window (`lrs_ar_variant.jl`, `guard(steps_of(1, 10))`); their
  effect there is not measured.

## Post hoc

- **2026-10-15, training hindcast (amendment 7), before this card was opened.** Job 28110732
  (`run_d6_train.sh LinReg1_pl`, gpu_a100, 18 min, ≈ 38 SBU), `out/D6train_LinReg1_pl` (50 member
  files, Snellius and laptop). `analysis/train_hindcast.jl LinReg1 LinReg1_pl LinReg7` (LinReg1
  reproduces results_LSTMS.md §25 exactly):

  | | LinReg1 | LinReg1_pl |
  |---|---|---|
  | diverged members | 0 | 0 |
  | gate firings | 246 of 60 000 steps, 2 members | 0 |
  | spread–skill ≤ 0.5 TU | .61 .69 .73 .73 .78 .78 | .63 .73 .74 .74 .79 .78 |
  | spread–skill 0.5–3 TU | 1.30 1.30 1.24 1.24 1.22 1.22 | 1.12 1.16 1.08 1.08 1.09 1.08 |
  | extreme ranks ≤ 0.5 TU (0.18 calibrated) | .35 .37 .22 .22 .19 .19 | .34 .36 .22 .21 .22 .23 |
  | ensemble-mean bias, 0.5 / 2.5 TU | −6.1 / +2.0 % | −6.3 / +6.2 % |

  **Defect rule: no defect** (ratio ≤ 0.5 TU to LinReg1's 1.03 1.06 1.01 1.01 1.01 1.01; first by
  hand, as 1.00 for E[16,32] from the rounded ratios; corrected 2026-10-16 when
  `train_hindcast.jl LinReg1 LinReg1_pl` got the partner entry). Spread and ensemble-mean error
  separately (LinReg1_pl / LinReg1, reprinted by the same run): ≤ 0.5 TU spread 1.00–1.04, error 0.98–0.99; 1–3 TU spread
  0.85–0.92, error 1.01–1.06. Member-steps at 0.5–3 TU with the level below its training range:
  2.0–3.4 % → 0.6–1.0 % (truth 0–1.9 %). The 2.5 TU bias shift comes mostly from the IC at 2.50 TU
  (+15–20 points), where two LinReg1 members dive towards zero (fans:
  `analysis/figures/train/train_fans_LinReg1_pl.pdf`); 5 ICs, so not a finding.
- **2026-10-15, full hindcast and long runs submitted (Rik: "submit … from Snellius as it is (697e074f)").**
  Code: the Snellius checkout, `conditional-density_time-series` @ `697e074f`, not this card's branch
  `step2-powerlaw-scale`: against `697e074f` that branch changes only comments in `src/`, the two
  builders (`lrs_ar_variant.jl`, `lrs_scale_variant.jl`, laptop-only), tests, analysis and this card,
  nothing the runs execute. Artefact: `TO_LRS/LinReg1_pl/LinReg.jld2`, sha256 1465e409… (equal on
  laptop and Snellius). Hindcast: job **28112414** (`run_d6_packed.sh`, gpu_a100, 10 tasks × 9 ICs,
  D6_CLOSURE = lrs, M = 10), output `out/D6_LinReg1_pl`. Long runs: job **28112415**
  (`run_online_array.sh LinReg1_pl on`, gpu_h100, replicas 1–5, 100 TU), output
  `out/TO_LRS/LinReg1_pl/data_online_tsim100.0_replica{1..5}.jld2`.
  Replicas 4 and 5 of 28112415 were cancelled by Snellius at dispatch, 3 s in (`AdminComment`
  `reason=budget`: the dispatch check reserves each task's full time limit, 3 h on gpu_h100 ≈ 576 SBU,
  while both packed hindcasts were reserved too); rerun unchanged as job **28112831** (`-t 00:30:00`,
  same script and replica seeds). All 900 member files and the 5 replicas are byte-identical on
  Snellius and the laptop (per-file sha256; member-list digest 4f26d1b4aea1, replica-set digest
  eebb4d3183b7).

- **2026-10-16, scored (GATE 4 passed: 90 ICs × 10 members, every `warm_identical = true`, 0
  diverged; 5 of 5 replicas with 40 001 columns, finite).** `score_d6.jl --compare D6_LinReg1_pl
  D6_LinReg1` (and against `D6_LinReg7`), lead-wise, 90 ICs; `longrun_stats.jl`; gate census as
  `gate_census.jl` (identically-zero dQ columns, from column 101 in the long runs). Every number
  below is reprinted by the repo commands in results_LSTMS.md §28, "Reproduce" (2026-10-16).

  | | LinReg1 | LinReg1_pl | − LinReg1 (90 % CI) | − LinReg7 (90 % CI) |
  |---|---|---|---|---|
  | CRPS_0.5 | 0.2208 | 0.2202 | −0.3 % [−1.0, +0.5] | −16.3 % [−22.9, −10.4] |
  | CRPS_all | 0.3903 | 0.3921 | +0.5 % [−0.7, +1.9] | −3.9 % [−7.1, −0.7] |
  | δ₀.₅ | 0.4505 | 0.4497 | −0.2 % [−0.6, +0.2] | −6.5 % [−12.4, −0.7] |
  | calibrated cells / 36 | 28 | 27 (ref. 35.7) | −1 [−2, +6] | +22 [+14, +29] |
  | bias, 0.5 / 2.5 TU | −2.9 / −5.9 % | −2.8 / −5.0 % | | |
  | diverged members | 3 | 0 | | |
  | guard, hindcast | 0.42 %, 35 members, 27 ICs | 0.20 % (2146 of 1 080 000), 16 members, 15 ICs | | |
  | long runs: b pooled [single] | −7.3 % [−9.5, −4.2] | −6.1 % [−7.4, −3.2] | | |
  | long runs: D pooled [single] | 0.84 [0.59, 1.01] | 0.79 [0.57, 0.95] | | |
  | long-run ratio | 1.53 | 1.36 | | |
  | stable runs | 5 | 5 | | |
  | guard, long runs | 1.13 % | 0.41 % (83 / 0 / 488 / 0 / 243 of 39 900) | | |

  Per lead, r is a few % lower than LinReg1's in almost every cell: leads 50–100 0.77–0.91 (LinReg1
  0.78–0.93), lead 400 1.06–1.13 (1.14–1.24), lead 1000 1.33–1.43 (1.37–1.48, all six still
  > 1.25). The cell lost is E[16,32] at lead 100 (0.80 → 0.79). The long-run guard first fires at
  58.9 TU in replicas 1 and 3 (shared forcing). The LinReg7 interval rows are the paired intervals
  of that comparison, in % of LinReg7's value.

  **Verdict vs LinReg1:** G pass (single-run D down to 0.57, below LinReg7's highest, 0.91; also
  against LinReg1), S fail (CRPS_0.5 CI upper end +0.5 %), C fail (count CI lower end −2) →
  **does not improve on LinReg1.** As expected for S; C did not come: the narrower ensembles do not
  bring the 2.5 TU cells inside [0.8, 1.25], and they cost a cell at lead 100. The guard fires about
  half as often and no member diverges, which the criterion does not score (stability is
  descriptive). The offline gain (0.48 nats/step held out) does not carry into the coupled scores
  (memory #75).
