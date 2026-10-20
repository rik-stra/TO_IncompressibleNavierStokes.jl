# Card: step 2 on LinReg7 — LinReg7 + power-law scale (S2b) and LinReg7 + AR(2) + power-law scale (S2c) (paper Sec. 6.3), opened 2026-10-16, closed —

WORKFLOW.md Stage 0. Rik, 2026-10-16: "do both", after step 2 on LinReg1 did not improve
(`analysis/cards/step2.md`, results_LSTMS.md §28). Methods text: `closures.tex` §Step 2, the paragraph
"Step 2 also asks whether the answer depends on the base" and Eq. `eq: power law AR`; Table
`tab: closures` rows S2b, S2c. Implementation check: V83 (`test/test_linreg_scale.jl`; the AR + scale
items added 2026-10-16) and the builder's offline sanity (results_LSTMS.md §29).

**Written before any run of S2b or S2c** (training hindcast included).

🔒 **Once any full-hindcast number of S2b or S2c exists, nothing above "Post hoc" changes.**
Late observations go under "Post hoc", dated.

- [ ] Rik has seen this card (date: ____)

## Closures

LinReg path (`D6_CLOSURE=lrs`), the mean of LinReg7 (h = 5, λ = 1 on every QoI, fitted on 1–10 TU),
deployed with the turbulence gate, member seeds `member_seed(k, m)`; each draws the same six standard
normals per step as its partner (V83).

| run | closure | the one change | matched partner | artefact (`TO_LRS/`) |
|---|---|---|---|---|
| S2b | LinReg7 + power-law scale | η = μ_η + s(q*) ⊙ ε, s_i = (clip(q*_i)/q_ref,i)^β_i, ε ~ N(0, Σ_ε), fitted to LinReg7's residual | LinReg7 (`D6_LinReg7`, its long runs; reused) | `LinReg7_pl` |
| S2c | LinReg7 + AR(2) + power-law scale | η = μ_η + s(q*) ⊙ z, z the constrained AR(2) fitted to LinReg7's residual ÷ s (Eq. power law AR) | S1.1 = LinReg7 + AR(2) (`D6_LinReg7_ar2n`, its long runs; reused) | `LinReg7_ar2n_pl` |

Both built on the laptop 2026-10-16 (`lrs_scale_variant.jl LinReg7` and `... LinReg7 --ar2n`).
β = 1.029 / 0.509 / 1.358 / 0.956 / 0.717 / 1.058 (Z/E[0,6], Z/E[7,15], Z/E[16,32]), the same in
both; fit-window gain 0.273 nats/step (held out 0.283, results_LSTMS.md §17). S2c's AR(2) on the
residual ÷ s: φ1 / φ2 = .826/0, .458/−.052, 1.215/−.325, 1.573/−.634, .873/0, .796/0 (S1.1's, refitted on
the laptop: .838/0, .451/−.047, 1.221/−.340, 1.608/−.669, .850/0, .833/−.054). Both are LinReg7's file
with keys added, every LinReg7 key byte-identical (V83).

**Expected, and why.**
- **S2b vs LinReg7.** On LinReg1 the scale narrowed the ensembles by up to a tenth at every lead with
  the mean error unchanged, and its gain was in stability (the guard), which LinReg7 does not need
  (no firing in its hindcast or long runs). LinReg7's ensembles are already too narrow (5 of 36
  cells calibrated; r ≤ 0.5 TU 0.4–0.6). Expected: CRPS_0.5 level with LinReg7, calibrated count no
  higher, possibly lower; long runs as LinReg7's.
- **S2c vs S1.1.** S1.1 is calibrated in all 36 cells, with r 0.91–1.05 up to 0.5 TU and 1.04–1.22
  beyond. A uniform narrowing could cost short-lead cells and help the long-lead ones. Expected:
  CRPS_0.5 level with S1.1; calibrated count within a few cells of 36; long runs as S1.1's.
- **Expected verdict vs LinReg1: no improvement for either.** The scale did not move skill on LinReg1,
  and both bases are behind LinReg1 on CRPS_0.5 (LinReg7 +19 %, S1.1 +5.9 % with C failing on the
  margin). They are run to test whether the fixed-form state dependence helps on a regularized and
  on a coloured base, and as the fixed-form counterpart of step 5c.

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1–7 apply unchanged)

Hindcast K = 90 ICs, M = 10, 100 + 1200 steps; window CRPS_0.5 (steps 1–200) and CRPS_all (1–1200);
δ₀.₅ (1–200); r, the calibrated count and the rank histograms per lead (36 cells); lead-wise
divergence rule; long runs 5 × 100 TU.

- **G** KS guard against **LinReg7**: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg7's.
- **S** CRPS_0.5 paired difference against LinReg1: 90 % CI upper end < 0.
- **C** calibrated cells (0.8 ≤ r ≤ 1.25, 36 cells) against LinReg1: paired CI of the count
  difference > 0, AND the CRPS_0.5 CI upper end < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

Against the matched partner (mechanism, secondary, no verdict; the partner replaces both baselines,
amendment 6): paired CRPS_0.5 and calibrated count; the KS guard against the partner. Reported, no
clause: CRPS_all, δ₀.₅, the calibrated count's reference, the rank-histogram figure, mean bias by lead,
long-run bias and spread–skill (per run), divergences, gate firings, and (as for step 2) the spread
and error split by lead and the share of member steps below the training range
(`train_hindcast.jl --full`).

Training hindcast (amendment 7) first, for both: a closure skips its full hindcast only for a
defect (a diverged member, or a ratio ≤ 0.5 TU more than a factor 2 from its partner's in any QoI).

Tools: `score_d6.jl --compare <D6_closure> <D6_LinReg1>` and `... <D6_partner>`; `train_hindcast.jl`
(partners entered), `longrun_stats.jl`, `gate_census.jl`, `baseline_extras.jl` (closure names).

## Settings and where they come from

- No offset and no noise scale fitted against coupled runs.
- Fixed by rule (Sec. 6.3): the scale as for LinReg1 + power-law scale (state q*^n, clip to the
  training range, q_ref the geometric mean, β and the variance at q_ref by ML with the mean fixed),
  fitted to LinReg7's residual. S2c: the AR(2) of step 1 (Eq. ACF fit, lags 1–20, φ1 ≥ 0, φ2 ≤ 0,
  Σ_ξ = DRD) fitted to the residual ÷ s; the stationary variance of z matched to var(residual ÷ s); the
  warm start sets z to the realized residuals ÷ s. Scaling the coloured residual, not the innovation
  (as step 5b does with a learned scale), keeps both fits unchanged; chosen by Claude, Rik's "do both"
  on the option described to him as (b).
- ⚠️ Disclosure, as for step 2: the fit rows are steps 400–4000; LinReg7 was trained on 405–3999.
- ⚠️ Code: S2c needs the AR + scale sampler, which the Snellius checkout (`697e074f`) does not have;
  S2b runs there as it is.

## Post hoc

- **2026-10-16, S2b training hindcast (amendment 7).** Job 28127114 (`run_d6_train.sh LinReg7_pl`,
  gpu_a100, 18 min, Snellius @ `697e074f`, whose white-scale path is the one tested here), 50 files,
  byte-identical on the laptop. `train_hindcast.jl LinReg7 LinReg7_pl`: no divergence, no guard
  firing (as LinReg7). Spread–skill ≤ 0.5 TU .42 .45 .41 .40 .43 .43 (LinReg7 .40 .44 .39 .39 .42 .42),
  0.5–3 TU .82–.88 (LinReg7 .78–.84). **Defect rule: no defect** (ratio to LinReg7 1.03 in every QoI).
  Spread / error against LinReg7: ≤ 0.5 TU 1.03 / 1.00, 1–3 TU 1.02–1.04 / 0.97–0.98, so slightly
  WIDER here (on LinReg1 the full hindcast narrowed; 5 ICs). Bias 0.5 / 2.5 TU −7.7 / +3.4 %
  (LinReg7 −7.6 / +6.7; 5 ICs). → S2b's full hindcast can run once Rik has seen this card.
- **S2c** has passed the local smoke run (results_LSTMS.md §29) but not its training hindcast: it
  needs the AR + scale sampler on Snellius (commit, push, pull).
