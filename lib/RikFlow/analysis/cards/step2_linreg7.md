# Card: step 2 on LinReg7 — LinReg7 + power-law scale (S2b) and LinReg7 + AR(2) + power-law scale (S2c) — and on LinReg^E + AR(2): LinReg^E + AR(2) + power-law scale (S2d) (paper Sec. 6.3), opened 2026-10-16, S2d added 2026-10-21, closed —

WORKFLOW.md Stage 0. Rik, 2026-10-16: "do both", after step 2 on LinReg1 did not improve
(`analysis/cards/step2.md`, results_LSTMS.md §28). Methods text: `closures.tex` §Step 2, the paragraph
"Step 2 also asks whether the answer depends on the base" and Eq. `eq: power law AR`; Table
`tab: closures` rows S2b, S2c. Implementation check: V83 (`test/test_linreg_scale.jl`; the AR + scale
items added 2026-10-16) and the builder's offline sanity (results_LSTMS.md §29).

**Written before any run of S2b or S2c** (training hindcast included).

**Amended 2026-10-21, before any run of S2d** (Rik: "add LinReg^E + AR(2) + power-law scale at lambda =
1 to the step-2 card (matched partner LinReg^E + AR(2)), switch the card from delta_0.5 to e_0.5"). At
that time the training hindcasts of S2b and S2c existed (Post hoc), no full hindcast of either, and every
full result of round 1 was known: S2d is chosen on them (below, "Why S2d"). The amendment adds S2d and
replaces δ₀.₅ by e₀.₅ for all three; nothing else above "Post hoc" changed.

🔒 **Once any full-hindcast number of S2b, S2c or S2d exists, nothing above "Post hoc" changes.**
Late observations go under "Post hoc", dated.

- [x] Rik has seen this card (date: 2026-10-21, with S2d: "The card is fine; tick it as seen today")

## Closures

LinReg path (`D6_CLOSURE=lrs`), the mean of LinReg7 (h = 5, λ = 1 on every QoI, fitted on 1–10 TU),
deployed with the turbulence gate, member seeds `member_seed(k, m)`; each draws the same six standard
normals per step as its partner (V83).

| run | closure | the one change | matched partner | artefact (`TO_LRS/`) |
|---|---|---|---|---|
| S2b | LinReg7 + power-law scale | η = μ_η + s(q*) ⊙ ε, s_i = (clip(q*_i)/q_ref,i)^β_i, ε ~ N(0, Σ_ε), fitted to LinReg7's residual | LinReg7 (`D6_LinReg7`, its long runs; reused) | `LinReg7_pl` |
| S2c | LinReg7 + AR(2) + power-law scale | η = μ_η + s(q*) ⊙ z, z the constrained AR(2) fitted to LinReg7's residual ÷ s (Eq. power law AR) | S1.1 = LinReg7 + AR(2) (`D6_LinReg7_ar2n`, its long runs; reused) | `LinReg7_ar2n_pl` |
| S2d | LinReg^E + AR(2) + power-law scale, λ = 1 | as S2c, on the mean of LinReg^E at λ = 1 (LinReg1 with the E[0,6] column of LinReg7): s fitted to LinReg^E's residual, z the constrained AR(2) fitted to that residual ÷ s | S1.3b = LinReg^E + AR(2) at λ = 1 (`D6_Splice1_E0x7_ar2n`, its long runs; reused) | `Splice1_E0x7_ar2n_pl` (not built) |

Both built on the laptop 2026-10-16 (`lrs_scale_variant.jl LinReg7` and `... LinReg7 --ar2n`).
β = 1.029 / 0.509 / 1.358 / 0.956 / 0.717 / 1.058 (Z/E[0,6], Z/E[7,15], Z/E[16,32]), the same in
both; fit-window gain 0.273 nats/step (held out 0.283, results_LSTMS.md §17). S2c's AR(2) on the
residual ÷ s: φ1 / φ2 = .826/0, .458/−.052, 1.215/−.325, 1.573/−.634, .873/0, .796/0 (S1.1's, refitted on
the laptop: .838/0, .451/−.047, 1.221/−.340, 1.608/−.669, .850/0, .833/−.054). Both are LinReg7's file
with keys added, every LinReg7 key byte-identical (V83).

**S2d, not built yet.** Builder `lrs_scale_variant.jl Splice1_E0x7 --ar2n` (the white LinReg^E at λ = 1 as
source, as S2c's source is the white LinReg7), unchanged; it has not been run on a spliced source before,
so its own checks are the implementation check: every source key byte-identical, the AR refit
reproduced, `ar_scale_sanity` (draws ÷ s with covariance Σ_ε, sd by quintile of q* following the
data's). What the fit should give, from fits that exist: the scale is fitted per QoI on that QoI's
residual alone, and LinReg^E's residual is LinReg1's in five QoIs and LinReg7's in E[0,6]; so β should be
S2a's in five QoIs and S2b's in E[0,6], 0.514 / **0.509** / 1.849 / 1.823 / 1.276 / 1.365, and the AR(2) of
E[0,6] S2c's, .458/−.052. The AR(2) of the other five, fitted to LinReg1's residual ÷ s, is not known yet;
on LinReg1's raw residual it is ≈ 0 (|φ| ≤ 0.004, Sec. 5). Σ_ε and Σ_ξ are new (the cross-covariances
mix the two residuals). A difference from these values is a finding to report before any run.

**Why S2d (2026-10-21).** In round 1, LinReg^E + AR(2) is the closure that improves on LinReg1: at λ = 1
by clause S (CRPS_0.5 −2.7 % [−5.2, −0.2]), at λ = 10 by C (paper Sec. 7.1.4). S2b and S2c scale bases
that trail LinReg1 (this card: "expected verdict: no improvement for either"); S2d asks whether the scale
adds to the one base that improves. λ = 1, the S pass, one value to keep the number of closures small
(round 1: ten closures, two narrow passes). The scale's gain on LinReg1 was in stability (guard firings
halved, no divergence, long runs D 0.84 → 0.79), and S1.3b still needs the guard (0.06 % of hindcast
steps, 0.05 % of long-run steps).

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
- **S2d vs S1.3b (added 2026-10-21).** In five QoIs the scale is S2a's, which on LinReg1 narrowed the
  ensembles by up to a tenth at every lead with the mean error unchanged; in E[0,6] it is S2c's, which
  was neutral on S1.1 (training hindcast). S1.3b is calibrated in 30 of 36 cells: too narrow at 0.25 TU
  in the middle and small scales (r 0.74–0.81) and too wide at 2.5 TU (r 1.09–1.29). A narrowing of up
  to a tenth would help the second and cost the first. Expected: CRPS_0.5 and e_0.5 level with S1.3b,
  calibrated count within a few cells of 30; fewer guard firings; a long-run spread–skill ratio below
  S1.3b's 1.31, long runs otherwise as S1.3b's.
- **S2d vs LinReg1: no expected change from S1.3b's verdict, with S on the margin either way.** S1.3b
  passes S with its interval ending at −0.2 %; on LinReg1 the scale moved CRPS_0.5 by −0.3 % [−1.0,
  +0.5], so a shift of that size can carry S2d to either side. C is unlikely (S1.3b: count +2 [−1, +9]).

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1–7 apply unchanged)

Hindcast K = 90 ICs, M = 10, 100 + 1200 steps; window CRPS_0.5 (steps 1–200) and CRPS_all (1–1200);
e₀.₅, the centre error (1–200; paper Eq. `centre error`, `score_d6.jl --compare` prints it beside δ₀.₅,
which it replaces: Rik, 2026-10-21); r, the calibrated count and the rank histograms per lead (36
cells); lead-wise divergence rule; long runs 5 × 100 TU. The clauses below apply to S2b, S2c and S2d
alike.

- **G** KS guard against **LinReg7**: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg7's.
- **S** CRPS_0.5 paired difference against LinReg1: 90 % CI upper end < 0.
- **C** calibrated cells (0.8 ≤ r ≤ 1.25, 36 cells) against LinReg1: paired CI of the count
  difference > 0, AND the CRPS_0.5 CI upper end < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

Against the matched partner (mechanism, secondary, no verdict; the partner replaces both baselines,
amendment 6): paired CRPS_0.5 and calibrated count; the KS guard against the partner. Reported, no
clause: CRPS_all, e₀.₅, the calibrated count's reference, the rank-histogram figure, mean bias by lead,
long-run bias and spread–skill (per run), divergences, gate firings, and (as for step 2) the spread
and error split by lead and the share of member steps below the training range
(`train_hindcast.jl --full`). Added 2026-10-21, for all three (post hoc measures of round 1, reported,
no clause): the calibrated cells per QoI, the long-run CRPS_100 and e_100 (`longrun_stats.jl`), and the
correction's autocorrelation in the long runs (as paper Fig. res-s1-dqacf).

Training hindcast (amendment 7) first, for all three: a closure skips its full hindcast only for a
defect (a diverged member, or a ratio ≤ 0.5 TU more than a factor 2 from its partner's in any QoI).
S2d's partner S1.3b has no training hindcast yet (`D6train_Splice1_E0x7_ar2n`): it runs with S2d's.

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
- S2d (2026-10-21): the same rules, fitted to LinReg^E's residual at λ = 1 (`Splice1_E0x7`: LinReg1's
  coefficients with the E[0,6] column of LinReg7, Sec. 6.2); no setting of its own. λ = 1 chosen by Claude
  as the S pass of round 1, Rik's "at lambda = 1".
- ⚠️ Disclosure, as for step 2: the fit rows are steps 400–4000; LinReg7 was trained on 405–3999 (the
  same for LinReg^E, whose columns come from LinReg1 and LinReg7).
- ⚠️ S2d is chosen after round 1's results, on the closure that passed; the paper says so where it
  introduces it (closures.tex §Step 2 and `tab: closures` get S2d once Rik has seen this card).
- ⚠️ Code: S2c needs the AR + scale sampler, which the Snellius checkout (`697e074f`) does not have;
  S2b runs there as it is. (2026-10-20: Snellius at `0495dbb6` has it; S2c's training hindcast ran
  there. S2d uses the same sampler; its artefact is built on the laptop and copied over.)

## Post hoc

- **2026-10-16, S2b training hindcast (amendment 7).** Job 28127114 (`run_d6_train.sh LinReg7_pl`,
  gpu_a100, 18 min, Snellius @ `697e074f`, whose white-scale path is the one tested here), 50 files,
  byte-identical on the laptop. `train_hindcast.jl LinReg7 LinReg7_pl`: no divergence, no guard
  firing (as LinReg7). Spread–skill ≤ 0.5 TU .42 .45 .41 .40 .43 .43 (LinReg7 .40 .44 .39 .39 .42 .42),
  0.5–3 TU .82–.88 (LinReg7 .78–.84). **Defect rule: no defect** (ratio to LinReg7 1.03 in every QoI).
  Spread / error against LinReg7: ≤ 0.5 TU 1.03 / 1.00, 1–3 TU 1.02–1.04 / 0.97–0.98, so slightly
  WIDER here (on LinReg1 the full hindcast narrowed; 5 ICs). Bias 0.5 / 2.5 TU −7.7 / +3.4 %
  (LinReg7 −7.6 / +6.7; 5 ICs). → S2b's full hindcast can run once Rik has seen this card.
- **S2c** passed the local smoke run (results_LSTMS.md §29); its training hindcast followed on
  2026-10-20 after the merge was pushed and pulled on Snellius (next entry).
- **2026-10-20, S2c training hindcast (amendment 7).** Job 28361794 (`run_d6_train.sh
  LinReg7_ar2n_pl`, gpu_a100, 19:31, Snellius @ `0495dbb6`; the log prints the AR(2) φ and β "on the
  AR process" as built), 50 files, byte-identical on the laptop. `train_hindcast.jl LinReg7_ar2n
  LinReg7_ar2n_pl`: no divergence, no guard firing (S1.1: 22 steps, one member). Spread–skill ≤ 0.5
  TU .74 .63 .85 .86 .89 .88 (S1.1 .73 .62 .83 .84 .87 .86), 0.5–3 TU 1.10–1.24 (S1.1 1.12–1.26).
  **Defect rule: no defect** (ratio to S1.1 1.01 1.03 1.02 1.02 1.03 1.03). Spread / error against
  S1.1: ≤ 0.5 TU 1.00–1.02 / 0.99–1.00, 1–3 TU 0.99–1.01 / 1.00–1.02, i.e. neutral. Member steps below
  the training range 1.3–1.7 % (S1.1 1.7–2.3 %). Bias 0.5 / 2.5 TU −8.6 / +4.5 % (S1.1 −8.5 / +4.5;
  5 ICs). Fans like S1.1's; no flat or exploding member (inspection agent, verified by a rerun). →
  S2c's full hindcast can run once Rik has seen this card.
- **2026-10-21, Rik has seen the card** (with S2d). S2b and S2c may run their full hindcasts.
- **2026-10-21, S2d built on the laptop** (`lrs_scale_variant.jl Splice1_E0x7 --ar2n`, not on Snellius
  yet; Rik: "show me its fit (beta, AR, sanity) before anything goes to Snellius"). β = 0.514 / 0.509 /
  1.849 / 1.823 / 1.276 / 1.365, **as predicted above** (S2a's in five QoIs, S2b's in E[0,6]). AR(2) on
  the residual ÷ s: E[0,6] .458/−.052, **as predicted** (S2c's); the other five φ1 = .004 / .044 / .041 /
  .047 / .048, φ2 ≈ 0 (−.006 to 0): dividing by s leaves a lag-one autocorrelation of 0.04–0.05 that
  the raw residual does not have (its fit, S1.3b's: φ1 = 0). σ_ξ / sd(Z ÷ s) 1.000 / 0.899 / 0.999 ×4;
  innovation lag-0 correlations −0.01 to 0.94. Builder checks: every source key identical, the AR refit
  reproduced; `ar_scale_sanity` (teacher-forced on 1–10 TU): warm start equals the data's Z ÷ s to
  3e−15, model z ACF at lags 1/2/5/10/20 on the fitted and the data ACF (E[0,6] .43/.15 against .44/.13),
  sd model/data 0.990–1.008, ξ sd / σ_ξ 0.995–1.000. ⚠️ The card above named the white path's checks
  (draws ÷ s against Σ_ε, sd by quintile of q*); the AR path prints these instead. The quintile check
  was run for the same β on S2a and S2b. Log: scratch build_s2d.log.
- **2026-10-21, submitted on Snellius** (Rik: "copy S2d to Snellius, run the training hindcasts of S2d
  and of its partner ..., and start the full hindcasts and long runs of S2b and S2c (gpu_a100)").
  Snellius at `0495dbb6`, clean, = the laptop's HEAD (the laptop's uncommitted changes are analysis
  only; no push or pull needed). `TO_LRS/Splice1_E0x7_ar2n_pl` copied by scp, sha256 equal
  (LinReg.jld2 `5319e309…a764`). S1.3b's training hindcast already exists there
  (`D6train_Splice1_E0x7_ar2n`, 50 files; it was on the laptop too, which Claude missed, and the pull
  below copied the Snellius files over it, the same job's output), so only S2d's runs. Jobs, all
  gpu_a100: S2d training hindcast 28422182; S2b full hindcast 28422183 (`D6_LinReg7_pl`), S2c
  28422184 (`D6_LinReg7_ar2n_pl`); long runs, 5 × 100 TU, gate on: S2b 28422185, S2c 28422186.
  Snellius home 82.4 % before the runs.
- **2026-10-21, S2d training hindcast (amendment 7).** Job 28422182 (18:39, gpu_a100, Snellius @
  `0495dbb6`), 50 files, pulled with S1.3b's. `train_hindcast.jl Splice1_E0x7_ar2n Splice1_E0x7_ar2n_pl`
  (S2d's partner added to its table): no divergence; guard 1 of 60 000 forecast steps (S1.3b: 44, two
  members). Spread–skill ≤ 0.5 TU .62 .66 .73 .73 .76 .75 (S1.3b .61 .64 .72 .73 .75 .75), 0.5–3 TU
  1.09–1.12 (S1.3b 1.15–1.20). **Defect rule: no defect** (ratio to S1.3b 1.02 1.03 1.01 1.01 1.01 1.01).
  Spread / error against S1.3b: ≤ 0.5 TU 1.00–1.03 / 0.99–1.00, 0.5–1 TU 0.99–1.00 / 0.98–1.00, 1–3 TU
  0.92–0.94 / 0.99–1.00: narrower beyond 1 TU at the same error, as expected. Member steps below the
  training range 1.3–2.2 % (S1.3b 2.0–3.1 %). Bias 0.5 / 2.5 TU −7.2 / +3.3 % (S1.3b −7.1 / +2.8; 5 ICs).
  → **Full runs submitted** (Rik: "if there is no defect submit S2d's full hindcast and long runs"):
  hindcast 28423750 (`D6_Splice1_E0x7_ar2n_pl`), long runs 28423751 (5 × 100 TU, gate on), gpu_a100.
- **2026-10-21, S2b and S2c long runs** (28422185, 28422186) completed, 5 replicas each on Snellius (not
  pulled yet); their full hindcasts were still running (514 / 422 of 900 files at the check).
- **2026-10-22, S2b, S2c, S2d scored** (all jobs completed; pulled, sizes equal; logs scratch
  scores2/; paper Table `tab: res s2 all`). Against LinReg1: S2b CRPS_0.5 +19.2 % [+12.3, +26.7], count
  4 (−24) → no; S2c +5.3 % [−0.5, +11.5], count 36 (+8 [+2, +13]) → no (C: upper end 11.5 % > 10 %);
  **S2d −3.1 % [−5.7, −0.5] → S passes: improves**, count 31 (+3 [−1, +10]). G passes for all (single-run
  D 0.52–0.80 / 0.56–1.17 / 0.45–0.78 against LinReg7's maximum 0.91). Against the partner, no score
  resolved: CRPS_0.5 +0.1 / −0.6 / −0.4 %, e_0.5 0.0 / −0.2 / −0.1 %, count −1 / 0 / +1. Guard firings
  down: hindcast 0 / 84 / 127 steps (partners 0 / 350 / 605), long runs 0 / 5 / 2 (0 / 115 / 101). Long
  runs: S2b D 0.65 → 0.60, CRPS_100 0.386 → 0.367; S2c D 0.84 → 0.88; S2d D 0.62 → 0.61, ratio 1.31 →
  1.23, CRPS_100 0.360 → 0.374 (no interval yet). As expected: the scale moves no hindcast score
  resolvably; S2d's S pass is its base's (upper end −0.2 % → −0.5 %).

### Amendment 2026-10-22: S2e = LinReg^E + power-law scale (white), λ = 1

🔒 **Written after the full results of S2b, S2c and S2d, before any run of S2e.** The lock above (no
change above "Post hoc" once a full-hindcast number exists) applies, so S2e is specified here, in
full, as a dated amendment. Rik, 2026-10-22: "should we also run linregE with scale dependent white
noise? Since this gave the best KS distance before", then "add S2e ... with these caveats; show me the
card before building or running".

- [x] Rik has seen this amendment (date: 2026-10-22: "The amendment is fine; tick it as seen today")

| run | closure | the one change | matched partner | artefact (`TO_LRS/`) |
|---|---|---|---|---|
| S2e | LinReg^E + power-law scale, λ = 1 | η = μ_η + s(q*) ⊙ ε, ε ~ N(0, Σ_ε), s fitted to LinReg^E's residual at λ = 1, white (Eq. power law) | S1.2b = LinReg^E at λ = 1 (`D6_Splice1_E0x7`, `D6train_Splice1_E0x7`, its long runs; all on the laptop, reused) | `Splice1_E0x7_pl` (not built) |

**Build.** `lrs_scale_variant.jl Splice1_E0x7` (the white path, as S2a's and S2b's; its own checks:
every source key identical, draws ÷ s with covariance Σ_ε, sd by quintile of q* following the data's).
Predicted fit: the scale keys exactly S2d's (`Splice1_E0x7_ar2n_pl` was built "with the scale keys
exactly as `<src>_pl`, same fit"), so β = 0.514 / 0.509 / 1.849 / 1.823 / 1.276 / 1.365 and the same
Σ_ε; any difference is a finding to report before any run.

**Why S2e.** (1) On white bases the scale improved the long runs: LinReg1 D 0.84 → 0.79, ratio 1.53 →
1.36 (S2a); LinReg7 D 0.65 → 0.60, CRPS_100 0.386 → 0.367, e_100 0.691 → 0.664 (S2b). On the coloured
bases it did not (S2c D 0.84 → 0.88; S2d 0.62 → 0.61). S1.2b has the smallest summed KS distance of all
closures so far, 0.53 (single runs 0.33–0.73). (2) It completes the 2 × 2 on LinReg^E at λ = 1 — white
or AR(2), each with or without the scale (S1.2b, S1.3b, S2e, S2d) — as S2b and S2c do on LinReg7, so the
scale's effect can be told from the colour's. (3) Cheap: about 800 SBU for the full hindcast and long
runs, 40 for the training hindcast.

**Expected, and why.**
- **S2e vs S1.2b.** Hindcast: CRPS_0.5 and e_0.5 level (the scale moved CRPS_0.5 by −0.6 to +0.1 % on
  every base so far, e_0.5 by at most 0.2 %); calibrated count within a few cells of S1.2b's 31. E[0,6]
  stays too narrow (S1.2b: 1 of 6 cells; r 0.55–0.83): the scale narrows beyond 1 TU, it does not widen.
  Fewer guard firings (S1.2b: 0.02 % of hindcast steps, 0.09 % of long-run steps). Long runs: summed KS
  at or below S1.2b's 0.53, CRPS_100 at or below its 0.373, spread–skill ratio from 1.12 towards 1.
- **S2e vs LinReg1: no improvement expected.** S1.2b's CRPS_0.5 is −1.4 % [−3.9, +1.1] against LinReg1
  and its count +3 [−5, +9]; shifts of the size the scale has produced cannot be expected to carry the
  upper end below 0 (S) or the count's lower end above 0 (C).

**Caveats (Rik: "with these caveats").**
- ⚠️ **It does not change the criterion's verdict.** S2e is run for its long runs; the KS distance enters
  the criterion only as the guard G, so a better D is reported, not scored.
- ⚠️ **"Best KS" is not a resolved ranking.** S1.2b's 0.53, S1.2c's 0.54 and S2b's 0.60 have single-run
  ranges that overlap widely (0.33–0.73, 0.48–0.72, 0.52–0.80). A smaller D for S2e is a difference only
  if its runs separate from S1.2b's (the guard's non-overlap rule, applied as a comparison, no clause).
- ⚠️ **Post hoc, a second time.** S2e is chosen after the results of steps 1 and 2, partly on a long-run
  result; the paper says so where it introduces it, as for S2d (closures.tex §Step 2 and `tab:
  closures` get S2e once Rik has seen this amendment). Multiplicity: step 2 then has five closures, of
  which one passes (S2d, on its base).

**Clauses.** As for S2b–S2d (above, unchanged): G against LinReg7, S and C against LinReg1, the
partner's comparison with no verdict; the same reported measures (e_0.5, CRPS_all, the post hoc
measures of round 1). Training hindcast first (amendment 7): defect rule against `D6train_Splice1_E0x7`.

**Runs.** Build on the laptop and show Rik the fit; copy to Snellius (scp, sha256 checked); training
hindcast (`run_d6_train.sh Splice1_E0x7_pl`); if no defect, full hindcast (`run_d6_packed.sh`,
`D6_Splice1_E0x7_pl`) and long runs (`run_online_array.sh Splice1_E0x7_pl on`, gpu_a100, 45 min).
Every ssh and scp asks Rik.

- **2026-10-22, S2e built on the laptop** (`lrs_scale_variant.jl Splice1_E0x7`; Rik: "Build S2e on the
  laptop and show me its fit before anything goes to Snellius"). **As predicted:** `scale_beta`,
  `scale_qref`, `scale_qclip`, `scale_sigma_eps`, `scale_logvar` all `isequal` to S2d's; β = 0.514 /
  0.509 / 1.849 / 1.823 / 1.276 / 1.365; no AR keys; the source keys identical (builder check). Fit-window
  NLL gain over the constant variance 0.4875 nats/step summed (0.009 / 0.009 in the large scales,
  0.11–0.12 in the others). Sanity (white path): draws ÷ s against Σ_ε, max |Δcov| / max diag 0.003
  (3601 draws); sd by quintile of q*, model / data: Z[7,15] 0.52–1.33 / 0.50–1.40, Z[16,32] 0.49–1.33 /
  0.55–1.38, large scales 0.84–1.10 / 0.82–1.12 (the law slightly too steep at the lowest quintile of the
  small scales, as for S2a, whose fit it is there). LinReg.jld2 sha256 `f8a92317…2996`. Log: scratch
  build_s2e.log. Not on Snellius yet.
- ⚠️ **2026-10-22, S2e's training hindcast waived (Rik: "copy to snellius and submit full hindcast and
  longruns. (we can skip training)").** A deviation from amendment 7 and from "Runs" above, decided
  by Rik: the full hindcast runs without the defect check. Its basis: S2e's fit is S2d's scale on S2d's
  white base, and S2d's training hindcast against S1.3b had no defect (ratio ≤ 0.5 TU 1.01–1.03).
- **2026-10-22, S2e submitted.** `TO_LRS/Splice1_E0x7_pl` copied by scp, sha256 equal; Snellius @
  `0495dbb6`. Full hindcast 28464658 (`D6_Splice1_E0x7_pl`), long runs 28464659 (5 × 100 TU, gate on),
  gpu_a100.
