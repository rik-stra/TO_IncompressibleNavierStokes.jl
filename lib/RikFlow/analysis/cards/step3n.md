# Card: step 3c — means trained with their own noise (paper Sec. 7.4 `sec:lever-net`), opened 2026-10-26

WORKFLOW.md Stage 0. Methods text: `closures.tex` §Step 3, the paragraph on training with the closure's own
noise and Eq. `eq: window noise` (**draft 2026-10-26 for Rik**, paper branch `step3-cnn-closure`); offline
evidence: `results_LSTMS.md` §34; the noise construction: `claude_memory.md` standing decision (2026-10-26,
Rik: "lock this way of adding noise"), `exp_square_HIT/tools/step3_innoise.jl`.

⚠️ **Motivated post hoc.** Rik proposed training with the noise after a1's held-out hindcast fans (2026-10-26),
i.e. after evaluation-window results of step 3a existed. The offline probe run before this card (§34)
**contradicted the motivating hypothesis**: a1 does not pass more of its noise into its mean than LinReg1
does, so the noise feedback does not explain a1 < LinReg1. Both closures are run anyway (Rik: "Build both"):
they test the noise feedback itself, which is large in both untrained means.

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**

- [ ] Rik has seen this card (date: ; quote: )

## Closures

| run | closure | the one change | matched partner | artefact (sha256) |
|---|---|---|---|---|
| n1 | LinReg1 trained with its noise (`LinReg1_nt`) | the mean fitted on histories carrying the closure's own window noise, its Σ self-consistent | LinReg1 (`D6_LinReg1`, `D6train_LinReg1`, its long runs; reused) | `TO_LRS/LinReg1_nt/LinReg.jld2` 3bf7358d… |
| n2 | CNN (RMSE) trained with its noise + MVG (`CNN_rmse_nt`) | the same for a1's network | a1 (`CNN_rmse_mvg`, its runs of 2026-10-23/26) | `TO_LSTM/step3/closures/CNN_rmse_nt/StochLSTM_seed1.jld2` b97099e5… |

The noise (both): per training window, η^{n−1} … η^{n−h} ~ N(0, Σ) i.i.d.; q^{n−k} → q^{n−k} − (η^{n−k+1} + … +
η^{n−1}), q^{n−k*} → q^{n−k*} − (η^{n−k} + … + η^{n−1}); q^{n*} and the target as recorded. Every correction in
the window carries one η; the sum is anchored at the prediction step.

- **n1** (`exp_square_HIT/tools/lrs_noise_variant.jl`, laptop, 2026-10-26): c' = (X'X + N S_dx(Σ)) \ X'Y on
  LinReg1's own rows (steps 405–3999; zero noise reproduces LinReg1's c to 4.9e-11), Σ = cov(y − c x) +
  c S_dx(Σ) c' iterated from LinReg1's Σ to a fixed point (52 iterations, 1e-12); μ_η the clean residual's
  mean, Σ the MLE, as LinReg1's `fit(MvNormal, ...)`. Only `c` and `stoch_distr` differ from LinReg1's file.
  Runs on the LinReg path (`D6_CLOSURE=lrs`, `run_online.sh lrsdeploy`), as LinReg1.
- **n2** (`cnn_mean_fit.jl`, RIKFLOW_CNN_INNOISE=fixed, laptop CPU, 2026-10-26): a1's network settings (RMSE,
  weight decay 0.3, seed 1, the two trainings), fresh draws every update; the validation loss (early stopping)
  on noisy histories, one fixed draw. Σ by an outer loop of five trainings, fixed in advance by the script:
  round 1 from a1's MVG, round k from round k−1's `S_deploy` (the refitted network's residual on 1–10 TU under
  its injected noise, about zero, 4 draws). `S_deploy` sd / injected sd: round 1 1.14 1.14 1.73 1.77 1.12
  1.24 … round 5 **1.00 1.00 1.01 1.03 0.99 0.99**. Deployed: round 5 (`step3/cnn_nt_rmse_wd0.3_s1_r5`), its
  `S_deploy` (`cnn_closure_export.jl … mvgnt`). Network path, as a1.

**Expected, and why** (§34; columns Z,E[0,6] Z,E[7,15] Z,E[16,32]).
- **Offline.** Gain = rms response of the mean to its own window noise / sd(η). LinReg1 0.89 0.92 2.43 2.35
  1.00 1.27; a1 0.73 0.73 2.45 2.11 0.79 1.16 (10–100 TU). n1 (in-sample) 0.46 0.39 0.66 0.76 0.35 0.46, n2
  0.42 0.39 0.65 0.77 0.30 0.44: both trained means pass on half to a third of their noise. The price: the
  clean one-step error, n1 1.09 1.08 1.95 2.10 1.08 1.22 × LinReg1's (in-sample), n2 1.15 1.10 2.16 2.53 1.46
  1.46 × (10–100 TU); and a larger Σ, sd 1.1–3.3 × LinReg1's, largest in the middle band. This is LinReg7's
  profile (gain 0.33–0.59, clean 1.2–11.5 ×), which is worse than LinReg1 in the hindcast (CRPS_0.5 +19 %,
  5 calibrated cells) and better in the long runs (summed KS 0.65 against 0.84), at a far smaller clean cost.
- **n1 vs LinReg1:** CRPS_0.5 worse, by less than LinReg7 (a few to ~15 %): the mean is worse in the middle
  band. Wider ensembles at every lead: better calibrated up to 0.5 TU (LinReg1's r 0.82–0.95), worse beyond
  (LinReg1 already over-dispersed there, r 1.27–1.40), so no clear gain in the calibrated count. Long runs:
  summed KS between LinReg7's and LinReg1's. Fewer gate firings.
- **n2 vs a1:** wider ensembles (a1 is under-dispersed up to 0.5 TU, r 0.70–0.87) and so more calibrated cells
  there; CRPS_0.5 not better than a1's (its mean is worse, and a1's early departure was not noise feedback).
- **Expected verdict vs LinReg1, both: does not improve.**

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1–7 apply unchanged)

As `step3a.md`: hindcast K = 90, M = 10, 100 + 1200 steps; window CRPS_0.5 (1–200) and CRPS_all (1–1200); δ₀.₅;
r, calibrated count and rank histograms per lead (36 cells); lead-wise divergence rule; long runs 5 × 100 TU.

- **G** KS guard against **LinReg7**: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg7's.
- **S** CRPS_0.5 paired difference against LinReg1: 90 % CI upper end < 0.
- **C** calibrated cells against LinReg1: paired CI of the count difference > 0, AND the CRPS_0.5 CI upper end
  < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

Against the matched partner (the mechanism, secondary; same rules): n2 vs a1 (n1's partner is LinReg1, above).

Reported for every closure, no clause: as `step3a.md` (CRPS_all, δ₀.₅, e₀.₅, rank histograms, bias by lead,
long-run bias, spread–skill, CRPS_100, e_100, divergences, gate firings, member steps below the training
range), plus the offline gain above. Tools: `score_d6.jl --compare`, `train_hindcast.jl` (PARTNER entries
added), `longrun_stats.jl` (n2's long runs by the relative path `../TO_LSTM/step3/closures/CNN_rmse_nt`).

## Settings and where they come from

- The noise construction: Rik (2026-10-26), locked; the anchor at the prediction step chosen over the window's
  start (which would reward reliance on the oldest entry).
- Σ self-consistent (Rik: "self-consistent Σ", "fixed Σ with the outer loop"); about zero for n2 as for a1;
  for n1 LinReg1's own convention (mean + MLE covariance), so that n1 differs from LinReg1 in the training only.
- n2's network settings are a1's, **not re-selected**: the weight decay 0.3 was chosen on clean validation
  (Sec. 6); the noise also regularizes, so another weight decay might win on noisy validation. One seed (1);
  no seed spread for the noise-trained network.
- `lstm_knobs` = input map only; no offset or noise scale fitted against coupled runs.
- ⚠️ Disclosures:
  - Post hoc motivation, and the probe's contradiction of it (above).
  - Not modelled in the noise: the solver's damping over a step (< 1 %), and the network's own errors feeding
    back (only the draws are injected).
  - n1's offline numbers are in-sample (its builder reads ≤ 10 TU); its 10–100 TU one-step scores were not
    computed.
  - n2 fails the offline check of Sec. 7.8, by more than a1 (clean validation likelihood −3.80 against least
    squares' −6.95; a1 −6.08), as expected of a mean trained for noisy histories.
  - The outer loop's five rounds were run in one script; the number was set before any round was seen.

## Prerun checks (WORKFLOW.md Stages 0–3)

- **GATE 0:** methods text — draft for Rik (paper branch `step3-cnn-closure`); this card — to be seen by Rik.
- **GATE 1 (implementation matches the text):**

  | text | checked by |
  |---|---|
  | Eq. `eq: window noise` (the pairing) | `step3_innoise.jl` self-check through the input map: every dQ and Dq* entry carries its own η once, q^{n−k*} the running sum, q^{n*} none (rel. 2e-16); `lrs_noise_variant.jl`'s `window_map` the same formulas, the regressor layout asserted on the record |
  | n1: the closed form; zero noise = LinReg1 | 4.9e-11 on LinReg1's rows 405–3999; LinReg1's stoch_distr = the MLE of its residual there (2e-15) |
  | n1: Σ the fixed point | 52 iterations to 1e-12; `--report` reproduces the file exactly (c and Σ, 0.0) |
  | n1: only the training differs | every other key of the file identical to LinReg1's (asserted); `RikFlow.LinReg` loads it, white |
  | n2: trained with the noise; the outer loop | smoke fits (2026-10-26): run 2's injected Σ == run 1's `S_deploy`; clean mode unchanged (smoke) |
  | n2: deployed = trained | translation check, 300 steps: max 3.4e-5 sd(dQ) |
  | n2 deploys `S_deploy` | `cnn_closure_export.jl … mvgnt`; `mvg` and the other modes refuse a noise-trained run |
- **GATE 2:** fits complete; n1 reproducible (`--report`); n2 seed 1 only (disclosed); offline check as a1's.
- **GATE 3 (smoke), on Snellius, as step 3a:** the training hindcasts (hindcast half; n1 runs the unchanged
  LinReg path, so its training hindcast is its smoke), and for n2 the long-run smoke
  (`smoke_online_lstm.sh step3/closures/CNN_rmse_nt`). n1's long-run path is LinReg1's (`lrsdeploy`), values
  only.
- **Training hindcast (amendment 7):** n1 `run_d6_train.sh LinReg1_nt` (defect rule vs LinReg1); n2
  `run_d6_train_lstm.sh step3/closures/CNN_rmse_nt` (vs a1).

## Production plan (after GATE 3 and Rik's go)

From `exp_square_HIT` on Snellius:
- n1 hindcast `D6_OUT=$PWD/output/D6_LinReg1_nt D6_CLOSURE=lrs D6_MODEL=$PWD/output/TO_LRS/LinReg1_nt/LinReg.jld2
  sbatch batch_scripts/run_d6_packed.sh`; long runs `sbatch -t 00:30:00 batch_scripts/run_online.sh lrsdeploy
  LinReg1_nt <r>`, r = 1…5.
- n2 hindcast `D6_OUT=$PWD/output/D6_CNN_rmse_nt D6_CLOSURE=lstm
  D6_MODEL=$PWD/output/TO_LSTM/step3/closures/CNN_rmse_nt/StochLSTM_seed1.jld2 sbatch batch_scripts/run_d6_packed.sh`;
  long runs `RIKFLOW_M4_MODEL_DIR=$PWD/output/TO_LSTM/step3/closures/CNN_rmse_nt sbatch -t 00:30:00
  batch_scripts/run_online.sh lstm 2 <r>`, r = 1…5.
- Cost ≈ 2 × 775 SBU + training hindcasts ~75 + smoke ~10; storage ≈ 3.2 GB (home ~29 GiB free on 2026-10-23).

## Post hoc

**2026-10-26, training hindcasts submitted** (Rik: "agreed run the training hindcasts", after the list of the two
closure setups; code `2287bdeb`, both closures on Snellius with sha256 equal): n1 `run_d6_train.sh LinReg1_nt`
job 28588678 → `output/D6train_LinReg1_nt`; n2 `run_d6_train_lstm.sh step3/closures/CNN_rmse_nt` job 28588679 →
`output/D6train_CNN_rmse_nt`. Then (Rik: "and smoke long runs") n2's long-run smoke `smoke_online_lstm.sh
step3/closures/CNN_rmse_nt` job 28588688. n1 has no short long-run smoke: `6_online_TO_LRS.jl` hard-codes
tsim = 100; none is run (Rik: "Skip it; LinReg1_nt uses LinReg1's long-run path unchanged").

**2026-10-26, GATE 3 and the training hindcasts** (Rik: "also inspect the training hindcasts for the noise runs";
jobs 28588678 (18 min), 28588679 (17 min), 28588688 (2 min), all COMPLETED; 50 files each and the logs copied to
the laptop; `train_hindcast.jl LinReg1 LinReg1_nt CNN_rmse_mvg CNN_rmse_nt`, scratch `scores3a/train_nt.txt`):
- GATE 3: n2's long-run smoke `PASS` (200 of 200 steps, warm-up bit-identical, finite, gate 0 / 100, the fit dir's
  model; log: `Deploying StochLSTM2` arch cnn, `Emission noise` none). Both training hindcasts: 50 / 50
  `warm_identical`, 0 diverged, `model_name` LinReg1_nt / step3/closures/CNN_rmse_nt, n2's `lstm_noise` white.
- Defect rule: **no defect** (spread–skill ≤ 0.5 TU / partner's: n1 0.74–0.78 vs LinReg1, n2 0.58–0.82 vs a1).
- Gate firings: n1 0 of 60 000 steps (LinReg1 246, 2 members); n2 16 steps, 3 members (a1 261, 2 members).
  Member steps below the training range: n1 0.1–1.5 % (LinReg1 2.0–3.4 %), n2 2.2–2.6 % (a1 2.8–3.0 %).
- ⚠️ **Not as expected: the ensembles are NARROWER, not wider.** Spread / partner's at every lead: n1 0.70–0.88,
  n2 0.67–0.91, although both draw a larger Σ (sd up to 3.2–3.3 × in the middle band). The partner's spread
  comes mostly from its mean answering its own draws (gain up to 2.4); cutting that feedback removes more spread
  than the larger draw adds. So r ≤ 0.5 TU falls: n1 0.45–0.61 (LinReg1 0.61–0.78), n2 0.47–0.74 (a1 0.77–0.91);
  0.5–3 TU n1 1.08–1.20 (LinReg1 1.22–1.30), n2 1.12–1.20 (a1 1.13–1.17).
- Error of the ensemble mean / partner's: n1 0.99–1.05 up to 0.5 TU, **0.83–0.87 at 1–3 TU**; n2 0.99–1.39 up to
  0.5 TU (E[0,6] 1.39), 0.77–0.91 at 1–3 TU. Bias at 2.5 TU: n1 +1.9 % (LinReg1 +2.0 %), n2 −5.5 % (a1 +4.5 %).
- In-sample (the 5 training ICs inside 1–10 TU), no clause; the expectations above are left as written.

**2026-10-26, production submitted** (Rik: "submit the full hindcasts and the five long runs for LinReg1_nt and
CNN_rmse_nt on gpu_a100, as in the step3n card's production plan"; code `2287bdeb`; budget 150 000 SBU, 11 076
used): n1 hindcast 28590706 (array 1-10) → `output/D6_LinReg1_nt`, long runs r = 1…5 28590707–11
(`lrsdeploy`, `-t 00:30:00`) → `TO_LRS/LinReg1_nt`; n2 hindcast 28590712 → `output/D6_CNN_rmse_nt`, long runs
28590713–17 (`lstm 2 <r>`, `-t 00:30:00`) → the closure dir.

**2026-10-26, n1 and n2 scored** (all jobs completed; 900 files per hindcast, counts and bytes equal; the 10 long
runs' sha256 equal; logs beside the results; `score_d6.jl --compare`, `longrun_stats.jl`, `train_hindcast.jl
--full`, scratch `scores3a/*nt*`):

| | CRPS_0.5 vs LinReg1 | calibrated (LinReg1 28) | G: single-run D (LinReg7 ≤ 0.91) | verdict vs LinReg1 |
|---|---|---|---|---|
| n1 LinReg1_nt | +7.0 % [+2.3, +11.9] | 15 (−13 [−19, −3]) | 0.54–0.88: pass | does not improve |
| n2 CNN_rmse_nt | +20.7 % [+12.7, +29.0] | 9 (−19 [−25, −8]) | 1.23–1.53: **FAIL** | does not improve |

Verdicts as expected. Against expectations otherwise:
- **n1:** CRPS_0.5 +7 % (expected "a few to ~15 %, less than LinReg7's +19 %"), as expected; CRPS_all +2.8 % [+0.4,
  +5.6]; δ₀.₅ +1.2 %, e₀.₅ +2.9 %, not resolved. ⚠️ Not as expected: **narrower**, not wider (spread 0.71–0.78 ×
  LinReg1's at every lead), so the calibration is reversed from the card: worse up to 0.5 TU (r 0.61–0.74 against
  0.82–0.95) and much better beyond (r 1.01–1.08 against 1.27–1.40); the count falls because the short-lead cells
  are lost. Divergences 0 (LinReg1 3); gate firings 0.03 % of steps, 4 members (0.42 %, 35); member steps below
  the training range 0.9–1.5 % (2.7–3.4 %); bias at 2.5 TU −4.0 % (−5.9 %). **Long runs better than LinReg1 on
  every measure** (as expected: summed KS between LinReg7's and LinReg1's): D 0.76 (LinReg1 0.84, LinReg7 0.65),
  bias −5.1 % (−7.3 %, −4.9 %), ratio 1.25 (1.53, 0.99), CRPS_100 **0.380** (0.387, 0.386, the lowest of all
  closures so far), e_100 0.679 (0.649, 0.691); 5 of 5 stable.
- **n2 vs a1 (partner):** CRPS_0.5 −4.0 % [−11.9, +3.4], not resolved; δ₀.₅ −6.9 % [−13.3, −0.4] (lower error up to
  0.5 TU: 0.90–1.01 × a1's); count 9 vs 11 (−2 [−12, +4]); G against the partner (single-run D 1.23–1.53 against
  a1's 0.73–1.26) passes under the paper's rule, so no on S and C. ⚠️ Not as expected: **narrower** (spread
  0.78–0.86 × a1's), not wider; worse at longer leads: CRPS_all +20.0 % [+14.6, +26.5], error 1.15–1.20 × a1's at
  0.5–1 TU, bias at 2.5 TU −10.0 % (−8.3 %); long runs worse (D 1.34, bias −10.8 %, CRPS_100 0.511). Gate firings
  0.25 % (0.40 %).
- Reading: the noise training removes the feedback in both, which makes the ensembles narrower (the partner's
  spread came mostly from it) and the long-lead and long-run statistics better for the linear mean. The CNN's mean
  keeps its out-of-range drift, which the noise does not touch.
