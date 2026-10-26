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
