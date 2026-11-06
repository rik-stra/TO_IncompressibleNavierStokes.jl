# Card: step 2b, more — the regularized baseline trained with its noise, and both noise-trained means + AR(2), opened 2026-11-04

Paper: Results step 2b (`sec:res-lin-noise`, scaffold "Coupled runs planned"); methods: Closures step 2b
(`sec:lever-noise`). Rik, 2026-11-04 (on the paper's step 2b): "[do we have coupled runs of the regularized baseline
trained with its noise?]" -> "I want to run the noise trained regularized model coupled. Also run the both noise
trained linear models with AR(2) coupled"; on the AR fit: "Yes, same way" (as the MVG of the noise-trained means: to
the residual given the noisy history).

⚠️ **Motivated post hoc.** All three are proposed after the coupled scores of LinReg1, LinReg7 and LinReg1_nt exist
(paper Table `tab: res s2b`), and after the offline step-4 table (`tab: skip-offline`) showed LinReg7 trained with its
noise to be close to LinReg7 (recorded 2.895 against 2.873, response 0.43/0.44 against 0.58/0.57 in the middle band).
The AR on the noise-trained residual was a planned closure since 2026-10-26 (story.md C5: "an AR fitted to the
residual of the noise-trained mean along its own draws, to restore the short-lead spread"; no card until this one).

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**

- [x] Rik asked for the runs (2026-11-04, quotes above); the card itself not yet seen.

## Closures

| run | closure | the one change | matched partner | artefact (sha256 of `LinReg.jld2`) |
|---|---|---|---|---|
| m1 | LinReg7 trained with its noise (`LinReg7_nt`) | the regularized baseline's mean fitted on histories carrying its own window noise, Σ self-consistent | LinReg7 (`D6_LinReg7`, `D6train_LinReg7`, its long runs; reused) | `e1522be6…401d76e` |
| m2 | LinReg1_nt + AR(2) (`LinReg1_nt_ar2nw`) | an AR(2) residual on step 2b's closure | LinReg1_nt (`D6_LinReg1_nt`, `D6train_LinReg1_nt`, its long runs; reused) | `9a5728a3…2c93f34` |
| m3 | LinReg7_nt + AR(2) (`LinReg7_nt_ar2nw`) | the same on m1 | m1 | `8a6600b3…b331801` |

- **m1** was built 2026-10-30 (`exp_square_HIT/tools/lrs_noise_variant.jl LinReg7`, laptop; `TO_LRS/LinReg7_nt/build.log`):
  c' = (X'X + λ P + N S_dx(Σ)) \ X'Y, λ = 1 kept, on LinReg7's own rows 405–3999; Σ self-consistent in 15 iterations;
  max |c − c_LinReg7| / max |c_LinReg7| = 0.074; sd(η) 1.07–1.11 × LinReg7's. Used offline in paper Table
  `tab: own-draws` (Sec. 5.3) and `tab: skip-offline`.
- **m2, m3** (`exp_square_HIT/tools/lrs_ar_variant.jl <src> 2nw`, new option `fit_ar_noisy`, desktop 2026-11-04; logs
  `TO_LRS/<name>/build.log`): the restricted AR(2) (φ1 ≥ 0, φ2 ≤ 0, the rule of 2026-10-07) fitted by `ar2_ls` to the
  ACF at lags 1–20 of the residual GIVEN THE NOISY HISTORY, computed exactly (no draws): Γ_k = the clean residual's
  lagged covariance on the source's fit rows (405–3999) + C' M (S_k ⊗ D Σ D) M' C, the response to the white window
  noise of the self-consistent Σ (consecutive windows share h − k draws). Γ_0 reproduces the source's Σ (rel. 5e-13,
  1.6e-13). Σ_ξ = D R D, R from the exact innovation covariance, D so that the AR's marginal variance is diag Σ. The
  mean and Σ (stoch_distr) are the source's, byte-identical; only `ar_phi`, `ar_sigma_xi`, `ar_provenance` added.
  - m2: φ1 0.37–0.72, φ2 0 to −0.22; noisy-residual ρ1 0.25–0.60 (clean 0.23–0.82); the fitted ACF follows it to
    within 0.03 at lags 1, 2.
  - m3: φ1 0.52–1.31, φ2 0 to −0.42; noisy-residual ρ1 0.52–0.86 (clean 0.51–0.97).
  - Approximation: the mean was trained with WHITE draws; the closure draws the AR. Not iterated.
- All three run on the LinReg path (`D6_CLOSURE=lrs`, `run_online.sh lrsdeploy`) with no code change: the Snellius
  checkout as it is (`2287bdeb`), which reads `ar_phi` / `ar_sigma_xi` as for the round-1 AR closures.

## Runs (as step 3c's n1)

- Training hindcast (defect rule against the partner): `sbatch --array=1-3 batch_scripts/run_d6_train.sh LinReg7_nt
  LinReg1_nt_ar2nw LinReg7_nt_ar2nw` (from `exp_square_HIT/`).
- Full hindcast: `D6_OUT=$PWD/output/D6_<name> D6_CLOSURE=lrs D6_MODEL=$PWD/output/TO_LRS/<name>/LinReg.jld2 sbatch
  batch_scripts/run_d6_packed.sh`; long runs `sbatch -t 00:30:00 batch_scripts/run_online.sh lrsdeploy <name> <r>`,
  r = 1…5; gpu_a100. Scored as Table `tab: res s2b` (score_d6.jl --compare against the partner and LinReg1;
  train_hindcast.jl --full; longrun_stats.jl; gate_census.jl).

## Post hoc

- 2026-11-04: artefacts copied laptop -> Snellius `output/TO_LRS/`; sha256 equal on both (LinReg1_nt there equals the
  laptop's, so m2 is built on the closure that ran coupled). Snellius checkout `2287bdeb`, unchanged.
- Jobs (gpu_a100): training hindcast array 28894594 (tasks 1-3: m1, m2, m3); full hindcasts and long runs held
  `afterok` on it: m1 hindcast 28894596, long runs 28894597-28894601; m2 hindcast 28894602, long runs 28894603,
  28894605, 28894607-28894609; m3 hindcast 28894610, long runs 28894611-28894615.
- 2026-11-04, training hindcast (amendment 7; job 28894594, 18 min per task; `analysis/train_hindcast.jl LinReg1 LinReg7
  LinReg1_nt LinReg7_nt LinReg1_nt_ar2nw LinReg7_nt_ar2nw`, partners added to its PARTNER table; scratch log
  `train_hindcast_2bmore.log`). No member diverged. Spread-skill ratio <= 0.5 TU, own and / partner's:
  - m1 LinReg7_nt 0.37-0.41, / LinReg7 0.93-0.95: no defect (about LinReg7; bias at 2.5 TU +5.2 % against +6.7 %).
  - m2 LinReg1_nt_ar2nw 0.67-0.92, / LinReg1_nt 1.39-1.53: no defect. The AR widens step 2b's too-narrow short-lead
    ensembles (extreme ranks 0.15-0.29 against 0.23-0.39; calibrated 0.18); 0.5-3 TU 1.35-1.43 (LinReg1_nt 1.08-1.20,
    LinReg1 1.22-1.30). Guard: 83 of 60 000 steps, one member (LinReg1_nt none).
  - m3 LinReg7_nt_ar2nw 0.65-0.87, / LinReg7_nt 1.61-2.17: **the rule flags it** (four QoIs above 2), as it flagged
    S1.1 (LinReg7 + AR(2) / LinReg7 1.42-2.14, round1.md, Rik: "run S1.1 anyway"): the partner is far too narrow
    (0.37-0.41) and the AR widens it toward calibration (extreme ranks 0.14-0.29 against 0.36-0.46). Guard: 80 steps,
    one member. Its full hindcast and long runs had started (held only on the training job's success). **Rik, 2026-11-04: "Yes, continue"** (the exception of S1.1, for the same reason: the rule's intent, a broken artefact, is not met).
- **2026-11-06, scored** (all jobs COMPLETED; copied to the laptop, 900 files and bytes equal per hindcast, the 15 long
  runs' sha256 equal; scratch logs `score_2bmore.log`, `full_2bmore.log`, `longrun_six.log`, `gate_six.log`; paper Table
  `tab: res s2b`, rows added). No member diverged; five stable long runs each.

  | | CRPS_0.5 vs partner | cal (partner) | G vs partner | vs partner | CRPS_0.5 vs LinReg1 | cal (LinReg1 28) | vs LinReg1 |
  |---|---|---|---|---|---|---|---|
  | m1 LinReg7_nt | +1.5 % [+0.6, +2.5] | 6 (5) | pass | no | +21.0 % [+13.7, +28.8] | 6 | no |
  | m2 LinReg1_nt_ar2nw | **−6.1 % [−8.5, −3.9]** | **27 (15)**, +12 [+3, +19] | pass (0.77–1.25 vs 0.54–0.88) | **improves** | +0.4 % [−3.6, +4.3] | 27 | no |
  | m3 LinReg7_nt_ar2nw | **−12.1 % [−15.7, −8.6]** | **34 (6)**, +28 [+19, +34] | pass (0.67–1.12 vs 0.60–0.82) | **improves** | +6.3 % [+0.3, +12.7] | 34 (+6 [0, +12]) | no |

  - m1 ≈ LinReg7, as expected (spread 0.94–0.99 x, error 0.99–1.01 x LinReg7's in every window); long runs D 0.69, b
    −5.0 %, ratio 0.96, CRPS_100 0.384 (LinReg7 0.65, −4.9, 0.99, 0.386).
  - The AR restores the short-lead spread the noise training took (spread 1.43–1.52 x LinReg1_nt's, 1.67–2.12 x
    LinReg7_nt's up to 0.5 TU; r up to 0.5 TU 0.92–1.04 / 0.90–1.05), and with it the calibrated cells: m2 is back at
    LinReg1's CRPS_0.5 and count; m3 has 34 of 36 calibrated cells (the AR on LinReg7 in step 1, S1.1: 36;
    corrected 2026-11-06, first written as "the most of any closure"). CRPS_3 rises (+2.4 / +1.3 % vs the partners): too wide beyond 0.5 TU (r 1.35–1.41 / 1.12–1.26).
  - **The long runs lose what the noise training won**: m2 D 1.00, b −6.6 %, ratio 1.48, CRPS_100 0.431 (LinReg1_nt
    0.76, −5.1, 1.25, 0.380); m3 D 0.90, b −6.9 %, ratio 1.30, CRPS_100 0.420 (LinReg7_nt 0.69, −5.0, 0.96, 0.384).
    As step 1's AR on LinReg7 (S1.1): the AR buys short-lead calibration with the long runs. Guard: hindcast 0.29 /
    0.10 % of steps (partners 0.03 / 0), long runs 0.48 / 0.14 %.
