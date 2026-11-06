# Card: step 3a, no weight decay — the three CNN mean models without regularization, coupled, opened 2026-11-04

Paper: Results step 3a (`sec:res-net-mean`, Table `tab: res s3a`, rows "no wd"; the scaffold "Coupled runs submitted
2026-11-04"); offline: Results step 3a "Offline" (Table `tab: net-loss`, bottom rows; Fig. `net-fitrows`). Rik,
2026-11-04: "I think we would need a coupled run of the CNN, wd 0, no stopping (overfit) as well?", "should we also run
the two other overfitted losses online?"; in the paper's text: "[Do run the overfitted networks now, do not do the
likelihood networks yet, except the overfitted one (I am curious how that one does).]"; then "Yes, all of it".

⚠️ **Motivated post hoc.** Proposed after the coupled scores of the six step-3a closures (all worse than both
baselines) and after the offline tables `tab: net-loss` and `tab: net-tend` existed. The question they answer is the
paper's C6 for step 3a: the networks without weight decay have a held-out summed RMSE 1.6-1.7 times that of the
networks with it, yet below the regularized baseline's; does that offline error show coupled?

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**

- [x] Rik asked for the runs (quotes above); the card itself not yet seen.

## Closures

Network path (`D6_CLOSURE=lstm`, `StochLSTM`), `arch = :cnn`, exactly as step 3a's a1 (card `step3a.md`), but the
network is the one trained WITHOUT weight decay and WITHOUT early stopping: `TO_LSTM/step3/fitrows_<loss>_wd0_refit_s1`
(seed 1, the paper's rule), refitted on 1-10 TU for all 100 000 updates (`RIKFLOW_CNN_REFIT_UPD = last`; Snellius jobs
28845486, 28845496 GPU, 28845551 CPU for the CRPS). The MVG refitted about zero on that network's residual on 1-10 TU
(`cnn_closure_export.jl <run> mvg <name>`, as a1, a2).

| run | closure | the one change | matched partner | artefact (`TO_LSTM/step3/closures/`, sha256 of `StochLSTM_seed1.jld2`) |
|---|---|---|---|---|
| w1 | CNN (RMSE) + MVG, no wd (`CNN_rmse_nowd`) | no weight decay, no early stopping | a1 `CNN_rmse_mvg` (reused) | `3d0b69b6…f3d3836b` |
| w2 | CNN (CRPS) + MVG, no wd (`CNN_crps_nowd`) | the same, CRPS loss | a2 `CNN_crps_mvg` (reused) | `0e8c99a0…09ed9935` |
| w3 | CNN (likelihood) + MVG, no wd (`CNN_nll_nowd`) | the loss: likelihood instead of RMSE | w1 | `932039de…f19e7bb9` |

w3 has no partner with weight decay: the likelihood network with weight decay is not coupled (Rik: "do not do the
likelihood networks yet, except the overfitted one"). Its partner is w1, the same training with the RMSE loss, as a2's
is a1. Reason, for the text: with weight decay the three networks are alike offline (held-out summed RMSE 1.234-1.256,
CRPS 0.677-0.682) and two are coupled; without it they differ, and the likelihood network overfits least (training
window 1.092 against the linear mean model's 1.121, held out 1.376 against 1.912 / 2.063).

Built 2026-11-04 on the desktop (`squamish`, code `c24e8bd2`; `src/`, the export and `lrs_scale_variant.jl` equal to
the laptop's `334c6074` and to Snellius's `2287bdeb`; the laptop's training environment fails to load), from the laptop's
`fitrows_*` dirs (md5 equal after copy). Build log: `TO_LSTM/step3/closures/export_nowd_build.log` (laptop copy). Residual mean / rms on 1-10 TU
0.05 / 0.04 at most (w1, w2), 0.01 (w3); translation check max 9.1e-5 / 9.2e-5 / 1.1e-4 sd(dQ), rms 1.0e-5 (tolerance
1e-3). A first attempt with juliaup's Julia hung on the network home (9 min, 3 s CPU; killed by PID); the local copy
`/export/scratch2/rik/julia-1.13.0` built all three in about a minute.

## Expected, and why (Table `tab: net-loss`, three-seed means; seed 1 within the seed ranges given there)

Offline on 10-100 TU, summed over the QoIs, against a1 / a2 (with weight decay): RMSE 1.912 / 2.063 against 1.234 /
1.238, CRPS 1.100 / 1.225 against 0.677 / 0.678; per QoI up to 2.6 x the unregularized baseline's in the middle band.
On 1-10 TU they fit better than the linear mean model (0.831 / 0.765 against 1.121), so their MVG, fitted to the
training residual, is narrower (Σ from a residual of summed RMSE 0.67-0.73 x the regularized networks'). w3: 1.376 held out,
1.092 on 1-10 TU.

Expected, w1 vs a1 and w2 vs a2: higher CRPS_0.5 and e_0.5, ensembles too narrow (spread-skill ratio below the
partner's), more drift beyond 0.5 TU, long runs further from the reference; possibly divergences or more guard firings
where the network extrapolates. In the training hindcast (in-sample) they may do better than their partners. w3 vs w1:
better on every score, between a1 and w1. **Expected verdict vs the unregularized baseline, every closure: does not
improve.** Not expected but possible: the offline error does not show coupled (C6), i.e. w1 / w2 about as a1 / a2.

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1-7 apply unchanged)

As step 3a: hindcast K = 90 ICs, M = 10; G (KS guard against the regularized baseline), S, C against the unregularized
baseline; against the matched partner the same rules with the partner in place of both baselines (Sec.
`sec:criterion`). Reported, no clause: as step 3a. Tools: `score_d6.jl --compare` (against the partner and LinReg1),
`train_hindcast.jl --full`, `longrun_stats.jl`, `gate_census.jl`.

## Runs (as step 3a / step 3c's n2; gpu_a100)

From `exp_square_HIT/` on Snellius:
- Training hindcast (amendment 7, defect rule against the partner; the hindcast half of GATE 3): `sbatch --array=1-3
  batch_scripts/run_d6_train_lstm.sh step3/closures/CNN_rmse_nowd step3/closures/CNN_crps_nowd
  step3/closures/CNN_nll_nowd`.
- Long-run smoke (GATE 3): `sbatch batch_scripts/smoke_online_lstm.sh` with the three fit dirs.
- Full hindcast: `D6_OUT=$PWD/output/D6_<name> D6_CLOSURE=lstm
  D6_MODEL=$PWD/output/TO_LSTM/step3/closures/<name>/StochLSTM_seed1.jld2 sbatch batch_scripts/run_d6_packed.sh`, held
  `afterok` on the training hindcast; long runs `RIKFLOW_M4_MODEL_DIR=$PWD/output/TO_LSTM/step3/closures/<name> sbatch
  -t 00:30:00 batch_scripts/run_online.sh lstm 2 <r>`, r = 1...5, held `afterok` on the smoke.
- Cost ≈ 3 x 775 SBU + training hindcasts ~110 + smoke ~10; storage ≈ 4.8 GB.

## Post hoc

- 2026-11-04: artefacts copied desktop -> laptop and Snellius (`output/TO_LSTM/step3/closures/`); sha256 equal on all
  three machines. Snellius checkout `2287bdeb`, unchanged. Budget EINF-21030/L1; ⚠️ home quota 96.7 % of 200 GiB at
  submission (these runs add ≈ 4.8 GB, the step-2b-more runs as much).
- Jobs (gpu_a100): training hindcast array 28900281 (tasks 1-3: w1, w2, w3); long-run smoke 28900284. Full hindcasts
  held `afterok` on 28900281: w1 28900286, w2 28900301, w3 28900313; long runs held `afterok` on the smoke: w1
  28900288, 28900291, 28900293, 28900298, 28900300; w2 28900303, 28900306, 28900307, 28900309, 28900311; w3 28900315,
  28900317, 28900319, 28900321, 28900323. PARTNER entries added to `analysis/train_hindcast.jl`.
- **2026-11-06, scored** (all jobs COMPLETED; long-run smoke 28900284 `PASS: 3 closure(s)`; copied to the laptop after
  one interrupted scp, 900 / 50 files and bytes equal per hindcast, the 15 long runs' sha256 equal; scratch logs
  `score_nowd.log`, `full_nowd.log`, `train_nowd.log`, `longrun_six.log`, `gate_six.log`; paper Table `tab: res s3a`,
  rows filled). No member diverged; five stable long runs each.
  - ⚠️ **Training hindcast, defect rule: w2 FLAGGED** (spread-skill <= 0.5 TU / a2's 1.40 1.24 1.58 1.48 2.07 2.05: above
    2 in the small scales; w1 0.83–1.19, w3 0.83–1.14: no defect). Its full runs had already run: the hindcast was held
    only on the training job's success, not on this rule (as in step2b_more.md, m3). The rule's intent, a broken
    artefact, is not met: w2's own ratio is 0.82–1.17, the partner's 0.56–0.66 (too narrow). **Rik, 2026-11-06: "Yes, keep it"** (as S1.1 and step2b_more.md m3).
  - As expected, every closure much worse than its partner and than LinReg1: CRPS_0.5 0.438 / 0.424 / 0.343 against a1
    0.278, a2 0.283, LinReg1 0.221 (w1 vs a1 +57.6 % [+40.9, +73.4], w2 vs a2 +49.7 % [+34.5, +64.3]); w3 vs w1 −21.6 %
    [−32.1, −10.9] (improves on its partner by the rule; G 1.44–2.30 vs 2.47–2.71). Calibrated 8 / 8 / 5 (LinReg1 28).
  - Coupled overfit, as offline: in the training window (5 ICs) CRPS_0.5 0.238 / 0.333 / 0.225 against LinReg1 0.263
    (w3 the lowest of the step-3a closures there), held out 0.438 / 0.424 / 0.343. The offline order holds coupled (likelihood
    least overfitted, best of the three); against the regularized baseline it does not (offline better, coupled CRPS_0.5
    0.34–0.44 against 0.263, D 1.77–2.56 against 0.65).
  - Not as expected in kind: w1's guard fires on 5.4 % of hindcast steps (675 of 900 members, every IC) and 14.3 % of
    long-run steps; its bias at 0.5 TU is +8.0 % (all others negative), then −11.7 % at 2.5 TU; long runs D 2.56,
    CRPS_100 1.182, the worst of all closures. w2 −13.0 % already at 0.5 TU. w3 guard 1.2 % / 2.8 %.
