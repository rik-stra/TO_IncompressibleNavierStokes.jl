# Card: step 4 — the CNN on a frozen linear mean model, trained with its noise, coupled; opened 2026-11-06

Paper: methods Closures step 4 (`sec:lever-mean`, Eq. `skip net`); offline Results step 4 (Table `tab: skip-offline`);
`results_LSTMS.md` §38h–38r. Proposed 2026-11-04 and parked (Rik: "we will pick this up later"); Rik asked for this work
2026-11-06; on the deployed noise of the CRPS head: "please do D1 a and b" (both variants below).

⚠️ **Motivated post hoc.** Chosen after Table `tab: skip-offline` existed: k1 is the only step-4 map that beats its own
skip offline (noisy-input RMSE 1.587 against 1.636, CRPS 0.886 against 0.915, seed means), and k2 is the noise-trained map
on the regularized baseline's skip. Also after the coupled scores of steps 2b, 2b-more, 3a and 3a without weight decay
(Tables `tab: res s2b`, `tab: res s3a`). The deployed noise (a / b) was chosen before any coupled step-4 number.

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**

- [ ] Rik has seen this card.

## Closures

Network path (`D6_CLOSURE=lstm`, `StochLSTM`), `arch = :cnn` with a frozen linear skip: the deployed mean is the
LinReg's predicted correction plus the network's (Eq. `skip net`), `y = g(z) + Ws z`, `Ws` the LinReg's map on the
network's mapped input (`RF.linreg_skip_map`). Networks as step 3a's (C 16, K 2, D 64, tanh, lr 1e-3, h 5, FEAT dqinc),
trained on the CRPS with the closure's own noise in head mode (`RIKFLOW_CNN_INNOISE=head`, `HEAD=inj`; the draws follow
the current learned head, stop-gradient; the skip reads the noisy history too), weight decay by the protocol, refitted on
all 1–10 TU (`RIKFLOW_CNN_REFIT=1`). Seed 1 (the paper's rule). Fits on Snellius 2026-11-03/04 (results_LSTMS.md §38o,
§38r), on the laptop since.

The CRPS head is diagonal, but the network's residual under its own noise is strongly correlated across the QoIs
(max |off-diagonal| 0.87 for k1, 0.82 for k2). Two deployed noises (Rik, "D1 a and b"):
- **a, `headcorrnt`**: the learned sds with the correlation of `S_noisy`, the residual on 1–10 TU under the network's own
  (diagonal) window noise, about zero (4 draws) — the noise-trained analogue of step 3a's a3' ("learned head, refitted
  correlation"); the partners also draw correlated noise.
- **b, `mvgnt`**: the head as trained, diagonal — what closures.tex step 4 says now ("the closure draws from the head of
  the trained network. For the CRPS the head is diagonal"); the noise behind the offline numbers of `tab: skip-offline`.

| run | closure | the one change | matched partner | fit dir (`TO_LSTM/step3/`) | artefact (`TO_LSTM/step3/closures/`) |
|---|---|---|---|---|---|
| k1a | LinReg1_nt + CNN (CRPS), trained with its noise, learned sds + refitted correlation | the CNN on step 2b's mean model, wd 1 | LinReg1_nt (reused) | `skipnth_LinReg1_nt_crps_wd1_s1` | `LinReg1_nt_CNN_crps_nt_headcorr` |
| k1b | the same, learned head (diagonal) | no correlation in the noise | k1a | same | `LinReg1_nt_CNN_crps_nt_head` |
| k2a | LinReg7 + CNN (CRPS), trained with its noise, learned sds + refitted correlation | the CNN on the regularized baseline's mean model (the skip stays LinReg7), wd 0.3 | LinReg7 (reused; Rik's D4) | `skipnth_LinReg7_crps_wd0.3_s1` | `LinReg7_CNN_crps_nt_headcorr` |
| k2b | the same, learned head (diagonal) | no correlation in the noise | k2a | same | `LinReg7_CNN_crps_nt_head` |

Partners of the b runs: the a run of the same network, the one change being the correlation (as step 3a pairs every
residual variant with one closure). Every closure is also scored against LinReg1 (the criterion) and the b runs against
the a run's partner too (reported, no clause). k3 (LinReg7 + CNN, likelihood, clean, wd 0.1, MVG, vs LinReg7) is
optional and not on this card unless Rik adds it.

Build: `cnn_closure_export.jl <fit dir> headcorrnt|mvgnt <name> skip=<LinReg1_nt|LinReg7>` on the desktop at Rik's commit,
in **Float64** (the least-squares skip's coefficients reach 9.4e3 target sd on the mapped input; StochLSTM's Float32 lag
window alone puts it 4.6e-3 sd off LinReg1_nt, above the 1e-3 rule; LinReg7 7e-5). Checks: `Ws` against the LinReg's own
predictions on every record step (Float64); translation check, deployed against Lux + skip (`pred3`), 300 steps,
tolerance 1e-3 target sd (here the sd of what the LinReg misses, a stricter unit than step 3a's sd(dQ)).
Dry run 2026-11-06 (laptop, `--project=.`, uncommitted code; artefacts deleted, the build is the desktop's): `Ws`
against the LinReg's own correction on 39 596 record steps, max rel. diff 1.2e-13 (LinReg1_nt, |Ws| up to 9.4e3) and
1.2e-14 (LinReg7, |Ws| up to 137); translation check max 9.1e-7 sd (k1a, k1b), 2.7e-6 (k2a, k2b), rms 1.5e-7 / 4.0e-7.
Clean residual on 1–10 TU, skip + network over skip alone: k1 1.00 0.99 0.87 0.95 0.97 0.95 (the network takes 13 % of
LinReg1_nt's residual in Z[7,15]); k2 0.59 0.92 0.22 0.19 0.54 0.43 (it rebuilds most of what the ridge leaves). Noise
(target sd units; per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32]):
- k1: learned sd / clean residual rms 1.11 1.06 1.25 1.53 1.00 1.06; the residual under its own noise / learned sd
  1.01 1.02 1.03 1.01 1.06 1.05 (self-consistent within 6 %); that residual's correlation up to 0.87.
- k2: learned sd / clean residual rms 1.11 1.06 1.26 1.46 1.00 1.07; residual under its noise / learned sd 1.01 1.02 1.02
  1.01 1.06 1.03; correlation up to 0.82.
**Built 2026-11-06 on the LAPTOP** (Rik: "Build on the laptop"; the desktop unreachable, ssh timed out), code `cd759ad3`
(the export and `src/` as `a38aec13`), Julia 1.12.7, `--project=.` (the export needs no Lux); log
`TO_LSTM/step3/closures/export_step4_build.log`. Every number as in the dry run above. Artefacts
(`StochLSTM_seed1.jld2`, 160 906 bytes each, sha256): k1a `5a974de9…58294de94`, k1b `b8d54227…a49cadb59`, k2a
`8879ba67…dac875cf`, k2b `f964242d…2f31bf`. Skips: `TO_LRS/LinReg1_nt/LinReg.jld2` `3bf7358d…3a70f5ec26`,
`TO_LRS/LinReg7/LinReg.jld2` `c9054f70…d673b779` (the translation check against `pred3`, written on Snellius, ties
them to the files the fits used).

## Expected, and why (Table `tab: skip-offline`, seed means; results_LSTMS.md §38o, §38q, §38r)

- **k1 vs LinReg1_nt.** Offline the map stays close to LinReg1_nt (0.21 / 0.08 / 0.12 of the LinReg1–LinReg7 distance
  per band), with the same response (0.65/0.76 in the middle band) and a smaller fitted noise in Z[7,15] and the small
  band (sd 2.13 / 1.04–1.26 × LinReg1's, against 2.58 / 1.15–1.38); on noisy inputs RMSE 1.587 against 1.636, CRPS 0.886
  against 0.915; clean 1.403 against 1.425. Expected coupled: close to LinReg1_nt (CRPS_0.5 0.236, e_0.5 0.469, D 0.76,
  CRPS_100 0.380), slightly better in Z[7,15], narrower in the bands where its noise is smaller. **Expected verdict vs
  LinReg1_nt: does not improve** (an offline gain of 3 % on one seed's map is small against the coupled differences
  resolved so far); possible: improves. **Vs LinReg1: does not improve** (LinReg1_nt does not: CRPS_0.5 0.236 against 0.221).
- **k2 vs LinReg7.** Offline the map lands near LinReg1_nt's fixed point (response 0.63/0.72, noise 2.23/3.21 × LinReg1's
  in the middle band), between LinReg1_nt and LinReg7 and nearer LinReg1_nt; clean 1.469 against LinReg7's 2.873, noisy
  1.645 against 3.182. Expected coupled: like LinReg1_nt rather than LinReg7 — CRPS_0.5 and e_0.5 lower than LinReg7's
  (0.263, 0.498), more calibrated cells (LinReg7: 5), long runs a little further from the reference (LinReg7 D 0.65
  against LinReg1_nt 0.76). **Expected verdict vs LinReg7: improves; vs LinReg1: does not improve.**
- **b vs a (both).** Step 3a's learned head against its correlated partners (CRPS_0.5 0.270 against 0.283 / 0.286; D 1.45
  against 1.35 / 1.33): expected lower CRPS_0.5 and more calibrated cells for b, about the same beyond 0.5 TU, long runs
  a little further. **Expected verdict b vs a: improves at 0.5 TU by the rule; not certain.**
- Possible problems: the a runs draw correlated noise the network never saw in training (its injected noise was
  diagonal), so their residual under it may not match the sds; the defect rule (training hindcast) would show it.

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1-7 apply unchanged)

As step 3a: hindcast K = 90 ICs, M = 10; G (KS guard against the regularized baseline), S, C against the unregularized
baseline; against the matched partner the same rules with the partner in place of both baselines (Sec.
`sec:criterion`). Reported, no clause: as step 3a. Tools: `score_d6.jl --compare` (`paired_compare` against the
partner and LinReg1), `train_hindcast.jl --full`, `longrun_stats.jl ../TO_LSTM/step3/closures/<name>`, `gate_census.jl`.

## Runs (as step 3a no-wd; gpu_a100)

From `exp_square_HIT/` on Snellius, at a commit holding `src/ts_lstm.jl`'s CNN skip (Rik commits, pushes, pulls):
1. Training hindcast (amendment 7, defect rule against the partner; the hindcast half of GATE 3): `sbatch --array=1-4
   batch_scripts/run_d6_train_lstm.sh step3/closures/<k1a> <k1b> <k2a> <k2b>`; long-run smoke (GATE 3):
   `sbatch batch_scripts/smoke_online_lstm.sh` with the four closure dirs.
2. **The defect rule is checked on the laptop (`analysis/train_hindcast.jl`, PARTNER entries added 2026-11-06) and Rik
   asked BEFORE the production runs are submitted** (two earlier batches were chained on job success only and flagged
   after the fact).
3. Full hindcast: `D6_OUT=$PWD/output/D6_<name> D6_CLOSURE=lstm
   D6_MODEL=$PWD/output/TO_LSTM/step3/closures/<name>/StochLSTM_seed1.jld2 sbatch batch_scripts/run_d6_packed.sh`; long
   runs `RIKFLOW_M4_MODEL_DIR=$PWD/output/TO_LSTM/step3/closures/<name> sbatch -t 00:30:00 batch_scripts/run_online.sh lstm
   2 <r>`, r = 1...5.
- Cost ≈ 4 x 775 SBU + training hindcasts ~150 + smoke ~10 ≈ 3,250 SBU; storage ≈ 6.4 GB (home ~78 % of 200 GiB).

## Post hoc

- 2026-11-06: Snellius pulled `2287bdeb` -> `cd759ad3` (`--ff-only`; Rik: "Yes, go ahead"); the three untracked
  `run_cnn_*.sh` there were byte-identical to the committed ones (git hash-object), moved aside for the pull, compared
  after it and removed. The four artefacts copied laptop -> Snellius, sha256 equal; the LinReg1_nt and LinReg7 skip files
  there equal the laptop's. Home 78.2 % of 200 GiB before the runs.
- Jobs (gpu_a100): training hindcast array 28939903 (tasks 1-4: k1a, k1b, k2a, k2b); long-run smoke 28939904. Nothing
  chained: production waits for the defect rule and Rik.
- Long-run smoke 28939904 COMPLETED (2:30, code `cd759ad3`): `PASS: 4 closure(s)`, the guard fired 0 / 100 steps after
  the warm-up for each.
- ⚠️ **Production submitted BEFORE the defect rule** (Rik, 2026-11-06: "start in the full hindcasts", "also start the
  full runs"; the training hindcast was then half run, tasks 1-2 running, 3-4 pending). At `cd759ad3`, gpu_a100, not
  chained. Full hindcasts (`run_d6_packed.sh`, array 1-10): k1a 28940364, k1b 28940370, k2a 28940376, k2b 28940383. Long
  runs (`run_online.sh lstm 2 <r>`, `-t 00:30:00`, r = 1-5): k1a 28940365-69, k1b 28940371-75, k2a 28940377, 78, 80, 81,
  82, k2b 28940384-88.
- Training hindcast 28939903 COMPLETED (tasks 1-4); copied to the laptop, 50 files per closure, summed bytes equal
  (9 882 196 / 9 882 000 / 9 882 050 / 9 881 850). **Defect rule (`train_hindcast.jl`, scratch log
  `train_step4.log`): none flagged** — no member diverged; spread-skill ≤ 0.5 TU / partner's: k1a / LinReg1_nt 1.05 1.00
  1.06 1.08 1.03 1.03; k1b / k1a 1.00 1.03 0.86 0.88 0.85 0.85; k2a / LinReg7 1.22 1.26 1.48 1.50 1.39 1.40; k2b / k2a 0.98
  1.01 0.85 0.86 0.86 0.86. Own spread-skill ≤ 0.5 TU: k1a 0.48-0.63, k1b 0.47-0.53, k2a 0.49-0.59, k2b 0.48-0.55
  (LinReg1_nt 0.45-0.61, LinReg7 0.39-0.44, LinReg1 0.61-0.78). Guard: 0 for the a runs; k1b 52, k2b 13 of 60 000
  forecast steps (one member each). Bias at 0.5 TU −5.1 to −6.9 % (LinReg1_nt −4.5, LinReg7 −7.6), at 2.5 TU +4.9 to
  +6.6 % (LinReg1_nt +1.9, LinReg7 +6.7).
