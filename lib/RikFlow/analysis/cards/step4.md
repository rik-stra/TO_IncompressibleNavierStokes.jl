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
- **2026-11-06, scored.** All 60 job records COMPLETED (4 x 10 hindcast tasks, 20 long runs). Copied to the laptop: 900
  files per hindcast, summed bytes equal (177 880 710 / 177 877 110 / 177 878 010 / 177 874 410); the 20 replicas'
  sha256 equal. No member diverged; five stable long runs each. Scratch logs `score_step4.log` (driver
  `score_step4.jl`, `paired_compare` x 10), `full_step4.log` (`train_hindcast.jl --full`), `longrun_step4.log`,
  `gate_hind_step4.log`, `gate_long_step4.log`. Snellius home 81.3 % after the runs.

  | closure | CRPS_0.5 | CRPS_3 | e_0.5 | cal. | b_0.5 | b_2.5 | guard | b | D | D range | CRPS_100 | e_100 | SS ratio | guard |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | LinReg1 (λ = 0) | 0.221 | 0.390 | 0.455 | 28 | −2.9 | −5.9 | 0.42 | −7.3 | 0.84 | 0.59–1.01 | 0.387 | 0.649 | 1.53 | 1.13 |
  | LinReg1_nt | 0.236 | 0.402 | 0.469 | 15 | −2.0 | −4.0 | 0.03 | −5.1 | 0.76 | 0.54–0.88 | 0.380 | 0.679 | 1.25 | 0.04 |
  | k1a LinReg1_nt + CNN (CRPS), learned sds + refitted corr. | 0.227 | 0.364 | 0.444 | 13 | −0.4 | −2.5 | 0.02 | −3.3 | 0.32 | 0.17–0.51 | 0.369 | 0.655 | 1.12 | <0.01 |
  | k1b the same, learned head | 0.231 | 0.365 | 0.446 | 15 | −0.3 | −2.3 | 0.02 | −2.6 | 0.29 | 0.19–0.38 | 0.333 | 0.581 | 1.27 | 0.01 |
  | LinReg7 (λ = 1) | 0.263 | 0.408 | 0.498 | 5 | −4.0 | −5.1 | 0 | −4.9 | 0.65 | 0.54–0.91 | 0.386 | 0.691 | 0.99 | 0 |
  | k2a LinReg7 + CNN (CRPS), with its noise, learned sds + refitted corr. | 0.230 | 0.387 | 0.456 | 12 | −3.2 | −3.0 | 0.05 | −3.2 | 0.33 | 0.30–0.58 | 0.375 | 0.667 | 1.18 | 0.01 |
  | k2b the same, learned head | 0.237 | 0.388 | 0.460 | 12 | −3.1 | −2.7 | 0.03 | −3.4 | 0.37 | 0.26–0.59 | 0.379 | 0.669 | 1.24 | 0.10 |

  Six digits: CRPS_0.5 0.22742 / 0.23139 / 0.23040 / 0.23742; CRPS_3 (own members, paired with the partner, which does
  not diverge) 0.36383 / 0.36490 / 0.38668 / 0.38831; e_0.5 0.4438 / 0.4461 / 0.4562 / 0.4603. Guard: hindcast 166 / 255
  / 499 / 287 of 1 080 000 steps (2 / 2 / 9 / 7 members, none in leads 1–200); long runs 3 / 10 / 24 / 201 of 199 500.
  Calibrated count from the pairing with the partner (none diverges); against LinReg1, k1b 14.

  Paired differences (90 % IC-block bootstrap, 10 000 resamples; % of the comparator). G: every closure's single-run
  D range overlaps the comparator's (LinReg7 0.54–0.91 for the criterion; the partner's otherwise): G passes throughout.
  | A vs B | CRPS_0.5 | CRPS_3 | e_0.5 | cal. A − B | S | C | verdict |
  |---|---|---|---|---|---|---|---|
  | k1a vs LinReg1_nt (partner) | −3.7 [−8.3, +0.6] | −9.4, resolved | −5.3, resolved | −2 [−5, +4] | fail | fail | **does not improve** |
  | k1a vs LinReg1 (criterion) | +3.0 [−0.9, +6.8] | −7.0, resolved | −2.6, n.r. | −15 [−20, −4] | fail | fail | **does not improve** |
  | k1b vs k1a (partner) | +1.7 [−0.0, +3.5] | +0.3, n.r. | +0.5, n.r. | +2 [−4, +4] | fail | fail | **does not improve** |
  | k1b vs LinReg1 (criterion) | +4.8 [−0.1, +9.6] | −6.7, resolved | −2.0, n.r. | −14 [−20, −4] | fail | fail | **does not improve** |
  | k1b vs LinReg1_nt (reported) | −2.0 [−7.3, +3.0] | −9.1, resolved | −4.8, n.r. | 0 [−5, +4] | fail | fail | no |
  | k2a vs LinReg7 (partner) | **−12.4 [−16.4, −8.5]** | −5.3, resolved | −8.5, resolved | +7 [+0, +12] | PASS | fail | **improves** |
  | k2a vs LinReg1 (criterion) | +4.4 [−0.1, +9.0] | −1.1, n.r. | +0.2, n.r. | −16 [−22, −5] | fail | fail | **does not improve** |
  | k2b vs k2a (partner) | +3.0 [+1.0, +5.2] (worse) | +0.4, n.r. | +0.9, n.r. | 0 [−5, +2] | fail | fail | **does not improve** |
  | k2b vs LinReg1 (criterion) | +7.5 [+2.1, +13.3] (worse) | −0.6, n.r. | +1.1, n.r. | −16 [−23, −6] | fail | fail | **does not improve** |
  | k2b vs LinReg7 (reported) | −9.7 [−13.7, −5.8] | −4.9, resolved | −7.7, resolved | +7 [+0, +11] | PASS | fail | improves |

  Against "Expected": as expected, k1 does not improve on LinReg1_nt, k2 improves on LinReg7, none on LinReg1. Not as
  expected: (1) **the long runs**, expected close to the partners, are the closest to the reference of every closure in
  the paper's results tables — D 0.29–0.37 against the lowest there, LinReg^E's 0.53 (LinReg7 0.65, LinReg1_nt 0.76),
  mean bias −2.6 to −3.4 % against −4.9 / −5.1 %; k1b's CRPS_100 0.333 and e_100 0.581 are below every row there (lowest
  0.360, LinReg^E + AR(2) λ = 1; 0.649, LinReg1). (2) The 3-TU hindcast:
  k1 CRPS_3 −9 % against LinReg1_nt, resolved, and the bias at 0.5 / 2.5 TU near zero for k1 (−0.4 / −2.5 % against
  −2.0 / −4.0 %). (3) b vs a went the other way from step 3a: the diagonal head is not better at 0.5 TU (k2b worse,
  resolved); for k1 it gives the better long runs. Calibration stays below LinReg1's (12–15 of 36 against 28).

- **2026-11-11, long-run figure** (Rik: "add a plot with long run trajectories and the long term distributions for
  these best performing models"). `plot_results_round1.jl --step s4 traj` (`fig_s4_traj`, laptop, analysis env) →
  `paper_tighten/figs/results_s4_traj.pdf`: the five runs of k1b and of k2a against the reference; pooled
  distributions of the reference, LinReg1, LinReg7, LinReg1_nt, k1b, k2a. Scratch scripts `s4_longrun_decomp.jl`,
  `s4_windows.jl`, `s4_firings.jl`, `s4_shape.jl` (logs alongside). Readings (hand-over `step4_longrun_figure.md`):
  summed D raw / mean matched / mean and sd matched: LinReg1 0.84 / 0.45 / 0.27, LinReg1_nt 0.76 / 0.34 / 0.10,
  k1b 0.29 / 0.22 / 0.22, LinReg7 0.65 / 0.14 / 0.08, k2a 0.33 / 0.24 / 0.27. The drop in D is mostly the mean shift
  (on λ = 0 also the width); the networks' shape is further from the reference than their skips' (skew −0.15 to
  −0.43 against +0.02 to +0.24: longer low tail, shorter high tail). Every long-run guard firing is a collapse of
  E[16,32] to the guard floor (−3.6 reference sds); replica 3 of LinReg1_nt, k1b and k2a fires within 81.6–82.0 TU,
  after a +3.0 / +3.5 sd peak of the reference. The runs share the reference's forcing (Sec. evaluation): per-run
  series correlation with the reference 0.4–0.6.

- **2026-11-11, zero-init offline rerun submitted** (Rik: "run the step-4 zero-init rerun", "Yes, pull and submit";
  runs_to_finalize.md item 1 / Q10). Snellius pulled to `d9c576dd` (fast-forward; it already had `ca84c79e`; the newer
  commits and the histnoise session's uncommitted edits there do not touch the CNN fit path). `run_cnn_skip.sh`,
  `ZTAG=z0`, gpu_a100: clean LinReg1 29130096, clean LinReg7 29130097, `skipntz0_LinReg1_nt` (fixed, lrs:LinReg1_nt)
  29130098, **k1** `skipnthz0_LinReg1_nt` (head, inj, lrs:LinReg1_nt, nll crps) 29130099, **k2** `skipnthz0_LinReg7`
  (head, inj, the head started at the old `step3/skip_LinReg7_nll_wd0.1_s1`'s Σ as before, nll crps) 29130101. Left
  out: the 7-round fixed-Σ RMSE loop on LinReg7 (`skipnt_LinReg7_rmse_wd0.3_r7`, not in the rule). The rule, fixed
  before any rerun number: keep the coupled k1/k2 if (i) the seed-mean summed RMSE 10-100 TU, clean and on own draws,
  is within 0.02 of the old, and (ii) the mean old-to-new map distance per seed (step4_converge.jl's unit) is below
  the old seed-to-seed distance in every band; else recouple (≈ 3 240 SBU) on Rik's word. Scratch
  `s4_z0_compare.jl` (self-test old vs old: RMSE 1.403 / 1.469, distance 0; old seed-to-seed k1 0.059 / 0.033 /
  0.043, k2 0.096 / 0.080 / 0.192). Then a SCORE pass for the own draws (`net_loss_skipnoise.jl`).
  SCORE pass (Rik: "Yes, both"): job 29130298, `afterok` on the five fits, script `$HOME/tmp_jobs/score_z0.sh` on
  Snellius (outside the repo), logs `step3/skip*z0_*_score.log`, scores `step3/innoise_scores/skip*z0_*.jld2`; k1 and
  k2 families at every wd, the others at the old closure's wd.
  **Scored, 2026-11-11: KEEP** (hand-over `step4_zeroinit_results.md`). All six jobs COMPLETED (fits 17-29 min,
  SCORE 1:46; ≈ 250 SBU); copied to the laptop, 923 files, sha256 equal. Protocol wd unchanged (k1 1, k2 0.3). k1:
  RMSE recorded 1.403 → 1.405, own draws 1.587 → 1.587, CRPS 0.886 → 0.887; map old → new 0.045 / 0.026 / 0.041
  against the old seed spread 0.059 / 0.033 / 0.043. k2: 1.469 → 1.470, 1.645 → 1.646, 0.918 → 0.918; 0.055 / 0.048 /
  0.107 against 0.096 / 0.080 / 0.192. Learned sds within 0.02. The update-0 rows are now the skip exactly (λ = 0 + CNN
  CRPS / RMSE 1.181, step 2b + CNN likelihood / RMSE 1.425). Moved more, not coupled: λ = 1 + CNN noise likelihood head
  1.528 → 1.541 (map outside the seed spread in [16,32]). The coupled step-4 results stand; the paper discloses the
  offset in one sentence.
- **2026-11-11, not run (Rik: "No, leave it"):** the CNN trained with its noise on a frozen clean LinReg1 (offline
  ≈ 40 SBU). Clean LinReg1 + CNN is LinReg1 offline (update 0, or 1.185 against 1.181), so it was never coupled; the
  noise-trained λ = 0 cell freezes LinReg1_nt instead (k1, Rik 2026-11-03: "LinReg1_nt skip").

## k1a noise-level test (opened 2026-11-13, before any run; Rik: "run the k1a noise test")

Question: is step 4's long-run gain fragile to the noise level at a fixed mean, or is step 5 B's collapse
(`k1_CNN_gen_nt`: k1's mean, noise ≈ 8 % wider in the middle band; long runs D 1.03, b −11.4 %, guard 2 %) specific to
its generator noise (its correlation, its small state-dependent part)?
- Closures: k1a with its deployed sd scaled on every QoI, Σ × f², f = 1.08 (and 0.92 if Rik includes it); mean,
  refitted correlation, everything else k1a's. Built on the laptop 2026-11-13 with `tools/scale_closure_noise.jl`
  (copy of `closures/LinReg1_nt_CNN_crps_nt_headcorr/StochLSTM_seed1.jld2`, `bd += log f` only; checks: Wd = 0, no
  uclip, every other weight identical, mean bit-identical on 20 random windows, Σ ratio f² to 4e-16):
  `closures/LinReg1_nt_CNN_crps_nt_headcorr_sx1.08` sha256 310b7e4a…, `_sx0.92` 3c5dd90a…, `_sx1.00` f11122ee… (control).
- Runs: long runs only, 5 replicas × 100 TU, `run_online.sh lstm 2 <r>` with k1a's replica seeds (index 2, deploy
  seed 1), gpu_a100, ≈ 125 SBU per arm. Control: `_sx1.00` replica 1 (≈ 25 SBU), expected bit-identical to k1a's
  replica 1; it shows the pairs are matched by code as well as by seed (the deploying code is now `d9c576dd`, k1a ran
  at `cd759ad3`).
- **Rule (fixed now), ×1.08 against k1a** (D 0.32, b −3.3 %): **noise level is the lever** if D ≥ 0.6 or |b| ≥ 6 %
  (the step-4 headline then needs a robustness caveat); **B's failure is generator-specific** if D ≤ 0.4 and |b| ≤ 4 %;
  graded in between. ×0.92 is read for direction only (sweet spot if it is also worse than k1a, slope if better).
- Expected: generator-specific, or the low end of graded (D 0.32–0.45, |b| 3–4.5 %): 8 % is the size of k1's own
  self-consistency error under its noise (residual / learned sd 1.01–1.06), and step 2b → k1 moved D far more than any
  sd change of this size moved a linear closure. ×0.92: about k1a's D, a slightly narrower long run (SS ratio below
  k1a's 1.12).
- Report: per arm against k1a and B: D (5-run range), b, CRPS_100, e_100, SS ratio, guard, and the KS split (raw /
  mean matched / mean and sd matched); hand-over `k1a_noise_test.md`.
- Submitted 2026-11-13 (Rik: "All three arms"): closures copied to Snellius (sha256 equal); deploying code `dff6b5b0`
  (its change to `ts_lstm.jl` touches generators only, plus a refactor of `cnn_features`: the control checks it).
  `RIKFLOW_M4_MODEL_DIR=<closure> sbatch -t 00:30:00 run_online.sh lstm 2 <r>`, gpu_a100: ×1.08 29217326, 27, 28,
  33, 34; ×0.92 29217336, 40, 44, 45, 47; ×1.00 (replica 1) 29217348. Snellius home 86.9 % before (+≈ 3.2 GB).
- **Seed check added 2026-11-13** (Rik: "also start in long runs for the two other trained k1a networks to check if
  training with a different seed still gives this optimum"): k1a's recipe exported from fit seeds 2 and 3 (old fits
  `skipnth_LinReg1_nt_crps_wd1_s2/_s3`, the same start offset as k1a; `cnn_closure_export.jl ... headcorrnt ...
  skip=LinReg1_nt`, laptop, `--project=.`). A seed-1 rebuild with today's code equals k1a's artefact (spec, weights,
  scaling; then deleted). Translation checks 8.2e-7 / 8.8e-7 target sd; learned sds within 1 % of k1a's.
  `closures/LinReg1_nt_CNN_crps_nt_headcorr_fs2` (StochLSTM_seed2, sha256 d160a0b2…), `_fs3` (seed3, f2b5070f…),
  sha256 equal on Snellius. Long runs `RIKFLOW_DEPLOY_SEED=<s>`, the same replica seeds as k1a: fs2 29218178-82, fs3
  29218189, 90, 91, 92, 99 (≈ 250 SBU). Read with the same measures; no rule fixed beyond reporting D, b and the KS
  split against k1a's 0.32 / −3.3 % and the ×1.08 arm.
- **Scored 2026-11-13** (all 21 COMPLETED, 10.5-11.5 min each, ≈ 300 SBU; copied to the laptop, sha256 equal; hand-over
  `k1a_noise_test.md`). Control ×1.00 replica 1 = k1a's replica 1 bit for bit (q, dQ, 40 001 steps): paired in code
  and seed. D / range / b / CRPS_100 / e_100 / SS / guard: k1a 0.32 / 0.17-0.51 / −3.3 / 0.369 / 0.655 / 1.12 / 3
  steps; **×1.08 0.32 / 0.27-0.38 / −2.6 / 0.368 / 0.645 / 1.17 / 0**; ×0.92 0.38 / 0.31-0.58 / −3.8 / 0.355 / 0.634 /
  1.15 / 9; seed 2 0.38 / 0.29-0.49 / −3.2 / 0.372 / 0.665 / 1.17 / 44; seed 3 0.40 / 0.32-0.56 / −3.6 / 0.369 / 0.672
  / 1.18 / 41; B 1.03 / 0.76-1.32 / −11.4 / 0.478 / 0.932 / 1.30 / 3 992. KS raw / mean matched / mean and sd matched:
  k1a 0.32 / 0.19 / 0.20, ×1.08 0.32 / 0.20 / 0.20, ×0.92 0.38 / 0.24 / 0.25, seed 2 0.38 / 0.22 / 0.23, seed 3 0.40 /
  0.19 / 0.22, B 1.03 / 0.67 / 0.53. **Rule: ×1.08 D ≤ 0.4 and |b| ≤ 4 % → B's failure is generator-specific** (as
  expected). ×0.92: a mild slope (b falls with the noise), not a sweet spot. Seeds: the optimum holds (D 0.32-0.40
  against ≥ 0.53 for every closure before step 4).

## Mechanism tests 1 and 2 (opened 2026-11-13, before any run; Rik: "do all of them"; review
`meta_files/step4_critical_review_2026-11-13.md`: k1's network changes its skip by ≈ 0 on average but by 0.1-0.5 ‰/step
at the level extremes, restoring)

**Measures, both tests** (against k1a and LinReg1_nt, k1a's replica seeds, 5 × 100 TU): D (5-run range), b, CRPS_100,
e_100, SS ratio, guard, the KS split (raw / mean matched / mean and sd matched), the tail split of b (samples below
the reference's 5 % quantile / middle 90 % / above 95 %, as the review's `tail_bias.jl`), collapse episodes per 100 TU
(Z[16,32] below the reference's 1 % quantile, `peak_collapse.jl`). Hand-over `mechanism_tests_step4.md`.

**Test 2: k1a with its network active only at the extremes / only in the bulk** (first; Rik's code).
- Gate (src `cnn_level_gate`, `level_gate_weights`, `cnn_window!(...; gate)`, V103; builder `tools/gate_closure.jl`):
  each QoI's NETWORK output `V1 h + cdec` × a weight set by THAT QoI's own level q*^n (per QoI, as the review's
  findings are per QoI; a joint criterion would switch a band's push on at another band's excursion). The skip `Ws x`,
  the target mean of the scaling (≈ 1e-3 target sd) and the noise (k1a's learned sds + refitted correlation) are never
  gated. Thresholds: quantiles of the tracked record's q* on 1-10 TU (columns 400-4000, the fits' window; no
  evaluation data): extremes weight 1 beyond the 2.5 / 97.5 % quantiles, 0 inside 5-95 %, half-cosine tapers between;
  bulk = 1 − extremes, so the two arms sum to k1a. Raw thresholds 2.5 / 5 / 95 / 97.5 %: Z06 572 / 639 / 1218 / 1236,
  E06 1.20 / 1.28 / 2.48 / 2.56, Z715 1127 / 1280 / 2762 / 2797, E715 0.149 / 0.165 / 0.348 / 0.353, Z1632 824 / 987 /
  2883 / 2963, E1632 0.0275 / 0.0329 / 0.0915 / 0.0940.
- ⚠️ The 1-10 TU record's low quantiles lie below the reference's: % of time below the record's 5 % / above its 95 %
  quantile (Z06 E06 Z715 E715 Z1632 E1632): reference 100 TU 1.4 2.0 1.1 1.1 1.2 1.2 / 7.9 7.6 7.6 8.4 5.1 5.6;
  LinReg1_nt 5.0 5.9 4.4 4.1 5.3 5.3 / 10.0 6.5 8.6 9.5 5.9 6.8; k1a 2.9 4.4 3.1 2.9 3.5 3.5 / 5.6 6.9 4.8 5.2 2.9 3.6.
  The low zone thus covers about the reference's lowest 1-2 % (where the review's push sits: bin 0-1 %), the high
  zone its top 5-8 %.
- Closures (laptop, `gate_closure.jl`, checks: weights and Cv the source's, gated output = skip + w .* network to
  1e-12 at levels in every zone): `closures/LinReg1_nt_CNN_crps_nt_headcorr_gext` (extremes-only) sha256 f2ba2d20…,
  `_gbulk` (bulk-only) 71211a0d…. V103 (126 / 126 CNN closure items, 660 / 660 generator + LSTM items pass): gate 1
  is k1a bit for bit, draws included; gate 0 is the skip, and with the target mean at zero exactly the frozen LinReg
  (1e-12); mixed zones give skip + w .* network.
- **Prediction:** extremes-only ≈ k1a, bulk-only ≈ LinReg1_nt. **Rule:** "≈ k1a" = D ≤ 0.45 and |b| ≤ 4.5 % (the five k1
  variants: D 0.29-0.40, b −2.6 to −3.8 %); "≈ LinReg1_nt" = D ≥ 0.6 and |b| ≥ 4.5 % (nt 0.76, −5.1 %). **The
  extremes carry the gain** if extremes-only ≈ k1a AND bulk-only ≈ nt; **the bulk carries it** if the reverse; **both
  contribute** if neither arm is ≈ k1a and neither ≈ nt; any other combination is reported as it falls (e.g. both ≈ k1a:
  the gain is not localised by level).

**Test 1: the linear-recipe control** (second).
- A LINEAR correction on the frozen LinReg1_nt, trained with k1's recipe: the same inputs as k1's CNN (the lag
  sequence of the mapped input and q*^n), `g = W [seq; q*] + b`, output at zero at the start (the skip), CRPS loss,
  learned constant diagonal head (`HEAD = inj`), `INNOISE = head` from `lrs:LinReg1_nt`, wd {0, .01, .03, .1, .3, 1} ×
  seeds 1-3, wd by the seed-mean validation loss, refit on 1-10 TU, seed 1 deployed as k1a (learned sds + refitted
  correlation). Fit-tool option `RIKFLOW_CNN_ARCH = linear` (cnn_mean_fit.jl); deployed on the network path with the
  linear map folded into the skip (`Ws' = Ws + M`, `V1 = 0`, `cdec = g(0)`): no src change; translation check
  against the fit's own predictions.
- Offline: how far it moves from the skip (RMSE recorded / own draws, response, the change by reference-quantile level
  bin as the review's §1A).
- **Prediction:** no gain: a linear map cannot push only at the extremes. **Rule:** D ≤ 0.45 → the recipe alone gives
  the gain; D ≥ 0.6 → the network is needed; graded in between.
