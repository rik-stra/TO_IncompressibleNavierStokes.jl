# Card: step 5 on other skips — the stochastic skip network trained with its own noise (A2) and on step 4's CNN mean (B); opened 2026-11-11

Paper: `sec:lever-stoch` (closures.tex), `sec:res-stoch` (Table `tab: s5-offline`, Table `tab: res s5`). Recipe: card
`step5.md` (everything there that this card does not change carries over).

**Origin.**
- Rik, via the paper session (2026-11-11): "start step 5 on both linreg1_nt and LinReg7". Why (paper session): step 4's
  networks sit on LinReg1_nt (k1) and LinReg7 (k2), step 5 on LinReg1, so step 4 against step 5 mixes what the network
  learns with which skip it sits on. C5 also predicts that step 5's short-lead loss shrinks on a skip that responds
  weakly to its own draws (LinReg1_nt 0.66 / 0.76, LinReg7 0.58, against LinReg1's 2.45).
- **Design amended the same day, before any production run** (paper session's HOLD, then Rik: **"A2, both skips; and
  B"**). The first default, training on the recorded history, was dropped: Table `tab: skip-offline` shows that every
  network trained on the clean history rebuilds LinReg1's map whatever its skip (on λ = 1 the clean CNN takes the
  recorded RMSE from 2.87 to 1.20 and the middle-band response from 0.58 to 2.31 / 1.99), so a clean-trained generator
  on these skips would likely reproduce step 5 on LinReg1. The clean-history grid already started on LinReg1_nt is kept
  as a **diagnostic** (below), not as a closure.

⚠️ **Motivated post hoc.** Proposed after step 5 (on LinReg1) and step 4 were scored coupled (Tables `tab: res s4`,
`tab: res s5`). No evaluation-window number of these closures exists when this card is written.

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**

- [ ] Rik has seen this card (offline gate summary to the paper session; GATE 1 = Rik's go for Snellius).

## Closures

| run | closure (artefact `TO_LSTM/step3/closures/<name>`) | the one change against step 5 | matched partner | also reported against |
|---|---|---|---|---|
| s5n | A2 on LinReg1_nt (`LinReg1_nt_CNN_gen_nt`) | skip LinReg1_nt; trained with its own noise in the history | LinReg1_nt (reused) | k1b `LinReg1_nt_CNN_crps_nt_head`; LinReg1 (criterion) |
| s5r | A2 on LinReg7 (`LinReg7_CNN_gen_nt`) | skip LinReg7 (λ = 1); trained with its own noise in the history | LinReg7 (reused) | k2a `LinReg7_CNN_crps_nt_headcorr`; LinReg1 (criterion) |
| s5k | B on k1's CNN mean (`k1_CNN_gen_nt`) | skip = k1's frozen CNN mean + LinReg1_nt; trained with its own noise in the history | k1a `LinReg1_nt_CNN_crps_nt_headcorr` (the same mean with learned sds + refitted correlation) | k1b; LinReg1 (criterion) |

Priority if not all fit before the last submission (Tue 17 Nov): s5n, then s5k, then s5r (paper session; Rik decides).

Every closure starts exactly at its partner: g_θ ≡ 0 at the start (output weights and bias zero; `fit_gen` asserts the
samples are L₀ ε at update 0), L₀ = the partner's noise.
- s5n / s5r: ŷ = W_s z + b + g_θ(z, q*, ε) + L ε with W_s z + b the skip LinReg's map (frozen, Float64), L₀ = chol Σ of the
  skip's own MVG and the target centred at its mean (`RIKFLOW_MLP_SKIP_CENTER=mvg`). For LinReg1_nt, Σ is its
  self-consistent Σ (its residual under its own window noise).
- s5k: ŷ = W_s z + m_k1(z, q*) + g_θ(z, q*, ε) + L ε with m_k1 k1's network (frozen) and W_s LinReg1_nt's map, in k1's
  target frame (`SKIP_CENTER=data`, as k1 was trained); L₀ = chol of k1a's deployed Σ (the refitted head's sds with the
  correlation of `S_noisy`). The start is k1a exactly. Deployed as a generator carrying k1's network as a frozen mean
  (`Cv.mean_net`, `RF.mean_net_from`; export `meannet=<k1 artefact>`).

## The history noise: A2 (all three closures)

🔒 **`RIKFLOW_CNN_GEN_HIST=own`**: every training window carries the closure's own noise, as step 4's `INNOISE = head`:
η = the **current** noise skip L times standard normals, read before each gradient (a stop-gradient), one η per
correction of the window, the levels the running sum anchored at the prediction step (`window_noise`); the frozen skip
reads the same noisy history (the linear skip's exact response `skip_dy`; s5k: k1's network evaluated on the noisy
window), so the target is what the noisy skip misses. Validation: one fixed set of standard normals times the current L.

**Disclosed:** A2 injects only the **constant** part of the closure's noise (L ε), not the network's state-dependent
part (the generator's response to ε). Step 4's head mode injected its learned (constant, diagonal) head likewise.

## Carried over from step 5 unchanged

- Network, ε in the first dense layer, L lower triangular and fully learnable with no weight decay.
- Loss: the fair energy score of M = 4 samples of the six-QoI standardised vector; validation with one fixed set of 16
  draws per row; the offline check with 256 fixed draws, paired with the stage-1 start on the same draws.
- Weight-decay grid {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}, seeds 1–3, the rule = the seed-mean validation loss; Adam,
  lr 1e-3, plateau schedule, batch 256, early stop on 8.2–10 TU, refit on 1–10 TU, seed 1 coupled.
- Deployed: emission `:generator`, ε from the member's stream before the forward pass (common random numbers with a
  LinReg partner), nothing added after the network.
- **Zero init.** The paper session asked for step 4's zero-init fix (`ca84c79e`). Step 5's generator has its own,
  equivalent fix since its build (2026-11-06, before `ca84c79e`): `mkmodel_gen` starts the output weights AND bias at zero
  and `fit_gen` asserts the start. **The original step-5 fit (`LinReg1_CNN_gen`) had the fix** (start = LinReg1 + MVG to
  1.5e-7 / 2.8e-6 sd, V98). s5k's frozen k1 is a separate matter (below).

## The held-out stage 1

Step 5's held-out early stop (Rik at GATE 1, 2026-11-06): stage 1 uses the skip refitted on 1–8.2 TU, so the early-stop
block 8.2–10 TU is out of sample for the skip too. For LinReg1 the code's "skip + least squares of its residual" is that
refit; for any other linear skip on the same regressor it is again least squares on 1–8.2 TU, which would make stage 1
skip-independent. 🔒 **s5n / s5r refit the skip's own recipe on the stage-1 fit rows** (`tools/skip_heldout.jl` →
`TO_LRS/<skip>_s1`, not deployed; `RIKFLOW_CNN_GEN_STAGE1=heldout_skip`): LinReg7 → ridge λ = 1 (bias unpenalised, the
same λ on fewer rows) and its MLE Gaussian; LinReg1_nt → the noise-trained closed form with its self-consistent Σ on those
rows; LinReg1 → least squares = step 5's stage 1 (asserted to round-off). Under A2 stage 1's noisy target uses the
refitted skip's own response.
- Built and checked 2026-11-11 (laptop; copied to the desktop, sha256 equal): each recipe reproduces its skip on the
  skip's own rows (c rel. ≤ 5e-11, Σ ≤ 6e-15). Rows 405–3280 (1.012–8.200 TU). Clean rms on the held-out rows
  3281–3999 over the skip's own: LinReg1_s1 0.93–1.14 (LinReg1 in sample 0.90–1.12); LinReg7_s1 0.95–1.18 (0.89–1.08);
  LinReg1_nt_s1 0.94–1.09 (0.92–1.07).
- **s5k's stage 1** uses k1's own stage-1 network (trained on 1–8.2 TU over LinReg1_nt fitted on 1–10 TU), exactly as
  k1's own early stop did; its LinReg part is therefore in sample on 8.2–10 TU (disclosed; refitting k1's stage 1 on a
  held-out skip is not done).

## s5k: which k1

The step-4 session reruns k1 with the zero-init fix (Snellius jobs 29130096–29130101); its keep-or-recouple verdict
decides the frozen mean: "keep" → the old fit `step3/skipnth_LinReg1_nt_crps_wd1_s1` and the deployed k1a / k1b;
"recouple" → the z0 fit and its artefacts. s5k's training starts only after that verdict (paper session).

## Expected, and why (A2 and B written 2026-11-11, before any A2 / B run; the smoke excepted)

- **A2 keeps the skip's weak response to its own draws, offline and coupled.** Trained with its noise in the history,
  the network cannot rebuild the least-squares map, whose large coefficients are what makes LinReg1 respond 2.45 to its
  own draws; as k1/k2 (step 4) it should learn a correction that holds up under the noise. So the response column
  (`gen_owndraws.jl`) should stay near the skip's (0.66 / 0.76; 0.58), well below step 5's 2.40.
- **Offline (each model under its own noise, the A2 check):** a gain against the held-out start of the same order as
  step 5's (−0.84 %), possibly larger for s5r (k2 rebuilt more of what the ridge leaves than k1 of LinReg1_nt). The
  spread follows the level as in step 5; L narrows from L₀ in the bands where the network takes over the scale.
- **Coupled against the partner:** by C5, a smaller short-lead loss than step 5's (+24.3 %), since the response is
  weak; the expected verdict is still "does not improve" for s5n (k1a did not improve on LinReg1_nt, −3.7 % n.r.). s5r
  may improve on LinReg7 at short leads through the mean, as k2a did (CRPS₀.₅ −12.4 %). Long runs near the skip's or
  better (step 5's were closer to the reference than LinReg1's; k1/k2's D 0.29–0.37 the best of the paper).
- **Against the step-4 network on the same skip (k1b, k2a):** A2 differs from step 4 only in the network's form (a
  generator with a full learnable L instead of a mean with a learned head). Expected about equal at short leads; the long
  runs no better than step 4's.
- **s5k against k1a:** it starts at k1a; with k1's mean frozen, what is left is the state-dependent noise. Expected about
  equal to k1a, the short-lead calibration possibly better (a state-dependent spread), the long runs about k1's.
- **Against the criterion (LinReg1):** none improves, as no network so far (calibration 12–15 against 28 for step 4).
- **Diagnostic (clean history on LinReg1_nt):** the generator rebuilds part of the least-squares map (its sample mean's
  error well below LinReg1_nt's on the recorded history, response to its own draws rising towards LinReg1's).

## Clauses (paper Sec. 3.5 `sec:criterion`; as `step5.md`)

Hindcast K = 90, M = 10, 100 + 1200 steps; CRPS₀.₅, CRPS_all, e₀.₅, r, calibrated cells, rank histograms; long runs
5 × 100 TU.
- **G:** the KS guard against LinReg7 (FAIL iff all five single-run summed KS distances lie above all five of LinReg7's).
- **S:** the CRPS₀.₅ paired difference against the baseline, 90 % CI upper end < 0.
- **C:** the calibrated cells against the baseline, paired CI of the count difference > 0, AND the CRPS₀.₅ CI upper end
  < +10 %.
- **Verdict:** IMPROVES iff G passes and (S or C), each baseline reported: (1) the matched partner (the step's verdict);
  (2) the step-4 network on the same skip (k1b / k2a; s5k: k1b); (3) LinReg1 (the criterion).
- The defect rule on the training hindcast against the partner (`train_hindcast.jl`) before production.

**Disclosures for the paper:**
- A2 injects only the constant noise part (L ε), not the state-dependent part.
- k1/k2 trained with their (diagonal) learned head's noise; A2 with the full L's. Step 4 against A2 differs in the
  network's form and in that noise.
- k1/k2's fits started with Lux's random output bias (up to 0.125 target sd; `ca84c79e` came after their runs); the
  generators start exactly at their partners. s5k uses whichever k1 the step-4 session's verdict keeps.
- The held-out stage 1 refits each skip's own recipe on 1–8.2 TU (s5n / s5r); s5k's stage 1 is k1's own.

## Offline gate (desktop CPU) and what is reported

- **Check (A2 PASS rule):** the fair CRPS of 256 fixed draws per row on 8.2–10 TU, each model under its own history noise
  (the fit's from its L, the start's from L₀, the same standard normals), against the stage-1 start on the same draws;
  PASS iff not above it. Beside it: the same on the recorded history, the energy score, the skip's own closed form.
- **Reported:** the spread per tenth of the level beside the power law; the network's share of the sample variance; L
  against L₀; the correlation against Σ; the sample mean's error against the skip's; error / sd; the response to its own
  draws and the error with its own draws in the history (`tools/gen_owndraws.jl`, Table `tab: s5-offline` cols 7–8).
- A failed check is reported and the closure is still run (step 5 D9).

## Runs (gpu_a100; as step 5)

Per closure: training hindcast (`run_d6_train_lstm.sh`), long-run smoke, full hindcast (`run_d6_packed.sh`, array 1–10),
5 long runs (`run_online.sh lstm 2 <r>`, `-t 00:30:00`). ≈ 810 SBU each, ≈ 2 430 SBU for three. Deadline: last submission
Tue 17 Nov, freeze Fri 20 Nov.

## Built (2026-11-11; uncommitted, Rik commits)

| item | code | test or check |
|---|---|---|
| held-out stage 1 on another skip | `tools/skip_heldout.jl` (includes `lrs_noise_variant.jl`; writes `TO_LRS/<skip>_s1`); `cnn_mean_fit.jl` `RIKFLOW_CNN_GEN_STAGE1=heldout_skip` (`gen_linreg_pred`, `GEN_S1`, its linear response `K`) | recipe reproduces each skip on its own rows; `gen_linreg_pred` reproduces `SKIPQ` and `SKIPK`; the refitted skip's response against its own predictions (< 1e-8); on LinReg1 = `heldout` to round-off (asserted), and the desktop rerun of step 5's wd 1 seed 1 reproduces it exactly (best @ 1 300, val 1.6645, check 3.4764 vs 3.5059) |
| A2: own noise in the history | `RIKFLOW_CNN_GEN_HIST=own`: η = `eta_from(L_current, ε)` per batch (own stream), `noisy_tensor`, target − `s1_dy` / `skip_dy` (− the frozen net's response for B); validation from one fixed set of normals × current L; `OWNCHECK` (each model under its own noise, paired) | desktop smoke (LinReg1_nt, 400 updates): runs through stage 1, OWNCHECK, refit; the start check (samples = L₀ ε at update 0) passes |
| A2c: the total noise injected (Rik, 2026-11-11: "A2c, check B too") | `RIKFLOW_CNN_GEN_HIST=owntot`: the injected factor = chol of the generator's average conditional sample covariance (network part included; `gen_tot_factor`, 512 fixed fit rows × 8 fixed draws), refreshed at every validation (stage 1) / every `val_every` updates (refit), outside the gradient; `ESCAPE CHECK` line per fit (injected sd / drawn sd, L / L₀, the network's share); `own` unchanged bit for bit | the LinReg7 grid (below) |
| B: frozen CNN mean, training | `RIKFLOW_CNN_GEN_NETSKIP=<step-4 run>` (`GEN_K`, `knet`, `knet_dy`, `knet_off`; `SKIP_CENTER=data`; Σ = the refitted head's sds × corr(S_noisy)); stage 1 = its stage-1 net | its stage-1 and refitted networks reproduce its `pred.jld2` (< 1e-4 target sd, asserted); smoke below |
| B: deployment | `src/ts_lstm.jl`: `cnn_mean_net`, `frozen_mean`, `mean_net_from`, `with_mean_net`, `_cnn_features`; `cnn_window!` adds `m(x)`; `check_shapes` (generator + skip only, shapes) | **V101** (40 items): at its start the generator = the step-4 closure's mean + L ε (forward pass, and online with the member's ε), refusals, save/load; V98 81/81 unchanged (121/121, laptop) |
| B: export | `cnn_closure_export.jl … gen <name> skip=LinReg1_nt meannet=<k1 closure>`: fit dir = the run's `netskip`, Ws and scaling equal, frozen mean + Ws z = the k1 closure's deployed mean on 300 steps; translation check on the samples | smoke below |
| any skip in the tools | `gen_owndraws.jl` (skip = RIKFLOW_MLP_SKIP; the closure's Ws checked against it), `gen_offline.jl` labels; `train_hindcast.jl` PARTNER: `LinReg1_nt_CNN_gen_nt` → LinReg1_nt, `LinReg7_CNN_gen_nt` → LinReg7, `k1_CNN_gen_nt` → k1a | — |

Desktop build copies (not the shared checkout): `/export/scratch2/rik/step5s_build` (A2's grid; HEAD `d9c576dd` + the
step-5-skips files as of the A2 code), `/export/scratch2/rik/step5b_build` (+ B). Grids: logs `/export/scratch2/rik/tmp/
step5s_grid/` (diagnostic), `step5s_a2/` (A2); fits `TO_LSTM/step5/gen_hs_<skip>_wd<wd>_s<s>` (diagnostic),
`gen_a2_<skip>_wd<wd>_s<s>` (A2).

## Offline

### Diagnostic: the clean-history generator on LinReg1_nt (started 16:01 before the HOLD; kept as a diagnostic)

Grid (8 weight decays × 3 seeds, held-out stage 1 = LinReg1_nt refitted on 1–8.2 TU): every fit learns, and far: the
check against the held-out start is −12 to −21 % (wd 0.3: −20.7 to −21.0 %), against step 5's −0.84 % on LinReg1. Rule:
wd 0.3 (seed-mean validation ES 1.3209; wd 1 1.3293). Least squares refitted on 1–8.2 TU scores 2.6567 there against the
start's 3.4700: LinReg1_nt's clean deficit to least squares is large, and the network takes most of it. Refit, 10–100
TU: the sample mean's error over LinReg1_nt's 0.945 0.952 **0.595 0.554** 0.955 0.870; sd / error 0.77–0.91; correlation
off Σ by 0.44. **Answer: yes, trained on the recorded history the generator rebuilds most of the least-squares map on
LinReg1_nt** (the middle band's error drops to 0.55–0.60 of LinReg1_nt's).

Response to its own draws (`gen_owndraws.jl`; the fit exported as the NOT-deployed `closures/diag_LinReg1_nt_CNN_gen_rec`,
translation 1.9e-6; 10–100 TU, 3 paths, 16-draw means; units LinReg1_nt's clean rms):

| per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32] | error, recorded history | error, own draws in the history | drawn noise sd | response |
|---|---|---|---|---|
| LinReg1_nt + MVG | 1.000 (all) | 1.11 1.08 1.31 1.49 1.06 1.12 | 1.07 1.02 1.29 1.46 1.05 1.12 | 0.46 0.39 **0.66 0.76** 0.35 0.46 |
| clean-history generator | 0.96 0.96 0.60 0.56 0.97 0.88 | 1.11 1.12 1.24 1.11 1.17 1.26 | 0.84 0.84 0.47 0.47 0.85 0.81 | 0.75 0.77 **2.34 2.06** 0.85 1.16 |
| (LinReg1 + MVG, step 5's table) | | | | 0.89 0.92 2.45 2.37 1.00 1.28 |

The LinReg1_nt row reproduces `tab: own-draws` (0.66 / 0.76). **The clean-trained generator responds to its own draws as
LinReg1 does (2.34 / 2.06 against 2.45 / 2.37)**: it rebuilds LinReg1's map, and with it LinReg1's response, so run
coupled it would be step 5 on LinReg1 again. The paper session's reason for A2 holds.

### s5n: A2 on LinReg1_nt (desktop grid 16:16–16:31; `gen_a2_LinReg1_nt_wd<wd>_s<s>`, logs `tmp/step5s_a2/`)

- **Grid** (8 weight decays × 3 seeds, own noise, held-out stage 1): wd 0–0.3 and 3–10 stay close to the start (best at
  200–2 300 updates, own-noise check −0.1 to −0.4 %, one seed slightly above: −0.0 to +1.1 % on the recorded history);
  **wd 1 learns, all seeds** (best at 14 400 / 17 600 / 22 300 updates). Rule (exact seed-mean validation ES, fits.jld2):
  **wd 1** (1.7793; wd 3 1.9661, wd 10 1.9661, the rest 1.975–1.976). As in step 5, wd 1 is the only weight decay at
  which the network moves far.
- 🔴 **How wd 1 learns under A2:** L shrinks (stage 1, L sd / L₀ in the middle and small bands 0.55–0.88) and the mean
  moves towards least squares (stage-1 error of the mean over the start's 0.69–0.94 in the middle band); seed 3 goes
  furthest (L 0.55–0.60, error 0.69–0.74, the network's share of the middle-band variance ~1.0). With A2 the injected
  history noise is L ε, so a smaller L means less noise in the training history, which lets the network rebuild more of
  the least-squares map: the drift the disclosure ("only L ε is injected") is about. The validation loss compares
  iterates each under its own (shrinking) noise, so part of the wd-1 gain is that self-reference.
- **Check, wd 1** (each model under its own history noise, paired; the A2 PASS rule): seed 1 **3.8951 against 4.1493
  (−6.13 %, PASS)**, seed 2 −8.25 %, seed 3 −15.83 %; on the recorded history −4.0 / −5.4 / −9.2 %.
- **Candidate `LinReg1_nt_CNN_gen_nt` = `step5/gen_a2_LinReg1_nt_wd1_s1`** (wd 1, seed 1, refit 14 400 updates). Exported
  (desktop build copy `step5b_build`, Float64): Ws vs LinReg1_nt's own correction 1.2e-13; translation check max 9.2e-7,
  rms 1.8e-7 target sd; `StochLSTM_seed1.jld2` 164 115 bytes, sha256 `f9b72b4bdf1e9867a1a27706533c4ea0a67b5692e59f3ce93ce731a3f9e21dbb`
  (desktop = laptop).
- **Refit, 10–100 TU** (`gen_offline.jl`): the sample mean's error over LinReg1_nt's 1.017 1.016 1.010 1.017 1.009
  1.001 (the mean is LinReg1_nt's; a 32-draw mean alone adds ~1.016); L L' sd / Σ sd 0.97 1.00 0.88 0.86 0.89 0.83;
  samples' correlation within 0.087 of Σ (L L' 0.225); the network's share of the variance 0.01 0.00 0.09 0.09 0.02
  0.07. **The spread does not follow the level**: per decile of q*/q_ref flat (Z[7,15] 0.96 at every decile, against the
  power law 0.36 → 1.05 and LinReg1_nt's residual 0.39 → 1.16), so error / sd runs 0.47 → 1.24 across the deciles
  (step 5 on LinReg1: spread 0.50 → 1.14). Overall error / sd 0.97 1.00 0.82 0.73 1.03 1.01 (over-dispersed in the middle
  band on the recorded history, as LinReg1_nt's own Σ is, being its residual under noise).
- **Response to its own draws** (`gen_owndraws.jl`; units LinReg1_nt's clean rms):

  | per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32] | error, recorded | error, own draws | drawn noise sd | noisy error / noise | response |
  |---|---|---|---|---|---|
  | LinReg1_nt + MVG | 1.000 | 1.11 1.08 1.31 1.49 1.06 1.12 | 1.07 1.02 1.29 1.46 1.05 1.12 | 1.04 1.06 1.02 1.02 1.01 1.01 | 0.46 0.39 0.66 0.76 0.35 0.46 |
  | s5n generator | 1.03 1.03 1.03 1.04 1.02 1.02 | 1.14 1.11 1.33 1.51 1.08 1.12 | 1.08 1.05 1.27 1.44 1.01 1.02 | 1.06 1.06 1.05 1.05 1.07 1.10 | **0.57 0.52 0.75 0.84 0.49 0.58** |

  The response stays near LinReg1_nt's (middle band 0.75 / 0.84 against 0.66 / 0.76), far below LinReg1's 2.45 / 2.37 and
  the clean-history diagnostic's 2.34 / 2.06: **A2 keeps the skip's weak response, as expected.** The residual met with
  its own draws is 1.05–1.10 × the drawn noise (LinReg1_nt 1.01–1.06).
- Against "Expected": response as expected; the offline gain (−6.1 %) larger than step 5's (−0.84 %), as expected; the
  spread does NOT follow the level (step 5's did), not expected; L narrows only 0–17 %.

- **s5n stays on A2** (Rik, 2026-11-11): its L keeps 0.83–1.00 of Σ (refit) and the network's share is ≤ 0.09, so the
  injected L ε is nearly all of the noise it draws; A2 and A2c coincide there in effect.

### s5r: A2 on LinReg7 (desktop grid 16:30–17:02; `gen_a2_LinReg7_wd<wd>_s<s>`, logs `tmp/step5s_a2/`)

- **Grid:** every weight decay learns far (check against the held-out start, LinReg7 refitted on 1–8.2 TU: −19 to −54 %;
  least squares on 1–8.2 TU 1.4010 against the start's 3.5516: the ridge's clean deficit is large). Rule (exact):
  **wd 0.3** (seed-mean ES 0.8943; wd 1 0.9249; the rest 1.145–1.523). Seed 1 best at 41 500 updates.
- 🔴 **A2 degenerates here: L collapses.** wd 0.3, stage 1: L sd / L₀ 0.27 0.46 **0.014 0.014** 0.16 0.16 (seeds 2, 3
  alike); the refit's L L' sd / Σ sd 0.28 0.45 0.06 0.09 0.10 0.10. The network carries the sample variance through its
  response to ε (share 0.74–1.04; 10–100 TU up to 1.8), so almost no noise is injected into the training history
  (η = L ε) and the fit is close to clean-history training: it rebuilds the ridge's deficit (stage-1 mean error over the
  start's 0.16–0.17 in the middle band; refit, 10–100 TU, over LinReg7's 0.58 0.91 **0.21 0.21** 0.62 0.49, as k2's
  0.59 0.92 0.22 0.19 0.54 0.43). This is the disclosed limitation of A2 at its worst: the state-dependent part of the
  closure's noise is not injected, and the optimiser moves the noise into it.
- Check (own noise, paired): seed 1 1.7309 against 3.9949 (−56.7 %, PASS); recorded history −52.7 %.
- **Exported `LinReg7_CNN_gen_nt`** (`gen_a2_LinReg7_wd0.3_s1`; desktop, Float64): Ws vs LinReg7 1.2e-14; translation
  1.0e-6 / rms 2.4e-7; sha256 `6b986c749c4a8090448ed7bd117f5294db3564a793deb9aff731c2c9d9aa0783`. 🔒 **Not submitted;
  kept as a diagnostic** (Rik, 2026-11-11, via the paper session: "A2c, check B too"; s5r is rebuilt with A2c, below).
  Disclosed in the paper as the reason for A2c.
- Refit, 10–100 TU: spread flat in the level (Z[7,15] 0.15–0.18 of LinReg7's constant sd at every decile, the power law
  0.51 → 1.36); error / sd 1.10 1.12 1.25 1.12 1.38 1.18 (narrower than its error on the recorded history already).
- **Response to its own draws:**

  | per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32] | error, recorded | error, own draws | drawn noise sd | noisy error / noise | response |
  |---|---|---|---|---|---|
  | LinReg7 + MVG | 1.000 | 1.11 1.05 1.16 1.13 1.11 1.09 | 0.95 0.96 1.00 0.94 0.97 1.00 | 1.16 1.10 1.16 1.21 1.14 1.10 | 0.49 0.34 0.58 0.56 0.50 0.43 |
  | s5r generator | 0.59 0.92 0.21 0.22 0.62 0.49 | 0.69 1.05 0.32 0.31 0.69 0.62 | 0.54 0.84 0.17 0.20 0.46 0.42 | **1.28 1.26 1.87 1.57 1.49 1.46** | **0.76 0.69 1.45 1.17 0.71 0.95** |

  The response doubles in the middle band (1.45 / 1.17 against 0.58 / 0.56), still below LinReg1's 2.45; with its own
  draws in the history its error is 1.3–1.9 × the noise it draws, so the coupled short-lead ensembles should be too
  narrow. Against "Expected": the mean correction is as expected (k2-like); the weak response is NOT kept.

### s5k: B on k1's frozen mean, A2 (desktop grid 16:59–17:16; `gen_b_k1_wd<wd>_s<s>`, logs `tmp/step5s_b/`; the old k1,
step-4 z0 verdict KEEP 2026-11-11)

- **Grid:** wd 0–0.3 and 3–10 stop early (best at 0–2 500 updates) and FAIL the recorded-history check (+0.9 to +3.8 %,
  a noise-trained fit scored clean); **wd 1 learns, all seeds** (best at 6 800 / 20 500 / 24 100). Rule (exact):
  **wd 1** (1.8352; the rest 1.943–1.971).
- **Check, wd 1** (own noise, paired, against k1's stage-1 network + k1a's Σ): seed 1 **3.8199 against 4.0997 (−6.83 %,
  PASS)**, seeds 2 / 3 −5.0 / −8.0 %; recorded history −0.8 / −1.2 / −1.9 %.
- 🔒 **Escape check (Rik: "A2c, check B too"): no escape, B stays on A2.** wd 1 stage 1, L sd / L₀: seed 1 0.94 1.03 0.92
  0.89 0.85 0.87 (seeds 2–3 0.66–1.04); the network's share of the variance 0.05 0.00 0.18 0.19 0.06 0.05 (seeds 2–3
  ≤ 0.26). Refit: L L' sd / Σ_k1a sd 1.01 1.03 0.98 0.94 0.95 0.88; on 10–100 TU the network's share 0.04 0.00 0.10 0.21
  0.02 0.05; the injected noise (L) over the noise the closure draws (own-draws table) 0.97 0.97 0.89 0.88 0.91 0.83.
  Against s5r's A2 (L 0.014, share ~1): no collapse.
- **Candidate `k1_CNN_gen_nt` = `step5/gen_b_k1_wd1_s1`** (refit 6 800 updates). Exported with `meannet=
  LinReg1_nt_CNN_crps_nt_headcorr`: frozen mean + Ws z against k1a's deployed mean on 300 steps **0.0**; translation
  8.0e-7 / rms 1.5e-7; sha256 `72f026f745616957e2149bfad2a946e4380090a17d754fbedbb3624faf959a6c`. 🔴 **Its deployment needs
  this card's `src/ts_lstm.jl` (`Cv.mean_net`)**: an older checkout loads the key and ignores it, i.e. deploys a
  different closure silently. Snellius must run at a commit that has it.
- Refit, 10–100 TU (units: k1a's constant sd / k1a's residual): the spread does not follow the level (Z[7,15] 1.04–1.07
  at every decile; the power law 0.39 → 1.10), error / sd 0.48 → 1.29 across the deciles; overall error / sd 0.96 1.00
  0.85 0.72 1.05 0.96; correlation within 0.148 of Σ.
- **Response to its own draws** (comparator k1a = k1's mean + its Gaussian; units k1a's clean rms):

  | per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32] | error, recorded | error, own draws | drawn noise sd | noisy error / noise | response |
  |---|---|---|---|---|---|
  | k1a (`LinReg1_nt_CNN_crps_nt_headcorr`) | 1.000 | 1.13 1.08 1.35 1.51 1.08 1.15 | 1.05 0.99 1.18 1.40 0.97 1.01 | 1.08 1.09 1.14 1.08 1.11 1.14 | 0.51 0.42 0.77 0.81 0.42 0.57 |
  | s5k generator | 1.03 1.03 1.09 1.07 1.03 1.01 | 1.15 1.12 1.44 1.60 1.11 1.17 | 1.09 1.04 1.29 1.50 1.00 1.07 | 1.06 1.08 1.11 1.07 1.10 1.09 | 0.58 0.55 0.81 0.87 0.53 0.64 |

  s5k is close to k1a: about the same response (middle band 0.81 / 0.87 against 0.77 / 0.81), a slightly wider noise
  (1.29 / 1.50 against 1.18 / 1.40), a residual under its own draws a little closer to its noise (1.06–1.11 against
  1.08–1.14), and a mean slightly worse in the middle band (1.07–1.09 × on the recorded history, 1.07 × with its own
  draws). Against "Expected": about equal to k1a, as expected; the state-dependent spread did not appear.

### B smoke (desktop, 400 updates, old k1 = `skipnth_LinReg1_nt_crps_wd1_s1`; a code test, not B's training)

The frozen k1 stage-1 and refitted networks reproduce its `pred.jld2`; the start check passes; OWNCHECK −1.15 % (the
recorded-history check +1.9 %, as expected of a noise-trained fit); export `smoke_k1_CNN_gen` (NOT deployed): frozen mean
+ Ws z against k1a's deployed mean on 300 record steps **0.0**, translation check 6.8e-7 target sd.

## Post hoc

- 2026-11-11, **s5n submitted** (Rik, asked "Copy LinReg1_nt_CNN_gen_nt to Snellius and submit its training hindcast +
  smoke on gpu_a100?": "Submit and chain production"). Snellius at `d9c576dd` (has the generator, `cb189c9f`; s5n needs no
  new src). Artefact copied laptop → Snellius, sha256 `f9b72b4b…1dbb` equal. Jobs (gpu_a100): training hindcast
  `29131087` (`run_d6_train_lstm.sh step3/closures/LinReg1_nt_CNN_gen_nt`), long-run smoke `29131088`; ⚠️ production
  chained `--dependency=afterok:29131087:29131088`, i.e. submitted BEFORE the defect rule (as step 5): full hindcast
  `29131089` (`run_d6_packed.sh`, array 1–10, `D6_OUT=output/D6_LinReg1_nt_CNN_gen_nt`), long runs (`run_online.sh lstm 2
  <r>`, `-t 00:30:00`) r = 1–5: `29131090`, `29131092`, `29131093`, `29131094`, `29131096`. The defect rule is checked on
  the training hindcast as soon as it lands; if it flags, Rik decides whether to cancel.
