# Results — M4, the stochastic LSTM, on HIT

Companion to [`results.md`](results.md) (M0 and the DDN). **Where a number here disagrees with a
design document, this file is the measurement.** M4 is `plan.md` §3's off-grid exploratory cell —
the stochastic LSTM of Barthel Sørensen et al. — and enters no attribution difference.

🔑 **This file was rewritten on 2026-09-24 to keep only what still stands.** The round-by-round
record — superseded protocols, the GPU investigation, scans run under flawed batching — is in git
history (last full version: commit `973e2d9a`). Everything below is either a definition, the
protocol as it is now, or a result that the current conclusions rest on.

**Status, 2026-09-29 (§10–§13; plan.md *Start here* has the order).**
- 🏁 **Finalist: `Splice1_E0x7_ar2`** — LinReg1 with LinReg7's (λ = 1) E[0,6] coefficient row and
  an AR(2) residual. It is linear; no network. At M = 10 on all 46 selection ICs it is **level with
  LinReg1 on CRPS (−0.02 %, CI ±4.2 %)** and ahead on calibration (30/30 vs 24/30 cells) and
  stability (tail 13 vs 22, gate 14 vs 26) (§13g). It goes to the confirmation block, scored once.
- **Nothing beats LinReg1 on skill.** No single λ + AR(2) does (§13e). Paper 3's per-QoI penalty
  rule is level at best, with worse calibration (§13i). The lever is **per-QoI λ + a coloured
  residual**: ridge helps only E[0,6], and ridge + white noise is always under-dispersed.
- **Nonlinear mean (M3ᶠ): closed at this data volume.** The MLP (§11) and the LSTM-memory variant
  (§13d) both return update 0.
- **Nonlinear noise (M0ᵛ): the only resolved nonlinear gain**, −8.9 % against its matched M0, but on
  an h = 2 mean 30 % behind LinReg1 (§13h). Next: the LSTM scale on the finalist's mean.
- ⚠️ M = 5 mini-D6s are too noisy for cells this close (§13g). Screen finalists at M = 10.

**Status, 2026-09-24.**
- Offline, the protocol is settled and the best fit (stride 100, `:storn`, 10 000 updates) is the
  best M4 model on held-out data by a wide margin (§5).
- 🔴 **Online, every fit that predicts the LEVEL `q` fails the same two ways** — a hard ceiling on
  the top band and a spurious flat state — and neither `beta` nor rollout fine-tuning fixes it (§6).
- ✅ **Predicting the CORRECTION instead removes both** (§7). With the latent entering the decoder
  as well as the cell (`:vrnn`), the correction's persistence also matches the reference's. What
  remains is over-dispersion. §7 is a screen — single short CPU runs — not a result yet.

---

## 1. The model

Four variants, one flag apart. With `x_t` the regressor row and `y_t` the output:

```
z_t  ~  q(z_t | x_t) = N( mu_z , sigma_z )     mu_z = B_mu e_t ,  sigma_z = softplus(B_sig e_t)
e_t  =  tanh( W_e x_t + b_e )                  dense encoder; n_encoder = 0 drops it (used here)
h_t  =  LSTM( [x_t ; z_t] , h_{t-1} )
y_t  =  V1 h_t  +  V2 z_t  +  c                linear output
```

| `arch` | `z` drawn | `z` → cell | `z` → decoder (`V2`) | is |
|---|---|---|---|---|
| `:lstm` | no | no | no | deterministic LSTM |
| `:vaernn` | yes | no | yes | noise at the output only |
| `:storn` | yes | yes | no | STORN — "upstream" noise only |
| `:vrnn` | yes | yes | yes | VRNN — upstream and at the output |

- 🔴 **The first line is the encoder `q(z|x_t)`, not the prior**; the prior is `N(0, I)`. The encoder
  sees only `x_t` — settled against the reference implementation (`ben-barthel/learning_dynamics`).
- **Dimensions:** input 19 (`q*^n`, `q^{n-1}`, `q*^{n-1}`, bias; `h = 1`, `N_Q = 6`), hidden **16**,
  latent **4**, no encoder layer — **~2 950 parameters** against 21 594 training target values. The
  source's 60/60/60 would be 43 128, twice the data; we test their architectural claim at `N_Q = 6`,
  not their model.
- **Emission head** (`emission`): `:none` (default — the decoder is deterministic and `z` is the only
  noise), `:constant`, or `:state_dependent` (`Sigma = D R D`, `log d` linear in `h_t`).
- **What the output means** (`scaling.target`, carried with every fit, read by the deployed closure):

| target | network predicts | closure returns `dQ =` |
|---|---|---|
| `:q` (every fit before 2026-09-23) | the level `q^n` | `qhat - q*` |
| `:dQ` | the recorded correction `dQ^n` | the prediction itself |
| `:logr` | `r = log1p(dQ / q*)` | `q* (exp(r) - 1)` — the level stays positive |

The inputs, including the level history, are the same for all three; only the output head differs.

## 2. The objective

```
L  =  sum_t [  -log p( y_t | mu_y(h_t, z_t), Sigma )  +  beta * KL( q(z_t|x_t) || N(0, I) )  ]
```

- `emission = :none`: the first term is a plain sum of squares — the source's objective. With a
  head it is a Gaussian log-density, so losses are on a different scale and never share a column.
- ⚠️ **`beta`, never `lambda`** — `lambda` is this project's ridge parameter.
- The ELBO uses one `z` draw; validation holds its noise draws fixed so the curve is a function of
  the parameters alone.
- Training carries a free precision factor for `Sigma`; deployment a correlation matrix. V41 checks
  the two give the same density.

## 3. Training protocol (as it stands)

| | |
|---|---|
| data | R1's tracking record, `train_range = (400, 4000)` (1–10 TU); trailing 20% of **rows** is the inner validation block (early stopping only) |
| segments | `L = 500`, `burn = 100` (unscored warm-up), **stride 100**, **full-length segments only, tiled from the END** of the block |
| batching | 32 segments, uniform random, last partial batch dropped (`batch_eff = min(32, nseg)`) |
| optimiser | Adam, `lr = 1e-2`, ×0.3 on a plateau of 20 validations **counted from the last decay**, floor **`min_lr = 1e-4`** |
| validation | every 2 updates; everything schedule-related is paced in **updates** |
| stopping | at `min_lr` and 100 validations without a new best, **or** < 0.5% gain over the last 500 updates; else the epoch cap |
| returned | the best-validation iterate (a warm start is validated at update 0 and can be returned) |
| device | **CPU** — one M4 update is a chain of ~20 000 dependent kernel launches; the GPU was measured ~7x slower than the same node's CPU, and no differentiable fused RNN exists in Julia |

`L` and `burn` come from the level's ACF crossings (1/e at 116–142 steps; ringing period ~400),
not from `T_int`. Precision: Float32 model, Float64 solver, converted at the closure's two
boundaries only.

**Fixes that change how older numbers read** (each has a test):

| fix | what it did before | test |
|---|---|---|
| split by row, not by segment list | at any stride < `L - burn`, validation rows were also training rows | V53 |
| full-length segments only | a lone short tail segment was batched alone and owned up to half the updates | V53/V55 |
| tile from the end | the dropped short segment was the rows next to validation (379 rows at stride 400, 19 at stride 20) | V39 |
| plateau counted from the last decay | a loss spike cascaded the rate to its floor in 160 updates | V56 |
| windowed stop + `min_lr` 1e-4 | a run at 1e-5 improved 36% more over 27 000 updates and never stopped | V57 |
| `:constant` emission really constant | `Wd` was trained under `:constant`, so it was state-dependent — the `:lstm` control `StochLSTM7` included | V59 |

## 4. Scoring

- Offline: the reconstruction term (`beta = 0`) on a **common validation set** (rows 2980:3599, the
  post-split fits' early-stopping set — it ranks, it does not test), and SSE on the **held-out
  window** (steps 4000–7600, teacher-forced, deployed step, one draw).
- ⚠️ **Step 6501 dominates the held-out error**: truth at the 96–100th percentile of all six QoIs
  at once; ±100 steps around it is 5.7% of the window and 17–58% of every model's squared error.
- Online: free-running in the solver (`12_online_StochLSTM.jl`). 🔴 **Summed KS on the level cannot
  carry the comparison** — the reference's own 10 TU windows span 0.38–1.32 (#61), and fig15's
  4 TU flat episode scored inside that band. The diagnostics that discriminate:

| diagnostic | on Z[16,32] unless named | reference, 100 TU | reference, ten 10 TU windows |
|---|---|---|---|
| **flat** | fraction of 0.5 TU windows with sd < 30% of the reference's median 0.5 TU sd | 3.6% | 0–7.4% |
| **>2900** | fraction of time above 2900 | ~5% | 0–12.9% |
| max / min | range reached | 3794 / 721 | 2780–3790 / 721–1270 |
| sd ratio | against the reference's 100 TU sd | 1 | 0.62–1.19 |
| **`dQ` lag-1** | lag-1 autocorrelation of the correction on E[0,6] | 0.743 | **0.72–0.77** |

## 5. Offline: the stride scan and held-out skill

`:storn`, `beta = 1e-4`, seed 1 (cell `StochLSTM2`). Rows 1, 3, 4 hit their 3000-update cap; stride
100 was re-run to 10 000 updates (still improving 2% per 500 at the end).

| point | updates | common-val recon | held-out SSE | SSE excl. ±100 of 6501 |
|---|---|---|---|---|
| **stride 100, b32, 10 000 updates** | 10 000 (cap) | **2.89e-4** | **3.63** | **1.96** |
| stride 100, b32 | 3000 (cap) | 4.97e-4 | 5.73 | 3.54 |
| stride 400, b32 | 3000 (cap) | 5.32e-4 | 7.75 | 4.16 |
| stride 50, b32 | 3000 (cap) | 6.22e-4 | 7.81 | 4.42 |
| pre-split `:storn` (flawed protocol) | 3000 | 6.22e-4 | 8.67 | 4.56 |
| pre-split `:vrnn` | 3000 | 6.93e-4 | 6.63 | 5.48 |
| `:lstm` control | — | (other units) | 15.59 | 12.34 |

- Held-out error falls with validation loss (5.73 → 3.63), so the longer run generalises rather than
  overfits.
- ⚠️ **The stride question is not settled**: only stride 100 got 10 000 updates, and at 3000 stride 50
  was worse than 400. The `(400, b2)` control cannot be read — its run was cut by a decay cascade
  (since fixed), and its 10 000-epoch re-run crawled at the old 1e-5 floor
  ([fig14](figures/fig14_lstm_b2_long_run.png)).
- `beta = 1e-2` (cell `StochLSTM3`, stride 100): held-out SSE 4.41, 21% worse than `1e-4`.

## 6. Online with the level target: a ceiling and a flat state

Cluster runs, 100 TU, 5 replicas (3 for `beta = 1e-2` teacher-forced); replica `i` shares IC, OU
forcing and latent seed across models.

![M4 online, 10 TU](figures/fig15_lstm_online_s100.png)
![M4 online ensembles, 100 TU](figures/fig16_lstm_online_ensembles.png)

| | flat | longest flat | summed KS | sd ratio Z16 | `dQ` lag-1 | Z16 max |
|---|---|---|---|---|---|---|
| `beta 1e-4` | 19–35% | 2.7–6.3 TU | 0.90–1.21 | 0.92–0.96 | 0.93–0.94 | 2868–2919 |
| `beta 1e-4`, rollout fine-tune | 16–20% | 3.7–7.5 TU | **0.82–0.97** | 0.90–0.94 | 0.96 | 2880–2901 |
| `beta 1e-2` | 2.5–5.2% | 0.6–0.9 TU | 1.46–1.61 | 0.79–0.89 | 0.94 | 2858–2871 |
| `beta 1e-2`, rollout fine-tune | 0.7–2.6% | 0.3–0.8 TU | 1.59–1.75 | 0.66–0.77 | 0.97 | 2874–2913 |
| LinReg1 | 3.3–5.7% | 0.6–1.0 TU | 0.59–1.01 | 1.19–1.42 | 0.76 | 3398–4652 |
| reference | 3.6% | — | — | 1 | 0.743 | 3794 |

- ✅ **Numerically stable**: every replica finite over 100 TU, clamp never fired.
- 🔴 **`beta = 1e-4` is bistable** — every replica switches between an active state and a flat one
  near Z16 ≈ 1600. `beta = 1e-2` removes the flat state but narrows the marginal.
- 🔴 **All 22 M4 replicas cap Z16 at 2858–2919** where the reference reaches 3794 and LinReg1 4652.
  Ruled out: the architecture's bound (`c ± Σ|V1|` = 4647); missing data (trained targets reach 3097);
  representation (teacher-forced, every fit reaches 3019–3068 at the 6501 excursion); the hidden
  state's age (teacher-forced error does not grow past the trained 500 steps). **It is the closed
  loop**: with its own history fed back, the model damps excursions.
- **The exposure, measured offline**: scored with its own output fed back into the level lags, the
  stride-100 fit's validation loss is **3.4x** its teacher-forced one (already at a 100-step horizon).
- **Rollout training** (`train_stochlstm(...; rollout = K)`, `tools/m4_rollout_train.jl`, V58) feeds
  the model's output back into the level lags after the burn-in, `q*` replayed. As a fine-tune
  (`L = 200`, `K = 100`, lr 1e-4, ~1400 updates) it cut the rollout loss 9–14% with teacher-forced
  unchanged — **but online it moves only the marginals** (better at `1e-4`, narrower at `1e-2`) and
  touches neither the ceiling nor the correction's persistence.

## 7. Online with the correction as the target — screening (2026-09-23/24)

🔴 **A screen, not results**: one seed, fits from scratch at the §3 protocol (they stop at the
floor after ~220–280 updates, but the **returned iterate is update 18–70** — see §7b), one 10 TU replica each, run on this workstation's CPU (a different backend from
the cluster runs: statistics, not trajectories, compare). Judged against the reference's own 10 TU
windows (§4). Fits by `tools/m4_explore_fit.jl`, scored by `analysis/m4_screen.jl`.

**Why the target**: a level-predicting network has to produce the level itself, and in the closed
loop it will not climb; predicting the correction lets the level come from the solver,
`q = q* + dQ`, with the network supplying a bounded adjustment (Rik). Offline this also removes the
exposure: the fed-back level is `q* + dQ`, so the model's own error barely enters its input
(**rollout/teacher-forced 1.00x**, against 3.4x for the level target).

| 10 TU, replica 1 | flat | >2900 | Z16 max | Z16 min | sd ratio | summed KS | `dQ` lag-1 | clamp |
|---|---|---|---|---|---|---|---|---|
| **reference, 10 TU windows** | **0–7.4%** | **0–12.9%** | **2780–3790** | **721–1270** | **0.62–1.19** | **0.38–1.32** | **0.72–0.77** | 0 |
| level target, `:storn` (cluster baselines, first 10 TU) | 0–42% | 0% | 2822–2868 | — | 0.69–1.00 | 0.86–2.61 | 0.91–0.98 | 0 |
| level target, `:vaernn` | 7.0% | 0% | 2801 | 1559 | 0.58 | 2.32 | 0.950 | 0 |
| `dQ`, `:storn`, beta 1e-4 | 6.4% | 14.8% | 5171 | **4** | 1.63 | 0.51 | 0.971 | 28 |
| `dQ`, `:storn`, beta 1e-2 | 2.3% | 20.1% | 6520 | 680 | 1.98 | 0.68 | 0.978 | 0 |
| `dQ`, `:storn`, beta 0.1 | 3.4% | 16.3% | 5097 | 412 | 1.66 | 0.53 | 0.947 | 0 |
| `dQ`, `:storn`, state-dependent emission | 5.5% | 17.6% | 4749 | 233 | 1.81 | 0.68 | 0.035 | 34 |
| `dQ`, `:storn`, constant emission | 2.2% | 28.2% | 8982 | 441 | 2.98 | 1.21 | 0.070 | 0 |
| **`dQ`, `:vrnn`, beta 1e-4** | **0.0%** | 16.9% | **4181** | **521** | **1.51** | 1.07 | **0.741** | **0** |
| → the five pending variants were screened at **20 TU × 3 replicas** instead: §7b | | | | | | | | |

1. ✅ **Predicting `dQ` removes the ceiling and the flat state together**: Z16 spends 15–28% of the
   time above 2900 (the level target: 0%), flat time is inside the null, and summed KS is the best
   of any M4 run (0.51). Mean offsets across the six QoIs are the smallest of any closure (+0.01 to
   +0.23 sd for `:storn`, `beta 1e-4`).
2. 🔴 **But it over-corrects.** Every `dQ` variant is wider than the reference (sd ratio 1.5–3.0). With
   noise in the cell only (`:storn`) it also fails downward: one run drained the top band — Z16 to 4
   against a reference minimum of 721, E[16,32] below the `1e-2` gate, the clamp firing 28 times. Its
   deployed correction is close to a conditional mean (dQ sd 0.21–0.55 of the reference's in the four
   lower bands; the network explains ~43% of the `dQ` variance).
3. 🔑 **Where the noise enters is the lever.**
   - cell only (`:storn`): the correction is far too persistent (0.95–0.98), at any `beta` up to 0.1;
   - white noise at the output (emission heads): far too white (0.04–0.07), and the widest runs;
   - **cell and decoder (`:vrnn`)**: **0.741, on the reference's 0.743**, and the only `dQ` fit with no
     drain, no clamp and the smallest overshoot.
4. 🔴 **The `dQ`/`logr` fits OVERFIT within tens of updates** — they are early-stopped, not converged.
   Validation bottoms out at update 26–70 and then rises while training keeps falling (`dq_storn`:
   best 1.70 at update 26, 2.30 at 200 with training at 1.10); the plateau rule then walks the rate
   to its floor and the stop fires ~200 updates later, so the deployed model is the update-26–70
   iterate. The level target keeps improving for 10 000 updates. Reason: the correction is mostly
   unpredictable (~43% of its variance explained) and the learnable part is found in ~30 updates;
   after that ~2 950 parameters memorise the 8 TU training block's noise. **Regularisation (smaller
   `n_hidden`, larger `beta`, weight decay, lower `lr`) or more training data is therefore a lever
   for this target in a way it was not for the level.**
5. What remains is **the upward swing — under-damped excursions** (the level target erred the other
   way). The multiplicative target `:logr` is the attack: its correction scales with `q*`, and its
   level cannot go negative.

### 7a. Regularisation against the correction target's overfitting (2026-09-24, laptop, branch `m4-regularization`)

⚠️ **Run in parallel with §7b–§7j on the desktop and merged afterwards; read it in that light.** Its
fits are on the architecture WITHOUT the linear skip, and §7c finds that architecture never learned
the linear part — which is the deeper reason these fits overfit, and why the ranking below says little
about the skip models. Its **weight decay is the UNCOUPLED form** (`decay_couple = false`: each update
shrinks every weight by `wd`, whatever the rate); the desktop fits (`m4_diag_fit.jl`,
`m4_window_fit.jl`) use the coupled default (`lr * wd`). The same `wd` is not the same decay across the
two — at `lr = 1e-2`, uncoupled `1e-3` is coupled `0.1`.

Offline only. `dQ` target, `:vrnn`, `L = 500`, seed 1, from scratch; each fit scored by the
**held-out R² of the correction** (`m4_heldout_skill`: teacher-forced, latent at its mean, per QoI,
on a window disjoint from training and from the stopping set). New: `train_stochlstm(...;
weight_decay, decay_couple = false)` (V74); `m4_explore_fit.jl` overrides `NHIDDEN`, `WD`, `TRAINRANGE`.
Figure: `analysis/plot_m4_reg_loss.jl` → [fig17](figures/fig17_lstm_reg_loss.png).

![loss decay, regularisation sweep](figures/fig17_lstm_reg_loss.png)

| fit | best at update | held-out R², steps 4000–7600 | lower four bands | top two | R², steps 12400–16000 |
|---|---|---|---|---|---|
| baseline (`n_hidden` 16) | 70 | 0.127 | −0.40 … −0.07 | 0.93 / 0.92 | 0.176 |
| `n_hidden` 10 | 70 | 0.172 | −0.40 … −0.03 | 0.93 / 0.91 | |
| **`n_hidden` 6** | 74 | **0.274** | **−0.06 … +0.01** | 0.89 / 0.83 | |
| **`n_hidden` 6 + wd 1e-3** | 74 | **0.279** | −0.05 … +0.01 | 0.89 / 0.83 | **0.291** |
| beta 1e-2 | 70 | 0.119 | −0.40 … −0.07 | 0.93 / 0.91 | |
| beta 1 | 110 | −0.028 | −0.64 … −0.11 | 0.90 / 0.90 | |
| wd 1e-4 / 1e-3 / 1e-2 | 70 / 84 / 132 | 0.124 / 0.126 / 0.129 | ≤ −0.03 | 0.93 / 0.92 | |
| lr 1e-3 | 384 | 0.179 | −0.30 … −0.06 | 0.88 / 0.91 | |
| lr 1e-3 + wd 1e-3 | **2434** | 0.262 | −0.07 … −0.01 | 0.85 / 0.89 | 0.230 |
| `:storn` + wd 1e-3 | 26 | 0.237 | −0.13 … −0.04 | 0.88 / 0.86 | |
| **training range 1–30 TU** (a protocol change) | 92 | — (inside its training range) | | | **0.337** (lower −0.05 … +0.15, top 0.93 / 0.93) |
| *level target, 10 000 updates (for scale)* | 9998 | *0.475* | *0.45 0.34 −0.04 0.37* | *0.94 / 0.78* | |

- 🔑 **The overfitting is capacity against data.** `n_hidden` 6 roughly doubles held-out skill
  (0.127 → 0.274; 0.176 → 0.291 on the second window) by taking the four lower bands from clearly
  negative to about zero, at a small cost in the top two; the train/val gap all but closes.
  **Three times the data does best** (0.337, lower bands positive, top bands kept at 0.93) — but
  moves the training range off the project's 1–10 TU partition, so it is Rik's call.
- **`beta` and weight decay alone do nothing** (beta 1 is worse); the best iterate stays at ~70.
- ⚠️ **lr 1e-3 + wd 1e-3 does not overfit, but for the wrong reason**: validation falls to update
  2434 (the sweep's lowest, 1.55) while training loss RISES from ~600 — the per-update decay is not
  scaled by the learning rate, so once the rate decays the decay dominates and shrinks the network.
  Regularisation by attrition; wd 1e-2 shows the same, stronger. (The coupled default avoids this.)
- None reaches the level target's lower-band skill (0.34–0.45): that part of the correction is
  learnable, the correction target does not learn it from 8 TU of data.
- **Next candidate, as written on the laptop**: `n_hidden` 6 (+ wd 1e-3) on 1–30 TU, then the online
  screen. ⚠️ Superseded as a plan by §7c–§8, which put the linear skip in first and found the
  recurrence then adds no residual mean on 1–10 TU. §7c scores held-out LOSS (0.5 × SSE, lower is
  better) on 50–100 TU, not this section's R², so the two tables do not compare number for number.

### 7b. 20 TU × 3 replicas on a GPU (2026-09-24)

Same fits, run on the shared desktop's RTX 3090 (`12_online_StochLSTM.jl`, CUDA, `RIKFLOW_ONLINE_IC`),
scored by `TSCREEN=20 m4_screen.jl` against the reference cut into five disjoint 20 TU windows.

🔑 **Why 20 TU, not 10**: 11% of the reference's 10 TU windows never exceed 2900 — a capped run can pass
the ceiling test by chance — against 0 of 81 sliding 20 TU windows. Over 20 TU the reference's flat
fraction is ≤ 6.9%, sd ratio ≤ 1.18 and `dQ` lag-1 0.727–0.753 (5th–95th percentile of sliding windows),
tight enough to fail every mode seen so far. Resolving an sd ratio of 1.05 against 1.2 needs ~50 TU;
that is the refit's job, not the screen's.

| 20 TU, replicas 1–3 | flat | >2900 | Z16 max | Z16 min | sd ratio | summed KS | `dQ` lag-1 | clamp |
|---|---|---|---|---|---|---|---|---|
| **reference, five 20 TU windows** | **0.2–6.9%** | **1.9–12.2%** | **3100–3790** | **721–1080** | **0.86–1.14** | **0.27–0.77** | **0.726–0.750** | 0 |
| `dQ`, `:vrnn`, beta 1e-4 | 0.8–2.5% | 7.5–10.9% | 3750–3815 | 419–743 | 1.26–1.37 | 0.69–0.99 | 0.61–0.67 | 0 |
| `dQ`, `:storn`, L 200 | 0.0–4.0% | 14.0–19.5% | 4903–6568 | 205–760 | 1.73–1.91 | 0.84–1.04 | 0.98 | 0 |
| `logr`, `:storn`, beta 1e-4 | 1.1–6.8% | 3.2–6.4% | 3353–4550 | 534–703 | 1.20–1.45 | 1.51–1.71 | 0.98 | 0 |
| `logr`, `:storn`, state-dependent emission | 0.0–0.8% | 8.5–13.8% | 3644–4323 | 580–692 | 1.34–1.51 | 1.06–1.16 | 0.03–0.04 | 0 |
| `logr`, `:vrnn`, beta 1e-4 | **NaN at t ≈ 8.45 TU in all three replicas** | | | | | | | |

1. 🔴 **No variant passes.** Every surviving fit is over-dispersed in both tails (sd ratio ≥ 1.20, a
   minimum below the reference's 721 in 11 of 12 runs).
2. 🔴 **The `:vrnn` + `dQ` lead does not hold up.** Its 0.741 lag-1 was one 10 TU replica: split by time,
   all three GPU replicas start near the reference (0.52–0.72 over 0.25–5 TU) and lose persistence
   later (0.60–0.65 over 10–20 TU, reference 0.72). One CPU replica cannot exclude a backend effect;
   replica 3's 0.515 in its first 5 TU says realisation spread is enough. Its correction on E[0,6]
   has 0.24–0.29 of the reference's `dQ` sd (CPU and GPU alike) — §7's conditional-mean finding.
3. 🔴 **`logr` does not fix the dispersion** (1.20–1.51, no better than `dQ`), and **with `:vrnn` it
   blows up**. The trigger is shared: all replicas replay the reference's OU forcing, which drives a
   large-scale excursion at t ≈ 8 (E[0,6] at z = 2.0 in the reference itself). The `dQ` fit's
   correction stays negative there and the level recovers by t = 8.25; the `logr` fit's correction on
   E[0,6] turns positive and grows with the level (+1.3 → +4.4 sd in 0.25 TU), and the run overflows
   0.2 TU later. Consistent with a biased multiplicative correction, `q*·(e^r − 1)`, feeding back in
   proportion to the level — a reading, not tested.
4. 🔑 **Persistence follows where the noise enters, not the target** — now for both targets: cell
   only (`:storn`) 0.98, emission head 0.03–0.04, cell + decoder (`:vrnn`) 0.61–0.67 (`dQ`).
5. ⚠️ **These are barely-trained networks.** Every fit's best validation came at update **18–70**,
   still at `lr = 1e-2`; thereafter training loss falls (to 1.1–1.5) while validation rises, the
   schedule decays to the floor and the stop rule fires ~200 updates later. `logr_b1e-4` and
   `logr_b1e-4_ems` are 18–20 full-batch steps from their initialisation. 24 segments from 9 TU of
   record is little data; more updates will not help, more data or regularisation might.

### 7c. Why the fits overfit: they never learned the linear part (2026-09-24)

Diagnosed with `tools/m4_diag_fit.jl`: every checkpoint scored on a **held-out 50–100 TU window**
(one teacher-forced pass, latent at its mean, 100-step warm-up), in the loss `elbo` trains on
(0.5 × SSE per step over the six QoIs, standardised `dQ`), beside a **least-squares map on the
same regressor**. 16 fits per round on the desktop CPU, ~2–10 min each.

🔴 **The LSTMs sat at the h = 0 linear floor.** A least-squares map on the fits' own inputs
(h = 1: `q*^n`, `q^{n-1}`, `q*^{n-1}`, bias) scores **0.284** held out, R² 0.93 / 0.57 / 0.96 /
0.98 / 0.995 / 0.99, with coefficients up to **119** — a precise cancellation between the level
lag and the predictor (`sd(q)/sd(dQ)` is 16–97). Without the lag (h = 0) it scores **1.75**. Every
M4 fit so far scored 1.6–2.1: **they never learned to use the lag**, fitted what `q*` alone gives,
then overfitted the rest. More data does not change that (no-skip `:vrnn` on 1–50 TU: 1.70 against
a floor of 0.262), so it is structural: the output `V1 tanh(·)` has no linear path from the input.

🔑 **Fix: a linear skip `y += Ws x`** (`LSTMSpec(; skip = true)`, V60), seeded with the
least-squares map and `V1 = V2 = 0`, so update 0 IS the linear model; `Ws` is then **frozen**
(`freeze = (:Ws,)`) and the recurrence models the residual. Freezing is a convenience, not a
necessity: joint and frozen `:vrnn` end at the same held-out 0.291 (best checkpoint 0.287). At the
least-squares optimum the `Ws` gradient is ~0 (norm 9e-3) but Adam normalises it, so every early
step moves each coefficient by ~`lr` and costs ~+0.03 held out; freezing avoids that.
⚠️ **Correction (same day):** the round-1 `:vrnn`/`:vaernn` skip fits were seeded with `V1 = 0` only,
so their decoder skip `V2 z`, `z = Bmu x`, added a random linear map: the seed scored **3.01**, not
0.284, and the drop "0.284 → 1.14 in two updates" first reported here was training recovering from
that, not destroying the seed. `:storn`/`:lstm` fits (incl. the lead candidate) were seeded exactly;
the driver now zeroes `V2` as well. Their end points are unaffected (the table below).

| round 1: residual mean (`:none` emission) | held out (linear floor) |
|---|---|
| no skip, 1–10 TU / 1–50 TU | **2.089** (0.284) / **1.704** (0.262) |
| skip, 1–10 TU: `:vrnn` `:storn` `:vaernn`, `n_hidden` 8–32, `L` 100–1000, `lr` 1e-2/1e-3, weight decay, joint or frozen | **0.285–0.303** (0.284) |
| skip, h = 2 | 0.252 (0.249) |
| skip, 1–50 TU | **0.253** (0.262) — the only fit that beats its floor, still improving at the stop |

**On 1–10 TU the recurrence finds no residual mean at all**; architecture, capacity and sequence
length are irrelevant once the linear part is in. With 5× the data it finds 3.4%.

**What the linear map leaves is noise** — held out: residual sd 0.07–0.27 of `dQ`'s in five
QoIs and **0.66 in row 2**, nearly white (lag-1 0.04–0.22, rows 3–4 0.77/0.87 decaying in ~5
steps), correlated within band pairs (0.90 / 0.67 / 0.89), excess kurtosis 0.7–1.5 — M0's η, not
something a mean model can predict. That is why a latent-only fit (`:none`) returned a correction
with 0.25 of the reference's sd online.

| rounds 2–3: noise heads on the frozen skip, 1–10 TU | held-out NLL / step | vs linear + constant η |
|---|---|---|
| linear mean + closed-form residual covariance (M0's structure) | −4.564 | — |
| constant head, trained from Σ = I | −3.1 to −3.7 | worse: the head needs ~440 updates, the mean overfits in ~100 and the schedule stops it |
| constant head seeded at the residual covariance (`SEEDHEAD`) + LSTM mean | −4.564, best iterate = update 0 | 0: the LSTM mean never helps |
| **linear mean + LSTM-driven state-dependent scale (`lin_sd`)** | **−4.881** | **+0.32** (h = 2: +0.40; 1–50 TU: +0.41) |
| LSTM mean + state-dependent scale | −4.82 to −4.87 | +0.25 to +0.30 (`L` 200–1000, `n_hidden` 16/32 alike) |
| `:storn` / `:vrnn` latent + state-dependent scale | −4.83 / −4.55 | +0.27 / 0 |

🔑 **Offline the well-trained M4 is: linear mean, learned state-dependent noise scale** — the L2
lever, learned by a recurrence. A latent path and a residual mean only cost, at this data volume.

### 7d. Online: the linear skip fixes persistence; the closed-loop bias is the mean map's (2026-09-24)

20 TU on the desktop's RTX 3090, scored against the reference cut into 20 TU windows (`m4_screen.jl`,
`M4_SCREEN_SUBDIR=diag`) plus per-QoI moments: the mean offset in reference sd (reference 20 TU
windows: −0.21…+0.29), the sd ratio, and `sd(dQ)` against the reference correction's. All `dQ`
target, trained on 1–10 TU. Linear-plus-η models (`rdg_h<h>_l<λ>`) are closed-form ridge fits
deployed through the same closure (`:lstm`, skip, `V1 = 0`, head seeded at each fit's own residual
covariance) — M0's structure in the M4 code path.

| model (replicas) | stable | mean offset | sd ratio | dQ lag-1 | >2900 | KS |
|---|---|---|---|---|---|---|
| **reference** | | **−0.2…+0.3** | **0.86–1.14** | **0.73–0.75** | **1.9–12.2%** | **0.27–0.77** |
| `:vrnn` latent on the skip, with or without an emission head (6) | **0/6** — NaN at t = 5–13 | | | | | |
| linear + LSTM-driven noise scale, h = 1 (3) | 3/3, clamp 34× in one | +0.4…+1.7 | **1.7–2.4** | 0.75–0.78 | 19–38% | 1.5–2.2 |
| linear + η, h = 1, λ = 0 (3) | 3/3 | **+0.8…+1.1** | 1.04–1.43 | 0.77–0.79 | 19–33% | 2.0–2.3 |
| — same, noise × 0.5 / noise ≈ 0 (3+3) | 6/6 | +1.0…+1.5 / **+0.8…+1.2** | 0.87–1.10 / 0.76–0.94 | 0.86 / 0.97 | 17–36% | 2.3–3.1 |
| linear + η, h = 1, λ = 3e-6 (2) | 2/2, clamp 120× in one | 0…+0.57 | 1.42–1.45 | 0.77 | 6–16% | 0.75–0.91 |
| **linear + η, h = 1, λ = 1e-5 (2)** | 2/2 | **−0.25…+0.12** | 1.33–1.44 | **0.74–0.75** | **9–11%** | **0.67–0.97** |
| linear + η, h = 1, λ = 3e-5 / 1e-4 / 3e-4 / 1e-3 (2 each) | 8/8, clamp at 1e-3 | −0.3…0 → −0.7…−0.3 | 1.5 → 2.3 | 0.71 → 0.23 | 9–15% | 1.0–1.7 |
| linear + η, h = 2, λ = 0 (3) | 3/3 | −0.3…+0.6 | 1.08–1.46 | 0.76–0.78 | 10–19% | 0.91–1.11 |
| linear + η, h = 3 / 5 / 10, λ = 0 (3 each) | 9/9 | +0.2…+2.0 / +1.0…+2.3 / +0.8…+1.6 | **0.81–1.20** | 0.76–0.79 | 22–53% | 2.7–3.9 |
| 🏆 **M4: linear skip h = 2 + LSTM-driven noise scale (`r3_lin_sd_h2`) (3)** | **3/3**, clamp 0 | −0.65…+0.36 | **1.02, 1.02**, 1.44 | 0.76–0.78 | **2.8, 5.6**, 14.8% | **0.47, 0.64**, 0.95 |

1. ✅ **The skip fixes what every earlier M4 fit got wrong**: the correction's persistence (lag-1
   0.74–0.79 against the reference's 0.73–0.75; the old fits read 0.03 or 0.98) and its size
   (row 2's `sd(dQ)` 0.99–1.09 with η, against 0.25 latent-only).
2. 🔴 **A `:vrnn` latent on the skip diverges** at the shared t ≈ 8 forcing excursion — 6/6, with
   or without an emission head. Its encoder `μ_z = B_μ x` is linear in the raw inputs, so an
   excursion off the training range drives it without bound (a reading, not tested).
3. 🔴 **The best offline model is worse online** (offline→online rank reversal): the LSTM-driven
   noise scale (+0.32 nats held out) over-disperses 1.7–2.4× in closed loop, plausibly an excursion
   → larger scale → larger excursion feedback that teacher-forced data cannot show.
4. 🔑 **The +1 sd bias of the linear map is the MEAN, not the noise**: it is unchanged at noise
   × 0.5 and ≈ 0. The noise sets the spread and the correction's whiteness — calibrated η gives the
   reference's lag-1 (0.78 vs 0.74); without it 0.97 and row 2 at 0.55 of its sd.
5. 🔑 **Ridge λ is a bias dial on the `dQ` target and it crosses zero near λ = 1e-5** (penalty
   `λ N`, standardised design): +1 sd at 0, −0.5 sd at 1e-3, monotone, while spread grows and
   persistence falls with λ. The same dial `results.md` §4b found for LinReg1 — and gotcha #63's
   round-off-regularised LinReg1 (h = 5, runs 6–9% low) sits on the other side of zero from the
   exact h = 5 fit here (+1 to +2.3 sd high).
   🔑 **Joint training does not choose λ for you.** From an exact seed a jointly trained skip stays
   on the unregularised least-squares map (`:storn` 0.03 output-sd from it, nearest ridge λ = 0;
   `:lstm` + η returns update 0) — the LS map IS the optimum of the teacher-forced one-step loss,
   while λ ≈ 1e-5 is set by the closed-loop bias, which no one-step objective contains. Only an
   objective through the solver (L5) or an online choice of λ (online selection — Rik's call) can.
6. 🔑 **Deeper lags get the spread right (sd ratio ≈ 1 at h ≥ 3) but not the bias**, which is not
   monotone in h (+1, +0.4, +1.8, +2, +1.5 sd at h = 1, 2, 3, 5, 10): a high-gain cancelling map
   (max |C| 119 → 2527) is sensitive in closed loop to small differences between fits. λ acts on the
   gain directly; the combination deep h + small λ is the unrun next step (set F/G below).

7. 🏆 **The first M4 variant that passes the screen: `r3_lin_sd_h2`** — frozen least-squares skip
   at h = 2, head seeded at the residual covariance, an LSTM (16 hidden, `:lstm`, `V1 = 0` frozen)
   driving only the log-scale `Wd h + bd`; best iterate at update 36. In **2 of 3 replicas** it sits
   inside the reference band on flat time, the ceiling (2.8 / 5.6%), both tails (Z16 3183–3272 /
   765–808 — the first M4 run above the reference's minimum of 721), the sd ratio (1.02) and summed KS
   (0.47 / 0.64); the misses are row 2's mean (−0.5 sd) and a slightly persistent correction (0.76–0.78).
   Replica 1 is over-dispersed (1.44). 🔑 **Here offline and online agree**: +0.40 nats held out over
   its linear base, and better than that base online (KS 0.47–0.95 vs 0.91–1.11) — whereas at h = 1
   the same head over-dispersed. The learned noise scale helps once the mean is good enough.

⚠️ **All of this is 20 TU, 2–3 replicas, one seed** — a screen. Shared GPU; noise-free timings are
not quoted.

### 7e. Window mode: short q*-only windows, reset every prediction (2026-09-24, branch `m4-short-lstm`)

The model (Rik): input `x_t = [q*_t; 1]` only (h = 0), `LSTMSpec(; window = W)`; the deployed closure
resets the state every step and replays the last `W` inputs, each with its OWN latent draw -- `eps_t`
is drawn once, when step `t` is first predicted, and reused in the `W - 1` later windows that replay it
(V61). Training: `L = W`, `stride = 1`, `burn = W - 1` (only the deployed prediction is scored), every
row a window end -- 2 870 windows, 89 updates/epoch. Driver `tools/m4_window_fit.jl`; held-out 50-100
TU, same unit as §7c (0.5 x SSE per step, standardised `dQ`). One seed, CPU, ~1 min per fit.

| W = 10, 1-10 TU | held out | best at update |
|---|---|---|
| linear, `[q*_n; 1]` | 1.752 | — |
| linear, the whole stacked window `[q*_{n-9..n}; 1]` | **1.460** | — |
| (§7c: linear with the level lag `q^{n-1}`) | (0.284) | — |
| `:vrnn` 16/4, no skip | 1.971 | 120 |
| — `lr = 1e-3` | 1.978 | 620 |
| — all 10 outputs scored (`burn = 0`) | 2.080 | 120 |
| — frozen linear skip on `x_n`, `V1 = V2 = 0` seed | 1.749 | 40 |

1. 🔴 **Dropping the level lag costs a factor 5 before any network is involved**: the best linear map
   on ten `q*` steps scores 1.46 against 0.284 with `q^{n-1}`. `q^{n-1} = q*^{n-1} + dQ^{n-1}`, so what
   is lost is the previous correction (lag-1 0.74) -- not recoverable from the predictor history.
2. 🔴 **No fit reaches even the `q*_n` floor without the skip** (1.97 vs 1.75), and none reaches the
   window floor; with the skip frozen the recurrence adds nothing (1.749). Not undertraining: train
   loss falls 5.4 -> 1.25 while inner validation and held out both turn up after ~120 updates.
   Stride 1 multiplies windows, not information -- they overlap in 9 of 10 steps of one 9 TU record.
3. **The source's KL (`kl_mode = :reference`, V62) changes nothing at the same `lam`**: 1.970 / 2.079
   / 1.736 against 1.971 / 2.080 / 1.749 (last / all / skip). Its KL is SUMMED over batch x steps,
   so its weight scales with the geometry: at the source's 32 x 100 and `lam = 1e-4` a `z`-vector's KL
   weighs 0.32 against one step's SSE; at 32 x 10 with the last step scored the same `lam` gives
   0.0032. Matched to 0.32 (`lam = 1e-2` last, `1e-3` all) the posterior collapses onto the prior
   (`sigma_z` 0.99 at `lam = 1e-2`, 0.5 at 1e-4) and held out is slightly worse, 2.07 / 2.11.

### 7f. The q*-only search: increments, the skip, validation, and a conditional-VAE encoder (2026-09-24)

Plan and checklist: [`plan_m4_search.md`](plan_m4_search.md). Driver `tools/m4_window_fit.jl`, W = 10,
16 hidden / 4 latent, one seed; held out 50-100 TU in dQ units (0.5 x SSE, CRPS over 32 draws,
spread/skill = ensemble sd / rmse of the ensemble mean). Online 20 TU x 3 on the desktop 3090.

**The linear structure.** Least squares on `[q*_n, q*_{n-1}]` scores 1.458 (on `q*_n` alone 1.752);
more lags add nothing (W = 10: 1.460, W = 40: 1.565). It needs coefficients of ~130 -- the increment
of q*, tiny in scaled units -- which no network learned. `FEAT = diff` feeds the standardised
increment as a fixed invertible `input_map` (V63): the no-skip LSTM then gains (1.97 -> 1.83), but
still does not reach the linear map. 🔴 **Online, every model with the `[q*, Δq*]` skip diverged,
0/9 by step 191**, the pure linear + eta included: `q*_n = S(q*_{n-1} + dQ_{n-1})`, so the increment
carries the model's own last correction and the gain-130 map closes a runaway loop.

**Two review findings, measured.** (i) Validation blocks spread through 1-10 TU (with embargo)
select WORSE fits than the trailing block -- held out 1.95-2.02 vs 1.736 -- because they share the
training regime and do not penalise the residual the LSTM grows on the skip; kept as `VAL=blocked`,
default `tail`. (ii) Held-out inputs beyond the training range explain nothing: 3.6-4.9% of rows,
and clamping them moves the linear floor 0.5%. Held-out dQ leaves the training range on 0.08-0.8% of
steps, and the fits track those steps at least as well as the rest (slope 0.35 no skip, 0.54-0.61
with the skip). 🔑 The linear skip is the unbounded output path (`:storn`/`:lstm` are otherwise capped
at `cdec +- sum|V1|`), so it is now on by default and frozen (joint training leaves the LS map
within ~20 updates, 1.96-2.00 held out).

**The latent is not a predictive distribution (the finding of the day).** With the source's encoder
`q(z | x)`, which never sees the target, spread/skill is **0.05-0.11** at any `beta` (1e-4 or 1e-2,
sigma_z ~ 1): the decoder ignores `z` because noise only raises the squared error. CRPS 0.500 against
linear + eta's 0.378. `posterior = :xy` (V65) trains `q(z_t | x_t, dQ_t)` and deploys the prior:

| q*-only, W = 10 | held out | CRPS | spread/skill |
|---|---|---|---|
| linear + eta | 1.752 | **0.378** | 0.85-1.08 |
| `:vrnn` + skip, source encoder | 1.736 | 0.500 | 0.05-0.11 |
| `:vrnn` + skip, **conditional encoder** | 1.806 | 0.382 | 0.71-0.88 |
| — state-dependent head / `beta` 0.3 / 3 / nz 2, 8 / W 5, 20 / last-step score | 1.75-1.93 | 0.379-0.392 | 0.64-1.29 |

With level history (h = 1, `q_star_q`) the same holds: CVAE CRPS 0.1287 vs linear + eta 0.1284,
spread/skill 0.95-0.99; source encoder 0.17-0.18, spread/skill 0.01-0.20.

**Online, q* only (20 TU x 3).**

| | stable | flat | >2900 | Z16 min | sd ratio | KS | `dQ` lag-1 | clamp |
|---|---|---|---|---|---|---|---|---|
| **reference** | | 0.2-6.9% | 1.9-12.2% | 721-1080 | 0.86-1.14 | 0.27-0.77 | 0.73-0.75 | 0 |
| `:storn` + skip, source encoder | 3/3 | 1.7-3.3% | 7.5-10.7% | 511-639 | 1.22-1.33 | **0.55-0.69** | 0.997 | 0 |
| `:vrnn` + skip, source encoder | 3/3 | 1.3-2.7% | 6.0-8.4% | 521-654 | 1.22-1.27 | 0.42-1.27 | 0.92-0.94 | 0 |
| — W = 5 / W = 20 | 2/3, 3/3 | | | 207-243 | 1.4-1.8 | 1.2-2.3 | 0.96-0.98 | 18-453 |
| linear + eta / + LSTM noise scale | 3/3 | | | 160-206 | 1.4-2.0 | 2.3-3.2 | 0.32-0.57 | 242-719 |
| `:vrnn` + skip, conditional encoder | 3/3 | 1.4-4.4% | 5.0-6.8% | 162-233 | 1.78 | 1.8-2.7 | 0.43-0.62 | 344-532 |
| no skip (raw / diff) | 2/3, 3/3 | | | 120-178 | 1.8-2.7 | 1.1-1.8 | 0.67-0.79 | 72-304 |

1. 🔴 **Nothing q*-only passes.** The mean is linear-limited, and the noise is either negligible
   (source encoder: stable, clamp-free, KS in band, but the correction is `q*`'s own persistence,
   0.93-0.997, and still 1.2-1.3x too wide) or calibrated for one step (linear + eta, the
   conditional VAE), which the closed loop amplifies to 1.4-2.0x with the clamp firing -- §7d's
   finding, now for a learned latent too.
2. 🔑 The window length does not set the persistence (W = 5 / 10 / 20: 0.98 / 0.93 / 0.97), so the
   0.93 is not the noise tying.

### 7g. With level history: the conditional VAE online (2026-09-24)

Same protocol as §7f, `x_t = [q*_t; q_{t-1}; q*_{t-1}; (q_{t-2}; q*_{t-2};) 1]` (`H`, `HIST_VAR=q_star_q`),
frozen skip seeded with least squares (`LAMBDA = 0`) or the ridge map. Offline every fit sits on its
linear floor (0.284 at h = 1, 0.249 at h = 2); the conditional VAE matches linear + eta on CRPS
(0.1287 vs 0.1284; h = 2 0.1097 vs 0.1095) with spread/skill 0.95-0.99, the source encoder does not
(0.17-0.18, spread/skill 0.01-0.20). Online, 20 TU x 3, all 27 runs finite:

| | sd ratio | KS | `dQ` lag-1 | clamp | mean offset (sd) | Z16 min | >2900 |
|---|---|---|---|---|---|---|---|
| **reference** | **0.86-1.14** | **0.27-0.77** | **0.73-0.75** | 0 | **-0.2..+0.3** | 721-1080 | 1.9-12.2% |
| CVAE h = 1, lambda 0 | **0.91-1.17** | 2.1-2.4 | 0.79-0.80 | 0 | +0.8..+1.1 | 1076-1530 | 15-25% |
| CVAE h = 1, lambda 3e-6 / 1e-5 | 1.3-1.7 / 1.6-2.3 | 0.8-1.0 / 0.6-1.2 | 0.77-0.80 | 0-272 | 0..+0.6 / -0.1..+0.2 | 56-590 | 11-20% |
| linear + eta h = 1, lambda 0 / 1e-5 | 0.96-1.28 / 1.37-1.69 | 0.8-1.2 / 1.1 | 0.74-0.78 | 0 / 0-236 | +0.1..+0.7 / -0.1..-0.5 | 78-978 | 6-20% |
| **linear + eta h = 2, lambda 0** | **1.11-1.52** | **0.77-1.10** | 0.76-0.78 | **0** | +0.1..+0.5 | 562-745 | 9.6-17% |
| CVAE h = 2, lambda 0 / 3e-6 | 1.23-1.35 / 1.5-2.0 | 1.6-1.7 / 0.9-1.3 | 0.78-0.81 | 0 / 0-268 | +0.5..+0.9 / +0.1..+0.6 | 48-1010 | 14-19% |
| `:storn` h = 1, source encoder | 0.83-0.88 | 2.4-3.0 | 0.96 | 0 | +0.8..+1.2 | 1114-1408 | 12-21% |

1. 🔴 **No variant passes the screen.** Nearest: linear + eta at h = 2 -- 2 of 3 replicas inside the band
   on flat, >2900 and sd ratio (1.11 / 1.12), KS 0.77 / 0.78; misses on the minimum (562-621) and the
   persistence (0.76-0.78).
2. 🔑 **One dial governs everything: the linear mean's lambda trades closed-loop bias for spread.**
   lambda = 0 runs high (+0.5 to +1.1 sd), any ridge removes the bias and over-disperses (1.3-2.3, the
   clamp firing) -- for the CVAE and for eta alike (§7d found it for eta).
3. 🔑 **The conditional VAE is the first latent that is a real predictive distribution**, and at
   h = 1 it has the best spread of any model (0.91-1.17 in all three replicas, clamp 0) with near-right
   persistence (0.79-0.80). Online it behaves like calibrated eta, not better: its learned temporal
   structure does not remove the mean's bias, which one-step training cannot see (§7d point 5).

### 7h. Two checks before closed-loop training (2026-09-25)

**(a) A QoI surrogate of the solver is NOT a usable training environment -- no-go.**
`analysis/m4_surrogate_check.jl` linearises the LF step around the recorded trajectory,
`q*_n = q*_n^rec + sum_k A_k (q_{n-k} - q_{n-k}^rec)` (A_k by least squares on 1-10 TU), and runs the
deployed closures through it exactly as online. One step it is near-perfect (R^2 0.998-0.99997), but
its propagator has spectral radius **1.013 (p = 1) / 1.60 (p = 2)** -- it AMPLIFIES deviations the real
LES damps -- and in closed loop it gets the bias's sign wrong for 5 of 6 fits (linear + eta h = 1:
-0.9..-2.3 sd against online -0.5..+0.7; h = 2: -0.3..-1.6 against -0.5..+0.5) and the spread 1.5-5x
too large; only the h = 1 CVAE's +1.0..+1.6 matched online's +0.7..+1.1. A least-squares Jacobian of a
near-identity map estimated from natural variability is confounded by the forcing; it is not the
solver's response. 🔑 A rollout that replays the recorded q* cannot see the bias either (§7), so closed-
loop training needs either the solver's measured QoI response or the solver itself.

**(b) Resetting the state per prediction (window) beats a persistent state.** Same inputs, CVAE,
frozen skip; persistent fits by `m4_diag_fit.jl` (`POST=xy`, L 500/200), 20 TU x 3:

| | sd ratio | flat | `dQ` lag-1 | clamp |
|---|---|---|---|---|
| reference | 0.86-1.14 | 0.2-6.9% | 0.73-0.75 | 0 |
| **window W = 10, h = 1** | **0.91-1.17** | 0.2-8.1% | **0.79-0.80** | **0** |
| persistent h = 1, L = 500 | 2.2-2.6 | 0.4-3.9% | 0.88-0.89 | 700-1180 |
| persistent h = 1, L = 200 | 0.63-1.15 | 5-8% | 0.86 | 0 |
| **window W = 10, h = 2** | 1.23-1.35 | 1.4-6.3% | 0.78-0.79 | 0 |
| persistent h = 2, L = 500 | 0.38-1.28 | **9-19%** (the flat state of §6 returns) | 0.84 | 0-107 |

The persistent state makes the same model over-dispersed or erratic across replicas, and more
persistent; the reset window is consistent. The +0.5..+1 sd mean bias is common to both.
⚠️ `elbo` now takes the KL over EVERY step for `posterior = :xy` (a burn-in `z` must not code its target
for free); window fits scored all steps and are unaffected.

**(c) A rollout with REPLAYED q* is the wrong environment -- tested directly.**
`analysis/m4_replay_rollout.jl` runs the deployed closure over 50-99.5 TU with the recorded q* supplied
each step, so the level lags are the model's own `q* + dQhat` and the predictor does not respond:

| fit | teacher-forced 0.5·SSE | replayed-q* rollout (mean / seeds) | online, real solver |
|---|---|---|---|
| linear + eta h = 1, lambda 0 | 0.284 | **explodes (~1e50)** | stable 3/3 |
| CVAE h = 1, lambda 0 | 0.285 | **explodes** | stable 3/3 |
| linear + eta / CVAE h = 2, lambda 0 | 0.249 / 0.250 | **NaN** | stable 3/3 |
| linear + eta h = 1, lambda 1e-5 | 0.341 | 2.76 / 3.8 | stable 3/3 |
| CVAE h = 1, lambda 1e-5 | 0.364 | 3.73 / 4.7-4.8 | stable 3/3 |

🔑 The fits' key feature is the cancellation `a q*_n + b q_{n-1}` (|a|, |b| ~ 119), i.e. the increment
`q*_n - q_{n-1}`, which is meaningful only because the solver makes `q*_n = S(q_{n-1})`. Replay `q*`
and the cancellation partner is gone: `q_n = q*_n + dQhat(q_{n-1})` has a loop gain of ~119. The
earlier "1.00x exposure" (§7) was measured on fits that ignored the lag (§7c) and does not transfer.
So closed-loop training needs a q* that RESPONDS to the correction: a surrogate of the solver.

**(d) An MLP surrogate on the current step only (Rik's spec) is unstable -- no-go.**
`analysis/m4_mlp_surrogate.jl`: `q*_{n+1} = q*_n + dQ_n + Delta`, `Delta ~ N(mu(u), sigma(u)^2)`,
`u = [q*_n; dQ_n]`, an MLP (2 x 64 tanh) and a linear head, Gaussian NLL on 1-10 TU. The MLP overfits
(best val NLL at epoch 200: -2.33 against -4.81 on training); in closed loop it **diverges in 18 of 18
runs within 240-1160 steps**, and does so even with the record's own corrections replayed (dmean -55 to
+82 sd): as a dynamical system it has no attractor. The linear head diverges in 14 of 18; its 4
survivors carry the online bias's sign (h = 1 CVAE +1.3..+1.7 against GPU +0.8..+1.1). 🔑 A map on the
six QoIs integrates its own step error; what holds the LES on its attractor lives in the unresolved
field and the forcing. The online bias is an equilibrium shift of exactly that kind.

**(e) The solver's MEASURED response to a correction (38 GPU replay runs).** `12_online_StochLSTM.jl` with
`RIKFLOW_M4_NWARM` >= the run length only replays the recorded dQ; `RIKFLOW_PERT=j,m,delta` adds a sustained
step (±0.5 sd(dQ_j) from step m, m = 200 / 1000 / 1800, 400 steps); `analysis/m4_response.jl` reads q* =
q[:, n+1] − dQ[:, n]. The GPU is deterministic (two identical baselines agree exactly) and the replay
tracks the record to ≤ 3e-3 q-sd over 2000 steps.
- 🔑 **The solver passes a correction on ~1:1.** One-step kernel diagonal 0.98 / 1.02 / 0.82 / 1.04 / 0.51 /
  1.03; the step response grows ~linearly (S_k ≈ k in four bands to k = 10–25): an integrator.
- Restoring appears only after ~50–100 steps in bands 1, 3, 5; bands 2, 4, 6 still hold 30–115 (per unit
  step) at k = 200. At k = 200 the spread across start times rivals the signal and the symmetric part is
  17–27% of the response (not linear).
- A linear-response surrogate on this kernel truncated at 200 steps has no restoring force and diverges
  (18/18 at lambda = 0, most at 1e-5). Not a training environment -- but it explains §7d/§7g: a constant bias
  in the correction is summed over the restoring time into a large LEVEL offset, and white one-step noise is
  summed into over-dispersion.

### 7i. Closed-loop calibration against the training period, and the dense VAE (2026-09-27)

Rik allowed closed-loop calibration against **training-period statistics only** (the level over 1-10 TU).
Tools: `scaling.dq_offset` (a per-QoI constant added to the deployed correction), `scaling.noise_scale`
(scales every latent and emission draw, window mode; V67), `tools/m4_offset_variant.jl`,
`tools/m4_calibrate.jl` (level Jacobian from +-offset runs, damped Newton step), `analysis/m4_calib_score.jl`.

**Why ridge + a noise scale cannot do it.** On the h = 1 CVAE, lambda removes the bias (|mean error| over
QoIs: 0.93 at 0, 0.16 at 3e-6, 0.10 at 1e-5) but the level's sd grows (1.0 -> 1.3 -> 1.8 x training), and
scaling the noise by 0.7 or 0.5 barely changes that at lambda > 0: the over-dispersion is DYNAMICAL -- the
shrunk map damps the loop less -- not the noise. An offset moves the mean without touching the gain.

**The level Jacobian (+-0.05 sd(dQ) per QoI, replica 1):** entries 5-64 q-sd per dQ-sd, cond 623, signs
physical (an enstrophy correction in the smallest bands lowers every level); -0.5 sd(dQ) offsets diverged,
and so did -0.05 in QoI 1. **The Newton step** is <= 0.03 sd(dQ) (bands 5-6), 0.002-0.006 elsewhere.

| h = 1 CVAE (`b_xy_h1`) | mean error vs 1-10 TU per QoI (sd) | sd / training sd |
|---|---|---|
| uncalibrated | +0.85..+1.10 | 0.86-1.13 |
| offset x 0.5 | +0.32..+0.55 | 0.90-1.40 |
| **offset x 1 (`xy1_N1`)** | **-0.10..+0.13 (all 3 replicas)** | 1.07-1.42 |
| offset x 1.5 | -0.35..-1.22 | 1.41-1.86 |

Reference-band screen, 20 TU:

| `xy1_N1` | flat | >2900 | Z16 max / min | sd ratio | KS | `dQ` lag-1 | clamp |
|---|---|---|---|---|---|---|---|
| **reference** | 0.2-6.9% | 1.9-12.2% | 3100-3790 / 721-1080 | 0.86-1.14 | 0.27-0.77 | 0.73-0.75 | 0 |
| r1 | 4.1% | 7.1% | 3952 / 191 | 1.35 | **0.69** | 0.790 | 177 |
| r2 | 5.8% | 10.3% | 3405 / 823 | 1.19 | **0.52** | 0.771 | 0 |
| r3 | 3.3% | 9.2% | 3585 / 794 | 1.18 | **0.43** | 0.784 | 0 |

🔑 **KS inside the reference band in all three replicas -- a first for any model in this search.** r2/r3 are
in band on everything but the sd ratio (1.18-1.19 vs <= 1.14) and persistence (0.77-0.78 vs <= 0.75).

**More replicas, and the drain.** Replicas 4-5 of `xy1_N1` (not used in the calibration) sit -0.4..-0.8 sd
low with the clamp firing; with r1, 3 of 5 replicas fall into a DRAINED state (Z16 min 166-281 vs the
reference's 721). A constant offset adds dissipation to bands 5-6 at every state, pushing low excursions into
collapse. **A state-proportional offset** (`scaling.offset_ref`, `c .* q* ./ mean_{1-10 TU}(q*)`, V68) -- the
same `c`, shrinking as a band drains -- removes it:

| `xy1_R1_s0.8` (state-proportional offset, noise x 0.8), 5 replicas | in band |
|---|---|
| flat 3.0-5.5% / >2900 3.1-9.3% / clamp 0 | 5/5 / 5/5 / 5/5 |
| **KS 0.36-0.66** | **5/5** |
| sd ratio 0.96-1.18 | 4/5 |
| Z16 max 3140-3825 / min 434-965 | 4/5 / 3/5 |
| **`dQ` lag-1 0.79-0.80** (reference 0.73-0.75) | **0/5** |

Training-period score |mean error| 0.09 sd, |log sd ratio| 0.06. The calibrated h = 2 linear + eta is worse
(full step drains, clamp > 1700; damped step sd 1.14-1.70). Persistence is the remaining gap.

**The fair comparison: calibrate the linear model the same way** (h = 1, state-proportional offset from its
own Jacobian, noise x 0.8; 20 TU x 5):

| | sd ratio | KS (in band) | Z16 min | `dQ` lag-1 | clamp |
|---|---|---|---|---|---|
| **calibrated deep (CVAE)** `xy1_R1_s0.8` | 0.96-1.18 | 0.36-0.66 (**5/5**) | 434-965 | 0.79-0.80 | 0 |
| calibrated linear + eta `le1_R1_s0.8` | 1.02-1.32 | 0.53-1.54 (3/5) | 256-807 | 0.78-0.80 | 0 |
| calibrated linear + AR(1) eta | 1.08-1.28 | 0.58-1.50 (2/5) | 257-706 | 0.81-0.83 | 2/5 |
| LinReg1-equivalent h = 5, one step (+1.2..+2.8 sd before) | 1.09-1.22 | 1.63-2.12 (0/3) | 251-302 | 0.77-0.78 | 2/3 |

🔑 **Calibration closes most of the gap; the deep model stays ahead** (KS in band 5/5 vs 3/5, tighter spread,
better lower tail). Most of the "improvement over LinReg" is the calibration, a smaller part the model. The
h = 5 map starts so far off (+1.2..+2.8 sd) that one Newton step overshoots into the drained state; a second
step is running.

**Over 100 TU the difference is clear** (`analysis/m4_screen_long.jl`, 3 replicas each, both calibrated the
same way on 1-10 TU):

| 100 TU | whole-run KS | sd ratio | mean offset vs the 100 TU reference (sd) |
|---|---|---|---|
| **calibrated deep** `xy1_R1_s0.8` | **0.35-0.49** | 1.12-1.22 | **+0.01..+0.11** |
| calibrated linear `le1_R1_s0.8` | 0.89-1.34 | 1.13-1.18 | −0.36..−0.64 |

🔑 The linear calibration does not hold beyond the 20 TU it was tuned on (it drifts half an sd low); the deep
one does. A second Newton step brings the h = 5 LinReg1-equivalent to KS 1.06-1.17, sd 1.09-1.31, >2900
10.7-18.2%, clamp 0 -- better, still behind.

**Seed robustness -- negative.** The recipe (h = 1 CVAE, its own Jacobian, one Newton step, state-proportional
offset, noise x 0.8) was repeated on training seeds 2 and 3. Offline the three fits are identical (held out
0.285-0.286, CRPS 0.128-0.129); online they are not:

| 20 TU x 5 | |mean err| / |log sd err| | behaviour |
|---|---|---|
| seed 1, own step (`xy1_R1_s0.8`) | 0.09 / 0.06 | healthy 5/5 |
| seed 2, own full step | 1.53 / 0.67 | 4/5 lock HIGH (Z16 never below its initial 1887, sd ratio 0.4-0.5) |
| seed 2, own damped step (mu = 100) | 0.27 / 0.39 | over-dispersed 1.3-1.7, clamp 5/5 |
| seed 2, seed 1's offset transferred | 0.84 / 0.63 | drains (Z16 min 51-129), clamp |
| seed 3, seed 1's offset transferred | 0.87 / 0.56 | locks high / drains / diverges |
| seed 3, own Jacobian | -- | cannot be formed: both +-0.05 runs in QoI 4 diverge; base replica 1 off by 4.4 sd |

🔴 **The closure is multistable** -- a healthy state, a drained one and a high-locked one -- and which basin a
fit lands in depends on sub-0.05 sd(dQ) differences in the correction and on the training seed; the level
Jacobian is ill-conditioned (cond ~1000-2800) and strongly nonlinear. **Seed 1's success is one basin, not a
recipe.** Any claim of the calibrated CVAE over LinReg needs a calibration that finds the healthy basin
reliably (a line search along the Newton direction is being tried on seed 2), or it rests on one seed.

**Also this round:** integrated regression (the skip fitted on k-step sums, `SUMK`) diverged 9/9 online;
**the dense VAE** (`arch = :dense`, V66: a 2-layer MLP on the window, no recurrence) matches the LSTM offline
exactly (floor 0.285, CRPS 0.1287) and online within noise (sd ratio 0.95-1.51, KS 1.7-2.1 uncalibrated) --
the recurrence neither helps nor hurts.

### 7j. White vs coloured noise -- and a model that LEARNS the colour (2026-09-27)

`analysis/m4_noise_colour.jl`: the closure run teacher-forced (K = 20 noise seeds) over a window; the
model's NOISE is each draw minus the ensemble mean, the data's RESIDUAL is the recorded dQ minus it. The
residual ACF is the colour the data asks for, the noise ACF the colour the model makes. Parameters are
fitted on the training window only (`RIKFLOW_COLOUR_TRAIN=1`); held out 50-100 TU confirms.

**The data's residual is coloured in the middle bands**, white elsewhere (training 1-10 TU, h = 1 mean):

| | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| residual ACF lag 1 / 2 / 10 | 0.11 / 0.12 / −0.02 | 0.04 / 0.04 / 0.00 | **0.73 / 0.40 / 0.00** | **0.82 / 0.66 / −0.23** | 0.02 / −0.11 / 0.05 | 0.18 / 0.02 / 0.05 |

**Every model so far makes WHITE noise** (ACF 0.00 at every lag, one-step spread calibrated 0.90-0.99):
linear + eta, the window CVAE with a tied latent, the untied CVAE, and the dense CVAE. 🔑 The tied window
gives the CVAE the MEANS for colour but no INCENTIVE: with an i.i.d. prior the conditional encoder explains
each step's residual by that step's own z, so past z's are useless to the decoder; and ~all of its noise
turns out to be the white emission head (colouring only the head reproduces the target colour).

**Imposed colour: AR(1) on eta** (`scaling.eta_ar`, V69), a = the training residual's lag-1 per QoI -- on
the linear mean this is M0^c (plan L7). It matches lags 1-2 (Z[7,15] 0.73 / 0.53 vs 0.73 / 0.40) but not
the negative lobe (lag 5: +0.21 vs −0.19).

**Learned colour: the VRNN prior** `p(z_t | h_{t-1})` (`LSTMSpec(; prior = :learned)`, V70), trained with
KL(q(z_t | x_t, dQ_t) || p(z_t | h_{t-1})). With an emission head the white head still absorbs the residual
(noise ACF 0); without one, beta = 1 and 0.1 collapse the posterior (spread 5-19% of the residual's). At
**beta = 0.001, no head (`lp_none_b0.001`)** it learns the colour:

| held out 50-100 TU | Z[7,15] lag 1 / 2 / 5 / 10 | E[7,15] lag 1 / 2 / 5 / 10 | white bands lag 1 | spread vs residual |
|---|---|---|---|---|
| data residual | 0.80 / 0.49 / −0.03 / 0.02 | 0.88 / 0.73 / 0.25 / −0.11 | 0.04-0.32 | 1 |
| **learned prior** | **0.75 / 0.50 / −0.01 / −0.13** | **0.81 / 0.61 / 0.15 / −0.13** | 0.00-0.31 | 0.75-0.91 |
| same model, standard prior (control) | −0.24 / −0.19 / 0.01 / 0.00 | −0.23 / −0.16 / 0.01 / 0.00 | −0.23..−0.38 | 1.5-3.0 |

🔑 **The learned prior reproduces the colour -- including the negative lobe AR(1) cannot -- and keeps the
white bands white; the identical model with the i.i.d. prior makes anti-correlated noise 1.5-3x too wide.**
Offline cost: mean 0.303 vs 0.285, CRPS 0.132 vs 0.128 (one-step metrics cannot see colour). Online: running.

**Online, 20 TU x 5 (uncalibrated unless named):**

| | sd ratio | KS | `dQ` lag-1 | clamp |
|---|---|---|---|---|
| **reference** | 0.86-1.14 | 0.27-0.77 | 0.73-0.75 | 0 |
| linear + white (M0) | 0.96-1.28 | 0.84-1.92 | **0.770-0.781** | 0 |
| linear + AR(1) (M0^c) | 0.99-1.24 | 1.06-2.07 | 0.804-0.812 | 0 |
| deep + white | 0.91-1.17 | 2.0-2.4 | 0.774-0.802 | 0 |
| deep + AR(1) | 0.97-1.28 | 2.1-2.4 | 0.804-0.820 | 0 |
| **calibrated deep + white** (`xy1_R1_s0.8`) | 0.96-1.18 | **0.36-0.66** | 0.79-0.80 | 0 |
| calibrated deep + AR(1) | 1.03-1.38 | 0.33-0.78 | 0.811-0.830 | 2/5 |
| deep + LEARNED colour (`lp_none_b0.001`) | 1.68-2.73 | 1.48-1.79 | 0.77-0.84 | 4/5 |

🔴 **Colour does not help online, imposed or learned.** AR(1) raises the correction's persistence (+0.03)
without improving KS; the learned-colour model is 1.7-2.7x too wide. 🔑 A reading consistent with §7h(e): the
solver passes a correction on ~1:1, so a persistent noise is SUMMED into the level -- with lag-1 0.8 its
summed variance is ~(1+a)/(1-a) ~ 9x a white one's. The residual's colour in the record is the colour of the
correction the TRACKED run needed, conditional on its state; reproducing it unconditionally over-disperses.
**Calibrating the learned-colour model does not rescue it**: every variant (full Newton step at noise x 1 /
0.7 / 0.5, a damped step) drains -- Z16 minimum 19-100, clamp 785-3823, sd ratio 2.2-3.2. It reproduces the
colour offline and runs away online: its coloured innovations are summed by the solver. A negative result
for the paper, with a mechanism.

**The factorial** (h = 1, identical inputs):

| mean \ noise | white | coloured, imposed (AR(1)) | coloured, learned |
|---|---|---|---|
| linear | `b_lin_eta_h1` (M0) | `colour/lin_h1_ar` (M0^c) | -- |
| deep (CVAE) | `b_xy_h1` / calibrated `calib/xy1_R1_s0.8` | `colour/xy_h1_ar` / `colour/xy1_R1_s0.8_ar` | `lp_none_b0.001` |

## 8. Where it stands, and what is open

**2026-09-29 — superseding the list below** (details §13):
- 🏁 **Finalist `Splice1_E0x7_ar2`** (§13g): level with LinReg1 on CRPS, ahead on calibration and
  stability. **Open (Rik):** when to spend the confirmation block (76–97 TU) on it, with LinReg1
  alongside, at full D6 density.
- **Next cell: M0ᵛ on the finalist's mean**, i.e. the LSTM noise scale (§13h's −8.9 %) on the h = 5
  LinReg1/splice mean instead of the h = 2 skip. It needs the skip seeded from a `LinReg.jld2`.
- **Closed:** the nonlinear mean at h = 2 (§11, §13d); single λ + AR (§13e); lag-1-exact AR(2)
  (§13c); taking Z[0,6] or the small-scale rows from ridge (§13f); AR(1).
- **Open, cheap:** the paper-3 base-λ scan's last point (LinReg16, §13i); the M = 5 low-priority
  cells (LinReg2 ± AR, LinReg8 ± AR), which only extend the λ ladder.
- `r3_lin_sd_h2` below is no longer the lead candidate. Its noise-scale idea survives as M0ᵛ.

**2026-09-24 (superseded):**

- 🏆 **Lead candidate (2026-09-24): `r3_lin_sd_h2`** — linear skip at h = 2 + LSTM-driven noise scale
  (§7c/§7d). Passes the 20 TU screen in 2 of 3 replicas; no earlier variant passed at all. The old
  architecture (no skip) never learned the linear part (§7c) — every §7/§7b fit sat at the h = 0 floor.
- **Before any M4 number is quoted**: refit it properly (5 seeds, S6), then 100 TU × 5 replicas
  against LinReg1 and the DDN.
- **Next screens** (built or trivially buildable with `tools/m4_diag_fit.jl` and the λ recipe in §7d):
  the (h, λ) grid around the bias zero-crossing (h = 2 at λ = 3e-6 / 1e-5, h = 5 at λ = 3e-6 / 1e-5 /
  3e-5, h = 10 at 1e-5) — then the LSTM noise scale on the best of those bases.
- **Open, Rik's call**: the calibrated η over-disperses in closed loop wherever the mean is weak;
  a noise amplitude fitted *online* would fix it but is online model selection, which the protocol
  rules out (decisions log 2026-08-14/17).
- ✅ **Why every fit overfit after 18–70 updates** (§7b point 5): answered in §7c — they never learned
  the linear map, so there was nothing left to fit but noise. `:logr` is answered (§7b point 3).
- **Open**: whether rollout training helps the `dQ`/`logr`
  targets online (offline exposure is already 1.00x, so it would have to act through the solver's
  response, which the replayed-`q*` surrogate cannot see); the stride question at matched budgets;
  the §6.3 capacity × `beta` × latent grid, which should now be run on the winning target.
- **The emission-head bug** (§3) means `StochLSTM7`, the `:lstm` control, was fitted with a
  state-dependent head; refit it before quoting it as a constant-noise control.

## 9. How to reproduce

```bash
# from lib/RikFlow. Training uses the `training` project (the only one with Lux).
julia --project=training -e 'using Pkg; Pkg.instantiate()'
julia --project=training exp_square_HIT/10_setup_lstm.jl          # the configuration table

# the stride scan (cluster: sbatch batch_scripts/run_m4_sweeps.sh stride); a subset of rows:
RIKFLOW_M4_POINTS=3 RIKFLOW_M4_EPOCHS=10000 \
  julia --project=training exp_square_HIT/tools/m4_stride_scan.jl 2 1
# export a scan point as a deployable fit
julia --project=training exp_square_HIT/tools/m4_export_point.jl \
  stride_scan_StochLSTM2_seed1_points3_cap10000.jld2 100 32
# offline scores
julia --project=training analysis/m4_common_val.jl
julia --project=training analysis/m4_traj_heldout.jl

# rollout fine-tune of an exported fit (cluster: run_m4_sweeps.sh rollout)
RIKFLOW_M4_INIT=StochLSTM2_s100b32_points3_cap10000 RIKFLOW_M4_L=200 RIKFLOW_M4_BURN=100 \
  julia --project=training exp_square_HIT/tools/m4_rollout_train.jl

# an exploratory fit (target, arch, beta, emission, L, rollout, warm start -- see the header)
RIKFLOW_X_TAG=dq_vrnn_b1e-4 RIKFLOW_X_TARGET=dQ RIKFLOW_X_ARCH=vrnn \
  julia --project=training exp_square_HIT/tools/m4_explore_fit.jl

# online, cluster (GPU partition, M4 on the CPU; the fit's cell index after `lstm`)
RIKFLOW_M4_MODEL_DIR=$PWD/exp_square_HIT/output/TO_LSTM/<fit dir> sbatch exp_square_HIT/batch_scripts/run_online.sh lstm 2 1
# online, locally on the CPU (~1.2 s/step; the IC extract is written once by tools/m4_extract_ic.jl)
RIKFLOW_ONLINE_DEVICE=cpu RIKFLOW_ONLINE_IC=$PWD/exp_square_HIT/output/online_ic_data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2 \
RIKFLOW_ONLINE_TSIM=10 RIKFLOW_M4_MODEL_DIR=$PWD/exp_square_HIT/output/TO_LSTM/explore/<tag> \
  julia -t 3 --project=. exp_square_HIT/12_online_StochLSTM.jl 2 1

# online scoring
julia --project=analysis analysis/m4_online_ensemble.jl     # 100 TU ensembles (fig16)
julia --project=analysis analysis/m4_screen.jl              # short runs against the 10 TU null

# §7c/§7d (2026-09-24): a fit with checkpoints scored on 50-100 TU, and the lead candidate
RIKFLOW_D_TAG=r3_lin_sd_h2 RIKFLOW_D_ARCH=lstm RIKFLOW_D_EMISSION=state_dependent RIKFLOW_D_H=2 \
RIKFLOW_D_SKIP=1 RIKFLOW_D_FREEZE=Ws,V1 RIKFLOW_D_SEEDHEAD=1 \
  julia --project=training exp_square_HIT/tools/m4_diag_fit.jl
# closed-form linear + eta models (M0's structure in the M4 code path), one per lambda
RH=1 LAMS=0,3e-6,1e-5,3e-5,1e-4 julia --project=training exp_square_HIT/tools/m4_linear_eta.jl
# online on a local GPU (desktop RTX 3090; cap CUDA.jl's pool on a shared card)
JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB RIKFLOW_ONLINE_IC=$PWD/exp_square_HIT/output/online_ic_data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2 \
RIKFLOW_ONLINE_TSIM=20 RIKFLOW_M4_MODEL_DIR=$PWD/exp_square_HIT/output/TO_LSTM/diag/r3_lin_sd_h2 \
  julia --project=. exp_square_HIT/12_online_StochLSTM.jl 2 1
TSCREEN=20 M4_SCREEN_SUBDIR=diag julia --project=analysis analysis/m4_screen.jl
julia --project=analysis analysis/m4_online_moments.jl diag/r3_lin_sd_h2 diag/rdg_h1_l1e-05
```

⚠️ **Julia 1.13 buffers stderr to a plain file until exit**, so `julia ... > log` shows nothing while
it runs (the desktop runs used `script -qfec "julia ..." log` for a TTY). A SLURM log is a plain file.

⚠️ On a 16 GB workstation keep it to three Julia processes and no analysis alongside; Claude Code's
low-memory reaper stopped background jobs twice at four.

## 10. Plan step 0 checks (2026-09-28): mini-D6 power and M0ᶜ gates

`analysis/d6_power.jl` (step 0(c)) and `analysis/m0c_checks.jl` (M0ᶜ checks (i), (ii) and the acceptance
diagnostic, plan §3 / §21 item 3 / gotcha #72(iii)). CPU only. **What ran on what:** M0ᶜ (i)–(iii) ran on
**real data** (R1's tracked cache, the existing 20 TU online runs, the measured response kernel). The mini-D6 power
analysis ran on a **synthetic surrogate only**, because the D6 member files were not on the desktop yet (see
10a for the command). 🔒 No step past 74 TU was read.

### 10a. Mini-D6 power — ✅ REAL D6 (LinReg1 vs LinReg7), run 2026-09-28

Real run: `analysis/d6_power.jl` on the full D6 member files, 87 paired ICs, t = 10.25–96.25 TU, M = 10.
Log: `analysis/output/d6_power_D6_LinReg1_D6_LinReg7.jld2`. The per-IC CRPS difference has ACF 0.12 / −0.05 /
0.10 at lags 1–3, inside the ±0.21 band, so b = 1–2 is adequate: b = 1/2/4 changes the full-set half-width by
≤ 8%.

**The known difference (full set, A − B).**
- Primary CRPS (grid): 0.1982 vs 0.2346, Δ = **−0.036**. LinReg1 is 16% better.
- Skill: 0.4957 vs 0.5099, Δ −0.014.
- In-band: 27 vs 5.

**Minimum detectable difference (MDD).** 90% CI half-width of the paired primary CRPS, in reference-sd units
(real data):

| pool | K | M | pairing | N_LEAD | CRPS MDD | resolves LinReg1−LinReg7? | skill MDD |
|---|---|---|---|---|---|---|---|
| all | 87 | 10 | CRN | 400 | 0.014 | yes | 0.019 (Δ −0.018: **no**) |
| all | 87 | 5 | noCRN | 400 | 0.021 | yes | 0.030 |
| t ≥ 52 | 45 | 10 | CRN | 400 | 0.017 | yes (Δ −0.028) | 0.022 |
| t ≥ 52 | 45 | 5 | CRN | 400 | 0.016 | yes | 0.024 |
| **t ≥ 52** | **45** | **5** | **noCRN** | **400** | **0.026** | **yes (Δ −0.028), barely** | 0.030 |
| t ≥ 52 | 30 | 5 | noCRN | 400 | 0.036 | 8–17% of subsets | 0.039 |
| all | 30 | 5 | noCRN | 400 | 0.031–0.036 | 50–83% | 0.039–0.047 |

**What this means for the funnel (plan §7).**
1. **Primary score: CRPS is the right choice.** Skill/climatology cannot resolve even the known LinReg1–LinReg7
   gap at any size tested (MDD 0.019–0.047 against |Δ| ≤ 0.027). The in-band count resolves that gap easily
   (MDD 6–9 against +22), but only because the gap is large.
2. **30 ICs is not enough. Use every package in the block.** For a cross-family screen (noCRN), 45 late ICs ×
   M = 5 give MDD ≈ 0.026 CRPS ≈ **13% of LinReg1's CRPS**. So the mini-D6 detects an improvement of about
   LinReg1-vs-LinReg7 size, 13–16%, and **not a few-percent gain**.
3. **M = 5 costs little against M = 10** (0.016 vs 0.017 with CRN). N_LEAD 400 costs nothing on the primary
   score, whose leads are ≤ 200 steps.
4. ⚠️ **The "t ≥ 52" pool here is D6's 1 TU-spaced ICs over 52–96 TU**, i.e. selection *and* confirmation
   blocks. The selection block alone (52–74 TU) has ~22 of D6's ICs, or ~46 packages at the full 0.48 TU
   spacing. Adjacent packages are ~0.5 TU apart, and whether they add independent information is unmeasured
   here: the lag-1 ACF at 1 TU is 0.12. So the table's 45-IC row is an **optimistic** bound for the selection
   block. Plan on MDD ≈ 0.026–0.036, i.e. a mini-D6 that sees **≥ 13–18% CRPS improvements only**.
5. **Consequence for the kill criterion (D-13).** A null at mini-D6 size means "no gain ≥ ~15%", not "no gain".
   A smaller real gain needs the confirmation D6 at M = 10 with CRN-free pairing, MDD ~0.02 at 45 ICs, or more ICs.

#### Synthetic validation (the machinery; written before the real files arrived)


Method. Paired A − B on the same ICs, policy A (an IC is dropped if either closure lost a member). Three scores:
- (a) **primary**: fair ensemble CRPS in reference-sd units, averaged over 6 bands × leads ≤ 0.5 TU. Two lead
  sets: the grid {25, 50, 100, 200} and every 5th step ("dense").
- (b) mean skill/climatology over the cells ≤ N_LEAD.
- (c) the in-band count.

Subsamples: K ∈ {30, 45, 90 (= all 87)}, drawn as contiguous or strided IC sets, from all ICs or only those with
t ≥ 52 TU. M ∈ {10, 5, 5-noCRN}, N_LEAD ∈ {1000, 400}. Each subsample gets an IC moving-block bootstrap (2000
replicates). The block length is b = ⌈1 TU / narrowest spacing⌉, which gives 1–2. MDD = the 90% CI half-width.

🔴 **"noCRN" is the row to size a cross-family screen on.** LinReg1 and LinReg7 share member seeds, so their
members 1..M are common random numbers. An LSTM-vs-MVG screen is paired by IC only (plan §7). noCRN scores A's
members 1–5 against B's members 6–10.

**Validation on the surrogate.** Six ringing AR(2) bands (1 TU period), D6's IC spacing, and B's error set 3%
larger with 0.8× the spread, so the difference is known. Checks:
- A scored against itself gives differences of exactly 0.
- Bootstrap 90% CI coverage of the known difference over 60 independent replicates is **83–90%**: CRPS 85/83–88%,
  skill 83–90%. It is slightly liberal, so read the MDDs as ~10% optimistic.
- The b ∈ {1, 2, 4} sensitivity moves the MDD by ≤ 20%.
- N_LEAD 400 vs 1000 does not change (a) at all, since its leads are ≤ 200 steps. It moves (b) and (c) only through
  the cell set.

The synthetic MDDs (CRPS, sd units, CRN / noCRN) are **not transferable numbers**. They depend on the surrogate's
error model: CRN pairing shrinks the MDD ~5× here (K = 45 strided: 0.005 vs 0.028). Only the machinery is
validated.

**To run on the real files once they are copied** (`analysis/output/D6_LinReg1/`, `D6_LinReg7/`,
`analysis/data/hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2`):

```bash
cd lib/RikFlow && julia --startup-file=no --project=analysis analysis/d6_power.jl   # ~minutes; caches the arrays
# other pair / dirs: D6_POWER_A=<dir> D6_POWER_B=<dir>;  D6_POWER_NBOOT, D6_POWER_NSUB, D6_POWER_TAU
```

It prints the full-set A − B, the per-IC ACF check on the block length, and one row per (pool, mode, K, M, N_LEAD).
Each row gives est / sd-across-subsamples / MDD / %-resolved for each score. Reference: §4c has skill 0.4957 vs
0.5099 (Δ −0.014) and in-band 27 vs 5.

### 10b. M0ᶜ check (i): does the closed loop with white η already make the middle-band persistence? — **yes**

`m0c_checks.jl i`. Every 20 TU online run of a linear + η closure on disk, steps 101–8000:
- **ONLINE dQ ACF**: the correction the closure actually applied.
- **online residual**: dQ − μ(x_online), with μ the deployed closure run `stochastic = false` on the online inputs.
  The method is validated by `colour/lin_h1_ar`, where it recovers the imposed AR(1) (0.73 / 0.82).
- The tracked record over the same 0.25–20 TU window, for comparison.

Middle bands, lag 1 / 5 / 20:

| | Z[7,15] dQ | E[7,15] dQ | Z/E[7,15] residual lag 1 |
|---|---|---|---|
| **tracked record** | 0.98 / 0.74 / 0.34 | 0.99 / 0.82 / 0.32 | 0.76 / 0.86 (data, h = 1 mean) |
| white M0, h = 1 (`r2_lin_const`, 3 runs) | 0.98 / 0.87 / 0.49 | 0.99 / 0.92 / 0.44 | −0.00 / −0.00 |
| same, noise × 0.5 / noise ≈ 0 (`_n05`, `_n0`) | 0.99 / 0.92 / 0.59 · 1.00 / 0.93 / 0.60 | 1.00 / 0.93 / 0.44 · 1.00 / 0.94 / 0.42 | ≈ 0 |
| white M0, h = 1, window (`b_lin_eta_h1`, 5) | 0.98 / 0.85 / 0.47 | 0.99 / 0.90 / 0.34 | ≈ 0 |
| white M0, h = 2 / 3 / 5 / 10 | 0.98–0.99 / 0.77–0.90 / 0.44–0.71 | 0.99 / 0.83–0.90 / 0.37–0.61 | ≈ 0 (data 0.28/0.18 at h = 2, ≈ 0 at h ≥ 3) |
| M0ᶜ, AR(1) η (`colour/lin_h1_ar`, 5) | 0.99 / 0.88 / 0.44 | 1.00 / 0.91 / 0.26 | 0.73 / 0.82 (imposed) |

🔑 **The white closed loop already makes all of the correction's persistence, and more.** At every h and every lag
≥ 1 the online middle-band dQ ACF is ≥ the tracked one. It is +0.1 to +0.2 too persistent at lags 5–20, and it
stays that way with the noise switched off. So the persistence comes from the mean map plus the solver, not from η.
The exception is ridge: `rdg_h1_l1e-05` (λ = 1e-5) has lag 1 of 0.91 / 0.90, below the data's, but it is still
above the data at lags ≥ 5.

The data residual's 0.76 / 0.86 is colour *relative to μ on the tracked states*. The loop reproduces it in dQ
without any help. AR(1) η adds nothing to dQ's lag 1 (0.99 → 0.99–1.00) and removes the lag-20 memory in E[7,15]
(0.44 → 0.26).

⚠️ `m4_screen.jl`'s "dQ lag-1 0.73–0.75 / 0.77–0.79" is **E[0,6]**; the same number here is 0.74 tracked and
0.77–0.79 online.

### 10c. M0ᶜ check (ii): the h control — **the middle-band colour vanishes at h ≥ 5**

`m0c_checks.jl ii`. Float64 least squares on the M0 design (`build_history`, `:q_star_q`, level target, bias last).
Residual ACF lags 1 / 2 / 5 / 10, in-sample → held-out 52–74 TU:

| fit | cond(X) raw / std | max\|C_std\| | Z[7,15] | E[7,15] | other bands, \|lag 1\| max (held) | sd held/fit |
|---|---|---|---|---|---|---|
| 1–10, h = 1 | 2.9e8 / 4.9e3 | 1.1 | 0.76 0.41 −0.20 0.00 → **0.78** 0.44 −0.14 | 0.86 0.69 0.12 −0.26 → **0.87** 0.70 0.16 −0.24 | 0.23 (E[16,32]) | 0.89–0.99 |
| 1–10, h = 2 | 2.3e9 / 4.1e4 | 2.1 | 0.30 0.04 → 0.35 0.02 | 0.18 0.11 → 0.24 0.07 | 0.06; lag 2 −0.24 (Z[16,32]) | 0.88–0.99 |
| 1–10, h = 5 | 7.9e10 / 1.9e6 | 18.7 | −0.01 −0.00 → **0.07** −0.07 | −0.02 0.00 → **0.05** −0.09 | 0.04 | 0.89–1.00 |
| 1–10, h = 6 | 1.3e11 / 2.8e6 | 39.3 | 0.01 → 0.09 −0.06 | 0.00 → 0.07 −0.07 | 0.05 | 0.90–1.00 |
| 1–50, h = 1 | 2.7e8 / 4.6e3 | 1.1 | 0.76 0.40 → 0.79 0.45 | 0.86 0.69 → 0.88 0.70 | 0.24 | 0.84–0.92 |
| 1–50, h = 2 | 2.1e9 / 3.7e4 | 2.1 | 0.28 → 0.37 | 0.18 → 0.27 | 0.08 | 0.83–0.93 |
| 1–50, h = 5 | 6.9e10 / 1.6e6 | 21.5 | −0.01 → **0.11** −0.03 | −0.02 → **0.09** −0.07 | 0.04 | 0.83–0.93 |
| 1–50, h = 6 | 1.1e11 / 2.6e6 | 17.6 | 0.00 → 0.12 −0.02 | 0.00 → 0.11 −0.05 | 0.04 | 0.83–0.93 |

🔑 **§7j's 0.73 / 0.82 is an h = 1 artefact.** It is the part of the dynamics that two more lags of q/q* absorb.
At h = 2 it is already 0.2–0.4, and at LinReg1's h = 5 the in-sample residual is white to ±0.02. Held out, a
residual lag-1 of 0.05–0.12 is left, with a small negative lag 2. h + 1 (h = 6) changes nothing further.

⚠️ The standardised design's cond is ~2e6 at h = 5–6 (the §3 identifiability warning stands). ⚠️ The held-out
residual sd is **0.83–0.93×** the in-sample sd on the 1–50 TU fits. The 52–74 TU block is quieter than the fit
window, which is not the "14–18% wider" of RH-1 on 50–100 TU.

### 10d. Acceptance diagnostic (iii): kernel-weighted noise power

`m0c_checks.jl iii`. P = Var(Σ_{k=1..200} G_k e_{n−k}), with G the measured impulse kernel (§7h(e),
`response/response_kernel.jld2`; its diagonal step response at k = 200 is −0.8 / 115.5 / −19.4 / 55.5 / −0.7 /
29.2). Every noise is set to the residual's variance. The table gives the ratio to white, diagonal kernel, per band;
"data" is the residual series itself filtered, in-sample / held-out 52–74 (the latter rescaled to the fit
variance).

| residual of | band | ρ₁ | AR(1) | AR(2)-YW | AR(2)-LS (lags 1–20) | data fit / held |
|---|---|---|---|---|---|---|
| **h = 5, 1–10 TU (M0)** | Z[7,15] | −0.01 | 0.98 | 0.98 | 0.98 | 1.08 / 1.14 |
| | E[7,15] | −0.02 | 0.97 | 0.97 | 0.96 | 1.07 / 1.21 |
| | other four | ≈ 0 | 1.00 | 1.00 | 1.00 | 0.89–0.98 / 0.87–1.30 |
| h = 5, 1–50 TU | all | ≈ 0 | 0.97–1.00 | 0.97–1.00 | 0.96–1.00 | 0.95–1.02 / 1.01–1.20 |
| h = 1, 1–10 TU (§7j's) | Z[7,15] | 0.76 | 5.12 | 3.07 | **2.73** | 2.74 / 2.86 |
| | E[7,15] | 0.86 | 11.13 | 8.17 | **4.58** | 3.54 / 4.42 |
| | Z/E[0,6], Z/E[16,32] | 0.01–0.21 | 1.02–1.50 | 0.82–1.45 | 0.73–1.54 | 1.00–1.79 / 1.53–2.22 |

With the full 6 × 6 kernel (simulated, lag-0 cross-correlation kept), every ratio lies in 0.92–1.11 for the AR fits
and 0.84–1.30 for the data. Cross-band terms dominate the output variance there, and the diagonal colour barely
registers.

🔑 **Where there is colour (h = 1), no AR fit is at or below white.** AR(1) is 5–11× white, which is §7j's
over-dispersion with its mechanism. The ACF-fitted AR(2) matches the data residual in Z[7,15] (2.73 vs 2.74 / 2.86)
and is 1.3× above it in-sample in E[7,15] (4.58 vs 3.54; held out 4.42). So the plan's corrected expectation holds
(#72(iii)): faithful colour *raises* the kernel-weighted power, it never lowers it. At h = 5 there is nothing to
fit. The AR fits collapse to white (0.96–1.00), and the data residual sits at 0.84–1.30× white, with no band-
specific excess.

⚠️ The kernel is truncated at 200 steps (0.5 TU) and bands 2 / 4 / 6 have not settled by then (§7h(e)), so P
under-weights the lowest frequencies. That strengthens the AR(1) conclusion.

### 10e. Verdict — **do not build M0ᶜ** (plan §21 item 3)

Both of §21 item 3's stop conditions hold, independently:
- (i) The online middle-band ACF of the white M0 already matches the data's persistence, and exceeds it at
  lags ≥ 5, at every h. With noise off it is the same.
- (ii) The colour vanishes at h = 5: in-sample |ρ₁| ≤ 0.02, held-out 0.05–0.12.

The acceptance diagnostic agrees. At the M0 design (h = 5) an AR(p) fit is white. At h = 1, where the colour exists,
every AR fit raises the kernel-weighted power 2.7–11× over white.

The one open end is the small held-out residual lag-1 of 0.05–0.12 at h = 5–6 (1–50 TU fit). It is too small to
carry a cell, and it is below the online loop's own excess persistence.

### 10f. Per-QoI breakdown (Rik asked: does the verdict hold for all six QoIs?) — 2026-09-28

Same `m0c_checks.jl i ii iii` run, all six QoIs. Full log: job scratch `m0c_full.log`.

**(ii) Tracked-record residual ACF of the LS mean, lags 1 / 2.** 1–10 TU fit, in-sample → held-out 52–74 TU.

| QoI | h = 1 | h = 2 | h = 3 (from (i)'s tracked resid) | **h = 5 (M0)** |
|---|---|---|---|---|
| Z[0,6] | 0.09 0.11 → 0.13 0.11 | −0.01 0.01 → 0.03 −0.00 | 0.03 −0.00 | −0.00 −0.00 → 0.04 −0.01 |
| E[0,6] | 0.03 0.02 → 0.06 0.02 | −0.00 −0.01 → 0.02 −0.02 | 0.02 0.01 | −0.00 0.00 → 0.03 −0.01 |
| Z[7,15] | **0.76 0.41** → 0.78 0.44 | **0.30** 0.04 → **0.35** 0.02 | 0.06 −0.01 | −0.01 −0.00 → 0.07 −0.07 |
| E[7,15] | **0.86 0.69** → 0.87 0.70 | **0.18** 0.11 → **0.24** 0.07 | 0.01 0.04 | −0.02 0.00 → 0.05 −0.09 |
| Z[16,32] | 0.01 −0.14 → 0.04 −0.14 | −0.01 **−0.22** → 0.01 **−0.24** | 0.00 0.03 (lag 5: 0.16) | −0.00 0.00 → 0.03 −0.04 |
| E[16,32] | 0.21 0.03 → 0.23 0.02 | 0.06 **−0.15** → 0.06 **−0.19** | −0.00 −0.01 (lag 5: 0.14) | 0.00 0.00 → 0.02 −0.05 |

On the 1–50 TU fit at h = 5, the held-out lag 1 is 0.01 / 0.01 / **0.11 / 0.09** / 0.04 / 0.04, and lag 2 is
−0.01…−0.07.

**(i) Online dQ ACF of the white-η linear closure at h = 5 (`diag/rdg_h5_l0`, 3 × 20 TU) against the tracked
record, lags 1 / 5 / 20.**

| QoI | online | tracked | online − tracked at lag 20 |
|---|---|---|---|
| Z[0,6] | 0.98 / 0.89 / 0.71 | 0.95 / 0.80 / 0.50 | +0.21 |
| E[0,6] | 0.78 / 0.34 / 0.06 | 0.74 / 0.21 / −0.02 | +0.08 |
| Z[7,15] | 0.99 / 0.80 / 0.50 | 0.98 / 0.74 / 0.34 | +0.16 |
| E[7,15] | 0.99 / 0.85 / 0.55 | 0.99 / 0.82 / 0.32 | +0.23 |
| Z[16,32] | 1.00 / 0.99 / 0.96 | 1.00 / 0.99 / 0.96 | 0.00 |
| E[16,32] | 1.00 / 0.98 / 0.95 | 1.00 / 0.97 / 0.94 | +0.01 |

The same pattern holds at h = 1, 2, 3 and 10 and in window mode: online ≥ tracked at every lag, in every QoI.

**(iii) Kernel-weighted power at h = 5, P/P_white.** AR(1) / AR(2) fits are 0.96–1.01 in all six QoIs. The data
residual is 0.89–1.08 in-sample and 0.87–1.30 held out.

**Per-QoI reading.**
1. **At h = 5 the M0ᶜ verdict holds for all six QoIs.** The in-sample residual is white to ±0.02 at lags 1, 2, 5
   and 10 everywhere. Held out, the largest residual is in the middle bands: lag 1 0.05–0.07 on the 1–10 TU fit,
   0.09–0.11 on 1–50. That is small, but twice the other bands'.
2. **Check (i) has no power in the small-scale bands.** Their dQ ACF is 0.94–1.00 out to lag 20 both tracked and
   online (the correction follows the slowly varying level), so "online ≥ tracked" is automatic there. The verdict
   for Z/E[16,32] rests on (ii) and (iii) alone.
3. **In the large and middle bands the loop is TOO persistent.** Online exceeds tracked by +0.08 to +0.23 at
   lag 20. That is a mean-map + solver property, not a missing-noise property, and it is a target for the mean
   (M3ᶠ), not for colour.
4. 🔴 **At h = 2, the design of M3ᶠ and M0ᵛ, colour remains in 4 of 6 QoIs.** The middle bands have lag 1
   0.18–0.37. The small-scale bands have a **negative lag 2 of −0.15 to −0.24**, which is new: it is absent at
   h = 1 and h = 5. The matched M0@h2 baseline carries the same residual, so the paired comparisons stay fair.
   But a closure at h = 2 leaves this structure in the noise, while h = 3 removes it (all |ρ₁,₂| ≤ 0.06, with a
   residual lag-5 of 0.14–0.16 in the small-scale bands).
5. ⚠️ **The online residuals at h ≥ 5 show small-scale ρ₁ −0.07…−0.16 and ρ₂ +0.11…+0.19.** The deployed noise
   is white, so this is a reconstruction artefact of μ(x_online): likely Float32 inputs through a cond ~2e6 design.
   It is not colour.

### 10g. Under ridge: the deployed λ ladder's residual is coloured, strongly from λ = 1e-2 on (2026-09-28, Rik asked)

`analysis/m0c_ridge.jl` applies every archived `TO_LRS/LinReg<n>` fit **exactly as deployed** (per-QoI scaling,
`c · [x; 1]`, output scaling, level target), teacher-forced on R1's tracked record at h = 5. The fit window is
1–10 TU (the ladder's train range) and the held-out window is 52–74 TU.

*Wiring check.* Deployed LinReg1 against the Float64 LS refit of (ii): the per-QoI residual ACFs agree to
≤ 0.01. The max point difference is 0.17 sd, the expected round-off spread of two fits at cond ~2e6 (#63).

Residual lag 1 per QoI, fit → held out (lags 2/5/10 in the log), and residual sd relative to LinReg1:

| fit | λ (LinReg, `:normal`) | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] | sd / sd(LinReg1) |
|---|---|---|---|---|---|---|---|---|
| LinReg1 | 0 | −0.00 → 0.04 | −0.00 → 0.03 | −0.01 → 0.07 | −0.02 → 0.05 | −0.00 → 0.02 | −0.00 → 0.02 | 1 |
| LinReg5 | 1e-5 | −0.00 → 0.04 | −0.00 → 0.03 | 0.05 → 0.13 | 0.03 → 0.09 | −0.00 → 0.02 | 0.01 → 0.03 | 1.00–1.02 |
| LinReg6 | 1e-4 | 0.00 → 0.05 | −0.00 → 0.03 | **0.20 → 0.27** | **0.21 → 0.27** | 0.00 → 0.03 | 0.03 → 0.05 | 1.00–1.08 |
| LinReg2 | 1e-2 | 0.11 → 0.14 | 0.02 → 0.05 | **0.76 → 0.78** | **0.83 → 0.85** | 0.19 → 0.21 | 0.30 → 0.32 | 1.0–2.6 |
| **LinReg7** | **1** (on the D6 front) | **0.77** | **0.44** | **0.95** | **0.97** | **0.81** | **0.85** | **1.2–11.4** |
| LinReg8 | 10 | 0.94 | 0.70 | 0.98 | 0.98 | 0.94 | 0.95 | 1.8–26 |
| LinReg9 | 100 | 0.97 | 0.89 | 0.99 | 0.99 | 0.97 | 0.97 | 3.4–47 |
| LinReg10 | 1e4 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 15–200 |

(From λ = 1 on, fit and held-out lag 1 agree to ±0.02.)

**Reading.**
1. **The λ = 0 verdict (10b–10f) does not transfer to ridge.** Ridge shrinks the mean's dynamics, and whatever it
   takes out reappears as a *coloured* residual.
   - First in the middle bands: λ = 1e-4 gives lag 1 0.20–0.27.
   - At λ = 1e-2 the middle bands are back at the h = 1 level (0.76–0.85).
   - At **λ = 1 (LinReg7) all six QoIs are coloured**: lag 1 0.44–0.97, lags 5–10 up to 0.53. The residual sd is
     1.2× (E[0,6]) to 11× (E[7,15]) LinReg1's.
2. **LinReg7's MVG noise is white with that inflated variance, and D6 still finds it under-dispersed** (spread–skill
   median 0.559, 5/36 in band, `results.md` §4c). The closed loop sets its spread, not the one-step Σ. The coloured
   residual has far more low-frequency power than the white draw at the same variance (§10d: AR(1) at 0.8 is
   5–11× white). So **a coloured residual on the ridge-stabilised mean is a plausible fix for exactly the
   calibration price ridge pays on the D6 front.** This is untested; it is a hypothesis, not a result.
3. **Consequence for M0ᶜ (plan §21 item 3):** the gate is closed for the λ = 0 baseline, but **open for the
   ridge-stabilised cells**: LinReg7-type M0, and M3ᶠ's ridge skip.
   - Next checks, all CPU: (i) the kernel-weighted power of an AR fit to LinReg7's residual against its white Σ;
     (ii) the D1 joint (mean, AR) fit at λ = 1, which also lets part of the dynamics back into the mean;
     (iii) the online dQ ACF of LinReg7's D6 members against the tracked record.
   - The GPU test would be a mini-D6 of LinReg7 + AR(p) against LinReg7, which is a same-family pairing.
4. **M3ᶠ inherits this.** Its frozen skip is ridge-fitted. Which skip λ matters depends on the M4-path convention,
   which is not LinReg's (#72(i)). So the skip's residual ACF must be measured per λ before its noise head is
   declared white.

### 10h. Ridge + colour, three CPU checks on LinReg7 (λ = 1, h = 5) — 2026-09-28

`analysis/m0c_ridge_colour.jl`. Log in job scratch: `m0c_ridge_colour.log`. Fit window 1–10 TU, held out
52–74 TU. The D6 ICs used end by 74 TU.

**(3) Online: LinReg7's white noise destroys the correction's persistence; LinReg1 keeps it.** Per-member dQ ACF
over the 1200-step forecasts, averaged, against the tracked record on the same steps. 63 ICs, ~630 members per
closure.

| QoI | tracked, lag 1 / 2 / 5 / 20 | **LinReg1** online | **LinReg7** online | sd(dQ) on/trk, LR1 / LR7 |
|---|---|---|---|---|
| Z[0,6] | 0.95 0.90 0.78 0.44 | 0.95 0.90 0.77 0.48 | **0.40** 0.52 0.42 0.22 | 1.01 / 0.73 |
| E[0,6] | 0.74 0.54 0.21 −0.01 | 0.75 0.57 0.27 0.02 | **0.39** 0.35 0.15 0.01 | 1.02 / 0.90 |
| Z[7,15] | 0.98 0.92 0.70 0.25 | 0.98 0.92 0.71 0.32 | **0.05** 0.30 0.25 0.13 | 1.06 / 1.01 |
| E[7,15] | 0.99 0.96 0.80 0.27 | 0.99 0.96 0.80 0.34 | **0.06** 0.35 0.28 0.12 | 1.04 / 0.83 |
| Z[16,32] | 0.99 0.99 0.98 0.92 | 1.00 0.99 0.98 0.94 | 0.96 0.97 0.96 0.91 | 1.26 / 1.01 |
| E[16,32] | 0.99 0.98 0.96 0.90 | 0.99 0.99 0.97 0.91 | 0.90 0.93 0.92 0.87 | 1.25 / 1.01 |

- **LinReg1 reproduces the tracked persistence to ≤ 0.07 at every lag, in every QoI.**
- **LinReg7 does not.** Its lag 1 is below its lag 2 in the four large/middle QoIs: a white component carries most
  of the variance on top of a persistent signal. That is the white MVG draw at the inflated residual variance of
  §10g (8–11× LinReg1's in the middle bands).
- So ridge does not remove the dynamics from the correction. It moves them from the mean into a residual, and then
  **samples that residual as white noise**.

**(1) Kernel-weighted power: LinReg7's white Σ under-delivers low-frequency power by ~3×.** P/P_white at LinReg7's
residual variance, measured solver kernel (§10d):

| QoI | ρ₁ | AR(1) | AR(2)-LS | data residual | full 6×6 kernel: AR(1) / AR(2)-LS / data |
|---|---|---|---|---|---|
| Z[0,6] | 0.77 | 6.4 | 8.4 | 11.0 | 2.72 / 2.55 / 2.80 |
| E[0,6] | 0.44 | 2.5 | 2.2 | 2.3 | 2.37 / 2.09 / 2.96 |
| Z[7,15] | 0.95 | 13.9 | 6.8 | 8.6 | 3.72 / 2.89 / 3.08 |
| E[7,15] | 0.97 | 33.6 | 9.9 | 10.7 | 4.21 / 3.02 / 3.18 |
| Z[16,32] | 0.81 | 6.3 | 7.2 | 6.3 | 3.21 / 2.72 / 2.72 |
| E[16,32] | 0.85 | 10.4 | 6.8 | 8.3 | 3.22 / 2.73 / 2.74 |

- With the full kernel, the data residual carries **2.7–3.2×** the white noise's power.
- The ACF-fitted AR(2) comes within ≈ 10% of it (2.1–3.0). AR(1) overshoots in the middle bands (3.7–4.2).
- 🔑 **A consistency check, not a proof.** A ~3× power deficit is a spread deficit of √(2.7–3.2) = 1.64–1.79.
  D6 measured LinReg7's spread–skill median at **0.559**, i.e. a deficit of **1/0.559 = 1.79** (`results.md`
  §4c). The two agree to within the precision of either.
- The white noise *also* over-delivers absolute power against LinReg1's (white7/white1 = 1.4–1.5, full kernel).
  So LinReg7 has more total noise than LinReg1 but in the wrong frequencies.

**(2) Joint (mean, AR) fit at λ = 1** (`fit_ridge(λ = 1)` reproduces the deployed LinReg7 `c` to 1.7e-3, so the
convention matches).

| model | held-out NLL/row | AR a₁ (a₂) | held-out innovation lag 1 | innovation sd / white sd | ‖C − c₇‖/‖c₇‖ |
|---|---|---|---|---|---|
| LinReg7 mean + AR(1), two-stage | −32.61 | 1.00 0.85 0.98 0.96 0.94 0.94 | −0.11 −0.20 0.77 0.83 0.12 0.34 | 0.25–0.98 | 0 |
| LinReg7 mean + AR(2), two-stage | −34.38 | a₁ 1.07–1.89, a₂ −0.16…−0.95 | −0.30…0.46 | 0.13–0.99 | 0 |
| **joint mean + AR(1)** | **−36.68** | 0.79 0.30 0.93 0.95 0.95 0.95 | **0.02–0.09** | 0.08–0.88 | 4.5 |
| joint mean + AR(2) | −36.68 | ≈ AR(1) (a₂ small) | 0.02–0.08 | 0.08–0.88 | 4.5 |

- The joint fit is the only one with white held-out innovations, and it beats the two-stage colour by 2.3–4.1
  nats/row. For reference, LinReg7 white is −21.43 under a full-covariance Gaussian; that is a different
  normalisation, so only the ordering is comparable.
- ⚠️ **One-step likelihood is exactly what failed to predict online before** (§25 of plan.md: LR+C had the best
  NLL and was +87% over-dispersed).
- ⚠️ **Both fits put AR poles at 0.93–1.00.** Two-stage AR(1) has **a = 1.00 in Z[0,6], a unit root**, and the
  two-stage AR(2) roots sit near the unit circle (a₁ 1.89, a₂ −0.95 in Z[7,15]). The joint fit re-estimates the mean
  wholesale (‖ΔC‖ = 4.5‖c₇‖) and lets a near-integrating residual carry the dynamics that ridge removed. That
  risks reintroducing the loop gain ridge was bought for.

**Verdict.**
- The hypothesis of §10g survives all three checks.
  - (3) LinReg7's online correction has lost the persistence LinReg1 keeps.
  - (1) Its white Σ is short of low-frequency power by the factor that matches its D6 under-dispersion.
  - (2) An AR(2) fitted to its residual ACF restores that power to within ~10%.
- **Candidate cell: "M0ᶜ-ridge" = LinReg7's mean + a stationary AR(2) fitted to the residual ACF** (two-stage,
  mean untouched, so ridge's stability argument is kept).
  - Prefer the ACF-fitted AR(2) (stationary by construction, power within ~10%) over the likelihood-optimal fits,
    whose poles are at the unit circle.
  - The joint fit is a second variant, flagged for stability.
  - Test: mini-D6 on the selection block against LinReg7, same member seeds (a same-family pairing, so CRN
    applies).
  - Pass: spread–skill moves toward 1 with no loss of stability and no loss of CRPS.
- **Needs code.** `LinReg` has no AR residual, and the M4 path's `scaling.eta_ar` is AR(1) only (V69). An AR(2)
  η on the LinReg sampler must also be warm-started, from the replayed residuals, not zero.

**Reproduce:** `julia --startup-file=no --project=training analysis/m0c_checks.jl [i] [ii] [iii]`. Part (i) takes
a few minutes, (ii) + (iii) about 1 min. It writes `analysis/output/m0c_checks_*.jld2`.

## 11. M3ᶠ and the matched M0@50 (plan step 1, 2026-09-28)

**Question.** On the existence window, does a small residual network on top of a frozen ridge skip beat
the matched linear model on the same inputs (plan.md §3, *The reduced ladder*; §7, *Screening funnel*)?

**Partition used (plan.md §7).** Training on R1, 1–50 TU. Fit rows are the first 80% (1.0–40.2 TU).
The tail (40.2–50 TU) is the early-stopping validation, as in every window fit. Offline held-out
scoring is on **52–74 TU only** (8801 one-step rows, teacher-forced). The reference for the online
scores is truncated to **t ≤ 74 TU** (`REF_TU_MAX=74`). Nothing here reads 76–97 TU.

**Design.**
- h = 2 inputs `[q*_n; q_{n-1}, q*_{n-1}; q_{n-2}, q*_{n-2}; 1]` (31 regressors).
- **M0@50** = ridge skip + constant correlated η seeded at the training-residual covariance
  (`tools/m4_linear_eta.jl`, deployed as `:lstm`, `V1 = 0`).
- **M3ᶠ** = the SAME skip, frozen, plus a 2-layer tanh MLP with `n_hidden = 16` on the same regressor.
  - It uses `arch = :dense`, `n_latent = 0`, `W = 1`: no latent path and no extra lags, so "same
    inputs" is literal.
  - The output layer is zero-initialised (`V1 = 0`), so update 0 **is** M0@50(λ).
  - The noise head is **constant, white, correlated Gaussian** (`emission = :constant`, `SEEDHEAD=1`,
    trained jointly).
  - Weight decay acts on the network only (`WD_EXCLUDE=bd,Araw`).
  - Training: `BATCH=256 LR=1e-2 VAL_EVERY=100 PATIENCE=30 STOP_PATIENCE=80 STOP_WINDOW=20000 EPOCHS=400`.
    That is 13–24k updates, about 1.5 min per fit on the CPU.
- **λ convention (M4 code path, not LinReg's; gotcha #72(i)).**
  - The objective is `‖Y − C X‖² + λ N_fit ‖C_{-bias}‖²`, with the regressor standardised by the
    1–50 TU training mean and sd and `dQ` standardised likewise.
  - So λ is a **per-row** penalty, and every column except the trailing bias is penalised (checked in
    `m4_window_fit.jl` and `m4_linear_eta.jl`: `Diagonal([fill(λ N, nin − 1); 0])`, where the bias is
    the last regressor).
- Weight decay is Optimisers' coupled AdamW: each update shrinks the weights by `lr × WD`.

### 11a. Offline, held-out 52–74 TU (one-step, teacher-forced; loss = 0.5 SSE/step over the six standardised dQ)

**The linear floor on the same inputs** is least squares, λ = 0: **held 0.2098, ensemble CRPS
0.10101**. Adding one more set of lags does not change this much: the stacked W = 10 window of the
full regressor, fitted by least squares, scores 0.2074 (−1.1%).

| model | held loss | vs M0(λ) | CRPS | vs M0(λ) | spread/skill per QoI | ‖g‖/‖Ws x‖ | ‖g‖/‖resid‖ |
|---|---|---|---|---|---|---|---|
| **M0@50 λ = 0** | **0.2098** | — | **0.10101** | — | 1.05 1.05 1.17 1.18 1.17 1.16 | — | — |
| M0@50 λ = 1e-5 | 0.2623 | — | 0.13361 | — | 1.06 1.06 1.07 1.11 1.15 1.12 | — | — |
| M0@50 λ = 1e-4 | 0.4478 | — | 0.18811 | — | 1.05 1.06 1.01 1.10 1.12 1.08 | — | — |
| M3ᶠ λ = 0, WD 1e-2 (s1/s2/s3) | 0.2098 ×3 | 0.00 / −0.00 / −0.00% | 0.10101 / 0.10090 / 0.10098 | 0.00 / −0.11 / −0.03% | ≈ M0 | 0 / 0.004 / 0.004 | 0 / 0.014 / 0.013 |
| M3ᶠ λ = 0, WD 1e-1 | 0.2098 ×3 | −0.01% ×3 | 0.10097 / 0.10090 / 0.10100 | −0.05 / −0.11 / −0.02% | ≈ M0 | 0.003 / 0.003 / 0.004 | 0.011 / 0.011 / 0.014 |
| M3ᶠ λ = 1e-5, WD 1e-2 | 0.2527 / 0.2490 / 0.2482 | −3.7 / −5.1 / −5.4% | 0.1292 / 0.1278 / 0.1274 | −3.3 / −4.3 / −4.6% | 1.05–1.14 | 0.059 / 0.060 / 0.069 | 0.17 / 0.17 / 0.20 |
| M3ᶠ λ = 1e-5, WD 1e-1 | 0.2617 / 0.2608 / 0.2613 | −0.2 / −0.6 / −0.4% | 0.1334 / 0.1329 / 0.1331 | −0.2 / −0.5 / −0.4% | 1.06–1.14 | 0.020 ×3 | 0.057 ×3 |

Held-out Gaussian NLL at λ = 0: M3ᶠ −7.207…−7.214 per step, against linear + η −7.210.

🔴 **Verdict: no network beats the linear floor on the same inputs.**
- **At λ = 0**, the network stays near its zero initialisation. Its output is 1–1.4% of the skip's
  held-out residual, and it changes CRPS by 0.00 to −0.11% (noise level). The held-out loss does not
  move.
  - The same holds for every pilot: `n_hidden` 32, batch 64 at `LR 3e-3`, and `W = 10` (which
    returned update 0 at `LR 1e-2`, and −0.09% CRPS / +0.03% loss at `3e-3`).
  - §7c's +3.4% for a recurrent `:vrnn` on 1–50 TU is **not** reproduced by a feed-forward residual
    on the matched h = 2 regressor. That gain may have come from its longer memory, not from
    nonlinearity; this was not tested here.
- **At λ = 1e-5**, the network "beats M0(λ)" by 3.7–5.4% (WD 1e-2), but **that gain is linear**.
  - Regress g(x) on x over the training rows. The linear part explains **86–92%** of g's held-out
    variance.
  - Skip + linear part scores **0.2475–0.2510, better than skip + g** (0.2482–0.2527). The
    nonlinear remainder hurts.
  - At WD 1e-1 the linear share is 0.46.
  - So the network is rebuilding the part of the least-squares map that ridge shrank (skip-residual
    sd grows 2.7–3.6× in the middle bands at 1e-5, §11c). It stays 18–20% worse than the λ = 0 floor
    (0.2098).
  - This is gotcha #72(i)'s concern seen from the other side. A frozen ridge skip does not hold the
    loop gain if the residual network is free to put it back. **Weight decay 1e-1 mostly stops it**:
    the gain falls to −0.2…−0.6%.
- The linear-share numbers come from `tools/m4_linfrac.jl`. Attribution in one sentence: **L1 (nonlinear mean)
  is zero offline at h = 2 on 1–50 TU.**

### 11b. Stage-2 smoke, 20 TU × 3 free-running (GPU, local IC extract), reference truncated to ≤ 74 TU

`m4_screen.jl` and `m4_online_moments.jl` with `REF_TU_MAX=74`.
- The reference null now has only 3 windows of 20 TU: flat 0.3–7%, >2900 1.9–12%, sd ratio
  0.93–1.12, KS 0.28–0.72, dQ lag-1 on E[0,6] 0.73–0.75, dmean ±0.28 sd.
- **All 15 runs are stable. The clamp/gate fired 0 times.**

| run (replicas r1/r2/r3) | KS | Z16 sd ratio | >2900 | Z16 min | dQ lag-1 E06 | mean offset, worst QoI (sd) | sd ratio range, six QoIs |
|---|---|---|---|---|---|---|---|
| M0@50 λ = 0 | 0.90 / 0.55 / 0.69 | 1.38 / 1.02 / 1.15 | 13 / 1.8 / 8.0% | 679 / 689 / 807 | 0.75–0.76 | E06 −0.40 / −0.56 / −0.32 | 1.02–1.45 |
| M0@50 λ = 1e-5 | 0.64 / 0.83 / 0.62 | 1.31 / 1.10 / 1.11 | 12 / 4.2 / 7.4% | 792 / 726 / 1027 | 0.72–0.74 | −0.07 / −0.27 / −0.19 | 1.01–1.37 |
| M0@50 λ = 1e-4 | 0.96 / 0.65 / 0.43 | 1.41 / 1.19 / 1.10 | 11 / 9.4 / 6.3% | 740 / 739 / 1090 | **0.55–0.57** | −0.17 / −0.19 / +0.07 | 0.95–1.41 |
| M3ᶠ λ = 0, WD 1e-1, s1 | 1.06 / 0.63 / 0.53 | 1.46 / 0.94 / 1.02 | 14 / 1.6 / 5.9% | 563 / 718 / 1012 | 0.75–0.76 | E06 −0.43 / −0.61 / −0.39 | 0.94–1.54 |
| M3ᶠ λ = 1e-5, WD 1e-2, s2 | 0.72 / 0.72 / 0.58 | 1.39 / 1.18 / 1.02 | 14 / 4.5 / 4.1% | 828 / 466 / 884 | 0.74–0.75 | −0.05 / −0.26 / −0.12 | 1.02–1.41 |

- **At h = 2 on 1–50 TU there is no large closed-loop bias to remove.** λ = 0 is within about
  ±0.2 sd everywhere except E[0,6] (−0.3…−0.6 sd). λ = 1e-5 is at −0.05…−0.27 sd in all QoIs. So
  the "zero crossing" is weak and lies between 0 and 1e-5 for E[0,6] only. The +1 sd bias of §7d
  (h = 1, 1–10 TU) does not appear here.
- λ = 1e-4 decorrelates the correction (lag-1 0.55 against the reference's 0.73–0.75) and shrinks
  sd(dQ) to 0.65–0.88 of the reference's.
- The M3ᶠ runs track their M0 replica by replica, as expected from 11a.
  - λ = 0: the same pattern, slightly wider in r1.
  - λ = 1e-5: the network restores sd(dQ) in the middle bands, 1.13–1.25 of the reference against
    0.71–0.77 for M0(1e-5), and moves the level statistics little.
- Every cell over-disperses in r1 (sd ratio 1.3–1.5). This is a replica effect common to all
  models: the same seed and IC give the same excursion.
- **Smoke only; no winner is claimed** (plan §7).

### 11c. Is a white constant head adequate at each skip λ? (coordinator request; cf. §10g)

This is the per-QoI ACF of the **skip-only (M0@50) residual**, teacher-forced on 52–74 TU, at lags
1/2/5, with the residual sd in standardised dQ units. Standardisation is by the 1–50 TU training
dQ statistics; QoIs are Z/E by band.

| skip λ | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| 0 | .01/−.01/.01 (sd .231) | .01/−.02/.01 (.589) | **.37**/.04/−.05 (.099) | **.27**/.10/−.04 (.052) | .02/**−.23**/.08 (.053) | .08/−.17/.08 (.059) |
| 1e-5 | .21/.13/.08 (.250) | .06/−.02/.01 (.589) | **.88/.64/.20** (.265) | **.93/.79/.32** (.185) | .30/−.02/.16 (.060) | .47/.17/.03 (.084) |
| 1e-4 | .61/.48/.43 (.332) | .27/.03/.04 (.616) | **.96/.86/.56** (.501) | **.97/.91/.63** (.359) | .71/.52/.53 (.085) | .76/.56/.28 (.119) |

- **λ = 0**: the white head is adequate in the outer bands. The middle bands carry a weak lag-1
  (0.27–0.37) that is gone by lag 2, and the small-scale band has a lag-2 lobe of −0.2.
- **λ = 1e-5**: the white head is **not** adequate in Z/E[7,15]. The residual there is lag-1
  0.88–0.93, and its sd is 2.7–3.6× the λ = 0 value. That persistent part is exactly what M3ᶠ's
  network re-learns linearly in 11a.
- **λ = 1e-4**: coloured in every QoI but E[0,6].
- This matches §10g's LinReg ladder at h = 5: ridge re-colours the residual.
- The numbers come from `tools/m4_residacf.jl`.

### 11d. Stage-4 long free runs (100 TU × 1 replica)

This is `m4_screen_long.jl` with `REF_TU_MAX=74`. Each chunk is scored against the 3-window
(0–74 TU) reference band. The whole run is scored against the 0–74 TU reference marginal. Stage 4 is
a pass/fail gate, not a selection score, and one replica is far short of the plan's 3.

| run | chunk KS, 0–20 … 80–100 TU | chunk Z16 sd ratio | Z16 min | clamp | dQ lag-1 E06 | whole run: KS / sd ratio | mean offset per QoI (sd) |
|---|---|---|---|---|---|---|---|
| M0@50 λ = 0 | 0.90 0.84 1.04 0.49 1.05 | 1.38 0.77 1.17 0.97 0.85 | **341** (40–60 TU) | 0 | 0.75–0.76 | 0.65 / 1.06 | −0.22 **−0.49** −0.05 −0.01 −0.27 −0.24 |
| M0@50 λ = 1e-5 | 0.64 0.65 1.18 0.41 0.92 | 1.31 0.93 1.04 1.15 0.84 | 716 | 0 | 0.73 | 0.59 / 1.08 | −0.20 −0.13 −0.18 −0.19 −0.25 −0.24 |
| M3ᶠ λ = 1e-5, WD 1e-2, s2 | 0.72 0.50 1.31 0.57 0.81 | 1.39 1.09 1.12 1.18 1.07 | 483 (80–100 TU) | 0 | 0.73–0.75 | 0.68 / 1.19 | −0.15 −0.07 −0.19 −0.18 −0.26 −0.25 |
| M3ᶠ λ = 0, WD 1e-1, s1 | 1.06 0.78 1.30 0.72 1.07 | 1.46 0.87 1.18 0.86 0.91 | **358** (40–60 TU) | 0 | 0.75–0.76 | 0.74 / 1.10 | −0.26 **−0.57** −0.08 −0.04 −0.30 −0.27 |

- **All runs are stable, with 0 clamp over 100 TU.**
- There is **no drift and no locked basin**. The offsets are a steady −0.1…−0.3 sd from about
  20 TU on, and E[0,6] sits at −0.5 sd at λ = 0.
- λ = 0 makes one deep Z[16,32] excursion, to 341 for M0 and 358 for M3ᶠ (reference minimum 721).
  It happens in the same 40–60 TU chunk, as the shared seed and IC would predict.
- M3ᶠ λ = 0 reproduces M0 λ = 0 chunk by chunk: same E[0,6] offset, KS 0.74 against 0.65.
- The M3ᶠ λ = 1e-5 run sits between its two M0 neighbours and is somewhat wider (sd 1.19).
- Nothing here separates M3ᶠ from M0.

### 11e. Verdict for the next stage

- **M3ᶠ does not pass stage 1 in any meaningful sense.**
  - At λ = 0 it passes the "no worse than M0 on CRPS" gate trivially, because it *is* M0 to within
    0.1%.
  - At λ = 1e-5 its offline gain is the ridge being undone, not a nonlinear mean.
  - Under the recommended kill criterion (D-13), the mini-D6 is expected to show no resolvable
    paired difference for λ = 0 M3ᶠ. That is the "stop and talk to Rik" branch.
- **If the mini-D6 is run anyway, it is cheap to include:**
  - (i) the pair **M0@50 λ = 0 vs M3ᶠ λ = 0, WD 1e-1, s1** — the L1 test proper; expect a null;
  - (ii) **M0@50 λ = 1e-5** — the smallest-bias M0 online here.
  - M3ᶠ λ = 1e-5 WD 1e-2 is effectively a partially un-ridged linear map, a point *between* M0(0) and
    M0(1e-5). It should not be read as L1.
- Not done here and worth one line to Rik: M3ᶠ with **more memory** (W = 10 or a recurrence on the
  h = 2 regressor) against the stacked-window linear floor (0.2074). That is the only place §7c's
  +3.4% could live, and it would be a memory effect, not a nonlinear-mean effect.

### 11f. Fits and runs (all under `exp_square_HIT/output/TO_LSTM/p4/`)

- M0@50: `m0_50_h2_l0`, `m0_50_h2_l1e-05`, `m0_50_h2_l0.0001`, each with 20 TU replicas 1–3.
  - `m0_50_h2_l0` and `m0_50_h2_l1e-05` also have a 100 TU replica 1.
- M3ᶠ grid (12 fits): `m3f_l{0,1e-5}_wd{1e-2,1e-1}_s{1,2,3}`.
  - 20 TU × 3 for `m3f_l0_wd1e-1_s1` and `m3f_l1e-5_wd1e-2_s2` (the median seeds by held-out CRPS).
  - 100 TU r1 for `m3f_l1e-5_wd1e-2_s2`.
- Pilots, not part of the grid:
  - `m3f_l0_wd0_s1` (default stopping, 2000 updates);
  - `pilot_a…j` (W, `n_hidden`, LR, batch, WD scans at λ = 0 / 1e-5);
  - `probe_m0_l1e-4` (1 epoch; only its update-0 = M0(1e-4) scores are used).

### 11g. Reproduce

```bash
cd lib/RikFlow
# M0@50 (closed form, seconds)
RH=2 LAMS=0,1e-5,1e-4 TRAIN_TU=50 SCORE_TU=52,74 OUTSUB=p4 TAGPFX=m0_50 \
  julia --project=training exp_square_HIT/tools/m4_linear_eta.jl
# one M3f fit (~1.5 min CPU); the grid is LAMBDA in {0,1e-5} x WD in {1e-2,1e-1} x SEED in {1,2,3}
RIKFLOW_W_TAG=m3f_l0_wd1e-1_s1 RIKFLOW_W_OUTSUB=p4 RIKFLOW_W_ARCH=dense RIKFLOW_W_NZ=0 RIKFLOW_W_NH=16 \
RIKFLOW_W_W=1 RIKFLOW_W_H=2 RIKFLOW_W_HIST_VAR=q_star_q RIKFLOW_W_EMISSION=constant RIKFLOW_W_SEEDHEAD=1 \
RIKFLOW_W_LAMBDA=0 RIKFLOW_W_WD=1e-1 RIKFLOW_W_WD_EXCLUDE=bd,Araw RIKFLOW_W_SEED=1 \
RIKFLOW_W_TRAIN_TU=50 RIKFLOW_W_SCORE_TU=52,74 RIKFLOW_W_BATCH=256 RIKFLOW_W_VAL_EVERY=100 \
RIKFLOW_W_PATIENCE=30 RIKFLOW_W_STOP_PATIENCE=80 RIKFLOW_W_STOP_WINDOW=20000 RIKFLOW_W_EPOCHS=400 \
  julia --project=training exp_square_HIT/tools/m4_window_fit.jl
# online smoke (GPU, ~3.5 min alone, ~7 min with 3 concurrent); TSIM=100 for stage 4
JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB RIKFLOW_ONLINE_TSIM=20 \
RIKFLOW_ONLINE_IC=$PWD/exp_square_HIT/output/online_ic_data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2 \
RIKFLOW_M4_MODEL_DIR=$PWD/exp_square_HIT/output/TO_LSTM/p4/m3f_l0_wd1e-1_s1 \
  script -qfec "julia --project=. exp_square_HIT/12_online_StochLSTM.jl 2 1" log
REF_TU_MAX=74 TSCREEN=20 M4_SCREEN_SUBDIR=p4 julia --project=analysis analysis/m4_screen.jl
REF_TU_MAX=74 julia --project=analysis analysis/m4_online_moments.jl p4/m0_50_h2_l0 p4/m3f_l0_wd1e-1_s1
REF_TU_MAX=74 julia --project=analysis analysis/m4_screen_long.jl p4/m0_50_h2_l1e-05 p4/m3f_l1e-5_wd1e-2_s2
```

```bash
# the linear share of g (11a) and the skip-residual ACF (11c); both hold the held-out block at 52-74 TU
julia --project=training exp_square_HIT/tools/m4_linfrac.jl m3f_l1e-5_wd1e-2_s1 m3f_l0_wd1e-1_s2
julia --project=training exp_square_HIT/tools/m4_residacf.jl m0_50_h2_l0 m0_50_h2_l1e-05 m0_50_h2_l0.0001
```

- **Code added this step (additive; all defaults reproduce old runs):**
  - `LSTMSpec` accepts `arch = :dense, n_latent = 0`, and refuses it without an emission head or
    with `posterior = :xy`.
  - `train_stochlstm(...; decay_exclude)`.
  - `m4_window_fit.jl`: `SCORE_TU`, `OUTSUB`, `WD_EXCLUDE`; prints the train, validation and score
    windows, the M0(λ) = update-0 scores, ‖g‖ ratios, and the stacked full-regressor floor.
  - `m4_linear_eta.jl`: `TRAIN_TU`, `SCORE_TU`, `OUTSUB`, `TAGPFX`.
  - `m4_diag_fit.jl`: `SCORE_TU`.
  - New: `tools/m4_linfrac.jl` and `tools/m4_residacf.jl`.
  - `m4_screen.jl`, `m4_screen_long.jl`, `m4_online_moments.jl`: `REF_TU_MAX`.
  - Test V71: 46 new tests. The Lux suite is **841/841**; the stdlib suite is 2071 pass / 8 broken.

## 12. M0ᶜ-ridge: LinReg7 + AR(2) residual, mini-D6 (2026-09-28)

**Question** (from §10g/§10h): LinReg7's residual is coloured, and its white Σ under-delivers kernel-weighted
low-frequency power by ~3×. Does sampling that residual as a stationary AR(2) fix LinReg7's D6
under-dispersion (spread–skill median 0.559) without costing CRPS or stability?

**Answer.** It fixes the calibration and improves CRPS by a clearly resolved margin. But it fails the
declared gate-census clause: one member of 230 collapsed to a low-energy state. There is also a
low-energy tail across the ensemble (11 vs 0 members with a QoI below half the truth at 1 TU).
**Verdict: not a pass as declared** (four of five clauses met). It is a strong candidate whose one
failure is a stability signal, and that signal must be run down before promotion.

### 12a. Code

- **`src/time_series_methods.jl`, `LinReg`: optional AR(p ≤ 2) residual.** The model is
  η_n = μ_η + z_n, with z_n = Σ φ_k ⊙ z_{n−k} + ξ_n and ξ ~ N(0, Σ_ξ). μ_η is `mean(stoch_distr)`. The AR is
  diagonal. It lives in scaled units and enters exactly where the white draw enters.
  - Two new optional JLD2 keys: `ar_phi` (p × 6) and `ar_sigma_xi` (6 × 6).
  - Malformed or non-stationary AR keys are refused.
  - **Without the keys, the behaviour is bit-identical, RNG stream included.** The white branch is the same
    `rand(rng, stoch_distr)` call. It now sits inside `draw_eta`.
  - The AR path makes one `MvNormal(0, Σ_ξ)` draw per step, the same number of normals as the white path.
    So the member seeds give common random numbers across the two variants.
- **Warm start.** The replay is unchanged (dQ is emitted verbatim).
  - On the last p warm-up steps whose history is full, the model also computes the realised residual,
    `scale(q* + dQ) − c[x; 1] − μ_η`, from its own history and pushes it into the AR state.
  - Fallback: if fewer than p lags were filled (warm-up shorter than h + p), the state is replaced at the
    first prediction by 2000 burn-in steps of the recursion. This never happens in D6 (nwarm = 100).
- **Gate.** When `TURBULENCE_GATE` zeroes dQ, the AR state still advances. The draw comes before the gate, so
  the member's stream is not shifted.
- **`tools/run_d6.jl`** records `ar_order` and `ar_phi` in every member file and prints them. The run
  identity already separates the variants, because `model_name` is the model directory
  (`LinReg7` vs `LinReg7_ar2`).

### 12b. Variants built

Built with `exp_square_HIT/tools/lrs_ar_variant.jl`.
- Each variant directory holds a **byte copy of LinReg7's `LinReg.jld2`** plus the keys `ar_phi`,
  `ar_sigma_xi` and `ar_provenance`, and a copy of `parameters.jld2`. Every source key was checked equal.
- The AR is fitted to LinReg7's own scaled residual on steps 400–4000 (1–10 TU), evaluated as deployed.
- Σ_ξ = D R D:
  - R is the lag-0 correlation of the implied innovations (off-diagonal −0.39…0.76).
  - D matches the AR's stationary marginal variance to var(z).
- `var(z)` equals the deployed white Σ's diagonal to ≤ 0.5%.

**`LinReg7_ar2`** (`ar2_ls` on ACF lags 1–20):

| QoI | ρ₁(resid) | φ₁ | φ₂ | poles \|·\| | σ_ξ/sd(z) |
|---|---|---|---|---|---|
| Z[0,6] | 0.772 | 0.446 | 0.358 | 0.862, 0.416 | 0.672 |
| E[0,6] | 0.438 | 0.451 | −0.047 | 0.290, 0.162 | 0.901 |
| Z[7,15] | 0.948 | 1.221 | −0.340 | 0.792, 0.429 | 0.387 |
| E[7,15] | 0.967 | 1.608 | −0.669 | 0.818, 0.818 | 0.200 |
| Z[16,32] | 0.810 | 0.389 | 0.427 | 0.876, 0.487 | 0.665 |
| E[16,32] | 0.854 | 0.833 | −0.054 | 0.762, 0.071 | 0.612 |

All poles are ≤ 0.88: well inside the unit circle, unlike the likelihood fits of §10h(2).

**Kernel-weighted power, P/P_white (full 6×6 kernel, raw units):**
- AR(2): **2.04–2.99**. The data residual gives 2.72–3.18. This reproduces §10h (2.09–3.02).
- As built (Σ_ξ against the deployed white Σ): 1.92–3.34.
- Diagonal kernel: AR(2) gives 8.4 / 2.2 / 6.8 / 9.9 / 7.2 / 6.8, identical to §10h.

`LinReg7_ar1` (φ = ρ₁) is also built. Its full-kernel power is 2.35–4.23, overshooting in the middle bands
as §10h found.

**Offline sanity.** The deployed `RikFlow.LinReg` was run on the new file, teacher-forced on 1–10 TU.
- The warm-start state equals the data residual to 1.3e-15.
- The model noise ACF (lags 1/2/5/10) reproduces the fitted AR ACF to ≤ 0.05. Examples: Z[7,15] 0.91 0.76
  0.36 0.10 against the AR's 0.91 0.77 0.41 0.13; E[7,15] 0.96 0.87 0.49 0.02 against 0.96 0.88 0.53 0.10.
- **Marginal spread is calibrated**: sd(model)/sd(resid) = 0.98–1.03.
- The fit's lag-1 is below the residual's in the middle bands (0.91/0.96 against 0.95/0.97), because it is an
  ACF least-squares fit over 20 lags. So the data's one-step innovation under the fitted φ is smaller than
  σ_ξ:
  - xi_data/σ_ξ = 1.01 / 1.00 / **0.62 / 0.72** / 1.04 / 0.84.
  - So **one-step spread is over-dispersed ~1.4–1.6× in Z/E[7,15]**, by construction. The marginal
    variance, not the one-step variance, was matched.

### 12c. Tests

- New `test/test_linreg_ar.jl` (V73): **84/84**. It includes `test/legacy_linreg.jl`, a verbatim copy of the
  pre-AR `LinReg`, as the oracle. It covers:
  - bit-identity with the legacy code, on synthetic files and on the archived LinReg1/LinReg7 files: dQ,
    q_hist, and the RNG position, with gate firings included;
  - that the AR(2) path gives the AR ACF, marginal variance and Σ_ξ;
  - the warm start from the replay, the "replay draws nothing" property, the first-step recursion, and the
    short-warm-up fallback (with the RNG consumption checked);
  - that the gate zeroes dQ, still advances the state, and does not shift the stream;
  - the file round trip, the built variants, and that malformed or non-stationary AR is refused.
- `test_sources.jl`: **41 pass, 1 broken** (pre-existing). V38 now accepts `draw_eta(` as the draw site.
- `test_d6_ics/lstm/score.jl`: 508 pass, 1 broken.
- **Online, same draws.** The fresh `D6mini_LinReg7` run was compared with the existing full-length
  `D6_LinReg7` run: same member seeds, 95 members whose full run ends by 74 TU.
  - dQ agrees to a max relative difference of **6.6e-9** over all 500 columns.
  - It is not bitwise equal: GPU round-off, and the IC's recomputed QoIs already differ at 3.5e-16.
  - A different RNG stream would give O(1) differences from the first forecast column. So the online no-AR
    path makes the same draws as the pre-AR code.
- The warm-up is bit-identical in **230/230** members of each run (the node-side check in `run_d6.jl`).

### 12d. Mini-D6

- Setup: selection block, K = 46 (ordinals 88–133), M = 5, nlead = 400, the same member seeds. Paired by IC
  with common random numbers.
- Both runs took ~19–47 s per member depending on how many other sims were on the card.
- No member diverged in either run.

**Primary score** (`score_d6.jl --paired D6mini_LinReg7_ar2 D6mini_LinReg7`: fair CRPS, leads ≤ 0.5 TU, 6
standardised bands):

| | A = LinReg7_ar2 | B = LinReg7 | A − B | 90% IC-block CI |
|---|---|---|---|---|
| CRPS | 0.17966 | 0.19870 | **−0.01904 (−9.6%)** | **[−0.02766, −0.01209]** |

Per band A − B: −0.021 / −0.015 / −0.018 / −0.019 / −0.021 / −0.020. The improvement is in every band.

**Spread–skill on the level q** (30 cells, 6 bands × 5 leads 25–400 steps; S7 band [0.8, 1.25]):

| | median | in band | median \|log r\| | per-band medians (Z0 E0 Z7 E7 Z16 E16) |
|---|---|---|---|---|
| LinReg7 | 0.589 | 2/30 | 0.529 | 0.57 0.69 0.54 0.54 0.52 0.52 |
| **LinReg7_ar2** | **1.127** | **26/30** | **0.119** | 1.17 1.04 1.15 1.14 1.08 1.08 |

- The 4 out-of-band AR(2) cells are all **over**-dispersed (1.26–1.30), in Z/E[7,15] and E[16,32], at
  leads 25–100. This fits the middle-band one-step over-dispersion of 12b.
- LinReg7's own mini-D6 median, 0.589, agrees with the full D6 value of 0.559.

**Online dQ ACF against tracked on the same steps.** 215 members / 43 ICs whose run ends by 74 TU; lags 1 / 2 / 5 / 20.

| QoI | tracked | LinReg7 | **LinReg7_ar2** | sd(dQ) on/trk, LR7 / ar2 |
|---|---|---|---|---|
| Z[0,6] | 0.93 0.88 0.72 0.32 | 0.31 0.45 0.33 0.14 | **0.84 0.82 0.63 0.20** | 0.84 / 1.00 |
| E[0,6] | 0.73 0.53 0.21 −0.01 | 0.38 0.34 0.14 0.00 | **0.71 0.54 0.23 0.00** | 0.99 / 1.16 |
| Z[7,15] | 0.97 0.91 0.66 0.17 | −0.02 0.25 0.20 0.08 | **0.93 0.86 0.67 0.17** | 1.17 / 1.09 |
| E[7,15] | 0.98 0.95 0.76 0.15 | −0.01 0.31 0.23 0.07 | **0.98 0.94 0.74 0.12** | 0.99 / 1.05 |
| Z[16,32] | 0.99 0.97 0.95 0.82 | 0.89 0.91 0.89 0.78 | **0.97 0.97 0.94 0.80** | 1.12 / 1.32 |
| E[16,32] | 0.98 0.96 0.90 0.76 | 0.78 0.83 0.80 0.70 | **0.96 0.95 0.90 0.75** | 1.13 / 1.30 |

- AR(2) restores the correction's persistence: lag 1 is within 0.09 of tracked in every QoI.
- The lag 1 < lag 2 signature of §10h(3) is gone.
- The cost is a larger dQ amplitude in the small-scale bands (1.30× tracked, against 1.13×).

**Stability and gate census.**

| | diverged | gate firings (forecast steps) | members with a QoI < 0.5× truth at lead 1 TU | mean level bias at lead 1 TU (sd units) |
|---|---|---|---|---|
| LinReg7 | 0/230 | 0 / 92000 | 0 / 215 | −0.04 … −0.07 |
| LinReg7_ar2 | 0/230 | **3 / 92000** (1 member) | **11 / 215** | **−0.05 … −0.16** |

The gate firings are all in `ic260_m1` (t_k = 64.75 TU), at leads 397–400, all through E[16,32] = 0.0097–0.0100.
- That member drained its energy: Z[0,6] 316 against the truth's 860, Z[16,32] 260 against 1640.
- Its same-seed LinReg7 twin also drifted low (Z[0,6] 675, E[16,32] 0.036) but stayed above the gate.
- The AR's extra low-frequency power carries the same draws further. The bias columns and the 11-member
  low tail show that this is a skewed low-energy tail, not one bad member.
- As `TURBULENCE_GATE`'s docstring says, a firing on HIT means the run has left the attractor.

### 12e. Verdict against the pass criterion (declared before looking)

| clause | result | |
|---|---|---|
| spread–skill median closer to 1 | 0.589 → 1.127 (\|log\| 0.529 → 0.119) | ✅ |
| more cells in [0.8, 1.25] | 2 → 26 of 30 | ✅ |
| paired CRPS: 90% CI upper bound ≤ +0.005 | −0.0190, CI [−0.0277, −0.0121] | ✅ (A better, resolved) |
| no diverged members | 0 / 230 | ✅ |
| gate census not worse | 0 → 3 firings (1 member) | ❌ |

**Not a pass as declared.**
- Calibration and CRPS improve by far more than the power analysis could resolve (≈ 0.016; the observed
  effect is 0.019 with a CI clear of 0).
- The failing clause is small in count, but it is backed by a real low-energy tail (11 vs 0 members at
  < 0.5× truth) and a 2–3× larger mean negative level bias at 1 TU.
- This is the loop-gain risk §10h flagged, in a milder form: the poles are stationary, but more power at low
  frequency lets closed-loop energy drain further.

Next checks before promoting M0ᶜ-ridge:
- (i) Look at the low tail over the full 3 TU D6 horizon: does it grow or recover?
- (ii) Test a variance-matched AR whose one-step σ_ξ is also calibrated in the middle bands, which would
  remove the 1.26–1.30 over-dispersed cells. For example, a Yule–Walker lag-1-exact AR(2), or ar2_ls with a
  lag-1 constraint.
- (iii) Check whether the tail is a property of the E[16,32] channel. E[16,32] parks low in the gate
  failures here, as in the 2026-09-15 runs.

⚠️ The selection-block forecasts read the truth up to 75.25 TU inside the scorer. That is in the 74–76 TU
embargo, not the confirmation block, and is the plan §7 protocol. Every analysis in this section beyond the
scorer is restricted to runs ending by 74 TU.

### 12e′. The AR(1) comparator (`LinReg7_ar1`), scored after the verdict (2026-09-28)

Same mini-D6: selection block, K = 46, M = 5, nlead 400, same seeds. AR(1) at each QoI's residual lag 1
(φ = 0.772 0.438 0.948 0.967 0.810 0.854), with the marginal variance matched.

| | LinReg7 | **LinReg7_ar1** | LinReg7_ar2 |
|---|---|---|---|
| paired CRPS vs LinReg7 | — | **−0.0120 (−6.0%)**, 90% CI [−0.0225, −0.0049] | −0.0190 (−9.6%), [−0.0277, −0.0121] |
| spread–skill on q: median / in [0.8, 1.25] | 0.589 / 2 of 30 | **≈ 1.14 / 21 of 30** (9 misses, all over-dispersed at 1.27–1.32) | 1.127 / 26 of 30 |
| diverged | 0 / 230 | 0 / 230 | 0 / 230 |
| gate firings (forecast steps) | 0 | **19 / 92000** | 3 / 92000 |
| dQ lag 5, Z/E[7,15] (tracked 0.66 / 0.76) | 0.20 / 0.23 | **0.82 / 0.88** (too persistent) | 0.67 / 0.74 |
| sd(dQ) on/trk, Z/E[7,15] | 1.17 / 0.99 | 1.23 / 1.23 | 1.09 / 1.05 |

**Reading.**
- AR(1) is better than LinReg7 on CRPS and calibration, but worse than AR(2) on every row.
- It overshoots the middle bands' persistence at lags ≥ 5. The negative lobe that AR(2) reproduces is missing
  (§10d: AR(1) has 3.7–4.2× white power against the data's 3.1–3.2).
- It fires the gate 6× as often as AR(2).
- This confirms the §12e reading: the low-energy tail grows with the noise's low-frequency power. AR(2) is the
  better of the two; neither passes the gate clause as declared.
- Scored by the main session. The logs `score_ar1_paired.log`, `score_ar1_main.log` and `census_ar.log` are in
  job scratch.

### 12f. Reproduce

```bash
cd lib/RikFlow
# variants (+ fit tables, kernel power, offline sanity); --report re-prints without writing
julia --startup-file=no --project=training exp_square_HIT/tools/lrs_ar_variant.jl LinReg7 2 1
# tests
julia --project=test -e 'using TestItemRunner; TestItemRunner.run_tests("test"; filter = t -> endswith(t.filename, "test_linreg_ar.jl"))'
# mini-D6 (from exp_square_HIT/), once per V in LinReg7, LinReg7_ar2
JULIA_CUDA_SOFT_MEMORY_LIMIT=4GiB D6_CLOSURE=lrs D6_MODEL=$PWD/output/TO_LRS/$V/LinReg.jld2 \
  D6_BLOCK=selection D6_MEMBERS=5 D6_NLEAD=400 D6_OUT=$PWD/../analysis/output/D6mini_$V \
  julia --startup-file=no --project=.. tools/run_d6.jl --all
# scoring
julia --startup-file=no --project=analysis analysis/score_d6.jl --paired analysis/output/D6mini_LinReg7_ar2 analysis/output/D6mini_LinReg7
D6_OUT=analysis/output/D6mini_LinReg7_ar2 julia --startup-file=no --project=analysis analysis/score_d6.jl   # and _LinReg7
julia --startup-file=no --project=training analysis/m0c_ar_online.jl D6mini_LinReg7 D6mini_LinReg7_ar2 --vs-full D6_LinReg7
```
The logs are in the job scratch: `lrs_ar_variant*.log`, `logs/D6mini_*.log`, `score_*.log`, `m0c_ar_online.log`.
The level-bias and low-tail census is a short ad-hoc script: for runs ending by 74 TU, it computes
(member − truth)/sd(truth over 1–74 TU) at leads 100/200/400. Since §13 it is `analysis/tail_census.jl`.

## 13. M0ᶜ-ridge to promotion or rejection, and the rest of the ladder (2026-09-28 evening → night)

Order and rationale: `plan.md` → *Start here → 2026-09-28 (evening)*. Rik, 2026-09-28: judge the
low-energy tail **against both** LinReg7 (the declared, same-family reference) and LinReg1 (the
target). Branch `overnight-0928` (from `upstream-merge` @ `6de5deea`).

**Where §13 ends (2026-09-29).** All scores are paired fair CRPS at leads ≤ 0.5 TU on the
selection block, against LinReg1. The rows marked M = 10 are the decisive ones.

| model | what it is | CRPS vs LinReg1 | calibrated cells | tail | § |
|---|---|---|---|---|---|
| LinReg1 | λ = 0, white noise (the target) | — | 24/30 | 22 / 430 | g |
| **Splice1_E0x7_ar2** (M = 10) | LinReg1 + E[0,6] row at λ = 1 + AR(2) | **−0.02 %** (±4.2 %) | **30/30** | **13** / 430 | g |
| LinReg14_ar2 (M = 10) | paper-3 per-QoI λ, base 0.03, + AR(2) | +0.8 % | 22/30 | 15 / 430 | i |
| LinReg15_ar2 (M = 10) | paper-3 per-QoI λ, base 0.1, + AR(2) | +2.4 % | 25/30 | 12 / 430 | i |
| LinReg11_ar2 (M = 5) | single λ = 0.1 + AR(2) | +2.3 % | 30/30 | 6 / 215 | e |
| LinReg7_ar2 (M = 5) | single λ = 1 + AR(2) | +5.0 %, resolved | 26/30 | 11 / 215 | b′ |
| LinReg7 (M = 5) | single λ = 1, white | +16 %, resolved | 2/30 | 0 / 215 | b′ |
| r3_lin_sd_h2 (M = 5) | h = 2 skip + LSTM noise scale | +18.5 %, resolved | 17/30 | 0 / 215 | h |

### 13a. Where the tail lives (check (c)) — the small-scale band

`analysis/tail_census.jl` (new): per QoI and lead, members with q < 0.5× / 0.25× the tracked truth,
the mean level bias (q − truth)/sd(truth, 1–74 TU), the worst QoI per low member, and whether
members low at 1 TU are still low at the last lead. The q-to-truth column offset is taken from the
replayed warm-up and asserted, not assumed. Only members ending by 74 TU are read.

Mini-D6, 1 TU, 215 members / 43 ICs each (count < 0.5× truth at leads 0.25 / 0.5 / 1 TU):

| QoI | LinReg7 | LinReg7_ar2 | LinReg7_ar1 |
|---|---|---|---|
| Z[0,6] | 0 / 0 / 0 | 0 / 2 / 2 | 0 / 3 / 2 |
| E[0,6] | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 2 / 1 |
| Z[7,15] | 0 / 0 / 0 | 0 / 0 / 4 | 0 / 1 / 5 |
| E[7,15] | 0 / 0 / 0 | 0 / 1 / 3 | 0 / 1 / 5 |
| **Z[16,32]** | 0 / 0 / 0 | 0 / 1 / **11** | 0 / 1 / **16** |
| **E[16,32]** | 0 / 0 / 0 | 0 / 1 / **10** | 0 / 1 / **13** |
| worst QoI of every low member | — | Z[16,32] in 11/11 | Z[16,32] in 16/16 |

- **The tail is the small-scale band**, Z[16,32] first and E[16,32] with it. It is absent at
  0.25 TU: the forecast builds it; the IC does not carry it.
- 🔑 **The target has a tail of its own.** Full D6, all 630 members ending by 74 TU (63 ICs, M = 10,
  mostly 10–50 TU ICs, which the selection block does not use):

  | lead | LinReg1: Z[16,32] < 0.5× (< 0.25×) | LinReg7 |
  |---|---|---|
  | 0.5 TU | 2 (0) | 0 |
  | 1 TU | 23 (1) | 6 (0) |
  | 2 TU | 37 (9) | 5 (0) |
  | 3 TU | 38 (10) | 7 (0) |

  LinReg1 also has 2 divergences and many gate firings (3507 all-zero forecast dQ columns). Its
  worst QoI is Z[16,32] in 35 of 40 low members. In both LinReg1 and LinReg7, **no member low at
  1 TU is still low at 3 TU**: they recover, and the 3 TU low members are new ones.
- So in LinReg1's terms, "LinReg7_ar2 has a low-energy tail" reads "LinReg7_ar2 moved toward
  LinReg1's small-scale behaviour". The like-for-like comparison is 13b.

### 13b. The 3 TU horizon, same 19 ICs × 10 members as the full D6 (check (a))

`D6sel3_LinReg7_ar2`: odd ordinals 89–125 (t 52.75–70.25 TU, K = 19), M = 10, nlead 1200, the full
D6's member seeds — so it pairs member-for-member with `D6_LinReg1` and `D6_LinReg7`, restricted to
the same 19 ICs (`D6_T_MIN=52.5 D6_T_MAX=70.75` at scoring). 190 members, ~26 s each, 0 diverged.

**Primary score** (paired fair CRPS, leads ≤ 0.5 TU, 90 % IC-block CI, block 2 ICs):

| A − B | A | B | A − B | 90 % CI | reading |
|---|---|---|---|---|---|
| LinReg7_ar2 − **LinReg7** | 0.2006 | 0.2167 | **−0.0160 (−7.4 %)** | [−0.0273, −0.0051] | A better, resolved (as §12) |
| LinReg7_ar2 − **LinReg1** | 0.2006 | 0.1789 | **+0.0218 (+12.2 %)** | [−0.0018, +0.0452] | A worse, not resolved (CI just touches 0) |

Per band vs LinReg1: +0.045 (Z[0,6]), +0.012, +0.021, +0.018, +0.017, +0.018 — worse in every band,
most in Z[0,6].

**Spread–skill on q** (36 cells = 6 bands × leads 25, 50, 100, 200, 400, 1000; S7 band [0.8, 1.25]):

| | cells in band, leads ≤ 1 TU | median ratio, leads ≤ 1 TU |
|---|---|---|
| LinReg7 | 2/30 | 0.57 |
| **LinReg7_ar2** | **25/30** | 1.03 |
| LinReg1 | 24/30 | 1.12 |

LinReg7_ar2 at 2.5 TU is over-dispersed (1.8–1.9 in five QoIs), where the skill has already
saturated below the spread. The counts come from `ssband.sh` (job scratch) on the scorer's
LEVEL-q tables.

**Tail, same 190 members** (count < 0.5× truth, with < 0.25× in brackets; gate = all-zero forecast
dQ columns, 228 000 forecast steps per run):

| lead | LinReg7 Z[16,32] | **LinReg7_ar2** Z[16,32] | LinReg1 Z[16,32] |
|---|---|---|---|
| 0.5 TU | 0 | 3 | 2 |
| 1 TU | 0 | 7 (1) | 3 |
| 2 TU | 0 | 5 | 8 (5) |
| 3 TU | 1 | 9 (1) | 13 (5) |
| low at 1 TU → still low at 3 TU | — | 1 of 7 | 0 of 4 |
| gate firings | 0 | 111 | 1842 |
| any QoI < 0.5× at 3 TU (members) | 1 | 9 | 15 |

**Verdict, judged against both (Rik).**
- **Against LinReg7** (the declared clause): the AR costs stability. The tail goes 1 → 9 members at
  3 TU and 0 → 111 gate firings, for −7.4 % CRPS and 2 → 25 calibrated cells. The declared clause
  "gate census not worse" fails, as in §12.
- **Against LinReg1** (the target): the tail is **smaller** (9 vs 15 members at 3 TU; 1 vs 5 below
  0.25×; 111 vs 1842 gate firings) and the calibration equal (25 vs 24 of 30), but the CRPS is
  **12 % worse** (not resolved at K = 19; the point estimate is the size of LinReg1's lead over
  LinReg7).
- 🔑 **So LinReg7_ar2 does not beat the target. It is a new point on S2′'s front, between LinReg1
  and LinReg7:** LinReg1's calibration, most of LinReg7's stability, and a CRPS 40 % of the way
  from LinReg7 back to LinReg1. The remaining gap is the **mean**: ridge at λ = 1 shrinks it, and
  a residual model cannot restore mean skill. The next cells therefore fill the λ gap between
  1e-2 and 1 with the AR on top (LinReg11 λ = 0.1, LinReg12 λ = 0.3; 13e).

### 13b′. The target on the mini-D6 itself: LinReg1, 46 selection ICs, M = 5, 1 TU

`D6mini_LinReg1` has the same ICs and member seeds as every `D6mini_*` run. Paired primary score
(90 % IC-block CI, block 3 ICs) and the calibration and tail on the same 215 members that end by 74 TU:

| A vs B = LinReg1 | A − B | 90 % CI | per band A − B (Z0 E0 Z7 E7 Z16 E16) | spread–skill in band / median (leads ≤ 1 TU) | Z[16,32] < 0.5× at 1 TU | gate firings |
|---|---|---|---|---|---|---|
| LinReg1 | — | — | — | 25/30, 1.10 | 14 (1 < 0.25×) | 0 |
| LinReg7 | **+16.1 %** (LinReg1 −13.9 % of LinReg7) | resolved | +.037 **−.012** +.035 +.033 +.037 +.037 | 2/30, 0.59 | 0 | 0 |
| LinReg7_ar1 | **+9.1 %** | [+0.0068, +0.0249] | +.018 **−.025** +.026 +.026 +.024 +.024 | 21/30, 1.14 | 16 | 19 |
| **LinReg7_ar2** | **+5.0 %** | [+0.0003, +0.0195] | +.015 **−.028** +.016 +.015 +.016 +.016 | 26/30, 1.13 | 11 (1) | 3 |

- **The target keeps the lead on CRPS by a resolved 5 %.** LinReg7_ar2 matches its calibration
  (26 vs 25 of 30) and its 1 TU tail (11 vs 14), as on the 3 TU set (§13b). It is a front point,
  not a win.
- 🔑 **E[0,6] is the exception, in every ridge cell.** Ridge helps E[0,6] and hurts the other five
  bands; the AR widens the E[0,6] gain (−0.012 → −0.028). The plan's band table already had "DDN
  beats LinReg1 on E[0,6]". Multi-output ridge solves each output row independently, so **λ can be
  chosen per QoI row** at no cost: LinReg1's rows where λ = 0 is best, and a ridge + AR row where
  ridge is best. The E[0,6] gain alone is worth −0.028/6 ≈ −0.005 on the primary score (≈ −2.7 %).
  The λ-gap runs (13e) give the per-band optimum.
- The 1 TU census in 13a said "LinReg7 has no tail"; LinReg1 on the same members has 14. The AR
  moves LinReg7 toward LinReg1's small-scale behaviour, not past it.

### 13c. Variants built this session

`exp_square_HIT/tools/lrs_ar_variant.jl` gained `2c`: **AR(2) with the residual's lag 1 exact**
(`ar2_ls_lag1`: φ1 = r1(1 − φ2), φ2 by least squares on the ACF at lags 2–20). The marginal variance
is matched as before.

| variant | kernel power AR / white Σ (full 6×6, as built) | one-step calibration xi_data/σ_ξ, six QoIs (1 = calibrated) |
|---|---|---|
| LinReg7_ar2 (§12) | 1.92–3.34 | 1.01 1.00 **0.62 0.72** 1.04 0.84 |
| **LinReg7_ar2c** | 1.89–3.29 | 1.04 1.00 **0.75 0.78** 1.06 0.96 |
| LinReg2_ar2 (λ = 1e-2) | **1.03–1.08** | 1.00 1.00 0.85 1.09 1.00 1.00 |
| LinReg2_ar2c | 1.03–1.07 | 1.00 1.00 0.93 0.97 1.00 1.00 |
| LinReg11_ar2 (λ = 0.1, new) | 1.18–1.45 | residual lag 1: 0.40 0.13 0.87 0.93 0.52 0.63 |
| LinReg12_ar2 (λ = 0.3, new) | 1.44–2.01 | residual lag 1: 0.59 0.26 0.92 0.95 0.67 0.76 |
| LinReg8_ar2 (λ = 10) | **2.87–6.78** | 0.90 0.95 0.57 0.62 0.80 0.85 |
| LinReg8_ar2c | 2.87–6.72 | 0.99 0.99 0.69 0.67 0.97 0.94 |

- The lag-1 constraint fixes the outer bands' one-step spread but only halves the middle bands'
  over-dispersion. The data's lag 2 (0.82 in Z[7,15]) sits below what a smooth AR(2) with lag 1 = 0.95
  can produce: there is a white "nugget" on top of the persistent part, which an AR(2) cannot carry
  without an extra (ARMA) term.
- **λ = 1e-2 barely colours the residual** (middle bands only, lag 1 0.76/0.83). The AR adds 3–8 %
  power, so LinReg2_ar2 should score as LinReg2. **λ = 10 colours everything** (poles 0.82–0.92), and
  the AR adds 3–7× the white power: the largest correction and the largest tail risk.
- Marginal spread is calibrated in every variant (sd model/resid 0.97–1.03).

**Check (b), the lag-1-exact AR(2) on the mini-D6: a null.** `LinReg7_ar2c` against `LinReg7_ar2`:
CRPS +0.00027 (+0.15 %), CI [−0.00075, +0.00097]. Per band ≤ 0.0006. Spread–skill 25/30 in band,
median 1.15 (ar2: 26/30, 1.13). The over-dispersed cells do not go away; they move (Z/E[7,15] at
leads 25 and 100: 1.27–1.33). Tail 12 vs 11 members at 1 TU; gate 7 vs 3. Against LinReg1: +5.1 %,
CI [+0.0002, +0.0198], as ar2. **The one-step innovation calibration does not drive the ensemble
score; `ar2_ls` stays the construction.**

### 13c′. Per-QoI λ: splicing coefficient rows (new tool)

`exp_square_HIT/tools/lrs_splice.jl <dst> <src_1> … <src_6>` takes row i of `c` from `src_i` and
refits the Gaussian noise (MLE mean and covariance) on the spliced residual. Because
`5_train_LinReg.jl`'s ridge solves one least-squares problem per output column, with the same design,
penalty and data scaling, this is **exactly** a ridge fit with a per-QoI λ. Only the noise couples
the rows. The training rows are rebuilt with `create_history`'s construction, and every run checks
that each source's stored `stoch_distr` is reproduced (LinReg1: 2.1e-16 / 2.3e-16; LinReg7:
1.5e-17 / 3.4e-16, |Δμ|/sd and |ΔΣ|/|Σ|).

First cell, from §13b′'s per-band read: **`Splice1_E0x7`** = LinReg1 with LinReg7's E[0,6] row, and
`Splice1_E0x7_ar2`. The AR fitted to the spliced residual is null on LinReg1's five rows (φ ≈ 0,
white at h = 5 as §10c found) and φ = (0.451, −0.047) on E[0,6], identical to LinReg7_ar2's E[0,6]
row. The deployed `LinReg` loads the file and warm-starts to 6e-15. Both are queued for the mini-D6.

### 13e. The λ gap with AR(2), mini-D6 (46 selection ICs, M = 5, 1 TU)

All paired with the same member seeds; tail = members with a QoI < 0.5× truth at 1 TU (215 members
ending by 74 TU); spread–skill = the 30 level cells at leads ≤ 1 TU.

| cell (λ) | vs **LinReg1**: A − B, 90 % CI | per band vs LinReg1 (Z0 E0 Z7 E7 Z16 E16) | vs LinReg7 | spread–skill in band / median | tail | gate |
|---|---|---|---|---|---|---|
| LinReg1 (0) | — | — | −13.9 % | 25/30, 1.10 | 14 | 0 |
| LinReg13_ar2 (0.03) | +1.71 %, [−0.0029, +0.0085], not resolved | +.003 −.016 +.006 +.003 +.011 +.011 | — | 29/30, 1.11 | 7 | 0 |
| LinReg11 (0.1), white noise | +4.79 %, [+0.0016, +0.0184], resolved | +.008 −.018 +.013 +.010 +.017 +.018 | — | 15/30, 0.80 | 5 | 0 |
| **LinReg11_ar2 (0.1)** | **+2.3 %**, [−0.0024, +0.0121], not resolved | +.002 **−.020** +.009 +.007 +.013 +.013 | −11.9 % | **30/30**, 1.11 | **6** | **0** |
| LinReg12_ar2 (0.3) | +3.05 %, [−0.0020, +0.0157], not resolved | +.004 **−.024** +.012 +.010 +.015 +.015 | — | 29/30, 1.09 | 8 | 0 |
| LinReg7_ar2 (1) | +5.0 %, [+0.0003, +0.0195] | +.015 **−.028** +.016 +.015 +.016 +.016 | −9.6 % | 26/30, 1.13 | 11 | 3 |
| LinReg7 (1) | +16.1 % | +.037 **−.012** +.035 +.033 +.037 +.037 | — | 2/30, 0.59 | 0 | 0 |

- 🔑 **LinReg11_ar2 is the first cell this project has found that is competitive with the target on
  every axis at once.** CRPS is not resolvably worse than LinReg1 (+2.3 %, CI contains 0). It has
  **all 30 cells calibrated** against LinReg1's 25, **less than half LinReg1's low-energy tail**
  (6 vs 14 members), and 0 gate firings. Against the plan's own test ("better calibration or stability
  at no loss of skill") it is a candidate. Whether "not resolvably worse" is "no loss" is exactly what
  the CI bounds: at most +0.012 (+7 %).
- The per-band column says where the remaining CRPS cost sits: **E[0,6] gains (−0.020), Z[0,6] is
  level, and the middle and small-scale bands lose 0.007–0.013.** That is the per-QoI λ split
  §13b′ predicted, now with a λ-dependence: at 0.1 the loss outside E[0,6] is half what it is at 1.
  A splice taking E[0,6] (and possibly Z[0,6]) from the ridge cell and the rest from LinReg1 is the
  obvious next cell (§13c′).
- **The AR residual is a resolved gain at λ = 0.1**, not only at λ = 1: LinReg11 (white) −
  LinReg11_ar2 = +2.4 %, CI [+0.0013, +0.0090], positive in every band. It also moves calibration
  from 15/30 (median 0.80, under-dispersed) to 30/30. Ridge + white noise is under-dispersed at
  every λ ≥ 0.1 measured here; ridge + AR(2) is not.
- **A single λ never passes LinReg1.** With LinReg13 (λ = 0.03) the curve is +1.7 / +2.3 / +3.05 /
  +5.0 % at λ = 0.03 / 0.1 / 0.3 / 1. It approaches LinReg1 from above as λ → 0, and every point
  has better calibration (29–30/30) and half the tail. Only a per-QoI λ (§13f) gets ahead.
- **The λ trend is monotone** over 0.1 → 0.3 → 1: CRPS against LinReg1 +2.3 → +3.05 → +5.0 %.
  E[0,6]'s gain grows with λ (−0.020 → −0.024 → −0.028) and the other bands' loss grows faster.
  LinReg12_ar2 − LinReg11_ar2 = +0.7 % (CI [−0.0014, +0.0049]). So the optimum of a *single* λ is
  at or below 0.1. **LinReg13 (λ = 0.03)** was appended to `4_setup_search.jl`, fitted and queued with
  `_ar2`: middle-band residual lag 1 0.82/0.89, AR power ×1.06–1.17.
- ⚠️ **Selection-block multiplicity.** About ten cells are now being compared on the same 46
  selection ICs, so whichever cell ends up best has an optimistically biased margin (winner's
  curse). The protection is plan §7's design: the finalist goes to the **confirmation block**
  (76–97 TU, untouched), scored once. No cell here is claimed as beating LinReg1 on selection-block
  evidence alone.

### 13f. Per-QoI λ splices on the mini-D6 (M = 5) — ahead of LinReg1 at M = 5, level at M = 10 (§13g)

| cell | vs **LinReg1**: A − B, 90 % CI | per band vs LinReg1 (Z0 E0 Z7 E7 Z16 E16) | spread–skill in band / median | tail | gate |
|---|---|---|---|---|---|
| LinReg1 | — | — | 25/30, 1.10 | 14 | 0 |
| LinReg11_ar2 (single λ = 0.1) | +2.3 %, [−0.0024, +0.0121] | +.002 −.020 +.009 +.007 +.013 +.013 | 30/30, 1.11 | 6 | 0 |
| Splice11_S16x1_ar2 (LinReg11, Z/E[16,32] rows from λ = 0) | +2.03 %, [−0.0030, +0.0114] | +.001 −.020 +.009 +.006 +.012 +.012 | 29/30, 1.09 | 6 | 0 |
| Splice1_E0x11_ar2 (E[0,6] row from λ = 0.1) | −1.56 %, [−0.0088, +0.0042] | −.005 −.018 +.002 +.003 +.001 +.001 | 30/30, 1.09 | 7 | 0 |
| **Splice1_E0x7_ar2** (LinReg1, E[0,6] row from λ = 1, + AR(2)) | **−2.34 %**, [−0.0110, +0.0030], not resolved | **−.007 −.025** +.002 +.003 +.002 +.001 | **30/30, 1.07** | **7** | **0** |
| Splice1_E0x7 (the same splice, **white** noise, no AR) | −0.81 %, [−0.0096, +0.0068] | −.008 −.017 +.004 +.004 +.004 +.005 | 23/30, 1.02 (E[0,6] 0.71–0.84) | **3** | 0 |
| Splice1_E0x8_ar2 (E[0,6] row from λ = 10) | −1.98 %, [−0.0114, +0.0046] | −.006 −.029 +.003 +.004 +.004 +.004 | 30/30, 1.10 | 7 | 0 |
| Splice1_Z0E0x7_ar2 (Z[0,6] **and** E[0,6] rows from λ = 1) | +2.02 %, [−0.0033, +0.0105]; **+4.5 % vs Splice1_E0x7_ar2, resolved** [+0.0024, +0.0126] | +.001 −.028 +.015 +.013 +.010 +.011 | 24/30, 1.13 | 6 | 0 |

- 🔑 **Splice1_E0x7_ar2 is the first cell ahead of LinReg1 on the point estimate of every axis at
  once**: CRPS −2.3 %, all 30 cells calibrated (against 25), half the low-energy tail (7 against 14
  members), no gate firings, no divergences. Against LinReg11_ar2 it is resolved: −4.5 %, CI
  [−0.0166, −0.0002].
- The per-band column is the design working: E[0,6] keeps the ridge + AR gain (−0.025), the four
  middle and small-scale bands are within +0.003 of LinReg1, and **Z[0,6] gains too (−0.007)** — the
  E[0,6] row changes the large-scale dynamics the Z[0,6] row sees.
- **The small-scale band's CRPS is not set by its own row.** Giving LinReg11 back LinReg1's
  Z/E[16,32] rows changes nothing (−0.27 % against LinReg11_ar2, per band ≤ 0.001; the small-scale
  loss stays +0.012). That loss follows the λ = 0.1 **middle-band** rows: the 16–32 band is slaved
  to the dynamics below it. So a splice keeps LinReg1's middle rows, and only the large-scale
  E[0,6] row is a free choice. **The E[0,6]-λ scan: λ = 0.1 → −1.56 %, λ = 1 → −2.34 %, λ = 10 → −1.98 %**
  against LinReg1. The neighbours differ from λ = 1 by +0.8 % and +0.4 % (neither resolved). At
  λ = 10 the E[0,6] band itself still improves (−0.029), but its larger AR noise (E[0,6] residual
  2.3× LinReg7's variance, lag 1 0.71) leaks into the other five bands (+0.003–0.004). **λ = 1 is
  the E[0,6] optimum on this grid.**
- **Both parts of the lead cell carry.** The ridge row alone (white noise) gives −0.8 %. The AR
  gives a further −1.6 % (Splice1_E0x7 − Splice1_E0x7_ar2 = +0.0026, CI [−0.0008, +0.0053]) and
  **fixes the calibration**: without it the E[0,6] row is under-dispersed (0.71–0.84, LinReg7's
  signature, §10h), 23/30 in band; with it 30/30. The white splice has the smallest tail measured
  (3 members); the AR brings it to 7, still half of LinReg1's 14.
- **Only the E[0,6] row should come from ridge.** Taking Z[0,6] from λ = 1 as well costs a
  resolved 4.5 %: that row's residual is persistent (AR poles 0.86 / 0.42), and its noise drives
  the middle bands (+0.010–0.013) and loses Z[0,6]'s own gain. Calibration falls to 24/30. Z[0,6]
  in the lead cell gains *through* the E[0,6] row, not through its own.
- The CRPS gain is not resolved at K = 46, M = 5 (CI upper bound +0.003, +1.8 %). Two checks are
  queued: the 3 TU run on the full D6's 19 selection ICs × 10 members (a second estimate at twice
  M, with the 3 TU tail), and the plain splice `Splice1_E0x7` (how much is the row, how much is the
  AR). `Splice1_E0x8_ar2` (E[0,6] from λ = 10; AR power ×4) tests whether E[0,6]'s optimum is
  higher still.

### 13g. The lead splice at 3 TU on the full D6's 19 ICs × 10 — the lead is not confirmed

`D6sel3_Splice1_E0x7_ar2`, exactly as §13b (odd ordinals 89–125, M = 10, the full D6's seeds),
paired with `D6_LinReg1` / `D6_LinReg7` on the same members.

| | vs LinReg1 | vs LinReg7 | spread–skill, leads ≤ 1 TU | Z[16,32] < 0.5× at 3 TU (< 0.25×) | gate firings |
|---|---|---|---|---|---|
| LinReg1 | — | — | 24/30 | 13 (5) | 1842 |
| LinReg7_ar2 (§13b) | +12.2 %, [−0.0018, +0.0452] | −7.4 % | 25/30 | 9 (1) | 111 |
| **Splice1_E0x7_ar2** | **+4.37 %**, [−0.0027, +0.0162], not resolved | **−13.8 %**, resolved | **27/30** | 12 (1) | 192 |

Per band vs LinReg1: +.007 **+.016** +.007 +.009 +.004 +.004 — **E[0,6], the band that gained on the
mini-D6 (−0.025), is here the band that loses most.**

**Why the two views disagree: sampling, not a defect.**
- The 19 ICs are the odd-ordinal half of the mini-D6's 46. Scoring the mini-D6 runs (M = 5)
  on exactly those 19 ICs (`D6_T_MIN=52.5 D6_T_MAX=70.75 D6_STRIDE=2`):

  | splice − LinReg1 | 46 ICs, M = 5 | same 19 ICs, M = 5 | same 19 ICs, M = 10 (3 TU runs) |
  |---|---|---|---|
  | Splice1_E0x7_ar2 | −2.3 % | +1.3 % [−0.0100, +0.0113] | +4.4 % |
  | LinReg7_ar2 | +5.0 % | +8.9 % [−0.0043, +0.0374] | +12.2 % |

  About 3.5 points is the IC subset: those 19 ICs favour LinReg1. About 3 points is members 6–10:
  adding them moves LinReg1's score 0.1850 → 0.1789, while the two AR cells move by ≤ 0.0007
  (0.1873 → 0.1867, 0.2013 → 0.2006).
- It is not hardware. The Snellius `D6_LinReg1` and the desktop `D6mini_LinReg1` make the same draws:
  dQ agrees to 3.2e-9 over the 95 shared members that end by 74 TU (`m0c_ar_online.jl --vs-full`).
- 🔑 **Honest reading.** On CRPS, Splice1_E0x7_ar2 is **level with LinReg1 to within about ±4 %**. The
  three estimates span −2.3 … +4.4 %, and none is resolved. It is ahead on calibration (30/30 vs 25/30
  mini; 27/30 vs 24/30 at 3 TU) and on stability (1 TU tail 7 vs 14; 3 TU gate firings 192 vs 1842;
  < 0.25× members 1 vs 5). That is plan §2's "better calibration and stability at no resolvable loss
  of skill". It is **not** "beats LinReg1 on skill".
- **The decisive test: M = 10 on all 46 selection ICs** (`D6mini10_*`; members 1–5 hard-linked from
  the M = 5 runs, 6–10 new, same seed rule):

  | | CRPS | vs LinReg1 (90 % CI) | per band vs LinReg1 | spread–skill in band / median | tail at 1 TU (of 430) | gate |
  |---|---|---|---|---|---|---|
  | LinReg1 | 0.17592 | — | — | 24/30, 1.06 | 22 | 26 |
  | **Splice1_E0x7_ar2** | 0.17588 | **−0.02 %**, [−0.0072, +0.0075] | +.000 −.012 +.004 +.005 +.002 +.001 | **30/30, 1.03** | **13** | **14** |

  🔑 **Verdict on the selection block: level with LinReg1 on skill (a dead heat, CI ±4.2 %), ahead
  on calibration (30 vs 24 of 30) and stability (tail 13 vs 22, gate 14 vs 26).** That is plan §2's
  "better calibration and stability at no loss of skill", not "beats LinReg1 on skill". The
  mini-D6's −2.3 % at M = 5 was member noise. **This is the finalist for the confirmation block.**

### 13h. M0ᵛ: the LSTM noise scale is a resolved gain over its matched linear model — on a weak mean

Mini-D6 (46 ICs, M = 5), `D6_CLOSURE=lstm`, gate parity with LinReg. `r3_lin_sd_h2` = frozen
least-squares skip at h = 2 + an LSTM-driven state-dependent noise scale; `r2_lin_const_h2` = the
same skip + constant correlated η (the matched M0). Both are fitted on 1–10 TU (§7d).

| | vs r2_lin_const_h2 | vs LinReg1 | per band vs matched M0 | spread–skill / median | tail | gate |
|---|---|---|---|---|---|---|
| r2_lin_const_h2 (M0, h = 2) | — | +30.1 %, resolved | — | 16/30, 0.81 | 0 | 0 |
| **r3_lin_sd_h2 (M0ᵛ)** | **−8.9 %**, [−0.0292, −0.0129], resolved | +18.5 %, resolved | −.037 −.024 −.015 −.018 −.013 −.012 | 17/30, 0.82 | 0 | 0 |

- 🔑 **First resolved gain from a nonlinear component in this project**: the state-dependent noise
  scale (lever L2) improves on its matched linear-plus-constant-noise model by 8.9 % in every
  band, with no stability cost (no tail, no gate firings).
- **But on the wrong mean**: the h = 2, 1–10 TU skip is 30 % behind LinReg1 (h = 5), and both h = 2
  cells are under-dispersed (median 0.81–0.82). The gain does not transfer by itself.
- **Next cell: M0ᵛ on the lead mean** — the LSTM scale on LinReg1's h = 5 mean (or the splice's),
  fitted on the same 1–10 TU, deployed through `D6_CLOSURE=lstm`. It needs a skip seeded from a
  `LinReg.jld2` rather than refitted, or an h = 5 refit that reproduces LinReg1 (λ = 0 least squares,
  same rows: the same map up to round-off).

### 13i. Paper 3's per-QoI penalty (Rik, 2026-09-29)

Paper 3 rescales the ridge penalty per QoI: column i of C uses **λ_i = λ (σ_i/σ_1)²**, with σ_i the
sd of the scaled correction, and quotes the base λ. It is now `lambda_scaling = :paper3` in
`5_train_LinReg.jl` (default `:none`). There σ_i is taken on the fit's own rows as the target row
minus the predictor row `q*`, and each column gets one exact ridge solve with its λ_i.
`4_setup_search.jl` builds cells by splatting the cell NamedTuple, which is identical field for
field for the old two-field cells; cells 1–13 were re-checked equal.

On R1's 1–10 TU rows: σ = 0.0173 0.0248 0.0084 0.0104 0.0518 0.0369, so the multipliers are
**1 / 2.06 / 0.23 / 0.36 / 9.0 / 4.6**. The middle bands get the least ridge, E[0,6] about twice
Z[0,6]'s, and the small scales the most. That is the direction §13f found by hand. The Z[0,6] row
of each paper-3 cell equals the single-λ fit at the same base λ to 3e-15 (LinReg14/15/16 against
LinReg13/11/12), which checks the per-column solve.

| cell | base λ | λ_i (Z0 E0 Z7 E7 Z16 E16) | AR power ×, as built |
|---|---|---|---|
| LinReg14 | 0.03 | 0.030 0.062 0.007 0.011 0.270 0.137 | 1.15–1.23 |
| LinReg15 | 0.1 | 0.100 0.206 0.023 0.036 0.899 0.456 | 1.38–1.61 |
| LinReg16 | 0.3 | 0.300 0.618 0.070 0.108 2.70 1.37 | 1.80–2.27 |

⚠️ Paper 3's rule gives Z[0,6] the base λ. §13f found Z[0,6] best at λ = 0, so the rule may cost
there at the larger base λ. Each `_ar2` cell is run at **M = 10 on all 46 selection ICs**, against
`D6mini10_LinReg1` and the finalist `D6mini10_Splice1_E0x7_ar2`.

| cell (M = 10, 46 ICs) | vs LinReg1 | vs finalist | per band vs LinReg1 (Z0 E0 Z7 E7 Z16 E16) | spread–skill / median | tail (of 430) | gate |
|---|---|---|---|---|---|---|
| LinReg1 | — | +0.02 % | — | 24/30, 1.06 | 22 | 26 |
| Splice1_E0x7_ar2 (finalist) | −0.02 %, [−0.0072, +0.0075] | — | +.000 −.012 +.004 +.005 +.002 +.001 | **30/30**, 1.03 | 13 | 14 |
| LinReg14_ar2 (paper 3, base 0.03) | +0.79 %, [−0.0054, +0.0100] | +0.81 %, [−0.0077, +0.0113] | −.000 **−.010** +.003 +.001 +.008 +.007 | 22/30, 1.12 | 15 | 23 |
| LinReg15_ar2 (paper 3, base 0.1) | +2.39 %, [−0.0035, +0.0150] | +2.41 %, [−0.0054, +0.0155] | +.004 **−.011** +.007 +.005 +.010 +.010 | 25/30, 1.08 | 12 | 9 |
| LinReg16_ar2 (paper 3, base 0.3) | +4.02 %, [−0.0004, +0.0180] | +4.04 %, [−0.0011, +0.0170] | +.011 **−.014** +.012 +.010 +.012 +.012 | 28/30, 1.09 | 11 | 3 |

- At base 0.1 the rule gets E[0,6] right (−0.011, as the finalist's −0.012), but it puts
  λ = 0.1 on Z[0,6] and 0.02–0.04 on the middle rows. Those rows cost 0.004–0.007 each, and the
  small-scale bands follow them (+0.010; §13f). Not resolved against either reference.
- At base 0.03 the rule is level with LinReg1 on skill (+0.8 %; Z[0,6] and the middle bands level,
  E[0,6] −0.010), but calibration and stability fall back toward LinReg1's (22/30, tail 15, gate 23).
  The small-scale bands still lose (+0.007) at λ_i 0.27 / 0.14 on their own rows. **Neither base λ
  reproduces the finalist's combination** of LinReg1's skill *and* 30/30 calibration. That needs
  E[0,6] alone at λ ≈ 1 with its AR, and λ = 0 everywhere else, i.e. a multiplier profile paper 3's
  σ-rule does not produce (it ranks Z[16,32] above E[0,6]).
- **The base-λ scan is a clean one-parameter trade-off**: base 0.03 / 0.1 / 0.3 gives CRPS +0.8 /
  +2.4 / +4.0 % against calibration 22 / 25 / 28 of 30, tail 15 / 12 / 11 and gate 23 / 9 / 3. Every
  point is dominated by the finalist (−0.02 %, 30/30, 13, 14). **Verdict on paper 3's rule: it is
  the right idea (per-QoI λ) with the wrong profile for this testbed.** The profile that works is
  "ridge on E[0,6] only".

### 13j. M0ᵛ on the finalist's mean — worse, stopped early (2026-09-29)

**Setup.** `m4_diag_fit.jl` gained `RIKFLOW_D_SKIP_FROM=<TO_LRS model>`. It seeds the frozen skip
from a deployed LinReg's mean, translated into the fit's scaled-dQ coordinates by least squares on
that model's own predictions over the training rows. This is exact: residual 4e-13 for
`Splice1_E0x7`, 6e-13 for LinReg1. Fits (h = 5, 1–10 TU, held out 52–74 TU, `:lstm` +
`EMISSION=state_dependent`, `SEEDHEAD=1`), in `TO_LSTM/diag/`:

| fit | mean | trained | held-out NLL / step (linear + constant η) |
|---|---|---|---|
| `m0v_fin_sd` | Splice1_E0x7 | LSTM + head | −7.546 (−6.934) |
| `m0v_fin_const` | Splice1_E0x7 | head only; LSTM frozen at its random init | −7.556 (−6.934) |
| `m0v_lr1_sd` | LinReg1 | LSTM + head | −8.233 (−7.580) |

All three early-stop at update 24–30 (the known fast overfit). The state-dependent head gains
≈ +0.6 nats/step over a constant η. **Training the LSTM adds nothing over its random
initialisation** (sd vs const). The "const" fit is therefore not a constant-noise control: a frozen
random LSTM still feeds the head.

**Online, partial** (`m0v_fin_sd`, M = 10, the first 12 of 46 selection ICs, t 52.25–57.5 TU; the
run was stopped at Rik's decision after this read):

| vs | A − B | 90 % CI | per band (Z0 E0 Z7 E7 Z16 E16) |
|---|---|---|---|
| finalist Splice1_E0x7_ar2 | **+12.7 %** | [+0.0048, +0.0353] | +.025 **+.044** +.011 +.015 +.009 +.003 |
| LinReg1 | **+12.2 %** | [+0.0048, +0.0386] | +.024 +.022 +.016 +.021 +.014 +.007 |

Spread–skill 19/30 in band, median 0.88: **under-dispersed**, worst in E[0,6] (0.57–0.98) and in
the small-scale bands at short leads (0.56–0.67).

**Reading.** The one-step NLL gain did not carry online, again (plan §25: the best one-step
likelihood has repeatedly been the wrong online selector). Two causes are visible:
- a head fitted by early-stopped one-step likelihood under-states the multi-step spread;
- the white head drops the AR(2) colour the E[0,6] ridge row needs (§13f: the white splice's E[0,6]
  was also under-dispersed, 0.71–0.84).

`m0v_lr1_sd` was not run online (the same head, the same training). **If M0ᵛ is pursued**, the
head must change, not the number of runs:
- (a) keep the AR(2) and let the LSTM scale only its innovation; or
- (b) train the head on a multi-step / ensemble (CRPS) objective.

§13h's −8.9 % was against an h = 2 constant-η model whose own spread was too small (0.81). The
state-dependent scale helps a badly calibrated base; it does not help a calibrated one.

### 13d. M3ᶠ with memory — a null (the open item of §11e)

An `:lstm` recurrence over a W = 10 window of the same h = 2 regressor, on the frozen λ = 0 skip, with
the constant seeded head; 1–50 TU, held out 52–74 TU (`m4_window_fit.jl`, `p4/m3fmem_lstm_l0_*`):

| fit | updates | best iterate | held loss | CRPS | ‖g‖/‖resid‖ |
|---|---|---|---|---|---|
| LR 1e-2, WD 1e-2, seed 1 | 17 300 | **update 0** | 0.2098 (= M0) | 0.10117 (+0.00 %) | 0.000 |
| LR 3e-3, WD 0, seed 1 | 10 700 | **update 0** | 0.2098 (= M0) | 0.10117 (+0.00 %) | 0.000 |

The held-out loss never goes below the linear floor at any checkpoint (best held-out iterate 0.210).
Together with §11a's feed-forward null and §7c (the old +3.4 % was h = 1 against an h = 1 floor, and
one extra lag gives the same), **L1 is closed at this data volume: neither nonlinearity nor recurrent
memory improves on least squares at h = 2.** The only mean gain on record is linear, from stacking
10 lags (0.2074, −1.1 %).

## 14. R2: one linear model, two code paths — the premise was wrong, the control is built (2026-10-06, laptop)

Implementation testing (`paper/todo.md` R2), not a result (restart, 2026-10-06). **Question:** the
"same" h = 5, λ = 0 linear model ran −6 to −9 % low in the LinReg path and +1.0 to +2.3 sd high in
the network path (§7d point 5). Which is the implementation's fault?

**R2-0, one metric** (`analysis/r2_bias.jl`): mean offset of the level in HF-reference sd (vs its
100 TU mean, as `m4_online_moments.jl`). LinReg1's five R2 long runs: **−0.18 to −0.54 sd over
0–20 TU** (the §7d window), −0.16 to −0.53 over 0–100 TU, every replica and QoI. Against §7d's
+1.0 to +2.3 sd the gap is real: ~1.5–2.5 sd, opposite sign. (Tracked record vs HF reference:
< 0.7 % of an sd apart, so the reference choice is immaterial.)

**R2-1:** the deployed `TO_LRS/LinReg1` IS the exact Float64 λ = 0 fit — refit coefficients 4.9e-11
relative (below cond·eps), training SSE equal to 4e-15 (`analysis/r2_offline.jl`; as #63 already
found on 2026-09-16; todo.md's "round-off-regularised" premise misread #63, which is about the
pre-R2 archive).

🔴 **The two paths never ran the same model.** `m4_linear_eta.jl` (the §7d `rdg_h5_l0`) refits on
the first 80 % of the rows, **1–8.2 TU**, not 1–10 TU. Offline, one step, teacher-forced, in
sd(dQ) (`r2_offline.jl`):

| change from the deployed LinReg1 | mean shift | rms |
|---|---|---|
| refit on 1–8.2 TU (Float64) | 2e-5 to 6e-3 | 5e-3 to 5e-2 |
| Float32 evaluation, network-path style (x, coefficients, product) | ~1e-6 | 2e-5 to 2e-4 |

The two windows' coefficients differ by **83 % (Frobenius)** while their one-step predictions agree
to 0.5–5 %: cond(X) = 1.9e6, and the λ = 0 map's near-null directions are set by which rows are in
the fit. A closed-loop bias that depends on those directions can differ between the two fits; the
one-step numbers cannot say whether it does.

⚠️ **The LinReg path is not pure Float64 either.** In `get_next_item_timeseries(::LinReg, …)`,
`pred = draw_eta(…) .|> Float32`, so `pred[…] += c * data` stores the scaled level in Float32 before
`scale_output` (~1e-5 sd(dQ) per step). `closures.tex`'s "differs … in its floating-point
precision" needs that qualification.

**The control (`tools/m4_control.jl`, `TO_LSTM/diag/control_LinReg1/StochLSTM_seed1.jld2`).**
LinReg1's map translated in closed form into the network path's scaled-dQ regression (no refit;
residual **3.1e-12 sd(dQ)** over all 39 995 rows), `arch = :lstm` with `V1 = 0` (LSTM off), a
`:constant` head equal to LinReg1's own MVG Σ (checked to 1e-16 against the MLE covariance of its
training residuals; MVG mean 1.5e-15), weights Float32, cell `StochLSTM2` (index 2). **GATE 2 on the
file as deployed:** Σ to 1.3e-6; the real `StochLSTM` closure (`stochastic = false`), replayed 100
recorded steps from 400 starts per window, against LinReg1's mean on the same history: rms
**2e-5 to 1.4e-4 sd(dQ)**, max 7e-4, mean ≤ 1e-5, on 1–10 and 10–100 TU — Float32 round-off. PASS.

**Card (declared before the runs, Rik 2026-10-06):** one arm settles R2 — the Float32 control,
5 × 100 TU on Snellius, against LinReg1's five R2 long runs (same IC, same forcing). **Pass:** in
0–20 and 0–100 TU and in every QoI, the 5-replica mean offset has LinReg1's sign and the replica
ranges overlap (`r2_bias.jl` evaluates it). Pass → the implementation is cleared and §7d's
discrepancy belongs to the 1–8.2 TU refit (by elimination; pilot, not quoted). Fail → add the
Float64 control (needs a precision switch in `12_online_StochLSTM.jl`). ⚠️ The rule is coarse:
LinReg7 (−0.25 sd mean) passes it against LinReg1 (−0.37 sd). It catches R2's sign flip, not a
0.1 sd shift; R2-5 (the control's full hindcast) is the fine check.

**Runs (Snellius, checkout `conditional-density_time-series` @ `1b3fb354`).** Smoke 27655353 (1 TU,
replica 1): completed; warm-up dQ bit-identical to the record, finite, gate 0/300, and the level over
the warm-up bit-identical to LinReg1's replica 1 (same IC, forcing, solver). Production: 5 × 100 TU,
jobs 27655506–09, 27655511 (`run_online.sh lstm 2 <r>`, `RIKFLOW_M4_MODEL_DIR=…/control_LinReg1`,
IC from the tracked record, as LinReg1's R2 runs). ~0.01 s/step.

**Result (2026-10-06): ✅ PASS — R2 is resolved; the second implementation is cleared.** `r2_bias.jl`, 5 replicas each, mean offset in reference sd:

| | 0–20 TU | 0–100 TU |
|---|---|---|
| LinReg1, LinReg path | −0.34 to −0.41 | −0.31 to −0.41 |
| control, network path (Float32) | −0.26 to −0.34 | −0.29 to −0.39 |

Same sign and overlapping replica ranges in every QoI and both windows; over 100 TU the means agree to ~0.02 sd (ratio of means −4 to −12 % per replica, as LinReg1's). All 10 runs complete (40 000 steps) and finite; gate fired on 0.80–1.45 % of steps (control) vs 0.40–1.84 % (LinReg1). So §7d's +1 to +2.3 sd belonged to the 1–8.2 TU refit, not to the code path (by elimination; pilot, not quoted). Network closures deploy at Float32 as they are. Still open: R2-5, the control's full hindcast (= C0), when the network path is run. Cost: 5 × 7:40 on H100 = ~125 SBU.

**Reproduce**
```bash
# from lib/RikFlow
julia --startup-file=no --project=. analysis/r2_offline.jl                    # R2-1 + the offline table
julia --startup-file=no --project=training exp_square_HIT/tools/m4_control.jl  # build + GATE 2 (~1 min)
julia --project=analysis analysis/r2_bias.jl TO_LRS/LinReg1 TO_LSTM/diag/control_LinReg1   # R2-0 / the pass rule
```
