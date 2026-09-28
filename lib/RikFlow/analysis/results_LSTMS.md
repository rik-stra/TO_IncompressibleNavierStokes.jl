# Results — M4, the stochastic LSTM, on HIT

Companion to [`results.md`](results.md) (M0 and the DDN). **Where a number here disagrees with a
design document, this file is the measurement.** M4 is `plan.md` §3's off-grid exploratory cell —
the stochastic LSTM of Barthel Sørensen et al. — and enters no attribution difference.

🔑 **This file was rewritten on 2026-09-24 to keep only what still stands.** The round-by-round
record — superseded protocols, the GPU investigation, scans run under flawed batching — is in git
history (last full version: commit `973e2d9a`). Everything below is either a definition, the
protocol as it is now, or a result that the current conclusions rest on.

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

🔴 **A screen, not results**: one seed, fits from scratch at the §3 protocol, one 10 TU replica each, run on this workstation's CPU (a different backend from
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
| ⏳ `dQ`, `:vrnn`, beta 1e-4, replica 2 | | | | | | | | |
| ⏳ `dQ`, `:storn`, L 200 | | | | | | | | |
| ⏳ `logr`, `:storn`, beta 1e-4 | | | | | | | | |
| ⏳ `logr`, `:storn`, state-dependent emission | | | | | | | | |
| ⏳ `logr`, `:vrnn`, beta 1e-4 | | | | | | | | |

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

### 7.1 Regularisation against the correction target's overfitting (2026-09-24, branch `m4-regularization`)

Offline only. `dQ` target, `:vrnn`, `L = 500`, seed 1, from scratch; each fit scored by the
**held-out R² of the correction** (`m4_heldout_skill`: teacher-forced, latent at its mean, per QoI,
on a window disjoint from training and from the stopping set). New: `train_stochlstm(...;
weight_decay)` (decoupled, V60); `m4_explore_fit.jl` overrides `NHIDDEN`, `WD`, `TRAINRANGE`.
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
  Regularisation by attrition; wd 1e-2 shows the same, stronger.
- None reaches the level target's lower-band skill (0.34–0.45): that part of the correction is
  learnable, the correction target does not learn it from 8 TU of data.
- **Next candidate**: `n_hidden` 6 (+ wd 1e-3) on 1–30 TU, then the online screen (§7's diagnostics).

## 8. Where it stands, and what is open

- **Lead candidate: `dQ` target, `:vrnn`, `beta = 1e-4`** — the only M4 variant that passes the flat,
  persistence and lower-tail screens; it fails on dispersion (sd ratio 1.51, 17% above 2900).
- **Before any M4 number is quoted**: a second replica (⏳), then a proper fit (5 seeds, S6) and a
  100 TU × 5-replica cluster run against LinReg1 and the DDN.
- **Open**: whether `:logr` fixes the dispersion (⏳); whether rollout training helps the `dQ`/`logr`
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
```

⚠️ On a 16 GB workstation keep it to three Julia processes and no analysis alongside; Claude Code's
low-memory reaper stopped background jobs twice at four.
