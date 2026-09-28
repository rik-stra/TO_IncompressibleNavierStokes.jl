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
4. What remains is **the upward swing — under-damped excursions** (the level target erred the other
   way). The multiplicative target `:logr` is the attack: its correction scales with `q*`, and its
   level cannot go negative.

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
