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
least-squares map and `V1 = 0`, so update 0 IS the linear model. Joint training at `lr = 1e-2`
wrecks the seed in two updates (0.284 → 1.14 held out) — the map is a cancellation that a 1e-2
Adam step on every coefficient destroys — so `Ws` is **frozen** (`freeze = (:Ws,)`) and the
recurrence models the residual.

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
