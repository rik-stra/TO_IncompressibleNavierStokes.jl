# Results — M4, the stochastic LSTM, on HIT

Companion to [`results.md`](results.md), which reports M0 and the DDN. Same convention: **where a
number here disagrees with a design document, this file is the measurement and the document is the
prediction.**

M4 is `plan.md` §3's off-grid exploratory cell — the stochastic LSTM of Barthel Sørensen et al.
(2026). It enters no attribution difference, is never the confirmatory cell, and is first on the
cut list. Design and build notes live in `meta_files/handoff_m4_stochastic_lstm.md`; this file is
only the measurements.

**Status, 2026-09-18.** Architecture, training code and both deployment drivers are built and
tested (1827 pass / 1 broken in `test/`, **412/412** in `training/`). The experiment is settled and
encoded in the configuration table. §6 is the first measurement made under it: **the loss curves,
which is all this round is for** — the held-out architecture comparison waits until the training
length and the learning rate are read off them.

Settled by this round: **`epochs` 300 → 3000**, because 300 was ~6× short of the plateau (§6), and
**`lr` stays at `1e-2`**, because the usable range above it is ~1.2× in steps and everything two
decades up diverges (§6.1). 🔴 **Still open and now a cluster job: §6.2**, whether overlapping
training segments help — the mechanism is implemented and tested, and unmeasured.

---

## 1. The architecture and its size

Four variants, one flag apart, because the source's claim is a *ranking* across them rather than a
statement about one model. With `x_t` the regressor row and `y_t` the prediction:

```
z_t  ~  q(z_t | x_t) = N( mu_z , sigma_z )     mu_z = B_mu e_t ,  sigma_z = softplus(B_sig e_t)
e_t  =  tanh( W_e x_t + b_e )                  dense encoder layer; n_encoder = 0 drops it
h_t  =  LSTM( [x_t ; z_t] , h_{t-1} )
y_t  =  V1 h_t  +  V2 z_t  +  c                linear output
```

| `arch` | `z` drawn | `z` → cell | `z` → decoder | is |
|---|---|---|---|---|
| `:lstm` | no | no | no | deterministic LSTM |
| `:vaernn` | yes | no | yes | stochasticity at the **output** only |
| `:storn` | yes | yes | no | **STORN** |
| `:vrnn` | yes | yes | yes | **VRNN** |

🔑 **"Upstream stochasticity" — the property Sørensen et al. report as the one that matters — is
exactly the `z → cell` column.** STORN and VRNN have it; the deterministic and output-only variants
do not.

🔴 **The first line is the ENCODER, the approximate posterior `q(z|x_t)` — not the prior.** The
prior is a fixed `N(0, I)`. Settled against the reference implementation
(`ben-barthel/learning_dynamics`, `ML_Code/networks_qg.py`), which is unambiguous: the encoder sees
**only `x_t`** — not the target, not `h_{t-1}` — and there is no separate recognition RNN. Reading
`methods_overview.tex`'s equation as the prior is the one way to get this structurally wrong.
Because `x_t` is observed at deployment, the same encoder serves training and inference, which is
what makes the architecture deployable at all.

### Every dimension, ours against the source's

| | source | here | why |
|---|---|---|---|
| what is predicted | a QG **field**, 2 channels on an `Ny x Nx` grid | **6 scalars**, the TO QoIs | different problem |
| input `x_t` | the field | **19 features** — `q*^n`, `q^{n-1}`, `q*^{n-1}`, bias, at `h = 1`, `N_Q = 6` | the shared `build_history` regressor |
| output | the field | **6** | `N_Q` |
| LSTM hidden | 60 | **16** | §sizing below |
| latent `z` | 60 | **4** | ⚠️ the latent dim is **not** `N_Q`; an early reading assumed it matched the output width and it does not |
| encoder | 60 (dense `tanh`) | **0** — the linear encoder `methods_overview.tex` writes | the tex and the repository disagree; `n_encoder = 0` is the tex's form |
| parameters | 43 128 at `h = 1` | **~2 950** (`:storn`) / **~2 980** (`:vrnn`) | |

**Sizing, and why it is not a taste question.** Carrying the source's 60/60/60 across gives
**43 128 parameters against 21 594 training target values** (3599 rows × 6 QoIs on the
`t ∈ [1,10]` TU window) — twice as many parameters as data. Their 60 units predict a field; we
predict six scalars. Copying the width across is copying a number, not the architecture.

| configuration | parameters |
|---|---|
| source dims, `h = 1` | 43 128 |
| source dims, `h = 5` | 57 528 |
| hidden 60, latent 6, no encoder | 21 672 |
| hidden 32, latent 8, no encoder | 8 464 |
| **hidden 16, latent 4, no encoder** | **~2 950** ← ours |

⚠️ **Consequence for the write-up:** shrinking means we are **not reproducing their model**. We are
testing their *architectural claim* — that stochasticity injected upstream beats the alternatives —
at `N_Q = 6`. That is the more meaningful test, but the paper must say which of the two it did.

🔑 **`h = 1`, and the recurrence carries the memory.** `h` is the explicit lag window in the
regressor; M0/TO-LRS uses `h = 5`. The level's 1/e time is **116–142 steps** (§3), so an `h = 5`
window is about 4% of one decay time — negligible either way. The recurrence has to do the memory
work regardless, and `h = 5` would buy nothing but 48 extra input features.

### Two deliberate deviations from the source

1. **The Gaussian emission head is implemented but OFF** (Rik, 2026-09-18). `plan.md` §3 specifies
   a *"Gaussian emission head"* for M4 and the code carries it — `Sigma^n = D^n R D^n` with
   `log d_i` linear in `h_t`, the L2 head of `methods_overview.tex`. 🔴 **`emission` now defaults
   to `:none` in `LSTMSpec`, in the configuration table and in `11_train_StochLSTM.jl`'s fallback**,
   so nothing gets the head without asking for it. The reason is the source's design: their decoder
   is deterministic and `z` is their *only* stochasticity, and a second noise channel with nothing
   in the objective allocating between them lets *"the noise is in the latent state"* stop being
   true of the fitted model even with the architecture flag set.
   - The reconstruction term becomes a plain sum of squares, which **is** the source's objective.
   - 🔴 **No predictive density, so no likelihood.** `iwae_nll` refuses for these cells rather than
     returning a number that looks like an NLL and is not one. Ensemble scoring is unaffected:
     drawing `z` still gives an ensemble, so `crps_ensemble`, the rank histogram with
     Jolliffe–Primo contrasts and spread–skill all read normally — and those are the metrics
     `plan.md` §9 makes ladder-wide precisely because they survive this kind of model.
   - ⚠️ **One exception, forced by construction: the `:lstm` control uses `emission = :constant`.**
     A deterministic backbone with no emission noise is a point predictor, cannot produce an
     ensemble, and `LSTMSpec` refuses the combination. Its spread is a constant `Sigma`, which is
     also the sharper contrast — noise upstream against noise at the output, state-independent,
     which is the DDN's design one level down. 🔴 **Its loss is therefore on a different scale**
     (Gaussian log-density, not sum of squares) and never shares an axis with the latent cells.
   - `:state_dependent` stays one keyword away for when S7 wants a likelihood back, and stays
     covered by the test suite.
   - 🔴 **Fixed while turning it off: `emission` was neither saved nor loaded.** A fit made with
     one mode came back deployed under whichever mode was the default, silently and with
     `check_shapes` passing, because the head's weights exist in every mode and only their *use*
     differs. `save_stochlstm`/`load_stochlstm` now carry it.
2. **The conservation-of-mass penalty is not imported.** It is L4, out of scope (`plan.md` §24).
   M4 takes the architecture, not the penalty.

⚠️ Two links, on purpose: **softplus** on the latent scale (the source's), **log** on the emission
scale (this project's, for the convexity and collapsing log-determinant that
`methods_overview.tex` argues for, and which explicitly rejects softplus there).

---

## 2. The loss function

Per-step negative ELBO, summed over the scored part of a segment:

```
L  =  sum_t [  -log p( q^n | mu_y(h_t, z_t), Sigma^n )  +  beta * KL( q(z_t|x_t) || N(0, I) )  ]
```

- With `emission = :none` the first term is a **plain sum of squares** — the source's objective.
  With `:constant` or `:state_dependent` it is a genuine Gaussian log-density.
- The KL is to a **fixed standard normal**, confirmed from `ML_Code/loss_funs_qg.py`.
- ⚠️ **`beta`, never `lambda`.** The source calls this weight `lambda` and uses `1e-4`; `lambda` is
  the ridge parameter throughout this project and a collision would corrupt the §8a analysis and
  the S2′ axis.

🔴 **The ELBO is an upper bound on the NLL, not the NLL.** The reconstruction term uses a single
`z` draw, so the training objective is a one-sample ELBO. For `:lstm` there is no latent path and
the objective *is* the exact NLL. **The two are not on the same scale, and a training or validation
loss must never be used to rank `:lstm` against the latent architectures.** Held-out comparison
uses ensemble CRPS and the rank histogram; the IWAE bound only where a likelihood exists at all.

**The covariance is parametrised differently in training and deployment** (and only matters when
an emission head is on). Training carries a free lower-triangular **precision** factor `A` —
`Sigma^{-1} = D^{-1} A' A D^{-1}`, `quad = ||A (r ./ d)||²`,
`log det Sigma = 2 sum(log d) − 2 sum(log diag(A))` — which is unconstrained, needs no triangular
solve and differentiates cleanly. Deployment carries the specified `Sigma = D R D` with `R` a
correlation matrix. Same family: `A` leaves `R`'s scale confounded with `d` by a per-coordinate
constant only, and the conversion normalises `R` to unit diagonal and folds the scale into `bd`.
**V41 is the test that the two give the same density**, not merely the same mean.

---

## 3. The training method

| | |
|---|---|
| optimiser | Adam, **`lr = 1e-2`**, decayed by `0.3` after `patience = 20` epochs without validation improvement, floor `1e-5` |
| batching | segments, not rows — **`batch = 32`** segments per step, ⚠️ unreachable at the current stride (see §6.2) |
| segments | length **`L = 500`**, burn-in **`burn = 100`**, stride `L - burn` |
| epochs | **3000** — raised from 300, which §6 measured to be ~6x too few |
| seeds | 5 per cell (S6: neural cells are fitted with five seeds, spread reported, **median seed deployed**) |
| precision | Float32 weights; the record and the solver are Float64 — see below |
| reparametrisation | `z = mu + sigma * eps` with `eps` drawn **outside** the differentiated function |

**Segments and burn-in.** `build_history` returns rows in increasing step order with their step
indices, so BPTT segments are contiguous slices. Two things the segmenter adds: segments **never
cross a discontinuity** in the step index, and the first `burn` rows of each segment charge the
hidden state and are **excluded from the loss**. Without the burn-in every segment contributes a
cold-start term from `h = 0`, which the model can only fit by learning to predict well from no
history — exactly the behaviour the recurrence exists to avoid.

### 🔑 `L` and `burn` come from the measured ACF, not from `T_int`

⚠️ **Do not size these with `T_int`.** `results.md` §1 is explicit about why: **the level's ACF is
not a decaying exponential.** It falls to ~0.1 within half a TU and then *rings* between roughly
−0.2 and +0.2 with a period near 1 TU, out to the end of the 10 TU window. Any integral-based
timescale is integrating that ringing, so "the" decorrelation time is not a well-defined property
of this series, and `T_exp` is meaningless on the level (`ρ₁(q) ≈ 0.9999` gives 3–34 TU).

**The robust statistics are the crossings** (level `q`, all six bands, from `results.md` §1):

| statistic | range over bands | in steps at `dt = 2.5e-3` |
|---|---|---|
| lag at ρ = 1/e | **0.290–0.355 TU** | **116–142** |
| lag at ρ = 0.1 | 0.430–0.600 TU | 172–240 |
| ringing period | ≈ 1 TU | ≈ 400 |
| ±2 Bartlett se | 0.114–0.132 | — |

The 1/e times span only **1.22×** across bands (against 2.18× for `T_int`), so a single `burn`
serves every QoI.

- **`L = 500`** covers more than one full ringing period. A recurrence is one of the few model
  classes that *can* represent an oscillating memory kernel — it is why `methods_overview.tex`
  reaches for a Hankel basis over a bank of real-pole exponentials, citing Gouasmi's
  decaying-sinusoid MZ kernel. At `L = 200` the model never sees a full period and cannot
  distinguish a ring from a decay, so it is denied the one thing the architecture is good for.
- **`burn = 100`** is D6's own warm-up length (`claude_memory.md` #67), just under the 1/e crossing
  at 116–142. The earlier `burn = 50` was under it on every band, so the hidden state was scored
  before it had seen one decay time of history.
- ⚠️ **The same argument applies to the online warm-up.** `nwarm = 100` was inherited from the
  linear cells, where the lag window is the only state; M4 also has to charge `(h, c)`.
  `tools/m4_warmup_probe.jl` is prerequisite P1 in `meta_files/handoff_m4_stochastic_lstm.md`
  §12.1, and it is not optional.
- ⚠️ **`analysis/postrun_lstm.jl` used its own fixed `burn = h + 50 = 51`** and now takes at least
  the fit's own `burn`. A scorer that charges the recurrence for less than training did is
  measuring cold starts.

### 🔴 Precision: Float32 model, Float64 solver, conversion at two named boundaries

The record is Float64 and **the solver stays Float64** — `12_online_StochLSTM.jl` sets `T = Float64`
and `params_track`'s Float32 `Re` is overridden, because a splat that carries it wins
(`claude_memory.md` #57). The model's weights and recurrent state are Float32. The conversions are
in `get_next_item_timeseries` and nowhere else:

- **in** — `q*` arrives at the solver's precision, is scaled, then cast **down** to the model's `T`;
  the lag window is stored in `T`, which is model state and not an output.
- **out** — the predicted level is cast **up** to `eltype(q*)` *before* `dQ = qhat − q*` is formed,
  so `dQ` leaves at the solver's precision whatever the weights and the `Scaling` happen to be.
- **the replayed warm-up is returned unconverted**, bit-identically, as `LinReg` and `MVG_sampler`
  do. D6's validation gate is `dQ` bit-identity over that window (`claude_memory.md` #48), and a
  conversion there would break it while passing every `≈` test in the suite (V44).

⚠️ **Promotion is not a policy.** It gave the right answer while every `Scaling` was Float64 and
would have stopped doing so the first time one fitted on a Float32 record was loaded. Training
converts the same way, once: `Xt, Yt = T.(X), T.(Y)`.

### 🔴 Three things the training loop must do, learned the hard way

1. **Return the best-validation iterate, not the last.** Without checkpointing, the fit that gets
   saved is whatever the final epoch happened to land on.
2. **Decay the learning rate on plateau.** At a constant `1e-3` the end of training oscillated by
   several nats — wider than the differences between architectures.
3. **Hold the validation noise draws fixed.** Re-drawing `eps` each epoch makes the validation loss
   a fresh one-sample estimate, so epoch-to-epoch comparison mixes "the model changed" with "the
   draw changed", and any best-validation rule selects partly on a lucky draw.

All three were absent in the first implementation and all three invalidated its results, which were
withdrawn in full. `V49` in `training/runtests_lux.jl` pins all three. 🔑 **The lesson generalises:
a fit is not a result until the curve is shown to have converged, and a training loop that returns
its last iterate will manufacture rankings out of optimiser noise.** That is why §6 exists and why
it comes before any architecture comparison.

### Scoring — what M4 breaks

🔴 A stochastic latent path has **no closed-form one-step predictive density**, so exact NLL,
closed-form CRPS and `companion`/ρ(C̃) are **undefined** for M4.

| metric | on M4 |
|---|---|
| exact one-step NLL · closed-form CRPS · ρ(C̃), starred gain, total block sum | ❌ undefined |
| `crps_ensemble` | ✅ unchanged |
| rank histogram + Jolliffe–Primo | ✅ unchanged — **the one axis the whole ladder reads on** |
| IWAE-K bound (`nll_iwae`) | ❌ undefined at `emission = :none`; ✅ but an upper bound on the NLL otherwise |

🔴 **`nll_iwae` is a lower bound on `log p`, hence an upper bound on the NLL. It is not comparable
to M0's exact NLL and must never share a column with one.** This is why `plan.md` §9 makes the
ensemble rank histogram, not PIT, the ladder-wide calibration metric: it is the only one that reads
the same on M0 and M4.

**Reading the rank histogram.** With `M` ensemble members there are `K = M + 1` bins and `M` degrees
of freedom, so **`chi2_eff ≈ M` is calibration**, not zero. ⚠️ And a value far *below* `M` is not a
pass — it is `results.md` §2's *"flat is not skilful"* trap from the other side. Always report the
paired dynamics statistic (lag-1 autocorrelation of the predicted mean `dQ`) beside it.

---

## 4. The experiment

🔑 **Sørensen et al.'s conclusion is that the noise belongs in the latent state** — the variants
that beat the others are the ones that feed `z` *into the recurrence*. So the first experiment is
those two, plus a control:

| cell | `arch` | `emission` | `z` into cell | `z` into decoder | role |
|---|---|---|---|---|---|
| 1–3 | `:storn` | `:none` | ✅ | — | upstream stochasticity, no output skip |
| 4–6 | `:vrnn` | `:none` | ✅ | ✅ | upstream stochasticity + skip |
| **7** | `:lstm` | `:constant` | — | — | **control** |
| 8–9 | `:vrnn` | `:none` | ✅ | ✅ | latent dimension 6, 8 |
| 10 | `:vaernn` | `:constant` | — | ✅ | output-only; the variant their result argues against |

Cells 1–6 are the **`beta` scan**, `{0, 1e-4, 1e-2}` × the two architectures, at **one seed**; the
winning `beta` is then re-run at all 5. Spending five seeds per point before knowing which `beta`
is sensible is an hour bought for nothing.

⚠️ **Keep the control even though it is not one of the two.** Without it there is no statement to
make: *"the latent path helped"* is only meaningful against a model that has none, and it costs
minutes. `:vaernn` is deliberately last — it is the variant their result argues against.

| | value | note |
|---|---|---|
| train window | `t ∈ [1, 10]` TU, `train_range = (400, 4000)` | the window the M0 cells use |
| selection window | `t ∈ [10, 19]` TU | disjoint from training **and** from the online reference |
| record | R1's tracking record, the `(q*, q)` pairing M0 is fitted to | the literal Sørensen setup — a free-running LF trajectory mapped to HF — is a different dataset and a different question; **our application is the closure** |
| inner validation | trailing `val_frac = 0.2` of segments | ⚠️ early stopping and diagnostics **only**; selection is the protocol's job, not this split's |

---

## 5. Cost

**The entire offline programme runs on this laptop in minutes. The online programme cannot run here
at all.** The model is ~10³ parameters on 6 QoIs; training is CPU-bound Julia and
`11_train_StochLSTM.jl` never touches CUDA.

### Offline, measured on this laptop

Over R1's tracked QoIs on `train_range = (400, 4000)` — 3599 rows, 19 features. Three of the four
steps below are lessons rather than tuning:

| state | s/epoch | note |
|---|---|---|
| as first written, 60/60/60, `L = 200` | 3.42 | one `L`-step loop per segment |
| + segments batched through one recurrence | 2.05 | `B` GEMVs become one GEMM (V50) |
| + right-sized net, 16/4/no encoder | 0.58 | 43 128 → ~2 950 parameters (§1) |
| **+ O(L) reverse pass, at `L = 400`** | **0.178** | the per-step slice was O(L²) |

🔴 **The last step was the big one and it was a bug, not a tuning knob.** `GX[:, t, :]` inside the
recurrence looks free, but Zygote's pullback for a slice allocates a *parent-sized* zero array and
scatters into it — so an `L`-step loop did `L` allocations of `4H × L × B`, making the reverse pass
quadratic in `L` while the forward pass was linear. One gradient step allocated 172 MiB for a
forward pass allocating 3.8 MiB. Slicing once behind an adjoint that accumulates into a single
buffer restored O(L): 246 ms → 51 ms, and reverse/forward from 26× down to 5.8×.

⚠️ Sizing the network down bought 3.5×, not the 14× the parameter count suggests — the cost was
dominated by Zygote's per-timestep overhead, not arithmetic. The remaining 3.3× came from the
O(L²) fix, which is why long segments are affordable at all.

⚠️ If several fits are run in parallel, set `OPENBLAS_NUM_THREADS=1`: Julia processes each spawning
a full BLAS pool contend badly enough to be slower than running sequentially.

🔑 **The offline programme is an experimentation loop, not a batch job.** There is no reason to run
it on a cluster and no reason to economise on seeds or epochs — §3's retracted first run is what
economising cost.

### 🔑 Why not a GPU — the parallelism is across fits, not inside one

Rik asked, 2026-09-18, and it is the right question to ask of anything that takes 45 minutes.
**Training M4 is not compute-bound and a GPU does not address what it is bound by.** Three
measurements say so, none of them about the device:

- **The arithmetic is negligible.** The largest matrix in the model is `Wx`, `64 x 23`. One
  gradient step is 500 timesteps forward and backward over ~2 950 parameters at `B = 7` —
  about **75 MFLOP**. Measured at ~0.2 s per gradient step, that is **~0.34 GFLOP/s**, one to two
  percent of a single CPU core. The whole five-point stride scan is **~1.1 TFLOP**, which an H100
  would finish in ~20 ms of arithmetic. The 45 minutes is not arithmetic.
- **The time is overhead, and §5 already located it**: ~400 µs per timestep for a matrix product
  that takes nanoseconds, i.e. Zygote's per-timestep tracing. Moving the FLOPs to a device does
  not touch that.
- **The long axis cannot be parallelised.** `h_t` depends on `h_{t-1}`, so the 500 timesteps are
  strictly sequential: ~1000+ *dependent* kernel launches per gradient step. At 5–10 µs of launch
  latency each that is 5–10 ms of pure latency per step, on work the device would do in
  nanoseconds. This is a throughput machine's worst case, and it is the same reason
  `ts_lstm_online.jl` keeps the **deployed** closure on the CPU: *"at H = 60 / N_Q = 6 the cell is
  far too small to amortise a kernel launch."*

🔑 **Where the speedup actually is, and it is large.** (i) **Across fits**: 6 `lr` points, 5
`stride` points, 5 seeds and 10 cells are all independent, so a SLURM array over CPU cores is
near-linear and needs no port. That is what makes the scans minutes on Snellius. (ii) **Batch is
the only parallel axis inside a fit and it is currently 7.** V50 already made segments share one
recurrence so `B` GEMVs become one GEMM — but there is nothing to amortise at `B = 7`. 🔴 **This is
a second, independent reason to care about §6.2**: at `stride = 50` there are 56 training segments,
so `B` could be 56 — eight times the work per launch at nearly the same overhead. That helps the
CPU directly, and it is the only thing that would make a GPU worth re-examining.

### ✅ The GPU path RUNS, and it is 5.7x SLOWER than the same node's CPU — measured

🔒 **First GPU run, Snellius `gpu_h100`, 2026-09-18: `M4 SMOKE PASS on device=cuda`.** Every stage
passed. Stage 9 then timed the whole scan geometry, and a second smoke at `M4_DEVICE=cpu` **on the
same node** gives the comparison that means something:

| stride | batch | node CPU s/epoch | GPU s/epoch | GPU / CPU |
|---|---|---|---|---|
| 400 | 32 | 0.139 | 0.851 | 6.1x slower |
| 400 | **2** | 0.124 | 1.481 | **11.9x slower** |
| 100 | 32 | 0.089 | 0.884 | 9.9x slower |
| 50 | 32 | 0.292 | 1.251 | 4.3x slower |
| 20 | 32 | 0.574 | 1.879 | 3.3x slower |
| **whole scan** | | **18 min** | **102 min** | **5.7x slower** |

🔴 **This supersedes an earlier table here that read "1.38x slower, and the GPU wins at stride 20".
That was wrong.** Its CPU column came from a workstation, not from the GPU node, and the node's CPU
is **4.1x faster** than that workstation — enough to invert the conclusion. The caveat was stated
at the time; it turned out to carry the entire result. 🔑 **A cross-machine ratio is not a ratio.**

🔑 **What survives, and it is the part worth keeping:** the GPU's disadvantage shrinks monotonically
as the stride shortens — 6.1x, 4.3x, 3.3x — which is §5's amortisation argument visible in the
data. Wider batches give each launch more work. **It simply never crosses 1.** And the `batch = 2`
control is the sharpest point: **11.9x** worse on the device, its worst by far, while being among
the cheapest on the CPU. That is launch-boundedness measured rather than argued.

🔒 **Conclusion: train M4 on the CPU. `M4_DEVICE` defaults to `cpu` in both batch scripts**
(Rik, 2026-09-18). The GPU works and stays fully supported — `M4_DEVICE=cuda sbatch -t 03:00:00 …`
— it simply costs 5.7x the wall time.

⚠️ **The partition stays `gpu_h100`; only the device moved.** That is the better-evidenced choice,
not an oversight: the 18 min was measured on *that node's* CPU, so running there with
`M4_DEVICE=cpu` is the configuration that was actually timed. A CPU partition (`rome`, `genoa`,
both already covered by the shared depot's `JULIA_CPU_TARGET`) would free the GPU but is
unmeasured — smoke it there before trusting a walltime. The cost accepted meanwhile is a GPU
requested and left idle.

🔒 **Walltimes follow**: `run_m4_sweeps.sh` **`-t 01:00:00`** (1.5 x 18 min + startup), raise to
3 h for `M4_DEVICE=cuda`; `run_train_lstm.sh` **`-t 02:00:00`**, which covers even the no-seed path
that fits all five seeds in sequence (~35 min on this CPU, ~3.6 h on the GPU).

⚠️ **The per-point CPU numbers above are noisy and the ordering among the fast ones is not real** —
7 segments timed at 0.139 s against 25 segments at 0.089 s is more work in less time. At ~0.1 s an
epoch, two timed epochs are dominated by the clock and by first-touch effects. The **totals** are
sound; the fast rows individually are not. `m4_smoke.jl` now grows the sample until each
measurement spans at least `RIKFLOW_M4_TIMING_MIN_SECS` (default 2 s) and prints how many epochs
each number rests on.

### How the GPU path is built

Built 2026-09-18 on Rik's instruction, *after* the argument above rather than against it: the point
is to let the question be **measured** instead of argued.

**How it works, and why there is no CUDA in the model code.** `lstm_forward` allocates its
recurrent state and hidden-state buffer with `similar(X, ...)` rather than `zeros(T, ...)`, so
every array it makes follows the array type it was *given*. Device choice then lives entirely at
the call site — `train_stochlstm(...; device = CuArray)` — and not one line of the recurrence,
the encoder, the decoder or the loss mentions a device.

⚠️ **`zero(similar(...))`, never `fill!(similar(...), 0)`.** The second is a `setindex!` Zygote can
see and is refused outright with *"Mutating arrays is not supported"*. Measured, not assumed.

🔑 **Every random number is drawn on the HOST and only then moved.** Parameters come from
`init_lstm_params(Xoshiro(seed))` and `eps` from the same stream; both are then placed. So **a
device fit and a host fit at the same seed are the same fit**, differing only by reduction order.
A CURAND stream would be faster and would void V49's reproducibility property and every number
already in this file. The copies are trivial — ~3 000 parameters, and 256 KiB of noise at the
largest geometry, against a gradient step of ~200 ms. Parameters are returned **on the host**,
because `LSTMWeights`, `save_stochlstm` and JLD2 all want plain arrays and a fit readable only on a
GPU node is not a fit.

**Selecting it.** `M4_DEVICE=cpu` or `cuda`, honoured by `11_train_StochLSTM.jl`, both scans and
the smoke through one resolver, `RikFlow.m4_device`. 🔴 **The Snellius batch scripts default it to `cuda`**
(Rik, 2026-09-18) — `run_train_lstm.sh` and `run_m4_sweeps.sh` both do
`export M4_DEVICE=${M4_DEVICE:-cuda}`, so the cluster trains on the device and
`M4_DEVICE=cpu sbatch ...` is the per-submission override (SLURM's default `--export=ALL` carries
it through). The library default stays `cpu`, so nothing that calls `train_stochlstm` directly
changes behaviour. 🔴 **`cuda` throws when no device is functional
rather than falling back to the host** — a silent fallback would report a GPU run that took CPU
time and put that number in a table. Verified: on this GPU-less workstation the drivers fail at
load with *"CUDA.functional() is false … Refusing to fall back to the CPU silently"*.

✅ **V54 tests the device path without a device, and it is not a compile check.** `JLArray`
(GPUArrays' host-backed reference array) **refuses scalar indexing exactly as `CuArray` does** and
its `similar` returns its own type, so it is a real proxy for the two failure modes a port actually
has: a host array silently mixed into a device graph, and a scalar `getindex` inside the
recurrence — neither of which a plain CPU run can see. The testset opens with a **positive
control** asserting `JLArray` really does refuse scalar indexing, because a proxy that does not
enforce the property proves nothing (V31's lesson, one level up). Measured, all four
architectures including the `:constant` emission head:

| arch | device run | max rel. difference on the validation curve vs host |
|---|---|---|
| `:storn` | ✅ | 1.6e-7 |
| `:vrnn` | ✅ | 2.1e-7 |
| `:lstm` / `:constant` | ✅ | 6.6e-8 |

i.e. Float32 round-off from a different reduction order, which is what "the same fit" looks like.

🔴 **What V54 does NOT establish, and it is the expensive half.** That CUDA.jl compiles these
kernels; that `m4_device("cuda")` finds a device; that the performance is anything but worse. Those
need a GPU node, and this repository has been bitten twice by exactly the class of defect a CPU
test cannot see — defeated constant propagation (#56) and non-isbits kernel arguments (#57), both
found only on the cluster. **Treat the first GPU run as a test, not as a measurement**, and run it
at a small budget (`RIKFLOW_M4_UPDATES=5`) before anything long.

⚠️ **The expectation is still that it will be slower at the current geometry**, for the reasons
measured above. The configuration in which it could pay is §6.2's short `stride` with `batch = 32`,
which widens each launch from 7 segments to 32.

✅ **Now measured — see the table above.** What follows was written before the GPU run and is
kept because the prediction it makes is the one the measurement confirmed. 🔑 **The cheap way to settle it is a `B` sweep on the CPU**: if wall time
per gradient step is flat from `B = 7` to `B = 56`, the fit is overhead-bound, which confirms the
diagnosis and answers the GPU question at the same time. A port would additionally have to survive
the two GPU-only failure classes this repository has already been bitten by — non-isbits kernel
arguments and defeated constant propagation (#56, #57).

### Online, not feasible locally

`12_online_StochLSTM.jl` couples the closure into a 64³ LES for 40 001 steps per replica, five
replicas per cell — a GPU and R1's 2.7 GB tracking record. On Snellius that is the same cost as an
LRS online run (~50 min for five replicas). **It has never been executed**; it is the one part of
the M4 build that has not been run end to end. D6 — the multi-IC ensemble S7-online needs — is
`K = 90 × M = 10` coupled runs and is a cluster job by any measure.

### S4, measured

`tools/m4_cost_probe.jl`, CPU, Float32, deployed closure only (`src/ts_lstm.jl` is hand-written on
plain arrays; Lux is never called from the solver loop):

| arch | h | hidden | latent | encoder | median / step | % of the S4 allowance |
|---|---|---|---|---|---|---|
| `:vrnn` | 1 | 60 | 60 | 60 | 43.9 µs | 15.8% |
| `:storn` | 1 | 60 | 60 | 60 | 35.7 µs | 12.8% |
| `:lstm` | 1 | 60 | 60 | 60 | 16.2 µs | 5.8% |
| `:vrnn` | 5 | 60 | 60 | 60 | 26.1 µs | 9.4% |
| `:vrnn` | 1 | 128 | 60 | 60 | 40.7 µs | 14.7% |
| `:vrnn` | 1 | 60 | 6 | 0 | 14.8 µs | 5.3% |

The allowance is 15% of the 1.85 ms/step surrogate share on HIT, i.e. **278 µs**. 🔑 **M4 is
admissible at the source's full dimensions with roughly six times the margin it needs** — and ours
is far smaller again. S4 is a kill criterion, not a target, and it is not the binding constraint
here. ⚠️ This measures the closure only, not the FFTs, the QoI computation or the host/device round
trip, all of which are already inside the 1.85 ms. ⚠️ The table predates `emission = :none`; the
probe now follows the deployed configuration, so it will read slightly cheaper on re-run.

---

## 6. The loss curves — what this round was for

Three cells, seed 1, at the settings of §3 and §4, run to **3000 epochs** via `RIKFLOW_M4_EPOCHS`
rather than the configured 300. `analysis/plot_lstm_losses.jl` draws them from the saved fits, so
the figure cannot disagree with what the driver printed.

⚠️ **These three fits were made under the pre-2026-09-18 train/validation split** — by position in
the segment list rather than by row (§6.2). At the tiling stride the two splits differ only in
which two segments form the validation set, so the shape of the curves and the epoch at which they
flatten are unaffected; the *values* will move in the third digit. The re-run under the new split
belongs on the cluster with everything else in §6.2 and has not been made here.

![M4 training curves](figures/fig11_lstm_losses.png)

| cell | arch | emission | params | val @300 | @1000 | @1500 | @2000 | @3000 | best | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| StochLSTM2 | `:storn` | `:none` | 2 952 | 5.28e-3 | 1.51e-3 | 9.65e-4 | 8.14e-4 | 8.14e-4 | **8.13e-4** | 2949 |
| StochLSTM5 | `:vrnn` | `:none` | 2 976 | 9.05e-3 | 1.84e-3 | 1.37e-3 | 1.19e-3 | 1.19e-3 | **1.19e-3** | 2896 |
| StochLSTM7 | `:lstm` | `:constant` | 2 696 | −13.8 | −16.9 | −18.0 | −18.0 | −18.1 | **−18.07** | 3000 |

🔴 **The configured 300 epochs is roughly 6× too short, and that is the finding of this round.**
At 300 epochs every curve is in the middle of a clean power-law descent with the learning rate
still at its starting value — nothing had converged, and the retracted first run is what reading a
fit at that stage costs. The two latent cells reach their floor at **~2000 epochs** and do not move
between 2000 and 3000 (8.14e-4 and 1.19e-3 to three digits at both). **`epochs` is now 3000** —
2000 is where it flattens, and the extra 1000 is insurance that costs only time, because
best-iterate selection means over-running cannot make the fit worse. 300 would have been read as a
ranking.

⚠️ **An epoch here is ≈ one optimiser step.** At `L = 500` on 3599 rows there are 9 segments, 7 of
them training — far fewer than `batch`, so the batch is never filled. 🔴 **Under the current
(row-based) split it is 2 updates, not 1**: the 2879-row training block gives 7 segments in two
*length* groups (6 × 500 and 1 × 479), and `group_by_length` batches each group separately. The
three fits in the table above predate that split, had 7 segments of one length, and so ran at 1
update per epoch — their x axis is **updates**, and the current configuration reaches the same
update count in half the epochs. A comparison against any other project's epoch count is
meaningless; compare updates, and quote the segment geometry beside them.

🔑 **The plateau is produced by the decay, not by the data running out.** The schedule first fires
at epoch **1655** (`:storn`) / **1691** (`:vrnn`) and reaches the `1e-5` floor within ~300 epochs of
that; the curve is flat only after. Before it fires the descent is still steep. So the run is
*learning-rate limited at the end and step limited in the middle* — which is exactly the regime in
which "train longer" and "start higher" are the two things worth testing, and §6.1 tests the second.

⚠️ **The control's curve is on a different scale and is not comparable to the other two** —
Gaussian log-density against sum of squares (§2). Read it only against itself. It is also visibly
the noisiest of the three between epochs 30 and 150, with spikes of several nats, and its schedule
fires far earlier (epochs **131** and **387**) because those spikes look like plateaus to a
patience rule. That is a property of the `:constant` head's objective, not of the backbone.

🔴 **Do NOT read `:storn` 8.13e-4 against `:vrnn` 1.19e-3 as the architecture result.** This is the
*inner* validation split — the trailing 20% of segments, used for early stopping and for drawing
these curves — at **one seed**, on the loss the model was trained on. The architecture comparison is
held-out, on the disjoint selection window, by ensemble CRPS and the rank histogram at five seeds
(§4, §8). Reporting an inner-split loss as a ranking is the same mistake as the retracted run in a
different costume.

⚠️ **No overfitting is visible yet** on the latent cells: at 3000 epochs train/val is
6.48e-4 / 8.13e-4 (`:storn`) and 1.20e-3 / 1.19e-3 (`:vrnn`) — a gap of 1.25× and none at all. With
~2 950 parameters against 21 594 target values that is what should happen, and it is the sizing
argument of §1 coming out right rather than a new result.

### 6.1 Does a higher starting learning rate converge in fewer steps? — ❌ no. 🔒 CLOSED

🔒 **`lr` is fixed at `1e-2` and is no longer swept** (Rik, 2026-09-18). The scan below is why;
`tools/m4_lr_scan.jl` stays for the record and for a future architecture, but nothing in the
current programme re-runs it.

`tools/m4_lr_scan.jl`, cell 2, seed 1, 1000 epochs per point. **Every point runs inside one
process on one prepared dataset**: same segments, same initialisation, same validation split, same
fixed validation noise draws, so the only thing that differs is `lr`. The thresholds are read off
the scan itself — the best validation loss any point reached, and 100× / 10× / 2× above it.

| lr | best val | best ep | epochs to 0.148 | to 0.0148 | to 0.00295 | schedule |
|---|---|---|---|---|---|---|
| 1e-3 | 1.077e-2 | 1000 | 164 | 821 | never | never fired |
| 3e-3 | 2.950e-3 | 995 | 61 | 312 | 995 | never fired |
| **1e-2** | **1.476e-3** | 999 | 31 | 146 | 477 | never fired |
| 3e-2 | 1.599e-3 | 1000 | 21 | 91 | 393 | 0.03 → 0.009 |
| 1e-1 | 0.127 | **37** | 34 | never | never | 🔴 **NaN** |
| 3e-1 | 1.953 | **3** | never | never | never | 🔴 **NaN** |

🔴 **The usable range is narrow and `1e-2` sits at the good end of it.** Above it the return is
~1.2× in steps and it is paid for in the minimum reached (3e-2 lands at 1.60e-3 against 1e-2's
1.48e-3), and it is the only surviving rate whose decay-on-plateau fired — i.e. it stalled and had
to be rescued. Two decades up the fit diverges to NaN.

🔑 **`1e-1` is the entry in the table worth remembering.** For the first ~34 epochs it is the
*fastest* point in the scan — it reaches the loosest threshold before any other rate — and it is
destroyed by epoch 37. **Early descent is not evidence of anything.** This is §3's retracted-run
lesson in the `lr` axis: read the best validation reached, never the first fifty epochs.

⚠️ Below `1e-2` the cost is steep and asymmetric: `3e-3` needs ~3× the steps for the same loss and
`1e-3` — the value this project used until 2026-09-18 — never reaches it at all inside 1000 epochs.
So the move from `1e-3` to `1e-2` was worth about **3× in steps**; there is no second such move
available above it.

✅ **Cross-check that the scan is a measurement and not a fixture:** its `lr = 1e-2` point ends at
1.5075e-3 after 1000 epochs, and the independent 3000-epoch fit of the same cell reads 1.51e-3 at
its own epoch 1000. Two processes, same number.

### 6.2 Do overlapping (shorter-stride) training segments help?

⏳ **PENDING — the scan is written and runs on the cluster, not here** (Rik, 2026-09-18: *"we can
do that on snellius in a matter of minutes"*). `tools/m4_stride_scan.jl` was started on this
workstation and stopped; five points at 3000 updates each is ~45 minutes here and minutes there.
The mechanism below is implemented, tested (V53) and unmeasured.

**The question, because it is not the obvious one.** Until 2026-09-18 the training segments were
computed **once** before the epoch loop and reused unchanged; only their order and the latent
noise draws varied per epoch. With `stride = L - burn` the scored windows exactly **tile** the
record — every row is scored once per epoch — which at `L = 500` leaves **7 training segments**,
fewer than `batch = 8`. So an epoch was **one full-batch update**, the shuffle was inert, and
"3000 epochs" in §6 means 3000 optimiser steps.

`stride` is now a keyword on `train_stochlstm` and a configuration field (defaulting to
`L - burn`, so nothing moved without being asked to). A shorter stride overlaps the segments and
makes more of them — 4× as many at `stride = 25`.

🔴 **It is augmentation, not data.** The same 3599 rows are re-scored at different offsets inside a
segment, so the extra gradients are correlated and `k ×` the segments is not `k ×` the information.
Two things it can genuinely buy, and they are different:

- **more optimiser steps per epoch** — a pure budget effect, nothing to do with the overlap;
  lowering `batch` at the tiling stride buys exactly the same thing;
- **variation in how long the hidden state has been charged** when a given row is scored — a real
  regulariser for a recurrence, and the only part the overlap alone can supply.

🔑 **So the scan matches every point on optimiser STEPS, and carries `(stride = 400, batch = 2)` as
the control that separates the two.** Comparing at equal epochs would confound them completely.

🔑 **`batch = 32` (Rik, 2026-09-18) is the other half of the same lever.** It does nothing at the
current stride — 7 segments cannot fill a batch of 8, let alone 32 — and takes effect only once the
stride shortens. Its purpose is the amortisation §5 identifies: V50 already made a chunk of
segments share one recurrence, so the per-timestep Zygote overhead is paid once for `B` segments
instead of `B` times. At `B = 7` there is nothing to amortise; at `B = 32` there is. **So `stride`
and `batch` are one lever, not two** — the stride creates the segments and the batch is what makes
them cheap — and it is also the only route by which a GPU could ever become relevant here (§5).

🔴 **The train/validation split had to change with it, and the old one was a latent trap.** It was
`segs[1:end-nval]` / `segs[end-nval+1:end]` — by position in the *segment list*. At the tiling
stride that is safe: consecutive segments overlap by exactly `burn` rows, and those rows are the
next segment's **burn-in**, which is not scored, so the scored ranges stay disjoint. At any shorter
stride the scored windows overlap and scored validation rows land inside scored training rows —
**the validation loss quietly becomes a training loss, with nothing looking wrong.** The split is
now by **row**, each side segmented independently, which cannot do that at any stride; and the
embargo comes free, because the validation block's first `burn = 100` rows are burn-in, putting the
first scored validation row about one 1/e time past the last training row. **V53** pins the whole
geometry at strides 400 / 75 / 50 / 25 / 1, and refuses a stride past `L - burn`, which would leave
rows nothing ever scores.

⚠️ **§6's table predates this change** and was measured on the old split. It is re-run under the
new one before it stands.

**The scan, as it will run** (5 points, each to the same 3000 optimiser steps, so the epoch count
is derived per point and is *not* the comparison axis):

| stride | batch | train segs | updates/epoch | epochs | isolates |
|---|---|---|---|---|---|
| 400 (tiling) | 32 | 7 | 2 | 1500 | the baseline |
| **400** | **2** | 7 | 4 | 750 | 🔑 **control — more steps, no overlap** |
| 100 | 32 | 25 | 2 | 1500 | overlap 4× |
| 50 | 32 | 49 | 3 | 1000 | overlap 8× |
| 20 | 32 | 120 | 5 | 600 | overlap 17× |

🔴 **Strides 100 / 50 / 20** (Rik, 2026-09-18). The first two rows are additions to that list and
are said so here rather than left to look like part of it: row 1 is the current default, which is
what the other rows have to be *read against*, and row 2 is the control without which "overlap
helped" cannot be separated from "more optimiser steps". Together they cost two points of five.

⚠️ **Segment counts are on the 2879-row TRAINING block, not the whole record**, and `updates/epoch`
accounts for the two length groups — both are things I got wrong before the smoke run printed them.
⚠️ **At `batch = 32` matched updates is no longer matched compute**: one update covers up to 32
segments, so at `stride = 50` an update sees ~9× the row-steps the tiling stride can offer. That is
the intended asymmetry — a bigger update is the point of a bigger batch — but the wall times the
scan prints are then not a cost ranking and must not be read as one.

🔑 **How to read it when it comes back.** If the short strides beat the baseline by no more than
the `(400, batch = 2)` row does, the gain was budget — more optimiser steps — and the overlap
itself bought nothing, in which case lower `batch` and leave `stride` alone, because it is
cheaper per step. Only a margin *over* that control is evidence for the context-position
augmentation, and that is the part worth carrying into the paper.

⚠️ **`epochs` and updates decouple the moment `stride` moves.** `epochs = 3000` at the tiling
stride is 3000 updates; at `stride = 50` it would be 21 000. Any change to `stride` requires the
epoch count to be re-derived from the intended update budget, which is what the scan does and what
the configuration table does **not** do automatically.

---

### 6.3 The capacity x regularisation grid — cells 11-28

🔴 **The programme after §6.2** (Rik, 2026-09-18). `lr` and `epochs` are now FIXED at `1e-2` and
3000 (§6, §6.1), so the remaining axes are the model's, not the optimiser's:

| axis | values | why |
|---|---|---|
| `beta` | **0, 1e-4, 1e-2** | the KL weight, and the latent path is now the *only* noise channel, so this is the whole regularisation of the stochasticity (§9 Q2) |
| `n_latent` | **4, 6** | 6 is `N_Q`, the matched case; 4 is under-complete and forces compression (§9 Q3) |
| `n_hidden` | **6, 10, 16** | how much the recurrence can do *without* the latent path |

**3 x 2 x 3 = 18 cells, appended to the configuration table as StochLSTM11-28**, generated by a
loop rather than written out so the table cannot drift from its own description.

🔑 **A grid, not three sweeps.** `beta` regularises the latent path and `n_latent` is how much
latent there is to regularise — they are not separable — and neither is independent of `n_hidden`,
which sets how much the model can do if the latent path contributes nothing. Sweeping them one at a
time answers a question nobody asked.

🔴 **Three of the eighteen are exact re-runs of cells 1-3.** The `n_hidden = 16, n_latent = 4`
column at the three betas *is* cells 1, 2, 3 — same configuration, same seeds, same fit. The table
is append-only so they stay. `10_setup_lstm.jl` now **detects and prints** duplicates with the
array range that skips them, rather than leaving them to be found as wasted cluster time:

```
3 duplicate configuration(s) — the same fit under two names:
  StochLSTM13 == StochLSTM1
  StochLSTM19 == StochLSTM2
  StochLSTM25 == StochLSTM3
  distinct rows: 25 of 28
```

⚠️ **The grid is `:storn` only, and that was not specified.** Upstream stochasticity with **no**
decoder skip, so the latent can act only *through* the recurrence — the cleanest test of the
source's claim, and the reason to prefer it when only one architecture can be afforded. 18 cells on
both is 36. `GRID_ARCH` in `10_setup_lstm.jl` switches it, or a second block appends `:vrnn`.
⚠️ `n_hidden = 6` with `n_latent = 6` is the corner where the cell is *smaller* than the latent.
Deliberate, and also the corner most likely to be simply bad — do not read a poor result there as
evidence about the latent dimension.

⚠️ **Run this after §6.2, not beside it.** The grid inherits whatever `stride` the stride scan
settles, and `stride` changes `updates/epoch`, so fitting the grid first would fix 18 cells at a
segmentation that is about to change.

---

## 7. Running the sweeps on Snellius

🔴 **The M4 batch scripts train on the GPU by default since 2026-09-18** (Rik):
`run_m4_sweeps.sh` (the scans) and `run_train_lstm.sh` (one cell) both set
`M4_DEVICE=${M4_DEVICE:-cuda}` on `gpu_h100 --gpus=1`. Both are single sequential Julia processes;
the batches are still assembled on the host, which is why `OPENBLAS_NUM_THREADS=1` is still set.

```bash
# on the login node, from lib/RikFlow (or from exp_square_HIT -- the script finds the drivers)
julia --project exp_square_HIT/10_setup_lstm.jl          # only if the table changed; no GPU

# 🔴 FIRST SUBMISSION, ALWAYS. ~3 min, finishes, exits 0 with `M4 SMOKE PASS` or names the phase
# that failed. Needs no inputs_lstm.jld2 and writes only to a temp dir.
sbatch batch_scripts/run_m4_sweeps.sh smoke

# then the real thing
sbatch batch_scripts/run_m4_sweeps.sh stride             # the stride scan (§6.2), on the GPU

# then the capacity x regularisation grid (§6.3), one cell per job, 1 seed to explore.
# 🔴 The list SKIPS 13, 19 and 25 -- they are exact re-runs of cells 1-3 (10_setup_lstm.jl prints
# this list). ⚠️ A LOOP of sbatch, not `--array`: run_train_lstm.sh takes the cell as $1 and does
# NOT read SLURM_ARRAY_TASK_ID, so an array would fit cell 11 fifteen times.
for i in 11 12 14 15 16 17 18 20 21 22 23 24 26 27 28; do
  sbatch batch_scripts/run_train_lstm.sh $i 1
done

# the CPU is one env var away, for the device comparison or if the GPU turns out not to pay
M4_DEVICE=cpu sbatch batch_scripts/run_m4_sweeps.sh stride

# budgets, from the submitting shell; the defaults are the scans' own
RIKFLOW_M4_UPDATES=3000 sbatch batch_scripts/run_m4_sweeps.sh stride
RIKFLOW_M4_EPOCHS=5     sbatch batch_scripts/run_train_lstm.sh 19 1    # smoke one grid cell

# the lr scan is CLOSED (§6.1) -- kept runnable, not part of the programme
# sbatch batch_scripts/run_m4_sweeps.sh lr
```

Each writes one file into `exp_square_HIT/output/TO_LSTM/` — `lr_scan_<cell>_seed<n>.jld2` or
`stride_scan_<cell>_seed<n>.jld2` — holding every point's full curve, so the summary table can be
recomputed without refitting.

**What to expect.** Measured on a workstation CPU core, so a cluster core is the right order.
🔴 **There is no GPU timing yet** — that is what the first submission produces:

| job | points | per point (CPU) | total (CPU) |
|---|---|---|---|
| `stride`, 3000 updates | 5 | ~10 min | **~50 min** |
| one grid cell, 3000 epochs | 1 | ~10 min | ~10 min |
| the 15 distinct grid cells, 1 seed | 15 | ~10 min | ~2.5 h **wall, if serial** — they are separate jobs, so in practice the queue decides |

The scripts ask for **2 h**. ⚠️ The **first** point of any scan includes Julia compilation — the
smoke run measured **100.3 s** against **0.7–1.9 s** for the four after it. Gotcha #44 in
miniature: never read the first point's wall time as a cost. That applies twice over on the GPU,
where the first kernel launch also pays CUDA compilation.

### 🔴 The GPU is the default, and the first run is a test

`run_m4_sweeps.sh` and `run_train_lstm.sh` both set `M4_DEVICE=${M4_DEVICE:-cuda}` on
`gpu_h100 --gpus=1`. Before 2026-09-18 both asked for a GPU and ran on the node's CPU — nothing in
the training path moved an array to a device — and `run_train_lstm.sh`'s header said so. §5 has the
mechanism; what matters operationally:

- 🔴 **The device path has never run on a GPU.** V54 verifies it against `JLArrays`, which enforces
  the same no-scalar-indexing semantics on the host, and all four architectures agree with the CPU
  fit to ~1e-7 — but that cannot see a CUDA **compilation** failure, which is the class this
  repository has been bitten by twice and both times only on the cluster (#56, #57).
  🔴 **So the first job on a device is `run_m4_sweeps.sh smoke`, never a scan.**
  `tools/m4_smoke.jl` runs a few seconds of arithmetic through all eight stages — extension,
  device, QoIs, regressor, fit, host return, save/load round-trip, one deployed step — printing a
  timestamped line at each, and exits 0 with `M4 SMOKE PASS` or non-zero naming the phase.
  ⚠️ **A scan is a bad first job**, which is how a 20-minute run came to be cancelled blind on
  2026-09-18: one point is thousands of updates and printed nothing until it finished, so a slow
  device and a hung one looked identical. Measured afterwards on CPU, that job was not hung —
  point 1 alone is ~12 min and the whole 5-point scan ~3.4 h, against a 2 h walltime (below).
  ✅ **`M4_DEVICE=jl` runs the same smoke through `JLArray`**, i.e. GPU *semantics* with no GPU.
  Verified 2026-09-18: PASS on `cpu` and on `jl`, with **identical** validation losses
  (1.609 → 1.209 → 1.034), which is the host-side RNG design doing what §5 claims.

### The stride scan's cost, measured

🔴 **Measured by the smoke's stage 9, which is what a walltime should be set from.** An earlier
paragraph here projected ~3.4 h by assuming per-epoch cost scales with the segment count; **that
was wrong by a factor of three and the error was extrapolating instead of measuring.** On CPU:

| stride | batch | segs | u/ep | epochs | s/epoch | point |
|---|---|---|---|---|---|---|
| 400 | 32 | 7 | 2 | 1500 | 0.597 | 14.9 min |
| 400 | 2 | 7 | 4 | 750 | 0.433 | 5.4 min |
| 100 | 32 | 25 | 2 | 1500 | 0.703 | 17.6 min |
| 50 | 32 | 49 | 3 | 1000 | 0.987 | 16.5 min |
| 20 | 32 | 120 | 5 | 600 | 1.986 | 19.9 min |
| | | | | | | **74 min total** |

🔑 **17× the segments costs only 3.3× the time, and that is `batch = 32` earning its keep.** More
segments per update means wider GEMMs, so the per-timestep Zygote overhead §5 identifies is
amortised over more work — the same amortisation argument that makes a short stride the only
geometry where a GPU could help. A naive "cost ∝ segments" projection misses it entirely.

⚠️ So the scan **does** fit a 2 h walltime on CPU, though not with much margin: the smoke suggests
`-t 03:00:00` (1.5× the measured total, plus 15 min of package load and compilation). Run the
smoke on the GPU and take its number rather than this one.

⚠️ **`jldsave` still runs only after all five points**, so a walltime kill produces no output at
all. Writing results incrementally is not done and is worth doing before a long run.
- ✅ **A mis-scheduled job fails loudly.** `m4_device` refuses `cuda` when `CUDA.functional()` is
  false rather than falling back to the host, so a job that lands without a device dies at load
  instead of reporting CPU time as GPU time. Verified on this workstation.
- ⚠️ **Expect it to be slower at the current geometry.** §5's measurement, not a guess about the
  hardware: ~75 MFLOP per gradient step at ~0.34 GFLOP/s over a strictly sequential 500-step
  recurrence. 🔑 **The case where it could pay is exactly what §6.2 sweeps** — a short `stride`
  with `batch = 32` widens each launch from 7 segments to 32. Read the device comparison off the
  stride scan rather than off the baseline row.
- **`M4_DEVICE=cpu sbatch ...`** is the per-submission override, and is how the device comparison
  is taken.

**Do the `VAR=value sbatch ...` prefixes actually reach the compute node?** Yes. `sbatch` defaults
to `--export=ALL`, so the job inherits the submitting environment — and the evidence is already in
this directory: **no script here sets `--export`, and every one of them needs an inherited `PATH`
just to find `julia`.** If the default were anything else, no job in this repository would ever
have run. Both M4 scripts now state it (`#SBATCH --export=ALL`) rather than rely on it, which also
protects against a site default of `NONE`.

🔴 **The trap is the explicit form.** `sbatch --export=RIKFLOW_M4_UPDATES=5 ...` **replaces** `ALL`
rather than adding to it, so the job loses `PATH` and dies before Julia starts. The additive form
is `--export=ALL,RIKFLOW_M4_UPDATES=5`. The prefix form, `RIKFLOW_M4_UPDATES=5 sbatch ...`, has no
such hazard and is what §7's examples use.

🔑 **And the log answers it rather than the reader inferring it.** `run_m4_sweeps.sh` echoes the
budgets it received before Julia starts, and the driver prints them again from its own side
(`@info "M4 stride scan" … updates=5`). Two lines, one on each side of the shell/Julia boundary, so
a value that failed to cross is visible instead of silently becoming the default.

⚠️ **A CPU partition remains available and is no longer an untested risk.** `m4_lr_scan.jl` —
`using RikFlow` **plus** the Lux extension, strictly more imports than the training driver — ran six
fits to completion on a machine with **no GPU at all**, so `run_train_lrs.sh`'s worry about *"CUDA.jl
initialising with no device"* does not bite. `--partition=rome` (Zen2) or `genoa` (Zen4) are both
covered by the shared depot's `JULIA_CPU_TARGET`, so neither needs a new depot; pair either with
`M4_DEVICE=cpu`. 🔴 Confirm the partition name with `sinfo` — it is the one thing here that could
not be checked from inside the repository.

⚠️ **`OPENBLAS_NUM_THREADS=1` is set on both scripts and still matters on the GPU path**: the
batches are assembled on the host, and on `64 x 23` matrices a full BLAS pool contends rather than
helps (§5).

### What is *not* a single job

🔑 **The grid and the production fits are many jobs, and that is where the parallelism is.**
`run_train_lstm.sh <cell> [seed]` fits one cell. §6.3's grid is 15 distinct cells at one seed to
explore; S6 then wants 5 seeds on whichever survive. These are independent fits, which is the
"parallelism is across fits, not inside them" point of §5 actually paying — and the reason a GPU
per job is a weaker lever than more jobs.

🔴 **Submit them as a LOOP of `sbatch`, not as `--array`.** `run_train_lstm.sh` takes the cell as
`$1` and does **not** read `SLURM_ARRAY_TASK_ID`, so `--array=11,12,...` would run cell 11 fifteen
times, with nothing in the output saying so.
⚠️ Its own header records a second trap: if the *seeds* are ever split across jobs, **one final
`11_train_StochLSTM.jl` pass without a seed argument is still needed** to write
`seed_summary.jld2`, or `12_online_StochLSTM.jl` silently deploys seed 1 instead of the median.

⚠️ **The scans could be split across jobs too and deliberately are not.** It would work — the data
preparation is deterministic given the cache and `train_range`, and the seed is passed explicitly —
but it costs the property that makes a scan a comparison (one process, one prepared dataset, one
initialisation) and needs an aggregation step for the thresholds, which are read off the best point
in the scan. At ~50 minutes that trade is not worth making.

---

## 8. How to reproduce

```bash
# once, per checkout: the training environment (the only one with Lux)
julia --project=lib/RikFlow/training -e 'using Pkg; Pkg.instantiate()'

# the configuration table (gitignored output, so it must be regenerated per checkout)
julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/10_setup_lstm.jl

# fit one cell.  A trailing seed fits that seed only; without it, all n_seeds in sequence.
# RIKFLOW_QOI_CACHE points at an extracted QoI cache; without it the driver reads the 2.7 GB
# tracking record instead.
RIKFLOW_QOI_CACHE=analysis/data/data_track_dns512_..._f64_lmwray3_qois.jld2 \
  julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/11_train_StochLSTM.jl 2 1

# the loss curves (§6)
julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/plot_lstm_losses.jl 2 5 7

# the two scans (§6.1, §6.2). Runnable here -- ~18 min and ~50 min -- but §7 is how they are
# meant to be run, and NOT as a GPU job (§5, "Why not a GPU").
RIKFLOW_M4_LR_EPOCHS=1000 \
  julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_lr_scan.jl 2 1
RIKFLOW_M4_UPDATES=3000 \
  julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/tools/m4_stride_scan.jl 2 1

# score it post-run (the faithful Sørensen setting: teacher-forced, no solver)
RIKFLOW_QOI_CACHE=... \
  julia --project=lib/RikFlow/training lib/RikFlow/analysis/postrun_lstm.jl 2

# per-step cost against the S4 budget (no Lux needed — the deployed path is stdlib)
julia --project=lib/RikFlow lib/RikFlow/exp_square_HIT/tools/m4_cost_probe.jl
```

⚠️ `RIKFLOW_M4_EPOCHS` overrides the configured epoch count and warns when it fires. It is a
smoke-test switch and the knob §6's long runs were made with — **not** a way to run the experiment
without changing the table.

---

## 9. Open questions

**Q2 — `beta`, and Q3 — the latent dimension. ⏳ OPEN, and now asked together.** At the source's
`1e-4` the KL contributes ~0.01 to a loss of order 10, so the latent is almost unregularised and
can carry arbitrary spread — which matters more, not less, now that it is the only noise channel.
And `n_latent` is *how much latent there is to regularise*, so the two are not separable; `n_hidden`
is the third, because it sets how much the model can do if the latent contributes nothing.
🔴 **§6.3's 18-cell grid is the answer to both**, cells 11–28.

**Q7 — which architecture carries the grid. ⏳ OPEN, decided provisionally as `:storn`.** Not
specified; 18 cells on `:storn` and `:vrnn` is 36. `:storn` has no decoder skip, so the latent can
act only through the recurrence, which is the sharper test of the source's claim. `GRID_ARCH`
switches it (§6.3).

**Q8 — does a GPU help. ⏳ OPEN, and now answerable.** The device path is implemented and verified
against `JLArrays` (§5); it has never run on a GPU, and the measured arithmetic intensity says it
should be slower except possibly at §6.2's shortest strides.

**Resolved, kept because re-deriving them wastes a session:** the emission head is off (§1,
deviation 1) · `h = 1`, because an `h = 5` window is 4% of one decay time (§1) · `L = 500`,
`burn = 100` from the crossings rather than `T_int` (§3) · the record stays R1's tracking record,
the pairing M0 is fitted to (§4) · **`lr = 1e-2`, fixed and no longer swept** (§6.1) ·
**`epochs = 3000`**, from the measured plateau at ~2000 (§6).
