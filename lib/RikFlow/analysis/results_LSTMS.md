# Results — M4, the stochastic LSTM, on HIT

Companion to [`results.md`](results.md), which reports M0 and the DDN. Same convention: **where a
number here disagrees with a design document, this file is the measurement and the document is the
prediction.**

M4 is `plan.md` §3's off-grid exploratory cell — the stochastic LSTM of Barthel Sørensen et al.
(2026). It enters no attribution difference, is never the confirmatory cell, and is first on the
cut list. Design and build notes live in `meta_files/handoff_m4_stochastic_lstm.md`; this file is
only the measurements.

**Status, 2026-09-17.** The architecture, the training code and both deployment drivers are built
and tested; the experiment in §8 is approved, encoded in the config table and verified by
construction. 🔴 **Nothing has been fitted since the training loop was repaired, so there are no
architecture results at all.** The one run that was made had its conclusions **retracted** (§5.1),
and §5.2 is deliberately empty rather than filled with numbers from a superseded loop.

**To produce §5.2** (~7 minutes, see §4):

```bash
for i in 1 2 3 4 5 6 7; do
  RIKFLOW_QOI_CACHE=... julia --project=lib/RikFlow/training \
    lib/RikFlow/exp_square_HIT/11_train_StochLSTM.jl $i 1
done
```

then `analysis/postrun_lstm.jl <i>` for each, and re-run the winning β at all five seeds.

---

## 1. The architecture

Four variants, one flag apart, because the source's claim is a *ranking* across them rather than a
statement about one model. With `x_t` the regressor row and `y_t` the prediction:

```
z_t  ~  q(z_t | x_t) = N( mu_z , sigma_z )     mu_z = B_mu e_t ,  sigma_z = softplus(B_sig e_t)
e_t  =  tanh( W_e x_t + b_e )                  single dense encoder layer
h_t  =  LSTM( [x_t ; z_t] , h_{t-1} )          hidden 60
y_t  =  V1 h_t  +  V2 z_t  +  c                linear output
```

| `arch` | `z` drawn | `z` → cell | `z` → decoder | is |
|---|---|---|---|---|
| `:lstm` | no | no | no | deterministic LSTM + Gaussian head |
| `:vaernn` | yes | no | yes | stochasticity at the **output** only |
| `:storn` | yes | yes | no | **STORN** |
| `:vrnn` | yes | yes | yes | **VRNN** |

🔑 **"Upstream stochasticity" — the property Sørensen et al. report as the one that matters — is
exactly the `z → cell` column.** STORN and VRNN have it; the deterministic and output-only variants
do not.

🔴 **The first line is the ENCODER, the approximate posterior `q(z|x_t)` — not the prior.** The
prior is a fixed `N(0, I)`. This was settled against the reference implementation
(`ben-barthel/learning_dynamics`, `ML_Code/networks_qg.py`), which is unambiguous: the encoder sees
**only `x_t`** — not the target, not `h_{t-1}` — and there is no separate recognition RNN. Reading
`methods_overview.tex`'s equation as the prior is the one way to get this structurally wrong.
Because `x_t` is observed at deployment, the same encoder serves training and inference, which is
what makes the architecture deployable at all.

### Dimensions, and what the input is

Hidden 60, latent 60, encoder 60 — the source's. ⚠️ The latent dimension is **not** `N_Q`; an early
reading assumed it matched the output width and it does not.

🔴 **These are the source's dimensions and we do not propose to use them.** At `h = 1` they give
43 128 parameters against 21 594 training target values — twice as many parameters as data. Their
60 units predict a QG *field*; we predict six scalars. **§8 is the sizing argument and the
architecture actually proposed for testing.**

The input `x_t` is **the same regressor every other cell on the ladder uses** — `build_history`
with a `HistorySpec`, so `q^{n*}` plus `h` lags of `(q, q*)` plus a bias. M4 introduces no new
input layout, which is what lets V1/V2 cover it and keeps the data budget comparable. At `h = 1`,
`N_Q = 6` that is 19 features.

### Two deliberate deviations from the source

1. 🔴 **A Gaussian emission head, where the source is deterministic.** Their decoder emits a point
   and the objective is a plain MSE, so their model has **no predictive density at all**. This
   project selects on held-out likelihood and has a calibrated-spread criterion (S7), and
   `plan.md` §3 specifies a *"Gaussian emission head"* for M4 in as many words. So the decoder here
   carries the L2 head, `Sigma^n = D^n R D^n` with `log d_i` linear in `h_t` and `R` a constant
   correlation matrix. **This is a deviation, not a reproduction, and it must be stated wherever
   M4 is reported.**
2. **The conservation-of-mass penalty is not imported.** It is L4, out of scope (`plan.md` §24).
   M4 takes the architecture, not the penalty.

⚠️ Two links, on purpose: **softplus** on the latent scale (the source's), **log** on the emission
scale (this project's, for the convexity and collapsing log-determinant that
`methods_overview.tex` argues for, and which explicitly rejects softplus there).

---

## 2. The loss function

Per-step negative ELBO, summed over the scored part of a segment:

```
L  =  sum_t [  -log N( q^n ; mu_y(h_t, z_t), Sigma^n )  +  beta * KL( q(z_t|x_t) || N(0, I) )  ]
```

- The reconstruction term is a **genuine Gaussian log-density**, not an MSE — that is deviation (1)
  above, and it is what makes a held-out likelihood definable at all.
- The KL is to a **fixed standard normal**, confirmed from `ML_Code/loss_funs_qg.py`.
- ⚠️ **`beta`, never `lambda`.** The source calls this weight `lambda` and uses `1e-4`; `lambda` is
  the ridge parameter throughout this project and a collision would corrupt the §8a analysis and
  the S2′ axis.

### The covariance is parametrised differently in training and deployment

Training carries a free lower-triangular **precision** factor `A`:

```
Sigma^{-1} = D^{-1} A' A D^{-1}      quad = || A (r ./ d) ||^2
log det Sigma = 2 sum(log d) - 2 sum(log diag(A))
```

which is unconstrained, needs no triangular solve and differentiates cleanly. Deployment carries
the specified `Sigma = D R D` with `R` a correlation matrix. These are the same family: `A` leaves
`R`'s scale confounded with `d` by a per-coordinate constant only, and the conversion normalises
`R` to unit diagonal and folds the scale into `bd`. **V41 is the test that the two give the same
density**, not merely the same mean.

### What the loss is NOT

🔴 **It is an upper bound on the NLL, not the NLL.** The reconstruction term uses a single `z`
draw, so the training objective is a one-sample ELBO. For `:lstm` there is no latent path and the
objective *is* the exact NLL. **The two are therefore not on the same scale, and a training or
validation loss must never be used to rank `:lstm` against the latent architectures.** Held-out
comparison uses the IWAE bound (§3) and ensemble CRPS.

---

## 3. The training method

| | |
|---|---|
| optimiser | Adam, `lr = 1e-3`, decayed by `0.3` after `patience = 20` epochs without validation improvement, floor `1e-5` |
| batching | segments, not rows — `batch = 8` segments per step |
| segments | length `L = 200`, burn-in `burn = 50`, stride `L - burn` — ⚠️ **both under review, see §8.5**: `burn = 50` is shorter than the level's 1/e time on every band |
| epochs | 300 |
| seeds | 5 per cell (S6: neural cells are fitted with five seeds, spread reported, **median seed deployed**) |
| precision | Float32 weights; the record is Float64 |
| reparametrisation | `z = mu + sigma * eps` with `eps` drawn **outside** the differentiated function |

**Segments and burn-in.** `build_history` returns rows in increasing step order with their step
indices, so BPTT segments are contiguous slices. Two things the segmenter adds: segments **never
cross a discontinuity** in the step index, and the first `burn` rows of each segment charge the
hidden state and are **excluded from the loss**. Without the burn-in every segment contributes a
cold-start term from `h = 0`, which the model can only fit by learning to predict well from no
history — exactly the behaviour the recurrence exists to avoid.

### 🔴 Three things the training loop must do, learned the hard way

All three were absent in the first implementation and all three invalidated its results (§5.1):

1. **Return the best-validation iterate, not the last.** Without checkpointing, the fit that gets
   saved is whatever the final epoch happened to land on.
2. **Decay the learning rate on plateau.** At a constant `1e-3` the end of training oscillated by
   several nats — wider than the differences between architectures.
3. **Hold the validation noise draws fixed.** Re-drawing `eps` each epoch makes the validation loss
   a fresh one-sample estimate, so epoch-to-epoch comparison mixes "the model changed" with "the
   draw changed", and any best-validation rule selects partly on a lucky draw.

`V49` in `training/runtests_lux.jl` pins all three.

### Scoring — what M4 breaks

🔴 A stochastic latent path has **no closed-form one-step predictive density**, so exact NLL,
closed-form CRPS and `companion`/ρ(C̃) are **undefined** for M4.

| metric | on M4 |
|---|---|
| exact one-step NLL | ❌ undefined |
| closed-form CRPS | ❌ undefined |
| ρ(C̃), starred gain, total block sum | ❌ undefined — no linear mean |
| `crps_ensemble` | ✅ unchanged |
| rank histogram + Jolliffe–Primo | ✅ unchanged — **the one axis the whole ladder reads on** |
| IWAE-K bound (`nll_iwae`) | ✅ but see below |

🔴 **`nll_iwae` is a lower bound on `log p`, hence an upper bound on the NLL. It is not comparable
to M0's exact NLL and must never share a column with one.** This is why `plan.md` §9 makes the
ensemble rank histogram, not PIT, the ladder-wide calibration metric: it is the only one that reads
the same on M0 and M4.

**Reading the rank histogram.** With `M` ensemble members there are `K = M + 1` bins and `M` degrees
of freedom, so **`chi2_eff ≈ M` is calibration**, not zero. ⚠️ And a value far *below* `M` is not a
pass — it is `results.md` §2's *"flat is not skilful"* trap from the other side. Always report the
paired dynamics statistic (lag-1 autocorrelation of the predicted mean `dQ`) beside it.

---

## 4. Can this be done locally? — yes for offline, no for online

**Short answer: the entire offline programme runs on this laptop in hours. The online programme
cannot run here at all.**

### Offline — feasible

The model is tiny: ~10⁴ parameters on 6 QoIs. Training is CPU-bound Julia; no GPU is involved, and
`11_train_StochLSTM.jl` never touches CUDA.

**Measured on this laptop**, on R1's tracked QoIs over `train_range = (400, 4000)` — 3599 rows,
19 features. The number moved a long way during the build, and the steps are worth recording
because three of them are lessons rather than tuning:

| state | s/epoch | note |
|---|---|---|
| as first written, 60/60/60, `L = 200` | 3.42 | one `L`-step loop per segment |
| + segments batched through one recurrence | 2.05 | `B` GEMVs become one GEMM (V50) |
| + right-sized net, 16/4/no encoder | 0.58 | 43 128 → 2 976 parameters (§8.1) |
| **+ O(L) reverse pass, at `L = 400`** | **0.178** | the per-step slice was O(L²) — see below |

🔴 **The last step was the big one and it was a bug, not a tuning knob.** `GX[:, t, :]` inside the
recurrence looks free, but Zygote's pullback for a slice allocates a *parent-sized* zero array and
scatters into it — so an `L`-step loop did `L` allocations of `4H × L × B`, making the reverse pass
quadratic in `L` while the forward pass was linear. One gradient step allocated 172 MiB for a
forward pass allocating 3.8 MiB. Slicing once behind an adjoint that accumulates into a single
buffer restored O(L): 246 ms → 51 ms, and reverse/forward from 26× down to 5.8×.

Note the last row is at **`L = 400`**, double the sequence length of the rows above it — so per
unit of sequence the improvement is about **13×**.

### The approved configuration, timed

| run | `arch` | `emission` | params | s/epoch | 300 epochs |
|---|---|---|---|---|---|
| A | `:storn` | `:none` | 2 952 | 0.178 | **0.9 min** |
| B | `:vrnn` | `:none` | 2 976 | 0.185 | **0.9 min** |
| C | `:lstm` | `:constant` | 2 696 | 0.245 | **1.2 min** |

| unit of work | cost |
|---|---|
| the six-point β scan at one seed, plus the control | **~7 min** |
| one cell at all 5 seeds (S6) | ~5 min |
| the full 10-row table × 5 seeds | **~50 min** |

🔑 **The offline programme is an experimentation loop, not a batch job.** There is no reason to run
it on a cluster and no reason to economise on seeds or epochs — §5.1 is what economising cost.

⚠️ If several fits are ever run in parallel, set `OPENBLAS_NUM_THREADS=1`: Julia processes each
spawning a full BLAS pool contend badly enough to be slower than running sequentially. At these
speeds parallelism is not needed.

### Online — not feasible locally

`12_online_StochLSTM.jl` couples the closure into a 64³ LES for 40 001 steps per replica, five
replicas per cell. That needs a GPU and R1's 2.7 GB tracking record. On Snellius this is the same
cost as an LRS online run (~50 min for five replicas, per `run_online.sh`'s estimate). **It has
never been executed** — it is the one part of the M4 build that has not been run end to end.

D6 — the multi-IC ensemble S7-online needs — is `K = 90 × M = 10` coupled runs, and is a cluster
job by any measure.

🔑 **So the split of work is clear: everything up to and including offline calibration can be done
here and now; S7-online and the calibration gate need the cluster.** That is a comfortable position
— the offline phase is where the architecture question is decided, and `plan.md` §7's whole
concern is that offline results may *not* transfer, which is a question you can only ask once the
offline numbers exist.

---

## 5. Offline results

### 5.1 🔴 The first run, and why it was retracted

Four architectures, one seed, 100 epochs, fitted on R1's tracked QoIs over `train_range = (400,
4000)` and scored post-run on the disjoint window `(4000, 7600)`.

**Its conclusions were withdrawn in full.** The check that broke them was simply asking whether
anything had converged:

| config | val @ epoch 91 | val @ epoch 100 (what was scored) |
|---|---|---|
| `:lstm` | −10.79 | −10.87 |
| `:vaernn` | **−12.14** | −4.21 |
| `:storn` | **−11.81** | −4.94 |
| `:vrnn` | **−12.57** | −3.54 |

At epoch 91 all three latent architectures were **better** than the deterministic backbone; nine
epochs later they were ~8 nats worse, and those were the iterates that got saved and scored. The
run therefore inverted the architecture ranking, and a calibration diagnosis built on that ranking
("the latent path injects spread not matched to the error") was wrong and is retracted.

🔑 **The lesson is recorded rather than buried, because it is a general one:** a fit is not a
result until the curve is shown to have converged, and a training loop that returns its last
iterate will manufacture rankings out of optimiser noise. The three fixes are §3's list.

**What survived:** the pipeline works end to end on real tracked data, and there is no posterior
collapse — KL 2.1–3.9 per latent dimension in all three latent architectures, so the latent path is
genuinely active.

### 5.2 The corrected run

*(pending — this section is filled by the run described in §4. The protocol is 4 architectures ×
`n` seeds × 300 epochs with best-iterate selection, scored by `analysis/postrun_lstm.jl` on the
window disjoint from training.)*

---

## 6. How to reproduce

```bash
# once, per checkout: the training environment (the only one with Lux)
julia --project=lib/RikFlow/training -e 'using Pkg; Pkg.instantiate()'

# the configuration table (gitignored output, so it must be regenerated per checkout)
julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/10_setup_lstm.jl

# fit one cell, all seeds.  RIKFLOW_QOI_CACHE points at an extracted QoI cache;
# without it the driver reads the 2.7 GB tracking record instead.
RIKFLOW_QOI_CACHE=analysis/data/data_track_dns512_..._qois.jld2 \
  julia --project=lib/RikFlow/training lib/RikFlow/exp_square_HIT/11_train_StochLSTM.jl 4

# score it post-run (the faithful Sørensen setting: teacher-forced, no solver)
RIKFLOW_QOI_CACHE=... \
  julia --project=lib/RikFlow/training lib/RikFlow/analysis/postrun_lstm.jl 4

# per-step cost against the S4 budget (no Lux needed — the deployed path is stdlib)
julia --project=lib/RikFlow lib/RikFlow/exp_square_HIT/tools/m4_cost_probe.jl
```

⚠️ `RIKFLOW_M4_EPOCHS` overrides the configured epoch count. It warns when it fires. It is a
smoke-test switch, **not** a way to run the experiment — §5.1 is what happens when a short run is
treated as a result.

---

## 7. Cost — S4, measured

The deployed closure is hand-written on plain arrays in `src/ts_lstm.jl`; Lux is never called from
the solver loop. Measured by `tools/m4_cost_probe.jl` (CPU, Float32):

| arch | h | hidden | latent | encoder | median / step | % of the S4 allowance |
|---|---|---|---|---|---|---|
| `:vrnn` | 1 | 60 | 60 | 60 | 43.9 µs | 15.8% |
| `:storn` | 1 | 60 | 60 | 60 | 35.7 µs | 12.8% |
| `:lstm` | 1 | 60 | 60 | 60 | 16.2 µs | 5.8% |
| `:vrnn` | 5 | 60 | 60 | 60 | 26.1 µs | 9.4% |
| `:vrnn` | 1 | 128 | 60 | 60 | 40.7 µs | 14.7% |
| `:vrnn` | 1 | 60 | 6 | 0 | 14.8 µs | 5.3% |

The allowance is 15% of the 1.85 ms/step surrogate share on HIT, i.e. **278 µs**. 🔑 **M4 is
admissible at the source's full dimensions with roughly six times the margin it needs.** S4 is a
kill criterion, not a target, and it is not the binding constraint here.

⚠️ This measures the closure only — not the FFTs, the QoI computation or the host/device round
trip, all of which are already inside the 1.85 ms.

---

## 8. Proposed experiment — ⚠️ FOR REVIEW, nothing has been run

Written 2026-09-17 for comment **before** any fitting. Every number below is either measured or a
choice; the choices are collected in §8.6 so they can be argued with individually.

### 8.1 Why the network shrinks

The source's dimensions are hidden 60, latent 60, encoder 60. Carried across unchanged, that is:

| configuration | parameters |
|---|---|
| source dims, `h = 1` | **43 128** |
| source dims, `h = 5` | 57 528 |
| hidden 60, latent 6, no encoder | 21 672 |
| **hidden 32, latent 8, no encoder** | **8 464** |
| **hidden 16, latent 4, no encoder** | **2 976** |

against **21 594 training target values** (3599 rows x 6 QoIs) on the `t in [1,10]` TU window.

🔴 **At the source's dimensions there are twice as many parameters as target values.** Their 60
units predict a quasi-geostrophic *field* — two channels over an `Ny x Nx` grid. We predict **six
scalars**. Copying the width across is copying a number, not the architecture.

⚠️ **Consequence for the write-up:** shrinking means we are **not reproducing their model**. We are
testing their *architectural claim* — that stochasticity injected upstream beats the alternatives —
at `N_Q = 6`. That is the more meaningful test, but the paper must say which of the two it did.

### 8.2 The proposed architecture

| | |
|---|---|
| input | the shared `build_history` regressor, `h = 1` → **19 features** (`q*^n`, `q^{n-1}`, `q*^{n-1}`, bias) |
| encoder | **none** (`n_encoder = 0`) — the linear encoder `methods_overview.tex` writes, not the repository's dense `tanh` layer |
| LSTM hidden | **16** (fallback 32) |
| latent | **4** (fallback 8) |
| decoder | linear, `V1 h_t + V2 z_t + c` |
| emission | **`:none`** — deterministic decoder, ✅ resolved in §8.6 Q1. `z` is the only stochasticity, as in the source |
| parameters | **~2 976** at 16/4, ~8 464 at 32/8 |

🔴 **With `emission = :none` there is no predictive density**, so these cells have no likelihood and
`iwae_nll` refuses. They are scored by **ensemble CRPS and the rank histogram**, which are defined
for them and are what `plan.md` §9 makes ladder-wide anyway.

### 8.3 Which variants, and why

🔑 **Sørensen et al.'s conclusion is that the noise belongs in the latent state** — the variants
that beat the others are the ones that feed `z` *into the recurrence*. So the first experiment is
those two:

| run | `arch` | `emission` | `z` into cell | `z` into decoder | role |
|---|---|---|---|---|---|
| **A** | `:storn` | `:none` | ✅ | — | upstream stochasticity, no output skip |
| **B** | `:vrnn` | `:none` | ✅ | ✅ | upstream stochasticity + skip |
| **C** | `:lstm` | `:constant` | — | — | **control** |

⚠️ **I would keep C even though it is not one of the two.** Without a control there is no statement
to make: *"the latent path helped"* is only meaningful against a model that has none, and C costs
three minutes.

🔴 **C must use `emission = :constant`, not `:none`.** A deterministic backbone with no emission
noise has no stochasticity at all — it is a point predictor, cannot produce an ensemble, and
`LSTMSpec` refuses the combination at construction. So the control's spread comes from a constant
`Sigma`, which is also the honest comparison: *"noise upstream"* against *"noise at the output,
state-independent"* — the latter being the DDN's design one level down.

`:vaernn` (output-only, state-dependent noise) is the one I would **drop** from the first pass. It
is the variant their result argues against, and it can be added later if A/B look promising.

### 8.4 Training setup

| | value | note |
|---|---|---|
| objective | negative ELBO, Gaussian reconstruction + `beta * KL(q || N(0,I))` | §2 |
| `beta` | **sweep {0, 1e-4, 1e-2}** | 1e-4 is theirs; at that weight the KL is nearly inactive — Q2 |
| optimiser | Adam, `lr = 1e-3` | decayed x0.3 after 20 epochs without improvement, floor 1e-5 |
| epochs | 300 | **best-validation iterate returned, not the last** (§3) |
| seeds | **3 for exploration, 5 for the record** | S6: spread reported, median seed deployed |
| batch | 8 segments | batched through one recurrence; V50 says this is a pure speed change |
| precision | Float32 weights | the record is Float64 |
| train window | `t in [1, 10]` TU, `train_range = (400, 4000)` | the window the M0 cells use |
| selection window | `t in [10, 19]` TU | disjoint from training **and** from the online reference |

### 8.5 🔑 Segment length and burn-in should come from the measured ACF, and currently do not

Current defaults are `L = 200`, `burn = 50`. At `dt = 2.5e-3` those are **0.5 TU** and **0.125 TU**.

⚠️ **Do not size these with `T_int`.** `results.md` §1 is explicit about why, and an earlier draft
of this section got it wrong: **the level's ACF is not a decaying exponential.** It falls to ~0.1
within half a TU and then *rings* between roughly −0.2 and +0.2 with a period near 1 TU, out to the
end of the 10 TU window. Any integral-based timescale is integrating that ringing, so "the"
decorrelation time is not a well-defined property of this series. `results.md` also settles a
factor-2 convention between the two integral estimators (#58) and notes that `T_exp` is meaningless
on the level, `ρ₁(q) ≈ 0.9999` giving 3–34 TU.

**The robust statistics are the crossings**, and they are what to size with (level `q`, all six
bands, from `results.md` §1):

| statistic | range over bands | in steps |
|---|---|---|
| lag at ρ = 1/e | **0.290–0.355 TU** | **116–142** |
| lag at ρ = 0.1 | 0.430–0.600 TU | 172–240 |
| ringing period | ≈ 1 TU | ≈ 400 |
| ±2 Bartlett se | 0.114–0.132 | — |

Note how tight the 1/e times are — they span only **1.22×** across bands, against 2.18× for `T_int`
— so a single `burn` serves every QoI.

🔴 **`burn = 50` is shorter than the 1/e crossing on every band**, so the hidden state is scored
before it has seen even one decay time of history — precisely the cold-start contamination the
burn-in exists to remove.

🔑 **And `L = 200` is half a ringing period, which is the more interesting problem.** A recurrence
is one of the few model classes that *can* represent an oscillating memory kernel — it is why
`methods_overview.tex` reaches for a Hankel basis over a bank of real-pole exponentials, citing
Gouasmi's decaying-sinusoid MZ kernel. At `L = 200` the model never sees a full period and cannot
distinguish a ring from a decay, so it is denied the one thing the architecture is good for.

**Proposed instead: `L = 400`, `burn = 150`** — one full ringing period, with the burn-in past the
1/e crossing on every band, leaving 250 scored steps. Cost is roughly unchanged: with
`stride = L - burn` the total scored steps per epoch is nearly the same, the same data cut into
fewer, longer pieces.

⚠️ **The same argument applies to the online warm-up.** `nwarm = 100` (0.25 TU) is about half the
level's `T_int`. It was inherited from the linear cells, where the lag window is the only state;
M4 also has to charge `(h, c)`. That is prerequisite P1 in
`meta_files/handoff_m4_stochastic_lstm.md` §12.1, and it is not optional.

### 8.6 Open questions — ✅ four resolved 2026-09-17, two still open

**Q1 — the emission head. ✅ RESOLVED: removed.** It was a second noise channel competing with the
latent path, so *"the noise is in the latent state"* could stop being true of the fitted model even
with the architecture flag set. `LSTMSpec` now takes `emission ∈ (:none, :constant,
:state_dependent)` and **the experiment uses `:none`** — the source's design, where `z` is the only
stochasticity and the decoder is deterministic.

Consequences, all stated rather than discovered later:

- The reconstruction term becomes a plain sum of squares, which is the source's objective.
- 🔴 **There is no predictive density, so there is no likelihood.** `iwae_nll` refuses for these
  cells rather than returning a number that looks like an NLL and is not one.
- **Ensemble scoring is unaffected**: drawing `z` still gives an ensemble, so `crps_ensemble`, the
  rank histogram with Jolliffe–Primo contrasts and spread–skill all read normally — and those are
  the metrics `plan.md` §9 makes ladder-wide precisely because they survive this kind of model.
- `:constant` remains one keyword away if a likelihood is wanted back for a particular comparison.
- ⚠️ `arch = :lstm` with `emission = :none` is refused at construction: it has no stochasticity at
  all and cannot produce an ensemble. So **the control C is `:lstm` with `emission = :constant`** —
  see §8.3's note below.

**Q2 — `beta`. ⏳ STILL OPEN.** At the source's `1e-4` the KL contributes ~0.01 to a loss of order
10, so the latent is almost unregularised and can carry arbitrary spread. Now that it is the *only*
noise channel this matters more, not less. Proposed sweep `{0, 1e-4, 1e-2}`.

**Q3 — latent dimension relative to `N_Q = 6`. ⏳ STILL OPEN.** 4 (under-complete, forces
compression), 6 (matched), or 8?

**Q4 — what it was, and the answer. ✅ RESOLVED: `h = 1`.** `h` is the length of the **explicit lag
window in the regressor** — how many past `(q, q*)` pairs are stacked into `x_t` alongside the
current `q*`. M0/TO-LRS uses `h = 5`. The question was whether to let the recurrence carry the
memory (`h = 1`) or hand the model the same explicit window M0 gets (`h = 5`) so the inputs match.

§8.5's measurement answers it: the level's 1/e time is **116–142 steps**. An `h = 5` lag window is
**five** steps — 0.0125 TU, about 4% of one decay time. It is negligible either way, so the
recurrence has to do the memory work regardless and `h = 5` buys nothing but 48 extra input
features. **`h = 1`.**

**Q5 — `L` and `burn`. ✅ RESOLVED: `L = 400`, `burn = 150`**, on §8.5's corrected reasoning — one
full ringing period, burn past the 1/e crossing on every band. ⚠️ The same argument says the online
warm-up `nwarm = 100` (0.25 TU) is under the 1/e crossing too; P1 measures it properly.

**Q6 — which record. ✅ RESOLVED: out of scope.** Stay with the tracking record's `(q*, q)`, the
same pairing M0 is fitted to. The literal Sørensen setup — a free-running LF trajectory mapped to
HF — is a different dataset and a different question; **our application is the closure**, and the
tracking record is what a closure is fitted to.
### 8.7 Cost, measured

See §4 for the full table and how it got there. At the approved settings —
hidden 16 / latent 4 / no encoder, `L = 400`, `burn = 150` — a 300-epoch fit is **0.9 min**
(`:storn`, `:vrnn`) or **1.2 min** (the `:constant` control), so:

| unit of work | cost |
|---|---|
| the β scan, 6 points at one seed + the control | **~7 min** |
| the winner re-run at all 5 seeds | ~5 min |

⚠️ Sizing the network down bought 3.5×, not the 14× the parameter count suggests — the cost was
dominated by Zygote's per-timestep overhead, not arithmetic. The remaining 3.3× came from fixing an
**O(L²) reverse pass**, which is a different kind of problem entirely and is the reason `L = 400`
is now affordable at all. Both are in §4.
