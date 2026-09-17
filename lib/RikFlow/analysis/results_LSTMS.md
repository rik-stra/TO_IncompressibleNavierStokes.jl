# Results — M4, the stochastic LSTM, on HIT

Companion to [`results.md`](results.md), which reports M0 and the DDN. Same convention: **where a
number here disagrees with a design document, this file is the measurement and the document is the
prediction.**

M4 is `plan.md` §3's off-grid exploratory cell — the stochastic LSTM of Barthel Sørensen et al.
(2026). It enters no attribution difference, is never the confirmatory cell, and is first on the
cut list. Design and build notes live in `meta_files/handoff_m4_stochastic_lstm.md`; this file is
only the measurements.

**Status, 2026-09-17.** The architecture, the training code and both deployment drivers are built
and tested. 🔴 **There are no valid architecture results yet.** A first run was made and its
conclusions were **retracted** — see §5.1. A corrected run is what §5.2 reports.

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

**Measured on this laptop**, fitting `StochLSTM4` (VRNN, h=1, hidden/latent/encoder 60) on R1's
tracked QoIs over `train_range = (400, 4000)` — 3599 rows, 19 features, `L = 200`, `burn = 50`:

| epochs | wall |
|---|---|
| 10 | 186 s |
| 110 | 528 s |

⇒ **3.42 s per epoch marginal**, with ~152 s of fixed overhead per process (Julia start, Zygote
compilation, reading the QoI cache). From which:

| unit of work | cost |
|---|---|
| one fit, 300 epochs | **~20 min** |
| one cell, 5 seeds (S6), sequential in one process | ~88 min |
| **4 architectures × 5 seeds × 300 epochs, 4 processes in parallel** | **~90 min** |
| the full 11-row config table × 5 seeds, 4 at a time | ~4 h |

🔑 **So the whole configured offline protocol is an overnight job at worst and a lunch break at
best.** There is no reason to run the offline phase on a cluster, and no reason to economise on
seeds or epochs — §5.1 is what economising cost.

⚠️ One practical note: run the parallel processes with `OPENBLAS_NUM_THREADS=1`. Four Julia
processes each spawning a full BLAS pool contend badly enough to be slower than running them
sequentially.

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
| emission | Gaussian, `Sigma = D R D` — see §8.6 Q1, the choice I am least sure about |
| parameters | **~2 976** at 16/4, ~8 464 at 32/8 |

### 8.3 Which variants, and why

🔑 **Sørensen et al.'s conclusion is that the noise belongs in the latent state** — the variants
that beat the others are the ones that feed `z` *into the recurrence*. So the first experiment is
those two:

| run | `arch` | `z` into cell | `z` into decoder | role |
|---|---|---|---|---|
| **A** | `:storn` | ✅ | — | upstream stochasticity, no output skip |
| **B** | `:vrnn` | ✅ | ✅ | upstream stochasticity + skip |
| **C** | `:lstm` | — | — | **control** |

⚠️ **I would keep C even though it is not one of the two.** Without a deterministic control there is
no statement to make: *"the latent path helped"* is only meaningful against a model that has none,
and C costs three minutes. `:vaernn` (output-only noise) is the one I would **drop** from the first
pass — it is the variant their result argues against, and it can be added later if A/B look
promising.

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
Against `results.md` §1's measured timescales for the **level** `q`, which is what the model
predicts:

| quantity | measured | in steps |
|---|---|---|
| `T_int(q)`, median | 0.474 TU | **~190** |
| `T_int(q)`, range | 0.249–0.543 TU | 100–217 |
| 1/e time of `q` | 0.29–0.36 TU | **116–144** |
| *(for contrast)* `T_int(dQ)`, median | 0.0914 TU | 37 |

🔴 **`burn = 50` is shorter than the level's 1/e time on every band, and about a quarter of its
`T_int`.** The hidden state is therefore being scored before it has seen one correlation time of
history — precisely the cold-start contamination the burn-in exists to remove. And `L = 200` leaves
only 150 scored steps, under one `T_int`.

**Proposed instead: `L = 400`, `burn = 150`** (1.0 TU and 0.375 TU; the burn-in is then ~0.8
`T_int` and past the 1/e time on every band). Cost is roughly unchanged: with `stride = L - burn`
the total number of scored steps per epoch is nearly the same — the same data cut into fewer,
longer pieces.

⚠️ **The same argument applies to the online warm-up.** `nwarm = 100` (0.25 TU) is about half the
level's `T_int`. It was inherited from the linear cells, where the lag window is the only state;
M4 also has to charge `(h, c)`. That is prerequisite P1 in
`meta_files/handoff_m4_stochastic_lstm.md` §12.1, and it is not optional.

### 8.6 Open questions — what I would like comments on

**Q1 — the emission head is a second noise channel, and it may undercut the whole point.**
🔴 The clearest issue in the current design. Sørensen's decoder is **deterministic**: the latent
path is their only stochasticity. I added a state-dependent Gaussian head (`log d` linear in `h_t`)
because this project selects on likelihood and has a calibrated-spread criterion (S7). But that
gives the model **two** ways to produce spread, with nothing in the objective allocating between
them — so *"the noise is in the latent state"* may stop being true of the fitted model even though
the architecture flag says it is.
Options: **(a)** constant `Sigma` — freeze `Wd = 0`, leave `bd` free, so the latent is the **only**
state-dependent noise source; **(b)** keep the state-dependent head; **(c)** run both.
**Recommendation: (a) first.** It is closest to their design while still giving a likelihood, and
it makes the latent path's contribution unambiguous. (b) then becomes a second rung with a clean
interpretation rather than a confound.

**Q2 — `beta`.** At the source's `1e-4` the KL contributes ~0.01 to a loss of order 10, so the
latent is almost unregularised and can carry arbitrary spread. If the claim is about the latent
path, `beta` is a model parameter and not a detail. Sweep `{0, 1e-4, 1e-2}`, or something else?

**Q3 — latent dimension relative to `N_Q = 6`.** 4 (under-complete, forces compression), 6
(matched), or 8? No strong prior here; their 60-for-a-field gives no guidance at this scale.

**Q4 — `h`.** `h = 1`, on the grounds that the recurrence is supposed to carry the memory? Or
`h = 5` to match M0's regressor exactly so the data budget is like-for-like? Both are cheap.

**Q5 — `L = 400`, `burn = 150`?** §8.5's argument, for confirmation or correction.

**Q6 — which record?** Currently the tracking record's `(q*, q)`, the same pairing M0 is fitted to.
The literal Sørensen setup instead maps a *free-running* LF trajectory to HF, which would use
`9_no_sgs.jl`'s output and is a different dataset. Worth doing, or out of scope?

### 8.7 Cost, measured

| configuration | params | s/epoch | 300 epochs |
|---|---|---|---|
| source dims (60/60/60) | 43 128 | 2.05 | 10.2 min |
| 32 / 8 / no encoder | 8 464 | 1.26 | 6.3 min |
| **16 / 4 / no encoder** | **2 976** | **0.58** | **2.9 min** |
| 16 / 4 / no encoder, `:storn` | 2 952 | 0.63 | 3.1 min |

Runs A + B + C at 3 seeds is therefore **~30 minutes** — an experimentation loop rather than an
overnight job.

⚠️ Note the shape of that speedup: 14x fewer parameters buys only 3.5x less time, because the cost
is dominated by Zygote's per-timestep overhead over the `L`-step recurrence rather than by
arithmetic. Raising `L` to 400 will therefore cost closer to linearly in `L` than the flop count
suggests. If this loop ever needs to be faster, the thing to fix is the AD overhead, not the model.
