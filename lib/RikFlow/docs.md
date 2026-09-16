# RikFlow — reference

Deeper reference for `lib/RikFlow`. See [`README.md`](README.md) for the pipeline and the
directory map, and [`analysis/results.md`](analysis/results.md) for what the numbers say.

Contents:

1. [Data objects on disk](#1-data-objects-on-disk)
2. [Closure types](#2-closure-types-srctime_series_methodsjl)
3. [The `ts_*` layer](#3-the-ts_-layer)
4. [Metrics and regimes](#4-metrics-and-regimes)
5. [Environment variables](#5-environment-variables)

Throughout, `N_Q = 6`, the LF grid is 64³, the LF step is `Δt = 2.5e-3` and the HF step is
`2.5e-4`. A 100 TU run is `nt = 40 000` LF steps.

---

## 1. Data objects on disk

### 1.1 HF reference — `2_HF_ref.jl`

`output/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0_f64_lmwray3.jld2`

| JLD2 key | contents |
|---|---|
| `data_train` | `(; data, comptime)` returned by `filtersaver` |
| `params_train` | the full parameter NamedTuple the run was built from |

`data_train.data` is an array indexed by (LES resolution, filter); with one of each, the record of
interest is `data_train.data[1]`, which is the `(; u, qoi_hist)` NamedTuple built by `lesdatagen`:

| field | shape | note |
|---|---|---|
| `data_train.data[1].u` | `Vector` of 401 arrays, each `(64, 64, 64, 3)` | filtered LES velocity fields, every `plotfreq = 1000` DNS steps ⇒ 0.25 TU |
| `data_train.data[1].qoi_hist` | `Vector` of 40 001 length-6 vectors | QoIs every `savefreq = 10` DNS steps ⇒ one sample per LF step |

The QoI sampling rate is what makes `3_track_ref.jl` able to index `qoi_hist[1:Int(tsim/Δt)+1]`
with the *LF* `Δt`: `savefreq * 2.5e-4 == 2.5e-3`. The driver asserts the resulting column count
rather than assuming it.

⚠️ `plotfreq` is only tested on steps where `n % savefreq == 0`, because `filtersaver` gates its
inner observable on `savefreq` first. A `plotfreq` that is not a multiple of `savefreq` silently
stores fields at `lcm(savefreq, plotfreq)`.

Checkpoints go to `output/checkpoints/checkpoint_n<n>.jld2` with keys `results` and `u_cpu`.
**Nothing in RikFlow reads a checkpoint back** — `create_ref_data` has no resume path.

### 1.2 Tracking record — `3_track_ref.jl`

`output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2`

| JLD2 key | contents |
|---|---|
| `data_track` | `(; dQ, tau, q, q_star, fields)` — the return value of `track_ref` |
| `params_track` | the parameter NamedTuple, including `ou_bodyforce` with `freeze = 1` |

| field | shape | note |
|---|---|---|
| `q` | `6 × 40001` | corrected QoIs. **Column 1 is `t = 0`**, so `q[:, n+1]` is step `n` |
| `q_star` | `6 × 40000` | the QoIs the uncorrected step would have produced, column `n` = step `n` |
| `dQ` | `6 × 40000` | `q_ref - q_star`; the training target |
| `tau` | `6 × 40000` | the TO amplitudes `dQ ./ src_Q` actually applied |
| `fields` | `Vector` of 401 entries with `.u :: (64,64,64,3)`, `.n`, `.t` | every `savefreq = 100` LF steps ⇒ 0.25 TU |

The `q` / `q_star` offset is not incidental: `qoisaver`'s initializer does `state[] = state[]`,
which fires every already-registered processor on the initial state, so `q` gets one extra leading
column. `fieldsaver` is registered *before* `qoisaver` in `LFsims.jl` precisely so it also sees
that poke and stores the `t = 0` field. Every downstream index convention rests on this.

This one record carries both the fit window (`train_range = (400, 4000)`, i.e. 1.00–10.00 TU) and
D6's IC pool (the 401 fields).

### 1.3 Online ensembles — `6_…` through `9_…`

| driver | path | `data_online` fields |
|---|---|---|
| `6_online_TO_LRS.jl` | `output/TO_LRS/LinReg<i>/data_online_tsim100.0_replica<r>.jld2` | `(; dQ, tau, q, fields)` |
| `7_online_DDN.jl` | `output/TO_DDN/DDN_data_online_tsim100.0_replica<i>.jld2` | `(; dQ, tau, q, fields)` |
| `8_smag_online.jl` | `output/smag/data_smag_0.07_tsim100.0.jld2` | `(; q, fields)` |
| `9_no_sgs.jl` | `output/no_model/data_no_sgs_tsim100.0.jld2` | `(; q, fields)` |

All four write two JLD2 keys, `data_online` and `params`. Shapes match §1.2 (`q` is
`6 × 40001`, `dQ`/`tau` `6 × 40000`), except that `savefreq = 1000` here, so `fields` holds 41
entries. The two baselines run through `solve_unsteady` directly with a `:CREATE_REF` `TO_Setup`,
which is why they carry no `dQ` or `tau`: there is no TO correction to record.

Five replicas per TO closure. The only per-replica state is the model RNG —
`Xoshiro(seeds.to + i + 2)` for the LRS, `Xoshiro(seeds.to + i)` for the DDN, with
`seeds.to = 234`. Because `online_sgs` rebuilds its OU force cache on every call and
`solve_unsteady` deep-copies `ustart`, an array task is **bit-identical** to the corresponding
iteration of the serial loop, and the two may be mixed in one ensemble.

### 1.4 Fitted TO-LRS model — `5_train_LinReg.jl`

`output/TO_LRS/LinReg<i>/LinReg.jld2`:

| key | contents |
|---|---|
| `c` | `N_Q × m` coefficient matrix (saved transposed from the solve) |
| `stoch_distr` | the fitted `MvNormal`/`DiagNormal` residual model, or `nothing` for `:no_noise` |
| `scaling` | `(; in_scaling, out_scaling)`, both the same legacy `(; mu, sigma)` NamedTuple |
| `hist_var`, `hist_len`, `include_predictor`, `fitted_qois` | the history spec, as separate keys |

`output/TO_LRS/LinReg<i>/parameters.jld2` holds one key, `parameters`, a NamedTuple recording
`name, hist_len, hist_var, n_replicas, normalization, include_predictor, lambda, ridge_solver,
penalize_intercept, train_range, track_file`. `lambda`, `ridge_solver` and `penalize_intercept`
were added so two fits at different penalties can be told apart on disk.

`output/TO_LRS/inputs_example.jld2` holds one key, `inputs`: the vector of configuration
NamedTuples written by `4_setup_search.jl`. `LinReg<i>` is entry `i` of that vector.

At the deployed configuration (`hist_len = 5`, `hist_var = :q_star_q`, `include_predictor = true`)
the design has `m = N_Q(2h + 1) + 1 = 67` columns and the `(400, 4000)` window gives 3595 usable
rows.

### 1.5 D6 initial-condition packages — `analysis/build_d6_ics.jl`

The geometry is fixed by named constants at the top of `build_d6_ics.jl`, and each carries a
docstring saying what set it:

| constant | value | what it is |
|---|---|---|
| `FIELD_STRIDE`, `FIELD_DT`, `N_FIELDS` | 100, 0.25, 401 | the tracking record's field grid, asserted rather than assumed |
| `N_REF` | 40000 | reference steps |
| `N_WARM` | 220 | warm-up replayed before the forecast — 1.01× the slowest **level** decorrelation time |
| `N_WARM_DRIVER` | 100 | what the online drivers replay. Deliberately does **not** follow `N_WARM`: the validation IC exists to reproduce a driver run column for column |
| `N_LEAD` | 2172 | forecast length, 10× the slowest **level** decorrelation time |
| `T_INT_MAX` | 0.5430 | that timescale, as the decay constant `T` in `ρ(τ) = exp(-τ/T)` |
| `DDN_TRAIN_RANGE` | `400:4000` | the DDN's training slice |
| `DRIVER_SEED_BASE` | 236 | the archived driver's seed base, `seeds.to + 2` |

`select_ics(; K = 180)` builds the IC set; **`K = 90` is the intended production setting** — all
180 packages are built (cheap, no GPU) and every second one submitted, because at `K = 180` the
achieved spacing is *below* the decorrelation time and adjacent ICs are genuinely correlated.

One file per selected IC, `d6_ic_<k>.jld2`, written to `analysis/output/d6_ics` by default:

| key | contents |
|---|---|
| `u` | the LF velocity field, `(64, 64, 64, 3)`, i.e. `fields[k].u` of the tracking record |
| `k` | the *field index* into `data_track.fields` |
| `n_k` | the tracking-run **step** that field is at |
| `t_k` | that step in time units |
| `ordinal` | the array-task ordinal `1..K` that maps to this `k` |
| `dQ_warm` | the recorded `dQ` slice replayed as the warm-up |
| `q_at_ic` | the record's `q` column at the IC |
| `q_window`, `q_window_offsets` | a window of `q` around the IC, plus its step offsets, for the on-node alignment check |
| `params` | the parameter subset `PARAM_KEYS = (:D, :Re, :lims, :qois, :nles, :Δt, :ou_bodyforce)` |
| `provenance` | source path and size, build time, `nwarm`, `nlead`, `K`, `spacing_tu`, `nref` |

Alongside them:

- **manifest** — keys `k, n, t, ordinal, K, kmin, kmax, spacing_tu, provenance, params`. `run_d6.jl`
  cross-checks it against `select_ics(; K)` on every load, which catches an IC set built from a
  different configuration than the driver assumes.
- **validation package** — same keys plus `validation = true` and `driver_seed_base`, at
  `n_k = 0`, `ordinal = 0`. It is deliberately *excluded* from `select_ics`: it sits inside the fit
  window and is the archived online runs' own initial condition, so it is the oracle, not a sample.
- **`d6_ddn_traindata.jld2`** — keys `dQ_train` (`6 × 3601`, the `DDN_TRAIN_RANGE = 400:4000`
  slice), `train_range`, `params`, `provenance`. The DDN fits its Gaussian inside the constructor,
  so a D6 task needs the training `dQ` itself rather than a fitted model file.

### 1.6 D6 member output — `tools/run_d6.jl`

`output/D6/d6_online_ic<k>_m<member>.jld2`, one per member. Keys:

`q`, `dQ`, `tau` (the trajectories), then `k, n_k, t_k, ordinal, member, seed, ou_advance,
validation, nwarm, nlead, M, model, closure, hist_len, hist_var, tsim, Δt, wall_seconds, julia,
device, written`.

🔴 **There is deliberately no `fields` key.** `savefreq` is set to `nt + 1`, which still leaves one
`t = 0` field in memory (see §1.2 on the initializer poke), and the driver asserts that at most
that one exists — but what actually keeps the disk cost at 160 MB rather than 80 GB over the full
run is that `jldsave` never writes them.

Three things in this driver are load-bearing:

- `ou_advance = n_k` replays the OU forcing chain to the step the IC was taken from. Without it
  every member's forcing is out of phase with its own initial condition, which inflates skill
  without inflating spread and biases the spread–skill ratio downward.
- The IC and the forcing are **shared** across the `M` members; only the model seed varies. That is
  what makes the rank histogram measure the surrogate's own dispersion and nothing else.
- Member seeds are `member_seed(k, member) = hash((:d6, k, member))`, and the value is *written
  into the output file* because `hash` is not promised stable across Julia versions. The validation
  ordinal instead reuses the archived driver's stream, `Xoshiro(DRIVER_SEED_BASE + member)` with
  `DRIVER_SEED_BASE = 236`, so its trajectory is comparable column by column rather than merely in
  distribution.

### 1.7 Cached QoI extracts — `analysis/extract_qois.jl`

`analysis/data/<source basename>_qois.jld2`, with keys `q`, `q_star`, and where present `dQ` and
`tau`, plus `key` (which of `data_track`/`data_train` it came from), `source`, `source_bytes` and
`extracted`. `load_qois(file)` extracts on demand and is what drivers should call — opening a
tracking file directly materialises the ~1.3 GB of velocity fields alongside the ~1 MB of QoIs,
because JLD2 stores the record as one compound dataset.

---

## 2. Closure types (`src/time_series_methods.jl`)

All five implement `get_next_item_timeseries`, dispatched on the type; the two that use the
current predictor take it as a second argument. `to_sgs_term`'s `:ONLINE` branch decides which
call to make by type.

| type | signature | state | what it returns |
|---|---|---|---|
| `Reference_reader` | `(m)` | a column index | the next column of a stored reference trajectory. Used in `:TRACK_REF` |
| `MVG_sampler` | `(m)` | RNG, optional warm-up counter | an i.i.d. draw from one `MvNormal` fitted to `dQ`. The **DDN** |
| `Resampler` | `(m)` | RNG | a uniformly drawn column of a stored `dQ` set |
| `ANN` | `(m, q_star)` | history buffer, counter | a Lux network's prediction. See the note below |
| `LinReg` | `(m, q_star)` | history buffer, counter, RNG | linear regression on the lagged history plus a Gaussian residual. **M0 / TO-LRS** |

**How they differ, in one line each.**

- `Reference_reader` is not a model: it replays the truth, and is how the tracking run is driven.
- `MVG_sampler` is **state-independent by construction — that is the point of it as a control.**
  It never sees `q_star`, so it cannot respond to the flow at all.
- `Resampler` is the empirical counterpart of `MVG_sampler`: same i.i.d. structure, empirical
  rather than Gaussian marginal.
- `LinReg` is the only deployed *conditional* model. Its `target` field is `:q`, so the network
  predicts the next **level** and the returned correction is `dQ = pred - q_star`.

**Warm-up.** Both `LinReg` and `MVG_sampler` accept `spinnup_data` and replay it column by column
before predicting or sampling. For `LinReg` this does two jobs — fill `q_hist` so the first
prediction has a valid history, and drive the solver along the recorded trajectory. The DDN has no
history, so only the second job exists, which is why its warm-up is called "pseudo": without it the
two closures would enter a matched-lead forecast from *different physical states*. In both, the
replay returns **before** the `rand` call, so a member seed means the same thing with and without a
warm-up and across warm-up lengths; and the replayed columns are returned unconverted, which is
what lets D6's validation gate be `dQ` bit-identity over the replayed window.

**The turbulence gate.** `const TURBULENCE_GATE = 1e-2`. In the `LinReg` (and `ANN`) path,
`any(abs.(q_star) .< TURBULENCE_GATE) && (dQ .= 0)`. The docstring states the intent plainly: it is
a **laminar-start gate, not a numerical guard**, introduced for the Taylor-Green vortex, which
begins laminar. The whole-vector zeroing and the `any` are therefore correct by intent — the
question is "is the flow turbulent yet?", answered over all bands at once. Consequences worth
knowing:

- On a statistically stationary turbulent testbed the correct firing rate is **zero**, so a nonzero
  rate is an alarm about the run rather than a property of the model.
- The value is testbed-specific and was chosen for Taylor-Green.
- The gate lives **only** in the `LinReg`/`ANN` path. `MVG_sampler` never receives `q_star`, so the
  DDN has no laminar-start gate at all, and the two closures are not treated alike.

⚠️ `ANN` is currently not constructible: `src/ANN.jl` defines `load_ANN` but is **not** `include`d
by `src/RikFlow.jl`, and the Lux dependencies are commented out at the top of the module. Treat the
`ANN` branch as dormant.

---

## 3. The `ts_*` layer

Seven files in `src/`, `include`d by `RikFlow.jl` but written to be usable on their own:

| file | what it holds |
|---|---|
| `ts_scaling.jl` | `_normalise`, `scale_input`, `scale_output`, and the `Scaling` struct that records the *convention* alongside `mu`/`sigma` |
| `ts_history.jl` | `HistorySpec`, `build_history` (batch), `HistoryBuffer` + `inputvec` (online), and the column-index helpers |
| `ts_models.jl` | `JointModel`, the AR parameterisation (`pacf_to_ar`, `ar_roots`, `decorrelation_time`) |
| `ts_fit.jl` | `fit_ridge` and `fit_joint` — block coordinate descent on one Gaussian likelihood |
| `ts_score.jl` | every offline metric: KS, Δρ, spread–skill, rank histograms, CRPS, NLL, clamp census |
| `ts_rollout.jl` | free-running (uncoupled) rollout of a fitted model, plus `closed_loop_matrix` / `spectral_radius` |
| `ts_spectrum.jl` | fit-time mechanism diagnostics: Gram spectrum, coefficient blocks, companion, starred gain |

### It is deliberately stdlib-only

That constraint is stated in `test/Project.toml` and `ts_scaling.jl`, and it is what lets the test
suite and the analysis drivers `include` these files **by path** without loading
IncompressibleNavierStokes, CUDA, Makie or Lux. `analysis/score_m0_ddn.jl` includes all seven in a
loop; `check_rollout.jl`, `fit_offline.jl`, `g0a_learning_curve.jl` and `plot_models.jl` include
the five they need; `plot_rebaseline.jl` and `score_d6.jl` include only `ts_score.jl`; and
`test/setup.jl` includes all seven.

One visible cost: there is no `erf` in the standard library, so `norm_cdf` is implemented in
`ts_score.jl` (`erf_series` below `|x| <= 2`, `erfc_cf` above) and checked against a BigFloat
series in the tests rather than against remembered constants.

### Conventions this layer fixes

- **History layout.** With `hist_var = :q_star_q` and `include_predictor = true`, one row is
  `[ q^{n*} | q^{n-1}, q^{n-1*} | … | q^{n-h}, q^{n-h*} | 1 ]`, of length `N_Q(2h+1) + 1`. **The
  bias column is always last.**
- **Normalization modes.** `:normal` centres and scales; `:standardise` sets `mu = 0` and only
  divides, leaving the mean in the data; `:minmax` uses midpoint and half-range; `:Id` is the
  identity. At `λ = 0` the first two give the same fit (a shift-invariance of least squares with an
  intercept), but they move the Gram spectrum, which is the diagnostic that decides what the
  penalty is doing. At `λ > 0` the penalty is not shift-invariant and they give *different fits*.
- **`Scaling` travels with the fit.** Before it, the convention was a call-site constant that was
  read and discarded, so a numerical `λ` did not mean the same thing on two testbeds and no saved
  model recorded which convention produced it. `assert_convention` is meant to be called by
  anything that reads a fitted model. Note that the deployed `LinReg.jld2` still stores the legacy
  `(; mu, sigma)` pair; `as_scaling` exists to adopt one, taking `normalization` from the matching
  `parameters.jld2`.
- **`JointModel` nests the ladder by zeroing blocks**: `ar_order = 0` gives a white residual, `W`
  zero outside its bias row gives a constant `Σ`, and `C` zero outside its bias row recovers the
  data-driven noise model. `Σ` is the covariance of the **innovation**, never of the marginal.
- **`fit_ridge` solves the augmented system** `[X; √λ P] \ [Y; 0]` rather than the normal
  equations, because `cond(X) ≈ 1.9e6` on this design and `X'X` would square that.
  `5_train_LinReg.jl` writes the identical construction inline, since it does not load this layer.

---

## 4. Metrics and regimes

`ts_score.jl` states the regime in every docstring, on the grounds that **the regime matters more
than the formula**. The four, quoting the file:

| regime | definition |
|---|---|
| **0** fit-time | "from the parameters and the design matrix, no trajectory" |
| **A** one-step | "teacher-forced on the true history; errors never accumulate" |
| **B** offline unrolled, `q*` replayed | "ARTIFICIAL: the feedback loop through the solver is cut" |
| **C** online | "the LF solver is in the loop; the only regime that measures deployment" |

`results.md` records which variable each regime is scored on: regime A is *identical* on `q` and
`dQ` (the two predictive laws differ by a shift common to forecast and truth); regime B is `dQ`
only, because the replayed `q*` pins the level — and **nothing in the report is regime B**; regime
C is `q`, with `dQ` secondary, because the level is what the physical claims are about; regime 0 is
in scaled q-space.

### What is implemented, by regime

| # | function | regime | note |
|---|---|---|---|
| 1 | `nll_gaussian(Y, mu, Sigma)` | A | per-sample NLL of a multivariate Gaussian predictive density. Unbounded, so one tail value can reverse a ranking — which is why #2 and #4 exist alongside it |
| 2 | `crps_gaussian`, `crps_ensemble`, `crps_ensemble_mean` | A (and shared by every ensemble tier) | closed form for a Gaussian margin; `fair = true` applies the finite-ensemble correction |
| 4 | `ranks`, `rank_histogram`, `jolliffe_primo`, `n_eff` | A | ties are broken **at random with a seeded RNG**, because the turbulence gate produces exact duplicates by construction |
| 10 | `summed_ks(traj, ref)` | C | per-replica KS summed over the six QoIs. "Summed" means over QoIs and **never** over replicas |
| 11 | `ensemble_ks(trajs, ref)` | C | replicas pooled into one empirical CDF *first*. A different object from #10; the two are never averaged together |
| 10/11 | `ks_noise_floor(ref)` | C | reference-vs-reference floor from two disjoint contiguous halves. Contiguous, not resampled, because the record is serially correlated |
| 12–15 | `delta_rho(traj, ref; lag, maxlag, dt)` | C | `Δρ(τ) = Σ_i abs(ρ_model − ρ_ref)`. `lag = 1` is #12, the integral-timescale lag #13, twice it #14, and the whole-curve integral #15 |
| 16 | `stability_fraction(trajs)` | C | fraction of replicas completing without a NaN, plus the first non-finite column of each |
| 17/18 | `spread_skill(ens, truth; correct)` | C | `ens` is `K × N_Q × M`. The finite-`M` factor `√((M+1)/M)` is applied by default and must stay on |
| 17 | `spread_skill_by_lead(fc, truth; grid, leads)` | C, D6 only | `fc` is `K × N_Q × M × L`; per-QoI leads, because `sd(dQ)` spans four orders of magnitude across the bands |
| RH-3 | `rank_histogram_by_lead(fc, truth; …)` | C, D6 only | |
| 23 | `gram_diagnostics(X; lambda)` | 0 | one SVD; classifies `λ` as `:inactive`, `:rank_deficiency_fix` or `:shrinks_determined_directions` |
| 24 | `coefficient_blocks`, `total_block_sum` | 0 | |
| 21/22 | `companion`/`rho`, `starred_gain` | 0 | spectral radius of the lifted companion, and the starred-block gain |
| 25 | `pinv_gap(X, Y)` | 0 | |
| 26 | `clamp_census(q_star; threshold = 1e-2)` | 0, B, C | how often the gate *would* fire on a record |

Supporting pieces: `autocorr`, `ljung_box`, `residual_battery`, `correlation_time`, `ks_distance`,
`block_bootstrap_indices`, `lead_grid`, `lead_positions`, `climatological_skill`,
`saturation_lead`, `norm_cdf`, `norm_pdf`.

### Warnings the code attaches to these, worth repeating

- **KS is a guard, never a selector.** It sees the histogram of values, not their order. On HIT,
  summed KS ranked the memoryless DDN best of nine while its correction had a lag-1
  autocorrelation of 0.0006 against a reference 0.9434.
- **`spread_skill` needs `K ≫ 1`.** Both spread and skill are expectations over initial
  conditions. Every archived online run is a single trajectory from one IC, which is exactly why
  D6 exists; the form that *is* computable from a single-IC ensemble is the **climatological** one
  (#18), and it must be reported labelled as climatological, never as a lead.
- **Apply the finite-`M` correction.** Uncorrected, a perfect ensemble reads `√(M/(M+1))` — 0.953
  at `M = 10` and 0.913 at `M = 5`, the archive's ensemble size.
- **One lead grid cannot serve six QoIs.** `T_int` spans a factor ~37 across the QoIs, so
  `lead_grid` builds one grid per QoI in physical time, and a lead the run cannot support raises
  rather than being clipped — a clipped lead reads as a saturated one, and saturation is the thing
  being measured.
- **`clamp_census` measures the record, not the model.** The rate is computable for any run,
  including one produced by a closure whose code path has no clamp. Firing may only be attributed
  to the `LinReg` lineage.
- **`autocorr`'s degeneracy guard is relative.** Testing `den == 0` almost never fires, and a
  constant series then returns `[1.0, 0.999]` — indistinguishable from a strongly autocorrelated
  one, which is precisely the wrong answer for a state-independent model. `CONSTANT_RTOL = 1e-12`,
  applied relative to the series' own variation.

---

## 5. Environment variables

Collected by grepping `get(ENV,` / `ENV[` across the package. Anything not listed has no
environment override.

### Data locations and dataset selection

| variable | read by | default / effect |
|---|---|---|
| `RIKFLOW_HF_REF` | `3_track_ref.jl` | path to the HF reference; defaults to the regenerated `…_f64_lmwray3.jld2` under `exp_square_HIT/output` |
| `RIKFLOW_TRACK_FILE` | `5_…`, `6_…`, `7_…`, `8_…`, `9_…`, `analysis/build_d6_ics.jl` | path to the tracking record; same default in all six |
| `RIKFLOW_VALIDATION_TRACK` | `analysis/build_d6_ics.jl` | record the validation IC is cut from; defaults to `RIKFLOW_TRACK_FILE`'s value |
| `RIKFLOW_ARCHIVE` | `analysis/extract_archive.jl`, `analysis/score_m0_ddn.jl`, `test/test_g1.jl` | paper 2's frozen archive root (outside the repository) |
| `RIKFLOW_DEV_ARCHIVE` | `analysis/extract_archive.jl`, `test/test_g1.jl` | the working archive root |
| `RIKFLOW_HF_NEW` | `analysis/extract_archive.jl` | path to the regenerated HF reference to cache |
| `RIKFLOW_REBASE_ROOT` | `analysis/extract_rebaseline.jl` | root of the R2 rebaselined outputs |
| `RIKFLOW_DATASET` | `analysis/score_m0_ddn.jl`, `analysis/plot_paper4.jl` | `archive` (default) or `new`; selects which dataset is scored and which output file is written |
| `RIKFLOW_NEW_REFERENCE` | `tools/check_ref_401.jl` | path to a regenerated reference to check; empty by default |
| `HF_REF_QOIS` | `analysis/plot_hf_new_vs_archive.jl` | override for the archive QoI source |

### Gates and run control

| variable | read by | default |
|---|---|---|
| `RIKFLOW_TRACK_GATE` | `3_track_ref.jl` | `1e-1`. **Deliberately loose** — a blow-up detector, not a tracking-quality check. The tight bound is to be set from the printed table, once, and recorded in `results.md`; do not tighten it by guessing |
| `HF_REF_SMOKE` | `2_HF_ref.jl` | `0`; `1` runs the 128³ / 0.5 TU variant (which then wants a 128³ spin-up the archive does not provide) |
| `SLURM_JOB_ID` | `1_spinnup.jl` | required (not a `get`) — the script indexes `ENV` directly for the log filename |

### D6

| variable | read by | default |
|---|---|---|
| `D6_IC_DIR` | `tools/run_d6.jl` | falls back to `exp_square_HIT/output/d6_ics`, then `analysis/output/d6_ics` |
| `D6_MODEL` | `tools/run_d6.jl` | `output/TO_LRS/LinReg1/LinReg.jld2` |
| `D6_CLOSURE` | `tools/run_d6.jl` | `lrs`; the other value is `ddn` |
| `D6_DDN_DATA` | `tools/run_d6.jl` | `<ic_dir>/d6_ddn_traindata.jld2` |
| `D6_OUT` | `tools/run_d6.jl`, `analysis/score_d6.jl` | `exp_square_HIT/output/D6` |
| `D6_MEMBERS` | `tools/run_d6.jl` | `10` |
| `D6_TRUTH` | `analysis/score_d6.jl` | `hf_reference` |
| `D6_SMOKE_LEAD` | `tools/smoke_d6.jl` | `300` |

### Probes

| variable | read by | default |
|---|---|---|
| `CFL_DEVICE`, `CFL_N_DNS`, `CFL_N_LES`, `CFL_TSIM`, `CFL_SAFETY`, `CFL_N_ADAPT`, `CFL_SYNTHETIC` | `cfl_probe.jl` | `gpu`, `512`, `64`, `0.05`, `0.9`, `1`, `0` |
| `HFT_DEVICE`, `HFT_N_DNS`, `HFT_N_LES`, `HFT_SAVEFREQ`, `HFT_PLOTFREQ`, `HFT_TARGET_TSIM`, `HFT_WALL_HOURS`, `HFT_N_CHECKPOINTS`, `HFT_WARMUP`, `HFT_BUDGET_S`, `HFT_MEASURE`, `HFT_MEASURE_MIN`, `HFT_MEASURE_MAX`, `HFT_SYNTHETIC`, `HFT_SAVE_FIELDS`, `HFT_KEEP_CHECKPOINT` | `hf_timing_probe.jl` | `gpu`, `512`, `64`, `10`, `1000`, `100`, `120`, `1`, `200`, `900`, `0`, `1000`, `20000`, `0`, `1`, `0` |
| `small_case.jl` reads arbitrary integer/float keys through its own `envint` / `envflt` helpers | | |

### Figures

| variable | read by | default |
|---|---|---|
| `HF_FIG_DIR`, `HF_SMOOTH_TU`, `HF_BLOCK_TU`, `HF_NBOOT`, `HF_NPERM`, `HF_ZOOM_COLS`, `HF_SEED` | `analysis/plot_hf_new_vs_archive.jl` | `analysis/figures`, `2`, `2`, `4000`, `2000`, `1000`, `20260914` |
| `REBASE_FIG_DIR`, `REBASE_STRIDE` | `analysis/plot_rebaseline.jl` | `analysis/figures`, `4` |

Batch scripts additionally set `JULIA_DEPOT_PATH` and `JULIA_CPU_TARGET`; the CPU target
multiversions the precompiled code across Zen2, Zen4 and Icelake-server so one depot serves both
the a100 and h100 partitions. That is also why each script runs `Pkg.instantiate(); Pkg.precompile()`
first: multiversioning changes the *content* of the precompiled images, and Julia errors on a
mismatched cache rather than rebuilding it silently.
