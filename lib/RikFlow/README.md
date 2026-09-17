# RikFlow

Tau-Orthogonal (TO) sub-grid-scale closure for turbulence, as a sub-package of a fork of
IncompressibleNavierStokes.jl. This directory is `lib/RikFlow`; it depends on the parent package
by path (`Project.toml`, `[sources]`).

Start here if you are returning to the code after a break. For data layouts, the closure types,
the metric definitions and the environment variables, see [`docs.md`](docs.md). For the current
measurements, see [`analysis/results.md`](analysis/results.md) — where a number in that file
disagrees with a design document, "this file is the measurement and they are the prediction".

---

## 1. What problem this solves

A high-fidelity (HF) direct numerical simulation of forced homogeneous isotropic turbulence is run
at 512³ and filtered onto a 64³ LES grid. A low-fidelity (LF) LES on that grid cannot reproduce the
HF statistics on its own. The TO method corrects it so that a *small set of Quantities of Interest*
tracks the HF reference, rather than trying to reproduce the whole sub-grid stress.

The six QoIs are spectral band integrals of enstrophy `Z` and energy `E` over three wavenumber
bands:

```julia
qois = [["Z",0,6], ["E",0,6], ["Z",7,15], ["E",7,15], ["Z",16,32], ["E",16,32]]
```

They are referred to throughout as `Z[0,6]`, `E[0,6]`, `Z[7,15]`, `E[7,15]`, `Z[16,32]`,
`E[16,32]`, in that order, and `N_Q = 6` everywhere.

At each LF step the solver computes the QoIs the uncorrected step would produce, `q*`; a *closure*
supplies a correction `dQ`; and `RikFlow.to_sgs_term` builds the sub-grid forcing that realises
exactly that correction. Mechanically (`src/RikFlow.jl`, `to_sgs_term`): per-QoI direction fields
`V_i` are evaluated in Fourier space, their inner products give coefficients `c_ij`, the
amplitudes are `tau = dQ ./ src_Q`, and the SGS term is `-Σ_i tau_i Σ_j c_ij T_j`, inverse-FFT'd
back to physical space.

`TO_Setup` has three modes, and the whole pipeline is built out of them:

| `to_mode` | what happens | used by |
|---|---|---|
| `:CREATE_REF` | QoIs are computed and stored; no SGS feedback | `2_HF_ref.jl`, `8_smag_online.jl`, `9_no_sgs.jl` |
| `:TRACK_REF` | `dQ = q_ref - q*`, read from a reference trajectory | `3_track_ref.jl` |
| `:ONLINE` | `dQ` comes from a fitted/sampled time-series model | `6_online_TO_LRS.jl`, `7_online_DDN.jl`, `tools/run_d6.jl` |

The tracking run (`:TRACK_REF`) is the training-data generator: the `dQ` it had to take in order to
stay on the reference is the target every closure is fitted to.

The closures actually deployed here are:

- **M0 / TO-LRS** — a linear regression on a lagged history of `q` and `q*`, plus a multivariate
  Gaussian residual. Paper 2's model. Fitted by `5_train_LinReg.jl`, deployed by
  `RikFlow.LinReg`.
- **DDN** — paper 1's data-driven noise model: one multivariate Gaussian fitted to `dQ` and
  sampled i.i.d. every step, state-independent by construction. `RikFlow.MVG_sampler`.
- **M4 / stochastic LSTM** — an LSTM with a latent path drawn upstream of the recurrence
  (STORN / VRNN, after Barthel Sørensen et al.), with a Gaussian emission head. Fitted by
  `11_train_StochLSTM.jl` under the Lux extension, deployed by `RikFlow.StochLSTM`.
  ⚠️ **Off-grid and exploratory** (`plan.md` §3): it enters no attribution difference, is never
  the confirmatory cell, and is first on the cut list. 🔴 The source applies this architecture as a
  **post-run corrector, outside the solver**; `analysis/postrun_lstm.jl` is that setting and
  `12_online_StochLSTM.jl` is the in-solver one, and **M4 and M0 must be compared in the same
  mode**.

Baselines: Smagorinsky (`8_smag_online.jl`) and no model at all (`9_no_sgs.jl`).

---

## 2. Directory map

| path | what it is |
|---|---|
| `src/` | the package. Solver bridge, TO machinery, closure types, and the stdlib-only `ts_*` layer |
| `src/HIT_setups/` | the three HIT drivers as library functions: `create_ref_data`, `spinnup`, `track_ref`, `online_sgs`, plus disk preflight (`storage.jl`) |
| `ext/` | `RikFlowMakieExt.jl` (plotting) and `RikFlowLuxExt.jl` (M4's training), both weak-dependency extensions |
| `training/` | the environment and test suite for M4's fit — the only place Lux is loaded |
| `attic/` | code kept for reference and **not loaded**: currently the retired `ANN.jl` |
| `exp_square_HIT/` | the HIT experiment: numbered driver scripts `1_…` … `12_…`, probes, and outputs |
| `exp_square_HIT/batch_scripts/` | SLURM submission scripts (Snellius), one per stage |
| `exp_square_HIT/tools/` | D6 driver and pre-flight, acceptance tests, the merge differential test, and `RUNBOOK.md` |
| `analysis/` | the measurement layer: extraction, offline fits, scoring, plots, and `results.md` |
| `test/` | the verification suite for the `ts_*` layer; runs without CUDA or INS |
| `channel/`, `taylor-green/`, `time_solvers/` | the other testbeds and a solver-timing study. Not part of the HIT pipeline; only the channel's tracked QoI cache has been scored so far |

Generated data is gitignored: `analysis/figures`, `analysis/data`, and the per-experiment
`output/` trees.

Four separate Julia environments, and the separation is deliberate:

```bash
julia --project                  # lib/RikFlow — drivers, needs CUDA + IncompressibleNavierStokes
julia --project=analysis         # analysis drivers — plotting, no CUDA/INS (ts_* is included by path)
julia --project=test             # the test suite — stdlib + Distributions/PDMats only
julia --project=training         # M4's fit — the ONLY environment with Lux/Optimisers/Zygote
```

🔑 **M4's split is the same principle one level down.** Training needs Lux and lives in
`ext/RikFlowLuxExt.jl`; what runs *inside the solver* is `lstm_step!` in `src/ts_lstm.jl`, on plain
arrays, with no Lux anywhere. That is what lets the stdlib-only suite test the deployed path, keeps
the ~900 M0/M1/M2 jobs from paying Lux load time, and makes the per-step cost controllable — see
`tools/m4_cost_probe.jl`.

---

## 3. The pipeline, in order

Every stage below is 64³ LF / 512³ HF, `Re = 2000`, Float64, `LMWray3`, unless stated. Run from
`exp_square_HIT/` (the batch scripts also accept being submitted from `lib/RikFlow`).

| # | driver | batch script | produces |
|---|---|---|---|
| 0 | `1_spinnup.jl` | `run_spinnup.sh` | `output_spinnup/u_start_spinnup_<n>_Re<Re>_freeze_<f>_tsim<t>.jld2` |
| 1 | `2_HF_ref.jl` | `run_HF_ref.sh` | `output/data_train_dns512_les64_Re2000.0_freeze_10_tsim100.0_f64_lmwray3.jld2` |
| 2 | `3_track_ref.jl` | `run_track_ref.sh` | `output/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2` |
| 3 | `4_setup_search.jl` | *(none — instant, no GPU)* | `output/TO_LRS/inputs_example.jld2` |
| 4 | `5_train_LinReg.jl <i>` | `run_train_lrs.sh <i>` | `output/TO_LRS/LinReg<i>/LinReg.jld2` and `parameters.jld2` |
| 5 | `6_online_TO_LRS.jl <i> [replica]` | `run_online.sh lrs <i>`, or `run_online_array.sh <i>` | `output/TO_LRS/LinReg<i>/data_online_tsim100.0_replica<r>.jld2` |
| 6 | `7_online_DDN.jl` | `run_online.sh ddn` | `output/TO_DDN/DDN_data_online_tsim100.0_replica<i>.jld2` |
| 7 | `8_smag_online.jl` | `run_online.sh smag` | `output/smag/data_smag_0.07_tsim100.0.jld2` |
| 8 | `9_no_sgs.jl` | `run_online.sh nomodel` | `output/no_model/data_no_sgs_tsim100.0.jld2` |
| 9 | `10_setup_lstm.jl` | *(none — instant, no GPU)* | `output/TO_LSTM/inputs_lstm.jld2` |
| 10 | `11_train_StochLSTM.jl <i> [seed]` | `run_train_lstm.sh <i> [seed]` | `output/TO_LSTM/StochLSTM<i>/StochLSTM_seed<s>.jld2`, `seed_summary.jld2` |
| 11 | `12_online_StochLSTM.jl <i> [replica]` | `run_online.sh lstm <i>` | `output/TO_LSTM/StochLSTM<i>/data_online_tsim100.0_replica<r>.jld2` |

`submit_lrs.sh <i> [array_spec]` chains stages 4 and 5 on the login node with
`--dependency=afterok`, so the ensemble array never starts on a failed fit. Use it rather than
submitting the two by hand.

### Stage notes

**0 — spin-up.** A plain forced DNS, no TO and no QoIs, producing one turbulent initial velocity
field. It is *reused from the archive* rather than re-run: it is only an initial field, it seeds a
different OU stream (`ou_spin = 123`) from the reference (`ou = 333`), and re-running it would
leave the new reference sharing nothing with the archive. Note that `1_spinnup.jl` writes to
`output_spinnup/` while everything downstream reads from `output/`, so a freshly produced spin-up
has to be moved.

**1 — HF reference.** 512³, 400 000 steps at `Δt = 2.5e-4`, `tsim = 100`. QoIs every
`savefreq = 10` DNS steps; filtered 64³ fields every `plotfreq = 1000`, giving 401 fields at
0.25 TU. A disk preflight (`ref_data_storage` / `check_output_space`, `hard = true`) refuses to
start unless the space is free, because `create_ref_data` holds everything in host memory and
writes only at the end. `HF_REF_SMOKE=1` runs a 128³ / 0.5 TU variant.

**2 — tracking run.** LF at `Δt = 2.5e-3`, `freeze = 1`, `tsim = 100`. Both the training data
(`dQ`, `q`, `q_star`, `tau`) and D6's initial-condition pool (401 LF fields at
`savefreq = 100` ⇒ 0.25 TU) come out of this one run. It ends by printing a per-QoI tracking-error
table and applying a deliberately loose blow-up gate (`RIKFLOW_TRACK_GATE`, default `1e-1`) —
the record is written *before* the gate is checked.

**3 — the configuration table.** `4_setup_search.jl` writes a vector of NamedTuples, one per
configuration. **The index is the identity**: `LinReg<i>` is the *i*-th entry of `CELLS`, and that
name is the output directory, the batch-script argument and how every results table refers to a
configuration. The list is append-only; inserting a cell renames every configuration after it.

**4 — fit.** A least-squares solve on the standardised lagged history, plus an MvNormal fitted to
the residual. `ridge_solver = :exact` solves the augmented ridge system in closed form;
`:admm` (via `RegularizedLeastSquares`) is kept only for reproducing paper 2's archived `λ > 0`
models and for the `:nuclear` regularizer.

**5–8 — online runs.** All four read the tracking record for the initial field, the parameters and
the OU forcing. `6_…` and `7_…` inherit `ou_bodyforce` from it and assert `freeze == 1`; `8_…` and
`9_…` rebuild it from local constants. Five replicas each for the two TO closures, differing only
in the model seed.

### Probes and one-off tools

| script | batch script | what it answers |
|---|---|---|
| `cfl_probe.jl` | `run_cfl_probe.sh` | what `Δt` an adaptive step picks at 512³ Float64/LMWray3, and whether it is stable |
| `hf_timing_probe.jl` | `run_hf_timing_probe.sh` | cost of the HF run before launching it: per-step time, projection, disk |
| `plot_spinnup_output.jl` | `plot_spinnup.sh` | plots the spin-up energy history |
| `tools/check_qoi_rate.jl` | — | verifies the HF reference computes *and* stores QoIs only every `savefreq`-th step |
| `tools/check_ref_401.jl` | — | full-scale acceptance test: reproduces the archive's 401 filtered fields |
| `tools/small_case.jl` | — | shrunk bit-for-bit differential test of the pipeline across the upstream merge (`generate` / `check`) |
| `tools/smoke_pass2.jl` | — | smoke test for the channel / Taylor-Green / `time_solvers` ports |

### D6 — the multi-initial-condition ensemble

Lead-resolved spread–skill and rank histograms need many initial conditions; every run in stages
5–8 is a single trajectory from one IC. D6 is the separate ensemble that supplies them.

1. `julia --project=analysis analysis/build_d6_ics.jl` — cut IC packages out of the tracking
   record, plus a manifest, a validation package and the DDN training slice.
2. `julia --project exp_square_HIT/tools/smoke_d6.jl` — pre-flight, before requesting an
   allocation.
3. `julia --project exp_square_HIT/tools/run_d6.jl 0` — **the validation task, run it first.**
   Ordinal 0 is the archived runs' own initial condition, so its trajectory can be compared
   column-by-column against an existing one.
4. `sbatch exp_square_HIT/batch_scripts/run_d6.sh` — one array task per IC, `M` members each,
   writing `output/D6/d6_online_ic<k>_m<member>.jld2`.
5. `julia --project=analysis analysis/score_d6.jl` — scoring.

`exp_square_HIT/tools/RUNBOOK.md` is the operational procedure for this (what to copy to the
cluster, in what order, what to check). Read it before running D6 on Snellius.

---

## 4. The analysis layer

Run the drivers with `--project=analysis`. The `ts_*` source files are `include`d by path, so
nothing here loads IncompressibleNavierStokes, CUDA or Lux — with one deliberate exception,
`ou_replay.jl`, noted below.

The two extraction steps exist because a tracking or online file is dominated by velocity fields
that none of the time-series work reads: JLD2 stores the record as one compound dataset, so `q`
cannot be read without materialising `fields` alongside it. `extract_qois.jl` pays that read once
and caches the QoI arrays under `analysis/data/…_qois.jld2`.

Order, for the primary (rebaselined) dataset:

```bash
cd lib/RikFlow
julia --startup-file=no --project=analysis analysis/extract_rebaseline.jl   # cache the R2 ensembles
RIKFLOW_DATASET=new julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl
julia --startup-file=no --project=analysis analysis/plot_rebaseline.jl     # figures/fig8_*.png
RIKFLOW_DATASET=new julia --startup-file=no --project=analysis analysis/plot_paper4.jl
```

and for paper 2's archive (reached through `RIKFLOW_ARCHIVE` / `RIKFLOW_DEV_ARCHIVE`, which live
outside the repository):

```bash
julia --startup-file=no --project=analysis analysis/extract_archive.jl     # cache D3 and D5
julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl        # -> output/paper4_scores.jld2
julia --startup-file=no --project=analysis analysis/plot_paper4.jl         # -> figures/*.png
```

`score_m0_ddn.jl` takes the dataset as a switch (`RIKFLOW_DATASET`, `archive` or `new`) and the two
runs write different files. **The two datasets must never be pooled or compared number for number**
— the rebaselined runs are post-`09954be1`, which changed the Nyquist convention and therefore `∂`,
`tau` and `dQ`; they are a *different dynamical system*, not a better measurement of the same one.

### Which script produces which artefact

**Tier 0 — caching.** All three write into `analysis/data/` and skip work already done. The
downstream drivers `include` them and will extract on demand, so running them by hand is an
optimisation, not a prerequisite.

| script | reads | writes |
|---|---|---|
| `extract_qois.jl <file…>` | a tracking record (`data_track`, else `data_train`) | `analysis/data/<basename>_qois.jld2` |
| `extract_archive.jl [targets]` | paper 2's archive roots + the regenerated HF reference | `hf_reference_tsim100.0_qois.jld2`, `hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2`, `online_<name>_<root>_qois.jld2` |
| `extract_rebaseline.jl [keys]` | the R2 online trees under `exp_square_HIT/output` | `online_new_<key>_qois.jld2` |

**Tier 1 — scoring and figures.**

| script | reads | writes |
|---|---|---|
| `score_m0_ddn.jl` | the tier-0 caches for the selected dataset | `analysis/output/paper4_scores.jld2` or `paper4_scores_new.jld2` |
| `plot_paper4.jl` | that score file | `analysis/figures/fig1…fig7[_new].png` |
| `plot_rebaseline.jl` | `hf_reference_new_…`, `online_new_<key>_…` | `analysis/figures/fig8_online_<key>.png`, plus §4b's three tables **to stdout** |
| `plot_hf_new_vs_archive.jl` | both HF reference caches | `analysis/figures/hf_new_vs_archive_{trajectories,deviation,distributions}.png` |

**Tier 2 — D6.** `ou_replay.jl` is the precondition: it measures, on a tiny CPU solve, how many
`OU_advance!` steps reproduce a solve of *n* steps, which is what licenses `ou_advance = n_k`. It
reads no files and writes none, and it is **the one script here that needs
IncompressibleNavierStokes**, so it runs under `--project=.` rather than `--project=analysis`.

| script | reads | writes |
|---|---|---|
| `build_d6_ics.jl [K\|validation] [--force]` | the raw tracking record (not the cache) | `analysis/output/d6_ics/{d6_ic_<k>,d6_ic_manifest,d6_ic_validation,d6_ddn_traindata}.jld2` |
| *(then `tools/run_d6.jl`, see §3)* | the IC packages + the fitted model | `exp_square_HIT/output/D6/d6_{online,valid}_ic<k>_m<m>.jld2` |
| `score_d6.jl [--preview]` | the D6 outputs, R1's cache, the new reference, `online_new_LinReg1` | `analysis/output/d6_scores.jld2` |

**Parked branch — the offline ladder.** `ladder_setup.jl` is an include-only library (partition,
standardisation, regressor build, rollout hand-off) shared by `fit_offline.jl`, `plot_models.jl`,
`check_rollout.jl` and `g0a_learning_curve.jl`. These take a tracking file as `ARGS[1]`, use
`train_frac = 0.75` rather than `results.md`'s partition, and say so in their own headers —
`check_rollout.jl` describes itself as "largely superseded" and `fit_offline.jl` as "parked rather
than retired". Their numbers are not comparable with the report's.

`analysis/results.md` is the report these drivers produce. Its section headers are the fastest way
in: §1 the records, §2 regime A, §3 regime C, §4 regime 0, §4b the rebaselined trajectories, §5–§7
findings and blockers, §8 test coverage.

---

**M4 — the stochastic LSTM.** Two drivers, two different experiments, and they are not
interchangeable:

| driver | environment | is |
|---|---|---|
| `analysis/postrun_lstm.jl <i>` | `--project=training` | **D-post** — the model as a post-run corrector, teacher-forced on the recorded history. The faithful Sørensen setting: no feedback, no exposure bias, no solver time. Reports ensemble CRPS, the rank histogram, the IWAE bound and the posterior-collapse diagnostic |
| `exp_square_HIT/12_online_StochLSTM.jl <i>` | `--project` | **D-online** — the model as a closure, inside `to_sgs_term`. What this project needs and what the source never did |

🔴 **M4 and M0 must be compared in the same mode.** An M4 scored as a post-processor against an M0
scored in-solver is not a comparison — it is the confound `plan.md` §22 item 9 exists to name.

🔴 **M4 has no closed-form one-step predictive density**, so exact NLL, closed-form CRPS and
`companion`/ρ(C̃) are undefined for it. Held-out likelihood becomes an **IWAE bound**, which is a
*lower* bound on `log p` and therefore **never shares a column with M0's exact NLL**. The
ensemble rank histogram is the metric that reads the same on both, which is exactly why §9 makes it
the ladder-wide calibration axis.

---

## 5. Tests

```bash
julia --startup-file=no --project=test test/runtests.jl              # the ts_* layer, stdlib only
julia --startup-file=no --project=training training/runtests_lux.jl  # M4's training side
```

The second suite is separate because it needs Lux. A failure there means M4 cannot be *trained*; a
failure in the first means M4 cannot be *run*, which is the more serious of the two — and it is the
reason the deployed `lstm_step!` was kept stdlib rather than written against Lux.

Run it directly, **not** through `Pkg.test`. The suite deliberately has no dependency on RikFlow:
the `ts_*.jl` files are stdlib-only and the tests `include` them by path, so the whole suite runs
without loading IncompressibleNavierStokes, CUDA, Makie or Lux. `Pkg.test` would force the package
under test — and its dependency tree — to load.

`test/Project.toml` adds exactly two non-stdlib packages, `Distributions` and `PDMats`, pinned to
the same versions as `Project.toml`, so the suite can *sample* the archived residual model and not
merely read it.

### The `V<number>` convention

Every `@testitem` is titled `"V<n> <what it asserts>"` (a few use `G1` or `SC-48` instead, and
seven carry no label at all). `V<n>` is a **numbered row of the verification matrix** in
`meta_files/plan.md` §14 — "all under `lib/RikFlow/test/` as `@testitem` blocks, ordered by what
it protects". The number, not the filename, is the stable identifier: it is how a test, a plan row
and a status line in `results.md` §8 refer to the same obligation.

Two caveats, both real as of this writing:

- **The registry only covers V0–V28.** V29–V38 are used by the tests but have no row in
  `plan.md` §14. V29/V30 appear in `results.md` §8 as statuses; V31, V34–V37 appear only in prose
  in the meta files; V32, V33 and V38 appear nowhere outside the test files themselves.
  ⚠️ **V39–V44 (M4) continue that unregistered sequence** and need rows adding when the gap is
  closed.
- **V23 and V25 are overloaded.** `plan.md` §14 numbers them for other things (chunked QR in
  `solve_C`; `load_qois` equivalence). The suite uses them for the Gram spectrum and the
  rank/`pinv` check — which are **metric** numbers #23 and #25 from `meta_files/metrics.md` §3.
  The suite follows `metrics.md` for these two and `plan.md` for V16/V17/V18/V26/V27, and nothing
  reconciles the two numbering schemes.

Roughly, what the suite covers:

| item | covers | file |
|---|---|---|
| V0 | the harness runs; the `ts_*` layer loads with no RikFlow in the process | `test_scaling.jl` |
| V1 | `build_history` against all five archived `create_history` copies, and the tracked record | `test_history.jl` |
| V2 | batch history ≡ the online `HistoryBuffer`, step by step | `test_history.jl` |
| V16 | ridge parity, QR vs the normal equations, intercept-penalty semantics | `test_fit.jl` |
| V17 | `erf`/normal CDF, NLL, CRPS, Δρ, all against closed forms or Monte Carlo | `test_score.jl` |
| V18 | coefficient blocks, total block sum, companion/ρ, starred gain, Float64 invariance | `test_spectrum.jl` |
| V23 | Gram-spectrum λ branches | `test_spectrum.jl` |
| V25 | rank deficit vs the `pinv` gap | `test_fit.jl`, `test_spectrum.jl` |
| V26 | level ≡ increment ranks | `test_score.jl` |
| V27 | rank-histogram uniformity, signed dispersion, χ²-on-`N_eff` | `test_score.jl` |
| V28 | D6's lead-resolved scorer, plus the whole driver on synthetic members | `test_d6_score.jl` |
| V29 | D6's IC selection and packaging | `test_d6_ics.jl` |
| V30 | the OU forcing replay | `test_ou.jl` |
| V31, V35–V37 | source-level scans: `@printf` literals, resolvable `using`s, soft-scope traps, `Val` | `test_sources.jl` |
| V32 | D6's validation verdict | `test_d6_score.jl` |
| V33, V34 | OU forcer type-genericity; setup fields stay GPU-kernel-safe | `test_ou.jl` |
| V38 | both samplers replay their warm-up before touching the RNG | `test_sources.jl` |
| V39 | M4's BPTT segmentation: coverage, burn-in mask, never crossing a record discontinuity | `test_lstm.jl` |
| V40 | `lstm_step!` against an independent reference forward, zero allocation per step, and the online closure against a batch pass | `test_lstm.jl`, `test_lstm_online.jl` |
| V41 | the **training** forward equals the **deployed** forward, across all four architectures and both encoder forms, including the covariance reparametrisation | `training/runtests_lux.jl` |
| V42 | M4's warm-up draws no randomness — V38 extended to a closure that cannot satisfy it by returning early | `test_lstm.jl`, `test_lstm_online.jl` |
| V43 | the likelihood arithmetic: Gaussian log-density, both KL forms, IWAE ≥ ELBO, architecture nesting | `test_lstm.jl` |
| V44 | M4's replayed warm-up is bit-identical and unconverted; the turbulence gate; the deterministic mode | `test_lstm_online.jl` |
| G1 | reproduction of paper 2's archived fits (`plan.md` §12) | `test_g1.jl` |

Several items are conditional: the `G1` items skip unless `RIKFLOW_ARCHIVE` /
`RIKFLOW_DEV_ARCHIVE` resolve, and several `V1`/`V18`/`V28`/`V29` items skip unless the matching
QoI cache exists under `analysis/data/`. A skip is reported, not silent.

⚠️ `test/data/` holds two golden files (`small_case_golden.jld2`,
`small_case_golden_float64_lmwray3.jld2`) that **no test in `test/` reads** — their only consumer
is `exp_square_HIT/tools/small_case.jl`, which is run by hand.

---

## 6. Conventions and traps worth knowing before you change anything

- **The `_f64_lmwray3` filename suffix is load-bearing.** A Float64/LMWray3 record must never be
  confusable with the archived Float32/RK44 one, which is otherwise identically named.
- **The time-stepping scheme is always stated, never inherited.** Upstream moved
  `solve_unsteady`'s default from RK44 to LMWray3 at the merge. Reproducing an archived run means
  passing `RKMethods.RK44(; T)` explicitly.
- **`freeze = 1` at the LF step is not a tidy-up target.** It advances the OU forcing chain on
  exactly the schedule the HF reference's `freeze = 10` at a ten-times-smaller step does. The two
  TO online drivers assert it after inheriting `ou_bodyforce`.
- **`Re = T(2_000)` is re-stated after every `params_track...` splat.** The splat carries the
  archive's Float32 parameters and a later key wins; without the override the setup is silently
  built at single precision.
- **Processor order matters.** `fieldsaver` is registered *before* `qoisaver` so that `qoisaver`'s
  initial `state[] = state[]` poke also reaches the field saver — which is why `q` has `nstep + 1`
  columns and why a `t = 0` field is always stored.
- **The turbulence gate.** `LinReg` zeroes the whole `dQ` vector whenever *any* `|q*_i|` falls
  below `TURBULENCE_GATE = 1e-2`. It is a laminar-start gate introduced for Taylor-Green, not a
  per-band numerical guard, and on a stationary turbulent testbed a nonzero firing rate is an alarm
  about the run. `MVG_sampler` has no such gate at all, so the two closures are not treated alike.
