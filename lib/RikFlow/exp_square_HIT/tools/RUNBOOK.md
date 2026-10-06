# D6 on Snellius — what to copy, what to run

The operational half of **P2r / R3 (was P2c)**: the multi-IC ensemble that metric #17 (lead-resolved
spread–skill) and RH-3 (lead-resolved rank histograms) need. Design and rationale live in
`meta_files/handoff_p2c_d6.md`; the numbers, once there are any, go in
`lib/RikFlow/analysis/results.md`.

---

## 🔴 Rebuilt on the rebaselined pipeline, 2026-09-15/16 — read this first

Everything below was rewritten. **The 2026-09-11 version of this file described a D6 built from
paper 2's archive and is wrong in every quantity that matters.** What changed:

| | was (≤ 2026-09-11) | is now |
|---|---|---|
| record the ICs are cut from | paper 2's archived `data_track2` | **R1**, `data_track_..._f64_lmwray3` |
| forecast length `N_LEAD` | 1208 steps (3.02 TU) | **1200 steps (3.00 TU)** |
| warm-up `N_WARM` | 100 steps | **100 steps** (was 220 between 2026-09-15 and 2026-09-16) |
| what sets the forecast length | `10 × T_int` of the **correction**, on the archive | the **reference's ACF**, measured (`plot_acf.jl`) |
| IC pool | `k ∈ [42, 387]`, 346 fields | **`k ∈ [42, 388]`, 347 fields** |
| ordinal 0's oracle | paper 2's archived LinReg1 | **R2's own LinReg1 replica 1** |
| `Z[16,32]` carve-out in the verdict | present | **removed** |
| closures that can run D6 | LRS only | **LRS and DDN** (`D6_CLOSURE`) |
| package size | 3.31 MB each | **6.61 MB each** (Float64) |

🔴 **No scored D6 run exists yet on the rebaselined pipeline.** The 2026-09-11 smoke and validation
passes were against the archive and do not transfer.

⚠️ `Distributions` and `PDMats` are still **pinned** (`=0.25.117`, `=0.11.32`). Do not lift them:
the archived `LinReg.jld2` stores its residual model as a two-parameter `PDMat`, and a newer PDMats
makes `rand(rng, stoch_distr)` — which every deployed step calls — fail (gotcha #42).

---

## What to copy

Build everything first, on the workstation — instant apart from one 2.6 GB read:

```bash
cd lib/RikFlow
julia --startup-file=no --project=analysis analysis/build_d6_ics.jl        # 180 + validation + DDN
```

That writes into `lib/RikFlow/analysis/output/d6_ics/`:

| file | size | why |
|---|---|---|
| `d6_ic_manifest.jld2` | 21 kB | 🔑 **not optional.** `ic_dir()` locates the directory *by* this file, and `load_ic` cross-checks it against `select_ics` — that check is what catches an IC set built with a different `K`, `nlead` or record from the one the driver assumes |
| `d6_ic_<k>.jld2` | 6.61 MB each | one per IC, `k = 42 … 388`; **1.19 GB** for all 180 |
| `d6_ic_validation.jld2` | 6.60 MB | ordinal 0 — R2's own online initial condition |
| `d6_ddn_traindata.jld2` | 0.18 MB | the `dQ` slice the DDN fits on; needed only for `D6_CLOSURE=ddn` |

Plus the fitted model for whichever closure you are running:

- **LRS:** `exp_square_HIT/output/TO_LRS/LinReg1/LinReg.jld2` and `.../LinReg7/LinReg.jld2` (15 kB
  each) — the two production cells. Any other `LinReg<n>` works via `D6_MODEL`.
- **DDN:** nothing extra — `d6_ddn_traindata.jld2` above is it.

**Deliberately not copied.** The 2.6 GB tracked record and the 2.6 GB HF reference: **scoring
happens locally** after the runs come back, so the truth never needs to be on Snellius.
`build_d6_ics.jl` pre-slices the record once, on the workstation, which is the whole reason for one
file per IC.

```bash
# from lib/RikFlow on the workstation; $SNEL = login, $REPO = fork root there
D=$REPO/lib/RikFlow/exp_square_HIT/output
ssh $SNEL "mkdir -p $D/d6_ics $D/TO_LRS/LinReg1 $D/D6"

# pilot: the validation package, the manifest, the DDN slice and the first five ICs
scp analysis/output/d6_ics/d6_ic_{validation,manifest}.jld2 \
    analysis/output/d6_ics/d6_ddn_traindata.jld2 \
    $SNEL:$D/d6_ics/
scp analysis/output/d6_ics/d6_ic_{42,44,46,48,49}.jld2 $SNEL:$D/d6_ics/
scp exp_square_HIT/output/TO_LRS/LinReg1/LinReg.jld2   $SNEL:$D/TO_LRS/LinReg1/
```

⚠️ **The pilot's five ordinals are 1–5, whose `k` are 42, 44, 46, 48, 49** — not 42/44/46/48/50 as
the old file said. The spacing changed with the pool. Get them from
`julia --project=analysis -e 'include("analysis/build_d6_ics.jl"); println(select_ics(; K=180).k[1:5])'`
rather than from memory.

Those are `run_d6.jl`'s default locations, so no environment variables are needed. To put them
elsewhere: `D6_IC_DIR`, `D6_MODEL`, `D6_OUT`, `D6_MEMBERS`, `D6_CLOSURE`, `D6_DDN_DATA`, `D6_TRUTH`.

---

## What to run

```bash
cd $REPO/lib/RikFlow/exp_square_HIT
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:      # trailing colon, as every script here has it
julia --project -e 'using Pkg; Pkg.instantiate()'
```

🔑 **Partition (Rik, 2026-10-06): paper production runs on `gpu_a100`** — `run_d6_packed.sh`
(hindcasts, 10 tasks × 9 ICs) and `run_online.sh` (long runs). Measured on the same packed smoke:
A100 18 s/member at 128 SBU/h (job 27660921), H100 12–13 s/member at 192 SBU/h (27660628), so a
packed hindcast is ~650 SBU on the A100 vs ~730 on the H100 and a long run costs the same on
either. The Sept baselines (`run_d6_linreg1.sh`, `run_d6_linreg7.sh`, `run_d6_ddn.sh`) and
`run_track_ref.sh` are on `gpu_a100` too; the training, probe and p4grid scripts stay on
`gpu_h100`. Every script uses the `julia_h100` depot with the same `JULIA_CPU_TARGET`
(`run_d6.sh` was moved off `julia_a1003` on 2026-09-16). `JULIA_CPU_TARGET` multiversioning means one depot serves both
partitions, so warm this one and every script benefits. ⚠️ Keep the target string identical across
`run_d6.sh`, the three `run_d6_*.sh`, `run_online_array.sh` and `run_train_lrs.sh` — Julia validates a compile cache against
the target it was built for, so a mismatch silently recompiles inside the walltime.

### 1 · Smoke test — first, and it is cheap

```bash
julia --project tools/smoke_d6.jl
```

Asserts completion, every expected output key, `q` with `nt + 1` columns, no NaN, the clamp count,
and two things that matter more than the rest:

- 🔴 **that no velocity fields were written.** `params_track` carries `savefreq = 100`; at 2392 steps
  that is ~24 snapshots × 3.3 MB per run — **over 100 GB** across a full 180 × 10 run against
  ~160 MB of QoIs. `savefreq = nt + 1` leaves **one** `t = 0` field in memory, not zero:
  `qoisaver`'s initializer does `state[] = state[]` (`RikFlow.jl:332`), which is what gives `q` its
  `nstep+1` columns and therefore the offset the whole index alignment rests on, and `fieldsaver` is
  registered before it so it fires at `n = 0`. What keeps the disk cost at zero is that
  `run_d6.jl`'s `jldsave` writes no `fields` key at all.
- 🔴 **that `ou_advance` actually changes the trajectory.** The unit tests prove the replay is
  correct arithmetic and `analysis/ou_replay.jl` proves the advance count is right, but neither goes
  through `online_sgs`. If the keyword were dropped between the driver and `Setup`, every member's
  forcing would be out of phase with its own initial condition, the spread–skill ratio would be
  biased *downward* — toward a false "over-confident" verdict — and nothing would say so.

⚠️ **Do not take a cost figure from this step.** At `M = 1` the whole run is compilation — the
2026-09-11 smoke printed 62.55 s/TU against a steady 3.56 s/TU, a factor 18 (gotcha #44).
`run_d6.jl` refuses to quote a rate at `M = 1` for that reason.

### 2 · Validation — one task, before the pilot

```bash
D6_MEMBERS=5 julia --project tools/run_d6.jl 0        # or sbatch with --array=0
```

Ordinal 0 is `fields[1]` of **R1** — the initial condition **R2's own online runs** launched from.
So `n_k = 0`, `ou_advance = 0` (the identity point of the replay), and the model seeds are the
driver's own `Xoshiro(236 + member)`. `D6_MEMBERS=5` matches R2's five replicas.

🔴 **The oracle moved from paper 2's archive to R2 on 2026-09-16.** The archive is a *different
dynamical system* — pre-`09954be1`, Float32, its own reference (gotchas #45, #46) — so a failed
reproduction against it could not distinguish a driver bug from the system difference. That is
exactly why the old criterion needed a `Z[16,32]` carve-out. R2's replica 1 shares the record, the
precision, the Nyquist convention, the solver and the reference, so **nothing is excluded from the
verdict any more**.

🔴 **What this step can and cannot ask for.** It is tempting to state the check as *"its `q` must
reproduce R2's replica over the whole run"*. That is unreachable, for one reason that remains even
with a same-system oracle:

**The statistic saturates.** Two replicas of one configuration — same code, same inputs, only the
model seed differing — separate to ~1 sd within a few TU, because the system is chaotic. A
full-window rms near that value is saturation, not a defect, which is why `compare_validation`
prints `replica_spread`'s own replica-to-replica yardstick beside it. Do not reintroduce a
full-trajectory tolerance.

So the criterion is two-part (#48):

- **Over the replayed warm-up**, `dQ` must be **bit-identical** — there the sampler emits stored
  columns verbatim, so anything else means the warm-up slice or the history layout is wrong.
- **After it**, a gate on `q` at the start and everything else *reported*, not gated.

✅ Ordinal 0 replays **100** warm-up steps (`N_WARM_DRIVER`), matching what R2's driver ran, and
since 2026-09-16 the scored set replays the same 100 (`N_WARM`). The validation therefore exercises
the deployed length, not just the mechanism. They remain separate constants: if `N_WARM` moves again,
`N_WARM_DRIVER` must not follow it, or the validation would check inputs nobody ever ran.

### 3 · Pilot

```bash
# batch_scripts/run_d6.sh ships with --array=1-5
sbatch batch_scripts/run_d6.sh
```

### 4 · Pull back, and check the validation *before* reading anything else

```bash
scp "$SNEL:$D/D6/d6_*.jld2" analysis/output/D6/
julia --startup-file=no --project=analysis analysis/score_d6.jl
```

`compare_validation` runs first, deliberately: if the D6 path does not reproduce R2 over the
replayed window, no scored number from the same path means anything.

### 5 · Full run

```bash
scp analysis/output/d6_ics/d6_ic_*.jld2 $SNEL:$D/d6_ics/    # all 180; 1.19 GB, rebuilt 2026-09-16

sbatch --array=0 batch_scripts/run_d6_linreg1.sh            # validation, one task — check it first
sbatch batch_scripts/run_d6_linreg1.sh                      # K = 90, M = 10 -> output/D6_LinReg1
sbatch batch_scripts/run_d6_linreg7.sh                      #                -> output/D6_LinReg7
sbatch batch_scripts/run_d6_ddn.sh                          #                -> output/D6_DDN
```

🔴 **One directory per closure, and the three scripts already set it** (`D6_OUT`). The scorer globs
`d6_online_ic*_m*.jld2` and cannot tell which model wrote a file, so a shared directory silently
pools three closures into one ensemble. Score each separately:

```bash
for m in LinReg1 LinReg7 DDN; do
  D6_OUT=analysis/output/D6_$m julia --startup-file=no --project=analysis analysis/score_d6.jl
done
```

### Rerunning individual ordinals

```bash
sbatch --array=67,81,141 batch_scripts/run_d6_linreg1.sh    # the 2026-09-16 divergences
```

`--array` on the command line overrides the `#SBATCH` directive, so the closure's own `D6_OUT` and
`D6_MODEL` are reused — never copy the config into a separate rerun script. `run_ic` skips members
whose files already exist, and the skip happens **before the seed is derived**, so only the missing
members run and every member keeps the seed it would have had.

🔴 **Ordinals 67, 81, 141 (`k` = 170, 197, 313) diverged**; the pre-2026-09-16 driver raised on the
short `q` and aborted the task, losing 14 unattempted members to 3 divergences. The patched driver
writes the truncated trajectory with `diverged = true` and continues, so the rerun measures what
metric #16 needs. The divergences **will** reproduce — the seeds and the IC are identical — and that
is the point, not a failure of the rerun.

⚠️ **Afterwards, drop `D6_EXCLUDE_ICS` when scoring.** It exists only to keep the three closures on a
common IC set while LinReg1 was short; once its ensembles are complete, all three score at K = 90.


🔑 **All three use the same `--array=1-179:2`, and that is the point.** D6 is *paired*: every closure
forecasts from the same 90 initial conditions, so realisation variance cancels in the comparison.
Changing the range in one script without the others silently breaks the pairing.

🔴 **The validation belongs to LinReg1 only.** Ordinal 0's oracle is R2's own LinReg1 replica 1.
Under LinReg7 or the DDN the warm-up gate still passes — the replayed `dQ` is model-independent —
but the post-warm-up comparison is then against a different model's trajectory, which reads as
divergence that is really chaos plus a model difference. That is what the 2026-09-16 pilot did.

🔑 **`--array=1-179:2` gives K = 90, and that is the production setting, not a shortcut.**
`select_ics` is strictly monotone, so every second ordinal is exactly the half-density IC set —
spacing 0.97 TU, **1.8× the slowest level timescale**. At the full K = 180 the spacing is 0.4832 TU,
which is **0.89×** that timescale: adjacent ICs are genuinely correlated. `--array=2-180:2` is the
fill-in if the extra density is ever wanted.

🔴 **The ordinal → `k` map MOVED on 2026-09-16.** `select_ics` derives it from `kmin`/`kmax`, and a
shorter `nlead` and a shorter `nwarm` free 11 more fields at the end of the record, so the pool grew
to `[42, 388]` and the selection re-spaced: ordinals 1–4 are still `k = 42, 44, 46, 48` but ordinal 5
is now **`k = 50`**, not 49, and every later ordinal shifts.
The 60 LinReg7 pilot outputs on disk stay scoreable — `load_members` reads each run's own
`nwarm`, and a 2172-lead run contains every lead of a 1200-lead grid — but their ICs are the old
selection, so
they are not paired with anything produced after the change. Rebuild the packages before the full run.

⚠️ The 9.65 GPU-hour figure in the old file was measured at `nlead = 1208`. At 1200 the per-member
cost is within a few percent of it, but GPU compilation now dominates a task; **budget from the
pilot's measured rate, not from this paragraph.**

### Running the DDN

```bash
sbatch batch_scripts/run_d6_ddn.sh      # D6_CLOSURE=ddn and D6_OUT=output/D6_DDN are set in it
```

New on 2026-09-16 and it is what makes the LRS-vs-DDN comparison meaningful: `MVG_sampler` now takes
the same `spinnup_data` the LRS does, so both closures are advanced through the identical recorded
warm-up and **enter the forecast from the same physical state** (gotcha #55). Without it the DDN
would start sampling immediately while the LRS was still being replayed, and every lead-resolved
difference would carry that offset.

⚠️ Set `D6_OUT` to a separate directory per closure, or the two write the same filenames.

### Running a StochLSTM (M4) fit — `D6_CLOSURE=lstm` (2026-09-28)

```bash
cd $REPO/lib/RikFlow/exp_square_HIT
D6_CLOSURE=lstm D6_MODEL=diag/r3_lin_sd_h2 D6_OUT=output/D6_r3_lin_sd_h2 \
  julia --project tools/run_d6.jl <ordinal>
```

🔑 **Same project as the LRS: `--project` = `lib/RikFlow`, NOT `lib/RikFlow/training`.** The deployed
M4 closure (`StochLSTM`, `src/ts_lstm_online.jl`) is stdlib and part of RikFlow proper; Lux,
Optimisers and Zygote are needed only to *train* (`RikFlowLuxExt`). `load_stochlstm` needs JLD2 only.
So the existing batch scripts and depot serve M4 unchanged — `12_online_StochLSTM.jl` has always run
this way. Do not add `using Lux` to the driver.

What to copy for M4: the fit directory, `exp_square_HIT/output/TO_LSTM/<subdir>/` —
`StochLSTM_seed<s>.jld2` (plus `seed_summary.jld2` if there is one). `D6_MODEL` is that directory,
absolute or relative to `output/TO_LSTM`; the **median seed** named by `seed_summary.jld2` is deployed,
else seed 1, exactly as `12_online_StochLSTM.jl` resolves it. A single `StochLSTM_seed<s>.jld2` path is
also accepted. Calibration / colour knobs (`noise_scale`, `dq_offset`, `eta_ar`, …) live in the fit's
`scaling` and are applied by `StochLSTM` itself; the output file lists which the fit carries
(`lstm_knobs`). `RIKFLOW_M4_UNTIED` (a diagnostic ablation) is not deployable through D6.

**The contract is `LinReg`'s**, and each part is pinned in `test/test_d6_lstm.jl` (V72):

- the same IC package and the same `dQ_warm`, replayed verbatim for `N_WARM = 100` steps, returned
  unconverted — 🔴 **`run_d6.jl` now checks `dQ[:, 1:nwarm]` bit-identity on the node, for every
  closure**, and aborts the task if it fails (`warm_identical = true` in every file written);
- the member seed `member_seed(k, member)`, and the replay draws nothing from it (V38/V42);
- 🔴 **`TURBULENCE_GATE`**, passed explicitly as `RikFlow.TURBULENCE_GATE` (not the constructor's
  literal default) and applied where `LinReg` applies it: on the final correction, after the output
  map and any calibration offset, before the step enters the history, never during the replay. Every
  output file — LRS, DDN and M4 — now records `gate_threshold` (NaN for the DDN, which has none by
  decision, #65), `gate_nfired` and `gate_first_lead`; the scorer's `clamp_report` counts the same
  identically-zero `dQ` columns. ✅ **Every M4 online run to date had the gate**: it has been in
  `StochLSTM` since the closure was written (`fce0d767`, 2026-09-17) and `12_online_StochLSTM.jl`
  never overrides it. (Only the offline replay analyses `m4_noise_colour.jl` and
  `m4_replay_rollout.jl` pass `gate = 0.0`; they are not in-solver runs.)
- divergence handling, `ou_advance = n_k`, `savefreq > nt` and the output format are unchanged; the
  file adds `model_name` (e.g. `diag/r3_lin_sd_h2`), the `lstm_*` fields and `deploy_seed`.

⚠️ The warm-up is 100 steps for every closure, so a window-mode fit needs `W − 1 <= 100` and
`h <= 100` (checked before member 1). `RIKFLOW_M4_NWARM` does not apply here.

### The mini-D6 and the partition (plan §7, 2026-09-28)

The partition is defined once, `D6_BLOCKS` in `analysis/build_d6_ics.jl` (V70):
**selection** = IC time `t ∈ [52, 74]` TU (46 packages at K = 180: ordinals 88–133), **confirmation**
= `t ∈ [76, 97]` TU, with 2 TU embargoes after the 1–50 TU training window and between the blocks.

| variable | meaning |
|---|---|
| `D6_BLOCK` | `selection` \| `confirmation` |
| `D6_T_MIN`, `D6_T_MAX` | an explicit IC-time window in TU instead (mutually exclusive with `D6_BLOCK`) |
| `D6_STRIDE` | every n-th IC of the window |
| `D6_NLEAD` | forecast steps after the warm-up; default 1200, **mini-D6 400** |
| `D6_MEMBERS` | M; mini-D6 5 |

```bash
# which ICs a filter selects (reads the manifest; no GPU)
D6_BLOCK=selection julia --project tools/run_d6.jl --list

# desktop: the whole block in ONE process, so the solver compiles once
D6_CLOSURE=lstm D6_MODEL=diag/r3_lin_sd_h2 D6_BLOCK=selection D6_MEMBERS=5 D6_NLEAD=400 \
  D6_OUT=output/D6mini_r3_lin_sd_h2 julia --project tools/run_d6.jl --all

# Snellius: the same --array for every cell; ordinals outside the block exit at once
D6_BLOCK=selection ... sbatch --array=88-133 batch_scripts/run_d6.sh
```

🔴 **One directory per closure, model, block and N_LEAD.** `run_d6.jl` refuses to write into a
directory holding members of another experiment, and `score_d6.jl` refuses to score one
(`d6_run_identity`). Stride and time window are *not* part of the identity, so a stride-2 run and its
fill-in may share a directory.

Scoring a mini-D6 needs nothing special — the lead grid is truncated to the run's own `nlead`
(25, 50, 100, 200, 400 at 400) and any IC subset scores. To score an existing *full* D6 on one block
(e.g. LinReg1 on the selection ICs), set `D6_BLOCK` for the scorer; the score file name then carries
the block. The plan's **primary score** — paired fair ensemble CRPS over leads <= 0.5 TU and the six
bands (each standardised by the truth's sd), with a 90% IC-block bootstrap CI (blocks of `floor(1 TU / spacing) + 1` ICs) — is

```bash
julia --startup-file=no --project=analysis analysis/score_d6.jl --paired <dir A> <dir B>
```

negative = A better. It pairs over the ICs both directories scored (dropped ICs are printed) and
refuses two different blocks. ⚠️ Pairing is by IC only: the same member seed is not common random
numbers across different samplers.

---

## Three traps worth repeating

⚠️ **`d6_valid_ic1_*` is `k = 1`, not ordinal 1.** The validation package hardcodes `k = 1`, and
output filenames carry `k`, not the ordinal. Scored ICs start at `k = 42`, so there is no collision —
but when you `scp` the validation results back the glob is `d6_valid_ic1_m*.jld2`, and it has
nothing to do with ordinal 1.

⚠️ **An empty `.out` means not started, not hung.** Julia buffers stdout when redirected; every
progress line in `run_d6.jl` and `smoke_d6.jl` is followed by `flush(stdout)`. Two healthy jobs have
already been killed on that misreading.

⚠️ **The validation IC is not part of the scored set and must not become part of it.** `t = 0` is
inside M0's fit window, so its short-lead spread would be measured on data the conditional mean has
already seen; and V28 requires D6's set disjoint from the online runs' own IC. The separation is by
filename — `d6_valid_ic1_m*.jld2` against `d6_online_ic<k>_m*.jld2` — so the scorer's glob cannot
see it and no filtering step can be forgotten.

---

## Without any data

```bash
julia --startup-file=no --project=analysis analysis/score_d6.jl --preview
```

Prints the per-QoI lead grids and the index alignment worked out for one IC, so the design can be
read rather than trusted. The grids — `{0.25, 0.5, 1, 2, 5, 10} × T_int(i)`, in steps:

| QoI | `T_int` [TU] | leads [steps] |
|---|---|---|
| Z[0,6] | 0.2489 | 25, 50, 100, 199, 498 |
| E[0,6] | 0.4732 | 47, 95, 189, 379, 946 |
| Z[7,15] | 0.4893 | 49, 98, 196, 391, 979 |
| E[7,15] | 0.4742 | 47, 95, 190, 379, 948 |
| Z[16,32] | 0.5430 | 54, 109, 217, 434, **1086** |
| E[16,32] | 0.5395 | 54, 108, 216, 432, 1079 |

**26 distinct leads in the union; the longest is 1086 of the 1200 available.** The `10 × T_int`
column was dropped on 2026-09-16: the reference's own autocorrelation is inside its ±2 Bartlett band
from `2 × T_int` onward (`results.md` §1), so the `5×` and `10×` leads measure climatology, not
forecast skill. `5×` is kept as the single saturation anchor for the spread–skill ratio.

🔴 **These are the LEVEL's decorrelation times, not the correction's** (Rik, 2026-09-15). D6 scores
the forecast of the QoI *level*, so the level's timescale is what has to saturate. The old grid used
the correction's, on the archive, and its longest lead was only 5.6× the slowest level timescale — a
grid that could have stopped before the slowest band saturated.

🔑 **One consequence worth knowing.** On the correction the six timescales spanned a factor **36.8**,
and that was the entire argument for a per-QoI grid. On the level they span **2.18** (0.5430 /
0.2489). The per-QoI grid is kept because it is still correct and costs nothing, but it is no longer
load-bearing, and a single shared grid would now be defensible.

---

## Snellius grid (plan step 1+, 2026-09-28)

This section implements the plan's screening funnel (plan.md §3, *reduced ladder*; §7, *Screening funnel*) as SLURM arrays.
**The grid values are provisional; the pipeline is not.** Every value lives in ONE declarative file,
`batch_scripts/p4grid/spec.toml`: per-h λ, weight decay, capacity and seeds. The table
`batch_scripts/p4grid/fits.csv` is generated from it by `tools/p4grid_generate.jl` and committed. Never
hand-edit the CSV, and never name an output directory by hand.

| stage | where | script | what |
|---|---|---|---|
| 0 λ calibration | desktop CPU, 25 s | `analysis/skip_lambda_calibration.jl` | maps M4-path skip λ ↔ LinReg λ (effective dof, held-out residual lag-1/2/5, SSE, ρ(C̃)); proposes 3 λ per h |
| A fits | Snellius array (CPU work on `gpu_h100`), or the desktop | `batch_scripts/run_p4grid_fit.sh` | one row of `fits.csv` per task: M0@50, M3ᶠ, M0ᵛ; fit on 1–50 TU, held out 52–74 TU |
| B offline selection | desktop | `analysis/p4grid_select.jl` | one row per config; rule below; writes `smoke_list.txt` |
| C 20 TU smoke | Snellius GPU array | `batch_scripts/run_p4grid_online.sh` | median seed of every config that passed offline, plus its matched M0; 3 replicas |
| C→D cap | desktop | `analysis/p4grid_select.jl --after-smoke` | smoke gate + cap at ≤ 6 configs; writes `d6_list.txt`, `d6_pairs.csv` |
| D mini-D6 | Snellius GPU array | `batch_scripts/run_p4grid_d6.sh` (+ `tools/p4grid_d6_chunk.jl`) | selection block (46 ICs, ordinals 88–133), M = 5, N_LEAD = 400; fit seeds 1–3 + matched M0 |
| E long runs | Snellius GPU array | `run_p4grid_online.sh` with `P4GRID_TSIM=100` | mini-D6 survivors + matched M0; 3 × 100 TU; pass/fail |
| F ridge + colour | Snellius GPU array | `batch_scripts/run_d6mini_lrs_ar.sh <LinRegN>` | mini-D6 of LinReg{2,7,8} and their `_ar2` variants (Agent D's builder, `tools/lrs_ar_variant.jl`) |
| G collect | desktop | `analysis/p4grid_collect.jl` | B tables, paired primary scores for D and F, E census; writes `long_list.txt` |

### Declared rules (2026-09-28)

- **B, offline.**
  - **M3ᶠ** is scored by held-out ensemble CRPS (`diag.held_crps`). Its matched M0(h, λ) is `diag.skip0.crps`: the same frozen skip with seeded η at update 0, which is the M0@50 fit of that (h, λ). The script checks the M0 directory's `Ws` against it.
  - **M0ᵛ** is scored by held-out NLL per step, against `diag.linear_nll`.
  - **PASS:** the gain over the best linear model across h, at the same λ, is > 0 for every seed, with at least 3 seeds.
  - Printed next to the verdict, for information only: per-seed min/median/max, the gain over the matched M0, and the gain over the best linear model across all h *and* λ. λ is the online-stability knob, which no one-step score can see.
- **C, the smoke is a gate, never a ranking.** Plan §24 rules out selecting on 20 TU free runs. A config passes if:
  - all 3 replicas complete and are finite;
  - the gate fires zero times after the warm-up;
  - its median level offset, max_k |Δmean_k|/sd_k, is no larger than max(the reference's own 20 TU windows, the matched M0's) + 0.1 sd (`P4GRID_SMOKE_TOL`).
  - The reference is used only up to t ≤ 74 TU.
- **Cap for D.** Take the configs that pass both offline and smoke. Keep the best per (cell, h, λ) by median-seed gain, then the top **6** (`P4GRID_MAX_D6`). Each kept config runs fit seeds 1–3, and the matched M0s run too.
- **D pass (plan §7, stage 3).** The `score_d6.jl --paired` CI excludes zero in the fit's favour in ≥ 2 of 3 seeds. ⚠️ The mini-D6 only resolves CRPS gains of ≥ 13–18% (`results_LSTMS.md` §10a).
- **E pass.** Every replica completes (40001 columns) and is finite, and the whole-run level offset is no larger than the matched M0's. For drift and basins across 20 TU chunks, use `m4_screen_long.jl`.

### What to copy to Snellius (from `lib/RikFlow` on the workstation)

```bash
# $SNEL = login node, $REPO = the fork root there (/gpfs/home6/rhoekstra/time_series_INS per the fits' provenance)
D=$REPO/lib/RikFlow/exp_square_HIT/output
# code: Rik commits and pushes; on Snellius, `git pull` (spec.toml, fits.csv, the scripts, tools/p4grid_*)
ssh $SNEL "mkdir -p $D/TO_LSTM $D/d6_ics $REPO/lib/RikFlow/analysis/data"
rsync -av analysis/data/data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2 \
      $SNEL:$REPO/lib/RikFlow/analysis/data/                                   # 7.7 MB: the fits' QoI cache
rsync -av exp_square_HIT/output/TO_LSTM/inputs_lstm.jld2 $SNEL:$D/TO_LSTM/        # 22 kB: the cfg row
rsync -av exp_square_HIT/output/online_ic_data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2 \
      $SNEL:$D/                                                                 # 7 MB: the stage C/E IC
rsync -av --ignore-existing analysis/output/d6_ics/ $SNEL:$D/d6_ics/              # manifest + packages (88-133 needed)
# stage F only, once Agent D has built the variants:
rsync -av exp_square_HIT/output/TO_LRS/LinReg{2,7,8}{,_ar2} $SNEL:$D/TO_LRS/
```

To print the `k` of the selection-block IC packages, run `D6_BLOCK=selection julia --project tools/run_d6.jl --list`.
The manifest is mandatory: `ic_dir()` finds the directory by it.

### Submit order (from `exp_square_HIT/` on Snellius)

```bash
export JULIA_DEPOT_PATH=$HOME/julia/julia_h100:
export JULIA_CPU_TARGET="generic;znver2,clone_all;znver4,clone_all;icelake-server,clone_all"
julia --project=../training -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'   # warm once, on the login node
julia --project=..          -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# A -- fits. The generator prints the exact range: 72 runnable rows with the provisional spec.
A=$(P4GRID_PACK=8 sbatch --parsable --array=1-9 batch_scripts/run_p4grid_fit.sh)   # or --array=1-72%20 unpacked
# F -- independent of A. One submission per base: base + _ar2, 2 IC chunks each.
for b in LinReg2 LinReg7 LinReg8; do sbatch --array=1-4 batch_scripts/run_d6mini_lrs_ar.sh $b; done
```

**Between A and C (desktop).** Check a few `output/TO_LSTM/p4grid/_logs/<tag>.log` files:
- `windows (TU)` should read train (1, 50) and held (52, 74);
- note `stop_reason`, and whether the M3ᶠ fits hit the epoch cap.

Then:

```bash
rsync -av --exclude 'data_online_*' $SNEL:$D/TO_LSTM/p4grid/ exp_square_HIT/output/TO_LSTM/p4grid/   # ~100 kB/fit
julia --startup-file=no --project=analysis analysis/p4grid_select.jl            # stage B -> analysis/output/p4grid/
scp analysis/output/p4grid/smoke_list.txt $SNEL:$REPO/lib/RikFlow/exp_square_HIT/batch_scripts/p4grid/
```

```bash
# C -- 20 TU smoke; the select script prints --array=1-<3 x lines>
sbatch -t 00:45:00 --array=1-<3n>%12 batch_scripts/run_p4grid_online.sh
```

**Between C and D.** Pull the replicas (61 MB each), then apply the gate and the cap:

```bash
rsync -av --include '*/' --include 'data_online_tsim20.0_*' --exclude '*' \
      $SNEL:$D/TO_LSTM/p4grid/ exp_square_HIT/output/TO_LSTM/p4grid/
julia --startup-file=no --project=analysis analysis/p4grid_select.jl --after-smoke
TSCREEN=20 REF_TU_MAX=74 M4_SCREEN_SUBDIR=p4grid julia --project=analysis analysis/m4_screen.jl   # the human-readable view
scp analysis/output/p4grid/d6_list.txt $SNEL:$REPO/lib/RikFlow/exp_square_HIT/batch_scripts/p4grid/
```

```bash
# D -- mini-D6, 2 IC chunks per closure; <= 24 closures -> --array=1-<=48
sbatch --array=1-<2n>%16 batch_scripts/run_p4grid_d6.sh
```

**Between D and E.** Pull and score. The collector reads `analysis/output/p4grid/d6_pairs.csv`:

```bash
mkdir -p analysis/output/snellius_p4grid
rsync -av $SNEL:$D/D6mini_p4grid analysis/output/snellius_p4grid/
rsync -av $SNEL:"$D/D6mini_LinReg*" analysis/output/snellius_p4grid/    # stage F
julia --startup-file=no --project=analysis analysis/p4grid_collect.jl     # B + D + F tables, long_list.txt
scp analysis/output/p4grid/long_list.txt $SNEL:$REPO/lib/RikFlow/exp_square_HIT/batch_scripts/p4grid/
```

🔑 `analysis/output/snellius_p4grid/` keeps the pulled D6 directories apart from the desktop's own
`analysis/output/D6mini_*`. Agent D's local LinReg7 runs have the same names.

```bash
# E -- 100 TU x 3 replicas of the survivors + matched M0
P4GRID_ONLINE_LIST=batch_scripts/p4grid/long_list.txt P4GRID_TSIM=100 \
  sbatch --array=1-<3n> batch_scripts/run_p4grid_online.sh
```

Pull `data_online_tsim100.0_*` (276 MB each) and rerun `p4grid_collect.jl`. It prints the
`m4_screen_long.jl` command for the drift/basin view.

`--dependency=afterok:<job>` only helps inside Snellius. Every stage boundary is a desktop check by
design, because the scores need the reference and the reference stays local. Use it to chain a rerun
onto its own stage's first submission, e.g.
`sbatch --dependency=afterok:$A --array=<lost rows> batch_scripts/run_p4grid_fit.sh`.
Finished rows, replicas and D6 members are skipped, so rerunning a whole range is also safe.

### Cost (budget; measured where stated)

| stage | tasks | per task | GPU-h |
|---|---|---|---|
| A | 72 rows, packed 8 per task → 9 tasks | M3ᶠ fit 0.7–1.5 min (measured: desktop, C's h = 2 fits) + ~1 min load | ~1.5 packed (~5 unpacked). **Desktop: 0 GPU-h, ~30 min at 4 in parallel** |
| C | ≤ 3 × (passing configs + their M0s), ~30 | 20 TU × 12 s/TU + 2 min compile ≈ 6 min | ≤ 3 |
| D | ≤ 24 closures × 2 chunks | 288 TU per closure × 12 s/TU / 2 + 1.5 min ≈ 30 min | ≤ 24 (≈ 13 at LRS's 6.5 s/TU) |
| E | ≤ 6 dirs × 3 | 100 TU × 12 s/TU ≈ 20 min | ≤ 6 |
| F | 3 bases × 2 models × 2 chunks | 288 TU × 6.5 s/TU / 2 ≈ 16 min | ~3.4 |

- **LSTM closure rate.** 12 s/TU is the upper end measured on the desktop 3090 (7.5–12 s/TU, shared GPU).
- **H100 rate.** Only LRS has been measured there: ~6.5 s/TU steady, ~60–90 s compile per process (`handoff_p2c_d6.md`).
- **Dry run.** Compile + one M = 1 ordinal took 66 s on the 3090.
- **SBU.** Multiply GPU-h by the partition's SBU rate from Snellius accounting.

### Dry runs (2026-09-28, desktop, toy size; outputs in the job scratch, not in `output/`)

The dry runs used these switches:

| variable | effect |
|---|---|
| `P4GRID_LOCAL=1` | keep the desktop depot and CPU target |
| `P4GRID_ROW=<r>` / `P4GRID_TASK=<t>` | run one row/task without SLURM |
| `P4GRID_OUTROOT=<abs dir>` | write fits there, never into `output/` |
| `P4GRID_EXTRA="RIKFLOW_W_EPOCHS=1 RIKFLOW_D_EPOCHS=1"` | cut the training budget |
| `P4GRID_D6_ROOT=<abs dir>` | write D6 output there |
| `P4GRID_NCHUNK=46` | one ordinal per chunk |

All of these ran clean:
- all three fit tools;
- the D6 chunk wrapper (ordinal 88, warm-up bit-identical, gate 0/25);
- the offline and after-smoke selection, on Agent C's `p4/`;
- the collector, including the paired primary score on a partial LinReg7_ar2 vs LinReg7.

### Known gaps (tool options that do not exist; not added here)

- **`m4_diag_fit.jl` has no skip-λ option**; its skip is always least squares. The M0ᵛ λ > 0 rows are
  generated with status `blocked:` and not run. They need a `RIKFLOW_D_LAMBDA` (Agent C's file).
- **`m4_diag_fit.jl` has no output-subdir option.** The generator routes its output with `RIKFLOW_D_TAG=../p4grid/<tag>`.
- **`m4_diag_fit.jl` does not record its held-out window** in `diag`; only the log shows 52–74 TU. So B
  cannot verify it from the file, as it does for M3ᶠ.
- **`m4_window_fit.jl` and `m4_linear_eta.jl` load the whole 0–100 TU QoI cache** and build the history over
  it. They fit only on 1–50 TU and score only on 52–74 TU (printed in each log), but data past 74 TU are in memory.
- **`m4_window_fit.jl` always names its file `StochLSTM_seed1.jld2`**, whatever the seed. This is harmless here
  (one seed per directory), and D6 and the online runs deploy that file.
