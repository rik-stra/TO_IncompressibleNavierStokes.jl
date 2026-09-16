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

✅ **Every GPU batch script in this directory is now on `gpu_h100` + `julia_h100` with the same
`JULIA_CPU_TARGET`** (`run_d6.sh` was moved there 2026-09-16; it had been the only one on `gpu_a100`/`julia_a1003`, and its `--partition` line had been missed when the depot was switched). The three production scripts
`run_d6_linreg1.sh`, `run_d6_linreg7.sh`, `run_d6_ddn.sh` were written to match. `JULIA_CPU_TARGET` multiversioning means one depot serves both
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
