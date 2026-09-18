# M4's configuration table.
#
#     julia --startup-file=no --project=lib/RikFlow lib/RikFlow/exp_square_HIT/10_setup_lstm.jl
#
# 🔴 **Deliberately NOT appended to `4_setup_search.jl`'s `CELLS`.** In that table the index *is*
# the identity and the output directory is literally `LinReg<i>` (README §3, stage 3), so an M4 row
# there would be a configuration called `LinReg` that is not a linear regression. M4 gets its own
# table, its own index space and `StochLSTM<i>` directories.
#
# ⚠️ **Append-only, exactly as `CELLS` is.** Inserting a row renames every configuration after it,
# and the name is what the batch scripts, the output directories and every results table refer to.

using JLD2

TO_folder = @__DIR__() * "/output/TO_LSTM"
mkpath(TO_folder)

# The defaults common to every row. `h = 1` is M4's natural setting -- the recurrence is supposed
# to carry the memory, which is the whole reason for using one, and the level's 1/e time is
# 116-142 steps, so M0's `h = 5` window would be about 4% of one decay time either way.
# 🔴 **These are NOT the source's dimensions, deliberately.** At hidden/latent/encoder = 60 the
# model has 43 128 parameters against 21 594 training target values -- twice as many parameters as
# data. Their 60 units predict a quasi-geostrophic *field*; we predict six scalars. See
# `analysis/results_LSTMS.md` §1.
const BASE = (;
    hist_var = :q_star_q,
    include_predictor = true,
    normalization = :normal,
    train_range = (400, 4000),      # t in [1, 10] TU, the same window the M0 cells are fitted on
    h = 1,                          # §1: an h = 5 lag window is 4% of one decay time
    n_hidden = 16,
    n_latent = 4,
    n_encoder = 0,                  # the tex's linear encoder, not the repository's dense layer
    emission = :none,               # the Gaussian head is OFF; the latent path is the only noise
    uclip = (-4.0, 4.0),            # unused when emission = :none; kept for the :constant rows
    beta = 1e-4,                    # the source's KL weight. NEVER called lambda here.
    # 🔑 L covers more than one ringing period of the level's ACF (~1 TU = 400 steps) and burn is
    # 100 steps, D6's own warm-up length (claude_memory.md #67), which is close to the 1/e
    # crossing at 116-142. Sized from the crossings, NOT from T_int, because the level's ACF
    # rings rather than decaying -- `results.md` §1, `results_LSTMS.md` §3.
    L = 500,
    burn = 100,
    # 🔴 300 -> 3000 (measured 2026-09-18, `results_LSTMS.md` §6). At lr = 1e-2 the validation
    # loss is still in a clean power-law descent at epoch 300 and flattens only at ~2000; the
    # decay-on-plateau schedule does not fire until epoch ~1650. 3000 rather than 2000 because
    # best-iterate selection makes over-running cost time and nothing else, and on the cluster the
    # extra steps are free.
    # ⚠️ An epoch here is TWO optimiser steps, not one: at L = 500 the 2879-row training block
    # yields 7 segments in two LENGTH groups (6 x 500 + 1 x 479), and `group_by_length` batches
    # each group separately. `batch` itself is unreachable. If `stride` is ever lowered this
    # number MUST be re-derived -- the update count is `epochs * updates/epoch`. See §6.2.
    epochs = 3000,
    # 🔴 32 (Rik, 2026-09-18), up from 8. ⚠️ **It changes nothing at the current stride** -- 7
    # segments cannot fill a batch of 8, let alone 32 -- and takes effect only once `stride` is
    # shortened, where it is the amortisation lever: the recurrence runs over `H x B` matrices
    # (V50), so a wider batch is more work per traced timestep at the same Zygote overhead, which
    # §5 measures as the thing this fit is actually bound by. At `stride = 50` there are 49
    # segments, so one update becomes a GEMM over 32 of them instead of 8.
    batch = 32,
    # 🔴 1e-2, not the 1e-3 of the first build (Rik, 2026-09-18). The decay-on-plateau schedule
    # stays: x0.3 after 20 epochs without a validation improvement, floor 1e-5. A high start with
    # a decay is what makes the loss curve readable in the first tens of epochs.
    lr = 1e-2,
    val_frac = 0.2,
    n_seeds = 5,                    # S6. The beta scan runs ONE seed via the driver's 2nd argument
    n_replicas = 5,
)

cell(i, arch; kwargs...) = (; name = "StochLSTM$(i)", arch, BASE..., kwargs...)

# The approved first experiment (Rik, 2026-09-17): Sørensen et al.'s conclusion is that the noise
# belongs in the **latent state**, so the two architectures that feed `z` into the recurrence come
# first, scanned over `beta` at one seed. `beta` regularises the *only* noise channel now that the
# emission head is gone, so it is a model parameter rather than a detail.
CELLS_LSTM = [
    # --- 1-6: the beta scan. A = :storn, B = :vrnn, beta in {0, 1e-4, 1e-2}.
    #
    # ⚠️ Run these with ONE seed -- `11_train_StochLSTM.jl <i> 1` -- then re-run the winner at all
    # `n_seeds`. Spending five seeds per point before knowing which beta is sensible is an hour
    # bought for nothing.
    cell(1, :storn; beta = 0.0),
    cell(2, :storn; beta = 1e-4),
    cell(3, :storn; beta = 1e-2),
    cell(4, :vrnn;  beta = 0.0),
    cell(5, :vrnn;  beta = 1e-4),
    cell(6, :vrnn;  beta = 1e-2),

    # --- 7: the control.
    #
    # 🔴 `emission = :constant`, NOT `:none`. A deterministic backbone with no emission noise has no
    # stochasticity at all -- it is a point predictor, cannot produce an ensemble, and `LSTMSpec`
    # refuses the combination. So the control's spread is a constant `Sigma`, which is also the
    # sharper contrast: noise upstream against noise at the output, state-independent.
    cell(7, :lstm; emission = :constant),

    # --- 8-9: the latent dimension, still open (§9 Q3). Run after beta is settled.
    cell(8, :vrnn; n_latent = 6),
    cell(9, :vrnn; n_latent = 8),

    # --- 10: the output-only variant. The one their result argues AGAINST, kept so the claim can
    # be tested rather than assumed, but not in the first pass.
    cell(10, :vaernn; emission = :constant),

    # --- 11-28: THE CAPACITY x REGULARISATION GRID (Rik, 2026-09-18).
    #
    # `beta` x `n_latent` x `n_hidden` = 3 x 2 x 3 = 18 cells, appended by `GRID` below rather than
    # written out, so the table cannot drift from the loop that generates it.
    #
    # 🔑 **Why a grid and not three separate sweeps.** `beta` regularises the latent path, and
    # `n_latent` is how much latent there is to regularise -- they are not separable, and neither
    # is independent of `n_hidden`, which sets how much the recurrence can do without the latent.
    # Sweeping them one at a time answers a question nobody asked.
    #
    # ⚠️ `n_latent = 6` is `N_Q`. That is the matched case and the reason 6 is in the list; 4 is
    # under-complete and forces compression. §9 Q3.
    # ⚠️ `n_hidden = 6` is SMALLER than `n_latent = 6`. Deliberate -- it is the corner where the
    # latent path has to carry state the cell cannot -- but it is also the corner most likely to
    # be simply bad, so do not read a poor result there as evidence about the latent dimension.
    # 🔴 **Three of the eighteen are EXACT re-runs of cells 1-3**: the grid's
    # `n_hidden = 16, n_latent = 4` column at the three betas is cells **13, 19, 25**, which is
    # cells 1, 2, 3. Same configuration, same seeds, therefore the same fit. The table is
    # append-only so they stay, but **skip them in the array** -- `--array=11,12,14-18,20-24,26-28`
    # is the 15 that are new -- or run them and use the pair as a free reproducibility check.
    # ⚠️ Cell 22 (`beta = 1e-4, n_latent = 6, n_hidden = 16`) is cell 8's geometry at `:storn`
    # rather than `:vrnn`, so it is NOT a duplicate; it is the arch contrast at that point.
]

# 🔴 ARCH FOR THE GRID. `:storn` -- upstream stochasticity with NO decoder skip, so the latent can
# only act THROUGH the recurrence, which is the cleanest test of the source's claim and the reason
# to prefer it over `:vrnn` when only one can be afforded. ⚠️ This was NOT specified; 18 cells on
# both architectures is 36, and the first pass does not justify it. Switch here and re-run
# `10_setup_lstm.jl` to move the grid to `:vrnn`, or append a second block for it.
const GRID_ARCH = :storn
const GRID_BETAS = (0.0, 1e-4, 1e-2)
const GRID_LATENT = (4, 6)
const GRID_HIDDEN = (6, 10, 16)

for b in GRID_BETAS, nz in GRID_LATENT, nh in GRID_HIDDEN
    push!(CELLS_LSTM,
          cell(length(CELLS_LSTM) + 1, GRID_ARCH; beta = b, n_latent = nz, n_hidden = nh))
end

for (i, c) in enumerate(CELLS_LSTM)
    c.name == "StochLSTM$(i)" ||
        error("row $i is named $(c.name): the index IS the identity, so the table is out of step")
end

# 🔑 Report duplicates rather than leave them to be discovered on the cluster. An append-only table
# plus a generated grid makes them inevitable, and a duplicate is not an error -- it is a re-run of
# the same fit, which is either wasted array time or a free reproducibility check, depending on
# whether the reader knows about it. `_dupes` is printed below and named in the array advice above.
const _CELLKEY = c -> (c.arch, c.beta, c.n_latent, c.n_hidden, c.n_encoder, c.emission, c.h,
                       c.L, c.burn, c.train_range)
const _dupes = let seen = Dict{Any,Int}(), out = Pair{Int,Int}[]
    for (i, c) in enumerate(CELLS_LSTM)
        k = _CELLKEY(c)
        haskey(seen, k) ? push!(out, i => seen[k]) : (seen[k] = i)
    end
    out
end

jldsave(TO_folder * "/inputs_lstm.jld2"; inputs = CELLS_LSTM)
println("wrote $(length(CELLS_LSTM)) configurations to $(TO_folder)/inputs_lstm.jld2")
for (i, c) in enumerate(CELLS_LSTM)
    println("  $(rpad(c.name, 14)) arch=$(rpad(string(c.arch), 8)) h=$(c.h) " *
            "hidden=$(c.n_hidden) latent=$(c.n_latent) enc=$(c.n_encoder) beta=$(c.beta)")
end

if isempty(_dupes)
    println("\nall $(length(CELLS_LSTM)) configurations are distinct")
else
    println("\n$(length(_dupes)) duplicate configuration(s) — the same fit under two names:")
    for (i, j) in _dupes
        println("  StochLSTM$i == StochLSTM$j")
    end
    keep = [i for i in 1:length(CELLS_LSTM) if !(i in first.(_dupes))]
    println("  distinct rows: $(length(keep)) of $(length(CELLS_LSTM)). To skip the re-runs:")
    println("    --array=$(join(keep, ","))")
end
