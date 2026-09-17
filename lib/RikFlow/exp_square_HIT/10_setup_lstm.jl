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
# to carry the memory, which is the whole reason for using one -- but rows 5-8 repeat the four
# architectures at M0's `h = 5` so the data budget can be compared like for like.
# 🔴 **These are NOT the source's dimensions, deliberately.** At hidden/latent/encoder = 60 the
# model has 43 128 parameters against 21 594 training target values -- twice as many parameters as
# data. Their 60 units predict a quasi-geostrophic *field*; we predict six scalars. See
# `analysis/results_LSTMS.md` §8.1.
const BASE = (;
    hist_var = :q_star_q,
    include_predictor = true,
    normalization = :normal,
    train_range = (400, 4000),      # t in [1, 10] TU, the same window the M0 cells are fitted on
    h = 1,                          # §8.6 Q4: an h = 5 lag window is 4% of one decay time
    n_hidden = 16,
    n_latent = 4,
    n_encoder = 0,                  # the tex's linear encoder, not the repository's dense layer
    emission = :none,               # §8.6 Q1: the latent path is the ONLY noise source
    uclip = (-4.0, 4.0),            # unused when emission = :none; kept for the :constant rows
    beta = 1e-4,                    # the source's KL weight. NEVER called lambda here.
    # 🔑 L is ONE RINGING PERIOD of the level's ACF (~1 TU = 400 steps) and burn is past its 1/e
    # crossing (116-142 steps on every band). Sized from the crossings, NOT from T_int, because
    # the level's ACF rings rather than decaying -- `results.md` §1, `results_LSTMS.md` §8.5.
    L = 400,
    burn = 150,
    epochs = 300,
    batch = 8,
    lr = 1e-3,
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

    # --- 8-9: the latent dimension, still open (§8.6 Q3). Run after beta is settled.
    cell(8, :vrnn; n_latent = 6),
    cell(9, :vrnn; n_latent = 8),

    # --- 10: the output-only variant. The one their result argues AGAINST, kept so the claim can
    # be tested rather than assumed, but not in the first pass.
    cell(10, :vaernn; emission = :constant),
]

for (i, c) in enumerate(CELLS_LSTM)
    c.name == "StochLSTM$(i)" ||
        error("row $i is named $(c.name): the index IS the identity, so the table is out of step")
end

jldsave(TO_folder * "/inputs_lstm.jld2"; inputs = CELLS_LSTM)
println("wrote $(length(CELLS_LSTM)) configurations to $(TO_folder)/inputs_lstm.jld2")
for (i, c) in enumerate(CELLS_LSTM)
    println("  $(rpad(c.name, 14)) arch=$(rpad(string(c.arch), 8)) h=$(c.h) " *
            "hidden=$(c.n_hidden) latent=$(c.n_latent) enc=$(c.n_encoder) beta=$(c.beta)")
end
