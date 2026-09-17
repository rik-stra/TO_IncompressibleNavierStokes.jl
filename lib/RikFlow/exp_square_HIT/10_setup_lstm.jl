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
const BASE = (;
    hist_var = :q_star_q,
    include_predictor = true,
    normalization = :normal,
    train_range = (400, 4000),      # t in [1, 10] TU, the same window the M0 cells are fitted on
    n_hidden = 60,                  # the source's dimensions
    n_latent = 60,
    n_encoder = 60,
    uclip = (-4.0, 4.0),            # the L2 head's clip, in log-scale units
    beta = 1e-4,                    # the source's KL weight. NEVER called lambda here.
    L = 200,                        # BPTT segment length
    burn = 50,                      # burn-in, excluded from the loss
    epochs = 300,
    batch = 8,
    lr = 1e-3,
    val_frac = 0.2,
    n_seeds = 5,                    # S6: neural cells are fitted with 5 seeds, median-seed online
    n_replicas = 5,
)

cell(i, arch; kwargs...) = (; name = "StochLSTM$(i)", arch, BASE..., kwargs...)

CELLS_LSTM = [
    # --- 1-4: the four architectures at h = 1. This IS the experiment M4 exists for: Sørensen et
    # al. report that the two with upstream stochasticity (STORN, VRNN) overfit less than the
    # deterministic LSTM and the output-only VAE-RNN, which is a claim on S1.
    cell(1, :lstm),
    cell(2, :vaernn),
    cell(3, :storn),
    cell(4, :vrnn),

    # --- 5-8: the same four at M0's h, so the regressor and the data budget match the linear cells
    cell(5, :lstm;   h = 5),
    cell(6, :vaernn; h = 5),
    cell(7, :storn;  h = 5),
    cell(8, :vrnn;   h = 5),

    # --- 9-10: beta sensitivity on the best upstream architecture. beta = 0 removes the KL
    # entirely, which turns STORN into a deterministic model with noise injection and is the
    # control for "did the variational part do anything".
    cell(9, :vrnn;  beta = 0.0),
    cell(10, :vrnn; beta = 1e-2),

    # --- 11: the tex's lean form -- a linear encoder and a latent as wide as the output. Cheapest
    # on S4 by a factor of three (see tools/m4_cost_probe.jl) and worth knowing if it suffices.
    cell(11, :vrnn; n_encoder = 0, n_latent = 6),
]

# `h` is in BASE only implicitly -- set it explicitly so every row records one.
CELLS_LSTM = [haskey(c, :h) ? c : (; c..., h = 1) for c in CELLS_LSTM]

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
