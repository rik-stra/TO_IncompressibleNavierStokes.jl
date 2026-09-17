# Shared fixtures.
#
# `TSLayer` includes the `ts_*.jl` sources bare. That works because they are stdlib-only by design
# and it is what keeps this suite runnable without IncompressibleNavierStokes or CUDA.
#
# `TrackedData` finds the extracted QoI caches under `analysis/data/`. Those are the 3.66 MB
# products of `analysis/extract_qois.jl`; the 1.3 GB tracking files they came from are not in this
# checkout, and neither is the rest of paper 2's archive. Tests that need a record ask
# `hit10()`/`hit100()`/`channel()` and skip themselves when it is absent, so the suite passes on a
# clean clone and gets stronger where the data is present.

@testmodule TSLayer begin
    using LinearAlgebra
    using Statistics
    using Random

    const SRC = normpath(joinpath(@__DIR__, "..", "src"))
    include(joinpath(SRC, "ts_scaling.jl"))
    include(joinpath(SRC, "ts_history.jl"))
    include(joinpath(SRC, "ts_models.jl"))
    # M4's inference core. It is in this list, and not behind a Lux guard, because that is the
    # entire point of the split: `lstm_step!` is what runs inside the solver, so it is what the
    # verification matrix has to be able to reach. Training lives in `ext/RikFlowLuxExt.jl` and is
    # not loaded here.
    include(joinpath(SRC, "ts_lstm.jl"))
    include(joinpath(SRC, "ts_lstm_online.jl"))
    include(joinpath(SRC, "ts_fit.jl"))
    include(joinpath(SRC, "ts_score.jl"))
    include(joinpath(SRC, "ts_rollout.jl"))
    include(joinpath(SRC, "ts_spectrum.jl"))
end

@testmodule TrackedData begin
    using JLD2

    const DATA = normpath(joinpath(@__DIR__, "..", "analysis", "data"))

    """
        load_cache(pattern)

    Load one extracted QoI cache whose file name contains `pattern`, or `nothing` when no such
    file is present. Returns `(; q, q_star, dQ, tau, name)`.

    ⚠️ Two archive key names for the same object -- HIT writes `data_track`, the channel
    `data_train` -- but `extract_qois.jl` has already normalised that away and stored the key it
    found under `"key"`, so this loader reads flat names.

    🔴 **An ambiguous pattern raises; it does not pick one.** This used to return
    `first(sort(hits))`, and `hit100()`'s pattern `"tsim100.0"` matched **eight** files in
    `analysis/data/` -- the tracked record, the HF reference and six `online_*` ensembles. It
    returned the right one only because `data_track2_...` happens to sort before `hf_reference_...`
    and `online_...`. Rename a file or add a record and a test asking for the tracked record would
    silently have been handed an online ensemble, which is a different object entirely and would
    not have looked wrong. Erroring is the whole point: a test that cannot find its data skips, but
    a test that is handed the wrong data passes.
    """
    function load_cache(pattern::AbstractString)
        isdir(DATA) || return nothing
        hits = filter(f -> occursin(pattern, f) && endswith(f, "_qois.jld2"), readdir(DATA))
        isempty(hits) && return nothing
        length(hits) == 1 || error("load_cache(\"$pattern\") is ambiguous: it matches " *
                                   "$(length(hits)) caches, $(sort(hits)). Narrow the pattern.")
        p = joinpath(DATA, only(hits))
        d = load(p)
        return (; q = d["q"], q_star = d["q_star"], dQ = get(d, "dQ", nothing),
                tau = get(d, "tau", nothing), name = basename(p))
    end

    # Each pattern is pinned to exactly one cache. `hit10` is the 4 000-step record V1, G1 and the
    # spectrum checks all run on -- small on purpose, so the suite stays fast; `hit100` is the
    # 40 000-step record D6's truth and `score_m0_ddn.jl` come from.
    hit10() = load_cache("data_track_trackingnoise_std_0.0")
    hit100() = load_cache("data_track2")

    # 🔴 R1's tracking record -- the one D6's IC packages are cut from since 2026-09-15. NOT
    # `hit100()`: that is paper 2's archived `data_track2`, Float32 and on the pre-`09954be1`
    # Nyquist convention, i.e. a different dynamical system (memory #45, #46, #60). A D6 test
    # comparing a package built from R1 against `hit100()` fails on every array, which is exactly
    # what it should do.
    hitR1() = load_cache("data_track_dns512")

    # ⚠️ The channel tracked record: extracted 2026-09-05 and **called by nothing**, four days
    # before the HIT-only focus (`claude_memory.md` decisions log, 2026-09-09). Kept, not deleted:
    # it is 386 kB in a gitignored directory and remaking it means another 39 MB read of a file that
    # lives outside the repository, and the channel archive is complete and waiting for the focus
    # to lift. 🔑 Worth knowing when it does: this record is **Float64** where both HIT records are
    # Float32, so its coefficient-level diagnostics -- `rho(Ctilde)`, the H-infinity gain -- are
    # trustworthy in a way HIT's provably are not (gotcha #26).
    channel() = load_cache("LF_6qoi_track_channel")

    """
        train_slices(rec, train_range)

    The exact arrays the archived training scripts build, and the pair `build_history` must be fed
    to match them.

    The scripts slice `q` twice -- once as the history stream `q[:, a:b-1]` and once as the target
    `q[:, a+1:b]` -- so the two are one column apart and `build_history`'s `(q_star, q)` argument
    pair is `(q_star[:, a:b-1], q[:, a:b])`. Getting this alignment wrong is exactly the off-by-one
    V1 exists to catch, so it is written once, here.
    """
    function train_slices(rec, train_range)
        a, b = train_range
        return (; q_hist = rec.q[:, a:(b - 1)],          # scripts' `q_scaled`
                q_star = rec.q_star[:, a:(b - 1)],       # scripts' `q_star_scaled`
                target = rec.q[:, (a + 1):b],            # scripts' `dQ_scaled` (the level, shifted)
                q_bh = rec.q[:, a:b],                    # `build_history`'s `q`, one column longer
                q_star_bh = rec.q_star[:, a:(b - 1)])    # `build_history`'s `q_star`
    end
end

# ---------------------------------------------------------------------------------------------
# the archived history builders, verbatim
# ---------------------------------------------------------------------------------------------

# The five copies of the history construction that `build_history` replaces, transcribed unchanged
# from the scripts so that V1 compares code against code and never against a document's assertion
# about the code. `plan.md` §8a is explicit about why: a synthetic system built with an assumed
# index convention confirms whichever convention was assumed.
#
# - `hit`, `chan`, `solvers` -- `exp_square_HIT/5_train_LinReg.jl:22,34`,
#   `channel/5_train_LinReg.jl:22,34`, `time_solvers/train_linreg.jl:18,30`. Character-identical to
#   each other; kept as three names so a future divergence is caught rather than assumed away.
# - `tg` -- `taylor-green/7_train_LinReg.jl:23,40`. Takes a `scaling` argument and **drops**
#   columns where any predictor QoI is small, which is P0.6's contiguity failure and is not
#   interchangeable with the others.
# - `online` -- the fifth, inline copy at `time_series_methods.jl:133-146`, as a function of the
#   ring buffer state.
@testmodule Legacy begin

    # --- exp_square_HIT/5_train_LinReg.jl:22,34 -------------------------------------------------
    function hit(hist_len, q_star, q, dQ; include_predictor = true)
        if hist_len == 0
            return q_star, dQ
        end
        qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
        if include_predictor
            return vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
        else
            return vcat(qs...), dQ[:,hist_len:end]
        end
    end

    function hit(hist_len, q_star, q, dQ, hist_var; include_predictor = true)
        if hist_var == :q
            inputs,outputs = hit(hist_len, q_star[:,:], q[:,:], dQ[:,:]; include_predictor)
        elseif hist_var == :q_star
            inputs,outputs = hit(hist_len, q_star[:,2:end], q_star[:,1:end-1], dQ[:,2:end]; include_predictor)
        elseif hist_var == :q_star_q
            inputs,outputs = hit(hist_len, q_star[:,2:end], cat(q[:,2:end],q_star[:,1:end-1],dims = 1), dQ[:,2:end]; include_predictor)
        end
        return inputs,outputs
    end

    # --- channel/5_train_LinReg.jl:22,34 -------------------------------------------------------
    chan(args...; kwargs...) = hit(args...; kwargs...)

    # --- time_solvers/train_linreg.jl:18,30 ---------------------------------------------------
    solvers(args...; kwargs...) = hit(args...; kwargs...)

    # --- taylor-green/7_train_LinReg.jl:23,40 -------------------------------------------------
    function tg(hist_len, q_star, q, dQ, scaling; include_predictor = true)
        if hist_len == 0
            inp, target = q_star, dQ
        else
            qs = [q[:,hist_len-i+1:end-i+1] for i in 1:hist_len]
            if include_predictor
                inp, target = vcat(q_star[:,hist_len:end], qs...), dQ[:,hist_len:end]
            else
                inp, target = vcat(qs...), dQ[:,hist_len:end]
            end
        end
        # remove data points where any of q_star = 0
        inp = inp[:,reshape(all( abs.(q_star[:,max(hist_len,1):end].*scaling.in_scaling.sigma) .> 0.5e-2, dims=1),:)]
        target = target[:,reshape(all( abs.(q_star[:,max(hist_len,1):end].*scaling.in_scaling.sigma) .> 0.5e-2, dims=1),:)]
        return inp, target
    end

    function tg(hist_len, q_star, q, dQ, hist_var, scaling; include_predictor = true)
        if hist_var == :q
            inputs,outputs = tg(hist_len, q_star[:,:], q[:,:], dQ[:,:], scaling; include_predictor)
        elseif hist_var == :q_star
            inputs,outputs = tg(hist_len, q_star[:,2:end], q_star[:,1:end-1], dQ[:,2:end], scaling; include_predictor)
        elseif hist_var == :q_star_q
            inputs,outputs = tg(hist_len, q_star[:,2:end], cat(q[:,2:end],q_star[:,1:end-1],dims = 1), dQ[:,2:end], scaling; include_predictor)
        end
        return inputs,outputs
    end

    # --- time_series_methods.jl:133-146, the inline online copy --------------------------------
    """
        online(q_hist, q_star, n_qoi, hist_var, include_predictor)

    The regressor row the deployed `LinReg` forms, transcribed from `get_next_item_timeseries`.
    `q_hist` is the ring buffer in its unscaled form, column 1 most recent; for `:q_star_q` its
    rows `1:n_qoi` hold the corrected QoIs and `n_qoi+1:end` the predictors.

    Scaling is left out on purpose: the deployed code applies `scale_input` to `q_star` and to each
    half of `q_hist` with the *same* `in_scaling`, which is per-QoI and therefore commutes with the
    stacking. What V1 has to pin down is the index layout, so the layout is compared unscaled.

    ⚠️ The `include_predictor = false` branch is transcribed with its defect intact: the deployed
    code assigns `input = q_hist_sc` without the `[:]` that the other branch applies, so for
    `h > 1` the row is a matrix and the construction is wrong. That is IM-17's first defect, and it
    is reproduced rather than repaired because V1's job is to record what the code does. Only the
    `include_predictor = true` path is asserted equal to `inputvec`.
    """
    function online(q_hist, q_star, n_qoi, hist_var, include_predictor)
        if hist_var == :q_star_q
            q_hist_sc = cat(q_hist[1:n_qoi,:], q_hist[n_qoi+1:end,:], dims = 1)
        else
            q_hist_sc = q_hist
        end
        if include_predictor
            input = vcat(q_star, q_hist_sc[:])
        else
            input = q_hist_sc          # verbatim: no `[:]`, which is the IM-17 defect
        end
        return vcat(input, ones(eltype(input), (1, 1)))
    end
end
