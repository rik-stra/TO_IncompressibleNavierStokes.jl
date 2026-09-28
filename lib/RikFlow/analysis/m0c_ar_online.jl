# M0ᶜ-ridge online checks on mini-D6 directories (results_LSTMS §12), 2026-09-28.
#
#     julia --startup-file=no --project=training analysis/m0c_ar_online.jl <dir> [<dir> ...]
#
# Per directory (relative to analysis/output or absolute):
#  (a) the run's metadata census: members, diverged, warm-up bit-identity (every member), gate
#      firings, recorded AR order/φ, seeds;
#  (b) online dQ ACF over the forecast columns (nwarm+1:end) per QoI, averaged over members, against
#      the tracked record on the same steps (m0c_ridge_colour.jl check3), and sd(dQ) online/tracked.
#      Only ICs whose run ends by 74 TU are read for (b) (🔒, guard());
#  (c) `--vs-full <dir>`: the first nwarm+nlead dQ columns against an existing full-
#      length D6 run of the same model and member seeds (same draws => round-off agreement; only ICs
#      ending by 74 TU in the full run are read).

include(joinpath(@__DIR__, "m0c_checks.jl"))       # acf, fmt, guard, DQR, LABELS, T_MAX_READ

resolve(d) = isabspath(d) ? d : joinpath(HERE, "output", d)
members(dir) = sort(filter(f -> occursin(r"^d6_online_ic\d+_m\d+\.jld2$", f), readdir(dir)))

function census(dir; io = stdout)
    fs = members(dir)
    nd = 0; nwi = 0; ng = 0; nsteps = 0; ks = Set{Int}(); ars = Set{Any}(); seeds = Dict{Tuple{Int,Int},UInt64}()
    for f in fs
        d = jldopen(joinpath(dir, f), "r") do fh
            (; diverged = fh["diverged"], warm = fh["warm_identical"], gate = fh["gate_nfired"],
             nlead = fh["nlead"], k = fh["k"], member = fh["member"], seed = fh["seed"],
             ar = haskey(fh, "ar_order") ? (fh["ar_order"], round.(fh["ar_phi"]; digits = 3)) : "(no key)",
             model = fh["model_name"])
        end
        nd += d.diverged; nwi += d.warm; ng += d.gate; nsteps += d.nlead
        push!(ks, d.k); push!(ars, (d.model, d.ar)); seeds[(d.k, d.member)] = d.seed
    end
    @printf(io, "%s: %d member files, %d ICs, diverged %d, warm-up bit-identical %d/%d, gate firings %d of %d forecast steps\n",
            basename(dir), length(fs), length(ks), nd, nwi, length(fs), ng, nsteps)
    for a in ars
        println(io, "  model / AR record: ", a)
    end
    return (; n = length(fs), nic = length(ks), diverged = nd, warm = nwi, gate = ng, seeds)
end

function online_acf(dir; io = stdout, lags = (1, 2, 5, 10, 20))
    on = [zeros(length(lags)) for _ in 1:NQ]; tr = [zeros(length(lags)) for _ in 1:NQ]
    sdr = zeros(NQ); n = 0; nic = Set{Int}()
    for f in members(dir)
        d = load(joinpath(dir, f))
        (d["t_k"] + d["tsim"] <= T_MAX_READ) || continue
        d["diverged"] && continue
        nw, nk = d["nwarm"], d["n_k"]
        cols = (nw + 1):size(d["dQ"], 2)
        tcols = guard(nk .+ cols)
        for i in 1:NQ
            on[i] .+= acf(d["dQ"][i, cols], lags)
            tr[i] .+= acf(DQR[i, tcols], lags)
            sdr[i] += std(d["dQ"][i, cols]) / std(DQR[i, tcols])
        end
        n += 1; push!(nic, d["k"])
    end
    @printf(io, "  online dQ ACF, forecast columns only, %d members / %d ICs ending by %g TU; lags %s\n", n,
            length(nic), T_MAX_READ, join(lags, " "))
    @printf(io, "    %-9s %-31s %-31s %s\n", "QoI", "online dQ ACF", "tracked dQ ACF (same steps)", "sd(dQ) on/trk")
    rows = []
    for i in 1:NQ
        @printf(io, "    %-9s %s   %s   %.2f\n", LABELS[i], fmt(on[i] ./ n), fmt(tr[i] ./ n), sdr[i] / n)
        push!(rows, (; qoi = LABELS[i], online = on[i] ./ n, tracked = tr[i] ./ n, sd_ratio = sdr[i] / n))
    end
    return rows
end

"""
Against an existing full-length D6 run of the same model and member seeds: the dQ over the mini
run's columns. Bitwise identity is not expected across runs on the GPU (the IC's own recomputed QoIs
already differ at ~1e-13 between two runs); what this checks is that the SAME draws were made: a
different RNG stream would show O(1) relative differences from the first forecast column.
"""
function vs_full(dir, full; io = stdout)
    neq = 0; ntot = 0; nskip = 0; worst = 0.0; worst_q0 = 0.0
    for f in members(dir)
        isfile(joinpath(full, f)) || continue
        tfull = jldopen(fh -> (fh["t_k"], fh["tsim"]), joinpath(full, f), "r")
        if tfull[1] + tfull[2] > T_MAX_READ
            nskip += 1
            continue
        end
        a = load(joinpath(dir, f)); b = load(joinpath(full, f))
        b["seed"] == a["seed"] || error("$f: seeds differ ($(a["seed"]) vs $(b["seed"]))")
        nc = size(a["dQ"], 2)
        ntot += 1
        neq += a["dQ"] == b["dQ"][:, 1:nc]
        worst = max(worst, maximum(abs.(a["dQ"] .- b["dQ"][:, 1:nc]) ./ (abs.(b["dQ"][:, 1:nc]) .+ 1e-30)))
        worst_q0 = max(worst_q0, maximum(abs.(a["q"][:, 1] .- b["q"][:, 1]) ./ abs.(b["q"][:, 1])))
    end
    @printf(io, "  vs %s (same seeds): %d of %d members bitwise equal; max rel |ΔdQ| over all columns %.1e (IC QoIs q[:,1] already differ by %.1e); %d skipped (full run past %g TU)\n",
            basename(full), neq, ntot, worst, worst_q0, nskip, T_MAX_READ)
    return (; neq, ntot, worst, worst_q0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    full = nothing
    if (i = findfirst(==("--vs-full"), args)) !== nothing
        full = resolve(args[i + 1]); deleteat!(args, i:(i + 1))
    end
    res = Dict{String,Any}()
    for d in args
        dir = resolve(d)
        c = census(dir)
        r = online_acf(dir)
        v = full === nothing ? nothing : vs_full(dir, full)
        res[basename(dir)] = (; census = (; c.n, c.nic, c.diverged, c.warm, c.gate), acf = r, vs_full = v)
        println()
    end
    jldsave(joinpath(HERE, "output", "m0c_ar_online.jld2"); res)
end
