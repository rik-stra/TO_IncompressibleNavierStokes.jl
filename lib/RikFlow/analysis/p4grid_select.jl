# Stages B and C->D of the Snellius grid (plan step 1+, 2026-09-28): offline selection of the fits,
# then the smoke-gated cap for the mini-D6.
#
#     julia --startup-file=no --project=analysis analysis/p4grid_select.jl [fitroot ...]
#     julia --startup-file=no --project=analysis analysis/p4grid_select.jl --after-smoke [fitroot ...]
#
# `fitroot` = directories whose sub-directories are fits (default exp_square_HIT/output/TO_LSTM/p4grid).
# Every fit is CLASSIFIED FROM ITS OWN FILE (StochLSTM_seed1.jld2 `extras`), not from its name, so
# Agent C's `p4/` fits and the grid's `p4grid/` fits are read the same way:
#   m3f  tools/m4_window_fit.jl, arch :dense   -> h, lambda, wd, nh, seed from `extras.diag`
#   m0v  tools/m4_diag_fit.jl, emission :state_dependent, Ws/V1 frozen
#   m0   tools/m4_linear_eta.jl (closed form; no score of its own -- see below)
# A config = (cell, h, lambda, wd, nh); its seeds are the fits sharing it.
#
# ---- Stage B: the offline rule (declared 2026-09-28) ---------------------------------------------
# Score, held-out 52-74 TU, one step, teacher-forced:
#   m3f  ensemble CRPS in dQ-sd units (`diag.held_crps`). Its matched M0(h, lambda) is `diag.skip0.crps`
#        -- the same frozen ridge skip + seeded eta at update 0, scored by the same estimator (K = 32,
#        rng 2024) -- which IS the M0@50 fit of that (h, lambda) (checked: the M0 dir's Ws against the
#        fit's frozen Ws, printed as `Ws check`).
#   m0v  Gaussian NLL / step (`diag.held_nll`); matched M0(h, 0) = `diag.linear_nll` (linear map + the
#        training residual's full covariance, same rows).
# gain_matched = (M0(h, lambda) - net) and gain_best = (best linear ACROSS h at the same lambda - net):
#   relative (%) for CRPS, nats/step for NLL; > 0 = the network is better.
# 🔑 PASS (offline) = gain_best > 0 for EVERY seed, with >= 3 seeds present. Reported beside it: the
# per-seed values and spread (min / median / max), gain_matched, and gain against the best linear over
# ALL h AND lambda (informational: lambda is the online-stability knob, which no offline score sees).
# Writes <out>/offline_fits.csv, offline_configs.csv and smoke_list.txt = the MEDIAN seed of every
# passing config + its matched M0, one fit dir per line (relative to output/TO_LSTM), for
# batch_scripts/run_p4grid_online.sh.
#
# ---- Stage C -> D: smoke gate and the cap (declared 2026-09-28) ----------------------------------
# `--after-smoke` reads the 20 TU replicas (data_online_tsim20.0_replica<i>.jld2) of every smoke_list
# fit. Reference: the tracked record, t <= 74 TU only; its null = its own 20 TU windows.
#   SMOKE PASS, per config, all of: (1) 3/3 replicas complete (8001 columns) and finite; (2) zero
#   TURBULENCE_GATE firings after the 100-step warm-up (identically-zero dQ columns); (3) the level
#   offset max_k |mean_k - mu_k| / sd_k (median over replicas) no larger than the larger of the
#   reference's own 20 TU windows' and the matched M0's smoke value, plus P4GRID_SMOKE_TOL (0.1 sd).
#   🔴 The smoke is a GATE, never a ranking score: plan §24 rules out selecting on 20 TU free runs.
#   CAP: of the offline-pass AND smoke-pass configs, keep the best per (cell, h, lambda) by median-seed
#   gain_best, then the top P4GRID_MAX_D6 (default 6) by that gain. Writes d6_list.txt: seeds 1-3 of
#   each kept config + their matched M0s (deduplicated), and d6_pairs.csv (config fit -> matched M0),
#   which the collect driver pairs.
#
# Environment: P4GRID_SELOUT (output dir, default analysis/output/p4grid), P4GRID_MAX_D6 (6),
# P4GRID_MIN_SEEDS (3), QOI_CACHE (the tracked record's QoI cache, for --after-smoke).

using JLD2, Statistics, Printf

const HERE = @__DIR__
const TO = normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "TO_LSTM"))
const DT = 2.5e-3
const NSMOKE = round(Int, 20 / DT) + 1
const NWARM = 100
const REF_TU_MAX = 74.0
# the smoke offset gate's tolerance in reference sd: 3 replicas of 20 TU are noisy, and a strict
# "<= matched M0" failed a fit at 0.43 against 0.40 (Agent C's p4, 2026-09-28)
const SMOKE_TOL = parse(Float64, get(ENV, "P4GRID_SMOKE_TOL", "0.1"))

g(x) = @sprintf("%g", x)
relto(p) = (r = relpath(p, TO); startswith(r, "..") ? p : r)     # fit dir as run_p4grid_*.sh want it

"Classify one fit directory from its own file; `nothing` if it is not one of the three cells."
function read_fit(dir)
    f = joinpath(dir, "StochLSTM_seed1.jld2")
    isfile(f) || return nothing
    e = try
        load(f, "extras")
    catch err
        @warn "unreadable fit" dir err
        return nothing
    end
    has(x, k) = x isa NamedTuple && haskey(x, k)
    tr = has(e, :train_range) ? e.train_range : (0, 0)
    ws = has(e, :ps) && has(e.ps, :Ws) ? Float64.(e.ps.Ws) : nothing
    base = (; dir, rel = relto(dir), tag = basename(dir), seed = has(e, :seed) ? Int(e.seed) : 0,
            train_tu = tr .* DT, Ws = ws)
    if has(e, :diag) && has(e.diag, :held_crps)
        ov = e.overrides
        (has(ov, :arch) && ov.arch === :dense) || return nothing
        d = e.diag
        return merge(base, (; cell = "m3f", h = Int(d.h), lambda = Float64(d.lambda), wd = Float64(d.wd),
                            nh = Int(ov.n_hidden), batch = Int(ov.batch), lr = Float64(ov.lr), score = Float64(d.held_crps),
                            matched = Float64(d.skip0.crps), held = Float64(d.held),
                            held_matched = Float64(d.skip0.held), metric = "crps",
                            score_tu = has(d, :score_tu) ? d.score_tu : (NaN, NaN),
                            nlr = has(d, :nlr) ? Float64(d.nlr.net_over_skip) : NaN,
                            stop = has(e, :losses) ? string(e.losses.stop_reason) : "?"))
    elseif has(e, :diag) && has(e.diag, :linear_nll)
        ov = e.overrides
        (has(ov, :emission) && ov.emission === :state_dependent) || return nothing
        d = e.diag
        return merge(base, (; cell = "m0v", h = Int(ov.h), lambda = has(d, :lambda) ? Float64(d.lambda) : 0.0,
                            wd = has(d, :wd) ? Float64(d.wd) : 0.0, nh = Int(ov.n_hidden),
                            batch = Int(ov.batch), lr = Float64(ov.lr),
                            score = Float64(d.held_nll), matched = Float64(d.linear_nll),
                            held = Float64(d.held), held_matched = Float64(d.linear_floor), metric = "nll",
                            score_tu = (NaN, NaN), nlr = NaN,
                            stop = has(e, :losses) ? string(e.losses.stop_reason) : "?"))
    elseif has(e, :lambda) && !has(e, :diag)
        return merge(base, (; cell = "m0", h = Int(e.cfg.h), lambda = Float64(e.lambda), wd = NaN, nh = 0,
                            batch = 0, lr = NaN,
                            score = NaN, matched = NaN, held = NaN, held_matched = NaN, metric = "",
                            score_tu = (NaN, NaN), nlr = NaN, stop = ""))
    end
    return nothing
end

scan(roots) = filter(!isnothing, [read_fit(joinpath(r, d)) for r in roots if isdir(r)
                                  for d in sort(readdir(r)) if isdir(joinpath(r, d))])

"Relative gain of `net` over `ref`: CRPS in % (lower is better), NLL in nats (lower is better)."
gain(metric, ref, net) = metric == "crps" ? 100 * (ref - net) / ref : ref - net

function offline(roots; io = stdout, minseeds = parse(Int, get(ENV, "P4GRID_MIN_SEEDS", "3")))
    fits = scan(roots)
    @printf(io, "scanned %s: %d fits (m3f %d, m0v %d, m0 %d)\n", join(roots, ", "), length(fits),
            count(f -> f.cell == "m3f", fits), count(f -> f.cell == "m0v", fits), count(f -> f.cell == "m0", fits))
    for f in fits
        f.train_tu[2] ≈ 50 || @warn "fit not on the 1-50 TU window" f.tag f.train_tu
        f.cell == "m3f" && !(f.score_tu[1] ≈ 52 && f.score_tu[2] ≈ 74) &&
            @warn "m3f fit not scored on 52-74 TU" f.tag f.score_tu
    end
    nets = filter(f -> f.cell in ("m3f", "m0v"), fits)
    # the linear reference per (metric, h, lambda): every fit of a cell carries its own M0(h, lambda)
    lin = Dict{Tuple{String,Int,Float64},Float64}()
    for f in nets
        k = (f.metric, f.h, f.lambda)
        v = get(lin, k, NaN)
        isnan(v) || abs(v - f.matched) <= 1e-9 * abs(v) ||
            @warn "two fits disagree on M0(h, lambda)" k v f.matched f.tag
        lin[k] = f.matched
    end
    bestlin(metric, lam) = minimum((v for ((m, _, l), v) in lin if m == metric && l == lam); init = Inf)
    bestall(metric) = minimum((v for ((m, _, _), v) in lin if m == metric); init = Inf)
    argbest(metric, lam) = (ks = [(v, h) for ((m, h, l), v) in lin if m == metric && l == lam]; isempty(ks) ? 0 : minimum(ks)[2])
    m0s = Dict((f.h, f.lambda) => f for f in fits if f.cell == "m0")

    println(io, "\n==== linear references M0@50(h, lambda), held-out 52-74 TU")
    for ((m, h, l), v) in sort(collect(lin); by = x -> (x[1][1], x[1][3], x[1][2]))
        m0 = get(m0s, (h, l), nothing)
        chk = if m0 === nothing
            "no M0 dir scanned"
        else
            ws = [f.Ws for f in nets if f.cell == "m3f" && f.h == h && f.lambda == l && f.Ws !== nothing]
            isempty(ws) || m0.Ws === nothing ? "M0 dir $(m0.rel)" :
            @sprintf("M0 dir %s, Ws check max|dWs| = %.1e", m0.rel, maximum(abs, ws[1] .- m0.Ws))
        end
        @printf(io, "  %-4s h=%d  lambda=%-7s %s = %.5f   %s\n", m, h, g(l), m, v, chk)
    end

    fitrows = [merge(f, (; gain_matched = gain(f.metric, f.matched, f.score),
                        gain_best = gain(f.metric, bestlin(f.metric, f.lambda), f.score),
                        gain_all = gain(f.metric, bestall(f.metric), f.score))) for f in nets]
    # batch and lr are in the key so that other recipes in the same root do not pool with the grid's
    key(f) = (f.cell, f.h, f.lambda, f.wd, f.nh, f.batch, f.lr)
    cfgs = []
    for k in sort(unique(key.(fitrows)))
        fs = sort(filter(f -> key(f) == k, fitrows); by = f -> f.seed)
        gb = [f.gain_best for f in fs]
        med = fs[sortperm(gb)[cld(length(gb), 2)]]
        m0 = get(m0s, (k[2], k[3]), nothing)
        push!(cfgs, (; cell = k[1], h = k[2], lambda = k[3], wd = k[4], nh = k[5], nseed = length(fs),
                     seeds = [f.seed for f in fs], gain_best = gb, gain_matched = [f.gain_matched for f in fs],
                     gain_all = [f.gain_all for f in fs], metric = fs[1].metric,
                     best_h = argbest(fs[1].metric, k[3]),
                     dup = !allunique([f.seed for f in fs]),
                     pass = length(fs) >= minseeds && allunique([f.seed for f in fs]) && all(>(0), gb),
                     median_fit = med,
                     matched_m0 = m0 === nothing ? "" : m0.rel, fits = fs))
    end

    println(io, "\n==== Stage B: per config (gain > 0 = network better; CRPS in %, NLL in nats/step)")
    @printf(io, "  %-4s %2s %-7s %-6s %3s %-5s | %-26s | %-26s | %-9s | %s\n", "cell", "h", "lambda", "wd", "nh",
            "seeds", "gain vs best lin across h", "gain vs matched M0(h,lambda)", "vs all", "PASS")
    for c in cfgs
        sp(v) = @sprintf("%+6.2f %+6.2f %+6.2f", minimum(v), median(v), maximum(v))
        @printf(io, "  %-4s %2d %-7s %-6s %3d %-5s | %s (h*=%d) | %s | %+8.2f | %s\n", c.cell, c.h, g(c.lambda),
                isnan(c.wd) ? "" : g(c.wd), c.nh, join(c.seeds, ""), sp(c.gain_best), c.best_h,
                sp(c.gain_matched) * "     ", median(c.gain_all),
                c.pass ? "PASS" : c.dup ? "DUPLICATE SEEDS (two recipes share this key)" :
                c.nseed < minseeds ? "incomplete ($(c.nseed) seeds)" : "fail")
    end
    println(io, "  (min / median / max over seeds; h* = the h of the best linear at that lambda)")
    return (; fits, fitrows, cfgs, lin, m0s)
end

function write_offline(res, out; io = stdout)
    mkpath(out)
    open(joinpath(out, "offline_fits.csv"), "w") do f
        println(f, "cell,h,lambda,wd,nh,seed,metric,score,matched,gain_matched,gain_best,gain_all,held,held_matched,net_over_skip,stop,fitdir")
        for r in res.fitrows
            println(f, join((r.cell, r.h, g(r.lambda), isnan(r.wd) ? "" : g(r.wd), r.nh, r.seed, r.metric,
                             g(r.score), g(r.matched), g(r.gain_matched), g(r.gain_best), g(r.gain_all),
                             g(r.held), g(r.held_matched), g(r.nlr), r.stop, r.rel), ","))
        end
    end
    open(joinpath(out, "offline_configs.csv"), "w") do f
        println(f, "cell,h,lambda,wd,nh,nseed,metric,gain_best_min,gain_best_median,gain_best_max,gain_matched_median,pass,median_fit,matched_m0")
        for c in res.cfgs
            println(f, join((c.cell, c.h, g(c.lambda), isnan(c.wd) ? "" : g(c.wd), c.nh, c.nseed, c.metric,
                             g(minimum(c.gain_best)), g(median(c.gain_best)), g(maximum(c.gain_best)),
                             g(median(c.gain_matched)), c.pass, c.median_fit.rel, c.matched_m0), ","))
        end
    end
    passing = filter(c -> c.pass, res.cfgs)
    lines = unique(vcat([c.median_fit.rel for c in passing], [c.matched_m0 for c in passing if !isempty(c.matched_m0)]))
    open(joinpath(out, "smoke_list.txt"), "w") do f
        foreach(l -> println(f, l), lines)
    end
    @printf(io, "\nwrote %s/{offline_fits,offline_configs}.csv and smoke_list.txt: %d passing config(s), %d smoke dir(s)\n",
            out, length(passing), length(lines))
    isempty(lines) || @printf(io, "  stage C: sbatch --array=1-%d batch_scripts/run_p4grid_online.sh   (3 replicas x %d dirs, 20 TU)\n",
                              3 * length(lines), length(lines))
    return lines
end

# ---- stage C -> D ----------------------------------------------------------------------------------

function reference()
    f = get(ENV, "QOI_CACHE", joinpath(HERE, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
    q = load(f, "q")
    q = q[:, 1:(round(Int, REF_TU_MAX / DT) + 1)]       # 🔒 nothing past 74 TU
    mu, sd = vec(mean(q; dims = 2)), vec(std(q; dims = 2))
    nul = [maximum(abs.((vec(mean(q[:, s:(s + NSMOKE - 1)]; dims = 2)) .- mu) ./ sd))
           for s in 1:(NSMOKE - 1):(size(q, 2) - NSMOKE + 1)]
    return (; mu, sd, null_max = maximum(nul), nnull = length(nul))
end

"Smoke census of one fit dir's 20 TU replicas."
function smoke(dir, ref)
    fs = sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d+\.jld2$", x), readdir(dir)))
    reps = map(fs) do x
        o, nw = jldopen(joinpath(dir, x), "r") do fh
            fh["data_online"], haskey(fh, "nwarm") ? fh["nwarm"] : NWARM
        end
        n = size(o.q, 2)
        ok = n >= NSMOKE && all(isfinite, o.q)
        dq = o.dQ[:, (nw + 1):min(size(o.dQ, 2), NSMOKE - 1)]
        clamp = count(j -> all(iszero, view(dq, :, j)), axes(dq, 2))
        q = o.q[:, 1:min(n, NSMOKE)]
        off = maximum(abs.((vec(mean(q; dims = 2)) .- ref.mu) ./ ref.sd))
        (; file = x, ok, n, clamp, off)
    end
    return (; dir, reps, nrep = length(reps), allok = length(reps) >= 3 && all(r -> r.ok, reps),
            clamp = sum((r.clamp for r in reps); init = 0),
            off = isempty(reps) ? NaN : median([r.off for r in reps]))
end

function after_smoke(res, out; io = stdout, maxd6 = parse(Int, get(ENV, "P4GRID_MAX_D6", "6")))
    ref = reference()
    @printf(io, "\n==== Stage C smoke gate (reference t <= %.0f TU; its %d 20 TU windows: max level offset %.2f sd)\n",
            REF_TU_MAX, ref.nnull, ref.null_max)
    sm = Dict{String,Any}()
    for c in filter(c -> c.pass, res.cfgs), d in (c.median_fit.dir, joinpath(TO, c.matched_m0))
        (isempty(d) || haskey(sm, d) || !isdir(d)) && continue
        sm[d] = smoke(d, ref)
    end
    elig = []
    for c in filter(c -> c.pass, res.cfgs)
        s = get(sm, c.median_fit.dir, nothing)
        m = isempty(c.matched_m0) ? nothing : get(sm, joinpath(TO, c.matched_m0), nothing)
        lim = max(ref.null_max, m === nothing ? -Inf : m.off) + SMOKE_TOL
        pass = s !== nothing && s.allok && s.clamp == 0 && s.off <= lim
        @printf(io, "  %-38s reps %d ok %-5s clamp %4d offset %.2f (limit %.2f; matched M0 %s) -> %s\n",
                c.median_fit.tag, s === nothing ? 0 : s.nrep, s === nothing ? "-" : string(s.allok),
                s === nothing ? 0 : s.clamp, s === nothing ? NaN : s.off, lim,
                m === nothing ? "not run" : @sprintf("%.2f", m.off), pass ? "SMOKE PASS" : "smoke fail")
        pass && push!(elig, c)
    end
    # the cap: best per (cell, h, lambda), then the top maxd6 by median-seed gain vs best linear
    best = Dict{Tuple{String,Int,Float64},Any}()
    for c in elig
        k = (c.cell, c.h, c.lambda)
        (!haskey(best, k) || median(c.gain_best) > median(best[k].gain_best)) && (best[k] = c)
    end
    ranked = sort(collect(values(best)); by = c -> -median(c.gain_best))
    kept = ranked[1:min(maxd6, length(ranked))]
    @printf(io, "\n==== Stage D selection: %d eligible, %d after one-per-(cell, h, lambda), %d kept (cap %d)\n",
            length(elig), length(ranked), length(kept), maxd6)
    lines, pairs = String[], Tuple{String,String}[]
    for c in kept
        @printf(io, "  %-4s h=%d lambda=%-7s wd=%-5s nh=%d  median gain %+.2f  seeds %s\n", c.cell, c.h, g(c.lambda),
                isnan(c.wd) ? "" : g(c.wd), c.nh, median(c.gain_best), join((f.tag for f in c.fits), " "))
        for f in c.fits
            push!(lines, f.rel); push!(pairs, (f.rel, c.matched_m0))
        end
    end
    append!(lines, unique([c.matched_m0 for c in kept if !isempty(c.matched_m0)]))
    lines = unique(lines)
    mkpath(out)
    open(io2 -> foreach(l -> println(io2, l), lines), joinpath(out, "d6_list.txt"), "w")
    open(joinpath(out, "d6_pairs.csv"), "w") do f
        println(f, "fit,matched_m0")
        foreach(p -> println(f, p[1], ",", p[2]), pairs)
    end
    @printf(io, "wrote %s/d6_list.txt (%d closures) and d6_pairs.csv (%d pairs)\n", out, length(lines), length(pairs))
    isempty(lines) || @printf(io, "  stage D: sbatch --array=1-%d batch_scripts/run_p4grid_d6.sh   (%d closures x 2 IC chunks)\n",
                              2 * length(lines), length(lines))
    return (; kept, lines, pairs)
end

function main(args = ARGS; io = stdout)
    aft = "--after-smoke" in args
    roots = filter(a -> !startswith(a, "--"), args)
    isempty(roots) && (roots = [joinpath(TO, "p4grid")])
    out = get(ENV, "P4GRID_SELOUT", joinpath(HERE, "output", "p4grid"))
    res = offline(roots; io)
    write_offline(res, out; io)
    aft && after_smoke(res, out; io)
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
