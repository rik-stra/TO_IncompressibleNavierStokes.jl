# Stage 0 of the Snellius grid (plan step 1+, 2026-09-28): calibrate the M4-path skip λ against the
# deployed LinReg λ ladder. CPU, local, ~1 min.
#
#     julia --startup-file=no --project=training analysis/skip_lambda_calibration.jl
#
# Why: the M4 path's ridge (`m4_linear_eta.jl`, `m4_window_fit.jl` RIKFLOW_W_LAMBDA) and `LinReg`'s
# ridge use different conventions (plan §3, "Stabilising the deep cells", Start here TODO-0), so a
# numerical λ means different things on the two paths:
#
#   M4 path : dQ target (standardised by its own 1-50 TU scaling), design standardised by q's
#             scaling, penalty `λ N` on every coefficient but the bias (N = fit rows, ~15.7k).
#   LinReg  : LEVEL target q^n (q's scaling), same design, penalty `λ` (not λN) on every coefficient
#             but the bias (`penalize_intercept = false`), fitted on 1-10 TU (N ~ 3.6k).
#
# So this script puts both on common axes, for each skip h in `CALIB_H` (default 2,3,5):
#   * effective degrees of freedom tr(X (X'X + P)^{-1} X'), exact, from the augmented QR (the Gram
#     matrix has cond ~1e12, so the normal equations are not used for it), and the penalised
#     fraction (dof - 1) / (n_in - 1) that is comparable across h;
#   * the skip-only one-step residual's ACF at lags 1, 2, 5 per QoI on held-out 52-74 TU;
#   * the one-step held-out loss 0.5 SSE / row in dQ-sd units (m4_linear_eta's "held mean");
#   * rho(C~), the lifted companion's spectral radius (metric #21, `RikFlow.rho`), with the dQ map
#     converted to the level map in q-scaled units (q = q* + dQ).
# and the same statistics for the deployed LinReg ladder at h = 5 (LinReg1/5/6/2/7/8, λ = 0...10),
# evaluated exactly as deployed (as `m0c_ridge.jl`: per-QoI input scaling, c [x; 1], level target).
#
# Then, per h, the M4-path λ that matches LinReg λ = 1e-4, 1e-2, 1, 10 on (a) the penalised dof
# fraction and (b) the mean held-out residual lag-1, and a PROPOSED 3-point λ ladder per h:
# 0, the LinReg2 (λ = 1e-2) equivalent and the LinReg7 (λ = 1) equivalent, both by dof, rounded to
# one significant digit -- printed as a TOML snippet for `batch_scripts/p4grid/spec.toml`.
#
# ⚠️ Neither match is an online equivalence. The closed-loop bias zero-crossing (λ ~ 1e-5 at h = 1,
# `results_LSTMS.md` §7d) needs online runs: it is a stage-1 smoke item, not computed here.
#
# 🔒 Nothing past 74 TU is read: the record is truncated to t <= 74 TU right after loading.
#
# Environment: CALIB_H (default "2,3,5"), CALIB_LAMS (comma list; default 0 + 12 log-spaced
# 1e-9...1e-2), CALIB_OUT (a .jld2 to write, default none).

using RikFlow, JLD2, Statistics, LinearAlgebra, Printf
const RF = RikFlow
include(normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "tools", "m4_data.jl")))

const DT = 2.5e-3
const NQ = 6
const LABELS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const T_MAX_READ = 74.0
const HELD_TU = (52.0, 74.0)
const LAGS = (1, 2, 5)
const TO = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LSTM"))
const LRS = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output", "TO_LRS"))
const LADDER = ["LinReg1", "LinReg5", "LinReg6", "LinReg2", "LinReg7", "LinReg8"]
const TARGETS = (("LinReg6", 1e-4), ("LinReg2", 1e-2), ("LinReg7", 1.0), ("LinReg8", 10.0))

acf(x, L) = (y = x .- mean(x); v = sum(abs2, y); [sum(y[1:(end - l)] .* y[(1 + l):end]) / v for l in L])
fmtv(v; d = 2) = join((@sprintf("%6.*f", d, x) for x in v), "")

"Exact effective dof tr(X (X'X + P)^{-1} X') for rows-as-samples `X` (N x m), diagonal penalty `p`."
function eff_dof(X::AbstractMatrix, p::AbstractVector)
    A = vcat(X, Diagonal(sqrt.(p)))
    Q = Matrix(qr(A).Q)                   # thin, (N + m) x m
    return sum(abs2, view(Q, 1:size(X, 1), :))
end

"The QoI record, truncated so that nothing past `T_MAX_READ` TU is ever used."
function load_truncated()
    rec = load_m4_qois(; track_file = "none")
    n = round(Int, T_MAX_READ / DT)
    return (; q = rec.q[:, 1:(n + 1)], q_star = rec.q_star[:, 1:n], dQ = rec.dQ[:, 1:n],
            source = rec.source)
end

"""
The M4 path's closed-form skip at skip-history `h`, as `m4_linear_eta.jl` builds it (the same
design, scaling, 80% training rows and `λ N` penalty), for every λ in `lams`.
"""
function m4_path(rec, base, h, lams)
    cfg = merge(base, (; arch = :lstm, emission = :constant, h, beta = 0.0,
                       train_range = (base.train_range[1], round(Int, 50 / DT))))
    cfg.hist_var === :q_star_q || error("expected hist_var = :q_star_q, got $(cfg.hist_var)")
    hist = RF.HistorySpec(; h, n_qoi = NQ, cfg.hist_var, cfg.include_predictor)
    dat = m4_training_data(rec, cfg, hist; target = :dQ)
    ntr = floor(Int, 0.8 * size(dat.Xc, 2))
    X, Y = Float64.(dat.Xc[:, 1:ntr]), Float64.(dat.Yc[:, 1:ntr])
    b = size(rec.q_star, 2); a = cfg.train_range[1]
    qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling)
    qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
    Xf, _, st = RF.build_history(hist, qss, qs); cols = (a - 1) .+ st
    Yf = RF.scale_input(rec.dQ[:, cols], dat.scaling.out_scaling); Xf = permutedims(Xf)
    H = findall(c -> HELD_TU[1] / DT <= c <= HELD_TU[2] / DT, cols)
    all(diff(cols[H]) .== 1) || error("held-out rows are not contiguous")
    first(cols[H]) > cfg.train_range[2] || error("held-out window overlaps the training range")
    nin = size(X, 1)
    sq, sd = vec(dat.scaling.in_scaling.sigma), vec(dat.scaling.out_scaling.sigma)
    qsc = RF.qstar_columns(hist)
    rows = map(lams) do lam
        p = [fill(lam * ntr, nin - 1); 0.0]
        C = lam == 0 ? X' \ Y' : (X * X' + Diagonal(p)) \ (X * Y')     # n_in x N_Q, as m4_linear_eta
        rh = Yf[:, H] .- C' * Xf[:, H]
        dof = lam == 0 ? Float64(nin) : eff_dof(permutedims(X), p)
        # level map in q-scaled units: q = q* + dQ  =>  C_q = C diag(sd_dQ / sd_q) + [I at q*_n]
        Cq = C .* (sd ./ sq)'
        qsc === nothing || (Cq[qsc, :] .+= Matrix{Float64}(I, NQ, NQ))
        (; lam, dof, frac = (dof - 1) / (nin - 1), held = 0.5 * sum(abs2, rh) / size(rh, 2),
         acf = reduce(hcat, [acf(rh[i, :], LAGS) for i in 1:NQ]),     # length(LAGS) x NQ
         rho = RF.rho(Cq, hist), maxC = maximum(abs, C[1:(end - 1), :]))
    end
    return (; h, nin, ntr, rows, sd_dQ = sd, held_rows = length(H),
            held_tu = (cols[H[1]] * DT, cols[H[end]] * DT))
end

"The deployed LinReg ladder at its own h, as deployed (m0c_ridge.jl's `deployed_resid`)."
function linreg_ladder(rec, sd_dQ_ref)
    out = []
    for name in LADDER
        f = joinpath(LRS, name, "LinReg.jld2")
        isfile(f) || (println("  (skipping $name: no $f)"); continue)
        m = load(f)
        p = load(joinpath(LRS, name, "parameters.jld2"))["parameters"]
        @assert m["hist_var"] == :q_star_q && m["include_predictor"] && m["fitted_qois"] == collect(1:NQ)
        h = m["hist_len"]
        lam = hasproperty(p, :lambda) ? p.lambda : 0.0
        pen_int = hasproperty(p, :penalize_intercept) ? p.penalize_intercept : false
        tr = hasproperty(p, :train_range) ? p.train_range : (400, 4000)
        hist = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
        si, so = m["scaling"].in_scaling, m["scaling"].out_scaling
        mu, sg = vec(si.mu), vec(si.sigma)
        c = Matrix{Float64}(m["c"])
        size(c, 1) == NQ || (c = permutedims(c))                         # N_Q x n_in
        function design(ns)
            a, b = first(ns) - h, last(ns)
            X, Y, st = RF.build_history(hist, rec.q_star[:, a:b], rec.q[:, a:(b + 1)])
            @assert st .+ (a - 1) == collect(ns)
            Xs = copy(X)
            for j in 1:(size(X, 2) - 1)
                i = mod1(j, NQ)
                Xs[:, j] = (X[:, j] .- mu[i]) ./ sg[i]
            end
            return Xs, Y
        end
        Xtr, _ = design((tr[1] + h):(tr[2] - 1))
        nin = size(Xtr, 2)
        pvec = [fill(Float64(lam), nin - 1); pen_int ? Float64(lam) : 0.0]
        dof = lam == 0 ? Float64(nin) : eff_dof(Xtr, pvec)
        Xh, Yh = design(round(Int, HELD_TU[1] / DT):round(Int, HELD_TU[2] / DT))
        pred = (c * Xh')' .* vec(so.sigma)' .+ vec(so.mu)'
        R = (Yh .- pred) ./ sd_dQ_ref'                    # physical residual in dQ-sd units
        push!(out, (; name, h, lam, dof, frac = (dof - 1) / (nin - 1), nin, ntr = size(Xtr, 1),
                    held = 0.5 * sum(abs2, R) / size(R, 1),
                    acf = reduce(hcat, [acf(R[:, i], LAGS) for i in 1:NQ]),
                    rho = RF.rho(permutedims(c), hist), maxC = maximum(abs, c[:, 1:(end - 1)])))
    end
    return out
end

"log-λ interpolation of the λ where `f(row)` crosses `target`, over rows with λ > 0 (monotone assumed)."
function match_lambda(rows, f, target)
    rs = filter(r -> r.lam > 0, rows)
    v = f.(rs); l = log10.(getfield.(rs, :lam))
    for k in 1:(length(rs) - 1)
        if (v[k] - target) * (v[k + 1] - target) <= 0 && v[k] != v[k + 1]
            w = (target - v[k]) / (v[k + 1] - v[k])
            return 10^(l[k] + w * (l[k + 1] - l[k]))
        end
    end
    return (target - v[1]) * (v[end] - v[1]) < 0 ? -Inf : Inf       # below / above the grid
end
round1(x) = !isfinite(x) || x <= 0 ? x : (e = floor(log10(x)); round(x / 10^e) * 10^e)
lamstr(x) = x == -Inf ? "< grid" : x == Inf ? "> grid" : @sprintf("%.2e", x)

function main(; io = stdout)
    hs = parse.(Int, split(get(ENV, "CALIB_H", "2,3,5"), ","))
    lams = let s = strip(get(ENV, "CALIB_LAMS", ""))
        isempty(s) ? [0.0; 10.0 .^ range(-9, -2; length = 12)] : parse.(Float64, split(s, ","))
    end
    base = load(joinpath(TO, "inputs_lstm.jld2"), "inputs")[2]
    rec = load_truncated()
    @printf(io, "QoI source %s, truncated to t <= %.0f TU (%d columns)\n", rec.source, T_MAX_READ,
            size(rec.q_star, 2))
    m4 = Dict(h => m4_path(rec, base, h, lams) for h in hs)
    ref_sd = m4[maximum(hs)].sd_dQ
    lr = linreg_ladder(rec, ref_sd)

    hdr = @sprintf("  %-10s %7s %6s %9s %8s %7s | %s\n", "λ", "dof", "frac", "held SSE", "rho(C~)",
                   "max|C|", "held residual ACF lag 1 / 2 / 5, per QoI: " * join(LABELS, " "))
    row(lbl, r) = @sprintf("  %-10s %7.2f %6.3f %9.4f %8.4f %7.1f | %s\n", lbl, r.dof, r.frac, r.held,
                           r.rho, r.maxC,
                           join((@sprintf("%5.2f/%5.2f/%5.2f", r.acf[1, i], r.acf[2, i], r.acf[3, i])
                                 for i in 1:NQ), " "))
    for h in hs
        m = m4[h]
        @printf(io, "\n==== M4-path skip, h = %d: n_in %d, %d fit rows (1-50 TU, first 80%%), held %d rows %.2f-%.2f TU\n",
                h, m.nin, m.ntr, m.held_rows, m.held_tu...)
        print(io, hdr)
        for r in m.rows
            print(io, row(@sprintf("%.2e", r.lam), r))
        end
    end
    println(io, "\n==== deployed LinReg ladder (fit 1-10 TU, penalty λ on the scaled design, level target)")
    print(io, hdr)
    for r in lr
        print(io, row(@sprintf("%s %g", r.name, r.lam), r))
    end
    @printf(io, "  (LinReg rows: held SSE in dQ-sd units of the h = %d M4 scaling; dof on its own %d fit rows)\n",
            maximum(hs), isempty(lr) ? 0 : lr[1].ntr)

    lag1m(r) = mean(r.acf[1, :])
    println(io, "\n==== M4-path λ equivalent to LinReg λ (log-interpolated on the grid)")
    @printf(io, "  %-8s %-10s | %s\n", "LinReg", "λ_LR", join((@sprintf("h=%d: by dof frac  by mean lag-1", h) for h in hs), " | "))
    table = Dict{Tuple{Int,String},Any}()
    for (nm, _) in TARGETS
        i = findfirst(r -> r.name == nm, lr)
        i === nothing && continue
        t = lr[i]
        cells = String[]
        for h in hs
            a = match_lambda(m4[h].rows, r -> r.frac, t.frac)
            b = match_lambda(m4[h].rows, lag1m, lag1m(t))
            table[(h, nm)] = (; dof = a, lag1 = b)
            push!(cells, @sprintf("   %-12s %-14s", lamstr(a), lamstr(b)))
        end
        @printf(io, "  %-8s %-10g | %s   (LinReg frac %.3f, mean lag-1 %.3f)\n", nm, t.lam,
                join(cells, " | "), t.frac, lag1m(t))
    end

    println(io, "\n==== PROPOSED skip-λ ladder per h (provisional): 0, LinReg2-equivalent, LinReg7-equivalent (by dof)")
    println(io, "  ⚠️ the closed-loop bias zero-crossing is NOT here: it is a stage-1 smoke item.")
    prop = Dict{Int,Vector{Float64}}()
    for h in hs
        l2 = get(table, (h, "LinReg2"), (; dof = NaN)).dof
        l7 = get(table, (h, "LinReg7"), (; dof = NaN)).dof
        prop[h] = [0.0; filter(x -> isfinite(x) && x > 0, round1.([l2, l7]))]
        @printf(io, "  [m3f.h%d]  lambda = [%s]\n", h, join((@sprintf("%g", x) for x in prop[h]), ", "))
    end
    outf = strip(get(ENV, "CALIB_OUT", ""))
    if !isempty(outf)
        isfile(outf) && error("CALIB_OUT = $outf exists; refusing to overwrite")
        jldsave(outf; m4, linreg = lr, table, proposal = prop, lams, hs, source = rec.source,
                t_max_read = T_MAX_READ)
        println(io, "\nwrote $outf")
    end
    return (; m4, lr, table, prop)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
