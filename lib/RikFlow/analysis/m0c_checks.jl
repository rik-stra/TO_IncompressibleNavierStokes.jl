# M0ᶜ gate checks (plan §3 *The reduced ladder* notes, §21 item 3, claude_memory #72(iii)).
#
#     julia --startup-file=no --project=training analysis/m0c_checks.jl [i] [ii] [iii]   # default all
#
# (`training` because part (i) evaluates the deployed closures' mean through RikFlow's StochLSTM;
# parts (ii) and (iii) need only the tracked cache and the response kernel.)
#
# (i)  ONLINE colour of the white-noise linear model. For each existing 20 TU online run of a
#      linear + eta closure (`output/TO_LSTM/{diag,window,colour}/...`), per band: the ACF of the
#      online correction dQ, and of the online residual dQ - mu(x_online), mu evaluated by the
#      deployed closure with `stochastic = false` on the ONLINE inputs (the level lag overwritten
#      with the online level every step, as m4_noise_colour.jl does with the recorded one). Beside
#      it the same fit teacher-forced on the TRACKED record over the same 0.25-20 TU window: the
#      data's dQ ACF and the data residual's ACF (m4_noise_colour.jl's definition).
# (ii) The h / h+1 control. Float64 least squares on the M0 design (`build_history`,
#      `:q_star_q`, predictor included, bias last) of dQ on [q*_n; q_{n-1}, q*_{n-1}; ...; 1] at
#      h = 1, 2, 5, 6, fitted on 1-10 TU and on 1-50 TU; residual ACF (lags 1, 2, 5, 10) per band on
#      the fit window and on held-out 52-74 TU. 🔒 Nothing past 74 TU is read (confirmation block).
#      cond(X) raw and column-standardised, and max |C| in the standardised basis.
# (iii) Acceptance diagnostic: kernel-weighted noise power P = Var(sum_k G_k e_{n-k}), G the
#      MEASURED solver impulse kernel (`response/response_kernel.jld2`, m4_response.jl, K = 200),
#      for white eta, AR(1) at the data lag-1, AR(2) by Yule-Walker, AR(2) least-squares on the
#      ACF at lags 1-20, and the data residual itself. Per band with the diagonal kernel, and with
#      the full 6x6 kernel (simulated). All noises at the data residual's variance.

using RikFlow, JLD2, Statistics, LinearAlgebra, Printf, Random
const RF = RikFlow

const HERE = @__DIR__
const DT = 2.5e-3
const NQ = 6
const LABELS = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const TO = normpath(joinpath(HERE, "..", "exp_square_HIT", "output", "TO_LSTM"))
const T_MAX_READ = 74.0            # 🔒 confirmation block starts at 76 TU; embargo 74-76
const REF = load(joinpath(HERE, "data", "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3_qois.jld2"))
const QR, QSR, DQR = REF["q"], REF["q_star"], REF["dQ"]

acf(x, L) = (y = x .- mean(x); v = sum(abs2, y); [sum(y[1:(end - l)] .* y[(1 + l):end]) / v for l in L])
fmt(v; d = 2) = join((@sprintf("%6.*f", d, x) for x in v), "")
steps_of(t0, t1) = round(Int, t0 / DT):round(Int, t1 / DT)
guard(r) = (maximum(r) * DT <= T_MAX_READ + 1e-9 ||
            error("🔒 step $(maximum(r)) = $(maximum(r) * DT) TU is past $T_MAX_READ TU"); r)

# ---------------------------------------------------------------------------------------------
# (ii) least squares on the M0 design
# ---------------------------------------------------------------------------------------------

"""
Design rows for steps `ns` (each needs n - h >= 1): X is N x (NQ(2h+1)+1), Y the LEVEL q^n, as the
LinReg path fits it (claude_memory #12). q* is a column of X, so the residual equals that of the
target q^n - q*_n; it differs from a recorded-dQ target by the tracking error (0.2-1.7% of sd(dQ)).
"""
function design(h, ns; q = QR, qs = QSR)
    spec = RF.HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    a, b = first(ns) - h, last(ns)
    X, Y, st = RF.build_history(spec, qs[:, a:b], q[:, a:(b + 1)])
    st = st .+ (a - 1)
    @assert st == collect(ns)
    return X, Y
end

function lsfit(h, fitr)
    guard(fitr)
    X, Y = design(h, fitr)
    C = X \ Y                         # QR, Float64
    mu = vec(mean(X[:, 1:(end - 1)]; dims = 1))
    sg = vec(std(X[:, 1:(end - 1)]; dims = 1))
    Xs = hcat((X[:, 1:(end - 1)] .- mu') ./ sg', ones(size(X, 1)))
    Cs = C[1:(end - 1), :] .* sg ./ vec(std(Y; dims = 1))'    # standardised coefficients
    return (; h, fitr, C, cond_raw = cond(X), cond_std = cond(Xs), maxC = maximum(abs, Cs),
            R = Y .- X * C)
end

resid(f, r) = (guard(r); (X, Y) = design(f.h, r); Y .- X * f.C)

const LAGS2 = (1, 2, 5, 10)
const WIN_FIT = (("1-10 TU", steps_of(1, 10)), ("1-50 TU", steps_of(1, 50)))
const WIN_HELD = steps_of(52, 74)

function part_ii(; io = stdout)
    println(io, "\n==== (ii) h control: least-squares M0 mean, residual ACF at lags ", LAGS2)
    fits = Dict{Tuple{Int,String},Any}()
    for (wn, wr) in WIN_FIT, h in (1, 2, 5, 6)
        f = lsfit(h, wr)
        fits[(h, wn)] = f
        Rh = resid(f, WIN_HELD)
        @printf(io, "\nfit %s  h = %d   cond(X) raw %.2e  standardised %.2e   max|C_std| %.1f\n", wn, h,
                f.cond_raw, f.cond_std, f.maxC)
        @printf(io, "  %-9s %-26s %-26s %s\n", "band", "fit window ACF 1 2 5 10", "held 52-74 ACF 1 2 5 10",
                "sd held/fit")
        for i in 1:NQ
            @printf(io, "  %-9s %s   %s   %.3f\n", LABELS[i], fmt(acf(f.R[:, i], LAGS2)), fmt(acf(Rh[:, i], LAGS2)),
                    std(Rh[:, i]) / std(f.R[:, i]))
        end
    end
    return fits
end

# ---------------------------------------------------------------------------------------------
# (iii) kernel-weighted noise power
# ---------------------------------------------------------------------------------------------

"AR(2) ACF from (phi1, phi2), lags 0:L."
function ar2_acf(p1, p2, L)
    r = zeros(L + 1)
    r[1] = 1
    r[2] = p1 / (1 - p2)
    for k in 2:L
        r[k + 1] = p1 * r[k] + p2 * r[k - 1]
    end
    return r
end
ar2_stationary(p1, p2) = abs(p2) < 1 && p2 + p1 < 1 && p2 - p1 < 1

function ar2_yw(r1, r2)
    p1 = r1 * (1 - r2) / (1 - r1^2)
    p2 = (r2 - r1^2) / (1 - r1^2)
    return p1, p2
end

"AR(2) whose ACF best matches the empirical one at lags 1..L (grid + local refine, stationary)."
function ar2_ls(racf; L = 20)
    best = (Inf, 0.0, 0.0)
    for p1 in range(-1.99, 1.99; length = 399), p2 in range(-0.99, 0.99; length = 199)
        ar2_stationary(p1, p2) || continue
        r = ar2_acf(p1, p2, L)
        e = sum(abs2, r[2:end] .- racf[2:(L + 1)])
        e < best[1] && (best = (e, p1, p2))
    end
    _, p1, p2 = best
    for s in (0.005, 0.001, 0.0002), _ in 1:3
        for d1 in (-2s, -s, 0, s, 2s), d2 in (-2s, -s, 0, s, 2s)
            a, b = p1 + d1, p2 + d2
            ar2_stationary(a, b) || continue
            e = sum(abs2, ar2_acf(a, b, L)[2:end] .- racf[2:(L + 1)])
            e < best[1] && (best = (e, a, b))
        end
        _, p1, p2 = best
    end
    return p1, p2
end

"sigma^2 g' T(rho) g: the variance of sum_k g_k e_{n-k} for a stationary e with ACF rho (lags 0:K-1)."
function kpower(g, rho, s2)
    K = length(g)
    P = 0.0
    for a in 1:K, b in 1:K
        P += g[a] * g[b] * rho[abs(a - b) + 1]
    end
    return s2 * P
end

"Direct: variance of the kernel-filtered series (valid part only)."
function kfilter_var(g, e)
    K = length(g)
    n = length(e)
    y = [sum(g[k] * e[t - k] for k in 1:K) for t in (K + 1):n]
    return var(y)
end

"Full 6x6 kernel applied to a 6 x n series (valid part), per-output-band variance."
function kfilter_var_full(G, E)
    K = size(G, 3)
    n = size(E, 2)
    Y = zeros(NQ, n - K)
    for (c, t) in enumerate((K + 1):n), k in 1:K
        Y[:, c] .+= view(G, :, :, k) * view(E, :, t - k)
    end
    return vec(var(Y; dims = 2))
end

"Simulate a diagonal AR(2) (p1, p2 per band) with lag-0-correlated innovations, marginal var s2."
function sim_ar(p1, p2, Cor, s2, n, rng)
    Lc = cholesky(Symmetric(Cor)).L
    E = zeros(NQ, n + 1000)
    for t in 3:size(E, 2)
        xi = Lc * randn(rng, NQ)
        @inbounds for i in 1:NQ
            E[i, t] = p1[i] * E[i, t - 1] + p2[i] * E[i, t - 2] + xi[i]
        end
    end
    E = E[:, 1001:end]
    return E .* sqrt.(s2 ./ vec(var(E; dims = 2)))
end

function part_iii(fits; io = stdout)
    G = load(joinpath(TO, "response", "response_kernel.jld2"), "G")
    K = size(G, 3)
    @printf(io, "\n==== (iii) kernel-weighted noise power, measured solver kernel (K = %d steps = %.2f TU)\n",
            K, K * DT)
    @printf(io, "kernel diag: G_1 %s | sum_k G_k (= step response at k = %d) %s\n",
            fmt([G[j, j, 1] for j in 1:NQ]), K, fmt([sum(G[j, j, :]) for j in 1:NQ]; d = 1))
    rng = Xoshiro(20260928)
    out = Dict{Any,Any}()
    for key in ((5, "1-10 TU"), (1, "1-10 TU"), (5, "1-50 TU"))
        f = fits[key]
        Rf = f.R                                   # N x NQ, fit window
        Rh = resid(f, WIN_HELD)
        @printf(io, "\nresidual of the h = %d LS mean fitted on %s  (P / P_white; data = the residual series itself, filtered)\n",
                key...)
        @printf(io, "  %-9s %6s %6s %6s | %8s %8s %8s %8s | %8s %8s | %s\n", "band", "rho1", "phi1", "phi2",
                "white", "AR(1)", "AR2-YW", "AR2-LS", "data-fit", "data-held", "AR2-LS (phi1, phi2)")
        rows = []
        for i in 1:NQ
            e = Rf[:, i]
            s2 = var(e)
            r = acf(e, 0:(K - 1))
            g = G[i, i, :]
            Pw = kpower(g, [1.0; zeros(K - 1)], s2)
            a = r[2]
            P1 = kpower(g, [a^k for k in 0:(K - 1)], s2)
            y1, y2 = ar2_yw(r[2], r[3])
            Pyw = ar2_stationary(y1, y2) ? kpower(g, ar2_acf(y1, y2, K - 1), s2) : NaN
            l1, l2 = ar2_ls(r; L = 20)
            Pls = kpower(g, ar2_acf(l1, l2, K - 1), s2)
            Pd = kfilter_var(g, e)
            eh = Rh[:, i] .* sqrt(s2 / var(Rh[:, i]))        # held-out residual at the fit's variance
            Ph = kfilter_var(g, eh)
            @printf(io, "  %-9s %6.2f %6.2f %6.2f | %8.2f %8.2f %8.2f %8.2f | %8.2f %8.2f | (%.3f, %.3f)\n",
                    LABELS[i], a, y1, y2, 1.0, P1 / Pw, Pyw / Pw, Pls / Pw, Pd / Pw, Ph / Pw, l1, l2)
            push!(rows, (; band = LABELS[i], rho1 = a, yw = (y1, y2), ls = (l1, l2),
                         ratio = (ar1 = P1 / Pw, yw = Pyw / Pw, ls = Pls / Pw, data = Pd / Pw, held = Ph / Pw),
                         Pwhite = Pw))
        end
        # full 6x6 kernel: simulated noises with the residual's lag-0 correlation, and the data itself
        s2v = vec(var(Rf; dims = 1))
        Cor = cor(Rf)
        n = 200_000
        W = sim_ar(zeros(NQ), zeros(NQ), Cor, s2v, n, rng)
        A1 = sim_ar([acf(Rf[:, i], 1:1)[1] for i in 1:NQ], zeros(NQ), Cor, s2v, n, rng)
        pls = [r.ls for r in rows]
        A2 = sim_ar(first.(pls), last.(pls), Cor, s2v, n, rng)
        pw = kfilter_var_full(G, W)
        @printf(io, "  full 6x6 kernel, P / P_white(correlated): AR(1) %s | AR2-LS %s | data-fit %s | data-held %s\n",
                fmt(kfilter_var_full(G, A1) ./ pw), fmt(kfilter_var_full(G, A2) ./ pw),
                fmt(kfilter_var_full(G, permutedims(Rf)) ./ pw),
                fmt(kfilter_var_full(G, permutedims(Rh .* sqrt.(s2v ./ vec(var(Rh; dims = 1)))')) ./ pw))
        out[key] = rows
    end
    return out
end

# ---------------------------------------------------------------------------------------------
# (i) online colour of the white-noise linear closures
# ---------------------------------------------------------------------------------------------

const ONLINE_FITS = ["diag/r2_lin_const", "diag/r2_lin_const_n05", "diag/r2_lin_const_n0",
                     "diag/r2_lin_const_h2", "diag/rdg_h3_l0", "diag/rdg_h5_l0", "diag/rdg_h10_l0",
                     "diag/rdg_h1_l1e-05", "window/b_lin_eta_h1", "window/b_lin_eta_h5",
                     "colour/lin_h1_ar"]
const LAGS1 = (1, 2, 5, 10, 20)

"Deterministic mean of a deployed closure along a trajectory (q: NQ x (n+1), dQ: NQ x n)."
function closure_mean(fit, q, dQ; nwarm = 100)
    n = size(dQ, 2)
    qs = q[:, 2:(n + 1)] .- dQ                    # q*_n = q[:, n+1] - dQ[:, n]
    m = RF.StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = dQ[:, 1:nwarm],
                     rng = Xoshiro(1), gate = 0.0, stochastic = false)
    mu = zeros(NQ, n)
    for t in 1:n
        mu[:, t] = RF.get_next_item_timeseries(m, qs[:, t])
        fit.spec.hist.h > 0 &&
            (m.buf.q[:, 1] .= eltype(m.buf.q).(vec(RF.scale_input(q[:, t + 1], fit.scaling.in_scaling))))
    end
    return mu
end

function part_i(; io = stdout)
    println(io, "\n==== (i) online colour of linear + eta closures (20 TU runs, steps 101-8000)")
    nstep = 8000
    cols = 101:nstep
    out = Dict{String,Any}()
    for d in ONLINE_FITS
        dir = joinpath(TO, d)
        fs = sort(filter(x -> occursin(r"^data_online_tsim20\.0_replica\d\.jld2$", x), readdir(dir)))
        isempty(fs) && (println(io, "  $d: no online runs"); continue)
        fit = RF.load_stochlstm(joinpath(dir, "StochLSTM_seed1.jld2"))
        # teacher-forced on the TRACKED record, same window
        mur = closure_mean(fit, QR[:, 1:(nstep + 1)], DQR[:, 1:nstep])
        Rr = DQR[:, cols] .- mur[:, cols]
        dA, rA, sdr = [], [], []
        for f in fs
            o = load(joinpath(dir, f), "data_online")
            size(o.dQ, 2) >= nstep || continue
            q, dQ = o.q[:, 1:(nstep + 1)], o.dQ[:, 1:nstep]
            all(isfinite, q) || continue
            mu = closure_mean(fit, q, dQ)
            R = dQ[:, cols] .- mu[:, cols]
            push!(dA, [acf(dQ[i, cols], LAGS1) for i in 1:NQ])
            push!(rA, [acf(R[i, :], LAGS1) for i in 1:NQ])
            push!(sdr, [std(R[i, :]) / std(Rr[i, :]) for i in 1:NQ])
        end
        nr = length(dA)
        @printf(io, "\n%s  (h = %d, window %d, %d replicas%s)\n", d, fit.spec.hist.h, fit.spec.window, nr,
                hasproperty(fit.scaling, :eta_ar) ? ", AR(1) eta" : "")
        @printf(io, "  %-9s %-31s %-31s | %-31s %-31s %s\n", "band", "ONLINE dQ ACF 1 2 5 10 20",
                "online resid ACF", "TRACKED dQ ACF", "tracked resid ACF", "sd(res on)/sd(res trk)")
        rows = []
        for i in 1:NQ
            da = mean(x[i] for x in dA)
            ra = mean(x[i] for x in rA)
            lo = minimum(x[i][1] for x in dA)
            hi = maximum(x[i][1] for x in dA)
            td = acf(DQR[i, cols], LAGS1)
            tr = acf(Rr[i, :], LAGS1)
            @printf(io, "  %-9s %s %s | %s %s  %.2f   (online dQ lag-1 range %.3f-%.3f)\n", LABELS[i], fmt(da), fmt(ra),
                    fmt(td), fmt(tr), mean(x[i] for x in sdr), lo, hi)
            push!(rows, (; band = LABELS[i], online_dq = da, online_res = ra, tracked_dq = td, tracked_res = tr,
                         lag1_range = (lo, hi)))
        end
        out[d] = rows
        flush(io)
    end
    return out
end

function main(parts)
    io = stdout
    res = Dict{String,Any}()
    "i" in parts && (res["i"] = part_i(; io))
    if "ii" in parts || "iii" in parts
        fits = part_ii(; io)
        res["ii"] = Dict(string(k) => (; v.h, v.cond_raw, v.cond_std, v.maxC) for (k, v) in fits)
        "iii" in parts && (res["iii"] = part_iii(fits; io))
    end
    mkpath(joinpath(HERE, "output"))
    jldsave(joinpath(HERE, "output", "m0c_checks_" * join(parts, "_") * ".jld2"); res = res)
    return res
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? ["i", "ii", "iii"] : ARGS)
end
