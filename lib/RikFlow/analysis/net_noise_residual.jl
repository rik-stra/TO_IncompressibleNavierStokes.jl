# The offline basis of input-noise training (results_LSTMS.md Sec. 36; 2026-10-26): the residual of means trained
# with the closure's own noise on their inputs (LinReg1_nt, the noise-trained CNN), against LinReg1 and the clean
# CNN, by the measures of paper Secs. 5-6, on the clean record AND along a history that carries the closure's own
# draws.
#
#     PAPER_FIG_DIR=<dir> julia --startup-file=no --project=lib/RikFlow/analysis \
#         lib/RikFlow/analysis/net_noise_residual.jl LinReg1=lrs:LinReg1 LinReg1_nt=lrs:LinReg1_nt \
#         CNN=step3/grid_cap_rmse_wd0.3_s1 CNN_nt=step3/cnn_nt_rmse_wd0.3_s1_r5
#
# `Label=lrs:<name>` is a deployed LinReg (`TO_LRS/<name>/LinReg.jld2`); `Label=dir[,dir...]` a CNN run under TO_LSTM
# written with RIKFLOW_CNN_REFIT (its refitted network, `pred3` in pred.jld2), scored by `cnn_mean_fit.jl`
# RIKFLOW_CNN_SCORE (its prediction along the noise path, `TO_LSTM/step3/innoise_scores/<basename>.jld2`, or
# NETNOISE_SCORES). The first label must be LinReg1: every ratio is against its clean one-step error.
#
# 🔑 Two residuals, both of the correction dQ^n = q^n - q^{n*} (raw units):
#  - CLEAN: on the record's own history, R = dQ^n - mean(x^n), the measure of Secs. 5-6;
#  - DEPLOYED: along ONE noise path, Rn = dQ^n - mean(x^n + dx^n), dx^n the window noise of the draws
#    η^{n-1} ... η^{n-h} (tools/step3_innoise.jl, anchored at the prediction step) of a path η^m ~ N(0, Σ) i.i.d.,
#    Σ the covariance the closure deploys. Online, a closure's history carries its own draws, so Rn is the residual it
#    meets, and Σ is what it draws to model it: a noise-trained closure deploys Σ = E[Rn Rn'] (self-consistent),
#    LinReg1 and the clean CNN deploy E[R R'] while meeting Rn. Rn = R - G, G = mean(x + dx) - mean(x) the mean's
#    response to its own draws (independent of the record), so Rn's lag-k autocovariance is R's plus G's, and G
#    reads h draws: an MA(h - 1) part at lags 1-4 on top of R's colour. One path per seed: the standard normals
#    `E` of seed s are shared by every model of that seed (common random numbers; the LinRegs take seed 1's).
# The gain: rms(G) / sd(η) per QoI (the probe's, results_LSTMS.md Sec. 34).
#
# Prints every number; writes (PAPER_FIG_DIR)
#   nt_residual_acf.pdf   ACF on 10-100 TU: clean (top block) and deployed (bottom), dQ dashed, first seed bold
#   nt_scale.pdf          spread about the bin mean of the DEPLOYED residual against the level, as net_scale.pdf
# (none with NETNOISE_NOFIG=1, for comparisons of more than six models)

include(joinpath(@__DIR__, "net_residual.jl"))           # plot_paper_secs45.jl, LinReg1's residual, ACF and scale tools

const NN_MAIN = abspath(PROGRAM_FILE) == @__FILE__
const NN_SCORES = get(ENV, "NETNOISE_SCORES", "step3/innoise_scores")
# LinReg1 (Okabe-Ito blue, as Secs. 4-6), LinReg1_nt (sky blue), clean CNN (bluish green, the RMSE network of
# Sec. 6), noise-trained CNN (reddish purple); further labels orange, vermillion
const NN_COL = [to_color("#0072B2"), to_color("#56B4E9"), to_color("#009E73"), to_color("#CC79A7"),
                to_color("#E69F00"), to_color("#D55E00")]

"Raw dQ^n = q^n - q^{n*} on record steps `n` (rows = steps)."
dq_rows(n) = permutedims(REC["q"][:, n .+ 1] .- REC["q_star"][:, n])

"""
A CNN run (comma-separated dirs, one seed each): per seed the clean and deployed residuals on the run's record steps,
the response G, the raw noise sd, and the steps' level q^{n*}. The path's standard normals come with it.
"""
function load_cnn(label, dirs)
    seeds = Dict{Int,Any}()
    ref = nothing
    for dir in split(dirs, ",")
        P = load(joinpath(OUTD, "TO_LSTM", dir, "pred.jld2"))
        S = load(joinpath(OUTD, "TO_LSTM", NN_SCORES, basename(dir) * ".jld2"))
        S["cols"] == P["cols"] || error("$label: score file and pred.jld2 on different steps")
        n = P["cols"]
        dq = dq_rows(n)
        for (s, o) in S["out"]
            haskey(seeds, s) && error("$label: seed $s twice")
            R = dq .- permutedims(P["pred3"][s])
            Rn = dq .- permutedims(o.pred3n)
            seeds[s] = (; R, Rn, G = Rn .- R, seta = S["so"] .* sqrt.(diag(o.Sigma)), Sigma_raw = S["so"] .* o.Sigma .* S["so"]',
                        E = o.E, m0 = S["m0"], val = o.val, wd = o.wd, src = o.sigma_src)
        end
        ref = (; n, m0 = S["m0"])
    end
    return (; label, kind = :cnn, n = ref.n, m0 = ref.m0, seeds)
end

"The mean and covariance of a stored MvNormal: JLD2 reconstructs it without Distributions (analysis project)."
mvg_parts(d) = (collect(Float64, d.μ), Matrix{Float64}(d.Σ.mat))

"""
A deployed LinReg on the steps `n` (those of the CNN runs): clean residual from its `c` (Sec. 5's `full_residual`, raw
units), and its response to the window noise of its own Σ (`stoch_distr`, raw) along the path of standard normals
`E` (first step `m0`), exactly (linear): dx in its scaled regressor, the layout `[q*^n; q^{n-1}, q^{n-1*}; ...; 1]`
asserted on the record.
"""
function load_lrs(label, name, n, E, m0)
    m = load(joinpath(OUTD, "TO_LRS", name, "LinReg.jld2"))
    C = permutedims(Matrix{Float64}(m["c"]))
    s = m["scaling"].in_scaling
    mu, sg = vec(s.mu), vec(s.sigma)
    h = m["hist_len"]
    hist = HistorySpec(; h, n_qoi = NQ, hist_var = :q_star_q, include_predictor = true)
    X, Y, st = build_history(hist, scale_input(REC["q_star"], s), scale_input(REC["q"], s))
    rows = indexin(n, st)
    any(isnothing, rows) && error("$name: some CNN steps have no LinReg row")
    rows = Int.(rows)
    X, Y = X[rows, :], Y[rows, :]
    # the layout the window map assumes, on the first step
    X[1, 1:NQ] ≈ (REC["q_star"][:, n[1]] .- mu) ./ sg && X[1, (NQ + 1):(2NQ)] ≈ (REC["q"][:, n[1]] .- mu) ./ sg &&
        X[1, (2NQ + 1):(3NQ)] ≈ (REC["q_star"][:, n[1] - 1] .- mu) ./ sg || error("$name: regressor layout differs")
    R = (Y .- X * C) .* sg'                                  # raw (level and correction share the scaling)
    maximum(abs, R .- (dq_rows(n) .- (X * C .- X[:, 1:NQ]) .* sg')) < 1e-8 * maximum(abs, R) ||
        error("$name: the level residual is not the correction's")
    mu_eta, S_eta = mvg_parts(m["stoch_distr"])
    Sraw = Diagonal(sg) * S_eta * Diagonal(sg)
    L = cholesky(Symmetric(Sraw)).L
    path = L * E                                             # raw η^m, m = m0, m0 + 1, ...
    G = zeros(length(n), NQ)                                 # the mean's response, raw
    for k in 1:h
        Sq = k == 1 ? zeros(NQ, length(n)) : -reduce(+, (path[:, n .- j .- m0 .+ 1] for j in 1:(k - 1)))
        Ss = -reduce(+, (path[:, n .- j .- m0 .+ 1] for j in 1:k))
        qr = NQ + 2(k - 1) * NQ .+ (1:NQ)
        G .+= (permutedims(Sq ./ sg) * C[qr, :] .+ permutedims(Ss ./ sg) * C[qr .+ NQ, :]) .* sg'
    end
    @printf("lrs: %-12s mean of its noise / sd %s (drawn about zero here)\n", name,
            pr(abs.(mu_eta) ./ sqrt.(diag(S_eta)); f = "%.1e"))
    return (; label, kind = :lrs, n, m0, seeds = Dict(1 => (; R, Rn = R .- G, G, seta = sqrt.(diag(Sraw)), Sigma_raw = Sraw)))
end

const MODELS = if NN_MAIN
    specs = [(String(a[1]), String(a[2])) for a in split.(ARGS, "=")]
    first(specs[1]) == "LinReg1" || error("the first label must be LinReg1")
    cnns = [load_cnn(l, d) for (l, d) in specs if !startswith(d, "lrs:")]
    isempty(cnns) && error("at least one CNN run: the noise path comes from its score file")
    c1 = cnns[1]
    E1 = c1.seeds[1].E
    map(specs) do (l, d)
        startswith(d, "lrs:") ? load_lrs(l, d[5:end], c1.n, E1, c1.m0) : only(c for c in cnns if c.label == l)
    end
else
    []
end

# rows: 1-10 TU (the closures' training window, steps 400-4000) and 10-100 TU (held out)
rows_tr(m) = findall(k -> 400 <= k <= 4000, m.n)
rows_ho(m) = findall(k -> 10 < k * DT <= 100, m.n)
qs_rows(m) = permutedims(REC["q_star"][:, m.n])
smean(f, m) = sum(f(v) for v in values(m.seeds)) / length(m.seeds)
first_seed(m) = m.seeds[minimum(keys(m.seeds))]

"Per model: sd(η) over LinReg1's clean rms; the one-step error clean and deployed over LinReg1's clean rms; the gain."
function report_error(L1)
    tr, ho = rows_tr(L1), rows_ho(L1)
    b = first_seed(L1).R
    rtr, rho = rmsq(b, tr), rmsq(b, ho)
    for m in MODELS
        @printf("error: %-12s (%d seeds) sd(eta) / LinReg1 rms (1-10 TU) %s\n", m.label, length(m.seeds), pr(smean(v -> v.seta ./ rtr, m)))
        @printf("error: %-12s 1-10 TU clean %s | deployed %s | gain %s\n", m.label, pr(smean(v -> rmsq(v.R, tr) ./ rtr, m)),
                pr(smean(v -> rmsq(v.Rn, tr) ./ rtr, m)), pr(smean(v -> rmsq(v.G, tr) ./ v.seta, m)))
        @printf("error: %-12s 10-100 TU clean %s | deployed %s | gain %s\n", m.label, pr(smean(v -> rmsq(v.R, ho) ./ rho, m)),
                pr(smean(v -> rmsq(v.Rn, ho) ./ rho, m)), pr(smean(v -> rmsq(v.G, ho) ./ v.seta, m)))
        # self-consistency: the deployed residual's sd on 1-10 TU over the sd the closure draws
        @printf("error: %-12s deployed residual sd (1-10 TU, about zero) / drawn sd %s\n", m.label,
                pr(smean(v -> sqrt.(vec(mean(abs2, v.Rn[tr, :]; dims = 1))) ./ v.seta, m); f = "%.3f"))
        length(m.seeds) > 1 && @printf("error: %-12s per seed, 10-100 TU clean: %s\n", m.label,
                                       join((pr(rmsq(m.seeds[s].R, ho) ./ rho) for s in sort(collect(keys(m.seeds)))), " / "))
    end
end

"""
10-100 TU steps with some q^{n*}_j outside its 1-10 TU range (as `range_split` of Sec. 6): the share, and per model the
error inside and outside over LinReg1's on the same steps, clean and deployed.
"""
function report_range(L1)
    tr, ho = rows_tr(L1), rows_ho(L1)
    q = qs_rows(L1)
    lo, hi = vec(minimum(q[tr, :]; dims = 1)), vec(maximum(q[tr, :]; dims = 1))
    out = [any((q[k, :] .< lo) .| (q[k, :] .> hi)) for k in ho]
    io, oo = ho[.!out], ho[out]
    b = first_seed(L1).R
    @printf("range: %.1f %% of the 10-100 TU steps have a level outside its 1-10 TU range\n", 100count(out) / length(out))
    for m in MODELS
        @printf("range: %-12s clean in %s | out %s || deployed in %s | out %s\n", m.label,
                pr(smean(v -> rmsq(v.R, io) ./ rmsq(b, io), m)), pr(smean(v -> rmsq(v.R, oo) ./ rmsq(b, oo), m)),
                pr(smean(v -> rmsq(v.Rn, io) ./ rmsq(b, io), m)), pr(smean(v -> rmsq(v.Rn, oo) ./ rmsq(b, oo), m)))
    end
end

acf_rows(R, rows, i) = autocorr(collect(R[rows, i]), KMAX)
"Seed-mean ACF (lags 0..KMAX x QoI) of field `f` (:R or :Rn) on `rows`."
acf_mean_(m, f, rows) = smean(v -> hcat((acf_rows(getfield(v, f), rows, i) for i in 1:NQ)...), m)

function report_acf()
    for (wl, rf) in (("10-100 TU", rows_ho), ("1-10 TU", rows_tr)), (nm, f) in (("clean", :R), ("deployed", :Rn))
        for m in MODELS
            a = acf_mean_(m, f, rf(m))
            @printf("acf: %-9s %-8s %-12s rho1 %s | rho2-5 %s | ratio %s\n", wl, nm, m.label, pr(a[2, :]),
                    join((pr(a[k, :]) for k in 3:6), " / "), pr([lrv(a[:, i]) for i in 1:NQ]; f = "%.1f"))
        end
    end
end

"The power law of step 2 (Sec. 5's `fit_powerlaw_scale`) on 1-10 TU for field `f`, first seed: β and the held-out gain."
function report_scale()
    for (nm, f) in (("clean", :R), ("deployed", :Rn)), m in MODELS
        v = first_seed(m)
        tr, ho = rows_tr(m), rows_ho(m)
        q = qs_rows(m)
        Rf = getfield(v, f)
        E = Rf ./ vec(std(Rf[tr, :]; dims = 1))'
        out = map(1:NQ) do i
            p = fit_powerlaw_scale(E[tr, i], q[tr, i])
            nllc(rows) = mean(0.5 .* (log(2pi) .+ log(mean(abs2, E[tr, i])) .+ E[rows, i] .^ 2 ./ mean(abs2, E[tr, i])))
            lv(rows) = p.logvar .+ 2p.beta .* log.(clamp.(q[rows, i], p.qclip...) ./ p.qref)
            nllp(rows) = mean(0.5 .* (log(2pi) .+ lv(rows) .+ E[rows, i] .^ 2 .* exp.(-lv(rows))))
            (p.beta, nllc(ho) - nllp(ho))
        end
        @printf("scale: %-8s %-12s beta %s | held-out NLL gain %s (sum %.2f)\n", nm, m.label, pr(first.(out)),
                pr(last.(out); f = "%.3f"), sum(last.(out)))
    end
end

function fig_acf()
    dqr = (; R = permutedims(REC["q"][:, 2:end] .- REC["q_star"]), ho = findall(k -> 10 < k * DT <= 100, 1:size(REC["q_star"], 2)))
    lags = 1:KMAX
    fig = Figure(size = (W, 470), figure_padding = (2, 6, 2, 2))
    for (b, (f, title)) in enumerate(((:R, "on the recorded history"), (:Rn, "along a history that carries the closure's own draws")))
        g = fig[b, 1] = GridLayout()
        Label(g[0, 1:3], "residual " * title; fontsize = 8, halign = :left, tellwidth = false)
        axs = [Axis(g[pos(i)...]; title = QLAB[i], LAGAX...) for i in 1:NQ]
        for i in 1:NQ
            ax = axs[i]
            hlines!(ax, [0.0]; color = :gray70, linewidth = 0.5)
            lines!(ax, lags, acf_ho(dqr, i)[2:end]; color = C_REF, linestyle = :dash, linewidth = 1.0)
            for (j, m) in enumerate(MODELS), (k, s) in enumerate(sort(collect(keys(m.seeds))))
                a = acf_rows(getfield(m.seeds[s], f), rows_ho(m), i)
                lines!(ax, lags, a[2:end]; color = NN_COL[j], linewidth = k == 1 ? 1.1 : 0.5)
            end
            r, c = pos(i)
            r == 2 ? (ax.xlabel = "lag (steps)") : (ax.xticklabelsvisible = false)
            c == 1 ? (ax.ylabel = "autocorrelation") : (ax.yticklabelsvisible = false)
            xlims!(ax, 1, KMAX); ylims!(ax, -0.55, 1.02)
        end
        rowgap!(g, 3); colgap!(g, 10)
    end
    els = Any[LineElement(color = C_REF, linestyle = :dash); [LineElement(color = NN_COL[j], linewidth = 1.1) for j in eachindex(MODELS)]]
    Legend(fig[3, 1], els, [L"dQ"; [m.label for m in MODELS]]; orientation = :horizontal)
    rowgap!(fig.layout, 4)
    savefig("nt_residual_acf.pdf", fig)
end

"As net_residual.jl's `fig_scale`, for the DEPLOYED residual of every model (first seed)."
function fig_scale(; nbin = 10)
    fig = Figure(size = (W, 255), figure_padding = (2, 6, 2, 2))
    axs = qoi_axes(fig; xscale = log10, yscale = log10, xticks = [0.5, 1, 1.5],
                   yticks = [0.3, 0.5, 0.7, 1, 1.4, 2, 3], xtickformat = vs -> [@sprintf("%g", v) for v in vs],
                   ytickformat = vs -> [@sprintf("%g", v) for v in vs])
    for i in 1:NQ
        hlines!(axs[i], [1.0]; color = :gray55, linestyle = :dash, linewidth = 0.6)
    end
    for (j, m) in enumerate(MODELS)
        v = first_seed(m)
        tr, ho = rows_tr(m), rows_ho(m)
        q = qs_rows(m)
        E = v.Rn ./ vec(std(v.Rn[tr, :]; dims = 1))'
        for i in 1:NQ
            p = fit_powerlaw_scale(E[tr, i], q[tr, i])
            x = q[:, i] ./ p.qref
            edges = quantile(x[tr], range(0, 1; length = nbin + 1)[2:(end - 1)])
            lo, hi = p.qclip ./ p.qref
            xx = exp.(range(log(0.3), log(2.1); length = 200))
            lines!(axs[i], xx, exp(p.logvar / 2) .* clamp.(xx, lo, hi) .^ p.beta; color = NN_COL[j], linewidth = 1.1)
            for (rows, filled) in ((tr, true), (ho, false))
                xm, ym, _ = binned(x, E[:, i], rows, edges)
                scatter!(axs[i], xm, ym; color = filled ? NN_COL[j] : :white, strokecolor = NN_COL[j], strokewidth = 0.7, markersize = 4.5)
            end
            _, str_, _ = binned(x, E[:, i], tr, edges)
            _, sho, mho = binned(x, E[:, i], ho, edges)
            @printf("scalefig: %-12s %-8s lowest/highest tenth sd %.2f/%.2f (1-10), %.2f/%.2f (10-100); 10-100 bin means %s\n",
                    m.label, LABELS[i], str_[1], str_[end], sho[1], sho[end], pr(mho; f = "%+.2f"))
        end
    end
    for i in 1:NQ
        r, c = pos(i)
        r == 2 ? (axs[i].xlabel = L"q^{n*}_i / q_{\mathrm{ref},i}") : (axs[i].xticklabelsvisible = false)
        c == 1 ? (axs[i].ylabel = "spread / constant scale") : (axs[i].yticklabelsvisible = false)
        xlims!(axs[i], 0.3, 2.1); ylims!(axs[i], 0.25, 3.2)
    end
    mk(c, filled) = MarkerElement(color = filled ? c : :white, strokecolor = c, strokewidth = 0.7, marker = :circle, markersize = 4.5)
    els = Any[[mk(NN_COL[j], true), LineElement(color = NN_COL[j])] for j in eachindex(MODELS)]
    labs = Any[m.label for m in MODELS]
    append!(els, [mk(:gray35, true), mk(:gray35, false), LineElement(color = :gray55, linestyle = :dash)])
    append!(labs, ["1–10 TU", "10–100 TU", "constant scale"])
    Legend(fig[3, 1:3], els, labs; orientation = :horizontal, nbanks = 2)
    rowgap!(fig.layout, 3); colgap!(fig.layout, 10)
    savefig("nt_scale.pdf", fig)
end

if NN_MAIN
    L1 = MODELS[1]
    report_error(L1)
    report_range(L1)
    report_acf()
    report_scale()
    # NETNOISE_NOFIG=1: numbers only (any number of models; the figures take at most six)
    if get(ENV, "NETNOISE_NOFIG", "0") != "1"
        fig_acf()
        fig_scale()
    end
end
