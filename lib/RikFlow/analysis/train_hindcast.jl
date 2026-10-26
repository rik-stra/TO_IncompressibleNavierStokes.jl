# The training hindcast's diagnostics (Rik, 2026-10-13; analysis/cards/round1.md, amendment 7):
# 5 ICs inside the fit window x 10 members per closure, in-sample. NOT scored against the held-out
# set, no clause, no tuning. Per closure:
#
#   divergences      members that stopped (and at which lead)
#   gate firings     identically-zero forecast dQ columns (`clamp_report`)
#   spread-skill     per QoI, pooled over the ICs and the leads of a window: sqrt(sum W V / sum E),
#                    leads 1-200 (<= 0.5 TU) and 201-1200 (0.5-3 TU)
#   defect           a diverged member, or a ratio over leads 1-200 more than a factor 2 from its
#                    matched partner's in any QoI (card amendment 7, as corrected 2026-10-13)
#   rank histogram   share of truths in the two extreme ranks (of M + 1) over leads 1-200, against
#                    2 / (M + 1) for a calibrated ensemble
#   bias             ensemble mean minus truth, % of the reference mean over the ICs, mean over the
#                    QoIs, at leads 200 and 1000 (0.5 and 2.5 TU)
#
# and, for every closure given together with its matched partner (step 2, 2026-10-16; paper Sec. 7.2,
# results_LSTMS.md §28):
#
#   spread / error   per QoI, closure / partner: the root of the pooled ensemble variance (spread) and
#                    of the pooled squared error of the ensemble mean (error), at the leads of the grid
#                    and over three windows, with the joint mask (a member that diverged in either
#                    closure is left out of both from its stop); r per lead of each closure on its own
#   below range      share of member steps at leads 201-1200 whose level lies below its training
#                    range: the clip floor of step 2's scale (`TO_LRS/LinReg1_pl`, the minimum of q*
#                    over steps 400-4000), every closure given and the reference
#
# Usage (from lib/RikFlow):
#   julia --project=analysis analysis/train_hindcast.jl LinReg1 LinReg6 LinReg7 ...
#   julia --project=analysis analysis/train_hindcast.jl --full LinReg1 LinReg1_pl
# reads exp_square_HIT/output/D6train_<name>; with `--full` the held-out hindcast D6_<name> instead
# (the defect rule is then not printed: it is the training hindcast's).


include(joinpath(@__DIR__, "score_d6.jl"))
using Printf, Statistics

const OUTD = normpath(joinpath(HERE, "..", "exp_square_HIT", "output"))
const LAB = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const WIN = (("<=0.5 TU", 1:200), ("0.5-3 TU", 201:1200))
# matched partners (analysis/cards/round1.md); the baselines have none here
const PARTNER = Dict("LinReg7_ar2n" => "LinReg7", "LinReg7_ar1ls" => "LinReg7", "Splice1_E0x11" => "LinReg1", "Splice1_E0x7" => "LinReg1",
                     "Splice1_E0x8" => "LinReg1", "Splice1_E0x11_ar2n" => "Splice1_E0x11",
                     "Splice1_E0x7_ar2n" => "Splice1_E0x7", "Splice1_E0x8_ar2n" => "Splice1_E0x8",
                     "LinReg14_ar2n" => "Splice1_E0x11_ar2n", "LinReg16_ar2n" => "Splice1_E0x7_ar2n",
                     "LinReg1_pl" => "LinReg1",      # step 2 (analysis/cards/step2.md)
                     "LinReg7_pl" => "LinReg7", "LinReg7_ar2n_pl" => "LinReg7_ar2n",   # step2_linreg7.md
                     "Splice1_E0x7_ar2n_pl" => "Splice1_E0x7_ar2n",   # S2d, added 2026-10-21
                     "Splice1_E0x7_pl" => "Splice1_E0x7",             # S2e, added 2026-10-22
                     # step 3a (analysis/cards/step3a.md, added 2026-10-23)
                     "CNN_rmse_mvg" => "LinReg1", "CNN_crps_mvg" => "CNN_rmse_mvg",
                     "CNN_crps_head" => "CNN_crps_mvg", "CNN_crps_headcorr" => "CNN_crps_mvg",
                     "CNN_rmse_ar2n" => "CNN_rmse_mvg", "CNN_rmse_pl" => "CNN_rmse_mvg",
                     # step 3c, trained with their own noise (analysis/cards/step3n.md, added 2026-10-26)
                     "LinReg1_nt" => "LinReg1", "CNN_rmse_nt" => "CNN_rmse_mvg")

function diagnose(name; truth = load_truth(), prefix = "D6train_")
    dir = joinpath(OUTD, prefix * name)
    ens = load_members(dir)
    fc, tr = assemble(ens, truth, 1:1200)
    K, nq, M, L = size(fc)
    cr = clamp_report(ens)
    stops = [findfirst(j -> any(isnan, view(fc, k, :, m, j)), 1:L) for k in 1:K, m in 1:M]
    ndiv = count(!isnothing, stops)
    @printf("\n== %s: %d ICs x %d members (t = %s TU)\n", name, K, M,
            join((@sprintf("%.2f", 0.25 * (k - 1)) for k in ens.ks), ", "))
    @printf("   diverged members %d%s; gate firings %d of %d forecast steps (%.2f %%), %d members\n",
            ndiv, ndiv == 0 ? "" : " (first stop at lead " * string(minimum(filter(!isnothing, stops))) * ")",
            cr.nfired, cr.nsteps, 100 * cr.rate, cr.members_fired)
    mom = cell_moments(fc, tr)
    r05 = Float64[]
    for (wl, w) in WIN
        r = [sqrt(sum(mom.W[:, i, w] .* mom.V[:, i, w]) / sum(mom.E[:, i, w])) for i in 1:nq]
        w == 1:200 && (r05 = r)
        @printf("   spread-skill %-9s %s\n", wl, join((@sprintf("%s %.2f", LAB[i], r[i]) for i in 1:nq), "  "))
    end
    ext = Float64[]
    for i in 1:nq
        n = 0; e = 0
        for k in 1:K, j in 1:200
            m = filter(!isnan, view(fc, k, i, :, j))
            length(m) == M || continue
            rk = count(<(tr[k, i, j]), m) + 1
            n += 1; e += (rk == 1 || rk == M + 1)
        end
        push!(ext, e / n)
    end
    @printf("   extreme ranks <=0.5 TU (calibrated %.2f): %s\n", 2 / (M + 1),
            join((@sprintf("%s %.2f", LAB[i], ext[i]) for i in 1:nq), "  "))
    for j in (200, 1000)
        mf = [mean(filter(!isnan, view(fc, k, i, :, j))) for k in 1:K, i in 1:nq]
        b = [100 * (mean(mf[:, i]) / mean(tr[:, i, j]) - 1) for i in 1:nq]
        @printf("   bias at lead %4d (%.1f TU): mean %+.1f %%  (%s)\n", j, j * 2.5e-3, mean(b),
                join((@sprintf("%+.1f", x) for x in b), " "))
    end
    return (; ndiv, r05)
end

"The card's defect rule against the matched partner's diagnostics."
function defect(name, res)
    d = res[name]
    p = get(PARTNER, name, nothing)
    p === nothing && return (d.ndiv > 0, "no partner (a baseline)")
    haskey(res, p) || return (d.ndiv > 0, "partner $p not analysed")
    q = d.r05 ./ res[p].r05
    return (d.ndiv > 0 || any(x -> !(0.5 <= x <= 2), q),
            "ratio / partner's ($p): " * join((@sprintf("%.2f", x) for x in q), " "))
end

fmtq(v; d = 3) = join((@sprintf("%.*f", d, x) for x in v), " ")

"Spread and error of `name` against `partner` (joint mask), and r per lead of each on its own."
function spread_error(name, partner; truth = load_truth(), prefix = "D6train_")
    ens = Dict(n => load_members(joinpath(OUTD, prefix * n)) for n in (name, partner))
    ks = sort(intersect(ens[name].ks, ens[partner].ks))
    own = Dict(n => assemble(restrict(ens[n], ks), truth, 1:1200) for n in (name, partner))
    jm = Dict(n => (copy(own[n][1]), own[n][2]) for n in (name, partner))
    mask_joint!([jm[name][1], jm[partner][1]])
    mo = Dict(n => cell_moments(own[n]...) for n in (name, partner))
    mj = Dict(n => cell_moments(jm[n]...) for n in (name, partner))
    nq = size(own[name][1], 2)
    ratio(F, w) = [sqrt(sum(getfield(mj[name], F)[:, i, w]) / sum(getfield(mj[partner], F)[:, i, w])) for i in 1:nq]
    r(n, j) = [sqrt(sum(mo[n].W[:, i, j] .* mo[n].V[:, i, j]) / sum(mo[n].E[:, i, j])) for i in 1:nq]
    @printf("\n== %s / %s, %s: %d ICs; spread = ensemble sd, error = rmse of the ensemble mean (joint mask);\n",
            name, partner, prefix, length(ks))
    println("   r of each closure on its own; columns ", join(LAB, " "))
    for j in LEADS
        @printf("   lead %4d  spread %s | error %s | r %s %s | r %s %s\n", j, fmtq(ratio(:V, j:j)),
                fmtq(ratio(:E, j:j)), name, fmtq(r(name, j); d = 2), partner, fmtq(r(partner, j); d = 2))
    end
    for (wl, w) in (("<=0.5 TU", 1:200), ("0.5-1 TU", 201:400), ("1-3 TU", 401:1200))
        @printf("   %-9s  spread %s | error %s\n", wl, fmtq(ratio(:V, w)), fmtq(ratio(:E, w)))
    end
end

"Share of member steps (leads 201-1200) below step 2's clip floor, per QoI, for `names` and the reference."
function below_range(names; truth = load_truth(), prefix = "D6train_", w = 201:1200)
    lo = load(joinpath(OUTD, "TO_LRS", "LinReg1_pl", "LinReg.jld2"), "scale_qclip")[:, 1]
    @printf("\n== member steps at leads %d-%d below the training range of the level (clip floor of LinReg1_pl), %s\n",
            first(w), last(w), prefix)
    tr = nothing
    for n in names
        fc, tr = assemble(load_members(joinpath(OUTD, prefix * n)), truth, 1:last(w))
        sh = [count(x -> !isnan(x) && x < lo[i], view(fc, :, i, :, w)) / count(!isnan, view(fc, :, i, :, w)) for i in eachindex(lo)]
        @printf("   %-20s %s\n", n, fmtq(sh))
    end
    tr === nothing || @printf("   %-20s %s\n", "reference", fmtq([count(<(lo[i]), view(tr, :, i, w)) / length(view(tr, :, i, w)) for i in eachindex(lo)]))
end

if abspath(PROGRAM_FILE) == @__FILE__
    full = "--full" in ARGS
    names = filter(!=("--full"), ARGS)
    prefix = full ? "D6_" : "D6train_"
    truth = load_truth()
    res = Dict(name => diagnose(name; truth, prefix) for name in names)
    if !full
        println("\nDEFECT RULE (a diverged member, or a ratio <= 0.5 TU more than a factor 2 from the partner's)")
        for name in names
            bad, why = defect(name, res)
            @printf("   %-20s %-4s %s\n", name, bad ? "YES" : "no", why)
        end
    end
    pairs_ = [(n, PARTNER[n]) for n in names if haskey(PARTNER, n) && PARTNER[n] in names]
    for (n, p) in pairs_
        spread_error(n, p; truth, prefix)
    end
    isempty(pairs_) || below_range(names; truth, prefix)
end
