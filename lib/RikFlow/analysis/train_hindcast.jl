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
# Usage (from lib/RikFlow):
#   julia --project=analysis analysis/train_hindcast.jl LinReg1 LinReg6 LinReg7 ...
# reads exp_square_HIT/output/D6train_<name>.


include(joinpath(@__DIR__, "score_d6.jl"))
using Printf, Statistics

const OUTD = normpath(joinpath(HERE, "..", "exp_square_HIT", "output"))
const LAB = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
const WIN = (("<=0.5 TU", 1:200), ("0.5-3 TU", 201:1200))
# matched partners (analysis/cards/round1.md); the baselines have none here
const PARTNER = Dict("LinReg7_ar2n" => "LinReg7", "LinReg7_ar1ls" => "LinReg7", "Splice1_E0x11" => "LinReg1", "Splice1_E0x7" => "LinReg1",
                     "Splice1_E0x8" => "LinReg1", "Splice1_E0x11_ar2n" => "Splice1_E0x11",
                     "Splice1_E0x7_ar2n" => "Splice1_E0x7", "Splice1_E0x8_ar2n" => "Splice1_E0x8",
                     "LinReg14_ar2n" => "Splice1_E0x11_ar2n", "LinReg16_ar2n" => "Splice1_E0x7_ar2n")

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

if abspath(PROGRAM_FILE) == @__FILE__
    truth = load_truth()
    res = Dict(name => diagnose(name; truth) for name in ARGS)
    println("\nDEFECT RULE (a diverged member, or a ratio <= 0.5 TU more than a factor 2 from the partner's)")
    for name in ARGS
        bad, why = defect(name, res)
        @printf("   %-20s %-4s %s\n", name, bad ? "YES" : "no", why)
    end
end
