# Paper Sec. 7.3 (results, step 3a; 2026-10-26): the numbers of the CNN closures that score_d6.jl,
# train_hindcast.jl, longrun_stats.jl and gate_census.jl do not print (as baseline_extras.jl for Sec. 4).
#
# 1. Where the hindcast gap to LinReg1 lies, split by the range of the training window. Offline (Sec. 6,
#    `sec:net-colour`) the CNN departs from LinReg1 most at the steps whose level lies outside its 1-10
#    TU range; this asks whether the coupled gap sits at the ICs whose reference leaves that range.
#    An IC is "outside" if the reference level leaves the 1-10 TU range of q* in any QoI at any step of
#    the CRPS_0.5 window (leads 1-200), "inside" otherwise. The range is the clip range of step 2's scale
#    (`TO_LRS/LinReg1_pl` `scale_qclip`, q* over steps 400-4000), as `train_hindcast.jl`'s below-range
#    share. Per closure, paired with LinReg1 as `score_d6.jl --compare` pairs them:
#      ICs        the number inside and outside (and of the outside ones, how many go below / above)
#      CRPS_0.5   the paired difference in % of LinReg1's within each group, with the IC-block interval
#                 over that group's ICs, and the share of the total difference from the outside ICs
#      error      the rmse of the ensemble mean / LinReg1's per QoI in each group, at lead 25 and pooled
#                 over leads 1-200
# 2. The deployed residual of each closure (`StochLSTM_seed1.jld2`: sds exp(bd), correlation LR LR'):
#    its sds against those of the MVG refitted on the same network (CNN (RMSE) or CNN (CRPS) + MVG), and
#    the correlation of Z and E within each band (for the AR(2) those of its innovation, for the power
#    law those of Sigma_eps, so their sd ratios are not the size of the drawn residual).
#
# Usage (from lib/RikFlow): julia --project=analysis analysis/step3a_extras.jl [closure ...]
# (default: the six step-3a closures; reads exp_square_HIT/output/D6_<closure>, D6_LinReg1 and
# exp_square_HIT/output/TO_LSTM/step3/closures/<closure>)

include(joinpath(@__DIR__, "score_d6.jl"))        # paired_load, paired_primary, ic_block_bootstrap_ci, LABELS

const OUTD = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
const CLOS = joinpath(OUTD, "TO_LSTM", "step3", "closures")
const CL = isempty(ARGS) ? ("CNN_rmse_mvg", "CNN_crps_mvg", "CNN_crps_head", "CNN_crps_headcorr", "CNN_rmse_ar2n",
                            "CNN_rmse_pl") : Tuple(String.(ARGS))
const W05 = PRIMARY_MAX_LEAD                        # leads 1-200 (0.5 TU)

"rmse of the ensemble mean over the ICs `ics` and the leads `js`, per QoI (NaN members skipped)."
function em_rmse(fc, tr, ics, js)
    nq = size(tr, 2)
    return [sqrt(mean((mean(filter(!isnan, view(fc, k, i, :, j))) - tr[k, i, j])^2 for k in ics, j in js))
            for i in 1:nq]
end

fmt6(v) = join((@sprintf("%5.2f", x) for x in v), " ")

function range_split(truth)
    clip = load(joinpath(OUTD, "TO_LRS", "LinReg1_pl", "LinReg.jld2"), "scale_qclip")      # nq x 2
    lo, hi = clip[:, 1], clip[:, 2]
    @printf("1. 1-10 TU range of q* (LinReg1_pl scale_qclip):\n   low  %s\n   high %s\n",
            join((@sprintf("%s %.4g", LABELS[i], lo[i]) for i in eachindex(lo)), "  "),
            join((@sprintf("%s %.4g", LABELS[i], hi[i]) for i in eachindex(hi)), "  "))
    for c in CL
        pl = paired_load(joinpath(OUTD, "D6_" * c), joinpath(OUTD, "D6_LinReg1"); truth, nsteps = W05)
        r = paired_primary(joinpath(OUTD, "D6_" * c), joinpath(OUTD, "D6_LinReg1"); truth, pl, io = devnull)
        tr = pl.tr[:, :, 1:W05]
        below = [any(tr[k, i, j] < lo[i] for i in axes(tr, 2), j in axes(tr, 3)) for k in axes(tr, 1)]
        above = [any(tr[k, i, j] > hi[i] for i in axes(tr, 2), j in axes(tr, 3)) for k in axes(tr, 1)]
        out = below .| above
        d = r.per_ic_a .- r.per_ic_b
        @printf("\n== %s against LinReg1: K = %d ICs, %d inside, %d outside (%d below, %d above, %d both)\n",
                c, length(d), count(!, out), count(out), count(below), count(above), count(below .& above))
        @printf("   CRPS_0.5 all      %+.1f %% of LinReg1's\n", 100 * mean(d) / mean(r.per_ic_b))
        for (lab, g) in (("inside", .!out), ("outside", out))
            ci = ic_block_bootstrap_ci(d[g], r.t[g])
            sb = mean(r.per_ic_b[g])
            @printf("   CRPS_0.5 %-8s %+.1f %% [%+.1f, %+.1f] of LinReg1's %.4f there (A %.4f)\n", lab,
                    100 * ci.mean / sb, 100 * ci.lo / sb, 100 * ci.hi / sb, sb, mean(r.per_ic_a[g]))
        end
        @printf("   share of the summed difference from the outside ICs %.2f (they are %.2f of the ICs)\n",
                sum(d[out]) / sum(d), count(out) / length(d))
        for (lab, g) in (("inside", findall(.!out)), ("outside", findall(out)))
            for (jl, js) in (("lead 25 ", 25:25), ("1-200   ", 1:W05))
                ea = em_rmse(pl.fa, pl.tr, g, js)
                eb = em_rmse(pl.fb, pl.tr, g, js)
                @printf("   error / LinReg1's %-8s %s %s\n", lab, jl, fmt6(ea ./ eb))
            end
        end
    end
end

"sds and correlation of a deployed closure's residual (the head `D LR ε`, D = diag(exp(bd)))."
function head_of(c)
    w = load(joinpath(CLOS, c, "StochLSTM_seed1.jld2"), "weights")
    LR = Float64.(w.LR)
    return (; sd = exp.(Float64.(w.bd)), R = LR * LR')
end

function heads()
    println("\n2. Deployed residual: sds against the MVG refitted on the same network, and corr(Z, E) per band")
    ref = Dict("rmse" => head_of("CNN_rmse_mvg"), "crps" => head_of("CNN_crps_mvg"))
    for c in CL
        h = head_of(c)
        b = occursin("crps", c) ? "crps" : "rmse"
        @printf("   %-18s sd / CNN_%s_mvg's %s | corr(Z, E) [0,6] [7,15] [16,32] %s\n", c, b,
                fmt6(h.sd ./ ref[b].sd), fmt6([h.R[1, 2], h.R[3, 4], h.R[5, 6]]))
    end
end

range_split(load_truth())
heads()
