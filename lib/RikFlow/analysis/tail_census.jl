# Low-energy tail census of D6 runs (results_LSTMS §13, 2026-09-28): which QoI goes low, how often,
# and whether it recovers over the forecast.
#
#     julia --startup-file=no --project=training analysis/tail_census.jl <dir> [<dir> ...] [--ics <dir>]
#
# Per directory (relative to analysis/output or absolute), over members whose run ends by 74 TU (🔒):
#  - per QoI and lead: members with q < 0.5x / 0.25x the tracked truth, and the mean level bias
#    (q − truth) / sd(truth over 1–74 TU);
#  - per member: the QoI with the lowest q/truth at the last lead ("which channel drains first");
#  - recovery: of the members low (< 0.5x in any QoI) at lead 1 TU, how many are still low at the
#    last lead.
# `--ics <dir>` restricts every directory to the (k, member) pairs present in <dir>, so runs of
# different length or member count are compared on the same members.
# The q-to-truth column offset is not assumed: it is taken from the replayed warm-up, where q must
# equal the tracked record, and asserted.

include(joinpath(@__DIR__, "m0c_checks.jl"))       # QR, guard, DT, NQ, LABELS, T_MAX_READ, steps_of

resolve(d) = isabspath(d) ? d : joinpath(HERE, "output", d)
members(dir) = sort(filter(f -> occursin(r"^d6_online_ic\d+_m\d+\.jld2$", f), readdir(dir)))

const SDQ = vec(std(QR[:, guard(steps_of(1, 74))]; dims = 2))

"Column offset o with q[:, c] == QR[:, n_k + c + o] on the warm-up, asserted to round-off."
function q_offset(q, nk, nw)
    for o in -2:2
        c = 2:nw
        err = maximum(abs.(q[:, c] .- QR[:, guard(nk .+ c .+ o)]) ./ abs.(QR[:, nk .+ c .+ o]))
        err < 1e-6 && return o
    end
    error("no warm-up column offset in -2:2 matches the tracked record")
end

function census(dir; leads = (100, 200, 400, 800, 1200), keep = nothing, io = stdout)
    nl = length(leads)
    low5 = zeros(Int, NQ, nl); low25 = zeros(Int, NQ, nl); bias = zeros(NQ, nl); nlead_ok = zeros(Int, nl)
    firstlow = zeros(Int, NQ)                  # members whose worst QoI at their last lead is i (if < 0.5x)
    low1 = 0; low1_still = 0; n = 0; ndiv = 0; ngate = 0
    for f in members(dir)
        keep === nothing || f in keep || continue
        d = load(joinpath(dir, f))
        (d["t_k"] + d["tsim"] <= T_MAX_READ) || continue
        n += 1
        q, nk, nw = d["q"], d["n_k"], d["nwarm"]
        # pre-2026-09-28 files (full D6) carry neither key: derive them
        div = get(d, "diverged", !all(isfinite, q))
        ndiv += div
        ngate += get(d, "gate_nfired", count(c -> all(iszero, view(d["dQ"], :, c)), (nw + 1):size(d["dQ"], 2)))
        div && continue
        o = q_offset(q, nk, nw)
        last_ok = 0
        lowat = Dict{Int,Bool}()
        for (j, L) in enumerate(leads)
            c = nw + 1 + L
            c <= size(q, 2) || continue
            tr = QR[:, guard([nk + c + o])][:, 1]
            r = q[:, c] ./ tr
            low5[:, j] .+= r .< 0.5; low25[:, j] .+= r .< 0.25
            bias[:, j] .+= (q[:, c] .- tr) ./ SDQ
            nlead_ok[j] += 1; last_ok = j
            lowat[L] = any(r .< 0.5)
        end
        cL = nw + 1 + leads[last_ok]
        rL = q[:, cL] ./ QR[:, guard([nk + cL + o])][:, 1]
        minimum(rL) < 0.5 && (firstlow[argmin(rL)] += 1)
        if get(lowat, 400, false)
            low1 += 1
            low1_still += lowat[leads[last_ok]]
        end
    end
    @printf(io, "\n%s: %d members ending by %g TU, diverged %d, gate firings %d\n", basename(dir), n, T_MAX_READ, ndiv, ngate)
    @printf(io, "  %-9s | %s\n", "QoI", join([@sprintf("lead %4d (%4.2f TU): <0.5x <0.25x  bias", L, L * DT) for (j, L) in enumerate(leads) if nlead_ok[j] > 0], " | "))
    for i in 1:NQ
        @printf(io, "  %-9s | %s\n", LABELS[i],
                join([@sprintf("%21d %6d %+6.2f", low5[i, j], low25[i, j], bias[i, j] / nlead_ok[j]) for j in 1:nl if nlead_ok[j] > 0], " | "))
    end
    @printf(io, "  worst QoI at the last lead, members below 0.5x: %s\n",
            join(["$(LABELS[i]) $(firstlow[i])" for i in 1:NQ], ", "))
    @printf(io, "  low (<0.5x, any QoI) at lead 400: %d members; still low at the last lead: %d\n", low1, low1_still)
    return (; n, ndiv, ngate, low5, low25, bias = bias ./ max.(nlead_ok', 1), firstlow, low1, low1_still, leads, nlead_ok)
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = copy(ARGS)
    keep = nothing
    if (i = findfirst(==("--ics"), args)) !== nothing
        keep = Set(members(resolve(args[i + 1])))
        deleteat!(args, i:(i + 1))
    end
    for d in args
        census(resolve(d); keep)
    end
end
