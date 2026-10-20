# The turbulence-gate census of the runs the paper uses (handoff_gate_off_2026-10-13, Step 1).
#
# A firing is a forecast column of `dQ` that is identically zero (#35: `online_sgs` does not return
# `q_star`, and `LinReg` gates with `dQ .= 0` on the whole vector), counted as `gate_census` in
# `exp_square_HIT/tools/d6_lib.jl` and `clamp_report` in `score_d6.jl` count it. The September D6
# baselines carry no `gate_nfired` key, so everything is counted from `dQ` here.
#
#   hindcast   out/D6_{LinReg1,LinReg7,DDN}: columns nwarm+1 .. end (lead = column - nwarm); the
#              CRPS_0.5 window is leads 1..200. A member with `diverged = true` is counted to its stop.
#   long runs  out/TO_LRS/<dir>/data_online_tsim100.0_replica{1..5}.jld2: from column 101 on, as
#              `plot_paper_secs45.jl dqacf`; t = column * dt (dt = 2.5e-3). An event is a run of
#              firings with gaps of at most EVENT_GAP steps.
#
# Usage: julia --project=analysis analysis/gate_census.jl [closure ...]
#
# 🔑 Closure names as arguments (2026-10-16, step 2): `gate_census.jl LinReg1_pl` counts out/D6_<name>
# and out/TO_LRS/<name> for those names instead of the default table (a name without one of the two
# directories is skipped there; the lambda column is then "-").

using JLD2, Printf

const OUTD = joinpath(@__DIR__, "..", "exp_square_HIT", "output")
const NAMES = String.(ARGS)
const D6_DIRS = isempty(NAMES) ? ("D6_LinReg1", "D6_LinReg6", "D6_LinReg7", "D6_DDN") :   # LinReg6 (lambda = 1e-4) added 2026-10-13
                Tuple("D6_" * n for n in NAMES if isdir(joinpath(OUTD, "D6_" * n)))
const LRS_DIRS = isempty(NAMES) ?
                 (("LinReg1", "0"), ("LinReg6", "1e-4"), ("LinReg7", "1"), ("LinReg8", "10"),
                  ("LinReg9", "100"), ("LinReg10", "1e4"), ("LinReg5", "- (not in paper)")) :
                 Tuple((n, "-") for n in NAMES if isdir(joinpath(OUTD, "TO_LRS", n)))
const CRPS05 = 200          # CRPS_0.5 window, leads 1..200
const LRS_SKIP = 100        # long runs: count from column 101 on
const DT = 2.5e-3
const EVENT_GAP = 10        # firings at most this many steps apart belong to one event

zerocols(dQ, r) = [c for c in r if all(iszero, view(dQ, :, c))]

"Group sorted columns into events: runs with gaps ≤ EVENT_GAP. Returns (first, last, n) per event."
function events(cols)
    ev = Tuple{Int,Int,Int}[]
    for c in cols
        if !isempty(ev) && c - ev[end][2] <= EVENT_GAP
            a, _, n = ev[end]
            ev[end] = (a, c, n + 1)
        else
            push!(ev, (c, c, 1))
        end
    end
    return ev
end

function census_d6(dir)
    files = filter(f -> occursin(r"^d6_online_ic\d+_m\d+\.jld2$", f), readdir(joinpath(OUTD, dir)))
    rows = NamedTuple[]
    for f in files
        m = match(r"^d6_online_ic(\d+)_m(\d+)\.jld2$", f)
        d = load(joinpath(OUTD, dir, f))
        dQ, nwarm = d["dQ"], d["nwarm"]
        dv = get(d, "diverged", false) === true
        nonfin = findfirst(c -> !all(isfinite, view(dQ, :, c)), 1:size(dQ, 2))
        last = nonfin === nothing ? size(dQ, 2) : nonfin - 1
        z = zerocols(dQ, (nwarm + 1):last)
        zw = zerocols(dQ, 1:min(nwarm, last))           # warm-up replay: must be 0
        push!(rows, (; ic = parse(Int, m[1]), mem = parse(Int, m[2]), nwarm,
                     nsteps = last - nwarm, leads = z .- nwarm, nwarmfired = length(zw),
                     diverged = dv, nonfinite = nonfin !== nothing,
                     key = get(d, "gate_nfired", missing), thr = get(d, "gate_threshold", missing)))
    end
    return rows
end

function report_d6(dir, rows)
    nf = sum(length(r.leads) for r in rows)
    ns = sum(r.nsteps for r in rows)
    hit = filter(r -> !isempty(r.leads), rows)
    ics = sort(unique(r.ic for r in hit))
    allics = unique(r.ic for r in rows)
    in05 = sum(count(<=(CRPS05), r.leads) for r in rows)
    @printf("\n== %s: %d members, %d ICs, nwarm %s\n", dir, length(rows), length(allics),
            join(unique(r.nwarm for r in rows), "/"))
    @printf("   firings %d / %d forecast steps (%.4f %%); members %d / %d; ICs %d / %d\n",
            nf, ns, 100nf / ns, length(hit), length(rows), length(ics), length(allics))
    @printf("   firings in leads 1..%d (CRPS_0.5 window): %d, in %d member(s)\n", CRPS05, in05,
            count(r -> any(<=(CRPS05), r.leads), rows))
    if !isempty(hit)
        fl = [first(r.leads) for r in hit]
        @printf("   first lead: min %d, median %d, max %d\n", minimum(fl), sort(fl)[cld(end, 2)], maximum(fl))
        nev = sum(length(events(r.leads)) for r in hit)
        @printf("   events (gap ≤ %d): %d\n", EVENT_GAP, nev)
        @printf("   ICs with firings: %s\n", join(ics, ", "))
        println("   per member (ic/m: n firings, first lead, events):")
        for r in sort(hit; by = r -> (r.ic, r.mem))
            @printf("     ic%-4d m%-2d %5d  first %4d  events %d%s\n", r.ic, r.mem, length(r.leads),
                    first(r.leads), length(events(r.leads)), r.diverged ? "  (diverged)" : "")
        end
    end
    nwf = sum(r.nwarmfired for r in rows)
    ndv = count(r -> r.diverged, rows)
    nnf = count(r -> r.nonfinite, rows)
    keyed = filter(r -> !ismissing(r.key), rows)
    @printf("   checks: warm-up zero columns %d; diverged flag %d; non-finite dQ %d; gate_nfired key on %d files%s\n",
            nwf, ndv, nnf, length(keyed),
            isempty(keyed) ? "" : @sprintf(" (agrees: %s)", all(r.key == length(r.leads) for r in keyed)))
    short = filter(r -> r.nsteps < maximum(x.nsteps for x in rows), rows)
    isempty(short) || @printf("   short members: %s\n",
                              join([@sprintf("ic%d m%d (%d steps)", r.ic, r.mem, r.nsteps) for r in short], ", "))
    return (; nf, ns, members = length(hit), nmem = length(rows), ics = length(ics), nics = length(allics), in05)
end

function report_lrs(dir, lam)
    pth = joinpath(OUTD, "TO_LRS", dir)
    files = sort(filter(f -> occursin(r"^data_online_tsim100\.0_replica\d\.jld2$", f), readdir(pth)))
    isempty(files) && return nothing
    @printf("\n== TO_LRS/%s (lambda = %s): %d replicas\n", dir, lam, length(files))
    tot, totn, nrep, allev = 0, 0, 0, 0
    for f in files
        dQ = jldopen(io -> io["data_online"].dQ, joinpath(pth, f))
        nonfin = findfirst(c -> !all(isfinite, view(dQ, :, c)), 1:size(dQ, 2))
        last = nonfin === nothing ? size(dQ, 2) : nonfin - 1
        z = zerocols(dQ, (LRS_SKIP + 1):last)
        zs = zerocols(dQ, 1:LRS_SKIP)
        ev = events(z)
        tot += length(z); totn += last - LRS_SKIP; nrep += !isempty(z); allev += length(ev)
        @printf("   %s: %d / %d steps, %d event(s)%s%s%s\n", replace(f, "data_online_tsim100.0_" => ""),
                length(z), last - LRS_SKIP, length(ev),
                isempty(z) ? "" : @sprintf(", first at t = %.4f TU", first(z) * DT),
                isempty(zs) ? "" : @sprintf("  [%d zero columns in 1..%d]", length(zs), LRS_SKIP),
                nonfin === nothing ? "" : @sprintf("  🔴 non-finite from column %d (t = %.2f)", nonfin, nonfin * DT))
        for (a, b, n) in ev
            @printf("       event t = %.4f-%.4f TU (columns %d-%d), %d firing(s)\n", a * DT, b * DT, a, b, n)
        end
    end
    @printf("   total %d / %d steps (%.4f %%); replicas with firings %d / %d; events %d\n",
            tot, totn, 100tot / totn, nrep, length(files), allev)
    return (; tot, totn, nrep, nfiles = length(files), allev)
end

println("Gate census -- firing = identically-zero forecast dQ column (#35)")
s6 = Dict(d => report_d6(d, census_d6(d)) for d in D6_DIRS)
sl = Dict(d => report_lrs(d, l) for (d, l) in LRS_DIRS)

println("\n== Summary")
for d in D6_DIRS
    s = s6[d]
    @printf("  %-12s %6d / %7d (%.3f %%)  members %3d/%d  ICs %2d/%d  in CRPS_0.5 window %d\n",
            d, s.nf, s.ns, 100s.nf / s.ns, s.members, s.nmem, s.ics, s.nics, s.in05)
end
for (d, _) in LRS_DIRS
    s = sl[d]; s === nothing && continue
    @printf("  %-12s %6d / %7d (%.3f %%)  replicas %d/%d  events %d\n",
            "LRS " * d, s.tot, s.totn, 100s.tot / s.totn, s.nrep, s.nfiles, s.allev)
end
