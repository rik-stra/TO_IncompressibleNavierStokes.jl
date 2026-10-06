# R2: the long-run mean bias of two closures in ONE metric, and the control's pass rule
# (`results_LSTMS.md` §14, `paper/todo.md` R2-0 / R2-3).
#
#     julia --project=lib/RikFlow/analysis lib/RikFlow/analysis/r2_bias.jl <dir A> [<dir B>]
#
# Each dir holds `data_online_tsim100.0_replica<i>.jld2` (relative dirs are taken under
# `exp_square_HIT/output/`), e.g. `TO_LRS/LinReg1` and `TO_LSTM/diag/control_LinReg1`.
#
# Metric, per replica and QoI: (mean of the level over the window - the reference's 100 TU mean) /
# the reference's 100 TU sd -- the unit of `m4_online_moments.jl` -- and the ratio of means - 1 in %
# (`results.md`'s "6-9 % low"). Windows: 0-20 TU (columns 1:8001, the §7d network runs' window) and
# 0-100 TU. The reference is the HF reference; the tracked record agrees with it to < 0.7 % of an sd.
#
# 🔒 Pass rule for B (the control) against A (LinReg1), fixed before the control runs existed
# (2026-10-06): in BOTH windows and in EVERY QoI, the 5-replica mean offset has A's sign and the
# replica ranges [min, max] of A and B overlap.

using JLD2, Statistics, Printf
out = normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "output"))
dirs = [isabspath(d) ? d : joinpath(out, d) for d in ARGS]
1 <= length(dirs) <= 2 || error("usage: r2_bias.jl <dir A> [<dir B>]")
qr = load(joinpath(@__DIR__, "data", "hf_reference_new_tsim100.0_f64_lmwray3_qois.jld2"), "q_ref")
mu, sd = vec(mean(qr; dims = 2)), vec(std(qr; dims = 2))
names = ("Z[0,6]", "E[0,6]", "Z[7,15]", "E[7,15]", "Z[16,32]", "E[16,32]")
windows = (("0-20 TU", 1:8001), ("0-100 TU", 1:40001))
fmt(x; d = 2) = join((@sprintf("%9.*f", d, v) for v in x), "")

function offsets(dir)
    files = sort(filter(f -> occursin(r"^data_online_tsim100\.0_replica\d+\.jld2$", f), readdir(dir)))
    isempty(files) && error("no data_online_tsim100.0_replica*.jld2 in $dir")
    res = Dict{String,Matrix{Float64}}()       # window => 6 x n_replicas, in reference sd
    pct = Dict{String,Matrix{Float64}}()
    for (lab, cols) in windows
        res[lab] = zeros(6, length(files)); pct[lab] = zeros(6, length(files))
    end
    for (r, f) in enumerate(files)
        q = jldopen(io -> io["data_online"].q, joinpath(dir, f))
        size(q, 2) == 40001 && all(isfinite, q) || @warn "$f: $(size(q, 2)) columns, finite = $(all(isfinite, q))"
        for (lab, cols) in windows
            m = vec(mean(q[:, cols]; dims = 2))
            res[lab][:, r] .= (m .- mu) ./ sd
            pct[lab][:, r] .= 100 .* (m ./ mu .- 1)
        end
    end
    return res, pct, files
end

R = [offsets(d) for d in dirs]
println(rpad("", 30), join((lpad(n, 9) for n in names), ""))
for (lab, _) in windows
    println("== $lab")
    for (d, (res, pct, files)) in zip(dirs, R)
        println("  ", relpath(d, out), "  ($(length(files)) replicas)")
        for r in axes(res[lab], 2)
            @printf("    %-24s %s   | ratio-1 %%: %s\n", "r$r offset (sd)", fmt(res[lab][:, r]), fmt(pct[lab][:, r]; d = 1))
        end
        @printf("    %-24s %s\n", "mean", fmt(vec(mean(res[lab]; dims = 2))))
    end
end

length(dirs) == 2 && let pass = true      # `let`: soft scope (claude_memory.md #47)
    for (lab, _) in windows
        a, b = R[1][1][lab], R[2][1][lab]
        sgn = sign.(vec(mean(a; dims = 2))) .== sign.(vec(mean(b; dims = 2)))
        ovl = (vec(minimum(a; dims = 2)) .<= vec(maximum(b; dims = 2))) .&
              (vec(minimum(b; dims = 2)) .<= vec(maximum(a; dims = 2)))
        @printf("%-9s same sign %s | ranges overlap %s\n", lab, join(sgn, " "), join(ovl, " "))
        pass &= all(sgn) & all(ovl)
    end
    println(pass ? "R2 control: PASS (B matches A's long-run bias)" : "R2 control: FAIL")
end
