# Generate the Snellius fit grid (stage A) from its declarative spec.
#
#     julia --startup-file=no exp_square_HIT/tools/p4grid_generate.jl [spec.toml] [fits.csv]
#
# Defaults: exp_square_HIT/batch_scripts/p4grid/spec.toml -> .../p4grid/fits.csv. Stdlib only (TOML,
# Printf), so it runs anywhere, in any project. Prints the row counts and the exact `sbatch --array`
# for `batch_scripts/run_p4grid_fit.sh`.
#
# One CSV row = one fit = one array task. Columns:
#   row          1-based, = SLURM_ARRAY_TASK_ID
#   cell         m0 | m3f | m0v
#   h, lambda, wd, nh, seed     the grid coordinates (wd / nh / seed empty where they do not apply)
#   tag          the fit's name, built from the coordinates, never by hand
#   outdir       where the tool writes it, relative to exp_square_HIT/output/TO_LSTM
#   matched_m0   outdir of the matched M0@50 (same h, same lambda) -- the paired baseline
#   tool         tools/<tool>.jl
#   status       ok | blocked:<reason> (a blocked row is not run; see the spec)
#   env          space-separated KEY=VALUE assignments for the tool (no value contains a space)
#
# Rows are ordered m0, m3f, m0v, blocked last, so the runnable rows are one contiguous range.

using TOML, Printf

const EXP = normpath(joinpath(@__DIR__, ".."))
spec_file = length(ARGS) >= 1 ? ARGS[1] : joinpath(EXP, "batch_scripts", "p4grid", "spec.toml")
csv_file = length(ARGS) >= 2 ? ARGS[2] : joinpath(dirname(spec_file), "fits.csv")

"Number formatting shared by every tag -- `%g`, which is also what m4_linear_eta.jl names its dirs with."
g(x) = @sprintf("%g", x)

S = TOML.parsefile(spec_file)
meta = S["meta"]
outsub = meta["outsub"]
train_tu, score_tu, seeds = meta["train_tu"], meta["score_tu"], Int.(meta["seeds"])
occursin(r"^\s*[0-9.]+\s*,\s*[0-9.]+\s*$", score_tu) || error("meta.score_tu must be \"a,b\"; got $score_tu")
parse(Float64, split(score_tu, ",")[2]) <= 74 || error("🔒 meta.score_tu reaches past 74 TU")

"The `[cell.hN]` sub-tables of a cell as (h, table), sorted by h."
function per_h(cell)
    out = Tuple{Int,Dict}[]
    for (k, v) in S[cell]
        m = match(r"^h(\d+)$", k)
        m === nothing || push!(out, (parse(Int, m[1]), v))
    end
    isempty(out) && error("[$cell] has no [$cell.hN] sub-table")
    return sort(out; by = first)
end
envstr(d) = join(("$k=$v" for (k, v) in sort(collect(d); by = first)), " ")
function check_env(d)
    for (k, v) in d
        occursin(r"\s", string(v)) && error("env value for $k contains whitespace: $(repr(v))")
    end
end

rows = NamedTuple[]
m0dir(h, lam) = "$outsub/$(S["m0"]["tagpfx"])_h$(h)_l$(g(lam))"

# --- M3ᶠ and M0ᵛ, collecting the (h, lambda) pairs their matched M0 needs ------------------------
pairs = Set{Tuple{Int,Float64}}()
m3f, m0v = NamedTuple[], NamedTuple[]
base_w = S["m3f"]["env"]; check_env(base_w)
for (h, t) in per_h("m3f"), lam in Float64.(t["lambda"]), wd in Float64.(t["wd"]), nh in Int.(t["nh"]),
    s in seeds

    tag = "m3f_h$(h)_l$(g(lam))_wd$(g(wd))_nh$(nh)_s$(s)"
    env = merge(base_w, Dict("RIKFLOW_W_TAG" => tag, "RIKFLOW_W_OUTSUB" => outsub, "RIKFLOW_W_H" => string(h),
                             "RIKFLOW_W_LAMBDA" => g(lam), "RIKFLOW_W_WD" => g(wd), "RIKFLOW_W_NH" => string(nh),
                             "RIKFLOW_W_SEED" => string(s), "RIKFLOW_W_TRAIN_TU" => string(train_tu),
                             "RIKFLOW_W_SCORE_TU" => score_tu))
    push!(pairs, (h, lam))
    push!(m3f, (; cell = "m3f", h, lambda = lam, wd = g(wd), nh = string(nh), seed = string(s), tag,
                outdir = "$outsub/$tag", matched_m0 = m0dir(h, lam), tool = S["m3f"]["tool"],
                status = "ok", env = envstr(env)))
end
base_d = S["m0v"]["env"]; check_env(base_d)
for (h, t) in per_h("m0v"), lam in Float64.(t["lambda"]), nh in Int.(t["nh"]), s in seeds
    tag = "m0v_h$(h)_l$(g(lam))_nh$(nh)_s$(s)"
    # m4_diag_fit.jl writes output/TO_LSTM/diag/<TAG> and has no subdir option: route it by the tag
    env = merge(base_d, Dict("RIKFLOW_D_TAG" => "../$outsub/$tag", "RIKFLOW_D_H" => string(h),
                             "RIKFLOW_D_NH" => string(nh), "RIKFLOW_D_SEED" => string(s),
                             "RIKFLOW_D_TRAIN_TU" => string(train_tu), "RIKFLOW_D_SCORE_TU" => score_tu))
    status = lam == 0 ? "ok" : "blocked:m4_diag_fit.jl-has-no-skip-lambda-option(RIKFLOW_D_LAMBDA)"
    push!(pairs, (h, lam))
    push!(m0v, (; cell = "m0v", h, lambda = lam, wd = "", nh = string(nh), seed = string(s), tag,
                outdir = "$outsub/$tag", matched_m0 = m0dir(h, lam), tool = S["m0v"]["tool"], status,
                env = envstr(env)))
end
m0 = NamedTuple[]
for (h, lam) in sort(collect(pairs))
    tag = "$(S["m0"]["tagpfx"])_h$(h)_l$(g(lam))"
    env = Dict("RH" => string(h), "LAMS" => g(lam), "TRAIN_TU" => string(train_tu), "SCORE_TU" => score_tu,
               "OUTSUB" => outsub, "TAGPFX" => S["m0"]["tagpfx"])
    push!(m0, (; cell = "m0", h, lambda = lam, wd = "", nh = "", seed = "", tag, outdir = "$outsub/$tag",
               matched_m0 = "$outsub/$tag", tool = S["m0"]["tool"], status = "ok", env = envstr(env)))
end
allrows = vcat(m0, m3f, m0v)
allrows = vcat(filter(r -> r.status == "ok", allrows), filter(r -> r.status != "ok", allrows))
length(unique(getfield.(allrows, :outdir))) == length(allrows) || error("two rows share an output directory")

cols = (:cell, :h, :lambda, :wd, :nh, :seed, :tag, :outdir, :matched_m0, :tool, :status, :env)
open(csv_file, "w") do io
    println(io, "row,", join(string.(cols), ","))
    for (i, r) in enumerate(allrows)
        vals = [c === :lambda ? g(r.lambda) : string(getfield(r, c)) for c in cols]
        any(v -> occursin(',', v) && v != vals[end], vals[1:(end - 1)]) && error("a comma in row $i")
        # env holds commas (FREEZE=Ws,V1, WD_EXCLUDE=bd,Araw), so it is quoted and last
        println(io, i, ",", join(vals[1:(end - 1)], ","), ",\"", vals[end], "\"")
    end
end

nok = count(r -> r.status == "ok", allrows)
@printf("wrote %s: %d rows (m0 %d, m3f %d, m0v %d), %d runnable, %d blocked%s\n", csv_file,
        length(allrows), length(m0), length(m3f), length(m0v), nok, length(allrows) - nok,
        get(meta, "provisional", false) ? "  -- ⚠️ PROVISIONAL spec values" : "")
for r in filter(r -> r.status != "ok", allrows)
    println("  blocked row: ", r.tag, "  (", r.status, ")")
end
println("submit (from exp_square_HIT/):  sbatch --array=1-$(nok)%20 batch_scripts/run_p4grid_fit.sh")
println("  packed, 8 fits per task:      P4GRID_PACK=8 sbatch --array=1-$(cld(nok, 8)) batch_scripts/run_p4grid_fit.sh")
