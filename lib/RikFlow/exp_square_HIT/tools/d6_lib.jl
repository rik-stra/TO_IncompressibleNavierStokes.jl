# D6's closure layer -- what `tools/run_d6.jl` deploys for `D6_CLOSURE=lstm`, in a file the offline
# suite can load.
#
# 🔑 Why a separate file: `run_d6.jl` loads CUDA and IncompressibleNavierStokes, which the offline
# suite (`test/Project.toml`) deliberately cannot, so nothing inside the driver is ever exercised
# locally (gotcha #47). Everything here is stdlib + JLD2 and refers to `StochLSTM`,
# `load_stochlstm` and `get_next_item_timeseries` by their bare names: in the driver they come from
# `using RikFlow`, in `test/test_d6_lstm.jl` from the bare `ts_*.jl` includes. The same code runs in
# both, which is the point.
#
# ⚠️ The deployed M4 closure is stdlib (`src/ts_lstm_online.jl`); **Lux is not needed to forecast**,
# only to train. So the driver runs under `--project=lib/RikFlow` for every closure, LSTM included --
# see RUNBOOK.md, "Running a StochLSTM (M4) fit".

using Random
using JLD2

"The closures `run_d6.jl` can deploy."
const D6_CLOSURES = (:lrs, :ddn, :lstm)

"""
    parse_closure(s) -> Symbol

`D6_CLOSURE`'s value as one of `D6_CLOSURES`. `stochlstm` and `m4` are accepted for `lstm`.
"""
function parse_closure(s::AbstractString)
    c = lowercase(strip(s))
    c in ("stochlstm", "m4") && (c = "lstm")
    Symbol(c) in D6_CLOSURES ||
        error("D6_CLOSURE must be one of $(join(D6_CLOSURES, ", ")) (or stochlstm/m4); got $(repr(s))")
    return Symbol(c)
end

"""
    resolve_lstm_model(spec; root) -> (; file, dir, deploy_seed, name)

Where the M4 fit named by `spec` (`D6_MODEL`) lives, resolved the way `12_online_StochLSTM.jl`
does it:

  * a **directory** (absolute, relative to the working directory, or relative to `root` =
    `exp_square_HIT/output/TO_LSTM`, e.g. `diag/r3_lin_sd_h2`): the fit is
    `StochLSTM_seed<s>.jld2` with `s` the directory's `seed_summary.jld2` `median_seed` when that
    file exists, else 1 (S6: the MEDIAN-seed fit goes online, not seed 1);
  * a **file**: deployed as is; the seed is read off a `StochLSTM_seed<s>.jld2` name, else 0.

`name` is the directory relative to `root` when it is under it (`diag/r3_lin_sd_h2`), else its
basename. It is what the output files record as `model_name` and what `d6_run_identity` compares.
"""
function resolve_lstm_model(spec::AbstractString; root::AbstractString)
    s = strip(spec)
    isempty(s) && error("D6_CLOSURE=lstm needs D6_MODEL: a fit directory under $root " *
                        "(e.g. diag/r3_lin_sd_h2) or a StochLSTM_seed<s>.jld2 file")
    p = ispath(s) ? abspath(s) : joinpath(root, s)
    ispath(p) || error("D6_MODEL = $(repr(s)) is neither a path nor a directory under $root")
    if isdir(p)
        dir = rstrip(p, '/')
        summary = joinpath(dir, "seed_summary.jld2")
        deploy_seed = isfile(summary) ? Int(load(summary, "median_seed")) : 1
        file = joinpath(dir, "StochLSTM_seed$(deploy_seed).jld2")
        isfile(file) || error("no fitted model at $file; $dir holds " *
                              join(filter(f -> endswith(f, ".jld2"), readdir(dir)), ", "))
    else
        file = p
        dir = dirname(p)
        m = match(r"StochLSTM_seed(\d+)\.jld2$", basename(p))
        deploy_seed = m === nothing ? 0 : parse(Int, m[1])
    end
    rroot = rstrip(abspath(root), '/')
    name = startswith(dir, rroot * "/") ? relpath(dir, rroot) : basename(dir)
    return (; file, dir, deploy_seed, name)
end

"""
    lstm_record(fit) -> NamedTuple

What the output file records about a deployed M4 fit, beside `model`/`model_name`: the
architecture, the target, and which of the deploy-time scaling knobs (`noise_scale`, `dq_offset`,
`offset_ref`, `eta_ar`, `emission_scale`, `input_map`) the fit carries. The knobs live in the fit's
`scaling` and are applied by `StochLSTM` itself, so D6 deploys a calibrated or coloured fit exactly
as `12_online_StochLSTM.jl` does, with no environment variable in between.
"""
function lstm_record(fit)
    sc = fit.scaling
    knobs = [string(k) for k in (:noise_scale, :dq_offset, :offset_ref, :eta_ar, :emission_scale,
                                 :input_map) if hasproperty(sc, k)]
    return (; lstm_arch = string(fit.spec.arch), lstm_emission = string(fit.spec.emission),
            lstm_window = fit.spec.window, lstm_skip = fit.spec.skip,
            lstm_target = string(hasproperty(sc, :target) ? sc.target : :q),
            lstm_knobs = join(knobs, ","))
end

"""
    make_lstm_sampler(fit, dQ_warm, seed; gate)

The M4 closure for one D6 member, under the same contract as `LinReg` in `run_d6.jl`:

  * **the same `dQ_warm`**, replayed verbatim for `size(dQ_warm, 2)` steps and returned
    **unconverted** -- the validation gate is `dQ[:, 1:nwarm]` bit-identity (#48). `StochLSTM` is
    CPU-side, so it is handed a host `Array` of the package's own element type, not `ArrayType{T}`;
  * **the member seed** `Xoshiro(seed)`, and the replay draws nothing from it (V38/V42), so a member
    seed means the same thing as for `LinReg` and `MVG_sampler`;
  * 🔴 **the turbulence gate, passed explicitly.** `gate` must be `RikFlow.TURBULENCE_GATE`, the
    constant `LinReg`'s four gate sites read. `StochLSTM` applies it at the same point `LinReg`
    does -- on the final correction, after the output map, before the step is pushed into the
    history -- and not during the replay, again as `LinReg`. Passing it rather than relying on the
    constructor's `1e-2` default keeps one number in one place (gotchas #59/#65: the DDN lacked
    the gate and that confounded `results.md` §4c).

Returns the `StochLSTM`. `tie_noise` stays at its default `true`; the untied variant is a
diagnostic ablation and is not deployable through D6.
"""
function make_lstm_sampler(fit, dQ_warm::AbstractMatrix, seed::Integer; gate::Real)
    fit.spec.hist.n_qoi == size(dQ_warm, 1) ||
        error("the M4 fit has N_Q = $(fit.spec.hist.n_qoi) but the IC package's warm-up has " *
              "$(size(dQ_warm, 1)) rows")
    return StochLSTM(fit.spec, fit.weights, fit.scaling; spinnup_data = Array(dQ_warm),
                     rng = Xoshiro(seed), gate)
end

"""
    gate_census(dQ, nwarm) -> (; nfired, nsteps, first_lead)

How often the turbulence gate fired over the forecast, counted exactly as `score_d6.jl`'s
`clamp_report` and `m4_screen.jl` count it: a forecast column of `dQ` that is **identically zero**.
Both `LinReg` and `StochLSTM` gate by `dQ .= 0` on the whole vector, so the census means the same
thing for either closure. The replayed warm-up (`1:nwarm`) is excluded. `first_lead` is the lead
(steps past the warm-up) of the first firing, `0` if none.
"""
function gate_census(dQ::AbstractMatrix, nwarm::Integer)
    nfired, first_lead = 0, 0
    for c in (nwarm + 1):size(dQ, 2)
        if all(iszero, view(dQ, :, c))
            nfired += 1
            first_lead == 0 && (first_lead = c - nwarm)
        end
    end
    return (; nfired, nsteps = max(size(dQ, 2) - nwarm, 0), first_lead)
end

"""
    check_out_dir(od, identity)

Refuse to write into `od` if it already holds D6 members of a different experiment (closure, model,
block or forecast length; see `d6_run_identity`). The scorer refuses such a directory too, but only
after the GPU time is spent.
"""
function check_out_dir(od::AbstractString, identity)
    isdir(od) || return nothing
    pat = r"^d6_online_ic\d+_m\d+\.jld2$"
    for f in readdir(od)
        occursin(pat, f) || continue
        other = d6_run_identity(load(joinpath(od, f)))
        other == identity || error(
            "D6_OUT = $od already holds members of a different experiment: $f is $other, this " *
            "run is $identity. One directory per closure, model, block and N_LEAD -- the " *
            "scorer's glob would pool them.")
        return nothing               # one file suffices: the directory was checked when it was written
    end
    return nothing
end
