module RikFlow
#= Implementation of tau-orthogonal method

=#
using IncompressibleNavierStokes
using FFTW
using Observables
using JLD2
using Infiltrator
#using TensorOperations
#import cuTENSOR
using KernelAbstractions
using LinearAlgebra
using Statistics
using Distributions
using Random
using Adapt
using CUDA
#using MLUtils
#using Lux, LuxCUDA
#using Optimisers, Zygote


# ---------------------------------------------------------------------------------------------
# Bridge to upstream IncompressibleNavierStokes >= 5.
#
# Upstream's `Setup` is a pure grid description: `Setup(; x, boundary_conditions, backend,
# workgroupsize)`, with no `Re`, `closure_model`, `bodyforce`, `temperature` or `ArrayType`, and
# with the grid fields hoisted to the top level instead of living under `setup.grid`. Physics moved
# to `params`, forcing moved to `force!` and its cache.
#
# 🔑 That setup is a plain `NamedTuple`, so RikFlow can extend it at its own call sites rather than
# forking `Setup`. `rf_setup` does exactly that: upstream's setup, plus the three fields our code
# genuinely needs. This is what let `src/setup.jl` be deleted in the merge instead of carried.
#
#   `Re`             — RikFlow reads it for the element type (`typeof(setup.Re)`) in a dozen places
#                      and as the physical parameter in `rf_params`. Kept as the single source of
#                      truth; `viscosity` is derived from it, never stored beside it.
#
# ⚠️ Only `Re`. Anything added here is passed into GPU kernels by upstream's operators and must be
# isbits after adaptation — `rf_arraytype` and the NaN flag in the force cache exist because
# `ArrayType` and `nans_detected` are not.
# ---------------------------------------------------------------------------------------------

"""
    rf_setup(; x, Re, boundary_conditions = periodic, ArrayType = Array, backend = CPU(), workgroupsize = 64)

Upstream's `Setup` extended with the fields RikFlow needs. Use this everywhere RikFlow used to call
`Setup(; x, Re, ArrayType, backend, ou_bodyforce)`.

⚠️ `ou_bodyforce` is **not** a keyword here. OU forcing is no longer a property of the setup: build
it with `ou_force_cache(setup; T_L, e_star, k_f, rng_seed, freeze)` and pass the result to
`solve_unsteady` as `force_cache`, together with `force! = ou_navierstokes!`. See `rf_solve`.

⚠️ `boundary_conditions` is keyed by field upstream — `(; u = ((PeriodicBC(), PeriodicBC()), …))`.
The default here is periodic in every direction, which is what every HIT case uses.

⚠️ `ArrayType` is **accepted and not stored**. Call sites pass it and RikFlow's own structures
(`TO_Setup`, `OU_setup`) still need it, but it cannot live in the setup — see below. Use
[`rf_arraytype`](@ref) to recover it from a setup.
"""
function rf_setup(;
    x,
    Re,
    boundary_conditions = (; u = ntuple(d -> (PeriodicBC(), PeriodicBC()), length(x))),
    ArrayType = Array,
    backend = IncompressibleNavierStokes.CPU(),
    workgroupsize = 64,
)
    # 🔴 The grid and `Re` must agree on precision. Several drivers splat an archived
    # `params_train`/`params_track` into their own parameters, and those archives carry a Float32
    # `Re`; splatted after a Float64 grid it wins silently and the run is mixed-precision, with
    # `typeof(setup.Re)` — which drives `TO_Setup`, `OU_setup` and every QoI buffer — quietly back
    # at Float32. Caught here rather than discovered in the output.
    eltype(x[1]) === typeof(Re) || error(
        "precision mismatch: grid is $(eltype(x[1])) but Re is $(typeof(Re)). " *
        "If Re came from an archived params tuple, override it after the splat.",
    )
    base = IncompressibleNavierStokes.Setup(; x, boundary_conditions, backend, workgroupsize)
    # 🔴 `Re` only. `ArrayType` and `nans_detected` used to live here and could not: upstream's
    # rewritten operators pass the whole `setup` into GPU kernels, where every field must be
    # isbits after adaptation. CUDA adapts `CuArray` to `CuDeviceArray`, but a `UnionAll` and a
    # host `Array{Bool,0}` adapt to nothing, so the kernel refused to compile with
    #
    #     KernelError: passing non-bitstype argument ... .ArrayType is of type UnionAll
    #                  ... .nans_detected is of type Array{Bool, 0}
    #
    # Base's operators took extracted grid arrays rather than the setup, which is why the fork
    # could get away with it before. `Re::Float64` is isbits and stays.
    #
    # ⚠️ Invisible on CPU: no kernel is compiled there, so every CPU test passes regardless. Do
    # not add a non-isbits field here on the strength of a green CPU run.
    (; base..., Re)
end

"""
    rf_params(setup)

The `params` named tuple `solve_unsteady` now requires. Upstream has no default for it.

🔴 `viscosity = 1 / Re`, matching what base's `diffusion!` computed internally
(`visc = use_viscosity ? 1 / Re : one(Re)`). Storing `Re` and deriving `viscosity` in this one
place keeps a single source of truth; holding both in the setup would let them drift silently.
"""
rf_params(setup) = (; viscosity = 1 / setup.Re)

"""
    rf_arraytype(setup)

The array constructor for a setup's backend — `Array` or `CuArray` — derived from the grid rather
than stored.

Replaces the `ArrayType` field `rf_setup` used to carry. That field made the setup non-isbits and
so uncompilable as a GPU kernel argument; deriving it costs nothing and cannot leak into a kernel.
"""
rf_arraytype(setup) = Base.typename(typeof(setup.x[1])).wrapper

"""
    rf_smag_force_cache(setup; c_s, ou_bodyforce = nothing)

Force cache for [`rf_smag_navierstokes!`](@ref): upstream's Smagorinsky closure, optionally on top
of the OU body force.

Replaces the old `setup = (; setup..., closure_model = smagorinsky_closure_natural)` plus
`solve_unsteady(; θ = c_s)`. Upstream removed both: `setup` no longer carries a closure and the
stepper no longer applies one, so the closure is part of the right-hand side like any other force,
and its constant travels in the cache rather than in `θ`.

🔴 The kernels are upstream's `smagorinsky_closure!`, not the `smagorinsky_closure_natural` this
fork used to call. Rik's decision of 2026-09-11 (map section 9, Q2). They are different code
computing the same model, so the Smagorinsky baseline is not numerically the one paper 2 reports,
and by the same decision (Q3) that difference is not measured.
"""
function rf_smag_force_cache(setup; c_s, ou_bodyforce = nothing)
    ou = isnothing(ou_bodyforce) ? (;) : ou_force_cache(setup; ou_bodyforce...)
    (;
        ou...,
        c_s,
        closure_force = vectorfield(setup),
        closure_cache = IncompressibleNavierStokes.get_cache(
            IncompressibleNavierStokes.smagorinsky_closure!,
            setup,
        ),
    )
end

"""
    rf_steady_force_cache(setup, f)

Force cache holding a precomputed steady body force, for use with
[`rf_bodyforce_navierstokes!`](@ref).

Replaces `Setup(; bodyforce = f, issteadybodyforce = true)`, which upstream removed along with
`applybodyforce!`. The channel cases use this for their constant streamwise driving force.

⚠️ `f` takes `(dim, x...)` and **not** a trailing `t`: upstream builds the field with
`velocityfield`, whose `ufunc` has no time argument. A steady force does not need one, but the old
signature did take it, so existing closures must drop the final parameter.

`doproject = false` matches what `applybodyforce!` did — the force is a right-hand-side term, not
an initial condition, so it is not made divergence free.
"""
rf_steady_force_cache(setup, f) =
    (; bodyforce = IncompressibleNavierStokes.velocityfield(setup, f; doproject = false),
       nans_detected = zeros(Bool))

"""
    rf_bodyforce_navierstokes!(force, state, t; setup, cache, viscosity)

Navier-Stokes plus whatever precomputed body force sits in `cache.bodyforce`.

Identical in effect to [`ou_navierstokes!`](@ref) — both add `cache.bodyforce` — and kept separate
only so call sites say which kind of forcing they mean. The difference is not in this function but
in the cache: an OU cache also carries `ou_setup`, which is what makes `solve_unsteady` advance the
chain once per step. A steady cache has no `ou_setup`, so nothing is advanced.
"""
rf_bodyforce_navierstokes!(force, state, t; setup, cache, viscosity) =
    ou_navierstokes!(force, state, t; setup, cache, viscosity)

"""
    rf_eddyvisc_force_cache(setup; model, ou_bodyforce = nothing, bodyforce = nothing)

Force cache for [`rf_eddyvisc_navierstokes!`](@ref): one of upstream's eddy-viscosity models on top
of Navier-Stokes, optionally with an OU or steady body force as well.

`model` is an `AbstractEddyViscosity` — `Smagorinsky(C)`, `WALE(C)`, `Vreman(C)` or `QR(C)`.
Replaces `setup.closure_model = wale_closure` plus `solve_unsteady(; θ = C)`; the constant now
lives inside the model object.

🔴 Upstream's kernels, not this fork's `wale_closure`/`smagvisc2!` (Rik's Q2 decision, 2026-09-11).
A different implementation of the same models, so the channel's `C_w = 2.20` calibration is not
numerically the one paper 3 reports.
"""
function rf_eddyvisc_force_cache(setup; model, ou_bodyforce = nothing, bodyforce = nothing)
    extra = if !isnothing(ou_bodyforce)
        ou_force_cache(setup; ou_bodyforce...)
    elseif !isnothing(bodyforce)
        rf_steady_force_cache(setup, bodyforce)
    else
        (;)
    end
    (;
        extra...,
        model,
        closure_force = vectorfield(setup),
        closure_cache = IncompressibleNavierStokes.get_cache(
            IncompressibleNavierStokes.eddy_viscosity_closure!,
            setup,
        ),
    )
end

"""
    rf_eddyvisc_navierstokes!(force, state, t; setup, cache, viscosity)

Navier-Stokes plus `cache.model`'s eddy viscosity, plus `cache.bodyforce` when present.

Like the other `force!` functions here this runs once per Runge-Kutta stage and never advances the
OU chain; `solve_unsteady` does that once per step.
"""
function rf_eddyvisc_navierstokes!(force, state, t; setup, cache, viscosity)
    IncompressibleNavierStokes.navierstokes!(force, state, t; setup, cache, viscosity)
    IncompressibleNavierStokes.eddy_viscosity_closure!(
        cache.model,
        cache.closure_force,
        state.u,
        cache.closure_cache,
        setup,
    )
    force.u .+= cache.closure_force
    haskey(cache, :bodyforce) && (force.u .+= cache.bodyforce)
    nothing
end

"""
    rf_smag_navierstokes!(force, state, t; setup, cache, viscosity)

Navier-Stokes plus a Smagorinsky closure, plus the OU body force when the cache carries one.

As with [`ou_navierstokes!`](@ref), this adds the *current* forcing and never advances the OU
chain: it runs once per Runge-Kutta stage. The advance is in `solve_unsteady`, once per step.
"""
function rf_smag_navierstokes!(force, state, t; setup, cache, viscosity)
    IncompressibleNavierStokes.navierstokes!(force, state, t; setup, cache, viscosity)
    IncompressibleNavierStokes.smagorinsky_closure!(
        cache.closure_force,
        state.u,
        cache.c_s,
        cache.closure_cache,
        setup,
    )
    force.u .+= cache.closure_force
    haskey(cache, :bodyforce) && (force.u .+= cache.bodyforce)
    nothing
end

include("time_series_methods.jl")
include("filter.jl")
export FaceAverage, VolumeAverage
export rf_setup, rf_params
export rf_smag_force_cache, rf_smag_navierstokes!
export rf_steady_force_cache, rf_bodyforce_navierstokes!
export rf_eddyvisc_force_cache, rf_eddyvisc_navierstokes!
# Upstream's eddy-viscosity model objects, re-exported so channel/taylor-green scripts can
# write `WALE(c)` without reaching into IncompressibleNavierStokes by hand.
using IncompressibleNavierStokes: Smagorinsky, WALE, Vreman, QR
export Smagorinsky, WALE, Vreman, QR

include("HIT_setups/create_ref_data.jl")
export create_ref_data
export spinnup

include("HIT_setups/storage.jl")

include("HIT_setups/LFsims.jl")
export track_ref
export online_sgs

# Time-series layer. `ts_scaling.jl` replaces the former `scale.jl`: same three functions, plus a
# `Scaling` struct that carries the normalization convention as data instead of leaving it in a
# call-site constant (`meta_files/plan.md` section 0 item 6). The rest of the `ts_*` files are
# stdlib-only by design so that a test or a driver can `include` them bare, without loading
# IncompressibleNavierStokes or CUDA; that is why they need `using LinearAlgebra` above rather than
# a dependency of their own.
include("ts_scaling.jl")
export Scaling, fit_scaling, as_scaling, scaling_pair, assert_convention, is_centred
export SCALING_VERSION

include("ts_history.jl")
export HistorySpec, build_history, HistoryBuffer, inputvec

include("ts_models.jl")
export JointModel, pacf_to_ar, ar_roots, decorrelation_time

# M4's inference core. Stdlib-only on purpose -- training lives in `ext/RikFlowLuxExt.jl` and
# nothing here loads Lux. See the header of ts_lstm.jl for why the split runs this way round.
include("ts_lstm.jl")
export LSTMSpec, LSTMWeights, LSTMState, lstm_step!, sample_emission!
export latent_sampled, latent_to_cell, latent_to_decoder, emission_noise
export segment_indices, gauss_logpdf, kl_diag_gaussian, kl_to_standard_normal, iwae_bound

# M4 as a deployed closure. Also stdlib-only, and included AFTER time_series_methods.jl so that
# `needs_qstar` and `get_next_item_timeseries` already exist to take a new method.
include("ts_lstm_online.jl")
export StochLSTM

# The one M4 file that needs JLD2, hence separate from the two stdlib ones above.
include("ts_lstm_io.jl")
export save_stochlstm, load_stochlstm
# ⚠️ `reset!`, `n_input`, `n_output`, `n_cell_input`, `n_encoder_out` and `check_shapes` are
# deliberately NOT exported: all six are names Lux, NNlib or IncompressibleNavierStokes could
# plausibly define, and the training environment loads RikFlow and Lux into the same session.
# Reach them as `RikFlow.n_input(spec)`.

include("ts_fit.jl")
export fit_ridge, fit_joint

include("ts_score.jl")
export autocorr, ljung_box, residual_battery, correlation_time, ks_distance
export nll_gaussian, crps_gaussian, crps_ensemble, crps_ensemble_mean
export rank_histogram, ranks, jolliffe_primo, n_eff, block_bootstrap_indices
export summed_ks, ensemble_ks, ks_noise_floor, delta_rho, stability_fraction, spread_skill
export clamp_census, norm_cdf, norm_pdf
# #17 lead-resolved and RH-3, both of which need the multi-IC ensemble D6.
export lead_grid, union_grid, lead_positions, spread_skill_by_lead, rank_histogram_by_lead
export climatological_skill, saturation_lead

include("ts_rollout.jl")
export rollout, closed_loop_matrix, spectral_radius, trajectory_stats

include("ts_spectrum.jl")
export gram_diagnostics, pinv_gap, coefficient_blocks, total_block_sum, companion, rho
export starred_gain

include("post_processing_funcs.jl")
export ks_dist
export rf_spectral_stuff
export getspectrum
# `energy_spectra_comparison` is declared empty in `post_processing_funcs.jl` and implemented in
# `ext/RikFlowMakieExt.jl`, so that RikFlow does not drag the Makie stack into every load.
export energy_spectra_comparison
# Same pattern: declared empty here, implemented in `ext/RikFlowMakieExt.jl`.
function rf_energy_spectrum_plot end
export rf_energy_spectrum_plot


"""
Create setup for Tau-orthogonal method (stored in a named tuple).
The tuple stores
- Basis info on QoIs
    - Masks for scale aware QoIs
- Location of QoI reference trajectories
- Relevant outputs (dQ, tau)
- Pre allocated functions for V_i, masks for c_ij, which are needed for fast computation of the SGS term
"""
function TO_Setup(; qois, to_mode, ArrayType, setup, nstep = nothing, time_series_method = nothing, tracking_noise = nothing, tracking_noise_seed = 56, mirror_y = false)
    T = typeof(setup.Re)
    masks, ∂ = get_masks_and_partials(qois, setup, ArrayType, mirror_y)
    N_qois = length(qois)
    to_setup = (; N_qois, qois, to_mode, masks, ∂, time_series_method, mirror_y)
    if !isnothing(tracking_noise) && (tracking_noise == 0.0)
        tracking_noise = nothing
    end

    if to_mode in [:TRACK_REF, :ONLINE]
        V_i = get_vi_functions(to_setup)
        cij_masks = get_cij_masks(to_setup)
        outputs = allocate_arrays_outputs(;nstep, N_qois, to_mode, T)
        if !isnothing(tracking_noise)
            tracking_rng = Xoshiro(tracking_noise_seed)
        else
            tracking_rng = nothing
        end
        to_setup = (; to_setup..., V_i, cij_masks, outputs, tracking_noise, tracking_rng)
    end
    return to_setup
end


function allocate_arrays_outputs(;nstep, N_qois, to_mode, T)
    dQ = Array{T}(undef, N_qois, nstep)
    tau = Array{T}(undef, N_qois, nstep)
    dic = (;)
    if to_mode == :TRACK_REF
        q_star = Array{T}(undef, N_qois, nstep)
        dic = (; q_star)
    end
    (; dQ, tau, dic...)
end

function get_cij_masks(to_setup)
    mask_A = ones(Bool,(to_setup.N_qois, to_setup.N_qois, to_setup.N_qois))
    mask_B = ones(Bool,(to_setup.N_qois, to_setup.N_qois))
    for i in 1:to_setup.N_qois
        mask_A[:,i,i] .= false
        mask_A[i,:,i] .= false
        mask_B[i,i] = false
    end
    return (A=mask_A, B=mask_B)
end

function get_vi_functions(to_setup)
    vi = []
    for i in 1:to_setup.N_qois
        if to_setup.qois[i][1] == "E"
            f = (u,w) -> to_setup.masks[i].*u
        elseif to_setup.qois[i][1] == "Z"
            f = (u,w) -> 2*curl(to_setup.masks[i].*w, to_setup)
            #f = (u,w) -> to_setup.masks[i].*w
        end
        push!(vi, f)
    end
    return vi
end

function get_masks_and_partials(QoIs, setup, ArrayType, mirror_y)
    N = setup.Np
    Lx = setup.xlims[1][2] - setup.xlims[1][1]
    Ly = setup.xlims[2][2] - setup.xlims[2][1]
    Lz = setup.xlims[3][2] - setup.xlims[3][1]
    if mirror_y
        N = N.*[1,2,1]
        Ly = Ly*2
    end

    T = typeof(setup.Re)
    k = convert.(T,fftfreq(N[1], N[1])./Lx)
    l = convert.(T,fftfreq(N[2], N[2])./Ly)
    m = convert.(T,fftfreq(N[3], N[3])./Lz)
    
    # create a list of bolean arrays
    masks_list = [Array{Bool, length(N)}(undef,N...) for i in 1:length(QoIs)]
    #println("masks_list: ", typeof(masks_list))
    for q in 1:length(QoIs)
        for r in 1:N[3], j in 1:N[2], i in 1:N[1]
            if (k[i]^2 + l[j]^2 + m[r]^2) >= maximum([0,QoIs[q][2]-0.5])^2 && (k[i]^2 + l[j]^2 + m[r]^2) <= (QoIs[q][3]+0.5)^2
                masks_list[q][i,j,r] = true
            else
                masks_list[q][i,j,r] = false
            end
        end
    end
    masks_list = ArrayType.(masks_list)
    #set nyquist frequency to 0
    iseven(size(k,1)) && (k[Int(end/2)+1] = 0)
    iseven(size(l,1)) && (l[Int(end/2)+1] = 0)
    iseven(size(m,1)) && (m[Int(end/2)+1] = 0)

    ∂ = [convert(T,2*pi).*reshape(k,(:,1,1))*1im,
    convert(T,2*pi).*reshape(l,(1,:,1))*1im,
    convert(T,2*pi).*reshape(m,(1,1,:))*1im]
    ∂ = ArrayType.(∂)
    
    return masks_list, ∂
end



"""
    compute the curl of a 3D flow field in Fourier space
"""
function curl(x, to_setup)
    (; ∂) = to_setup
    return stack(
        (
            ∂[2].*x[:,:,:,3] .- ∂[3].*x[:,:,:,2],
            ∂[3].*x[:,:,:,1] .- ∂[1].*x[:,:,:,3],
            ∂[1].*x[:,:,:,2] .- ∂[2].*x[:,:,:,1],
        ),
        dims = 4
    )
end


"""
    Compute the QoIs from the Fourier transformed fields
"""
function compute_QoI(u_hat, w_hat, to_setup, setup)
    (; dimension, xlims) = setup
    ArrayType = rf_arraytype(setup)
    D = dimension()
    L = [xlims[a][2] - xlims[a][1] for a in 1:D]
    if to_setup.mirror_y
        L[2] = L[2]*2
    end
    N = size(u_hat)
    q = Array{typeof(setup.Re)}(undef, to_setup.N_qois)  # if slow make this into a CuArray

    for i in 1:to_setup.N_qois
        if to_setup.qois[i][1] == "E"
            E = sum(abs2, u_hat, dims = 4)
            q[i]= sum(E.*to_setup.masks[i])*(prod(L)/(2*prod(N[1:D])^2))
        elseif to_setup.qois[i][1] == "Z"
            Z = sum(abs2, w_hat, dims = 4)
            q[i] = sum(Z.*to_setup.masks[i])*(prod(L)/(prod(N[1:D])^2))
        else
            error("QoI not recognized")
        end 
    end
    
    return q
end

"""
    Compute QoI densities from the Fourier transformed fields
"""
function compute_filtered_qoi_fields(u_hat, w_hat, to_setup, setup)
    (; dimension, xlims) = setup
    ArrayType = rf_arraytype(setup)
    D = dimension()
    L = [xlims[a][2] - xlims[a][1] for a in 1:D]
    if to_setup.mirror_y
        L[2] = L[2]*2
    end
    N = size(u_hat)
    qs = []

    for i in 1:to_setup.N_qois
        if to_setup.qois[i][1] == "E"
            push!(qs, u_hat.*to_setup.masks[i])
        elseif to_setup.qois[i][1] == "Z"
            
            push!(qs, w_hat.*to_setup.masks[i])
        else
            error("QoI not recognized")
        end 
    end
    
    return qs
end

"""
    get_u_hat(u::Tuple, setup, TO_Setup)
Compute the Fourier transform of the field. Returns an 4D array, velocity components stacked along last dimension.
"""
function get_u_hat(u, setup, TO_Setup)
    (; dimension) = setup
    d = dimension()
    # interpolate u to cell centers
    #u_c = interpolate_u_p(u, setup)
    if TO_Setup.mirror_y
        u1 = u[setup.Iu[1], 1]
        u1 = cat(u1, -1 .*reverse(u1, dims=2), dims= 2)
        u2 = u[setup.Iu[2], 2]
        #z = zeros(typeof(setup.Re),size(u2)[1], 1, size(u2)[3])
        z = fill!(similar(setup.x[1], size(u2)[1], 1, size(u2)[3]), 0)
        u2 = cat(u2, z, -1 .*reverse(u2, dims=2),z, dims= 2)
        u3 = u[setup.Iu[3], 3]
        u3 = cat(u3, -1 .*reverse(u3, dims=2), dims= 2)

        u = stack([u1, u2, u3], dims=4)
    else
        u = stack([u[select_physical_fourier_points(a, setup), a] for a=1:d], dims=4)
    end

    u_hat = fft(u, [1,2,3])
    return u_hat
end

function select_physical_fourier_points(a, setup)
    # ⚠️ `boundary_conditions` is keyed by field since the upstream merge: `.u[a]`, not `[a]`.
    bc = setup.boundary_conditions.u[a]
    if eltype(bc) == PeriodicBC
        return setup.Iu[a]
    elseif eltype(bc) == DirichletBC{Nothing}
        return setup.Ip
    else
        error("Boundary condition not recognized: $(eltype(bc)) in direction $a")
    end
end


"""
    get_w_hat_from_u_hat(u_hat, to_setup)
Compute the vorticity field from the velocity field in Fourier space.
"""
function get_w_hat_from_u_hat(u_hat, to_setup)
    # compute vorticity
    w_hat = curl(u_hat, to_setup)
    return w_hat
end

"""
Create processor that stores the QoI values every `nupdate` time step.
"""
qoisaver(; setup, to_setup, nupdate = 1, nan_limit = 1f5, nans_detected = nothing) =
    processor() do state
        T = typeof(setup.Re)
        qoi_hist = fill(zeros(T,0), 0)
        on(state) do state
            state.n % nupdate == 0 || return
            u_hat = get_u_hat(state.u, setup, to_setup)
            w_hat = get_w_hat_from_u_hat(u_hat, to_setup)
            q = compute_QoI(u_hat, w_hat, to_setup,setup)
            if any(q .> nan_limit)
                @warn "Unreasonable large QoI at n = $(state.n)"
                # The flag lives in the force cache, not the setup: see `rf_setup`.
                isnothing(nans_detected) || (nans_detected[] = true)
            end
            push!(qoi_hist, q)
        end
        state[] = state[]  # invokes all processors on initial state!
        qoi_hist
    end

"""
    to_sgs_term(u, setup, to_setup, stepper)
    
"""
function to_sgs_term(u, setup, to_setup, stepper)
    # get u_hat v_hat
    u_hat = get_u_hat(u, setup, to_setup);
    w_hat = get_w_hat_from_u_hat(u_hat, to_setup);
    # get dQ
    if to_setup.to_mode == :TRACK_REF
        q_star = compute_QoI(u_hat, w_hat, to_setup,setup)
        to_setup.outputs.q_star[:,stepper.n] = q_star
        q_ref = get_next_item_timeseries(to_setup.time_series_method)
        dQ = q_ref-q_star
    elseif to_setup.to_mode == :ONLINE
        # Which closures see `q*` is a property OF THE CLOSURE, declared next to its definition --
        # see `needs_qstar` in time_series_methods.jl. This used to be two literal type lists here.
        if needs_qstar(to_setup.time_series_method)
            q_star = rf_arraytype(setup)(compute_QoI(u_hat, w_hat, to_setup,setup))
            dQ = get_next_item_timeseries(to_setup.time_series_method, q_star)
        else
            dQ = get_next_item_timeseries(to_setup.time_series_method)
        end
    end
    dQ = Array(dQ)
    to_setup.outputs.dQ[:,stepper.n] = dQ

    if to_setup.to_mode == :TRACK_REF && !isnothing(to_setup.tracking_noise)
        if typeof(to_setup.tracking_noise)<:Sampleable
            dQ += convert.(eltype(dQ),rand(to_setup.tracking_rng, to_setup.tracking_noise))
        else
            dQ += randn(to_setup.tracking_rng, eltype(dQ), size(dQ)).*to_setup.time_series_method.stds.*convert(eltype(dQ),to_setup.tracking_noise)
        end
    end

    # get V_i
    vi = [to_setup.V_i[i](u_hat, w_hat) for i in 1:to_setup.N_qois]
    vi = stack(vi, dims=5)
    # get T_i
    ti = copy(vi)
    # compute innerproducts (returns ip on CPU)
    ip = innerpoducts(vi,ti,setup; mirror_y = to_setup.mirror_y)
    # compute c_ij
    cij = compute_cij(ip, to_setup)
    src_Q = reshape(sum(-conj(cij).*ip, dims = 1),:)
    tau = dQ./src_Q
    to_setup.outputs.tau[:,stepper.n] = real(tau)
    # move to GPU
    cij = rf_arraytype(setup)(cij)
    tau = rf_arraytype(setup)(tau)

    # construct SGS term
    #@tensor P_hat2[c,d,e,f,b] := cij[a,b]* ti[c,d,e,f,a]
    cij = reshape(cij, 1,1,1,1,size(cij,1), size(cij,2))
    ti = reshape(ti, size(ti,1), size(ti,2), size(ti,3), size(ti,4), size(ti,5), 1)
    P_hat = sum(cij .* ti; dims = 5)
    P_hat = reshape(P_hat, size(P_hat,1), size(P_hat,2), size(P_hat,3), size(P_hat,4), size(cij,6))

    #@tensor sgs_hat2[b,c,d,e] := -tau[a] * P_hat[b,c,d,e,a]
    tau = reshape(tau, 1,1,1,1,:)
    @. P_hat = -tau * P_hat
    sgs_hat = sum(P_hat; dims = 5)
    sgs_hat = reshape(sgs_hat, size(sgs_hat,1), size(sgs_hat,2), size(sgs_hat,3), size(sgs_hat,4))

    sgs = real(ifft(sgs_hat, [1,2,3]))

    if to_setup.mirror_y
        sgs = sgs[:,1:Int(end//2),:,:]
    end
    return sgs
end

function innerpoducts(x,y,setup; mirror_y = false)
    (; dimension, xlims) = setup
    D = dimension()
    L = [xlims[a][2] - xlims[a][1] for a in 1:D]
    if mirror_y
        L[2] = L[2]*2
    end
    N = size(x)[1:D]
    #@tensor ip2[e,f] := x[a,b,c,d,e]* conj(y)[a,b,c,d,f]
    x = reshape(x, size(x,1), size(x,2), size(x,3), size(x,4), size(x,5), 1)
    y = reshape(y, size(y,1), size(y,2), size(y,3), size(y,4), 1, size(y,5))
    ip =reshape(sum(x .* conj.(y); dims = (1,2,3,4)), size(x,5), size(y,6))
    Array(ip).*(prod(L)/(prod(N)^2))
end

function compute_cij(ip, to_setup)
    T = typeof(ip[1,1]) 
    N = to_setup.N_qois
    cij = ones(T, (N, N)).*-1
    for i in 1:N
        A = reshape(ip[to_setup.cij_masks.A[:,:,i]], (N-1, N-1))
        b = ip[:,i][to_setup.cij_masks.B[:,i]]
        cij[to_setup.cij_masks.B[:,i],i] = A\b  # each column of cij is the solution of a linear system
    end
    return cij
end

using IncompressibleNavierStokes: timestep!, create_stepper, get_state, default_psolver,
    get_cache, AbstractODEMethod, AbstractRungeKuttaMethod, RKMethods, processor, apply_bc_u!

"The element type of any `AbstractODEMethod`, taken from its supertype parameter."
_method_eltype(::AbstractODEMethod{T}) where {T} = T

"""
AbstractODEMethod for the Tau-orthogonal method, wrapping an inner Runge-Kutta scheme.
"""
struct TOMethod{T,R,TOS} <: AbstractODEMethod{T}
    rk_method::R
    to_setup::TOS
    # ⚠️ `_method_eltype`, not `eltype(rk_method.A)`. Only `ExplicitRungeKuttaMethod` has a Butcher
    # tableau; `LMWray3` is `LMWray3{T}` with no fields at all, so the old expression made TOMethod
    # unusable with any low-storage scheme. Both are `AbstractODEMethod{T}`, so the supertype
    # parameter is the general answer and gives the identical result for RK44.
    # LMWray3 by Rik's decision of 2026-09-11, matching `solve_unsteady`'s own default. The
    # archived TO runs used RK44; reproducing one means passing `rk_method` explicitly.
    TOMethod(; rk_method = LMWray3(), to_setup) =
        new{_method_eltype(rk_method),typeof(rk_method),typeof(to_setup)}(rk_method, to_setup)
end

export TOMethod

# The three methods TOMethod has to provide, ported to the IncompressibleNavierStokes >= 5 stepper
# interface. Three things changed and all three are here:
#
#   `create_stepper`    `u, temp` became a single `state` container (`(; u)` for us).
#   `ode_method_cache`  renamed to `get_cache` and re-signatured: it now takes `state` as well,
#                       because the cache is built by `map(similar, state)`.
#   `timestep!`         takes the right-hand side `force!` as a positional argument, `θ` became
#                       `params`, and `cache` split into `ode_cache` and `force_cache`.
#
# The TO step itself is unchanged: run the RK step, then add the SGS term. The SGS term is applied
# *after* the full step rather than inside it, which is why upstream dropping the in-stepper
# closure hook does not affect us.

IncompressibleNavierStokes.create_stepper(method::TOMethod; setup, psolver, state, t, n = 0) =
    create_stepper(method.rk_method; setup, psolver, state, t, n)

IncompressibleNavierStokes.get_cache(method::TOMethod, state, setup) =
    get_cache(method.rk_method, state, setup)

function IncompressibleNavierStokes.timestep!(
    method::TOMethod,
    force!,
    stepper,
    Δt;
    params = nothing,
    ode_cache,
    force_cache,
)
    (; rk_method, to_setup) = method
    (; setup) = stepper
    (; dimension) = setup
    D = dimension()

    # RK step
    stepper = timestep!(rk_method, force!, stepper, Δt; params, ode_cache, force_cache)

    # to method
    u = stepper.state.u
    sgs = to_sgs_term(u, setup, to_setup, stepper)
    # add SGS term to u
    for a in 1:D
        u[select_physical_fourier_points(a, setup),a] .+= sgs[:,:,:,a]
    end

    apply_bc_u!(u, stepper.t, setup)
    stepper
end


end # module RikFlow