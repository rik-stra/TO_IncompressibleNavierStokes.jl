# The data of step 3 (paper Sec. 6, closures.tex step 3): the tracked record, the target, the input map
# P of the network means, every check of the input against the raw record, the least-squares floor
# and the whole-record design for the held-out score. Shared by `mlp_mean_fit.jl` and
# `cnn_mean_fit.jl` (2026-10-22, split out of mlp_mean_fit.jl unchanged); `include` it after
# `using RikFlow, LinearAlgebra, Statistics, Printf, JLD2` and `const RF = RikFlow`.

TO_folder = normpath(joinpath(@__DIR__, "..", "output", "TO_LSTM"))
include(joinpath(@__DIR__, "m4_data.jl"))

envs(k, d) = (v = strip(get(ENV, "RIKFLOW_MLP_" * k, "")); isempty(v) ? d : v)
envv(k, d, T) = parse.(T, strip.(split(envs(k, d), ",")))
const DT = 2.5e-3
const T = Float32

base = load(joinpath(TO_folder, "inputs_lstm.jld2"), "inputs")[2]
cfg = merge(base, (; arch = :dense, n_latent = 0, n_encoder = 0, emission = :constant,
                   h = parse(Int, envs("H", "5")), hist_var = :q_star_q, include_predictor = true,
                   train_range = (base.train_range[1], round(Int, 10 / DT))))
cfg.train_range == (400, 4000) || error("training range $(cfg.train_range), expected (400, 4000) = 1-10 TU")

rec = load_m4_qois(; track_file = joinpath(TO_folder, "..",
                                           "data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2"))
nq = size(rec.q, 1)
hist = RF.HistorySpec(; h = cfg.h, n_qoi = nq, cfg.hist_var, cfg.include_predictor)
nreg = RF.nfeatures(hist)                                   # LinReg1's regressor, bias last
nreg == (2cfg.h + 1) * nq + 1 || error("regressor width $nreg, expected $((2cfg.h + 1) * nq + 1)")

dat0 = m4_training_data(rec, cfg, hist; target = :dQ)
# 🔴 THE TARGET is dQ^n = q^n - q^{n*} computed from the record's levels (paper notation: column n + 1
# of `q` minus column n of `q_star`), NOT the recorded correction `rec.dQ` the M4 loader returns (Rik,
# 2026-10-22, "dQ_n = q_n - q*_n"; the history inputs use the same difference). The two differ by up to
# 0.12 sd(dQ) per step, 0.04 rms in the small scales -- two thirds of LinReg1's whole residual there
# (found by the analysis dry run, 2026-10-22). With this target, least squares on the MLP's input is
# LinReg1's fit exactly (LinReg1 predicts the level q^n from the same regressor, and q^{n*} is in it).
let cols = (cfg.train_range[1] - 1) .+ dat0.steps
    D = rec.q[:, cols .+ 1] .- rec.q_star[:, cols]
    Dn, out_scaling = RikFlow._normalise(D; normalization = cfg.normalization)
    global dat = merge(dat0, (; Yc = Dn, scaling = (; dat0.scaling.in_scaling, out_scaling, target = :dQ)))
end
ncol = size(dat.Xc, 2)
ntr = floor(Int, (1 - cfg.val_frac) * ncol)                 # the library's split (val_frac = 0.2)
tu(k) = (cfg.train_range[1] - 1 + dat.steps[k]) * DT

# --- the input: the predictor history, the dQ history and the predictor increments (Rik, 2026-10-22) ---
# 🔑 FEAT = dqinc (default): the current predictor q^{n*}; in every lag k = 1..h the predictor q^{n-k*}
# and the correction dQ^{n-k} = q^{n-k} - q^{n-k*} (Rik: "giving the history as q*n and dQ = qn - q*n");
# and, as EXTRA inputs, the predictor increments Dq*^{n-k} = q^{n-k+1*} - q^{n-k*}, k = 1..h (with
# q^{n-0*} = q^{n*}; Rik: "add all five increments ... as extra inputs"). The level lag q^{n-k} of
# LinReg1's regressor is replaced by dQ^{n-k}; dQ^{n-k} and Dq*^{n-k} are standardised by their own mean
# and sd on 1-10 TU. The input is
#     [q*^n | dQ^{n-1}, q*^{n-1} | ... | dQ^{n-h}, q*^{n-h} | 1 | Dq*^{n-1} | ... | Dq*^{n-h}],
# a fixed affine map P of LinReg1's regressor x (both streams share one in_scaling, so a scaled
# difference is a raw difference / sigma_q), (nreg + h nq) x nreg: the first nreg rows are invertible,
# the extra rows are combinations of them, so no information is added or lost. What changes is the
# conditioning: the network is handed the small differences instead of having to form them from nearly
# equal inputs (on 1-10 TU the condition number on the input's range drops from 1.9e6 to 2.3e4; the dQ
# history alone leaves it at 2.1e6). Stored as `scaling.input_map`, which the deployed closure applies
# (`_apply_input_map`, V63; the widened input, `LSTMSpec`'s `n_in`, V91). Least squares on P x is least
# squares on x, and every entry of P x is checked against the record below.
# FEAT = dq: the dQ history without the extra inputs; FEAT = raw: LinReg1's regressor as it is.
const FEAT = Symbol(envs("FEAT", "dqinc"))
FEAT in (:raw, :dq, :dqinc) || error("RIKFLOW_MLP_FEAT must be raw, dq or dqinc; got $FEAT")
"Rows of LinReg1's regressor holding q^{n-k} and, right after it, q^{n-k*} (layout of `build_history`)."
lagrows(k) = (q = (nq + 2(k - 1) * nq + 1):(nq + (2k - 1) * nq); (; q, s = q .+ nq))
"Rows of q^{n-k*} in the regressor, k = 0 being the current predictor q^{n*}."
predrows(k) = k == 0 ? (1:nq) : lagrows(k).s
P0 = Matrix{Float64}(I, nreg, nreg)
dq_m, dq_s = zeros(nq, cfg.h), zeros(nq, cfg.h)              # the standardisation of dQ^{n-k}
in_m, in_s = zeros(nq, cfg.h), zeros(nq, cfg.h)              # ... and of Dq*^{n-k}
Pinc = zeros(cfg.h * nq, nreg)
if FEAT !== :raw
    cfg.include_predictor && cfg.hist_var === :q_star_q || error("FEAT = $FEAT needs the :q_star_q history with the predictor")
    for k in 1:cfg.h
        r = lagrows(k)
        d = Float64.(dat.Xc[r.q, :] .- dat.Xc[r.s, :])          # dQ^{n-k} / sigma_q on 1-10 TU
        dq_m[:, k], dq_s[:, k] = vec(mean(d; dims = 2)), vec(std(d; dims = 2))
        P0[r.q, r.q] .= Diagonal(1 ./ dq_s[:, k])               # the q row now holds dQ: q* stays
        P0[r.q, r.s] .= -Diagonal(1 ./ dq_s[:, k])
        P0[r.q, nreg] .= -dq_m[:, k] ./ dq_s[:, k]
        # the increment q^{n-k+1*} - q^{n-k*}
        a_, b_, blk = predrows(k - 1), predrows(k), ((k - 1) * nq + 1):(k * nq)
        di = Float64.(dat.Xc[a_, :] .- dat.Xc[b_, :])
        in_m[:, k], in_s[:, k] = vec(mean(di; dims = 2)), vec(std(di; dims = 2))
        Pinc[blk, a_] .= Diagonal(1 ./ in_s[:, k])
        Pinc[blk, b_] .= -Diagonal(1 ./ in_s[:, k])
        Pinc[blk, nreg] .= -in_m[:, k] ./ in_s[:, k]
    end
end
P = FEAT === :dqinc ? vcat(P0, Pinc) : P0
nin = size(P, 1)                                            # the width the network reads
scaling = FEAT === :raw ? dat.scaling : merge(dat.scaling, (; input_map = P))
Xc = T.(P * Float64.(dat.Xc))
"Condition number on the range: largest over smallest singular value above 1e-10 of the largest."
cond_range(A) = (s = svdvals(A); s[1] / s[findlast(>(s[1] * 1e-10), s)])
# The map is right iff least squares on P x (Float64, minimum norm where the extra rows make it
# rank-deficient) predicts what least squares on x does, to round-off; asserted. The network then reads
# P x in Float32, as every M4 fit reads its input -- that rounding is checked entry by entry below.
let X64 = Float64.(dat.Xc), Z64 = P * X64, Y = Float64.(dat.Yc)
    yx = (X64' \ Y')' * X64
    err = maximum(abs, yx .- (pinv(Z64') * Y')' * Z64) / maximum(abs, yx)
    err < 1e-8 || error("least squares on the mapped input differs from LinReg1's regressor by $err")
    @info "input" FEAT nreg nin ls_rel_diff = err rank = rank(Z64) cond_raw = cond(X64) cond_input_range = cond_range(Z64)
end

# 🔴 Every entry of the input and the target, rebuilt from the RAW record by indexing alone -- not
# through `build_history` -- and compared (Rik, 2026-10-22: "do carefully check"). Notation of the
# paper: q^{m*} is column m of `q_star`, q^m (the level after the correction at step m) is column
# m + 1 of `q`, and row r of the design is the step n = cols[r] whose predictor is q^{n*}.
function check_input(Z, cols)
    mu, sg = vec(dat.scaling.in_scaling.mu), vec(dat.scaling.in_scaling.sigma)
    sc(v) = (v .- mu) ./ sg
    worst = 0.0
    for (r, n) in enumerate(cols)
        z = Z[:, r]
        e = zeros(nin)
        e[1:nq] = sc(rec.q_star[:, n])                          # q^{n*}, the current predictor
        for k in 1:cfg.h
            lr = lagrows(k)
            qk, qsk = rec.q[:, n - k + 1], rec.q_star[:, n - k] # q^{n-k} and q^{n-k*}
            e[lr.q] = FEAT === :raw ? sc(qk) : ((qk .- qsk) ./ sg .- dq_m[:, k]) ./ dq_s[:, k]
            e[lr.s] = sc(qsk)
            if FEAT === :dqinc                                  # q^{n-k+1*} - q^{n-k*}
                e[nreg .+ (((k - 1) * nq + 1):(k * nq))] =
                    ((rec.q_star[:, n - k + 1] .- qsk) ./ sg .- in_m[:, k]) ./ in_s[:, k]
            end
        end
        e[nreg] = 1
        worst = max(worst, maximum(abs, z .- e))
    end
    return worst
end
cols_tr = (cfg.train_range[1] - 1) .+ dat.steps               # record column n of q^{n*}, per row
let w64 = check_input(P * Float64.(dat.Xc), cols_tr), w32 = check_input(Float64.(Xc), cols_tr)
    w64 < 1e-9 || error("the input differs from the record by $w64 (Float64)")
    # the target: q^n - q^{n*} at step n (columns n + 1 of q, n of q_star), standardised
    so_, mo_ = vec(dat.scaling.out_scaling.sigma), vec(dat.scaling.out_scaling.mu)
    wy = maximum(abs, Float64.(dat.Yc) .- (rec.q[:, cols_tr .+ 1] .- rec.q_star[:, cols_tr] .- mo_) ./ so_)
    wy < 1e-9 || error("the target differs from q^n - q^{n*} by $wy")
    @info "input checked entry by entry against the record (1-10 TU rows)" max_abs_diff_float64 = w64 max_abs_diff_float32 = w32 target_max_abs_diff = wy
end
# How the dQ history the network reads in training (q^{n-k} - q^{n-k*} from the tracked record) relates
# to the RECORDED correction dQ^{n-k}, which is what the deployed closure's history holds (it stores
# q^{n-k} = q^{n-k*} + dQ^{n-k}). Per QoI, the rms difference over the record in units of sd(dQ); one
# step off (dQ^{n-k-1}) for scale -- a misaligned history would show up as the larger number.
let n = 2:(size(rec.q_star, 2) - 1), sdq = vec(std(rec.dQ; dims = 2))
    d0 = rec.q[:, n .+ 1] .- rec.q_star[:, n]
    @info "q^{m} - q^{m*} against the recorded dQ^m, rms / sd(dQ) per QoI" same_step = round.(vec(sqrt.(mean(abs2, d0 .- rec.dQ[:, n]; dims = 2))) ./ sdq; sigdigits = 2) one_step_off = round.(vec(sqrt.(mean(abs2, d0 .- rec.dQ[:, n .- 1]; dims = 2))) ./ sdq; sigdigits = 2)
end

"Gaussian NLL per step, summed over the QoIs, of residuals `r` (nq x N) under covariance `S`."
gnll(r, S) = 0.5 * (nq * log(2pi) + logdet(S) + mean(sum(r .* (S \ r); dims = 1)))
headcov(w) = Diagonal(exp.(Float64.(w.bd))) * (Float64.(w.LR) * Float64.(w.LR)') * Diagonal(exp.(Float64.(w.bd)))

# --- the LS floor: least squares on LinReg1's regressor, the stage-1 fit rows ----------------------
# On x itself (the same predictions as on P x, asserted above), so the extra rows' rank deficiency
# never enters it. The network's validation rows are the mapped ones (Xva).
Rtr_, Rva_ = Float64.(dat.Xc[:, 1:ntr]), Float64.(dat.Xc[:, (ntr + 1):end])
Ytr, Yva = Float64.(dat.Yc[:, 1:ntr]), Float64.(dat.Yc[:, (ntr + 1):end])
Xva = Float64.(Xc[:, (ntr + 1):end])
Cls = Rtr_' \ Ytr'
Sls = (Rls = Ytr .- Cls' * Rtr_; cov(permutedims(Rls)))
ls_val = gnll(Yva .- Cls' * Rva_, Sls)
ls_sse = 0.5 * sum(abs2, Yva .- Cls' * Rva_) / size(Yva, 2)
@info "LS floor on 8.2-10 TU (fit on 1-8.2 TU)" nll = ls_val half_sse = ls_sse

# --- the whole record in the same input: a held-out score of every fit on 10-100 TU ------------------
a, b = cfg.train_range[1], size(rec.q_star, 2)
qs = RF.scale_input(rec.q[:, a:(b + 1)], dat.scaling.in_scaling)
qss = RF.scale_input(rec.q_star[:, a:b], dat.scaling.in_scaling)
Xr, _, stepsf = RF.build_history(hist, qss, qs)
Xr64 = permutedims(Float64.(Xr))                            # LinReg1's regressor, whole record
Xf = P * Xr64                                               # the same input map
cols = (a - 1) .+ stepsf                                    # record column n of q*^n
(wf = check_input(Xf, cols)) < 1e-9 || error("the whole-record input differs from the record by $wf")
mo, so = vec(dat.scaling.out_scaling.mu), vec(dat.scaling.out_scaling.sigma)
Yf = (rec.q[:, cols .+ 1] .- rec.q_star[:, cols] .- mo) ./ so   # the target, standardised
HO = findall(n -> 10 < n * DT <= 100, cols)
rmsq(r) = vec(sqrt.(mean(abs2, r; dims = 2)))
# the comparator: least squares fitted on the same 1-8.2 TU rows as the stage-1 networks
ls_ho = rmsq(Yf[:, HO] .- Cls' * Xr64[:, HO])

# LinReg1's least squares (h = 5) as a FIXED reference for every h (2026-10-22: histories 10 and 15): fitted
# on the same 1-8.2 TU record columns as the networks' fit rows, in the same standardised target, scored on
# the same validation and 10-100 TU columns. At h = 5 it is the floor above (checked).
hist5 = RF.HistorySpec(; h = 5, n_qoi = nq, hist_var = :q_star_q, include_predictor = true)
X5r, _, steps5 = RF.build_history(hist5, qss, qs)
X5 = permutedims(Float64.(X5r))
cols5 = (a - 1) .+ steps5
at5(c) = c .- cols5[1] .+ 1                               # record column -> column of X5 (contiguous)
Y5 = (rec.q[:, cols5 .+ 1] .- rec.q_star[:, cols5] .- mo) ./ so
fit5, val5, ho5 = at5(cols_tr[1:ntr]), at5(cols_tr[(ntr + 1):end]), at5(cols[HO])
C5 = X5[:, fit5]' \ Y5[:, fit5]'
ls1_sse = 0.5 * sum(abs2, Y5[:, val5] .- C5' * X5[:, val5]) / length(val5)
ls1_ho = rmsq(Y5[:, ho5] .- C5' * X5[:, ho5])
cfg.h == 5 && (maximum(abs, ls1_ho .- ls_ho) < 1e-10 || error("the LinReg1 reference differs from the h = 5 floor"))
@info "LinReg1 reference (h = 5, fit on the same 1-8.2 TU columns)" h = cfg.h val_half_sse = ls1_sse

"""
The constant Gaussian head of the residuals `r` (nq x N, standardised target units), in the library's
parametrisation: `bd = log sd`, `Araw` with `A'A = R^{-1}` (`A` lower triangular, log diagonal), so that
the deployed head is `D R D` (`LSTMWeights`). Used to SEED a likelihood head at least squares' residual
covariance (Rik, 2026-10-22), as `m4_window_fit.jl`'s SEEDHEAD does.
"""
function head_from_residuals(r)
    sd = vec(std(r; dims = 2))
    Pm = inv(Symmetric(cor(permutedims(r))))
    J = reverse(Matrix{Float64}(I, nq, nq); dims = 1)
    A = J * cholesky(Symmetric(J * Pm * J)).U * J          # lower triangular, A'A = R^{-1}
    return (; bd = T.(log.(sd)), Araw = T.(tril(A, -1) + Diagonal(log.(diag(A)))))
end
const LS_HEAD = head_from_residuals(Rls)                   # least squares' residual, 1-8.2 TU fit rows
"Sum over the QoIs of each QoI's RMSE (the per-QoI RMSE loss), for prediction and target nq x N."
rmse_sum(Yhat, Y) = sum(sqrt.(mean(abs2, Yhat .- Y; dims = 2)))
