# Offline: the AR(2) by overdetermined Yule–Walker (`ar2_eyw`) against the ACF fit (`ar2_ls`) on the
# residuals of the round-1 AR(2) closures (2026-10-13, Rik asked; results_LSTMS.md §25).
#
#     julia --startup-file=no --project=training analysis/ar2_eyw_compare.jl
#
# Both fit lags 1-20 of the residual's sample ACF on 1-10 TU; they differ in what they match:
#
#   LS    ar2_ls:  Σ_k (ρ^AR(k; φ) − ρ̂(k))²                 the AR's own ACF against ρ̂  (paper Eq. ACF fit)
#   EYW   ar2_eyw: Σ_k (ρ̂(k) − φ1 ρ̂(k−1) − φ2 ρ̂(k−2))²     the Yule–Walker recursion, linear in φ
#
# each free (stationary) and restricted to φ1 ≥ 0, φ2 ≤ 0 (`n`, the round-1 rule, §16). Residual: each
# source's own, as deployed (`scaled_resid` of lrs_ar_variant.jl, scaled units). Per source, QoI and fit:
# φ1, φ2, the poles; the RMS ACF misfit at lags 1-20 on 1-10 TU (LS's objective) and on 10-74 TU
# (outside the fit window, same ρ^AR); the RMS Yule–Walker misfit (EYW's objective); ρ1 of the AR vs
# the data; the one-step calibration xi_d/σξ = sd of the data's innovations under φ over σ_ξ of
# `ar_innovation_var` (1 = calibrated, > 1 = the AR is one-step under-dispersed); and the long-run
# variance ratio AR / data, (1 + 2 Σ_{k ≤ 200} ρ_k) each, the data's on 1-10 TU (noisy: ~18 τ_int).
# Check: `ar2_ls(...; nonneg = true)` must reproduce the deployed `<src>_ar2n` φ exactly.
# 🔒 Nothing past 74 TU is read (guard()).

include(normpath(joinpath(@__DIR__, "..", "exp_square_HIT", "tools", "lrs_ar_variant.jl")))

const SOURCES = ["LinReg7", "Splice1_E0x11", "Splice1_E0x7", "Splice1_E0x8", "LinReg14", "LinReg16"]
const FITS = (("LS", r -> ar2_ls(r; L = 20)), ("LSn", r -> ar2_ls(r; L = 20, nonneg = true)),
              ("EYW", r -> ar2_eyw(r; L = 20)), ("EYWn", r -> ar2_eyw(r; L = 20, nonneg = true)))
const KLRV = 200

rms(v) = sqrt(mean(abs2, v))
yw_misfit(p1, p2, r) = rms([r[k + 1] - p1 * r[abs(k - 1) + 1] - p2 * r[abs(k - 2) + 1] for k in 1:20])
lrv(rho) = 1 + 2 * sum(rho[2:end])

"Signed poles of z² − φ1 z − φ2 as a string: real pair, or modulus ∠ angle for a complex pair."
function poles_str(p1, p2)
    d = p1^2 + 4p2
    d >= 0 && return @sprintf("%6.3f %6.3f", (p1 + sqrt(d)) / 2, (p1 - sqrt(d)) / 2)
    return @sprintf("%5.3f∠%4.0f°", sqrt(-p2), rad2deg(atan(sqrt(-d), p1)))
end

function compare(src; io = stdout)
    fitr = guard(steps_of(1, 10))
    outr = guard(steps_of(10, 74))
    Z = scaled_resid(src, fitr).Z
    Zo = scaled_resid(src, outr).Z
    dep = load(joinpath(LRS, "$(src)_ar2n", "LinReg.jld2"), "ar_phi")
    println(io, "\n==== $src: residual on 1-10 TU (fit) and 10-74 TU")
    @printf(io, "  %-9s %-5s %7s %7s  %-13s %8s %8s %8s  %5s/%5s %8s %8s\n", "QoI", "fit", "phi1", "phi2", "poles",
            "ACF 1-10", "ACF 10-74", "YW", "rho1", "data", "xi_d/sx", "LRV a/d")
    rows = []
    for i in 1:NQ
        r = acf(Z[:, i], 0:KLRV)
        ro = acf(Zo[:, i], 0:20)
        s2 = var(Z[:, i])
        for (name, fit) in FITS
            p1, p2 = fit(r)
            if name == "LSn"
                @assert (p1, p2) == (dep[1, i], dep[2, i]) "$src QoI $i: LSn refit $((p1, p2)) ≠ deployed $(dep[:, i])"
            end
            ra = ar2_acf(p1, p2, KLRV)
            xi = Z[3:end, i] .- p1 .* Z[2:(end - 1), i] .- p2 .* Z[1:(end - 2), i]
            row = (; src, qoi = LABELS[i], fit = name, p1, p2,
                   acf_in = rms(ra[2:21] .- r[2:21]), acf_out = rms(ra[2:21] .- ro[2:21]),
                   yw = yw_misfit(p1, p2, r), rho1 = ra[2], rho1_data = r[2],
                   xi_ratio = std(xi) / sqrt(ar_innovation_var(s2, p1, p2)), lrv_ratio = lrv(ra) / lrv(r))
            push!(rows, row)
            @printf(io, "  %-9s %-5s %7.3f %7.3f  %-13s %8.4f %8.4f %8.4f  %5.3f/%5.3f %8.3f %8.2f\n",
                    name == "LS" ? LABELS[i] : "", name, p1, p2,
                    poles_str(p1, p2), row.acf_in, row.acf_out, row.yw, row.rho1, row.rho1_data, row.xi_ratio,
                    row.lrv_ratio)
        end
    end
    println(io, "  (LSn = the deployed round-1 φ, checked equal to $(src)_ar2n)")
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    rows = reduce(vcat, [compare(s) for s in SOURCES])
    # The splices share LinReg1's (near-white) rows and each other's E[0,6]: summarise the distinct
    # coloured cells only (ρ̂1 ≥ 0.2, one per distinct residual ACF).
    cells = unique(x -> (x.qoi, round(x.rho1_data; digits = 6)), filter(x -> x.rho1_data >= 0.2, rows))
    keep(x) = any(c -> c.src == x.src && c.qoi == x.qoi, cells)
    println("\n==== summary over the $(length(cells)) distinct coloured cells (ρ̂1 ≥ 0.2): EYW against LS (same restriction)")
    println("  cells: ", join(("$(c.src) $(c.qoi)" for c in cells), ", "))
    for (a, b) in (("EYW", "LS"), ("EYWn", "LSn"))
        A = filter(x -> x.fit == a && keep(x), rows)
        B = filter(x -> x.fit == b && keep(x), rows)
        dphi = maximum(max(abs(x.p1 - y.p1), abs(x.p2 - y.p2)) for (x, y) in zip(A, B))
        @printf("  %-4s vs %-3s: max |Δφ| %.3f | median ACF misfit 1-10 %.4f vs %.4f, 10-74 %.4f vs %.4f | median |xi_d/sx − 1| %.3f vs %.3f | median |log LRV a/d| %.2f vs %.2f\n",
                a, b, dphi,
                median(x.acf_in for x in A), median(x.acf_in for x in B),
                median(x.acf_out for x in A), median(x.acf_out for x in B),
                median(abs(x.xi_ratio - 1) for x in A), median(abs(x.xi_ratio - 1) for x in B),
                median(abs(log(x.lrv_ratio)) for x in A), median(abs(log(x.lrv_ratio)) for x in B))
        @printf("              |ρ1 AR − ρ̂1| median %.3f vs %.3f; EYW better in xi_d/sx %d of %d, ACF 10-74 %d of %d, |log LRV| %d of %d\n",
                median(abs(x.rho1 - x.rho1_data) for x in A), median(abs(x.rho1 - x.rho1_data) for x in B),
                count(abs(x.xi_ratio - 1) < abs(y.xi_ratio - 1) for (x, y) in zip(A, B)), length(A),
                count(x.acf_out < y.acf_out for (x, y) in zip(A, B)), length(A),
                count(abs(log(x.lrv_ratio)) < abs(log(y.lrv_ratio)) for (x, y) in zip(A, B)), length(A))
    end
end
