# Fan plots of the training hindcast (Rik, 2026-10-13): one figure per closure, the 6 QoIs (rows) x the
# 5 training ICs (columns); thin lines the 10 members from the end of the replayed warm-up, black the
# reference, the warm-up shaded. Usage (from lib/RikFlow):
#   julia --project=analysis analysis/plot_train_fans.jl LinReg1 LinReg7 ...
# reads exp_square_HIT/output/D6train_<name>, writes analysis/figures/train/train_fans_<name>.pdf.

ENV["PAPER_FIG_DIR"] = joinpath(@__DIR__, "figures", "train")
include(joinpath(@__DIR__, "plot_paper_secs45.jl"))

const FAN_COL = Dict("LinReg1" => COL["LinReg1"], "LinReg6" => to_color("#56B4E9"), "LinReg7" => COL["LinReg7"])

function fig_train_fans(name)
    ens = load_members(joinpath(OUTD, "D6train_" * name))
    col = get(FAN_COL, name, to_color("#009E73"))
    ks = sort(ens.ks)
    nk = Dict(k => jldopen(f -> f["n_k"], last(first(ens.files[k]))) for k in ks)
    fig = Figure(size = (W, 520), figure_padding = (2, 8, 2, 2))
    q_ref = TRUTH.q
    for (c, k) in enumerate(ks)
        nwarm = jldopen(f -> f["nwarm"], last(first(ens.files[k])))
        for i in 1:NQ
            ax = Axis(fig[i, c]; xticklabelsvisible = i == NQ, yticklabelsvisible = c == 1,
                      xlabelsize = 9, ylabelsize = 9, xticklabelsize = 7, yticklabelsize = 7,
                      xticks = [0, 1, 2, 3])
            i == 1 && (ax.title = @sprintf("t = %.2f TU", nk[k] * DT); ax.titlesize = 9)
            c == 1 && (ax.ylabel = QLAB[i])
            i == NQ && (ax.xlabel = "lead (TU)")
            ncol = 0
            for (_, p) in ens.files[k]
                q = jldopen(f -> Float64.(f["q"]), p)
                ncol = max(ncol, size(q, 2))
                tt = ((nwarm):(size(q, 2) - 1)) .* DT .- nwarm * DT     # lead 0 = end of the warm-up
                lines!(ax, tt, q[i, (nwarm + 1):end]; color = (col, 0.55), linewidth = 0.45)
            end
            cr = (nk[k] + 1):(nk[k] + ncol)                                # truth columns of the member
            lines!(ax, ((0:(ncol - 1)) .- nwarm) .* DT, q_ref[i, cr]; color = C_REF, linewidth = 0.8)
            vspan!(ax, -nwarm * DT, 0; color = (:gray60, 0.15))
            xlims!(ax, -nwarm * DT, (ncol - 1 - nwarm) * DT)
        end
    end
    for i in 1:NQ
        linkyaxes!(contents(fig[i, :])...)
    end
    Label(fig[0, :], name * ", training hindcast (in-sample)"; font = :bold, fontsize = 10)
    rowgap!(fig.layout, 2); colgap!(fig.layout, 4)
    savefig("train_fans_$(name).pdf", fig)
end

"""
Fans of the held-out hindcast (`D6_<name>`), the 6 QoIs (rows) x the closures (columns), at the four ICs
of the paper's Fig. 2 (t = 10.25, 15.00, 20.00, 24.75 TU) on one time axis.
"""
function fig_full_fans(names; tics = (10.25, 15.0, 20.0, 24.75), file = "full_fans.pdf")
    fig = Figure(size = (W, 560), figure_padding = (2, 8, 2, 2))
    q_ref = TRUTH.q
    t0, t1 = first(tics) - 0.4, last(tics) + 3.4
    cr = round(Int, t0 / DT):round(Int, t1 / DT)
    for (c, name) in enumerate(names)
        ens = load_members(joinpath(OUTD, "D6_" * name))
        col = get(FAN_COL, name, to_color("#009E73"))
        ks = [k for k in ens.ks if any(t -> isapprox(0.25 * (k - 1), t; atol = 1e-6), tics)]
        for i in 1:NQ
            ax = Axis(fig[i, c]; xticklabelsvisible = i == NQ, yticklabelsvisible = c == 1,
                      xticklabelsize = 7, yticklabelsize = 7, xlabelsize = 9, ylabelsize = 9)
            i == 1 && (ax.title = name; ax.titlesize = 9)
            c == 1 && (ax.ylabel = QLAB[i])
            i == NQ && (ax.xlabel = "t (TU)")
            for k in ks
                nk = jldopen(f -> f["n_k"], last(first(ens.files[k])))
                nwarm = jldopen(f -> f["nwarm"], last(first(ens.files[k])))
                for (_, p) in ens.files[k]
                    q = jldopen(f -> Float64.(f["q"]), p)
                    tt = ((nk + nwarm):(nk + size(q, 2) - 1)) .* DT
                    lines!(ax, tt, q[i, (nwarm + 1):end]; color = (col, 0.5), linewidth = 0.45)
                end
                vlines!(ax, [(nk + nwarm) * DT]; color = (:gray40, 0.8), linestyle = :dash, linewidth = 0.5)
            end
            lines!(ax, (cr .- 1) .* DT, q_ref[i, cr]; color = C_REF, linewidth = 0.8)
            xlims!(ax, t0, t1)
        end
    end
    for i in 1:NQ
        linkyaxes!(contents(fig[i, :])...)
    end
    rowgap!(fig.layout, 2); colgap!(fig.layout, 4)
    savefig(file, fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    if !isempty(ARGS) && ARGS[1] == "--full"
        fig_full_fans(ARGS[2:end]; file = "full_fans_" * join(ARGS[2:end], "_") * ".pdf")
    else
        foreach(fig_train_fans, ARGS)
    end
end
