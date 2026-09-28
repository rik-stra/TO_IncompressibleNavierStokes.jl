# One IC CHUNK of a mini-D6, in one process, so the solver compiles once per chunk rather than once
# per IC (compile ~60-90 s per process on the H100 against ~60-100 s of stepping per IC at M = 5,
# N_LEAD = 400 -- handoff_p2c_d6.md §2).
#
#     D6_BLOCK=selection D6_CLOSURE=... D6_MODEL=... D6_OUT=... D6_MEMBERS=5 D6_NLEAD=400 \
#       julia --project tools/p4grid_d6_chunk.jl <chunk> <nchunk>
#
# The IC filter (D6_BLOCK / D6_T_MIN / D6_T_MAX / D6_STRIDE) selects ordinals exactly as
# `run_d6.jl --all` does; this runs the <chunk>-th of <nchunk> CONTIGUOUS slices of them, through
# run_d6.jl's own `run_ic` (included, not copied -- same seeds, same warm-up check, same skip of
# existing members, same output identity). Because the filter is the named block, every chunk writes
# `block = "selection"`, so chunks of one closure share one D6_OUT and pair with any other
# selection-block run (a D6_T_MIN/D6_T_MAX window would record block "" and refuse to pair).
#
# `--list` prints the chunk's ordinals and stops.

include(joinpath(@__DIR__, "run_d6.jl"))

function chunk_main(args = ARGS)
    length(args) >= 2 || error("usage: p4grid_d6_chunk.jl <chunk> <nchunk> [--list]")
    c, n = parse(Int, args[1]), parse(Int, args[2])
    1 <= c <= n || error("chunk $c is not in 1:$n")
    filt = ic_filter_from_env()
    filt.active || error("set D6_BLOCK (e.g. selection): a chunk of the unfiltered pool is not a mini-D6")
    ords, man = selected_ordinals(filt)
    isempty(ords) && error("the IC filter selects no ordinal")
    edges = round.(Int, range(0, length(ords); length = n + 1))
    mine = ords[(edges[c] + 1):edges[c + 1]]
    @printf("IC filter %s: %d ordinals; chunk %d/%d = ordinals %s (k %s)\n", filt.label, length(ords), c, n,
            join(mine, ","), join(man["k"][mine], ","))
    flush(stdout)
    ("--list" in args || isempty(mine)) && return mine
    for o in mine
        run_ic(o; filt)
    end
    return mine
end

chunk_main()
