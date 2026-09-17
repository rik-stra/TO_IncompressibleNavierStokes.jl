# attic

Code kept for reference and **not loaded by the package**. Nothing here is `include`d, tested, or
maintained. Read it for what it once did; do not call it.

## `ANN.jl`

Moved here 2026-09-17, on `plan.md` §13 P5's own instruction (*"Move it to `attic/`* — it is dead
and its formula is actively misleading*").

Why it was dead where it sat:

- `src/RikFlow.jl` never `include`d it, and `using Lux, LuxCUDA` at `RikFlow.jl:20` is commented
  out, so none of its names ever existed at runtime.
- `dev` (used in `setup_ANN` and `load_ANN`) was never defined anywhere in the package.
- There were zero call sites. The `ANN` struct in `src/time_series_methods.jl` calls `load_ANN`,
  which this file defines — so that constructor could never have run either.
- 🔴 The formula `in_size = n_qois * (1 + hist_len)` **cannot express `hist_var = :q_star_q`**,
  which carries two streams per lag and needs `n_qois * (2*hist_len + 1)`. Anything built on it
  would have been silently mis-shaped against the regressor every other model in the package uses.

Its replacement is **M4**, the stochastic LSTM: `src/ts_lstm.jl` (stdlib-only inference) plus
`ext/RikFlowLuxExt.jl` (Lux training). See `meta_files/handoff_m4_stochastic_lstm.md`.
