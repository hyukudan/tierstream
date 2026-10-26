# Reproducing the device-gated results

This page lists what is needed to rebuild tierstream, rerun its gates and reproduce the DeepSeek V4.1 Flash numbers
quoted in the README's device-gated section. The engine side (the DeepSeek CUDA replica that links tierstream) is in
[strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab); its `docs/reproduce.md` has the full
command lines for the model runs. This page covers the library part.

## What is on by default

| Feature | Default | Switch back to the old path |
|---|---|---|
| Pipelined startup loader (overlapped reads, staging and device copies, bounded buffers) | on | `DG_LOAD_SERIAL=1` (serial loader) |
| Deferred, bounded write-backs (`dg_wb_defer`) | on (`ss_options_defaults` sets 1) | `dg_wb_defer = 0` at open, engine option `--no-wb-defer`, or `ss_dg_set_sched(c, 0, flush_ready)` between tokens |
| Demand-stream flush after each ready (`dg_flush_ready`) | off | `ss_dg_set_sched(c, wb_defer, 1)` |
| Cache reset to the fresh-load placement (`ss_dg_reset_open`) | API call | not called |
| Residency snapshot in recency order (`ss_dg_residency`) | API call | not called |
| Priced window prefetch (`winpf`) | off | exact, but no repeatable gain was measured |
| Scored eviction (`SS_POLICY_SCORED`) | off (LRU) | opt-in policy; changes only which victims the boundary picks |

`ss_options_defaults` (and the sized variant for older option structs) sets `dg_wb_defer = 1`; a caller that fills
`ss_options` without it keeps `0`, the immediate boundary write-backs. Both give the same bytes; only the copy
scheduling differs.

## Hardware and software used

- GPU tests and model runs: one NVIDIA RTX PRO 6000 Blackwell (96 GB), Windows 11, WDDM driver mode, CUDA 13.0,
  MSVC 2022 + Ninja. The 4090 in the same machine was not used.
- CPU tests: the same Windows build, and Linux / WSL with GCC.
- Host: enough RAM for the pinned expert cache (the model runs used about 74 GiB of pinned host memory) and an NVMe
  drive for the model file; load times are bound by that drive.

## Build and unit gates

```
# Linux / WSL, CPU
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
# Windows, CPU and CUDA
tools\build-win.bat
tools\build-win.bat cuda
ctest --test-dir build
```

Every CPU test must pass (`ctest` reports 100 %). The device-gated GPU gate is `ts-test-dg-cuda` in the CUDA build;
run it alone on an idle GPU:

```
build-cuda\ts-test-dg-cuda.exe                       # every case
build-cuda\ts-test-dg-cuda.exe --only dg_cuda_wb_defer  # cases whose name contains the text
```

The cases that cover this release's features: `dg_cuda_wb_defer*` (write-back deferral, partial parts, tight spares,
counters, stopped service), `dg_cuda_load_serial_pipeline` (serial and pipelined loads give the same placement and the
same bytes), `dg_cuda_reset_open` (the slot table after a reset equals the one at open, with and without deferral), `dg_cuda_flush_ready`
and the `dg_cuda_winpf*` cases. Each case prints its counters and ends with `PASS` or the failing check.

`dg_cuda_winpf_poison_with_prior_in_flight` cannot stage its premise on a GPU with a single copy engine (the RTX PRO
under WDDM): the gated prefetch copy blocks the step's own copies, so the failed token only runs to the gate's
timeout. Expect it to fail there; the close/drain ordering it targets is covered by the CPU tests. Every other case
must pass.

## Full-model gates (DeepSeek V4.1 Flash, RTX PRO alone)

The model results were taken with the engine-lab DeepSeek CUDA replica linked against this library. Exactness was
checked the same way for every configuration:

1. Every run starts with the engine's exact gates (plain greedy decode, forced write-backs, forced prefetch, the
   speculative path and plain decode again); each must reproduce the stored reference tokens.
2. Every verify window of the speculative decoder is checked against plain greedy decode (`verify 64/0` = 64 windows,
   0 mismatches) and every chat's token vector must be identical to the plain greedy one.
3. Load-time features are compared against a fresh process: the persistent worker's requests, a request after
   `ss_dg_reset_open`, and requests with write-back deferral switched at runtime through `ss_dg_set_sched` all produced
   bitwise-identical output records to fresh processes started with the same options (6 of 6 identical).
4. Pipelined versus serial loading: identical placement and identical outputs; only the load timers change.

Load time (pipelined loader, `DG_LOAD_SERIAL` unset) is reported by the engine's load timers; with the serial loader for
comparison, run the same command with `DG_LOAD_SERIAL=1`. Expect the expert fill to be bound by the drive's read rate.

Throughput varies between process loads on this machine (by up to about 10-17 % between loads, from host memory and
PCIe conditions), so compare configurations inside one load (the persistent worker's runtime switches) or with
back-to-back A/B/A loads.
