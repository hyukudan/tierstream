# strix-stream

Model-generic weight streaming for inference engines whose model is larger than the device. It keeps a resident set
of weight units (experts, layer blobs) in device memory and streams the rest from a pinned host tier or from NVMe
read unbuffered. Engines read every unit through a per-layer pointer table, so streaming never changes a bit.

- C ABI: `include/strix_stream/ss.h` (prefix `ss_`)
- Design, descriptor schema and engine contract: [docs/design.md](docs/design.md)
- Backends:
  - CPU (tests, unified);
  - CUDA (RTX PRO 6000 / 4090);
  - HIP stub (Strix Halo, unified memory: resident = mapped, miss = local NVMe read in place).

## Build

```
# Linux / WSL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
# Windows (MSVC + Ninja), CPU or CUDA
tools\build-win.bat          &  tools\build-win.bat cuda
```

The tests write small synthetic model files under `SS_TEST_DIR`, or the system temp directory if it is not set. The
CUDA test `ss-test-cuda` needs a GPU. It checksums every streamed unit on the device and probes pinned and NVMe
bandwidth with about 256 MiB each.

`ss-plan DESC.json --device-gib X --host-gib Y` validates a descriptor and prints its tier plan, in the same format
hyperloom's planner reads.

## Status

v1, local only. The CPU tests pass on MSVC and on GCC with ASan, UBSan and TSan. The CUDA backend is built; its GPU
test runs in one short window. The HIP backend has not been built yet (no ROCm on this machine).

## Credits

The streaming mechanisms here (scored residency, next-layer router lookahead, unbuffered expert reads into device
slots, hits-first gating) follow published descriptions of weight-streaming runtimes for large MoE models. No
third-party source code is included: every file in this repository is new code written for this project (see
NOTICE). If a ported file is ever added, it keeps its original license notice in the file and in NOTICE.

## License

MIT, see [LICENSE](LICENSE).
