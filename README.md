# KoreGP

KoreGP.org

Test implementation of the KoreGP API — the proposal lives in the
[KoreGP-website](../KoreGP-website/README.md) repository — as a native
frontend inside Mesa, below the Vulkan entry points. First backend: NVIDIA
Turing through NVK's internals (NAK, NIL, nouveau).

Status: the compute subset (the proposal's M0) and most of textures and the
heap (M1) are implemented and run correctly on an RTX 2070 SUPER, with
recording a dispatch costing four to seven times less CPU than through
Mesa's Vulkan driver on the same card. An offline test tier runs without a
GPU and checks that KoreGP sends exactly what that driver sends for the same
work. Raster is written up to a first triangle and passes that offline
check; it has not run on the GPU yet.

- [`docs/plan.md`](docs/plan.md) — milestones, what is done, what is next
- [`docs/results-n1.md`](docs/results-n1.md) — test and benchmark results so far
- [`docs/nvk-mapping.md`](docs/nvk-mapping.md) — the proposal restated for NVIDIA hardware
- [`docs/test-host.md`](docs/test-host.md) — how the test machine is used without breaking its driver
- [`include/koregp/kgp.h`](include/koregp/kgp.h) — the API so far
- `src/koregp/` — the frontend; `tests/` — tests, their Vulkan twins, the benchmark
- `mesa-patches/`, `scripts/` — the two Mesa patches, and build and driver-swap tooling

It builds only as part of Mesa (`scripts/host/build-mesa.sh`); there is no
standalone build.
