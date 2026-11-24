# KoreGP

Test implementation of the KoreGP API (spec: `../KoreGP-website/README.md`)
as a native frontend inside Mesa, on NVK internals, tested on one RTX 2070
SUPER.

Read before working:

- `docs/plan.md` — milestones, task breakdown, what is done.
- `docs/nvk-mapping.md` — the spec restated in NVIDIA terms, with deltas D1–D9.
- `docs/results-n1.md` — what has been measured, and on what.
- `docs/test-host.md` — **the rules for the test host. Not optional.**

## Test host rules (summary; `docs/test-host.md` is authoritative)

The host (`ssh argon-isotopia`, user `argon-pixel`) is someone's workstation
with one GPU on the proprietary NVIDIA driver, other projects, and a boot path
that may not come back from a reboot.

- Work only under `~/koregp/` on the host.
- Install Mesa only to `~/koregp/prefix` / `prefix-release`. Never a system path.
- **Never reboot or power off the host.**
- Never edit modprobe/grub config, run `update-grub`, `update-initramfs`,
  `dkms`, or `apt upgrade`; never install or remove `nvidia*`, `linux-*`,
  `mesa*`, `libdrm*`, firmware.
- **Never run `to-nouveau.sh`, `with-nouveau.sh` or `w0.sh` without the
  owner's go-ahead for that specific window.** A previous go-ahead does not
  carry over. `to-nvidia.sh`, `gpu-status.sh` and `hardware-suite.sh` (which
  refuses outside a window) are always fine.
- If nvidia cannot be restored: stop, report, do not reboot.

## Working loop

```
scripts/push.sh                                    # workstation -> host
ssh argon-isotopia ~/koregp/KoreGP/scripts/host/build-mesa.sh
ssh argon-isotopia 'cd ~/koregp/build && meson test --suite koregp-offline'
```

The offline suite runs on Mesa's DRM shim with nvidia still bound and needs
no window. Make it green first; windows are not for fixing compile errors.
Hardware tests go through `scripts/host/w0.sh swap` (or `with-nouveau.sh
hardware-suite.sh`) so nvidia is always restored.

`KGP_DEBUG=push,sync,vm,gart,asm,nir` on any KoreGP program: pushbuf dump,
wait-after-every-submit, VA bind log, force system memory, NAK assembly, NIR.
`rootv1` selects the experimental root convention. `KGP_DEBUG=1` (or
`validate`, to combine it with the others) is the validation mode:
`src/koregp/kgp_validate.c` says what it checks.

Measure only on the release flavor (`KGP_FLAVOR=release build-mesa.sh`).
`KGP_FLAVOR=asan` builds with sanitizers; run its programs through
`kgp_offline`, not `meson test`.

## Code conventions

- The repo root is symlinked into the Mesa tree as `src/koregp`; the root
  `meson.build` is a Mesa subdir file. Upstream Mesa is changed only through
  `mesa-patches/`. Do not use `..` paths that cross the symlink.
- C, Mesa style, Meson (it is part of Mesa's build).
- Extraction, not forking: copy a function from NVK and cut its Vulkan-object
  plumbing; name the ancestor in a comment so the file stays diffable.
- Call `nvkmd`, NIL, NAK, `nv_push` directly; do not reimplement them.
- New command emission comes with a Vulkan twin in `tests/vk_baseline.c` and
  a `push_matches_nvk` test: KoreGP must push what NVK pushes for the same
  work, or the difference must be explained in the script's mask list.
- Mesa pin: `MESA_TAG` in `scripts/host/env.sh`. Source references in the
  docs are to that tag.
- Editing from Windows: write files with the editor tools, not shell
  heredocs — backslashes and quotes in heredocs get mangled on the way.
