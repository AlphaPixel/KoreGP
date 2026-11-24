# Test host: argon-pixel

How KoreGP is built and tested on the one NVIDIA machine we have, without
breaking the driver that machine depends on. Read this before running
anything on the host. Facts are as inventoried on 2026-11-20.

## Access

| | |
|---|---|
| SSH | `ssh argon-isotopia` (an alias in the workstation's `~/.ssh/config`) |
| User | `argon-pixel` |
| Work dir | `~/koregp/` and nothing else |

## Inventory

| | |
|---|---|
| GPU | RTX 2070 SUPER, TU104, `10de:1e84`, PCI `0000:09:00.0`. **Only GPU** (Ryzen 7 2700X has no iGPU) |
| BARs | BAR0 16 MB, BAR1 256 MB (resizable, not resized), BAR3 32 MB |
| Driver in use | proprietary `nvidia` 580.159.03 (DKMS, built for all three installed kernels) |
| Kernel | 7.0.0-38-generic running; 7.0.0-43 installed and **pending** (next boot changes kernel) |
| OS | Ubuntu 24.04.5, system Mesa 25.2.8 |
| nouveau | module present for the running kernel; blacklisted by `/lib/modprobe.d/nvidia-graphics-drivers.conf` (`blacklist nouveau` + `alias nouveau off`) |
| GSP firmware | `/lib/firmware/nvidia/tu104/gsp` (→ tu102): 535.113.01 and 570.144 |
| Display | gdm greeter only (Wayland, `gnome-shell` + `Xwayland`, ~205 MiB VRAM). Nobody logs in at the desk; the box is used over SSH |
| Other tenants | Docker containers and other projects in `~`, **no GPU use** |
| CPU / RAM / disk | 16 threads, 46 GB, 944 GB free on `/` (ext4) |

## Why this is delicate

1. **One GPU, two kernel drivers, mutually exclusive.** NVK needs the
   `nouveau` kernel module bound to the card. The desktop and everything else
   on the box needs `nvidia`. They cannot both hold `0000:09:00.0`.
2. **Rebooting this host is not safe to do remotely.** `~/BOOTINFO.md` on the
   host documents it: Linux boots in legacy BIOS/CSM from `nvme2n1`, Windows is
   installed UEFI on `nvme0n1`, and the firmware may fall back to Windows on a
   power cycle. The boot log has a nine-day gap (2026-10-19 → 10-28) consistent
   with exactly that. GRUB has a hidden menu and zero timeout. If the box boots
   Windows, SSH is gone until someone is physically there. On top of that the
   next boot picks up kernel 7.0.0-43.

So the design rule is: **no reboots, no persistent system state, and a
plain reboot must always land back on stock nvidia.**

## Safety model

Three rings. Agents may act freely in ring 1, ring 2 is done and closed,
ring 3 needs the owner's go-ahead every time.

### Ring 1 — user-local (always allowed)

Everything under `~/koregp/`:

| Path | What |
|---|---|
| `~/koregp/KoreGP` | copy of this repo (`scripts/push.sh`) |
| `~/koregp/mesa` | Mesa checkout at the pinned tag (`MESA_TAG` in `scripts/host/env.sh`) |
| `~/koregp/build`, `build-release`, `build-asan` | Meson build dirs (debug with asserts; release for measuring; sanitizers) |
| `~/koregp/prefix`, `prefix-release`, `prefix-asan` | **private install prefixes** — the only places our Mesa is ever installed |
| `~/koregp/logs` | one directory per hardware run (`hardware-suite.sh`) |
| `~/koregp/venv` | Python venv with Meson |
| `~/.cargo/bin/{bindgen,cbindgen}` | Rust build tools |

Our Mesa is selected per process via `kgp_run` (`VK_DRIVER_FILES` +
`LD_LIBRARY_PATH`, see `env.sh`). It is never on a default search path, so no
other program on the host can pick it up. The build is headless
(`-Dplatforms=`, no GL/EGL/GBM/GLX), so it does not even contain the pieces
that could shadow the desktop stack.

Verified 2026-11-20: with the private build installed, system `vulkaninfo`
still reports `NVIDIA GeForce RTX 2070 SUPER / 580.159.03`.

### Ring 2 — system packages (done; do not extend without asking)

Eleven additive dev packages were installed on 2026-11-20, behind a gate
that simulates the transaction and refuses unless it is `0 upgraded, 0 to
remove` and touches no `nvidia*`, `linux-*`, `mesa*`, `libdrm*`, firmware,
grub or initramfs package:

```
bison flex glslang-tools libclang-19-dev libclang-cpp19-dev libclc-19-dev
libelf-dev libllvmspirvlib-19-dev libllvmspirvlib19.1 spirv-tools vulkan-tools
```

Undo: `sudo apt-get remove` that list. Any further package needs the same
gate and the owner's OK.

### Ring 3 — the driver swap (owner go-ahead per window)

A **runtime** swap, no reboot, nothing written outside `/run`:

```
scripts/host/to-nouveau.sh [minutes]   # nvidia -> nouveau, arms auto-restore
scripts/host/to-nvidia.sh              # nouveau -> nvidia, any time, any state
scripts/host/with-nouveau.sh [-w min] <cmd...>   # swap, run, always restore
scripts/host/w0.sh swap|deadman|bar1 <n>|display # the first windows, scripted
scripts/host/hardware-suite.sh         # everything that needs the GPU; inside a window only
scripts/host/gpu-status.sh             # read-only
```

`to-nvidia.sh`, `gpu-status.sh` and `hardware-suite.sh` are safe to run at any
time (the last refuses unless the GPU is already on nouveau). The others are
ring 3.

`to-nouveau.sh`:

1. Refuses unless the GPU is on `nvidia`, nobody but the gdm greeter is on
   `seat0`, and no compute process is using the GPU.
2. Records which of `gdm` / `nvidia-persistenced` were running.
3. **Arms a dead-man timer** (`systemd-run --on-active`) that runs
   `to-nvidia.sh` when the window expires. A dropped SSH session or a crashed
   agent cannot leave the host on nouveau.
4. Stops those units, unloads `nvidia_uvm nvidia_drm nvidia_modeset nvidia`.
5. Loads nouveau with `modprobe -C <empty dir>`, which sidesteps the blacklist
   for this one call without editing it. Default args: `modeset=2` (headless,
   so nouveau never takes the console and unloads cleanly) and
   `config=NvGspRm=1` (GSP firmware, for real clocks).
6. Any failure after step 3 triggers `to-nvidia.sh`.

`to-nvidia.sh` disarms the timer, kills whatever still holds the nouveau
node, unloads nouveau, loads nvidia, restarts the recorded units and checks
`nvidia-smi`. If it cannot get nvidia back it says so and stops. **It never
reboots.**

What a window costs the host: the gdm greeter and the monitor output go away
for the duration; CUDA/NVENC are unavailable. Docker stacks, SSH and
everything non-GPU keep running.

Checked without opening a window (2026-11-20): all scripts pass `bash -n`;
`to-nvidia.sh` was run from a root systemd transient unit exactly as the
dead-man arms it, with the GPU on nvidia, where it is a no-op. That dry run
found the dead-man would have died on an unset `$HOME`; fixed, and re-run
clean with and without `KGP_ROOT`. A dry run in that state cannot exercise
the path that matters, unloading nouveau and loading nvidia, and that is
where the first real window failed ("The first window", below).

`scripts/host/break-nvidia-udev-loop.sh` is a recovery tool for one specific
failure described there; it is not part of a normal window.

#### BAR1 resize (optional, extra risk)

BAR1 is the CPU's window onto VRAM: 256 MiB on this card, advertised as
resizable up to 8 GiB, never resized. `kgpMalloc`'s default memory is mapped
through it (delta D1 in `nvk-mapping.md`). The kernel can resize a BAR at run
time through `/sys/bus/pci/devices/0000:09:00.0/resource1_resize` while no
driver is bound, which is exactly the gap between unloading nvidia and loading
nouveau.

`KGP_BAR1_SIZE=13 to-nouveau.sh` (or `w0.sh bar1 13`) tries it: 13 means
2^13 MiB. Nothing persists; `to-nvidia.sh` writes the old size back before
loading nvidia.

Why it is not on by default:

- The host boots through CSM. Firmware in that mode usually leaves no 64-bit
  bridge window for an 8 GiB BAR, so the likely outcome is a clean refusal
  (the kernel keeps the old size). That is still an answer worth one attempt.
- If the kernel half-succeeds and leaves BAR1 unassigned, neither driver
  works until a reboot. `to-nouveau.sh` checks for that and restores at once,
  but the restore can then fail too. That is a failure mode the plain swap
  does not have.
- Whether the proprietary driver is happy with a BAR1 that was resized and
  put back within one boot is unknown.

So: only after `w0.sh swap` and `w0.sh deadman` have both passed, and only
with the owner's go-ahead for that specific risk. N1 does not need it.

#### Display windows (extra risk)

Ordinary windows load nouveau with `modeset=2`: it never touches the
monitor or the console, which is part of why it unloads cleanly. Putting a
KoreGP picture on the monitor (plan R10) needs `modeset=1`: nouveau then
drives the display, and the kernel console moves onto its framebuffer.

`w0.sh display` is the shakedown for that, and runs nothing of KoreGP: swap
with `modeset=1`, `display-report.sh` (read-only: connectors, modes,
console, the kernel's messages), swap back. Facts as of 2026-11-23, under
nvidia: one monitor, 3840x2160 on `HDMI-A-2`; the console is bound to
nvidia's framebuffer and has come back there after every window.

The shakedown ran on 2026-11-23 15:00 and passed (`results-n1.md`, eighth
window): the unload below went through, with the console bound. One pass,
with nothing rendering. A window in which a KoreGP program was DRM master
(`w0.sh present`) ran on 2026-11-23 19:17 and restored cleanly too
(eleventh window). A display window still needs its own go-ahead.

`w0.sh present` also passes `atomic=1`: Mesa's display WSI presents
through atomic modesetting only, and nouveau keeps that ioctl from user
space by default. The shakedown ran without it; inside the kernel nouveau
is atomic either way.

What is different from an ordinary window:

- **The unload.** With the console on its framebuffer nouveau has one more
  user. `to-nvidia.sh` takes the console off first (`vtcon*/bind`), which is
  the usual sequence, and nvidia_drm has unloaded in the same position in
  every window; but if nouveau does not unload, nvidia cannot be loaded and
  the only way back is the reboot this host must not take remotely. The
  dead-man does not help with that: it runs the same script.
- **The monitor** shows the kernel's text console during the window instead
  of going dark.
- `to-nouveau.sh` also hands `/dev/dri/card*` to the test user, for the
  window only: presenting needs the primary node.

So: owner's go-ahead for that specific risk, with the owner able to reach
the machine.

### Never

- `meson install` / `ninja install` to anything but `~/koregp/prefix`.
- Touch `/usr/share/vulkan`, `/etc/vulkan`, `/etc/ld.so.conf*`, `ldconfig`.
- Edit `/etc/modprobe.d`, `/lib/modprobe.d`, `/etc/default/grub`,
  `/etc/grub.d`; run `update-grub`, `update-initramfs`, `dkms`.
- `apt upgrade`, or install/remove anything matching `nvidia*`, `linux-*`,
  `mesa*`, `libdrm*`, `linux-firmware`.
- **Reboot or power off the host.** If a reboot looks necessary, stop and tell
  the owner.
- Touch Docker, or any directory in `~` other than `koregp`.
- Run a ring-3 script without the owner's go-ahead for that window.

## The first window (2026-11-20, 23:46)

The owner ran `w0.sh swap`. The swap to nouveau worked; the tests did not
run; the restore failed; the recovery took the rest of the hour. No reboot.
What it established, and what each failure changed:

| What happened | Cause | Now |
|---|---|---|
| nouveau loaded with `modeset=2 config=NvGspRm=1` right after nvidia, no PCI reset: GSP firmware 570.144, 8192 MiB VRAM, a render node, "Cannot find any crtc" (headless, as intended) | — | three rows of the unknowns table answered: yes, yes, yes |
| every test reported "no GPU" | the render node is `root:render 0660` and the test user is in neither `render` nor `video`; the proprietary driver never needed it | `to-nouveau.sh` hands the node to the invoking user for the window; `hardware-suite.sh` checks it first |
| nouveau was not unloaded on restore | `lsmod \| grep -q` under `pipefail` reads false when grep exits early; and `modprobe -r nouveau` is a silent no-op here, because the nvidia package's `alias nouveau off` applies to removal too | tests read `/proc/modules`; removal bypasses the modprobe config like the load does |
| nvidia modules then loaded and unloaded every 7 s, indefinitely | see below | see below |

**The udev loop.** Ubuntu's `/lib/udev/rules.d/71-nvidia.rules` runs
`modprobe nvidia-modeset/-drm/-uvm` when the nvidia PCI driver registers and
`modprobe -r` of the same when it unregisters. Loading nvidia while nouveau
still holds the card registers and at once unregisters the driver, queueing
one event of each kind; the failed restores did that several times, and the
script's own "reset and retry" added more. From then on each queued "removed"
event fully unloaded the driver (emitting a new "removed") and each queued
"added" loaded it again (emitting a new "added"). gdm, restarted in the
middle of this, could not get a display and held the device, which stretched
the cycles and let the queue grow.

What stopped it: holding `/dev/nvidiactl` open. With the module's use count
above zero the unload half cannot complete, so no new driver events are
produced and udev only has its backlog left
(`scripts/host/break-nvidia-udev-loop.sh`). The backlog was long enough
that, after ten minutes of watching it drain at nine events a minute, the
owner discarded it with `systemctl restart systemd-udevd`. After that the
modules loaded once and stayed, gdm came back with its usual 205 MiB, and
the module use counts matched the values from before the window.

One difference from the baseline remained until 07:14 the next morning:
`nvidia-persistenced` was inactive. Its unit is `StopWhenUnneeded=true` and
is pulled in by the udev event for the nvidia driver appearing
(`SYSTEMD_WANTS` in `71-nvidia.rules`); that event was among those discarded,
so a manual `systemctl start` was stopped again at once. Replaying the event
with `sudo udevadm trigger --action=add /sys/bus/pci/drivers/nvidia` brought
it back, with the module use counts unchanged and udev quiet afterwards. The
host has matched the baseline since then.

Rules that follow, now in the scripts:

- Never load nvidia while nouveau is still bound; if nouveau will not
  unload, stop and say so.
- `udevadm settle` before and after every module change; `to-nouveau.sh`
  refuses to start while udev is busy.
- No automatic unload-and-reload of nvidia as a "retry".
- `to-nvidia.sh` pins nvidia for two minutes while udev reacts to it.
- A failed restore is reported once. Running the restore script again and
  again is what turned a recoverable state into the loop.

The dead-man was not exercised in that window (the swap's own restore ran
first and failed). It was on 2026-11-21 at 07:41, as its own three-minute
window (`w0.sh deadman`): the timer fired, the corrected `to-nvidia.sh`
unloaded nouveau, loaded nvidia and restarted `nvidia-persistenced` and gdm
with nobody touching the host, and the state afterwards equals the baseline.
A `w0.sh swap` started by mistake while that window was open was refused and
changed nothing; `w0.sh` and `to-nouveau.sh` now refuse that case by name,
and `to-nouveau.sh` also refuses during the two minutes a restore keeps
nvidia pinned.

## Unknowns the first window (W0) has to settle

None of these can be answered without trying; W0 exists to answer them
before any KoreGP code depends on the answers.

Answers from the windows of 2026-11-20 and 2026-11-21 are filled in. Only
the optional BAR1 row is unanswered.

| Question | Answer / if the answer is bad |
|---|---|
| Does `nvidia_drm` unload cleanly once gdm stops? | **Yes** (2026-11-20). |
| Does nouveau's GSP boot work after the proprietary driver has run in the same boot? | **Yes**, with no PCI reset. |
| Is GSP used on Turing with this kernel, and which firmware loads? | **Yes: 570.144.** |
| Does `modeset=2` expose a render node NVK accepts? | **Yes** (2026-11-21): NVK and KoreGP both run on it, once `to-nouveau.sh` hands the node to the test user. |
| Does `nvidia` reload and gdm come back without a reboot? | **Yes**, through the script alone (2026-11-21). The first window's restore needed manual recovery; the corrected `to-nvidia.sh` has since done it once from a clean start. |
| Does the dead-man fire and restore correctly on its own? | **Yes** (2026-11-21, `w0.sh deadman`). |
| Can BAR1 be resized between the two drivers, and does it restore? (optional) | leave it at 256 MiB; D1 stands as written |

W0 procedure and exit criteria are in `plan.md`. `hardware-suite.sh` saves
nouveau's dmesg lines, which answer the GSP and render-node rows.

## Rejected alternatives

- **GRUB one-shot nouveau boot entry.** Needs a reboot per switch, edits
  `/etc/grub.d` and regenerates `grub.cfg`, and bets on the fragile boot path
  twice per window.
- **System-wide Mesa (PPA, `/usr/local`, distro package rebuild).** Changes
  what the desktop loads. The private prefix gives the same result for our
  processes only.
- **VFIO passthrough to a VM.** Still unbinds nvidia from the host, plus a VM.
- **A second GPU.** The clean answer if windows become a nuisance: any spare
  card for the desktop on nvidia-or-whatever, the 2070 permanently on nouveau
  (or the reverse). Needs hardware and a free slot; not assumed.

## Offline tier (no window needed)

Mesa's `libnouveau_noop_drm_shim.so` fakes the nouveau kernel interface
(`VM_INIT`, `VM_BIND`, `EXEC` are no-ops) and is built into the private prefix
(`-Dtools=drm-shim`). With `LD_PRELOAD` and `NOUVEAU_CHIPSET=164` the userspace
driver runs as a TU104 **while nvidia stays bound**. Nothing executes, but
device open, allocation, shader compilation and command recording all run.

`kgp_offline <cmd>` in `env.sh` sets that up for one command; the Meson suite
`koregp-offline` does the same for the tests.

Stock Mesa 26.3.1 does not get this far in an assert-enabled build: the
winsys probes for a video-decode class the shim does not advertise and
asserts. `mesa-patches/0002` makes the probe fail quietly instead. (An
earlier note here blamed a missing compute class; the shim does advertise
one.) With the patch, both stock NVK and KoreGP run on the shim, which is
what makes the NVK-versus-KoreGP pushbuf comparison possible.

The shim stubs the syncobj ioctls and never signals anything, so KoreGP
treats every submit as complete when `NOUVEAU_DISABLE_FENCES` is set (the
shim sets it), and waits return immediately.

### Root, and what it was used for

Besides the ring 2 package install, root has been used on this host for two
things, both on 2026-11-20 and neither touching the GPU's driver:

- `sudo perf stat` / `perf record` on our own benchmark running against the
  shim (`perf_event_paranoid` is 4, so unprivileged perf is off; the setting
  was not changed).
- four transient systemd units for the dead-man dry run described above.
  They are gone.
