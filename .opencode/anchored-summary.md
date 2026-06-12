## Goal
- Get ROCm GPU compute working on user's WSL2 setup with RX 6700 XT (RDNA2) to train/run a world model

## Constraints & Preferences
- WSL2 Ubuntu 24.04, Windows 11, AMD Adrenalin 26.6.1 driver
- ROCm 7.2.4 installed in WSL2 (NOT 7.2.3 as README references)
- Windows SDK 10.0.26100.0 already installed
- GCC 13.3.0, CMake 3.28.3 available
- User wants research on internet before trying fixes on own
- User prefers committing useful code, deleting throwaway scripts
- `wsl -d Ubuntu-24.04` fails; correct name is `wsl -d Ubuntu`
- 32GB system RAM available
- Adrenalin driver warning: "Windows driver is old, please update it" (non-fatal)

## Progress
### Done
- Built librocdxg v1.2.0 from source (commit 29a33c9, latest) using cmake toolchain file
- Installed to `/opt/rocm/lib/librocdxg.so.1.2.0`
- Added `0x73DF,10,3,1` (RX 6700 XT, gfx1031) to `/opt/rocm/share/rocdxg/dids.conf`
- `rocminfo` shows GPU agent correctly: `gfx1031`, AMD Radeon RX 6700 XT, 12.8GB VRAM
- Made env vars persistent in `~/.bashrc`: `HSA_ENABLE_DXG_DETECTION=1`, `PATH=/opt/rocm/bin:$PATH`
- Documented full GPU setup in `AGENTS.md`, updated `.gitignore`
- Deleted root `test_chunks.py` and `test_training.py` (one-off smoke tests)
- Installed numpy 2.4.6
- Installed **TheRock multi-arch PyTorch** `torch==2.12.0+rocm7.14.0a20260611` from `https://rocm.nightlies.amd.com/whl-multi-arch/` with `torch[device-gfx1031]`
  - Also installed: `amd-torch-device-gfx1031`, `rocm-sdk-core 7.14.0a`, `rocm-sdk-libraries`, `rocm-sdk-device-gfx1031`, `rocm-sdk-devel`, `triton 3.7.0`
- Ran `rocm-sdk init` — devel contents expanded to `/home/josema/.local/lib/python3.12/site-packages/_rocm_sdk_devel`
- Ran `rocm-sdk test` — 23/26 pass, 3 expected WSL2 failures (amd-smi, rocm-smi, hipconfig)
- **`torch.cuda.is_available()` = True natively** (without manual hsa_init)
- Verified at low-level via ctypes what works/doesn't:
  - `hipInit(0)` returns success, GPU enumerated
  - `hipMalloc(1MB)` returns success, `hipFree` returns success (GPU memory: 12.8GB free)
  - `hipMemcpy H2D` **HANGS** indefinitely
  - Compiled HIP kernel with `hipcc` also hangs (confirmed not PyTorch-specific)
- Identified exact failure point: at SDMA command submission in librocdxg
  - HIP creates SWq/HWq queue objects successfully
  - "Number of allocated hardware queues with low priority: 0, with normal priority: 0, with high priority: 0"
  - Allocates 4MB HSA host memory staging buffer successfully
  - **Hangs immediately after, during D3DKMT command submission to HWq**
- Found via `git log` that librocdxg is up to date (no newer commits)

### In Progress
- None — diagnostic complete, blocker is fundamental

### Blocked
- **Any HIP kernel dispatch (including DMA copy) hangs indefinitely** in WSL2 via librocdxg
- Root cause: librocdxg v1.2.0 cannot submit commands to RDNA2 (gfx1031) hardware queues in WSL2 GPU-P
- `rocm-smi` still fails: `/usr/bin/env: 'python3': No such file or directory`

## Key Decisions
- Used cmake toolchain file (`/home/josema/rocdxg_toolchain.cmake`) for librocdxg build
- Added RX 6700 XT as `0x73DF,10,3,1` overriding librocdxg's RDNA3-only compatibility matrix
- Deleted root `test_*.py` scripts (hardcoded `/workspace/` paths) instead of moving to `tests/`
- Ignored `tools/` (all scripts have hardcoded paths, not reusable)
- Switched to TheRock multi-arch wheels (nightly `7.14.0a` from June 11, 2026) — official PyTorch wheels for ROCm 7.0+ ship with broken gfx1030 kernels
- Skipped PyTorch 2.5+rocm6.2 (segfaults, ABI mismatch with system HIP 7.2) and 2.7-2.9 wheels (don't detect GPU on system HIP 7.2)
- **Concluded WSL2 compute path is dead** — must pivot to alternative approach

## Next Steps
1. **Pivot to one of these (need user decision):**
   - **Windows-native ROCm via TheRock multi-arch**: Install PyTorch on Windows directly. Adrenalin driver exposes D3D12 compute natively, no librocdxg needed. `pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1031]"` in Windows Python.
   - **DirectML on Windows**: AMD-supported, works with RX 6700 XT, less performant for compute but works
   - **CPU training**: Works immediately for development, slow for production
   - **Wait for librocdxg fixes**: Active project but no clear timeline
2. Document final decision in `AGENTS.md` and add Windows-native setup section

## Critical Context
- **RX 6700 XT is NOT in ROCm 7 official supported GPUs list** — docs say: "prebuilt ROCm libraries are not officially supported and will cause runtime errors"
- TheRock community multi-arch wheels explicitly support gfx1031 (RX 6700 XT)
- **TheRock SUPPORTED_GPUS.md warns**: "✅ Build Passing does not imply runtime is functional" — gfx1031 is Build Passing but not Sanity Tested
- **librocdxg v1.2.0 cannot submit compute commands to RDNA2 (gfx1031) hardware queues in WSL2** — confirmed via raw ctypes tests bypassing PyTorch
- PyTorch wheels tested: 2.5+rocm6.2 (segfault), 2.7-2.9+rocm6.3 (no GPU detect), 2.10+rocm7.0 (kernels fail "invalid device function"), 2.12.0+rocm7.14.0a (detects GPU but compute hangs)
- `libtorch_hip.so` hardcodes bundled HIP/HSA via RPATH (verified with `ldd`)
- HIP 7.0 and HIP 7.2 are ABI-incompatible (LD_PRELOAD causes segfault)
- `libamdhip64.so.6` symlink to v7 lib caused segfault
- ROCm 7.2.4 has `libamdhip64.so.7.2.70204`, `libhsa-runtime64.so.1.18.70204`
- `librocdxg` is loaded from `/opt/rocm/lib/librocdxg.so.1.2.0` (system, not bundled)
- Kernel packs (.kpack format) located at:
  - `/home/josema/.local/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack/` (blas, rand, fft, rccl)
  - `/home/josema/.local/lib/python3.12/site-packages/torch/.kpack/torch_gfx1031.kpack`
- `dmesg` `dxgkio_query_adapter_info: Ioctl failed: -22` is benign
- `rocminfo` shows benign `Warning: Windows driver is old`
- WSL2 `/tmp` cleaned between sessions; use `~/` for persistent builds
- Adrenalin 26.6.1 may be too old for WSL2 compute — librocdxg emits "Windows driver is old, please update it" warning
- PowerShell issue: `2>/dev/null` and `2>&1` get mangled by PowerShell redirect parser; use `>` with `tee` or write to file
- PowerShell issue: `sed` and `ls -la` get intercepted; use `Select-String`, `Get-ChildItem`
- TheRock multi-arch install command: `pip3 install --break-system-packages --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1031]"`
- HIP logging `AMD_LOG_LEVEL=4` reveals:
  - HIP Version: 7.2.53211.97f5574fe2
  - Direct Dispatch: 1
  - HMM support: 0, XNACK: 0, Direct host access: 0
  - Max SDMA Read Mask: 0x3, Max SDMA Write Mask: 0x3
- librocdxg code path on hang: `SubmitToHwQueue` → `SubmitCommandToHwQueue` (D3DKMT ioctl) → `SignalSynchronizationObjectFromGpu` → `WaitForSynchronizationObjectFromCpu` (fence never signaled)

## Relevant Files
- `~/librocdxg/`: librocdxg source (v1.2.0, develop branch, commit 29a33c9)
- `~/rocdxg_toolchain.cmake`: cmake toolchain with `WIN_SDK` path
- `/opt/rocm/lib/librocdxg.so.1.2.0`: installed library
- `/opt/rocm/share/rocdxg/dids.conf`: GPU device ID config (has `0x73DF,10,3,1`)
- `librocdxg/src/wddm/device.cpp`: Contains `SubmitToSwQueue`, `CreateHwQueue`, `SubmitToHwQueue`, `GpuSignal` (the failing function)
- `D:\Projects\Personal\artificial-intelligence\wally\AGENTS.md`: updated with GPU setup section
- `D:\Projects\Personal\artificial-intelligence\wally\.gitignore`: updated with data/logs/archives/dev-scripts
- `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared\`: Windows SDK headers
- `/usr/lib/wsl/lib/libdxcore.so`: Microsoft DXCore library in WSL2
- `/dev/dxg`: WSL2 GPU-P device file
- `https://github.com/ROCm/librocdxg/issues/22`: RX 9060 XT gfx1200 VramAvail bug (RDNA4, different from our case)
- `https://github.com/ROCm/librocdxg/issues/34`: RX 7800 XT WSL2 issue (RDNA3, no useful details)
- `https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md`: gfx1031 supported, warns Build Passing != runtime
- `https://github.com/ROCm/TheRock/pull/5283`: draft PR "Package WSL ROCDXG with ROCm runtime" (May 2026)
- `https://github.com/ROCm/TheRock/pull/1629`: merged "Adding support for RDNA2 gfx103X cards"
- `https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html`: official GPU support list (no RX 6700 XT)
- `https://rocm.nightlies.amd.com/whl-multi-arch/`: TheRock multi-arch wheel index
- `/home/josema/.local/lib/python3.12/site-packages/torch/lib/`: PyTorch bundled HIP/HSA libs
- `/home/josema/.local/lib/python3.12/site-packages/_rocm_sdk_devel/`: expanded devel tree (post `rocm-sdk init`)
- `/home/josema/.local/lib/python3.12/site-packages/_rocm_sdk_libraries/.kpack/`: gfx1031 kernel packs
- `/home/josema/.local/lib/python3.12/site-packages/torch/.kpack/torch_gfx1031.kpack`: PyTorch gfx1031 kernel pack
- `~/d/Projects/Personal/artificial-intelligence/wally/`: main project directory (Windows path)
