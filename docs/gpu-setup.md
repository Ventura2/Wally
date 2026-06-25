# GPU setup

AMD RX 6700 XT (RDNA 2, gfx1031) GPU setup. The project ships two paths:

- **Windows-native** via TheRock multi-arch PyTorch (recommended for training, planning, validation, deployment)
- **WSL2** for the `wally-dev` Podman container (data collection, `wally-play --relay`)

WSL2 GPU compute is **broken** (librocdxg SDMA hang — see "WSL2 compute status" below). Do not attempt training or planning in WSL2.

---

## Training always uses GPU

`wally-train` and `wally-train-curriculum` always train on a GPU. They
default to `--device cuda` and **exit with code 2** if
`torch.cuda.is_available()` is False. CPU is exposed only as an explicit
`--device cpu` escape hatch used by a handful of fast smoke tests on tiny
configs (it logs a warning). There is no environment variable that
re-enables a silent CPU fallback and there is no auto-fallback.

If `wally-train` exits with the "Training requires a GPU but
torch.cuda.is_available() is False" error, the active venv almost
certainly has a CPU-only torch build. Reinstall from the TheRock
multi-arch index below.

---

## Windows — recommended for training

Use **TheRock multi-arch PyTorch** nightly wheels with the `device-gfx1031` extra. This installs PyTorch with the AMD ROCm runtime and gfx1031-specific (RX 6700 XT) kernel packs, using AMD's official Adrenalin driver D3D12 compute path on Windows directly.

### Prerequisites

- **Windows**: AMD Adrenalin driver (any recent version that ships with the D3D12 driver)
- **Python**: 3.12 or 3.13 (3.14 not yet supported by TheRock wheels as of June 2026)
- **No WSL2 librocdxg needed** for training on Windows

### Install TheRock PyTorch

```powershell
# Create venv
python -m venv .venv-windows

# Upgrade pip
.\.venv-windows\Scripts\python.exe -m pip install --upgrade pip

# Install TheRock multi-arch PyTorch with gfx1031 (RX 6700 XT) kernel packs
.\.venv-windows\Scripts\python.exe -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ "torch[device-gfx1031]"

# Install torchvision from the same index (otherwise pip installs the CPU/wrong-ABI wheel)
.\.venv-windows\Scripts\python.exe -m pip install --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ --force-reinstall --no-deps torchvision

# Install the wally project in editable mode (this will also pull numpy, etc.)
.\.venv-windows\Scripts\python.exe -m pip install -e .
```

Total download: ~1GB (rocm-sdk-core is 745MB, rocm-sdk-libraries 116MB, amd-torch-device-gfx1031 45MB).

### Verify GPU compute

```powershell
.\.venv-windows\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True AMD Radeon RX 6700 XT
```

Quick perf check (8K matmul should reach ~10 TFLOPS FP32):

```python
import torch, time
a = torch.randn(8192, 8192, device='cuda'); b = torch.randn(8192, 8192, device='cuda')
torch.cuda.synchronize(); t0 = time.time()
for _ in range(10): c = a @ b
torch.cuda.synchronize()
print(f'{10/(time.time()-t0):.1f} matmul/s')
```

### Known issues (Windows)

- MIOpen warning: `CK grouped conv library not found for device gfx1031: No se puede encontrar el m�dulo especificado.` — benign, falls back to a non-CK kernel path.
- Pip may downgrade numpy to 1.26.4 when installing wally (some transitive dep). This is fine — PyTorch handles both numpy 1.x and 2.x.

---

## Best execution path for the RX 6700 XT (gfx1031)

The Windows-native TheRock setup above is **the best and only well-supported path** for ML training on this card. This section documents why, what alternatives were considered, and what the hardware can and cannot do.

### Why TheRock multi-arch PyTorch on Windows

For an AMD RX 6700 XT (RDNA 2, LLVM target `gfx1031`) running PyTorch training on Windows, the options in 2026 are:

| Path | Status for gfx1031 / Windows | Verdict |
|------|------------------------------|---------|
| **TheRock multi-arch PyTorch** (`torch[device-gfx1031]` from `rocm.nightlies.amd.com/whl-multi-arch/`) | HIP runtime ✅, HIP SDK ❌, TheRock "Build Passing" ✅, "Sanity Tested" ❌ (see [TheRock SUPPORTED_GPUS.md](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)) | **Use this** — only path with first-class PyTorch + ROCm + gfx1031 kernel packs |
| DirectML (`torch-directml`) | Deprecated upstream (Microsoft archived the project in 2024); no PyTorch 2.4+ wheels, no bf16 autocast parity | Rejected — abandoned |
| ZLUDA (CUDA-on-AMD via PTX translation) | Unofficial, requires patching every PyTorch wheel, breaks on every minor torch bump, no CI for gfx1031 | Rejected — unmaintained and unreliable for production training |
| HIP SDK native (`hipcc` + MIOpen, hand-rolled) | HIP SDK column is ❌ for gfx1031 on Windows in the [HIP SDK support table](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/reference/system-requirements.html) — runtime works, debug tools do not | Rejected — no official gfx1031 support; we get the same runtime via TheRock without the toolchain burden |
| CPU torch (no GPU) | Works, ~50–100× slower than the GPU path; cannot train LeWorldModel at any usable step count | Rejected — the trainer is now hard-coded to refuse CPU (see "Training always uses GPU" above) |

TheRock ships gfx1031-specific kernel packs via the `device-gfx1031` extra and exposes them through PyTorch's normal `cuda` API. The same Adrenalin driver D3D12 compute path is used — there is no separate ROCm install, no WSL2 librocdxg, and no `rocm-smi` required.

### RX 6700 XT hardware characteristics relevant to training

| Spec | Value | Implication for training |
|------|-------|--------------------------|
| Architecture | RDNA 2, LLVM target `gfx1031` | bf16 / fp16 native, no fp8 hardware (fp8 is RDNA 3+) |
| Compute units | 40 CUs | Smaller than RDNA 3 (60+ CUs); expect lower per-GPU throughput than an RX 7900 XT |
| VRAM | 12.8 GB GDDR6 | Tight for batch_size > 32 with seq_length > 16 at full fp32 activations — see perf tips below |
| Memory bandwidth | 384 GB/s | Same envelope as RDNA 3 mid-tier; dataloader must keep up or the GPU starves |
| Infinity Cache | 96 MB | Helps matmul-heavy transformer blocks; ineffective for purely streaming workloads |
| BF16 tensor cores | 2× FP16 throughput on RDNA 2 | Always train in bf16 on this card — there is no fp16 win, and bf16 has fp32's exponent range so no `GradScaler` is needed |
| FP8 | Not supported | Do not enable any `torch.float8_e4m3fn` / `e5m2` dtypes — they will silently fall back to fp32 or error |

### Performance tuning tips for 12.8 GB VRAM

The default `configs/lewm_default.yaml` is sized to fit this card. The numbers below assume the default `batch_size: 16`, `seq_length: 16`, `embed_dim: 192`, `vit_tiny_patch16_224` encoder:

- **Keep `use_amp: true` and `amp_dtype: bfloat16`.** bf16 is the only dtype that benefits from RDNA 2 tensor cores and skips the `GradScaler` overhead. Switching to fp16 buys nothing and reintroduces the NaN-gradient risk that the AdaLN-Zero predictor was designed to eliminate.
- **`batch_size: 16`, `seq_length: 16` leaves ~6–8 GB of headroom** on the 12.8 GB budget for activations + optimizer state (Adam = 2× model params in fp32). If you bump `batch_size` to 32 you will OOM at the encoder output. Use gradient accumulation (`--gradient_accumulation_steps N` if you wire it up) for a larger effective batch instead of cranking the per-step batch.
- **Channels-last memory format** (`model.to(memory_format=torch.channels_last)`) is a free win for the CNN encoder path; the default config currently uses `encoder_type: cnn` and the activations are NCHW. Switching to channels-last typically shaves 5–15% off the encoder forward time on RDNA 2. Verify with `torch.cuda.memory_stats()` before/after if you change it.
- **Pin memory + non-blocking H2D.** The dataloader already auto-enables `pin_memory=True` when CUDA is available (`src/wally/data/dataloader.py:58-59`), and the trainer moves `frames` / `actions` with `.to(self.device, non_blocking=True)` (`src/wally/training/trainer.py:92-93`). The `non_blocking=True` flag lets the H2D copy overlap with kernel execution; without it the H2D copy forces a host-side synchronize before the next op, which is the typical 3–8% wall-clock loss on this card.
- **`num_workers: 8`, `prefetch_factor: 4`, `persistent_workers: true`** (default config) is the right zone for a 16-core host. Going higher just thrashes the page cache and the dataloader stops keeping up. The 384 GB/s memory bandwidth is the bottleneck; more workers will not help.
- **Do not enable `torch.compile`** on gfx1031 unless you have measured a win. TheRock ships `torch.compile` but the Inductor backend for ROCm is not as well tuned as the CUDA one; on small models like the LeWM default the compilation overhead dwarfs the runtime savings. Stick to eager mode.
- **MIOpen CK warning is benign.** The "CK grouped conv library not found for device gfx1031" message means MIOpen fell back to a non-Composable-Kernel path. There is nothing to do — gfx1031 does not have an official CK build. The non-CK path is still using the GPU's tensor cores via MIOpen's standard HIP backend.

### Verifying the GPU is actually being used

After `wally-train` starts, the first log line is `Using device: cuda`. To confirm the GPU is real (not a CPU-only torch build that was renamed):

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# True AMD Radeon RX 6700 XT (10, 3, 1)
```

`torch.cuda.get_device_capability(0)` should return `(10, 3, 1)` for the RX 6700 XT (gfx1031). If it returns anything else — or if `torch.cuda.is_available()` is `False` — the active venv has a CPU-only torch build; reinstall from the TheRock multi-arch index as shown in "Install TheRock PyTorch" above.

---

## WSL2 — for collector only

The RX 6700 XT (RDNA2) is **not** in AMD's official WSL2 ROCm compatibility matrix (which only lists RDNA3/RDNA4 and Ryzen AI APUs). However, GPU **detection** works with a custom `dids.conf` entry. The setup requires building AMD's open-source `librocdxg` library from source.

### Prerequisites

- **Windows**: AMD Adrenalin driver (≥ 26.2.2 for WSL GPU-P support), Windows SDK 10.0.26100.0
- **WSL2 Ubuntu 24.04**: ROCm 7.2.x, GCC ≥ 11.4, CMake ≥ 3.15
- Windows SDK must be installed at `C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\`

### Build librocdxg

```sh
# Clone the repo (to a persistent path, NOT /tmp which gets cleaned)
git clone https://github.com/ROCm/librocdxg.git ~/librocdxg
cd ~/librocdxg

# Create a cmake toolchain file to pass WIN_SDK (path has spaces, breaks -D flag)
cat > ~/rocdxg_toolchain.cmake << 'EOF'
set(WIN_SDK "/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0/shared")
EOF

# Configure and build
mkdir -p build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/rocdxg_toolchain.cmake
make -j$(nproc)
sudo make install
```

### Configure dids.conf for RX 6700 XT

The `dids.conf` at `/opt/rocm/share/rocdxg/dids.conf` allows adding unsupported device IDs:

```
# Add this line (device_id, gfx_major, gfx_minor, gfx_stepping)
0x73DF,10,3,1    # Radeon RX 6700 XT, gfx1031
```

### Enable GPU detection

```sh
# Add to ~/.bashrc for persistence
echo 'export HSA_ENABLE_DXG_DETECTION=1' >> ~/.bashrc
echo 'export PATH=/opt/rocm/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

### Verify

```sh
rocminfo
# Look for Agent with: gfx1031, AMD Radeon RX 6700 XT, Chip ID 0x73df
```

### WSL2 compute status: BROKEN

**librocdxg v1.2.0 cannot submit compute commands to RDNA2 (gfx1031) hardware queues in WSL2.** The D3DKMT command submission to the GPU's hardware queue processor never completes — the GPU never executes the command, so fences are never signaled, and HIP waits forever.

**What works in WSL2:**
- `rocminfo` enumerates GPU correctly (gfx1031, 12.8GB VRAM)
- `hsa_init` succeeds, GPU agent detected
- `hipInit(0)` returns success
- `hipMalloc` / `hipFree` (GPU memory allocation)
- `torch.cuda.is_available()` = True
- `torch.empty(...)` (memory allocation)

**What hangs (librocdxg SDMA submission failure):**
- `hipMemcpy` (any direction)
- Any HIP kernel launch
- `torch.zeros`, `tensor.to('cuda')` (PyTorch ops)
- Custom HIP kernels compiled with hipcc

**Diagnostic log from `AMD_LOG_LEVEL=4`:**
```
:3:rocdevice.cpp :2871: Number of allocated hardware queues with low priority: 0, with normal priority: 0, with high priority: 0
:3:rocdevice.cpp :2952: Created SWq=0x... to map on HWq=0x...
:4:rocdevice.cpp :2019: Allocate hsa host memory 0x..., size 0x400000
<HANGS HERE — D3DKMT command submission>
```

The HWq object is created in userspace but the D3DKMT command submission to the GPU's hardware queue processor never completes. This is a librocdxg limitation, not something fixable from the user side. No known GitHub issues for "command queue hang" or "RDNA2 compute" in librocdxg.

**Do not attempt training, planning, or any compute workload in WSL2.** Use the Windows-native TheRock setup instead.

### Known issues (WSL2, collector only)

- `rocm-smi` is a Python script and may fail if `python3` isn't on PATH in the shell
- `dmesg` will show `dxgkio_query_adapter_info: Ioctl failed: -22` — this is from the amdgpu kernel module and is **benign** (userspace path works fine)
- `rocminfo` shows `Warning: Windows driver is old` — this is a non-fatal warning
- Adrenalin 26.6.1 driver triggers this warning; older or newer drivers may or may not

---

## How to launch training on GPU

Concrete checklist for `wally-train` on the RX 6700 XT. The pitfalls below
are all real and have cost hours of debugging; this section exists so the
next person doesn't repeat them.

### 1. Activate the right venv

Only the **Windows-native venv** (`.venv-windows`) works for training. The
WSL2 collector venv cannot submit compute to RDNA2 (see WSL2 section above).

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\Activate.ps1"
```

### 2. Set env vars (every shell)

These two are required for `torch` to find the ROCm runtime DLLs and for
`wally` to be importable from the `src/` layout:

```powershell
$env:PATH = "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Lib\site-packages\_rocm_sdk_core\bin;$env:PATH"
$env:PYTHONPATH = "D:\Projects\Personal\artificial-intelligence\wally\src"
```

The `_rocm_sdk_devel\bin` path (which also has the HIP DLLs) is an
alternative if you installed `rocm[devel]`. Either is fine.

### 3. Verify GPU before training (always do this)

```powershell
& "D:\Projects\Personal\artificial-intelligence\wally\.venv-windows\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# Should print: True AMD Radeon RX 6700 XT (10, 3, 1)
```

If `cuda.is_available()` is `False`, the active venv has a CPU-only torch
build - go back to [Install TheRock PyTorch](#install-therock-pytorch).

### 4. Pre-flight: convert raw shards to training shards

`wally-train` does **not** read raw collector shards. You must run
`wally-convert` first:

```powershell
& .venv-windows\Scripts\python.exe -m wally.cli.convert `
    --input data/raw/<task> `
    --output data/shards/<task> `
    --config configs/converter_default.yaml
```

Raw → converted is slow on CPU (~1-2 min per 1 GB raw shard, mostly JPEG
decode + resize). Plan accordingly.

### 5. Pick the data subset via the YAML config

The training CLI takes no `--data-dir` flag. The only way to select shards
is the `training.data_dir` field in the YAML config:

```yaml
training:
  data_dir: data/shards/treechop_full   # train on this shard directory
  output_dir: checkpoints/my_run
  num_workers: 4                       # 0 = single-threaded (slow!), 4-8 is the right zone
```

`wally-train` globs `**/*.tar` under `data_dir`, so subdirectories work.
To train on a single task, point `data_dir` at `data/shards/<task>/` (each
subdirectory under `data/shards/` is one task).

### 6. Run training

```powershell
& .venv-windows\Scripts\python.exe -m wally.cli.train `
    --config configs/lewm_default.yaml `
    --log-file logs/train.log
```

The first log line should be `Using device: cuda`. If it's `cpu`, **stop** -
the venv is wrong, the run is useless, and you'll burn hours for nothing.

### 7. Monitor GPU usage (Task Manager is misleading on AMD)

**Windows Task Manager → Performance → GPU** does **not** show ROCm/HIP
compute utilization. It tracks the 3D engine (DirectX/OpenGL), which is a
different hardware queue than the one compute workloads use. You will see
"3D" stay at 0% even during a busy training run. Don't be fooled.

Reliable ways to verify the GPU is actually working:
- **AMD Adrenalin software** (right-click desktop → AMD Adrenalin →
  Performance tab) — shows real-time GPU clock, temperature, utilization.
  **Caveat:** Adrenalin samples utilization once per second and averages
  it. A small model (e.g. `vit_tiny_patch16_224`, `depth: 4`, batch 16,
  seq 8) does its forward+backward in ~60 ms on the 6700 XT — well
  under the 1 s sampling window. The meter can show 0% even while the
  GPU is doing real work, because each 60 ms burst is lost in the
  average. Watch GPU **clock** or **temperature** instead (they respond
  in <100 ms), or run a synthetic 5-10 s GPU stress (a loop of
  `a @ b` on 8192² matrices) to confirm the meter moves at all.
- **HWiNFO64** (free, third-party) → Sensors tab → GPU Temperature and
  GPU Clock jump during a workload
- **`torch.cuda.memory_allocated()`** from your training script - if
  it's > 0, the GPU is in use
- **The training log** - loss decreasing on a `cuda` device is the
  definitive signal. The trainer logs `fetch=Xs gpu=Ys total=Zs` per
  step (`src/wally/training/trainer.py:245-263`) — if `gpu` is non-zero
  and `fetch` is the dominant term, the data loader is the bottleneck,
  not the GPU.

### 8. Common perf pitfalls

- **`num_workers: 0` is the default** in every wood-* smoke config and
  makes data loading the bottleneck. With the current chunked data
  format (64-frame `.npz`, 4-7 MB each), 4 workers is the sweet spot:
  cold start ~7 s, then 0.5-0.8 s per batch. 8 workers does not help
  and can thrash the page cache on a 16-core host. **Before the
  chunking change**, each `.npz` was a full episode (144-335 MB) and
  per-step fetch time was 30-130 s — that is no longer the case, do
  not "fix" the new numbers by reducing `num_workers`.
- **`checkpoint_interval` must be `<= max_steps`** or no checkpoint is
  ever written.
- **`skip_short: true`** (default) drops whole episodes shorter than
  `seq_length`. With the chunked format this is per-chunk, not
  per-episode — short episodes still contribute their non-empty chunks.
  If your data has many short episodes, this kills your effective
  dataset size silently.
- **Per-step timing in the log** (`fetch=Xs gpu=Ys total=Zs`) is the
  fastest way to tell which side of the bottleneck you're on:
  - `fetch >> gpu` → data loader is starving the GPU. Check
    `num_workers`, disk I/O (NVMe vs SATA), and that you re-converted
    after the chunked format went in (see `src/wally/AGENTS.md` data
    format section).
  - `gpu >> fetch` → data loader is keeping up, model or batch is
    too big. Consider `torch.compile` only after measuring a win on
    gfx1031 (Inductor is less tuned for ROCm than CUDA).
  - Both large → genuine compute-bound, time to scale up the model or
    batch.

---

## Troubleshooting

### Antivirus keeps quarantining `torch_hip.dll`

**Symptom:** pip install of TheRock torch fails with `PermissionError` or
`[WinError 5] Acceso denegado` on `torch_hip.dll` (~70 MB), even after
pip downloads and starts extracting the wheel. The other ~30 DLLs in the
same wheel extract fine.

**Cause:** `torch_hip.dll` matches a heuristic signature (e.g.
`Gen:Variant.Barys.516138` in Bitdefender, similar in Norton) because
the AMD HIP runtime does low-level hardware operations that look like
rootkit behavior. The other PyTorch DLLs don't trip it. **This is a
documented false positive** (search "torch_hip.dll antivirus" - it's a
recurring issue across AMD ROCm wheels).

**Fix in order of effort:**

1. **Process exclusion** in your AV (most reliable). Add both
   `python.exe` and `pip.exe` (or `pip3.12.exe`) from your venv
   `Scripts\` directory as trusted applications. **A folder exclusion
   alone is not enough** if your AV has a "Safe Files" / ransomware
   protection layer (Bitdefender does, Windows Defender does under
   "Controlled folder access") - that layer intercepts writes by
   *process* not by path. Adding python/pip to the trusted-app list
   unlocks the writes.

2. **Threat allow-list entry** in your AV for the specific detection
   name (e.g. `Gen:Variant.Barys.516138`). Available in Bitdefender
   Advanced Threat Defense, Norton, and most enterprise AVs.

3. **Hardlink workaround** as a last resort if no AV config will let
   the write through. The AV appears to flag the file by name, not by
   content - so a write to a different name passes, and a Windows
   hardlink (which is a metadata-only operation) does too. From Python:
   ```python
   import zipfile, shutil, os
   whl = r'...path-to-torch-...whl'
   tmp_alt = r'C:\Users\YOU\AppData\Local\Temp\amd_hip_runtime.bin'
   with zipfile.ZipFile(whl).open('torch/lib/torch_hip.dll') as s, open(tmp_alt, 'wb') as d:
       shutil.copyfileobj(s, d)
   lib_dir = r'...venv...\Lib\site-packages\torch\lib'
   shutil.copy(tmp_alt, os.path.join(lib_dir, 'amd_hip_runtime.bin'))
   os.system(f'cmd /c mklink /H "{lib_dir}\\torch_hip.dll" "{lib_dir}\\amd_hip_runtime.bin"')
   ```
   The hardlinked `torch_hip.dll` is the same file the AV tried to
   quarantine; the AV doesn't see a write because the file already
   exists. This unblocks the install without disabling protection.

4. **Do NOT disable real-time protection globally** as a first resort.
   The hardlink trick and the process exclusion are both strictly safer.

### pip overwrote my TheRock torch with the PyPI build

**Symptom:** After `pip install numpy Pillow ...` (training deps), torch
silently upgraded to PyPI's `torch 2.12.1` (CPU/CUDA) and now
`torch.cuda.is_available()` is `True` but `a @ b` raises
`CUDA error: device kernel image is invalid`. The kpack file is gone
from `torch/.kpack/`.

**Cause:** pip saw the training deps had no pinned torch version and
helpfully "upgraded" to the latest PyPI release, which has no
`amd-torch-device-gfx1031` kernel pack.

**Fix:**
1. Force-reinstall TheRock torch: `pip install --index-url
   https://rocm.nightlies.amd.com/whl-multi-arch/ --force-reinstall
   --no-deps "torch==2.12.0+rocm7.14.0a20260620"`
2. Reinstall the kpack: `pip install --index-url
   https://rocm.nightlies.amd.com/whl-multi-arch/ --force-reinstall
   --no-deps amd-torch-device-gfx1031`
3. Reinstall torchvision from the same index: `pip install --index-url
   https://rocm.nightlies.amd.com/whl-multi-arch/ --force-reinstall
   --no-deps torchvision`
4. Install other training deps with `--no-deps` so pip doesn't pull a
   newer torch: `pip install --no-deps numpy Pillow pyyaml webdataset
   timm einops pydantic wandb matplotlib braceexpand`

### `ModuleNotFoundError: No module named 'torchgen'`

**Symptom:** `import torch` fails partway with this error after a pip
install that errored mid-extract.

**Cause:** pip extracted most of the wheel but bailed before writing
the top-level `torchgen/` directory. The torch package is left
inconsistent.

**Fix:** Reinstall the wheel (force-reinstall). If pip keeps failing
on `torch_hip.dll`, use the hardlink trick above to get the wheel fully
extracted.

### `CUDA error: device kernel image is invalid`

The active torch build doesn't have the gfx1031 kernel pack. Either you
have stock PyPI torch (reinstall from TheRock index) or the
`amd-torch-device-gfx1031` package is missing or stale (reinstall it).

### `torch.cuda.is_available()` returns True but `import torch` takes 30+ seconds and prints a flood of `xnack` warnings

This is fine. The xnack warnings come from ROCm probing CPU/GPU
features. They're informational. The slow import is one-time
initialization. Both go away after the first import.

### `ModuleNotFoundError: No module named 'cv2'` when running the agent tests

**Symptom:** `pytest -m smoke` errors at collection time with
`ImportError: No module named 'cv2'` in `tests/test_viewer.py`,
`test_agent_buffer.py`, `test_play_cli.py`, `test_relay.py`, and a
handful of other tests under `tests/`. The training code (`wally-train`,
`wally-convert`) and the live agent (`wally-play --relay`) import cv2
via `src/wally/agent/relay.py:10` and `src/wally/agent/viewer.py`.

**Cause:** `opencv-python` is not declared in `pyproject.toml`
dependencies. The agent code uses it for JPEG encoding (relay) and the
optional OpenCV viewer; the lean training venv set up per the recipe
above doesn't install it. The training pipeline itself does NOT need
cv2 — only the agent and viewer paths do — so this is only a problem
when you try to run the agent tests.

**Fix:** install it directly:
```powershell
.\.venv-windows\Scripts\python.exe -m pip install --no-deps opencv-python
```

Or add it to `pyproject.toml` `dependencies` and reinstall the project
(`pip install -e .`). After this, all 42 smoke tests pass (8 skipped,
842 deselected — same numbers as the training-only venv).

### Adrenalin shows 0% GPU utilization during a training run

**Symptom:** Task Manager and AMD Adrenalin both show 0% GPU
utilization (and ~6 W power) while `wally-train` is happily logging
decreasing loss and per-step GPU times of 50-70 ms. CPU is at 50%+.

**Cause:** The 6700 XT does ~70k 8192² matmuls/s in pure compute, but
small models (e.g. `vit_tiny_patch16_224` with `depth: 4` and
`batch_size: 16, seq_length: 8`) do each step's forward+backward in
~60 ms. Adrenalin samples utilization once per second and averages —
a 60 ms burst is ~6% of the window, often rounded to 0%. Task
Manager's "3D" panel is worse: it tracks the DirectX/OpenGL hardware
queue, which compute workloads do not touch, so it shows 0%
unconditionally.

**How to confirm the GPU is actually working** (any one is enough):

1. **Loss is decreasing on `cuda` device** in the training log. The
   optimizer's `.step()` only updates weights that the autograd graph
   reached, and autograd only runs on the device the tensors live on.
2. **`torch.cuda.memory_allocated() > 0`** in your script (the trainer
   logs this implicitly via per-step loss).
3. **GPU temperature rises** during a sustained workload (HWiNFO64
   sensors tab is the most responsive).
4. **Synthetic 5-10 s stress** as a one-off sanity check:
   ```python
   import torch, time
   a = torch.randn(8192, 8192, device='cuda')
   b = torch.randn(8192, 8192, device='cuda')
   t0 = time.time(); n = 0
   while time.time() - t0 < 5:
       a @ b; torch.cuda.synchronize(); n += 1
   print(f'{n/5:.1f} matmuls/s')
   # Should print ~9 matmuls/s; Adrenalin should show 50%+ during this
   ```
   If Adrenalin still shows 0% during a 5 s all-GPU loop, the
   Adrenalin telemetry service is broken (rare; restart it).

**Fix:** None needed — the GPU is fine. If you want higher Adrenalin
utilization, scale up the model (`depth: 8-12`, `embed_dim: 384`) or
the per-step work (`batch_size: 32`, `seq_length: 16`) so each step
spans most of the 1 s sampling window.

### `ValueError: <path>.tar: no gopen handler defined`

**Symptom:** `wally-train` (or any `wds.WebDataset(...)` consumer)
crashes on the first iteration with
`ValueError: D:\Projects\...\shard_000001.tar: no gopen handler defined`.

**Cause:** WebDataset 1.0.2 has a bug on Windows when paths are
**absolute** with a drive letter. `urllib.parse.urlparse` sees
`D:\foo\bar.tar` and reports `scheme='d'`, which is not in
`gopen_schemes` (only `http`, `https`, `pipe`, `s3`, `gs`, etc. are
registered), so `gopen` falls through to the `gopen_error` default
handler. Relative paths (`data/shards/treechop/shard_000001.tar`)
parse with `scheme=''` and the local-file branch in
`gopen_file` is used instead.

**Fix:** Always pass a **relative** `data_dir` in the training YAML.
The default `data_dir: data/shards/treechop_full` is already relative
and works. If you need to point at an absolute path, prefix it with
`file:///` (`file:///D:/Projects/.../shard.tar`) — both the
`file:/D:/...` and `file:///D:/...` forms are accepted by `gopen`.
Do not patch `webdataset.gopen` itself; the bug is upstream and the
relative-path workaround is a one-line change in the YAML.
