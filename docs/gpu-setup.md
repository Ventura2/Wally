# GPU setup

AMD RX 6700 XT (RDNA 2, gfx1031) GPU setup. The project ships two paths:

- **Windows-native** via TheRock multi-arch PyTorch (recommended for training, planning, validation, deployment)
- **WSL2** for the `wally-dev` Podman container (data collection, `wally-play --relay`)

WSL2 GPU compute is **broken** (librocdxg SDMA hang — see "WSL2 compute status" below). Do not attempt training or planning in WSL2.

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
