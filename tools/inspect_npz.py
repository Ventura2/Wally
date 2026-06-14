import numpy as np
import os

data_dir = '/workspace/data/external/navigate/MineRLNavigate-v0'
traj = sorted(os.listdir(data_dir))[0]
npz_path = os.path.join(data_dir, traj, 'rendered.npz')
data = np.load(npz_path)
print('Keys:', list(data.keys()))
for k in data.keys():
    v = data[k]
    print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    if v.dtype.kind == 'f' and v.ndim >= 1:
        print(f'      range: [{v.min():.4f}, {v.max():.4f}]')
        if v.ndim == 1 and v.size < 30:
            print(f'      values: {v}')
    elif v.dtype.kind in ('U', 'S', 'O'):
        flat = v.flat[0] if v.size > 0 else 'empty'
        print(f'      sample: {flat}')
    elif v.dtype.kind == 'i' or v.dtype.kind == 'u':
        print(f'      range: [{v.min()}, {v.max()}], unique: {len(np.unique(v))}')
