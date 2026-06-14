import zipfile, subprocess, sys

# Get all trajectories from ZIP
path = "/workspace/data/external/MineRLObtainDiamond-v0.zip"
with zipfile.ZipFile(path) as zf:
    trajs = set()
    for name in zf.namelist():
        parts = name.strip("/").split("/")
        if len(parts) >= 2 and parts[1].startswith("v"):
            trajs.add(parts[1])
trajs = sorted(trajs)
print(f"Total trajectories in ZIP: {len(trajs)}")

# Get episodes we have in shard 1 and 2
have_eps = set()
for shard_num in [1, 2]:
    shard = f"/workspace/data/raw/minerl_diamond/shard_{shard_num:06d}.tar"
    result = subprocess.run(["tar", "tf", shard], capture_output=True, text=True, timeout=120)
    for line in result.stdout.splitlines():
        if line.endswith(".json"):
            ep = line.rsplit("_", 1)[0]
            have_eps.add(ep)
print(f"Have unique episodes: {len(have_eps)}")

# Find missing - match traj dir name against the MineRLObtainDiamond-v0_ prefix in tar entries
missing = []
for t in trajs:
    prefix = "MineRLObtainDiamond-v0_" + t
    if prefix not in have_eps:
        missing.append(t)

print(f"Missing episodes: {len(missing)}")
for m in missing:
    print(f"  {m}")
