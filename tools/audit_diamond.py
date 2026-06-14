"""Audit Diamond shards - list all episode IDs."""
import tarfile, sys
from pathlib import Path

shards = sorted(Path(sys.argv[1]).glob("shard_*.tar"))
all_eps = set()
for s in shards:
    try:
        with tarfile.open(str(s), "r") as t:
            for m in t:
                if m.name.endswith(".json"):
                    ep = m.name.rsplit("_", 1)[0]
                    all_eps.add(ep)
        print(f"{s.name}: OK")
    except Exception as e:
        print(f"{s.name}: ERROR - {e}")

print(f"\nTotal unique episodes: {len(all_eps)}")
for ep in sorted(all_eps):
    print(f"  {ep}")
