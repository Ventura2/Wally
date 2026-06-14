"""Quickly verify all raw shards in a dataset directory."""
import tarfile, json, sys, os
from pathlib import Path

def verify_shard(path):
    """Verify a tar shard is readable and has paired jpg+json entries."""
    try:
        with tarfile.open(path, "r") as t:
            names = t.getnames()
            jsons = sorted(n for n in names if n.endswith(".json"))
            jpgs = sorted(n for n in names if n.endswith(".jpg"))
            if not jsons or not jpgs:
                return (0, 0, "no entries")
            if len(jsons) != len(jpgs):
                return (0, 0, f"mismatch: {len(jsons)} json vs {len(jpgs)} jpg")
            # Check first and last JSON are valid
            for sample in [jsons[0], jsons[-1]]:
                member = t.getmember(sample)
                f = t.extractfile(member)
                data = json.loads(f.read())
                assert "action" in data, "missing action"
            # Count unique episode IDs
            eps = set()
            for j in jsons:
                ep = j.rsplit("_", 1)[0]
                eps.add(ep)
            return (len(eps), len(jsons), "OK")
    except Exception as e:
        return (0, 0, str(e))

errors = []
for dataset in sys.argv[1:]:
    d = Path(dataset)
    if not d.is_dir():
        errors.append(f"{dataset}: not a directory")
        continue
    shards = sorted(d.glob("shard_*.tar"))
    if not shards:
        errors.append(f"{dataset}: no shards found")
        continue
    total_eps = 0
    total_steps = 0
    for s in shards:
        eps, steps, status = verify_shard(s)
        if status != "OK":
            errors.append(f"{s.name}: {status}")
        total_eps += eps
        total_steps += steps
    total_gb = sum(s.stat().st_size for s in shards) / (1024**3)
    print(f"{d.name}: {len(shards)} shards, {total_eps} eps, {total_steps} steps, {total_gb:.1f} GB - {'ERROR' if any(s.name in e for e in errors) else 'OK'}")

if errors:
    print("\nERRORS:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
