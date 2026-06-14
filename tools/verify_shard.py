"""Verify a raw shard output."""
import tarfile, json

path = "/workspace/data/raw/minerl_treechop/shard_000001.tar"
t = tarfile.open(path)
members = t.getmembers()
print(f"Total members: {len(members)}")
print(f"First 5 names: {[m.name for m in members[:5]]}")
print(f"Last 5 names: {[m.name for m in members[-5:]]}")

# Extract a sample JSON to check
json_members = [m for m in members if m.name.endswith(".json")]
if json_members:
    sample = t.extractfile(json_members[0])
    meta = json.loads(sample.read())
    print(f"\nSample JSON keys: {list(meta.keys())}")
    print(f"Action keys: {list(meta['action'].keys())[:10]}...")
    print(f"Sample action: {meta['action']}")

jpg_members = [m for m in members if m.name.endswith(".jpg")]
print(f"\nJPG count: {len(jpg_members)}")
print(f"JSON count: {len(json_members)}")
# Check they match
jpg_keys = set(m.name.replace(".jpg", "") for m in jpg_members)
json_keys = set(m.name.replace(".json", "") for m in json_members)
print(f"JPG-JSON match: {jpg_keys == json_keys}")
