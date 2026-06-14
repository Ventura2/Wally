import tarfile, sys

def check(path, label):
    try:
        with tarfile.open(path, "r") as t:
            names = t.getnames()
            jsons = [n for n in names if n.endswith(".json")]
            print(f"{label}: {len(names)} entries ({len(jsons)} steps)")
            if jsons:
                print(f"  First: {jsons[0]}, Last: {jsons[-1]}")
                # extract episode IDs
                eps = set()
                for j in jsons:
                    parts = j.rsplit("_", 1)[0]
                    eps.add(parts)
                print(f"  Episodes: {len(eps)}")
    except Exception as e:
        print(f"{label}: CORRUPT - {e}")

check("/workspace/data/raw/minerl_iron/shard_000005.tar", "Iron shard 5")
check("/workspace/data/raw/minerl_diamond/shard_000001.tar", "Diamond shard 1")
