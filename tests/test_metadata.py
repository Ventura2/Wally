import json

from wally.exporter.metadata import generate_manifest


class TestGenerateManifestCreatesFile:
    def test_creates_manifest_json(self, tmp_path):
        shard_infos = [("shard_000000.tar", 100), ("shard_000001.tar", 50)]
        generate_manifest(shard_infos, tmp_path)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

    def test_creates_output_dir_if_missing(self, tmp_path):
        out = tmp_path / "nested" / "dir"
        shard_infos = [("shard_000000.tar", 10)]
        generate_manifest(shard_infos, out)
        assert (out / "manifest.json").exists()


class TestGenerateManifestFields:
    def test_manifest_has_all_required_keys(self, tmp_path):
        shard_infos = [("shard_000000.tar", 100)]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert "total_transitions" in manifest
        assert "total_episodes" in manifest
        assert "shard_count" in manifest
        assert "shards" in manifest

    def test_total_transitions_sum(self, tmp_path):
        shard_infos = [
            ("shard_000000.tar", 100),
            ("shard_000001.tar", 200),
            ("shard_000002.tar", 50),
        ]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert manifest["total_transitions"] == 350

    def test_shard_count_matches(self, tmp_path):
        shard_infos = [
            ("shard_000000.tar", 100),
            ("shard_000001.tar", 200),
        ]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert manifest["shard_count"] == 2

    def test_shards_list_length(self, tmp_path):
        shard_infos = [
            ("shard_000000.tar", 100),
            ("shard_000001.tar", 200),
            ("shard_000002.tar", 50),
        ]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert len(manifest["shards"]) == 3

    def test_shard_entries_have_path_and_transitions(self, tmp_path):
        shard_infos = [("/data/shard_000000.tar", 42)]
        manifest = generate_manifest(shard_infos, tmp_path)
        shard = manifest["shards"][0]
        assert shard["path"] == "shard_000000.tar"
        assert shard["transitions"] == 42

    def test_shard_path_is_basename_only(self, tmp_path):
        shard_infos = [("/long/path/to/shard_000000.tar", 10)]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert manifest["shards"][0]["path"] == "shard_000000.tar"


class TestGenerateManifestEpisodeIds:
    def test_total_episodes_from_episode_ids(self, tmp_path):
        shard_infos = [("shard_000000.tar", 100)]
        episode_ids = {"ep01", "ep02", "ep03"}
        manifest = generate_manifest(shard_infos, tmp_path, episode_ids=episode_ids)
        assert manifest["total_episodes"] == 3

    def test_total_episodes_zero_without_episode_ids(self, tmp_path):
        shard_infos = [("shard_000000.tar", 100)]
        manifest = generate_manifest(shard_infos, tmp_path)
        assert manifest["total_episodes"] == 0

    def test_total_episodes_none_gives_zero(self, tmp_path):
        shard_infos = [("shard_000000.tar", 100)]
        manifest = generate_manifest(shard_infos, tmp_path, episode_ids=None)
        assert manifest["total_episodes"] == 0


class TestGenerateManifestJsonContent:
    def test_written_json_matches_returned_dict(self, tmp_path):
        shard_infos = [("shard_000000.tar", 10)]
        manifest = generate_manifest(shard_infos, tmp_path)
        with open(tmp_path / "manifest.json", "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded == manifest

    def test_empty_shard_infos(self, tmp_path):
        manifest = generate_manifest([], tmp_path)
        assert manifest["total_transitions"] == 0
        assert manifest["shard_count"] == 0
        assert manifest["shards"] == []
