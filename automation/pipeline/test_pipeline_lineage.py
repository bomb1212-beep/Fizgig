"""Offline checks for provenance: quick hashes, the generation rules and the snapshot caps."""
import sys
import tempfile
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import lineage as lin  # noqa: E402


class FileHashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_a_small_file_is_hashed_in_full(self):
        path = self.dir / "small.safetensors"
        path.write_bytes(b"x" * 4096)
        entry = lin.quick_hash(path)
        self.assertEqual(entry["mode"], "full")
        self.assertEqual(entry["sha256"], lin.sha256_file(path))

    def test_a_large_file_is_hashed_at_both_ends(self):
        path = self.dir / "big.safetensors"
        size = lin.SMALL_FILE + 10
        path.write_bytes(b"a" * lin.HEAD_BYTES + b"m" * (size - 2 * lin.HEAD_BYTES) + b"z" * lin.HEAD_BYTES)
        entry = lin.quick_hash(path)
        self.assertEqual(entry["mode"], "edges")
        self.assertEqual(entry["size"], size)
        self.assertEqual(len(entry["head_sha256"]), 64)
        self.assertNotEqual(entry["head_sha256"], entry["tail_sha256"])

    def test_changing_the_middle_of_a_large_file_changes_its_key(self):
        path = self.dir / "big.safetensors"
        path.write_bytes(b"a" * (lin.SMALL_FILE + 10))
        before = lin.hash_key(lin.quick_hash(path))
        with open(path, "r+b") as handle:
            handle.seek(lin.HEAD_BYTES + 1)
            handle.write(b"zzzz")
        after = lin.hash_key(lin.quick_hash(path))
        self.assertNotEqual(before, after)

    def test_hash_key_accepts_a_bare_sha256(self):
        self.assertEqual(lin.hash_key("abc"), "abc")
        self.assertEqual(lin.hash_key({}), "")


class GenerationRuleTests(unittest.TestCase):
    def test_a_lora_output_is_one_generation_past_its_snapshot(self):
        loras = [{"trained_on_generation": 0}, {"trained_on_generation": 1}]
        self.assertEqual(lin.generation_from_loras(loras), 2)
        self.assertEqual(lin.generation_from_loras([]), 0)

    def test_an_edit_keeps_its_inputs_generation(self):
        self.assertEqual(lin.generation_from_edit(1), 1)
        self.assertEqual(lin.generation_from_edit(2), 2)

    def test_a_cap_refuses_a_later_generation(self):
        record = {"file": "x.png", "train_generation": 3}
        with self.assertRaises(lin.LineageError):
            lin.check_generation(record, limit=2)
        self.assertTrue(lin.check_generation({"train_generation": 2}, limit=2))

    def test_filter_for_snapshot_splits_kept_and_refused(self):
        records = [{"id": "a", "file": "a.png", "train_generation": 1, "edit_depth": 0},
                   {"id": "b", "file": "b.png", "train_generation": 3, "edit_depth": 0},
                   {"id": "c", "file": "c.png", "train_generation": 1, "edit_depth": 4}]
        kept, refused = lin.filter_for_snapshot(records, {"max_train_generation": 2, "max_edit_depth": 1})
        self.assertEqual([r["id"] for r in kept], ["a"])
        self.assertEqual({r["id"] for r in refused}, {"b", "c"})
        self.assertTrue(all("cap" in r["reason"] or "depth" in r["reason"] for r in refused))


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _image(self, name="a.png", data=b"PNG"):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_a_record_carries_the_digest_of_its_bytes(self):
        path = self._image()
        record = lin.record(path, generator="krea2_lora", parent_ids=["s_1"], train_generation=1,
                            graph_version="krea2_t2i_turbo@1", seed=1003)
        self.assertEqual(record["sha256"], lin.sha256_file(path))
        self.assertEqual(record["id"], lin.image_id(record["sha256"]))
        self.assertEqual(record["graph_version"], "krea2_t2i_turbo@1")

    def test_a_generated_image_without_parents_is_refused(self):
        path = self._image()
        with self.assertRaises(lin.LineageError):
            lin.record(path, generator="krea2_lora", parent_ids=[], train_generation=1)

    def test_an_unknown_generator_is_refused(self):
        path = self._image()
        with self.assertRaises(lin.LineageError):
            lin.record(path, generator="photoshop")

    def test_a_source_record_marks_origin(self):
        path = self._image()
        record = lin.source_record(path, origin="synthetic", parent_ids=["c_1"], note="tool2 anchor")
        self.assertEqual(record["origin"], "synthetic")
        self.assertEqual(record["id"], lin.source_id(record["sha256"]))
        with self.assertRaises(lin.LineageError):
            lin.source_record(path, origin="unknown")


class JsonlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_appending_the_same_record_twice_writes_it_once(self):
        path = self.dir / "candidates.jsonl"
        record = {"id": "c_1", "file": "a.png"}
        self.assertEqual(lin.append_jsonl(path, [record]), 1)
        self.assertEqual(lin.append_jsonl(path, [record]), 0)
        self.assertEqual(len(lin.read_jsonl(path)), 1)

    def test_reading_skips_a_broken_line(self):
        path = self.dir / "candidates.jsonl"
        path.write_text('{"id": "c_1"}\nnot json\n', encoding="utf-8")
        self.assertEqual(len(lin.read_jsonl(path)), 1)

    def test_write_jsonl_replaces_the_file(self):
        path = self.dir / "candidates.jsonl"
        lin.write_jsonl(path, [{"id": "c_1"}, {"id": "c_2"}])
        lin.write_jsonl(path, [{"id": "c_3"}])
        self.assertEqual([r["id"] for r in lin.read_jsonl(path)], ["c_3"])

    def test_trace_reaches_the_origin_and_reports_a_missing_parent(self):
        index = {"c_2": {"id": "c_2", "parent_ids": ["c_1"]},
                 "c_1": {"id": "c_1", "parent_ids": ["s_1"]}}
        seen, unknown = lin.trace_to_origin({"id": "c_3", "parent_ids": ["c_2", "s_9"]}, index)
        self.assertEqual(set(seen), {"c_2", "s_9", "c_1", "s_1"})
        # s_9 is referenced but nothing in the index describes it; s_1 is reachable only through c_1, which the index
        # does describe, so both count as unknown ids to the caller.
        self.assertEqual(set(unknown), {"s_9", "s_1"})


if __name__ == "__main__":
    unittest.main()
