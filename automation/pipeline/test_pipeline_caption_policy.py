"""N6: an inherited caption obeys the same `omit`/`include` policy as one built from the card.

The audit's `caption_omit` reproduced the defect: with `caption_traits=omit`, a caption inherited from tool2 kept
`black hair` and `beige dress` while a caption built from the card dropped them. The two paths then described the same
character differently, and which one a picture got depended on whether tool2 happened to caption it.

The contract these tests pin, in the order the spec states it:

* `omit` drops the card's fixed traits and the default outfit from an exported copy, and keeps everything else - the
  pose, the expression, the background, a non-default outfit;
* a tag list is edited tag-wise (no substring surgery: `black hairband` is not `black hair`);
* free prose is edited as whole phrases, and the preview says so with a note a person should read;
* `include` appends what is missing and never duplicates what is there;
* the image bytes never change; a policy change moves the caption hash and the snapshot id, and the previous export
  stays exactly as it was.
"""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
for entry in (str(AUTO.parent / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import export, seed  # noqa: E402


def card(**over):
    base = {"id": "fx", "trigger": "fixture", "caption_base": "webtoon style",
            "caption_traits": "omit",
            "fixed_traits": ["black hair"],
            "outfits": {"default": "beige dress"}}
    base.update(over)
    return base


class TagListTests(unittest.TestCase):
    """The tag path, which is what this pipeline's captions actually are."""

    def test_the_audit_fixture_loses_exactly_the_two_fixed_items(self):
        # spec N6's own completion condition, verbatim.
        out = seed.add_traits("fixture, black hair, beige dress, running, red umbrella", card())
        self.assertEqual(out, "fixture, running, red umbrella")

    def test_the_pose_expression_background_and_other_outfits_survive(self):
        out = seed.add_traits("fixture, black hair, beige dress, sitting, smiling, on a rooftop, navy coat", card())
        for kept in ("sitting", "smiling", "on a rooftop", "navy coat"):
            self.assertIn(kept, out)
        self.assertNotIn("black hair", out)
        self.assertNotIn("beige dress", out)

    def test_matching_is_case_insensitive_and_whole_tag_only(self):
        # Substring surgery would turn `black hairband` into `band`; a tag either is the term or it is not.
        out = seed.add_traits("Fixture, BLACK HAIR, black hairband, Beige Dress, running", card())
        self.assertEqual(out, "Fixture, black hairband, running")

    def test_a_compound_trait_is_matched_only_when_it_is_its_own_tag(self):
        card_ = card(fixed_traits=["long black hair"], outfits={"default": "white shirt"})
        out = seed.add_traits("fixture, long black hair, short black hair, white shirt, waving", card_)
        self.assertEqual(out, "fixture, short black hair, waving")

    def test_the_trigger_and_the_style_are_never_dropped_even_if_they_overlap_a_trait(self):
        # A card whose default outfit happens to be its own style line: omitting the outfit must not strip the style.
        card_ = card(caption_base="webtoon style", fixed_traits=["fixture"], outfits={"default": "webtoon style"})
        out = seed.add_traits("fixture, webtoon style, running", card_)
        self.assertIn("webtoon style", out)
        self.assertIn("fixture", out)
        self.assertIn("running", out)

    def test_a_caption_with_nothing_to_drop_is_unchanged(self):
        text = "fixture, running, red umbrella"
        self.assertEqual(seed.add_traits(text, card()), text)


class SentenceTests(unittest.TestCase):
    """Free prose: the edit is narrower and it says so."""

    def test_a_sentence_loses_the_phrases_and_keeps_its_shape(self):
        text = "A girl with long black hair wearing a beige dress, running through the rain."
        result = seed.caption_policy_preview(text, card(fixed_traits=["long black hair"]))
        self.assertEqual(result["kind"], "sentence")
        self.assertNotIn("long black hair", result["after"])
        self.assertNotIn("beige dress", result["after"])
        self.assertIn("running through the rain", result["after"])
        self.assertIn("free prose", result["note"], "the caller has to know this one needs reading")

    def test_a_phrase_inside_a_longer_word_is_left_alone(self):
        text = "She has blackish hair and a beardress-print scarf."
        result = seed.caption_policy_preview(text, card())
        self.assertEqual(result["after"], text)
        self.assertEqual(result["dropped"], [])

    def test_a_sentence_without_the_terms_is_unchanged_and_says_nothing(self):
        text = "A girl running through the rain."
        result = seed.caption_policy_preview(text, card())
        self.assertEqual(result["after"], text)
        self.assertEqual(result["note"], "")


class IncludeTests(unittest.TestCase):
    """`include` still appends, and never duplicates."""

    def test_missing_traits_are_appended(self):
        out = seed.add_traits("fixture, running", card(caption_traits="include"))
        self.assertEqual(out, "fixture, running, black hair, beige dress")

    def test_present_traits_are_not_duplicated(self):
        text = "fixture, Black Hair, running"
        out = seed.add_traits(text, card(caption_traits="include"))
        self.assertEqual(out, "fixture, Black Hair, running, beige dress")

    def test_a_caption_that_already_carries_everything_is_unchanged(self):
        text = "fixture, black hair, beige dress, running"
        self.assertEqual(seed.add_traits(text, card(caption_traits="include")), text)

    def test_an_empty_caption_stays_empty_and_is_reported_as_built(self):
        result = seed.caption_policy_preview("", card())
        self.assertEqual(result["after"], "")
        self.assertEqual(result["kind"], "built")


class ExportTests(unittest.TestCase):
    """What actually goes into a published export, and what a policy change does to the snapshot."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_project(self, *, traits="omit"):
        from pipeline import project as pj
        card_ = card(caption_traits=traits, holdout={"ids": []})
        project = pj.Project.create("fx", card_, folder=self.root / f"proj_{traits}")
        incoming = self.root / "incoming"
        incoming.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for index, colour in enumerate([(10, 20, 30), (40, 50, 60)]):
            Image.new("RGB", (8, 8), colour).save(incoming / f"src{index}.png")
        project.import_sources(sorted(incoming.iterdir()), log=lambda *_: None)
        for record in project.source_records():
            project.decide(record["id"], "accept")
        return project

    def set_caption(self, project, record_id, text):
        """Write a caption onto a candidate record the way a generator would have.

        `collect` reads `entry["caption"]` off the record it got from `audit`, so this rewrites the row in
        `candidates.jsonl`. The pipeline has no separate caption-setter API: a caption arrives with the generator's
        record, which is exactly the path under test here.
        """
        path = project.dir / "candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if row["id"] == record_id:
                row["caption"] = text
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        return project

    def test_an_inherited_caption_obeys_omit_in_the_export(self):
        project = self.make_project()
        records = project.source_records()
        # The picture's caption as the pipeline recorded it, carrying both fixed items. Sources are user originals, so
        # the caption has to come from a candidate: make one that points at the same bytes.
        self.add_candidate(project, records[0], "fixture, black hair, beige dress, running, red umbrella")
        entries, _report, _holdout, _caps = export.collect(project, log=lambda *_: None)
        entry = next(e for e in entries if e["kind"] == "candidate")
        self.assertEqual(entry["caption"], "fixture, running, red umbrella")
        self.assertEqual(entry["caption_source"], "the candidate's own caption",
                         "a caption the generator recorded is the one the export carries")
        self.assertEqual(entry["caption_policy"]["policy"], "omit")
        self.assertEqual(entry["caption_policy"]["dropped"], ["black hair", "beige dress"])
        self.assertEqual(entry["caption_policy"]["original_caption"],
                         "fixture, black hair, beige dress, running, red umbrella")
        self.assertEqual(entry["caption_policy"]["policy_version"], seed.CAPTION_POLICY_VERSION)

    def add_candidate(self, project, source_record, caption):
        """Register an accepted candidate that carries `caption`, drawn from `source_record`'s bytes' parentage."""
        from pipeline import lineage as lineage_mod
        from PIL import Image
        # The candidate's own bytes: a picture identical to a kept original is one picture, and the audit would refuse
        # it as a duplicate rather than let the export carry it twice.
        generated = Path(project.dir) / "generated" / f"{source_record['id']}_edit.png"
        generated.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (200, 30, 90)).save(generated)
        record = lineage_mod.record(generated, generator="qwen_edit",
                                    parent_ids=[source_record["id"]], train_generation=1,
                                    edit_depth=1, slot={"slot_id": "frm_front"}, prompt="p")
        record["caption"] = caption
        project.import_candidates([record], log=lambda *_: None)
        project.decide(record["id"], "accept")
        return record

    def test_include_and_omit_leave_the_image_bytes_alone(self):
        omit_project = self.make_project(traits="omit")
        include_project = self.make_project(traits="include")
        omit_entries, *_ = export.collect(omit_project, log=lambda *_: None)
        include_entries, *_ = export.collect(include_project, log=lambda *_: None)
        self.assertEqual({e["sha256"] for e in omit_entries}, {e["sha256"] for e in include_entries},
                         "a caption policy must never change which bytes are exported")
        self.assertNotEqual({e["caption_sha256"] for e in omit_entries},
                            {e["caption_sha256"] for e in include_entries})

    def test_each_entry_carries_the_original_text_and_its_hash(self):
        project = self.make_project()
        records = project.source_records()
        self.add_candidate(project, records[0], "fixture, black hair, walking")
        entries, *_ = export.collect(project, log=lambda *_: None)
        entry = next(e for e in entries if e["kind"] == "candidate")
        self.assertEqual(entry["caption"], "fixture, walking")
        self.assertEqual(entry["caption_sha256"], hashlib.sha256(entry["caption"].encode("utf-8")).hexdigest())
        self.assertEqual(entry["caption_policy"]["original_caption"], "fixture, black hair, walking")

    def test_the_caption_policy_version_is_in_the_snapshot_identity(self):
        # spec N6: a change to how the policy edits a caption makes a different snapshot, so the old export keeps its
        # own meaning instead of being silently re-read under the new rule.
        entries = [{"id": "c1", "lineage_ids": ["c1"], "sha256": "a" * 64, "caption": "fixture, running",
                    "kind": "candidate", "slot": "frm_front"}]
        caps = {"max_train_generation": 2, "seed_max_generation": 1}
        holdout = {"shas": []}

        class _P:
            card = {"id": "fx"}

            def digest(self):
                return "digest"

        first = seed.snapshot_identity(_P(), entries, caps=caps, holdout=holdout)
        original = seed.CAPTION_POLICY_VERSION
        seed.CAPTION_POLICY_VERSION = "caption-v3"
        try:
            second = seed.snapshot_identity(_P(), entries, caps=caps, holdout=holdout)
        finally:
            seed.CAPTION_POLICY_VERSION = original
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
