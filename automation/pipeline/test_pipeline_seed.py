"""Tests for the project folder, the review page and the seed snapshot (spec 3.1, 4 M3-1/M4 core, tasks T1-T3)."""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
for entry in (str(ROOT / "src"), str(AUTO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from pipeline import lineage, project as pj, review, seed  # noqa: E402

CARD = {"trigger": "trg", "kind": "anime", "caption_base": "korean webtoon style",
        "caption_traits": "omit", "lineage": {"max_train_generation": 2, "seed_max_generation": 1,
                                              "max_edit_depth": 1},
        "holdout": {"min_sources": 3, "count": 1, "pick": []},
        "slot_quotas": {"ang_front_close": 2, "pos_sitting": 1}}


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.chars = self.root / "chars"
        self.chars.mkdir()
        self.project = pj.Project.create("demo", dict(CARD), folder=self.root / "demo")

    def tearDown(self):
        self.temp.cleanup()

    def original(self, name, data=None):
        path = self.chars / name
        path.write_bytes(data or ("ORIGINAL-" + name).encode())
        return path

    def import_originals(self, count):
        """Import `count` originals and return {file name: source id}."""
        paths = [self.original(f"o{i}.png") for i in range(count)]
        self.project.import_sources(paths, log=lambda *_: None)
        return {Path(r["src_path"]).name: r["id"] for r in self.project.source_records()}

    def candidate(self, name, *, generator="qwen_edit", generation=1, depth=1, caption="trg, front view",
                  parents=("s_1",), slot=None, models=None):
        path = self.root / name
        path.write_bytes(("IMAGE-" + name).encode())
        if not parents and not models:
            # a generated image must point somewhere: a LoRA snapshot counts, an empty parent list does not
            models = {"loras": [{"name": "seed.safetensors", "trained_on_snapshot": "seed@1",
                                 "trained_on_generation": 1, "sha256": "x", "strength": 1.0}]}
        record = lineage.record(path, generator=generator, parent_ids=list(parents), train_generation=generation,
                                edit_depth=depth, caption=caption, slot=slot or {"category": "angles",
                                                                                 "slot_id": "ang_front_close"},
                                models=models, seed=1003)
        record["caption"] = caption
        lineage.append_jsonl(self.project.dir / "candidates.jsonl", [record])
        return self.project.candidate_by_id(record["id"])

    def patch_candidate(self, candidate_id, **changes):
        """Rewrite one candidate record in place (parent ids, models, ...), for the lineage cases."""
        records = [dict(r) for r in self.project.candidates()]
        for record in records:
            if record["id"] == candidate_id:
                record.update(changes)
        lineage.write_jsonl(self.project.dir / "candidates.jsonl", records)
        return self.project.candidate_by_id(candidate_id)

    def codes(self, report):
        return {row["id"]: row["code"] for row in report["excluded"]}


class ProjectTests(Base):
    def test_a_project_is_created_with_its_files_and_defaults(self):
        self.assertTrue((self.project.dir / "project.json").is_file())
        self.assertTrue((self.project.dir / "review.json").is_file())
        for folder in ("sources", "candidates", "exports"):
            self.assertTrue((self.project.dir / folder).is_dir())
        self.assertEqual(self.project.card["id"], "demo")
        self.assertEqual(self.project.card["lineage"]["max_train_generation"], 2)

    def test_a_second_project_in_the_same_folder_is_refused(self):
        with self.assertRaises(pj.ProjectError):
            pj.Project.create("demo", dict(CARD), folder=self.project.dir)

    def test_opening_a_folder_without_a_project_is_refused(self):
        with self.assertRaises(pj.ProjectError):
            pj.Project.open(self.root / "nothing")

    def test_sources_are_named_by_their_content_and_not_copied_twice(self):
        first = self.original("a.png")
        added, reused = self.project.import_sources([first], log=lambda *_: None)
        self.assertEqual((added, reused), (1, 0))
        again = self.root / "same-content-different-name.png"
        again.write_bytes(first.read_bytes())
        added, reused = self.project.import_sources([again], log=lambda *_: None)
        self.assertEqual((added, reused), (0, 1))
        records = self.project.source_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(Path(records[0]["file"]).name, records[0]["sha256"][:16] + ".png")

    def test_a_synthetic_input_is_marked_and_the_reason_recorded(self):
        listed = self.original("anchor_x.png")
        by_prefix = self.original("anchor_refined_fullbody_a_s1.png")
        plain = self.original("front.png")
        self.project.import_sources([listed, by_prefix, plain],
                                    origin_by_name={"anchor_x.png": "synthetic"}, log=lambda *_: None)
        by_name = {Path(r["src_path"]).name: r for r in self.project.source_records()}
        self.assertEqual(by_name["anchor_x.png"]["origin"], "synthetic")
        self.assertIn("caller", by_name["anchor_x.png"]["decided_by"])
        self.assertEqual(by_name["anchor_refined_fullbody_a_s1.png"]["origin"], "synthetic")
        self.assertIn("prefix", by_name["anchor_refined_fullbody_a_s1.png"]["decided_by"])
        self.assertEqual(by_name["front.png"]["origin"], "user")
        share, counts = self.project.source_share()
        self.assertEqual(counts, {"user": 1, "synthetic": 2})
        self.assertAlmostEqual(share, 2 / 3)

    def test_the_originals_are_copied_not_moved(self):
        path = self.original("front.png")
        before = path.read_bytes()
        self.project.import_sources([path], log=lambda *_: None)
        self.assertEqual(path.read_bytes(), before)

    def test_holdout_needs_enough_user_originals(self):
        self.project.import_sources([self.original(f"o{i}.png") for i in range(2)], log=lambda *_: None)
        ids, reason = self.project.holdout_ids()
        self.assertEqual(ids, [])
        self.assertIn("below holdout.min_sources", reason)

    def test_holdout_takes_a_front_view_and_picks_named_files(self):
        names = ["side.png", "front.png", "back.png"]
        self.project.import_sources([self.original(n) for n in names], log=lambda *_: None)
        ids, reason = self.project.holdout_ids()
        by_name = {Path(r["src_path"]).name: r["id"] for r in self.project.source_records()}
        self.assertEqual(ids, [by_name["front.png"]])
        self.assertIn("front view first", reason)
        self.project.card["holdout"] = {"min_sources": 3, "count": 2, "pick": ["side.png", "back.png"]}
        ids, reason = self.project.holdout_ids()
        self.assertEqual(set(ids), {by_name["side.png"], by_name["back.png"]})
        self.assertIn("named by the card", reason)

    def test_a_frozen_holdout_keeps_its_ids_and_files(self):
        by_name = self.import_originals(4)
        entry = self.project.freeze_holdout(log=lambda *_: None)
        self.assertTrue(entry["frozen"])
        self.assertEqual(entry["ids"], [by_name["o0.png"]])
        self.assertEqual(entry["files"], ["o0.png"])
        self.assertEqual(len(entry["shas"]), 1)
        # importing more originals does not move the frozen set
        self.project.import_sources([self.original("late.png")], log=lambda *_: None)
        again = self.project.holdout_entry()
        self.assertTrue(again["frozen"])
        self.assertEqual(again["ids"], [by_name["o0.png"]])
        self.assertIn("frozen", again["reason"])

    def test_a_decision_is_recorded_and_can_be_taken_back(self):
        record = self.candidate("a.png")
        self.project.decide(record["id"], "accept", note="good face")
        self.assertEqual(self.project.review()[record["id"]]["decision"], "accept")
        self.assertEqual(self.project.review()[record["id"]]["note"], "good face")
        self.assertEqual(self.project.review()[record["id"]]["origin"], "user")
        self.project.undecide(record["id"])
        self.assertEqual(self.project.review(), {})

    def test_a_null_decision_is_the_same_as_taking_it_back(self):
        record = self.candidate("a.png")
        self.project.decide(record["id"], "accept")
        self.project.decide(record["id"], None)
        self.assertEqual(self.project.review(), {})

    def test_an_unknown_decision_is_refused(self):
        record = self.candidate("a.png")
        with self.assertRaises(pj.ProjectError):
            self.project.decide(record["id"], "maybe")

    def test_an_unknown_reason_is_refused(self):
        record = self.candidate("a.png")
        with self.assertRaises(pj.ProjectError):
            self.project.decide(record["id"], "reject", reason="i-dont-like-it")

    def test_a_decision_from_the_m0_sheet_is_marked_as_such(self):
        record = self.candidate("a.png")
        self.project.decide(record["id"], "accept", by="user (M0 scores)")
        self.assertEqual(self.project.state_of(record["id"])["origin"], pj.ORIGIN_M0)

    def test_accepted_can_be_capped_by_generation(self):
        first = self.candidate("a.png", generation=1)
        second = self.candidate("b.png", generator="krea2_lora", generation=2, parents=())
        for record in (first, second):
            self.project.decide(record["id"], "accept")
        self.assertEqual([r["id"] for r in self.project.accepted()], [first["id"], second["id"]])
        self.assertEqual([r["id"] for r in self.project.accepted(limit_generation=1)], [first["id"]])

    def test_undecided_and_by_decision_partition_the_candidates(self):
        first = self.candidate("a.png")
        second = self.candidate("b.png")
        self.project.decide(first["id"], "reject")
        self.assertEqual([r["id"] for r in self.project.by_decision("reject")], [first["id"]])
        self.assertEqual([r["id"] for r in self.project.undecided()], [second["id"]])

    def test_quotas_count_accepted_candidates_per_slot(self):
        first = self.candidate("a.png")
        second = self.candidate("b.png")
        self.project.decide(first["id"], "accept")
        quotas = self.project.quotas(self.project.card["slot_quotas"])
        self.assertEqual(quotas["ang_front_close"], {"target": 2, "accepted": 1, "short": 1})
        self.assertEqual(quotas["pos_sitting"]["short"], 1)
        self.project.decide(second["id"], "accept")
        self.assertEqual(self.project.quotas(self.project.card["slot_quotas"])["ang_front_close"]["accepted"], 2)

    def test_a_candidate_traces_to_its_original(self):
        source_path = self.original("front.png")
        self.project.import_sources([source_path], log=lambda *_: None)
        source = self.project.source_records()[0]
        candidate = self.candidate("edited.png", parents=(source["id"],))
        visited, unknown = self.project.trace(candidate["id"])
        self.assertEqual(visited, [source["id"]])
        self.assertEqual(unknown, [])

    def test_candidates_are_copied_in_under_their_content_hash(self):
        self.original("front.png")
        self.project.import_sources([self.chars / "front.png"], log=lambda *_: None)
        made = self.root / "elsewhere" / "edit.png"
        made.parent.mkdir()
        made.write_bytes(b"EDIT")
        record = lineage.record(made, generator="qwen_edit", parent_ids=["s_1"], train_generation=1, edit_depth=1,
                                slot={"category": "angles"}, caption="trg, smile")
        num, skipped = self.project.import_candidates([record], log=lambda *_: None)
        self.assertEqual((num, skipped), (1, []))
        stored = Path(self.project.candidates()[0]["file"])
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.read_bytes(), b"EDIT")
        self.assertEqual(stored.parent.parent.name, "qwen_edit")
        self.assertEqual(stored.stem, record["sha256"][:16])
        # the original the generator worked from is untouched
        self.assertEqual((self.chars / "front.png").read_bytes(), b"ORIGINAL-front.png")

    def test_importing_the_same_candidate_twice_adds_nothing(self):
        made = self.root / "edit.png"
        made.write_bytes(b"EDIT")
        record = lineage.record(made, generator="qwen_edit", parent_ids=["s_1"], train_generation=1, edit_depth=1)
        self.project.import_candidates([record], log=lambda *_: None)
        num, skipped = self.project.import_candidates([record], log=lambda *_: None)
        self.assertEqual((num, len(skipped)), (0, 1))
        self.assertEqual(len(self.project.candidates()), 1)

    def test_a_record_whose_file_is_gone_is_reported(self):
        record = {"id": "c_dead", "file": str(self.root / "nope.png"), "generator": "qwen_edit"}
        num, skipped = self.project.import_candidates([record], log=lambda *_: None)
        self.assertEqual(num, 0)
        self.assertIn("not found", skipped[0]["reason"])

    def test_a_record_written_relative_to_the_automation_root_still_resolves(self):
        # early runs recorded `automation/projects/<id>/...` instead of a project-relative path
        stored = self.project.dir / "candidates" / "qwen_edit" / "manual"
        stored.mkdir(parents=True)
        (stored / "abc.png").write_bytes(b"OLD")
        record = {"id": "c_relative", "file": "automation/projects/demo/candidates/qwen_edit/manual/abc.png",
                  "generator": "qwen_edit"}
        self.assertEqual(self.project.resolve_file(record["file"]).name, "abc.png")

    def test_the_lock_refuses_a_second_owner_and_only_its_owner_releases_it(self):
        with self.project.lock():
            lock_path = self.project.dir / "pipeline.lock"
            self.assertTrue(lock_path.is_file())
            owner = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(owner["pid"], __import__("os").getpid())
            self.assertEqual(owner["thread"], threading.get_ident())
            self.assertTrue(owner["token"])
            # a lock file written by someone else (another live process) is refused
            lock_path.write_text(json.dumps({"pid": 4, "at": 1, "what": "other"}), encoding="utf-8")
            with self.assertRaises(pj.ProjectError):
                with self.project.lock():
                    pass
            # ... and the release does not delete a record that is no longer ours
            self.assertTrue(lock_path.is_file())

    def test_a_second_thread_cannot_take_the_lock(self):
        failures = []

        def other():
            try:
                with self.project.lock():
                    failures.append("took it")
            except pj.ProjectError:
                failures.append("refused")

        with self.project.lock():
            thread = threading.Thread(target=other)
            thread.start()
            thread.join(timeout=10)
        self.assertEqual(failures, ["refused"])

    def test_a_stale_lock_is_reclaimed(self):
        (self.project.dir / "pipeline.lock").write_text(json.dumps({"pid": 999999999, "at": 1}), encoding="utf-8")
        with self.project.lock() as lock:
            self.assertTrue(lock.acquired)
        self.assertFalse((self.project.dir / "pipeline.lock").exists())

    def test_concurrent_decisions_from_threads_are_all_kept(self):
        records = [self.candidate(f"c{i}.png") for i in range(12)]
        errors = []

        def decide(index, record):
            try:
                self.project.decide(record["id"], "accept" if index % 2 == 0 else "reject")
            except Exception as exc:                      # noqa: BLE001 - the test reports whatever happened
                errors.append(exc)

        threads = [threading.Thread(target=decide, args=(i, r)) for i, r in enumerate(records)]
        [t.start() for t in threads]
        [t.join(timeout=30) for t in threads]
        self.assertEqual(errors, [])
        review = self.project.review()
        self.assertEqual(len(review), len(records))
        self.assertEqual(sum(1 for state in review.values() if state["decision"] == "accept"), 6)

    def test_the_card_digest_changes_with_the_card(self):
        before = self.project.digest()
        self.project.card["trigger"] = "other"
        self.assertNotEqual(before, self.project.digest())

    def test_the_review_migration_moves_the_stated_reason_and_tags_the_origin(self):
        first = self.candidate("a.png")
        second = self.candidate("b.png")
        pj.write_json(self.project.dir / "review.json", {
            first["id"]: {"decision": "reject", "by": "user (M0 scores)", "note": "M0 정체성 - · 다른 캐릭터"},
            second["id"]: {"decision": "accept", "by": "user (M0 scores)", "note": "M0 정체성 5"}})
        plan = self.project.review_plan()
        self.assertEqual(plan["counts"]["changed"], 2)
        self.assertEqual(plan["counts"]["different_character"], 1)
        # reporting does not write anything
        self.assertNotIn("reason", self.project.review()[first["id"]])
        self.project.migrate_review(log=lambda *_: None)
        review = self.project.review()
        self.assertEqual(review[first["id"]]["reason"], "different_character")
        self.assertEqual(review[first["id"]]["origin"], pj.ORIGIN_M0)
        self.assertNotIn("reason", review[second["id"]])


class ReviewTests(Base):
    def test_the_page_has_a_card_and_four_buttons_per_candidate(self):
        self.candidate("a.png")
        self.candidate("b.png")
        page = review.write_page(self.project, self.root / "review.html", log=lambda *_: None)
        text = page.read_text(encoding="utf-8")
        self.assertEqual(text.count('class="card"'), 2)
        # 채택 / 기각 / 보정 / 다른 캐릭터
        self.assertEqual(text.count('class="dec'), 8)
        self.assertIn("quota", text)
        self.assertIn("A=채택", text)

    def test_rows_can_be_filtered(self):
        first = self.candidate("a.png")
        self.candidate("b.png", generator="krea2_lora", slot={"category": "poses", "slot_id": "pos_sitting"})
        self.project.decide(first["id"], "accept")
        self.assertEqual(len(review.candidate_rows(self.project)), 2)
        self.assertEqual([r["id"] for r in review.candidate_rows(self.project, decision="accept")], [first["id"]])
        self.assertEqual(len(review.candidate_rows(self.project, generator="krea2_lora")), 1)
        self.assertEqual(len(review.candidate_rows(self.project, slot="pos_sitting")), 1)

    def test_the_quota_board_marks_what_is_short(self):
        record = self.candidate("a.png")
        self.project.decide(record["id"], "accept")
        text = review.write_page(self.project, self.root / "review.html", log=lambda *_: None).read_text("utf-8")
        self.assertIn("채택", text)
        self.assertIn("ang_front_close", text)

    def test_a_different_character_row_cannot_be_corrected(self):
        record = self.candidate("a.png")
        self.project.decide(record["id"], "reject", reason="different_character")
        row = review.candidate_rows(self.project)[0]
        self.assertFalse(row["fixable"])
        self.assertEqual(row["reason"], "different_character")
        text = review.write_page(self.project, self.root / "review.html", log=lambda *_: None).read_text("utf-8")
        self.assertIn("보정 불가", text)

    def test_the_page_urls_carry_the_files_own_suffix(self):
        made = self.root / "edit.jpg"
        made.write_bytes(b"JPEGDATA")
        record = lineage.record(made, generator="qwen_edit", parent_ids=["s_1"], train_generation=1, edit_depth=1)
        self.project.import_candidates([record], log=lambda *_: None)
        row = review.candidate_rows(self.project)[0]
        self.assertEqual(review.image_url(row), f"/image/{row['id']}.jpg")


class ReviewServerTests(Base):
    def setUp(self):
        super().setUp()
        self.candidate("a.png")
        self.server = review.make_server(self.project, port=0, log=lambda *_: None,
                                         quota_plan=self.project.card["slot_quotas"])
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        time.sleep(0.3)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def get(self, path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as response:
                return response.status, response.headers.get("Content-Type", ""), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers.get("Content-Type", ""), exc.read()

    def post(self, payload):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/decide", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_it_serves_the_page_and_the_images(self):
        status, ctype, _body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        candidate = self.project.candidates()[0]
        self.assertEqual(self.get(f"/image/{candidate['id']}")[0], 200)
        self.assertEqual(self.get("/image/nope")[0], 404)

    def test_every_image_url_the_page_shows_is_delivered(self):
        import re
        page = self.get("/")[2].decode("utf-8")
        urls = sorted(set(re.findall(r'(?:src|href)="(/image/[^"]+)"', page)))
        self.assertTrue(urls)
        for url in urls:
            status, ctype, body = self.get(url)
            self.assertEqual(status, 200, url)
            self.assertEqual(ctype, "image/png", url)
            self.assertTrue(body)

    def test_a_decision_is_posted_into_review_json(self):
        candidate = self.project.candidates()[0]
        status, body = self.post({"id": candidate["id"], "decision": "accept"})
        self.assertEqual((status, body["decision"]), (200, "accept"))
        self.assertEqual(self.project.review()[candidate["id"]]["decision"], "accept")

    def test_a_null_decision_clears_it_in_one_request(self):
        candidate = self.project.candidates()[0]
        self.post({"id": candidate["id"], "decision": "fix"})
        status, body = self.post({"id": candidate["id"], "decision": None})
        self.assertEqual((status, body["decision"]), (200, None))
        self.assertEqual(self.project.review(), {})

    def test_an_unknown_decision_is_a_client_error(self):
        candidate = self.project.candidates()[0]
        status, body = self.post({"id": candidate["id"], "decision": "maybe"})
        self.assertEqual(status, 400)
        self.assertIn("decision", body["error"])

    def test_a_decision_about_an_unknown_id_is_refused(self):
        status, body = self.post({"id": "c_doesnotexist", "decision": "accept"})
        self.assertEqual(status, 400)
        self.assertIn("존재하지", body["error"])
        self.assertNotIn("c_doesnotexist", self.project.review())

    def test_a_reasoned_rejection_is_stored_and_blocks_accept_and_fix(self):
        candidate = self.project.candidates()[0]
        status, body = self.post({"id": candidate["id"], "decision": "reject", "reason": "different_character"})
        self.assertEqual((status, body["reason"]), (200, "different_character"))
        self.assertEqual(self.project.review()[candidate["id"]]["reason"], "different_character")
        self.assertFalse(body["fixable"])
        status, body = self.post({"id": candidate["id"], "decision": "accept"})
        self.assertEqual(status, 400)
        self.assertIn("보정할 수 없습니다", body["error"])
        status, _body = self.post({"id": candidate["id"], "decision": "fix"})
        self.assertEqual(status, 400)
        # rejecting it again (or clearing it) is still allowed
        self.assertEqual(self.post({"id": candidate["id"], "decision": "reject"})[0], 200)

    def test_a_different_character_cannot_be_marked_for_correction(self):
        candidate = self.project.candidates()[0]
        self.project.decide(candidate["id"], "reject", reason=review.FIX_BLOCKED)
        status, body = self.post({"id": candidate["id"], "decision": "fix"})
        self.assertEqual(status, 400)
        self.assertIn("보정할 수 없습니다", body["error"])

    def test_two_concurrent_posts_on_two_cards_are_both_kept(self):
        second = self.candidate("b.png")
        first = self.project.candidates()[0]
        results = []

        def post_one(candidate_id, decision):
            results.append(self.post({"id": candidate_id, "decision": decision}))

        threads = [threading.Thread(target=post_one, args=(first["id"], "accept")),
                   threading.Thread(target=post_one, args=(second["id"], "reject"))]
        [t.start() for t in threads]
        [t.join(timeout=30) for t in threads]
        self.assertEqual(sorted(status for status, _body in results), [200, 200])
        review = self.project.review()
        self.assertEqual(review[first["id"]]["decision"], "accept")
        self.assertEqual(review[second["id"]]["decision"], "reject")


class SeedEligibilityTests(Base):
    """T1: what may and may not be a seed input, and why."""

    def test_an_edit_of_a_holdout_original_is_refused(self):
        by_name = self.import_originals(4)
        holdout, _why = self.project.holdout_ids()
        edit = self.candidate("bad.png", parents=(holdout[0],))
        self.project.decide(edit["id"], "accept")
        report = seed.audit(self.project)
        self.assertEqual(self.codes(report)[edit["id"]], "holdout_direct")
        self.assertNotIn(edit["id"], {entry["id"] for entry in report["kept"]})

    def test_an_edit_of_an_edit_of_a_holdout_is_refused_too(self):
        by_name = self.import_originals(4)
        holdout, _why = self.project.holdout_ids()
        first = self.candidate("first.png", parents=(holdout[0],))
        second = self.candidate("second.png", parents=(first["id"],))
        for record in (first, second):
            self.project.decide(record["id"], "accept")
        codes = self.codes(seed.audit(self.project))
        self.assertEqual(codes[first["id"]], "holdout_direct")
        self.assertEqual(codes[second["id"]], "holdout_indirect")

    def test_a_synthetic_anchor_with_no_recorded_parent_is_quarantined_with_its_children(self):
        anchor = self.original("anchor_refined_fullbody_x_s1.png")
        self.project.import_sources([anchor] + [self.original(f"o{i}.png") for i in range(3)],
                                    log=lambda *_: None)
        anchor_id = {Path(r["src_path"]).name: r["id"]
                     for r in self.project.source_records()}["anchor_refined_fullbody_x_s1.png"]
        edit = self.candidate("from-anchor.png", parents=(anchor_id,))
        self.project.decide(edit["id"], "accept")
        report = seed.audit(self.project)
        codes = self.codes(report)
        self.assertEqual(codes[anchor_id], "synthetic_source")
        self.assertIn("no recorded parent", [row for row in report["excluded"]
                                             if row["id"] == anchor_id][0]["reason"])
        self.assertEqual(codes[edit["id"]], "unverified_ancestor")
        self.assertEqual(report["quarantined"], [anchor_id])

    def test_only_accepted_and_reviewed_edits_enter(self):
        self.import_originals(4)
        source = self.project.source_records()[1]["id"]
        undecided = self.candidate("undecided.png", parents=(source,))
        rejected = self.candidate("rejected.png", parents=(source,))
        fixing = self.candidate("fixing.png", parents=(source,))
        other = self.candidate("other.png", parents=(source,))
        self.project.decide(rejected["id"], "reject")
        self.project.decide(fixing["id"], "fix")
        self.project.decide(other["id"], "reject", reason="different_character")
        codes = self.codes(seed.audit(self.project))
        self.assertEqual(codes[undecided["id"]], "not_reviewed")
        self.assertEqual(codes[rejected["id"]], "rejected")
        self.assertEqual(codes[fixing["id"]], "marked_for_fix")
        self.assertEqual(codes[other["id"]], "different_character")

    def test_a_candidate_from_another_generator_is_not_a_seed_input(self):
        self.import_originals(4)
        made = self.candidate("lora.png", generator="krea2_lora", parents=())
        self.project.decide(made["id"], "accept")
        self.assertEqual(self.codes(seed.audit(self.project))[made["id"]], "generator_not_eligible")

    def test_a_missing_parent_a_cycle_and_a_deep_generation_are_each_named(self):
        by_name = self.import_originals(4)
        unknown = self.candidate("unknown-parent.png", parents=("s_deadbeefdeadbeef",))
        deep = self.candidate("deep.png", parents=(by_name["o1.png"],), generation=2)
        loop_a = self.candidate("loop-a.png", parents=(by_name["o1.png"],))
        loop_b = self.candidate("loop-b.png", parents=(loop_a["id"],))
        self.patch_candidate(loop_a["id"], parent_ids=[loop_b["id"]])
        for record in (unknown, deep, loop_a, loop_b):
            self.project.decide(record["id"], "accept")
        codes = self.codes(seed.audit(self.project))
        self.assertEqual(codes[unknown["id"]], "missing_parent")
        self.assertEqual(codes[deep["id"]], "generation_cap")
        self.assertIn(codes[loop_b["id"]], ("cyclic_lineage",))
        self.assertIn(codes[loop_a["id"]], ("cyclic_lineage",))

    def test_a_lora_trained_on_an_unrecorded_snapshot_is_not_trusted(self):
        self.import_originals(4)
        edit = self.candidate("lora-made.png", parents=(), models={"loras": [
            {"name": "seed.safetensors", "trained_on_snapshot": "nosuchsnap", "trained_on_generation": 1}]})
        self.project.decide(edit["id"], "accept")
        self.assertEqual(self.codes(seed.audit(self.project))[edit["id"]], "lora_snapshot_unverified")

    def test_a_candidate_whose_bytes_changed_is_refused(self):
        self.import_originals(4)
        source = self.project.source_records()[1]["id"]
        edit = self.candidate("changed.png", parents=(source,))
        self.project.decide(edit["id"], "accept")
        Path(self.project.candidate_by_id(edit["id"])["file"]).write_bytes(b"SOMETHING ELSE")
        self.assertEqual(self.codes(seed.audit(self.project))[edit["id"]], "tampered_file")

    def test_the_audit_tells_accepted_from_accepted_by_the_m0_sheet(self):
        self.import_originals(4)
        source = self.project.source_records()[1]["id"]
        m0 = self.candidate("m0.png", parents=(source,))
        fresh = self.candidate("fresh.png", parents=(source,))
        self.project.decide(m0["id"], "accept", by="user (M0 scores)")
        self.project.decide(fresh["id"], "accept")
        counts = seed.audit(self.project)["counts"]
        self.assertEqual((counts["accepted"], counts["accepted_from_m0"], counts["accepted_by_review"]), (2, 1, 1))
        origins = {entry["id"]: entry["decision_origin"] for entry in seed.audit(self.project)["kept"]}
        self.assertEqual(origins[m0["id"]], pj.ORIGIN_M0)
        self.assertEqual(origins[fresh["id"]], pj.ORIGIN_USER)

    def test_an_empty_eligible_set_says_which_reasons_emptied_it(self):
        self.import_originals(4)
        holdout, _why = self.project.holdout_ids()
        edit = self.candidate("bad.png", parents=(holdout[0],))
        self.project.decide(edit["id"], "accept")
        # drop every source record: nothing user-supplied is left to train on
        (self.project.dir / "sources.jsonl").write_text("", encoding="utf-8")
        with self.assertRaises(pj.ProjectError) as caught:
            seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertIn("missing_parent", str(caught.exception))


class SeedSnapshotTests(Base):
    """T2: the snapshot's identity, integrity and behaviour on a rerun."""

    def _seed(self, originals=4):
        by_name = self.import_originals(originals)
        return by_name

    def _accepted_edit(self, name="edit1.png", generation=1, caption="trg, korean webtoon style, smile",
                       parent=None):
        parent = parent or self.project.source_records()[1]["id"]
        record = self.candidate(name, generation=generation, caption=caption, parents=(parent,))
        self.project.decide(record["id"], "accept")
        return record

    def test_the_seed_set_is_the_originals_plus_accepted_edits(self):
        self._seed()
        edit = self._accepted_edit()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        ids = {e["id"] for e in snapshot["inputs"]}
        self.assertIn(edit["id"], ids)
        # 4 originals, one of them held out, plus the accepted edit
        self.assertEqual(snapshot["counts"]["total"], 4)
        self.assertEqual(snapshot["counts"]["synthetic"], 1)
        self.assertEqual(snapshot["counts"]["user"], 3)

    def test_an_edit_past_the_seed_cap_is_refused_not_silently_dropped(self):
        self._seed()
        self._accepted_edit("deep.png", generation=2)
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        # the holdout original and the too-deep edit; each for its own recorded reason
        self.assertEqual(snapshot["excluded_counts"]["generation_cap"], 1)
        self.assertEqual(snapshot["excluded_counts"]["holdout_source"], 1)
        generation_row = [row for row in snapshot["excluded"] if row["code"] == "generation_cap"][0]
        self.assertIn("seed_max_generation", generation_row["reason"])
        self.assertNotIn(2, [e["generation"] for e in snapshot["inputs"]])

    def test_the_holdout_never_enters_the_seed_set(self):
        self._seed()
        holdout, _reason = self.project.holdout_ids()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertNotIn(holdout[0], {e["id"] for e in snapshot["inputs"]})
        self.assertEqual(snapshot["holdout_ids"], sorted(holdout))

    def test_captions_are_written_beside_every_image(self):
        self._seed()
        self._accepted_edit()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        dataset = Path(snapshot["dataset"])
        images = sorted(p for p in dataset.iterdir() if p.suffix == ".png")
        self.assertEqual(len(images), snapshot["counts"]["total"])
        for image in images:
            caption = image.with_suffix(".txt").read_text(encoding="utf-8").strip()
            self.assertTrue(caption)
            self.assertTrue(caption.startswith("trg"), caption)

    def test_a_tool2_caption_for_the_same_bytes_is_preferred(self):
        source = self.original("front.png")
        self.project.import_sources([source], log=lambda *_: None)
        tool2_dir = self.root / "tool2dataset"
        tool2_dir.mkdir()
        (tool2_dir / "originals_front_abc123.png").write_bytes(source.read_bytes())
        (tool2_dir / "originals_front_abc123.txt").write_text("trg, korean webtoon style, upper body\n",
                                                             encoding="utf-8")
        self.project.card["tool2"] = {"dataset": str(tool2_dir)}
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertIn("tool2", snapshot["caption_sources"][snapshot["inputs"][0]["id"]])
        caption = (Path(snapshot["dataset"]) / snapshot["inputs"][0]["file"]).with_suffix(".txt").read_text("utf-8")
        self.assertIn("upper body", caption)

    def test_the_include_policy_appends_the_fixed_traits(self):
        self.project.card["caption_traits"] = "include"
        self.project.card["fixed_traits"] = ["long dark hair"]
        self.project.card["outfits"] = {"default": "white blouse"}
        self.project.import_sources([self.original("front.png")], log=lambda *_: None)
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        caption = (Path(snapshot["dataset"]) / snapshot["inputs"][0]["file"]).with_suffix(".txt").read_text("utf-8")
        self.assertIn("long dark hair", caption)
        self.assertIn("white blouse", caption)

    def test_an_empty_seed_set_is_refused_with_a_reason(self):
        with self.assertRaises(pj.ProjectError):
            seed.build_snapshot(self.project, log=lambda *_: None)

    def test_the_same_inputs_give_the_same_snapshot_id(self):
        self._seed()
        first = seed.build_snapshot(self.project, log=lambda *_: None)
        second = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertEqual(first["id"], second["id"])

    def test_a_new_accepted_edit_changes_the_snapshot_id(self):
        self._seed()
        first = seed.build_snapshot(self.project, log=lambda *_: None)
        self._accepted_edit()
        second = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertNotEqual(first["id"], second["id"])

    def test_one_picture_is_one_training_image_even_under_two_ids(self):
        self.import_originals(3)
        source = self.original("twin.png")
        self.project.import_sources([source], log=lambda *_: None)
        source_id = {Path(r["src_path"]).name: r["id"] for r in self.project.source_records()}["twin.png"]
        copy = self.root / "twin-again.png"
        copy.write_bytes(source.read_bytes())
        record = lineage.record(copy, generator="qwen_edit", parent_ids=[source_id], train_generation=1,
                                edit_depth=1, caption="trg, same bytes")
        self.project.import_candidates([record], log=lambda *_: None)
        self.project.decide(record["id"], "accept")
        report = seed.audit(self.project)
        self.assertEqual(self.codes(report)[record["id"]], "duplicate_bytes")
        kept = {entry["id"]: entry for entry in report["kept"]}
        self.assertIn(source_id, kept)
        self.assertEqual(kept[source_id]["lineage_ids"], sorted([source_id, record["id"]]))
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        shas = [entry["sha256"] for entry in snapshot["inputs"]]
        self.assertEqual(len(shas), len(set(shas)))
        self.assertEqual(snapshot["counts"]["total"], len(seed.inventory(snapshot)))
        self.assertTrue(seed.validate_snapshot(snapshot["base"]) == [])

    def test_the_manifest_the_dataset_and_the_captions_agree(self):
        self._seed()
        self._accepted_edit()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        dataset = Path(snapshot["dataset"])
        images = seed.inventory(snapshot)
        self.assertEqual(len({e["sha256"] for e in snapshot["inputs"]}), len(images))
        self.assertEqual(len(list(dataset.glob("*.txt"))), len(images))
        for entry in snapshot["inputs"]:
            self.assertEqual(entry["caption_sha256"],
                             __import__("hashlib").sha256(entry["caption"].encode()).hexdigest())

    def test_a_rerun_reuses_the_published_snapshot_and_keeps_the_files(self):
        self._seed()
        first = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertFalse(first["reused"])
        images = seed.inventory(first)
        stamps = {p.name: p.stat().st_mtime_ns for p in images}
        time.sleep(0.05)
        second = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertTrue(second["reused"])
        self.assertEqual(first["id"], second["id"])
        after = seed.inventory(second)
        self.assertEqual([p.name for p in after], [p.name for p in images])
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in after}, stamps)

    def test_a_caption_change_gives_a_new_snapshot_and_leaves_the_old_one_alone(self):
        self._seed()
        first = seed.build_snapshot(self.project, log=lambda *_: None)
        first_files = {p.name: p.stat().st_mtime_ns for p in seed.inventory(first)}
        self.project.card["caption_traits"] = "include"
        self.project.card["fixed_traits"] = ["long dark hair"]
        self.project.save()
        second = seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertNotEqual(first["id"], second["id"])
        self.assertTrue(Path(first["base"]).is_dir())
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in seed.inventory(first)}, first_files)

    def test_a_damaged_snapshot_is_reported_and_not_overwritten(self):
        self._seed()
        first = seed.build_snapshot(self.project, log=lambda *_: None)
        victim = seed.inventory(first)[0]
        victim.write_bytes(b"NOT AN IMAGE ANY MORE")
        with self.assertRaises(pj.ProjectError) as caught:
            seed.build_snapshot(self.project, log=lambda *_: None)
        self.assertIn("not intact", str(caught.exception))
        self.assertTrue(victim.is_file())

    def test_the_snapshot_records_why_images_were_left_out(self):
        self._seed()
        holdout, _why = self.project.holdout_ids()
        self._accepted_edit("bad.png", parent=holdout[0])
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        codes = {row["code"] for row in snapshot["excluded"]}
        self.assertIn("holdout_direct", codes)
        self.assertIn("holdout_source", codes)
        self.assertEqual(snapshot["excluded_counts"]["holdout_direct"], 1)

    def test_the_runner_settings_ask_for_preserve_captions_and_the_project_family(self):
        self._seed()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        config = seed.runner_config(self.project, snapshot, epochs=10)
        self.assertEqual(config["family"], "krea2")
        self.assertEqual(config["caption_policy"], "preserve")
        self.assertEqual(config["kind"], "anime")
        self.assertEqual(config["trigger"], "trg")
        self.assertEqual(config["target_steps"], snapshot["counts"]["total"] * 10)
        self.assertEqual(config["dataset_root"], snapshot["dataset"])
        self.assertEqual(config["snapshot_id"], snapshot["id"])
        self.assertEqual(config["seed_policy"], snapshot["policy"])

    def test_the_runner_folder_carries_its_own_settings_and_instructions(self):
        self._seed()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        folder = seed.prepare_runner(self.project, snapshot, config=seed.runner_config(self.project, snapshot),
                                     log=lambda *_: None)
        for name in ("config.json", "todo.json", "caption_instructions_anime.txt"):
            self.assertTrue((folder / name).is_file(), name)
        todo = json.loads((folder / "todo.json").read_text(encoding="utf-8"))
        self.assertEqual(todo[0]["trigger"], "trg")

    def test_training_runs_the_runner_as_its_own_process_with_the_project_paths(self):
        self._seed()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        # A stand-in for the runner: it records its arguments and creates the run folder the caller looks for.
        fake = self.root / "fake_runner.py"
        fake.write_text(
            "import json, sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "print(json.dumps({'kind': 'run', 'path': args[args.index('--runs-dir') + 1]}))\n"
            "print('@@FIZGIG_EVENT@@' + json.dumps({'kind': 'stage', 'stage': 'Training'}))\n"
            "runs = Path(args[args.index('--runs-dir') + 1]) / 'demo_seed-abc'\n"
            "(runs / 'output').mkdir(parents=True, exist_ok=True)\n"
            "(runs / 'output' / 'demo_seed.safetensors').write_bytes(b'LORA')\n"
            "(runs / 'state.json').write_text(json.dumps({'complete': True, 'family': 'krea2'}), encoding='utf-8')\n",
            encoding="utf-8")
        config = seed.runner_config(self.project, snapshot, epochs=2)
        import pipeline.seed as seed_module
        events = []
        with _patched(seed_module, "AUTO", self.root), _patched(seed_module, "ROOT", self.root):
            (self.root / "runner.py").write_bytes(fake.read_bytes())
            runs = seed.train(self.project, snapshot, config=config, python=sys.executable, log=lambda *_: None,
                              on_event=events.append)
        # only @@FIZGIG_EVENT@@ lines become events; the runner's other lines are the human log
        self.assertEqual([e["kind"] for e in events], ["stage"])
        self.assertEqual(events[0]["stage"], "Training")
        self.assertTrue(Path(runs).is_dir())
        self.assertTrue(seed.lora_dir(runs, "demo_seed").joinpath("demo_seed.safetensors").is_file())

    def test_a_chatty_stderr_does_not_block_the_runner(self):
        """T5's regression, checked here because the seed run is the caller: 1 MiB of stderr must not deadlock."""
        self._seed()
        snapshot = seed.build_snapshot(self.project, log=lambda *_: None)
        noise = self.root / "noisy_runner.py"
        noise.write_text(
            "import json, sys\n"
            "for i in range(20000):\n"
            "    sys.stderr.write('noise %d ' % i + 'x' * 80 + '\\n')\n"
            "sys.stderr.flush()\n"
            "print('@@FIZGIG_EVENT@@' + json.dumps({'kind': 'stage', 'stage': 'Training'}), flush=True)\n",
            encoding="utf-8")
        config = seed.runner_config(self.project, snapshot, epochs=1)
        import pipeline.seed as seed_module
        events = []
        with _patched(seed_module, "AUTO", self.root), _patched(seed_module, "ROOT", self.root):
            (self.root / "runner.py").write_bytes(noise.read_bytes())
            seed.train(self.project, snapshot, config=config, python=sys.executable, log=lambda *_: None,
                       on_event=events.append)
        self.assertEqual([e["stage"] for e in events], ["Training"])


class _patched:
    """Tiny context manager, so the test can point seed.AUTO at a scratch folder without unittest.mock."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *_exc):
        setattr(self.module, self.name, self.saved)
        return False


if __name__ == "__main__":
    unittest.main()
