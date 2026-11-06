"""The project folder: sources, candidates, review decisions, snapshots and the lock between them.

    projects/<id>/
      project.json       the character card (spec 3.1)
      sources/           copies of the originals, named <sha256[:16]>.<ext>
      sources.jsonl      one record per source, with where it came from
      candidates.jsonl   one record per candidate image (spec 3.3)
      review.json        {candidate_id: {decision: accept|reject|fix, reason: ..., ...}}
      exports/<snapshot>/  the frozen training set (spec 3.5)
      pipeline.lock      one writer per project, across processes
      pipeline.write.lock  the guard file behind every project write (advisory, never deleted)

Origin is recorded, never inferred from a folder: tool2 writes generated anchors straight into `characters/<name>/`,
so a file sitting next to the user's photos is not evidence that the user drew it.

Two kinds of exclusion, because they answer different questions:

* `Lock` is the named, long-lived claim on a project ("this process is building the seed snapshot"). It refuses a
  second live owner, names the pid and thread in the file, and is released only by the owner that took it.
* `FileGuard` is the short write guard. Every read-modify-write of a project file goes through it, so two threads in
  one process and two processes on one folder cannot both read, change and write back the same JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

AUTO = Path(__file__).resolve().parents[1]
ROOT = AUTO.parent
if str(AUTO) not in sys.path:
    sys.path.insert(0, str(AUTO))

from pipeline import lineage, models  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_LINEAGE = {"max_train_generation": 2, "seed_max_generation": 1, "max_edit_depth": 1}
# What tool2's own candidate generator writes into a character folder; used as evidence, and recorded as such.
SYNTHETIC_PREFIXES = ("anchor_refined_", "anchor_", "retired_", "refined_", "sheet_")
DECISIONS = ("accept", "reject", "fix")
DEFAULT_HOLDOUT = {"min_sources": 15, "count": 2, "pick": []}
# The structured reasons a reviewer can give. `different_character` is the one that also changes what may be done
# with the image: a picture of another person cannot be corrected into this one, so it may only be rejected.
REVIEW_REASONS = ("different_character", "wrong_style", "artifact", "other")
FIX_BLOCKED_REASONS = ("different_character",)
# How a decision was reached. A decision carried over from the M0 score sheet is a person's earlier judgement about a
# different question; it is kept, and it stays distinguishable from a fresh look at the review page.
ORIGIN_M0 = "m0_score"
ORIGIN_USER = "user"
M0_BY_PREFIX = "user (m0"
GUARD_NAME = "pipeline.write.lock"


class ProjectError(ValueError):
    """A project that cannot be read, or an operation it does not allow."""


def projects_dir(cfg=None):
    block = (cfg if cfg is not None else models.config()).get("pipeline") or {}
    return Path(models.resolve_path(block.get("projects_dir") or "automation/projects"))


def path_of(project_id, cfg=None):
    return projects_dir(cfg) / project_id


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {} if default is None else default


def _temp_beside(path):
    """A temp name unique to this process and call, so two writers can never share one temp file."""
    return path.with_name(f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{os.urandom(4).hex()}")


def write_json(path, data):
    """Write JSON so a reader sees either the old file or the new one, never a half-written one."""
    return write_text(path, json.dumps(data, ensure_ascii=False, indent=1))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temp_beside(path)
    try:
        with open(temp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return path


# ---------------------------------------------------------------------- the write guard
_GUARD_LOCK = threading.Lock()
_GUARD_BY_KEY = {}
_HELD = threading.local()                 # {guard path: depth}, for this thread only


def _thread_lock(key):
    with _GUARD_LOCK:
        return _GUARD_BY_KEY.setdefault(key, threading.RLock())


def _held():
    paths = getattr(_HELD, "paths", None)
    if paths is None:
        paths = _HELD.paths = {}
    return paths


def _os_lock(handle, *, deadline):
    """Take an exclusive byte-range lock on an open file, waiting until the deadline."""
    if os.name == "nt":
        import msvcrt
        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        return
    import fcntl
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _os_unlock(handle):
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


class FileGuard:
    """One writer at a time inside a folder, across threads and processes.

    Threads are serialised by an in-process lock keyed on the guard file; processes by an OS byte-range lock on that
    same file. The guard file is created once and never deleted: unlinking it would let two processes lock different
    inodes and both believe they own the folder. The OS releases the lock when its holder dies, so there is no stale
    lock to clean up. A guard taken again by the thread that already holds it only deepens the count - the OS lock is
    not re-taken, because a second handle on the same byte range would deadlock against the first.
    """

    def __init__(self, folder, name=GUARD_NAME, timeout=20.0):
        self.path = Path(folder) / name
        self.timeout = float(timeout)
        self._handle = None
        self._key = None
        self._thread_lock = None
        self._nested = False
        self.depth = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        key = str(self.path.resolve())
        self._key = key
        self._thread_lock = _thread_lock(key)
        self._thread_lock.acquire()
        held = _held()
        if held.get(key):
            held[key] += 1
            self._nested = True
            self.depth += 1
            return self
        try:
            handle = open(self.path, "a+b")
        except OSError as exc:                          # a read-only folder is reported, not ignored
            self._thread_lock.release()
            raise ProjectError(f"cannot open the write guard {self.path}: {exc}") from exc
        try:
            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\n")
                handle.flush()
            _os_lock(handle, deadline=time.monotonic() + self.timeout)
        except OSError as exc:
            handle.close()
            self._thread_lock.release()
            raise ProjectError(f"{self.path.parent.name} stayed locked for {self.timeout:g}s; "
                               f"another writer is using it ({exc})") from exc
        held[key] = 1
        self._handle = handle
        self.depth += 1
        return self

    def __exit__(self, *_exc):
        self.depth = max(0, self.depth - 1)
        if self._nested:
            held = _held()
            held[self._key] = max(0, held.get(self._key, 1) - 1)
            self._thread_lock.release()
            return False
        if self._handle is not None:
            _os_unlock(self._handle)
            self._handle.close()
            self._handle = None
            _held().pop(self._key, None)
        if self._thread_lock is not None:
            self._thread_lock.release()
        return False


class Lock:
    """The named claim on a project: one live owner, and release only by the owner.

    The owner record carries a process id, a thread id and a token. A lock file whose owner is gone is reclaimed; one
    whose owner is alive is refused, and a release never deletes a record that is no longer ours.
    """

    def __init__(self, folder, name="pipeline.lock"):
        self.path = Path(folder) / name
        self.acquired = False
        self.token = None
        self.depth = 0
        self.what = ""

    def owner(self):
        data = read_json(self.path, None)
        if not isinstance(data, dict):
            return None
        try:
            return {"pid": int(data["pid"]), "at": data.get("at"), "what": data.get("what", ""),
                    "thread": data.get("thread"), "token": data.get("token")}
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _alive(pid):
        if pid <= 0:
            return False
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5
            try:
                code = wintypes.DWORD()
                return not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def live_owner(self):
        owner = self.owner()
        return owner if owner and self._alive(owner["pid"]) else None

    def held_by_me(self):
        """Is the record on disk the one this object wrote?"""
        return bool(self.token) and (self.owner() or {}).get("token") == self.token

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.depth and self.held_by_me():                 # the same holder, once more
            self.depth += 1
            return self
        for _attempt in range(20):
            live = self.live_owner()
            if live and not (live["pid"] == os.getpid() and live.get("thread") == threading.get_ident()):
                raise ProjectError(
                    f"{self.path.parent.name} is already in use (pid {live['pid']}"
                    f"{', thread ' + str(live['thread']) if live.get('thread') else ''}: {live['what']}); "
                    f"remove {self.path} if that process is gone")
            self.token = os.urandom(6).hex()
            self.what = " ".join(sys.argv[:3])
            guard = FileGuard(self.path.parent)
            with guard:                                     # two processes cannot both claim it
                live = self.live_owner()
                if live and live.get("token") not in (None, self.token):
                    continue
                write_json(self.path, {"pid": os.getpid(), "at": time.time(), "what": self.what,
                                       "thread": threading.get_ident(), "token": self.token})
                if self.held_by_me():
                    self.acquired = True
                    self.depth = 1
                    return self
        raise ProjectError(f"{self.path.parent.name}: could not take the lock at {self.path}")

    def __exit__(self, *_exc):
        if not self.depth:
            return False
        self.depth -= 1
        if self.depth:
            return False
        if self.held_by_me():
            guard = FileGuard(self.path.parent)
            with guard:
                if self.held_by_me():
                    self.path.unlink(missing_ok=True)
        self.acquired = False
        self.token = None
        return False


class Project:
    """One character's folder: its card, its sources, its candidates and its review decisions."""

    def __init__(self, folder, card):
        # Absolute from the start: a record written with a relative path means something different in every process
        # that opens the project, and `file` values are recorded for whoever opens it later.
        self.dir = Path(folder).resolve()
        self.card = card

    # ------------------------------------------------------------------ opening
    @classmethod
    def create(cls, project_id, card, folder=None):
        folder = Path(folder) if folder else path_of(project_id)
        if (folder / "project.json").is_file():
            raise ProjectError(f"{folder} already holds a project")
        card = {**card, "id": project_id}
        card.setdefault("lineage", dict(DEFAULT_LINEAGE))
        card.setdefault("holdout", dict(DEFAULT_HOLDOUT))
        card.setdefault("caption_traits", "omit")
        card.setdefault("final_families", ["qwen_image21", "krea2"])
        for name in ("sources", "candidates", "exports"):
            (folder / name).mkdir(parents=True, exist_ok=True)
        write_json(folder / "project.json", card)
        write_json(folder / "review.json", {})
        return cls(folder, card)

    @classmethod
    def open(cls, folder):
        folder = Path(folder)
        card = read_json(folder / "project.json")
        if not card:
            raise ProjectError(f"no project.json in {folder}")
        return cls(folder, card)

    def save(self):
        with self.guard():
            write_json(self.dir / "project.json", self.card)

    def lock(self):
        return Lock(self.dir)

    def guard(self, name=GUARD_NAME, timeout=20.0):
        """The short write guard: hold it around any read-modify-write of a project file."""
        return FileGuard(self.dir, name=name, timeout=timeout)

    # ------------------------------------------------------------------ sources
    def source_records(self):
        return lineage.read_jsonl(self.dir / "sources.jsonl")

    def candidates(self):
        return lineage.read_jsonl(self.dir / "candidates.jsonl")

    def candidate_by_id(self, candidate_id):
        for record in self.candidates():
            if record.get("id") == candidate_id:
                return record
        return None

    def resolve_file(self, value, *, base=None):
        """A project file path, from `file`/`src_path` as recorded, whichever way it was written.

        Records have been written three ways: relative to the project folder, relative to the automation root (an
        early run), and absolute. All three are accepted, and the first one that exists wins, so a record keeps
        pointing at its bytes no matter which working directory wrote it.
        """
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        else:
            for root in (Path(base) if base else None, self.dir, ROOT, AUTO):
                if root is not None:
                    candidates.append(Path(root) / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0] if candidates else None

    def import_sources(self, paths, *, origin_by_name=None, log=print):
        """Copy originals into `sources/` under their content hash and record where each came from.

        Bytes are never modified, and a file already imported (same hash) is not copied twice: a source's identity is
        its content, so the same photo under two names is one source.
        """
        origin_by_name = origin_by_name or {}
        records, added, reused = [], 0, 0
        with self.guard():
            known = {r["sha256"] for r in self.source_records()}
            for path in paths:
                path = Path(path)
                if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                    continue
                sha = lineage.sha256_file(path)
                origin, why = self._origin_for(path.name, origin_by_name)
                if sha in known:
                    reused += 1
                    continue
                dest = self.dir / "sources" / (sha[:16] + path.suffix.lower())
                if not dest.exists():
                    shutil.copy2(path, dest)
                record = lineage.source_record(path, origin=origin, note=why)
                record["file"] = str(dest.relative_to(self.dir))
                record["decided_by"] = why
                records.append(record)
                known.add(sha)
                added += 1
            written = lineage.append_jsonl(self.dir / "sources.jsonl", records)
        log(f"[{self.card.get('id')}] sources: {written} new, {reused} already known")
        return written, reused

    def _origin_for(self, name, origin_by_name):
        if name in origin_by_name:
            return origin_by_name[name], "listed by the caller"
        if name in set(self.card.get("synthetic_sources") or []):
            return "synthetic", "listed in the card's synthetic_sources"
        if any(name.startswith(prefix) for prefix in SYNTHETIC_PREFIXES):
            return "synthetic", "name matches tool2's own candidate prefixes " + ", ".join(SYNTHETIC_PREFIXES)
        return "user", "assumed an original the user supplied"

    def source_share(self):
        """(synthetic fraction, counts) over the imported sources."""
        records = self.source_records()
        if not records:
            return 0.0, {"user": 0, "synthetic": 0}
        counts = {key: sum(1 for r in records if r.get("origin") == key) for key in ("user", "synthetic")}
        return counts["synthetic"] / len(records), counts

    # ------------------------------------------------------------------ holdout (spec 7 Q6)
    def holdout_entry(self):
        """The holdout set: {ids, files, shas, reason, frozen}.

        Once frozen in the card the set never moves again, because the images that judge a LoRA must not change when
        sources are re-imported, renamed or reordered. An unfrozen card is still resolved the old way, so an existing
        project keeps working; `freeze_holdout()` records the answer.
        """
        holdout = {**DEFAULT_HOLDOUT, **(self.card.get("holdout") or {})}
        records = {r["id"]: r for r in self.source_records()}
        chosen, reason = [], ""
        frozen = list(holdout.get("ids") or [])
        if frozen:
            missing = [i for i in frozen if i not in records]
            chosen = [i for i in frozen if i in records]
            reason = f"{len(chosen)} source(s) frozen in the card"
            if missing:
                reason += f"; not in sources.jsonl: {', '.join(missing)}"
        else:
            users = [r for r in records.values() if r.get("origin") == "user"]
            if len(users) < int(holdout["min_sources"]):
                return {"ids": [], "files": [], "shas": [], "frozen": False,
                        "reason": f"only {len(users)} user original(s); below holdout.min_sources "
                                  f"{holdout['min_sources']}"}
            picked = list(holdout.get("pick") or [])
            if picked:
                by_name = {Path(r["src_path"]).name: r for r in users}
                chosen = [by_name[name]["id"] for name in picked if name in by_name]
                missing = [name for name in picked if name not in by_name]
                reason = f"{len(chosen)} named by the card"
                if missing:
                    reason += f"; not found: {', '.join(missing)}"
            else:
                front = next((r for r in users if "front" in Path(r["src_path"]).name.lower()), users[0])
                rest = [r for r in users if r["id"] != front["id"]]
                chosen = [front["id"]] + [r["id"] for r in rest[: max(0, int(holdout["count"]) - 1)]]
                reason = "front view first, then the next original"
        return {"ids": chosen,
                "files": [Path(records[i].get("src_path") or "").name for i in chosen if i in records],
                "shas": [records[i]["sha256"] for i in chosen if i in records],
                "frozen": bool(frozen), "reason": reason}

    def holdout_ids(self):
        """The source ids set aside for evaluation: only when there are enough user originals."""
        entry = self.holdout_entry()
        return entry["ids"], entry["reason"]

    def freeze_holdout(self, *, log=print, apply=True):
        """Write the resolved holdout into the card, so later source changes cannot move it."""
        entry = self.holdout_entry()
        if entry["frozen"]:
            log(f"[holdout] already frozen: {len(entry['ids'])} source(s)")
            return entry
        if not entry["ids"]:
            log(f"[holdout] nothing to freeze: {entry['reason']}")
            return entry
        log(f"[holdout] freezing {len(entry['ids'])} source(s): {', '.join(entry['files'])}")
        for source_id, name, sha in zip(entry["ids"], entry["files"], entry["shas"]):
            log(f"  {source_id}  {name}  {sha[:12]}")
        if apply:
            holdout = {**DEFAULT_HOLDOUT, **(self.card.get("holdout") or {})}
            holdout.update({"ids": entry["ids"], "files": entry["files"], "shas": entry["shas"],
                            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": entry["reason"]})
            self.card["holdout"] = holdout
            self.save()
            entry = self.holdout_entry()
        return entry

    # ------------------------------------------------------------------ review (spec 3.1/4 M4 core)
    def review(self):
        return read_json(self.dir / "review.json", {})

    def decide(self, candidate_id, decision, *, note="", by="user", reason=None, extra=None):
        """Record (or clear) one decision. Read-modify-write under the write guard, so nothing is lost."""
        if decision is not None and decision not in DECISIONS:
            raise ProjectError(f"decision must be one of {', '.join(DECISIONS)}, not {decision!r}")
        if reason is not None and reason not in REVIEW_REASONS:
            raise ProjectError(f"reason must be one of {', '.join(REVIEW_REASONS)}, not {reason!r}")
        if decision is None:
            return self.undecide(candidate_id)
        origin = ORIGIN_M0 if str(by).lower().startswith(M0_BY_PREFIX) else (
            ORIGIN_USER if by == "user" else str(by))
        with self.guard():
            review = self.review()
            review[candidate_id] = {"decision": decision, "by": by, "origin": origin,
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "note": note,
                                    **({"reason": reason} if reason else {}), **(extra or {})}
            write_json(self.dir / "review.json", review)
            return review[candidate_id]

    def undecide(self, candidate_id):
        with self.guard():
            review = self.review()
            removed = review.pop(candidate_id, None)
            write_json(self.dir / "review.json", review)
            return removed

    def state_of(self, candidate_id):
        return self.review().get(candidate_id) or {}

    def accepted(self, *, limit_generation=None):
        """The candidates marked accept, oldest first, optionally capped by training generation."""
        review = self.review()
        out = []
        for record in self.candidates():
            if (review.get(record["id"]) or {}).get("decision") != "accept":
                continue
            if limit_generation is not None and int(record.get("train_generation") or 0) > limit_generation:
                continue
            out.append(record)
        return out

    def by_decision(self, decision):
        review = self.review()
        return [r for r in self.candidates() if (review.get(r["id"]) or {}).get("decision") == decision]

    def undecided(self):
        review = self.review()
        return [r for r in self.candidates() if r["id"] not in review]

    def quotas(self, planned_slots):
        """Accepted count per slot category against what the plan asked for.

        `planned_slots` is a mapping of slot id -> target count; a slot is short when too few of its candidates were
        accepted. Slot ids are what the plan used (`ang_front_close`), so the category is their first field.
        """
        plan = {slot: int(target) for slot, target in (planned_slots or {}).items()}
        counts = {slot: 0 for slot in plan}
        for record in self.accepted():
            slot = (record.get("slot") or {}).get("slot_id") or (record.get("slot") or {}).get("id")
            if slot is None:
                category = (record.get("slot") or {}).get("category")
                for candidate_slot in plan:
                    if category and candidate_slot.startswith(category + "_"):
                        slot = candidate_slot
                        break
            if slot in counts:
                counts[slot] += 1
        return {slot: {"target": target, "accepted": counts.get(slot, 0),
                       "short": max(0, target - counts.get(slot, 0))} for slot, target in plan.items()}

    def import_candidates(self, records, *, base=None, log=print):
        """Copy candidate images into `candidates/<generator>/<batch>/` and record them.

        A generator writes its images wherever it likes (a tool2 snapshot, an experiment folder); the project keeps its
        own copy under the content hash so a later cleanup of those folders cannot break a lineage record. A record's
        `file` may be relative - an earlier run recorded it relative to the folder that run used - which is what `base`
        resolves. Records already known by id are left alone: importing the same batch twice adds nothing.
        """
        written, skipped = [], []
        with self.guard():
            known = {r["id"] for r in self.candidates()}
            for record in records:
                source = self.resolve_file(record.get("file"), base=base)
                if record.get("id") in known:
                    skipped.append({"id": record.get("id"), "reason": "already in the project"})
                    continue
                if not source or not source.is_file():
                    skipped.append({"id": record.get("id"), "reason": f"file not found: {record.get('file')}"})
                    continue
                sha = record.get("sha256") or lineage.sha256_file(source)
                batch = str(record.get("snapshot") or record.get("batch") or "manual")[:24]
                folder = self.dir / "candidates" / str(record.get("generator") or "unknown") / batch
                folder.mkdir(parents=True, exist_ok=True)
                dest = folder / (sha[:16] + source.suffix.lower())
                if not dest.exists():
                    shutil.copy2(source, dest)
                record = {**record, "file": str(dest), "sha256": sha}
                written.append(record)
                known.add(record["id"])
            num = lineage.append_jsonl(self.dir / "candidates.jsonl", written)
        log(f"[{self.card.get('id')}] candidates: {num} imported, {len(skipped)} skipped")
        return num, skipped

    # ------------------------------------------------------------------ migration
    def sync_source_parents(self, *, apply=True, log=print):
        """Copy parents from a candidate row onto the source row with the same bytes.

        A generated picture can arrive in the project twice: as the candidate that was reviewed, and as an input file a
        later run copied into the character folder (which `import_sources` records as a source). The two rows describe
        one picture, but only the candidate knows where it came from - and the seed audit walks parent ids from the
        source row, so an anchor whose parents were recorded on its candidate looked like an ancestor of unknown
        origin and took everything drawn from it out of the seed with it.
        """
        sources = self.source_records()
        by_sha = {}
        for record in self.candidates():
            if record.get("parent_ids"):
                by_sha.setdefault(record.get("sha256"), record)
        changed, unresolved = [], []
        for source in sources:
            if source.get("parent_ids") or not source.get("sha256"):
                continue
            twin = by_sha.get(source["sha256"])
            if twin is None:
                if source.get("origin") == "synthetic":
                    unresolved.append({"id": source["id"], "file": source.get("file")})
                continue
            # A candidate that is a copy of a source records that source as its parent; copying that onto the source
            # row would make the picture its own ancestor, and a self-parent is a cycle, not a provenance.
            parents = [parent for parent in twin["parent_ids"] if parent != source["id"]]
            if not parents:
                if source.get("origin") == "synthetic":
                    unresolved.append({"id": source["id"], "file": source.get("file")})
                continue
            source["parent_ids"] = parents
            source["parents_from"] = twin["id"]
            changed.append({"id": source["id"], "from": twin["id"], "parents": source["parent_ids"]})
        log(f"[lineage] {len(changed)} source row(s) took their parents from a candidate, "
            f"{len(unresolved)} synthetic source(s) still have none")
        if apply and changed:
            with self.guard():
                lineage.write_jsonl(self.dir / "sources.jsonl", sources)
        return {"changed": changed, "unresolved": unresolved}

    def normalize_paths(self, *, apply=True, log=print):
        """Rewrite recorded file paths to absolute ones, so a record stops depending on the working directory.

        Records were written three ways over the project's life: relative to the project, relative to `automation/`,
        and absolute. `resolve_file` reads all three, but a new record should not add a fourth interpretation to the
        pile; this rewrites what exists and reports what it could not resolve instead of guessing.
        """
        records = self.candidates()
        changed, unresolved = [], []
        for record in records:
            raw = str(record.get("file") or "")
            if not raw:
                continue
            resolved = self.resolve_file(raw)
            if resolved is None or not resolved.is_file():
                unresolved.append({"id": record["id"], "file": raw})
                continue
            if raw != str(resolved):
                record["file"] = str(resolved)
                changed.append({"id": record["id"], "from": raw, "to": str(resolved)})
        sources = self.source_records()
        for record in sources:
            raw = str(record.get("file") or "")
            if not raw:
                continue
            resolved = self.resolve_file(raw)
            if resolved is not None and resolved.is_file() and raw != str(resolved):
                record["file"] = str(resolved)
                changed.append({"id": record["id"], "from": raw, "to": str(resolved)})
        log(f"[paths] {len(changed)} record(s) made absolute, {len(unresolved)} unresolved")
        if apply and changed:
            with self.guard():
                lineage.write_jsonl(self.dir / "candidates.jsonl", records)
                lineage.write_jsonl(self.dir / "sources.jsonl", sources)
        return {"changed": changed, "unresolved": unresolved}

    def review_plan(self):
        """What the review history would look like after the migration, and why (reported before it is written).

        Two things are moved out of free text: the stated `different_character` reason (which the server needs as a
        value, not as prose) and how each decision was reached (`origin`), so a decision carried over from the M0 score
        sheet can be told from a fresh one.
        """
        review = self.review()
        changes = []
        for candidate_id, state in review.items():
            note = str(state.get("note") or "")
            wanted = dict(state)
            if state.get("reason") in REVIEW_REASONS:
                pass
            elif "다른 캐릭터" in note or "different character" in note.lower():
                wanted["reason"] = "different_character"
            if not wanted.get("origin"):
                wanted["origin"] = (ORIGIN_M0 if str(state.get("by") or "").lower().startswith(M0_BY_PREFIX)
                                    else ORIGIN_USER)
            if wanted != state:
                changes.append({"id": candidate_id, "from": state, "to": wanted,
                                "why": ", ".join(
                                    part for part in (
                                        "reason <- note" if "reason" in wanted and "reason" not in state else "",
                                        f"origin={wanted.get('origin')}" if "origin" not in state else "")
                                    if part)})
        holdout = self.holdout_entry()
        return {"review_changes": changes, "holdout": holdout,
                "counts": {"review": len(review), "changed": len(changes),
                           "different_character": sum(1 for c in changes
                                                      if c["to"].get("reason") == "different_character"),
                           "m0_derived": sum(1 for c in changes if c["to"].get("origin") == ORIGIN_M0)}}

    def migrate_review(self, *, apply=True, log=print):
        plan = self.review_plan()
        log(f"[migrate] review entries: {plan['counts']['review']}, to change: {plan['counts']['changed']} "
            f"(reasons: {plan['counts']['different_character']}, origin tags: {plan['counts']['m0_derived']})")
        for change in plan["review_changes"]:
            log(f"  {change['id']}: {change['why']}")
        if apply and plan["review_changes"]:
            with self.guard():
                review = self.review()
                for change in plan["review_changes"]:
                    review[change["id"]] = change["to"]
                write_json(self.dir / "review.json", review)
        if apply:
            plan["holdout"] = self.freeze_holdout(log=log, apply=True)
        return plan

    # ------------------------------------------------------------------ lineage
    def index(self):
        """{id: record} over sources and candidates, for tracing to an original."""
        out = {}
        for record in self.source_records() + self.candidates():
            out[record["id"]] = record
        return out

    def trace(self, candidate_id):
        """(visited ids, unknown parent ids) from a candidate back to the originals."""
        record = self.candidate_by_id(candidate_id)
        if record is None:
            raise ProjectError(f"no candidate {candidate_id}")
        return lineage.trace_to_origin(record, self.index())

    def digest(self):
        """A short fingerprint of the card, so a snapshot can record which project state produced it."""
        return hashlib.sha256(json.dumps(self.card, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
