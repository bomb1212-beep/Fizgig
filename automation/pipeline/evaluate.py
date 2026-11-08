"""M4: judging the finished LoRA on pictures it was never trained on, and checking the scorers against a person.

Two reports live here, and they answer different questions with the same restraint:

**Holdout evaluation.** The holdout is a small set of originals frozen before any training, kept out of every snapshot
and every export (spec Q6). Once a final LoRA exists, the holdout prompts are drawn with it and compared to the holdout
originals by the identity scorer of that family. The number that matters is family by family: a LoRA's Krea2 identity
score says nothing about its Qwen score, and averaging them would hide exactly the difference the two families exist to
show.

**Agreement with a person.** The spec asks how the automatic scores line up with the M0 human scores: quartile
agreement, the cases a person rejected that scored high, and side and back views counted separately. This is a
calibration report, not a gate. A disagreement is a fact about the scorer, and the answer to "the scorer says 0.9 but
the reviewer said no" is to look at the picture, not to lower a threshold.

Neither report writes to the project. Both return data a caller renders; nothing is accepted, rejected or deleted.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

# Views where a whole-frame embedding is known to be a weak proxy: the face is small, turned away, or absent. Counting
# them in the overall agreement would flatter or punish the scorer for something it cannot see, so they are also
# reported on their own.
HARD_VIEW_SLOTS = ("ang_back", "ang_side", "frm_back", "frm_profile", "from_above", "dutch")

# What a holdout measurement is *for* (spec N4.3). The three are not interchangeable and the report has to say which
# one it is, because "0.77 with the LoRA" and "0.62 without it" only mean something as a pair, and a seed-run number
# answers a different question from a final-LoRA one.
HOLDOUT_PURPOSES = ("seed", "final", "baseline")


class EvalError(RuntimeError):
    """An evaluation that cannot be run as asked."""


def is_hard_view(slot_id):
    """Is this slot one where the character's face is not visible to a whole-frame scorer?"""
    slot_id = str(slot_id or "")
    return any(slot_id.startswith(prefix) or prefix in slot_id for prefix in HARD_VIEW_SLOTS)


def quartiles(values):
    """{q1, median, q3} for a list of numbers, or None when there are too few to split.

    Four values is the floor: three splits need at least enough points to sit either side of the median.
    """
    values = sorted(float(v) for v in values if v is not None)
    if len(values) < 4:
        return None
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"q1": q[0], "median": q[1], "q3": q[2], "n": len(values)}


def weighted_human(row):
    """The same 0.45/0.30/0.25 weighting M0 used, or None when the row was not fully scored."""
    try:
        identity = float(row.get("identity") or "")
        instruction = float(row.get("instruction") or "")
        quality = float(row.get("quality") or "")
    except (TypeError, ValueError):
        return None
    return identity * 0.45 + instruction * 0.30 + quality * 0.25


def read_scores(path):
    """The M0 human scores as rows, with `human` (the weighted mark) filled in when it can be."""
    path = Path(path)
    if not path.is_file():
        raise EvalError(f"no scores file at {path}")
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["human"] = weighted_human(row)
        row["rejected"] = str(row.get("error") or "").strip() not in ("", "0")
    return rows


def load_experiment_tasks(path):
    """(tasks, experiment_dir) from a saved experiment manifest: the slot, method, seed and image of every cell.

    The manifest is the record of what the run drew, saved next to the images it names - which is why the join for the
    agreement report runs through it (spec R5).
    """
    path = Path(path)
    if not path.is_file():
        raise EvalError(f"no experiment manifest at {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    tasks = data.get("tasks") or []
    if not tasks:
        raise EvalError(f"the experiment manifest at {path} records no tasks")
    return tasks, path.parent


def match_sheet_to_automatic(scores_rows, automatic_rows, tasks, project, *, experiment_dir, log=print):
    """(matched, matching): the human sheet joined to automatic scores through the saved experiment manifest.

    The join is by picture, not by name (spec R5): a sheet row's slot, method and seed name the manifest task that drew
    it, the task names the image file, and only the sha256 of those bytes - the same bytes the project imported as a
    candidate - connects the row to an automatic score. Anything that cannot prove it is the same picture stays
    unmatched with its reason counted, because a guessed pairing would flatter or punish the scorer for someone else's
    picture.
    """
    from . import lineage
    sha_to_candidate = {record["sha256"]: record["id"] for record in project.candidates() if record.get("sha256")}
    score_by_id = {row.get("id"): row for row in automatic_rows if row.get("id")}
    matched, unmatched = {}, {}

    def unmatch(reason, task_key=""):
        unmatched[reason] = unmatched.get(reason, 0) + 1
        examples = unmatched.setdefault("_examples_" + reason, [])
        if len(examples) < 5:
            examples.append(task_key)

    for task in tasks:
        task_key = f'{task.get("slot_id") or task.get("slot") or ""}__{task.get("method")}__s{task.get("seed")}'
        file_ref = str(task.get("file") or "")
        if not file_ref:
            unmatch("the task names no image file", task_key)
            continue
        path = Path(file_ref)
        if not path.is_absolute():
            path = Path(experiment_dir) / path
        if not path.is_file():
            unmatch("the task's image is not on disk", task_key)
            continue
        sha = lineage.sha256_file(path)
        candidate_id = sha_to_candidate.get(sha)
        if not candidate_id:
            unmatch("the task's bytes are not in the project", task_key)
            continue
        auto_row = score_by_id.get(candidate_id)
        if auto_row is None:
            unmatch("no automatic score for this picture", task_key)
            continue
        matched[task_key] = {"score_row": auto_row, "candidate": candidate_id, "sha256": sha}
    return matched, {"unmatched": unmatched}


def agreement_from_manifest(scores_rows, automatic_rows, tasks, project, *, experiment_dir,
                            score_field="clip_max_cosine", log=print):
    """The agreement report with the manifest's join recorded, pair by pair (spec R5).

    `automatic_rows` are the rows of a scores.json; the match goes through the experiment manifest's slot, method, seed
    and image sha, and every pair carries the candidate id and sha that vouch for it. The matching block counts the
    rows that could not be joined and why, so the reader can see how much of the sheet the comparison covers.
    """
    matched, matching = match_sheet_to_automatic(scores_rows, automatic_rows, tasks, project,
                                                 experiment_dir=experiment_dir, log=log)
    table = {}
    for task_key, entry in matched.items():
        value = entry["score_row"].get(score_field)
        if value is not None:
            table[task_key] = value
    report = agreement(scores_rows, table, score_field=score_field)
    for pair in report.get("pairs") or []:
        entry = matched.get(pair.get("key")) or {}
        pair["match"] = {"candidate": entry.get("candidate"), "sha256": entry.get("sha256"),
                         "basis": "experiment manifest (slot, method, seed) joined by image sha256"}
    counts = {k: v for k, v in (matching.get("unmatched") or {}).items() if not k.startswith("_examples_")}
    examples = {k[len("_examples_"):]: v for k, v in (matching.get("unmatched") or {}).items()
                if k.startswith("_examples_")}
    report["matching"] = {"basis": "experiment manifest: slot, method, seed, image sha256",
                          "score_field": score_field,
                          "manifest_tasks": len(tasks),
                          "matched_by_manifest": len(matched),
                          "with_score": len(table),
                          "unmatched": counts,
                          "unmatched_examples": examples}
    return report


def agreement(scores, automatic, *, score_field="max_cosine"):
    """How the automatic scores line up with a person's marks, overall and for the hard views.

    `scores` are M0 rows (with `task` or `slot`), `automatic` is `{task or slot: score}`. Only rows that have both a
    human mark and an automatic score take part; everything else is counted as unpaired so the reader can see how much
    of the sheet the comparison actually covers.
    """
    paired = []
    unpaired = []
    for row in scores:
        key = row.get("task") or row.get("slot")
        auto = automatic.get(key)
        if auto is None:
            unpaired.append({"key": key, "reason": "no automatic score"})
            continue
        if row.get("human") is None:
            unpaired.append({"key": key, "reason": "not scored by hand"})
            continue
        paired.append({
            "key": key, "slot": row.get("slot"), "method": row.get("method"),
            "human": row["human"], "auto": float(auto), "rejected": bool(row.get("rejected")),
            "hard_view": is_hard_view(row.get("slot")),
        })
    if not paired:
        raise EvalError("no row has both a human mark and an automatic score, so nothing can be compared")

    human_quartiles = quartiles([row["human"] for row in paired])
    auto_quartiles = quartiles([row["auto"] for row in paired])

    def split(rows):
        if len(rows) < 4:
            return {"pairs": len(rows), "note": "too few pairs to split into quartiles", "agreement": None}
        hi_human = {r["key"] for r in rows if r["human"] >= _threshold(rows, "human", 0.75)}
        hi_auto = {r["key"] for r in rows if r["auto"] >= _threshold(rows, "auto", 0.75)}
        lo_auto = {r["key"] for r in rows if r["auto"] <= _threshold(rows, "auto", 0.25)}
        hi_human_lo_auto = hi_human & lo_auto
        false_high = sorted({r["key"] for r in rows if r["rejected"] and r["auto"] >= auto_quartiles["median"]})
        return {
            "pairs": len(rows),
            "top_quartile_agreement": round(len(hi_human & hi_auto) / max(1, len(hi_human)), 3),
            "high_human_low_auto": sorted(hi_human_lo_auto),
            "rejected_but_scored_high": false_high,
            "spearman": spearman([(r["human"], r["auto"]) for r in rows]),
        }

    report = {
        "paired": len(paired),
        "unpaired": unpaired[:50],
        "unpaired_count": len(unpaired),
        "score_field": score_field,
        "overall": split(paired),
        "easy_views": split([r for r in paired if not r["hard_view"]]),
        "hard_views": split([r for r in paired if r["hard_view"]]),
        "human_quartiles": human_quartiles,
        "auto_quartiles": auto_quartiles,
        "pairs": paired,
    }
    return report


def _threshold(rows, field, fraction):
    values = sorted(float(r[field]) for r in rows)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, int(round(fraction * (len(values) - 1)))))
    return values[index]


def spearman(pairs):
    """Rank correlation without a scipy dependency: 1.0 means the two orders agree completely.

    Ties get average ranks, which is what makes it usable here - human marks are 1-5 with many repeats.
    """
    pairs = [(float(a), float(b)) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    ranks_a = _average_ranks([p[0] for p in pairs])
    ranks_b = _average_ranks([p[1] for p in pairs])
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    den = (sum((a - mean_a) ** 2 for a in ranks_a) * sum((b - mean_b) ** 2 for b in ranks_b)) ** 0.5
    return round(num / den, 4) if den else None


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def holdout_family_report(family, scores, *, references, cells_audit=None, purpose=None):
    """One family's holdout result, with its references named and its cells accounted for.

    `scores` is a scorer result whose rows carry `max_cosine`; `references` are the holdout originals it was compared
    against. Reported per family because a number from one family is not evidence about another, and the cells audit is
    part of the number: a measurement over vouched cells says something different from one over files that merely sat
    in the folder (spec R5). `purpose` names which experiment this is (seed/final/baseline, spec N4.3) so two numbers
    for one family can never be read as the same measurement.
    """
    if not references:
        raise EvalError(f"the holdout evaluation for {family} has no references to compare against")
    values = [row["max_cosine"] for row in scores["rows"] if row.get("max_cosine") is not None]
    missing = [row["id"] for row in scores["rows"] if row.get("max_cosine") is None]
    report = {
        "family": family,
        "purpose": purpose,
        "scorer": scores["scorer"],
        "references": [str(r) for r in references],
        "images": len(values),
        "unscored": missing,
        "max": round(max(values), 4) if values else None,
        "mean": round(sum(values) / len(values), 4) if values else None,
        "median": round(statistics.median(values), 4) if values else None,
        "rows": scores["rows"],
        "note": ("no image could be scored" if not values else
                 f"{len(missing)} image(s) carry no score and are not counted as zero"),
    }
    if cells_audit is not None:
        report["cells"] = cells_audit
        if cells_audit.get("expected") and cells_audit.get("scored") != cells_audit.get("expected"):
            report["partial"] = True
            report["note"] += (f"; the plan has {cells_audit['expected']} cell(s) and {cells_audit['scored']} were "
                               "scored, so this is a partial measurement")
    return report



def compare_families(reports):
    """A table of holdout results, one row per family, with the caveat stated rather than implied."""
    rows = [{"family": r["family"], "scorer": r["scorer"], "images": r["images"],
             "max": r["max"], "mean": r["mean"], "median": r["median"]} for r in reports]
    families = [r["family"] for r in reports]
    note = None
    if len(families) > 1 and len({r["scorer"] for r in reports}) > 1:
        note = ("these families were judged by different scorers, so their numbers are not directly comparable; "
                "read each against its own baseline")
    return {"families": rows, "note": note}


def write_report(path, report, *, title="holdout evaluation"):
    """Write a small markdown report; the numbers are already in the caller's data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if "families" in report:
        lines += ["| family | scorer | images | max | mean | median |", "| --- | --- | --- | --- | --- | --- |"]
        for row in report["families"]:
            lines.append(f'| {row["family"]} | {row["scorer"]} | {row["images"]} | '
                         f'{_fmt(row["max"])} | {_fmt(row["mean"])} | {_fmt(row["median"])} |')
        if report.get("note"):
            lines += ["", f'> {report["note"]}']
    if "overall" in report:
        lines += ["", "## agreement with the human sheet", ""]
        for name in ("overall", "easy_views", "hard_views"):
            block = report.get(name) or {}
            lines.append(f'- **{name}**: {block.get("pairs", 0)} pair(s), '
                         f'top-quartile agreement {_fmt(block.get("top_quartile_agreement"))}, '
                         f'spearman {_fmt(block.get("spearman"))}')
            if block.get("rejected_but_scored_high"):
                lines.append(f'  - rejected but scored high: {", ".join(block["rejected_but_scored_high"])}')
            if block.get("high_human_low_auto"):
                lines.append(f'  - marked high but scored low: {", ".join(block["high_human_low_auto"])}')
        lines += ["", f'unpaired rows: {report.get("unpaired_count", 0)}']
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def holdout_prompts(project):
    """The holdout originals as prompts, so a final LoRA can be asked to draw what the set never saw.

    Each holdout gets its own caption, not the card's boilerplate: tool2 captioned the originals in its dataset, and
    where that caption exists for this image it is the verified wording (spec R5). Where none exists the card's caption
    stands in and the report says so - a prompt whose provenance is unknown should be visible as such, not silently
    blended with the verified ones.
    """
    from . import seed as seed_mod
    out = []
    holdout = project.holdout_entry()
    index = project.index()
    captions = seed_mod.tool2_captions((project.card.get("tool2") or {}).get("dataset"))
    for record_id in holdout.get("ids") or []:
        record = index.get(record_id) or {}
        name = record.get("file") or record.get("src_path") or record_id
        sha = record.get("sha256")
        own = captions.get(sha or "")
        if own:
            prompt, source = seed_mod.add_traits(own, project.card), "tool2's caption for the same image"
        else:
            prompt, source = seed_caption(project), "the card's caption (no verified caption for this image)"
        out.append({"id": record_id, "file": str(name), "sha256": sha, "prompt": prompt, "caption_source": source})
    return out


def seed_caption(project):
    """The card's caption under its `caption_traits` policy, built the way the seed and export build it."""
    from . import seed as seed_mod
    return seed_mod.add_traits(seed_mod.card_caption(project.card), project.card)


def holdout_scores(project, generated, *, family, scorer="clip_vision", embedder=None):
    """Score freshly drawn holdout pictures against the frozen holdout originals, for one family.

    `generated` is `[(id, path), ...]`. Only the holdout originals are references here - comparing a holdout draw with
    the whole training set would ask "does it look like the character" instead of "did it learn what it never saw".

    The reference set is resolved by purpose (`score.reference_paths(..., purpose="holdout")`), which returns exactly
    the frozen holdout ids. That is the only place in the product where those two pictures are a reference at all
    (spec N4.5): every other scoring path excludes them.
    """
    from . import score as score_mod
    from .scorers import clip_vision
    refs = score_mod.reference_paths(project, purpose="holdout")
    if not refs:
        raise EvalError("the holdout originals are not on disk, so the holdout cannot be scored")
    if not generated:
        raise EvalError("no holdout picture was drawn, so there is nothing to score")
    ids = [ident for ident, _path in generated]
    paths = [path for _ident, path in generated]
    if scorer == "clip_vision":
        scores = clip_vision.identity_scores(paths, refs, ids=ids, embedder=embedder)
    else:
        from .scorers import arcface
        scores = arcface.face_scores(paths, refs, ids=ids)
    return scores, refs, {"family": family}



def holdout_dir(project, family, *, purpose=None):
    """Where one purpose's holdout cells for one family live (spec N4.3).

    `seed` keeps the historical path `experiments/holdout/<family>/` so the existing 0.7696/0.6197 measurement stays
    where it is; the other purposes get their own folder, because a final LoRA's number and a baseline's number are
    different experiments and must not overwrite each other's cells.
    """
    base = project.dir / "experiments" / "holdout"
    if purpose in (None, "seed"):
        return base / family
    return base / purpose / family


def holdout_manifest_path(project, family, *, purpose=None):
    return holdout_dir(project, family, purpose=purpose) / "holdout.json"


def read_holdout_manifest(project, family, *, purpose=None):
    path = holdout_manifest_path(project, family, purpose=purpose)
    data = json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None
    return data if isinstance(data, dict) and data.get("cells") is not None else None


def write_holdout_manifest(project, family, manifest, *, purpose=None):
    from . import project as pj
    pj.write_json(holdout_manifest_path(project, family, purpose=purpose), manifest)


def _graph_version_for(family, purpose):
    """Which graph drew these cells, recorded by its own version string (spec N4.3)."""
    from . import krea2_gen
    if purpose == "baseline":
        return {"krea2": krea2_gen.GRAPH_VERSION["b"], "character_lora": None}
    return {"krea2": krea2_gen.GRAPH_VERSION["b"]}



def holdout_pin(project, *, purpose="seed", family="krea2", run=None, checkpoint=None):
    """The checkpoint a holdout draw is pinned to, for one purpose (spec N4.3/R5).

    A cell drawn with one LoRA is evidence about that LoRA and about nothing else (spec R5): the pin is what keeps a
    cell from sliding from one experiment into another model's measurement. `checkpoint_matches` travels with the pin,
    because a selection whose file changed underneath it is not the checkpoint the cells were drawn with either.

    The three purposes are kept apart because they answer different questions:

    * `seed` - the epoch chosen from the seed-run grid. What the M3-2 comparison picked, and what the existing
      `0.7696 vs 0.6197` measurement was made with. This is the only purpose that reads the card's choice.
    * `final` - one of the project's `final_families`, each trained from the same export. Requires an explicit
      `run` and `checkpoint`: a seed checkpoint must never stand in for a final LoRA, which is what silently
      substituting one would do.
    * `baseline` - the same prompts drawn with no character LoRA at all. There is no checkpoint; the pin is the fact
      that it had none, which is what makes the difference meaningful rather than a bare number.

    An unavailable `final` pin returns None and the caller names the family's waiting reason; it never falls back.
    """
    if purpose == "baseline":
        return {"purpose": "baseline", "family": family, "character_lora": None,
                "note": "drawn without the character LoRA, so this is the prompt-only reference"}
    if purpose == "final":
        if family not in (project.card.get("final_families") or []):
            return None
        if not run or not checkpoint:
            return None
        path = Path(checkpoint)
        if not path.is_file():
            return None
        from . import lineage
        return {"purpose": "final", "family": family, "run_id": Path(run).name,
                "checkpoint_path": str(path), "checkpoint_sha256": lineage.sha256_file(path),
                "checkpoint_matches": True,
                "character_lora": "the family's own final LoRA, trained on the chosen export"}
    from . import batch_gen
    chosen = batch_gen.epochs_mod.chosen(project)
    if not chosen:
        return None
    return {"purpose": "seed", "family": "krea2", "run_id": chosen.get("run_id"), "epoch": chosen.get("epoch"),
            "checkpoint_path": chosen.get("checkpoint_path"),
            "checkpoint_sha256": chosen.get("checkpoint_sha256"),
            "checkpoint_matches": chosen.get("checkpoint_matches")}


def holdout_waiting_reason(project, purpose, family):
    """Why a purpose/family combination cannot be measured yet, in words a person can act on (spec N4.3).

    Returns None only when the combination is supported; whether the *checkpoint* exists is a separate question the
    caller answers from `holdout_pin`.
    """
    if purpose not in HOLDOUT_PURPOSES:
        return f"unknown purpose {purpose!r}; the holdout is measured as one of {', '.join(HOLDOUT_PURPOSES)}"
    if purpose == "baseline":
        return None
    if purpose == "seed":
        if family != "krea2":
            return (f"the seed-run holdout is drawn with the krea2 epoch grid, so there is no {family} seed "
                    "measurement; use --purpose final with that family's run and checkpoint")
        return None
    families = project.card.get("final_families") or []
    if family not in families:
        return (f"{family} is not one of this project's final families "
                f"({', '.join(families) or 'none set'})")
    return None


def _missing_pin_reason(project, purpose, family, run, checkpoint):
    """The actionable message when the policy allows this measurement but no checkpoint can be pinned for it."""
    if purpose == "seed":
        return ("no epoch has been chosen for this project, so no holdout can be drawn; "
                "run `pipeline.epochs choose --epoch N` first")
    if purpose == "final":
        if not run or not checkpoint:
            return (f"a final {family} measurement needs that family's own LoRA: pass --run and --checkpoint. "
                    "A seed checkpoint is not a substitute for one trained on the final export")
        return (f"the checkpoint given for the final {family} measurement is not a readable file: {checkpoint}")
    return f"no checkpoint can be pinned for the {purpose}/{family} holdout"




def draw_holdout(project, *, family="krea2", seeds=(7001, 7002), seeds_always=True, client=None, log=print,
                 cancel=None, purpose="seed", run=None, checkpoint=None):
    """Draw the holdout prompts with one LoRA, for one family and purpose, under a pinned manifest.

    Every cell drawn is recorded in `experiments/holdout/<purpose>/<family>/holdout.json` with the checkpoint that drew
    it and the bytes it produced (spec R5, N4.3), so a rerun submits nothing new and a scoring pass can vouch for
    exactly the cells the pin names. The rules that keep it honest:

    * the same plan again submits nothing - a recorded cell whose bytes still hash to what was drawn is reused;
    * the chosen checkpoint changed since the manifest pinned it: refuse, rather than mixing cells drawn by two models;
    * files on disk that no manifest records are refused too - an unvouched picture must not enter a measurement, and
      the operator decides whether to move it aside;
    * the purpose decides which checkpoint may be used (spec N4.3). A `final` measurement needs that family's own run
      and checkpoint, an unavailable one returns a waiting reason, and a seed checkpoint never substitutes for it.
    """
    from . import batch_gen, krea2_gen, epochs as epochs_mod

    if purpose not in HOLDOUT_PURPOSES:
        raise EvalError(f"unknown purpose {purpose!r}; the holdout is measured as one of {', '.join(HOLDOUT_PURPOSES)}")
    if purpose == "seed" and family != "krea2":
        # The epoch grid draws with krea2 checkpoints, so there is no such thing as a qwen seed-run measurement.
        raise EvalError(holdout_waiting_reason(project, purpose, family))
    if purpose == "final" and family not in ("krea2", "qwen_image21"):
        raise EvalError(f"no inference path exists for {family!r}; the final holdout can be drawn for "
                        + ", ".join(project.card.get("final_families") or []))
    # Refusing stray cells comes first: it is a question about the folder, not about this project's choice, and the
    # operator needs to hear it even when nothing else is ready (spec R5).
    out_dir = holdout_dir(project, family, purpose=purpose)
    manifest = read_holdout_manifest(project, family, purpose=purpose)
    if manifest is None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stray = sorted(path.name for path in out_dir.glob("*.png"))
        if stray:
            raise EvalError(f"{len(stray)} image(s) sit in {out_dir} that no holdout manifest records "
                            f"({', '.join(stray[:3])}...); move them aside, then draw - an unvouched picture "
                            "cannot say which checkpoint drew it")
    pin = holdout_pin(project, purpose=purpose, family=family, run=run, checkpoint=checkpoint)
    if pin is None:
        raise EvalError(_missing_pin_reason(project, purpose, family, run, checkpoint))
    prompts = holdout_prompts(project)
    if not prompts:
        raise EvalError("no holdout is frozen for this project")
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (project.card.get("krea2_inference") or {}).get("size") or {}
    width = int(size.get("width") or 1024)
    height = int(size.get("height") or 1536)

    if manifest is None:
        manifest = {"kind": "holdout", "family": family, "purpose": purpose, "created_at": _now_stamp(),
                    "pin": pin, "prompts": prompts, "seeds": list(seeds),
                    "size": {"width": width, "height": height},
                    "graph_version": _graph_version_for(family, purpose), "cells": []}
    else:
        recorded_pin = manifest.get("pin") or {}
        if str(recorded_pin.get("purpose") or "seed") != purpose:
            raise EvalError(f"the cells in {out_dir} were drawn for the {recorded_pin.get('purpose')!r} purpose, "
                            f"not {purpose!r}; they cannot share a measurement")
        recorded = str(recorded_pin.get("checkpoint_sha256") or "")
        current = str(pin.get("checkpoint_sha256") or "")
        if current and recorded and recorded != current:
            raise EvalError(f"the holdout cells were drawn with checkpoint {recorded[:12]}, but this draw pins "
                            f"{current[:12]}; the two cannot share a measurement. Move the folder aside "
                            "or restore the choice, then draw again")
        if pin.get("checkpoint_matches") is False:
            raise EvalError(f"the pinned checkpoint's file changed since it was chosen "
                            f"({pin.get('checkpoint_path')}); the holdout cells were drawn with the model as it "
                            "was, so the two cannot share a measurement")
        prompts = manifest.get("prompts") or prompts
        seeds = tuple(manifest.get("seeds") or seeds)


    recorded = {(cell.get("id"), int(cell.get("seed") or 0)): cell for cell in manifest["cells"]}
    drawn, skipped, failed = [], 0, []

    def cell_path(row, seed):
        return out_dir / f'{row["id"]}__s{seed}.png'

    def reusable(path, cell):
        """Is this file a cell this manifest already vouches for? (spec N4.2)

        `cell` may be None - a file on disk with no manifest entry - which used to crash on `cell.get()` and now simply
        means "not vouched for". A recorded cell must still decode, still hash to what was drawn, and still be pinned to
        this plan; anything else is drawn again rather than adopted, because an orphan picture in the holdout folder
        cannot say which checkpoint drew it.
        """
        if not path.is_file() or not lineage_image_ok(path):
            return False
        if not isinstance(cell, dict):
            return False
        recorded_sha = cell.get("sha256")
        if not recorded_sha:
            return False
        return lineage.sha256_file(path) == recorded_sha

    from . import lineage
    pending = []
    for row in prompts:
        for seed in seeds:
            out = cell_path(row, seed)
            cell = recorded.get((row["id"], int(seed)))
            if reusable(out, cell):
                drawn.append((f'{row["id"]}__s{seed}', out))
                skipped += 1
                continue
            pending.append((row, seed, out, cell))
    if not pending:
        write_holdout_manifest(project, family, manifest)
        return drawn, skipped, failed

    # Something must actually be drawn: only now is a client contacted and the pinned checkpoint required.
    client = client or epochs_mod.make_client(None)
    names = names_for_holdout(project, client, family)
    for row, seed, out, _cell in pending:
        if cancel is not None and cancel():
            log("cancelled")
            write_holdout_manifest(project, family, manifest)
            return drawn, skipped, failed
        try:
            graph = krea2_gen.build(
                "b", unet=names["b_unet"], clip=names["clip"], vae=names["vae"], prompt=row["prompt"],
                width=width, height=height, seed=seed,
                character_lora=names["character_lora"], speed_lora=names["speed_lora"],
                filename_prefix=f"pipeline_holdout/{row['id']}_s{seed}",
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            client.run(graph, out, slot_id=row["id"], method="krea2_b_holdout", seed=seed)
            manifest["cells"] = [c for c in manifest["cells"]
                                 if not (c.get("id") == row["id"] and int(c.get("seed") or 0) == int(seed))]
            manifest["cells"].append({"id": row["id"], "seed": int(seed), "file": out.name,
                                      "sha256": lineage.sha256_file(out), "status": "done"})
            write_holdout_manifest(project, family, manifest)
            drawn.append((f'{row["id"]}__s{seed}', out))
            log(f'  {row["id"]} s{seed}: drawn')
        except Exception as exc:  # noqa: BLE001 - one failed cell must not lose the others
            failed.append({"id": row["id"], "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
            log(f'  {row["id"]} s{seed}: FAILED {exc}')
    write_holdout_manifest(project, family, manifest)
    return drawn, skipped, failed


def _now_stamp():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def lineage_image_ok(path):
    from . import lineage
    return lineage.image_decodes(path)


def pixel_hash_of(path):
    """Decoded-pixel identity, or None when the file does not decode (spec N4.1)."""
    from . import lineage
    return lineage.pixel_hash(path)


def names_for_holdout(project, client, family):
    """The model names one holdout draw needs, resolved once per call."""
    from . import batch_gen, epochs as epochs_mod, models
    client = client or epochs_mod.make_client(None)
    if not client.ping():
        raise EvalError(f"ComfyUI is not answering at {client.url}")
    resolved = models.resolve("krea2")
    lora_path = batch_gen.character_lora_path(project)
    return {
        "b_unet": client.name_for("unet", resolved["krea2_raw_dit"]),
        "speed_lora": client.name_for("lora", resolved["krea2_turbo_lora"]),
        "clip": client.name_for("clip", resolved["krea2_text_encoder"]),
        "vae": client.name_for("vae", resolved["krea2_vae"]),
        "character_lora": client.name_for("lora", str(lora_path)),
    }


def holdout_cells(project, family, *, log=print, purpose=None):
    """The cells a scoring pass may read, from the manifest, with everything else named apart (spec R5, N4.1/N4.2).

    Returns `(pairs, audit)`. `pairs` are `(id__seed, path)` for cells the plan expected, the manifest recorded, that
    still decode, whose bytes are still what was drawn, and that are not a pixel duplicate of another cell.

    `audit` names everything else, and the distinction between "the plan never got this cell" and "the folder holds a
    file nobody recorded" is the point (spec N4.1):

    * `expected` - how many cells the manifest's own prompts x seeds say there should be;
    * `missing` - expected, unrecorded or recorded-but-absent;
    * `no_sha` - recorded without a SHA, which cannot be vouched for and is therefore not scored;
    * `undecoded`, `changed` - present but not evidence;
    * `unlisted` - files on disk the manifest does not name (spec N4.2: never silently trusted);
    * `duplicate_pixels` - cells whose decoded pixels are identical, so one picture is one measurement.

    A partial draw is never reported as a whole one: `expected` vs `scored` is in the audit, and the caller prints it.
    """
    from . import lineage
    manifest = read_holdout_manifest(project, family, purpose=purpose)
    if manifest is None:
        raise EvalError(f"no holdout manifest records what drew the cells in "
                        f"{holdout_dir(project, family, purpose=purpose)}; "
                        "score only what a draw pinned, or draw again")
    out_dir = holdout_dir(project, family, purpose=purpose)
    prompts = manifest.get("prompts") or []
    seeds = [int(s) for s in (manifest.get("seeds") or [])]
    expected_keys = {(str(row.get("id")), seed) for row in prompts for seed in seeds}
    pairs, missing, undecoded, changed, no_sha = [], [], [], [], []
    seen_pixels = {}
    recorded_keys = set()
    for cell in manifest.get("cells") or []:
        key = (str(cell.get("id")), int(cell.get("seed") or 0))
        recorded_keys.add(key)
        name = str(cell.get("file") or "")
        path = out_dir / name
        if not path.is_file():
            missing.append(name or f"{key[0]}_s{key[1]}")
            continue
        if not lineage.image_decodes(path):
            undecoded.append(name)
            continue
        if not cell.get("sha256"):
            # spec N4.1: a cell with no recorded SHA cannot be distinguished from a file somebody dropped in the
            # folder, so it is reported rather than scored. The old reader accepted it (`missing_sha_accepted`).
            no_sha.append(name)
            continue
        actual = lineage.sha256_file(path)
        if actual != cell["sha256"]:
            changed.append(name)
            continue
        pixels = pixel_hash_of(path)
        if pixels in seen_pixels:
            # Two files, one picture. Counting it twice would weight one drawing twice in the mean.
            continue
        seen_pixels[pixels] = name
        pairs.append((f'{cell.get("id")}__s{cell.get("seed")}', path))
    for key in sorted(expected_keys - recorded_keys):
        missing.append(f"{key[0]}_s{key[1]} (never recorded)")
    listed = {str(cell.get("file") or "") for cell in manifest.get("cells") or []}
    unlisted = sorted(path.name for path in out_dir.glob("*.png") if path.name not in listed)
    duplicate_files = []
    counts = {}
    for cell in manifest.get("cells") or []:
        path = out_dir / str(cell.get("file") or "")
        if not path.is_file():
            continue
        pixels = pixel_hash_of(path)
        if pixels is None:
            continue
        counts.setdefault(pixels, []).append(path.name)
    for names in counts.values():
        if len(names) > 1:
            # The first file of a pixel-identical group is the measurement; the rest are the duplicates reported here.
            duplicate_files.extend(sorted(names)[1:])
    audit = {"missing": sorted(missing), "undecoded": undecoded, "changed": changed,
             "no_sha": no_sha, "unlisted": unlisted, "duplicate_pixels": sorted(duplicate_files),
             "expected": len(expected_keys), "recorded": len(manifest.get("cells") or []),
             "scored": len(pairs),
             "complete": not (missing or undecoded or changed or no_sha or unlisted),
             "pin": manifest.get("pin")}
    return pairs, audit


def _fmt(value):
    return "—" if value is None else f"{value:.4f}"


def main(argv=None):
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="pipeline.evaluate", description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("holdout", "agreement"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--scores", help="path to the M0 scores.csv for the agreement report")
    parser.add_argument("--automatic", help="path to a scores.json from pipeline.score")
    parser.add_argument("--experiment", help="path to the experiment.json that drew the scored images "
                                             "(default: next to the scores file)")
    parser.add_argument("--score-field", default="clip_max_cosine",
                        help="which automatic number to compare (clip_max_cosine, arcface_max_cosine, ...)")
    parser.add_argument("--family", default="krea2", help="which family to draw and score (holdout)")
    parser.add_argument("--purpose", default="seed", choices=HOLDOUT_PURPOSES,
                        help="what the measurement is for: seed (the M3-2 choice), final (a family's own final LoRA) "
                             "or baseline (no character LoRA at all)")
    parser.add_argument("--run", help="with --purpose final: the run folder that trained this family's LoRA")
    parser.add_argument("--checkpoint", help="with --purpose final: the .safetensors that should be measured")
    parser.add_argument("--seeds", default="7001,7002", help="comma-separated seeds (holdout)")
    parser.add_argument("--draw", action="store_true", help="really draw the holdout prompts (needs ComfyUI)")
    parser.add_argument("--score", action="store_true", help="score what was drawn against the holdout originals")
    parser.add_argument("--comfy", default=None)
    parser.add_argument("--out", help="where to write the report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from . import project as pj
    project = pj.Project.open(args.project)

    if args.command == "holdout":
        prompts = holdout_prompts(project)
        if args.json and not (args.draw or args.score):
            print(json.dumps(prompts, ensure_ascii=False, indent=1))
            return 0
        if not prompts:
            print("no holdout is frozen for this project")
            return 0

        drawn, skipped, failed = [], 0, []
        if args.draw:
            seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
            from . import gpu
            claim = gpu.claim_generation(what="the holdout draw")
            try:
                with claim:
                    drawn, skipped, failed = draw_holdout(project, family=args.family, seeds=seeds,
                                                          client=None if not args.comfy else _client(args.comfy),
                                                          purpose=args.purpose, run=args.run,
                                                          checkpoint=args.checkpoint)
            finally:
                claim.release()
            print(f"holdout draw ({args.purpose}/{args.family}): {len(drawn)} image(s), {skipped} already present, "
                  f"{len(failed)} failed")
            for row in failed:
                print(f'  FAILED {row["id"]} s{row["seed"]}: {row["error"][:160]}')
            if failed and not drawn:
                return 1
        if not args.score:
            if not args.draw:
                waiting = holdout_waiting_reason(project, args.purpose, args.family)
                if waiting:
                    print(f"{args.purpose}/{args.family}: {waiting}")
                    return 1
                print(f"{len(prompts)} holdout original(s); draw these prompts with the {args.purpose} LoRA of each "
                      "family:")
                for row in prompts:
                    print(f'  {row["id"]}  {Path(row["file"]).name}')
                print(f'  prompt: {prompts[0]["prompt"]}')
                print("\nadd --draw to draw them, or --score to score what is already drawn")
            return 0

        folder = holdout_dir(project, args.family, purpose=args.purpose)
        if not holdout_manifest_path(project, args.family, purpose=args.purpose).is_file():
            raise EvalError(f"no holdout manifest in {folder} records what drew the cells there; "
                            "score only what a draw pinned - run the draw first")
        pairs, audit = holdout_cells(project, args.family, purpose=args.purpose)
        if not pairs:
            raise EvalError(f"none of the recorded cells in {folder} can be vouched for: "
                            + "; ".join(f"{k} {len(v)}" for k, v in audit.items() if isinstance(v, list) and v))
        scores, refs, _meta = holdout_scores(project, pairs, family=args.family)
        report = holdout_family_report(args.family, scores, references=refs, cells_audit=audit,
                                       purpose=args.purpose)
        print(f'{args.purpose}/{args.family}: {report["images"]} image(s) scored, max {_fmt(report["max"])}, '
              f'mean {_fmt(report["mean"])}, median {_fmt(report["median"])}')
        # A measurement over part of the plan is printed as such, not as a number that looks whole (spec N4.1).
        if audit.get("expected") and audit.get("scored") != audit.get("expected"):
            print(f'  cells: {audit.get("scored")}/{audit.get("expected")} of the plan were scored'
                  + ("" if audit.get("complete") else " - THIS IS A PARTIAL MEASUREMENT"))
        audit_bits = ", ".join(f"{name} {len(values)}" for name, values in
                              ((k, audit.get(k)) for k in ("missing", "undecoded", "changed", "no_sha", "unlisted",
                                                           "duplicate_pixels")) if values)
        if audit_bits:
            print(f"  cells apart from the measurement: {audit_bits}")
        if report["unscored"]:
            print(f'  {len(report["unscored"])} image(s) carry no score: {", ".join(report["unscored"][:5])}')
        if args.out:
            path = write_report(args.out, {"families": [report]},
                                title=f"{args.purpose}/{args.family} holdout evaluation")
            print(f"wrote {path}")

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0

    if not args.scores or not args.automatic:
        raise EvalError("the agreement report needs --scores (a scores.csv) and --automatic (a scores.json)")
    rows = read_scores(args.scores)
    automatic = json.loads(Path(args.automatic).read_text(encoding="utf-8"))
    automatic_rows = automatic.get("rows") or []
    manifest_path = Path(args.experiment) if args.experiment else Path(args.scores).parent / "experiment.json"
    if manifest_path.is_file():
        # The sheet is joined to the automatic scores through the manifest that recorded what each cell drew
        # (spec R5): the same picture, proven by sha256, never a name match.
        tasks, experiment_dir = load_experiment_tasks(manifest_path)
        report = agreement_from_manifest(rows, automatic_rows, tasks, project, experiment_dir=experiment_dir,
                                         score_field=args.score_field)
    else:
        report = agreement(rows, {row["id"]: row.get(args.score_field) for row in automatic_rows},
                           score_field=args.score_field)
        report["matching"] = {"basis": "candidate id, without an experiment manifest",
                              "score_field": args.score_field,
                              "unmatched": {"no automatic score": report["unpaired_count"]}}
    if args.json:
        # The JSON is the whole report and nothing else: a caller parses this stream, and the pairs carry each row's
        # match basis, so they stay in.
        print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0
    matching = report.get("matching") or {}
    print(f'paired {report["paired"]} of {report["paired"] + report["unpaired_count"]} sheet row(s) '
          f'via {matching.get("basis")}')
    for reason, count in (matching.get("unmatched") or {}).items():
        print(f"  unmatched: {reason} x{count}")
    if args.out:
        path = write_report(args.out, report, title="automatic scores against the human sheet")
        print(f"wrote {path}")
    return 0


def _client(url):
    from . import epochs as epochs_mod
    return epochs_mod.make_client(url)


if __name__ == "__main__":
    import sys

    sys.exit(main())
