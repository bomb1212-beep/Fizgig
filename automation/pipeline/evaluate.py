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
import statistics
from pathlib import Path

# Views where a whole-frame embedding is known to be a weak proxy: the face is small, turned away, or absent. Counting
# them in the overall agreement would flatter or punish the scorer for something it cannot see, so they are also
# reported on their own.
HARD_VIEW_SLOTS = ("ang_back", "ang_side", "frm_back", "frm_profile", "from_above", "dutch")


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


def holdout_family_report(family, scores, *, references):
    """One family's holdout result, with its references named.

    `scores` is a scorer result whose rows carry `max_cosine`; `references` are the holdout originals it was compared
    against. Reported per family because a number from one family is not evidence about another.
    """
    if not references:
        raise EvalError(f"the holdout evaluation for {family} has no references to compare against")
    values = [row["max_cosine"] for row in scores["rows"] if row.get("max_cosine") is not None]
    missing = [row["id"] for row in scores["rows"] if row.get("max_cosine") is None]
    return {
        "family": family,
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

    The prompt for a holdout is the card's caption plus the traits under the project's caption policy - the same wording
    an export uses - because the point is to compare the LoRA's output with the *original* under the same description.
    """
    out = []
    holdout = project.holdout_entry()
    review_of = project.index()
    for record_id in holdout.get("ids") or []:
        record = review_of.get(record_id) or {}
        name = record.get("file") or record.get("src_path") or record_id
        out.append({"id": record_id, "file": str(name),
                    "sha256": record.get("sha256"),
                    "prompt": seed_caption(project)})
    return out


def seed_caption(project):
    """The card's caption under its `caption_traits` policy, built the way the seed and export build it."""
    from . import seed as seed_mod
    return seed_mod.add_traits(seed_mod.card_caption(project.card), project.card)


def holdout_scores(project, generated, *, family, scorer="clip_vision", embedder=None):
    """Score freshly drawn holdout pictures against the frozen holdout originals, for one family.

    `generated` is `[(id, path), ...]`. Only the holdout originals are references here - comparing a holdout draw with
    the whole training set would ask "does it look like the character" instead of "did it learn what it never saw".
    """
    from .scorers import clip_vision
    holdout = project.holdout_entry()
    refs = []
    for record_id in holdout.get("ids") or []:
        record = project.index().get(record_id) or {}
        path = project.resolve_file(record.get("file") or record.get("src_path"))
        if path and Path(path).is_file():
            refs.append(str(path))
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


def holdout_dir(project, family):
    return project.dir / "experiments" / "holdout" / family


def draw_holdout(project, *, family="krea2", seeds=(7001, 7002), seeds_always=True, client=None, log=print,
                 cancel=None):
    """Draw the holdout prompts with the chosen final LoRA, for one family.

    This is the half the CLI used to leave to the operator. It reuses the same graph the epoch grid uses, so a holdout
    picture is drawn exactly the way a training-set picture is - the only difference is that the prompt was never
    trained on. A cell that is already on disk is kept, so a rerun after a crash does not redraw what exists.
    """
    from . import batch_gen, krea2_gen, epochs as epochs_mod, models

    if family != "krea2":
        raise EvalError(f"the holdout draw is only wired for krea2 so far; asked for {family!r}")
    prompts = holdout_prompts(project)
    if not prompts:
        raise EvalError("no holdout is frozen for this project")
    out_dir = holdout_dir(project, family)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = (project.card.get("krea2_inference") or {}).get("size") or {}
    width = int(size.get("width") or 1024)
    height = int(size.get("height") or 1536)

    drawn, skipped, failed = [], 0, []

    def _existing():
        found = []
        for row in prompts:
            for seed in seeds:
                out = out_dir / f'{row["id"]}__s{seed}.png'
                if out.is_file():
                    found.append((f'{row["id"]}__s{seed}', out))
        return found

    # Everything asked for may already be on disk; then no LoRA is looked up at all, so a rerun after choosing a
    # different epoch cannot be blocked by a project that no longer has one.
    present = _existing()
    if len(present) == len(prompts) * len(seeds):
        return present, len(present), failed

    lora_path = batch_gen.character_lora_path(project)
    client = client or epochs_mod.make_client(None)
    if not client.ping():
        raise EvalError(f"ComfyUI is not answering at {client.url}")

    resolved = models.resolve(family)
    names = {
        "b_unet": client.name_for("unet", resolved["krea2_raw_dit"]),
        "speed_lora": client.name_for("lora", resolved["krea2_turbo_lora"]),
        "clip": client.name_for("clip", resolved["krea2_text_encoder"]),
        "vae": client.name_for("vae", resolved["krea2_vae"]),
        "character_lora": client.name_for("lora", str(lora_path)),
    }

    for row in prompts:
        for seed in seeds:
            out = out_dir / f'{row["id"]}__s{seed}.png'
            if out.is_file():
                drawn.append((f'{row["id"]}__s{seed}', out))
                skipped += 1
                continue
            if cancel is not None and cancel():
                log("cancelled")
                return drawn, skipped, failed
            try:
                graph = krea2_gen.build(
                    "b", unet=names["b_unet"], clip=names["clip"], vae=names["vae"], prompt=row["prompt"],
                    width=width, height=height, seed=seed, character_lora=names["character_lora"],
                    speed_lora=names["speed_lora"],
                    filename_prefix=f"pipeline_holdout/{row['id']}_s{seed}",
                )
                client.run(graph, out, slot_id=row["id"], method="krea2_b_holdout", seed=seed)
                drawn.append((f'{row["id"]}__s{seed}', out))
                log(f'  {row["id"]} s{seed}: drawn')
            except Exception as exc:  # noqa: BLE001 - one failed cell must not lose the others
                failed.append({"id": row["id"], "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
                log(f'  {row["id"]} s{seed}: FAILED {exc}')
    return drawn, skipped, failed


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
    parser.add_argument("--family", default="krea2", help="which family to draw and score (holdout)")
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
            drawn, skipped, failed = draw_holdout(project, family=args.family, seeds=seeds,
                                                  client=None if not args.comfy else _client(args.comfy))
            print(f"holdout draw: {len(drawn)} image(s), {skipped} already present, {len(failed)} failed")
            for row in failed:
                print(f'  FAILED {row["id"]} s{row["seed"]}: {row["error"][:160]}')
            if failed and not drawn:
                return 1
        if not args.score:
            if not args.draw:
                print(f"{len(prompts)} holdout original(s); draw these prompts with the final LoRA of each family:")
                for row in prompts:
                    print(f'  {row["id"]}  {Path(row["file"]).name}')
                print(f'  prompt: {prompts[0]["prompt"]}')
                print("\nadd --draw to draw them, or --score to score what is already drawn")
            return 0

        folder = holdout_dir(project, args.family)
        on_disk = sorted(folder.glob("*.png")) if folder.is_dir() else []
        pairs = [(path.stem, path) for path in on_disk]
        if not pairs:
            raise EvalError(f"nothing drawn in {folder}; run again with --draw")
        scores, refs, _meta = holdout_scores(project, pairs, family=args.family)
        report = holdout_family_report(args.family, scores, references=refs)
        print(f'{args.family}: {report["images"]} image(s) scored, max {_fmt(report["max"])}, '
              f'mean {_fmt(report["mean"])}, median {_fmt(report["median"])}')
        if report["unscored"]:
            print(f'  {len(report["unscored"])} image(s) carry no score: {", ".join(report["unscored"][:5])}')
        if args.out:
            path = write_report(args.out, {"families": [report]}, title=f"{args.family} holdout evaluation")
            print(f"wrote {path}")
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        return 0

    if not args.scores or not args.automatic:
        raise EvalError("the agreement report needs --scores (a scores.csv) and --automatic (a scores.json)")
    rows = read_scores(args.scores)
    automatic = json.loads(Path(args.automatic).read_text(encoding="utf-8"))
    table = {row["id"]: row.get("clip_max_cosine") for row in automatic.get("rows") or []}
    report = agreement(rows, table)
    if args.out:
        path = write_report(args.out, report, title="automatic scores against the human sheet")
        print(f"wrote {path}")
    if args.json:
        print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, ensure_ascii=False, indent=1))
    return 0


def _client(url):
    from . import epochs as epochs_mod
    return epochs_mod.make_client(url)


if __name__ == "__main__":
    import sys

    sys.exit(main())
