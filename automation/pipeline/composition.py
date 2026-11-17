"""The composition plan: a dataset of `target_images` pictures, half body shots and half face shots, with the shot size,
view, pose and expression spread so that no combination repeats.

A card opts in with `"slot_plan": "composition"`; the plan then replaces `slot_quotas` everywhere a slot board is read
(`Project.slot_plan`). Each cell is one slot with a target of one picture.

* body half: full body, cowboy shot, upper body, weighted 2:1:1 toward full body (SHOT_WEIGHT) - the whole figure is
  what a character LoRA learns proportions from, and the first suamom batch had too few.   face half: bust shot,
  portrait, close-up, evenly.
* views front / three-quarter / side profile / back (VIEW_WEIGHT). Side profiles get as much room as the front views
  on body shots and nearly as much on face shots (the first batch was short of them). A close-up takes no back view
  (it turns the face to the front); a portrait takes no back view either; a body shot from behind keeps a neutral
  face.
* poses standing / sitting / walking / dynamic only on body shots; expressions neutral / smiling / angry / sad /
  surprised spread evenly over the whole set.
* the originals that enter training (user originals, holdout excluded) fill their own cells first. Their shot, view,
  pose and expression are read from tool2's `original_tags` for the character; only the rest become cells, and the
  cells lean toward what the originals lack, so the totals - originals plus cells - come out balanced.

The plan is a pure function of the target and the originals: the same project plans the same cells, with the same ids,
every time. A cell id spells out its combination (`cmp_full_quarter_walking_angry`), so a person reading a slot board
knows what the slot asks for without a lookup.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import settings

SHOTS = {"body": ("full body", "cowboy shot", "upper body"), "face": ("bust shot", "portrait", "close-up")}
VIEWS = ("front view", "three-quarter view", "side profile", "back view")
SHOT_WEIGHT = {"body": (2, 1, 1), "face": (1, 1, 1)}
VIEW_WEIGHT = {"body": (6, 6, 7, 5), "face": (8, 8, 7, 2)}
ALLOWED_VIEWS = {"close-up": VIEWS[:3], "portrait": VIEWS[:3]}
POSES = ("standing", "sitting", "walking", "dynamic pose")
EXPRESSIONS = ("neutral", "smiling", "angry", "sad", "surprised")

CODE = {"full body": "full", "cowboy shot": "cowboy", "upper body": "upper", "bust shot": "bust",
        "portrait": "portrait", "close-up": "closeup", "front view": "front", "three-quarter view": "quarter",
        "side profile": "side", "back view": "back", "standing": "standing", "sitting": "sitting",
        "walking": "walking", "dynamic pose": "dynamic", "neutral": "neutral", "smiling": "smiling",
        "angry": "angry", "sad": "sad", "surprised": "surprised"}

# The words each axis is asked for, in the tag vocabulary the training captions use. Full body and side profile carry
# extra plain words: "full body" alone came back cropped at the knees when sitting, and "from side, profile" alone came
# back as a three-quarter view.
SHOT_TAGS = {"full body": "full body, whole body from head to toe, feet and shoes visible",
             "cowboy shot": "cowboy shot", "upper body": "upper body",
             "bust shot": "upper body, bust", "portrait": "portrait", "close-up": "close-up, face focus"}
VIEW_TAGS = {"front view": "front view, looking at viewer", "three-quarter view": "three-quarter view",
             "side profile": "from side, side view, profile, facing sideways, not looking at viewer",
             "back view": "from behind"}
BACK_FACE_TAGS = "from behind, looking back"
POSE_TAGS = {"standing": "standing", "sitting": "sitting", "walking": "walking", "dynamic pose": "dynamic pose"}
EXPRESSION_TAGS = {"neutral": "closed mouth", "smiling": "smile", "angry": "angry, frown",
                   "sad": "sad, teary eyes", "surprised": "surprised, open mouth"}
PLAIN_BACKGROUND = "white background, simple background"
BACKGROUNDS = ("living room, indoors, day", "park, cherry blossoms, outdoors, day", "cafe, indoors, day",
               "city street, night, neon lights", "kitchen, indoors, morning", "riverside, golden hour, outdoors")


class CompositionError(ValueError):
    """A plan that cannot be made: a target too small for the originals, or an unknown slot."""


def split(total, weights):
    """Largest-remainder split of `total` by `weights`."""
    weights = [max(0.0, float(w)) for w in weights]
    if total <= 0 or not sum(weights):
        return [0] * len(weights)
    raw = [total * w / sum(weights) for w in weights]
    out = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: (out[i] - raw[i], i))[:total - sum(out)]:
        out[i] += 1
    return out


def classify(tags):
    """{half, shot, view, pose, expression} for an original, read from tool2's `original_tags` words."""
    shot_words = str((tags or {}).get("shot", ""))
    for word, shot in (("full body", "full body"), ("cowboy", "cowboy shot"), ("upper body", "upper body"),
                       ("bust", "bust shot"), ("portrait", "portrait"), ("close-up", "close-up")):
        if word in shot_words:
            break
    else:
        shot = "full body"            # crouching, kneeling, no size named: the whole figure is in the picture
    if "from behind" in shot_words:
        view = "back view"
    elif "from side" in shot_words or "profile" in shot_words:
        view = "side profile"
    elif "three-quarter" in shot_words:
        view = "three-quarter view"
    else:
        view = "front view"
    if "sitting" in shot_words:
        pose = "sitting"
    elif "walking" in shot_words:
        pose = "walking"
    elif any(w in shot_words for w in ("crouching", "on one knee", "reaching", "running", "jumping")):
        pose = "dynamic pose"
    else:
        pose = "standing"
    expr_words = str((tags or {}).get("expr", ""))
    if "smile" in expr_words:
        expression = "smiling"
    elif "angry" in expr_words:
        expression = "angry"
    elif any(w in expr_words for w in ("sad", "worried", "crying")):
        expression = "sad"
    elif "surprised" in expr_words:
        expression = "surprised"
    else:
        expression = "neutral"
    half = "body" if shot in SHOTS["body"] else "face"
    return {"half": half, "shot": shot, "view": view, "pose": pose if half == "body" else "",
            "expression": expression}


def tool2_tags(card):
    """{original file name: tags} from the tool2 config the card names, or {} when it cannot be read."""
    block = card.get("tool2") or {}
    try:
        jobs = json.loads(Path(block.get("config") or "").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    for char in jobs.get("characters") or []:
        if char.get("folder") == block.get("character"):
            return dict(char.get("original_tags") or {})
    return {}


def originals(project):
    """The originals that will be in training, classified: user originals, holdout refused by id and by sha."""
    holdout = project.holdout_entry()
    ids, shas = set(holdout.get("ids") or []), set(holdout.get("shas") or [])
    tags = tool2_tags(project.card)
    out = []
    for record in project.source_records():
        if record.get("origin") != "user" or record["id"] in ids or record.get("sha256") in shas:
            continue
        name = Path(str(record.get("src_path") or record.get("file") or "")).name
        out.append({"id": record["id"], "name": name, "tagged": name in tags, **classify(tags.get(name))})
    return out


def plan_cells(have, target):
    """The cells to draw for `target` pictures when the originals `have` are already in the set."""
    target = int(target)
    half_target = {"body": target - target // 2, "face": target // 2}
    cells = []
    expr_need = Counter(dict(zip(EXPRESSIONS, split(target, [1] * len(EXPRESSIONS)))))
    expr_need.subtract(Counter(o["expression"] for o in have))
    for half in ("body", "face"):
        mine = [o for o in have if o["half"] == half]
        gap = max(0, half_target[half] - len(mine))
        shots = SHOTS[half]
        shot_need = Counter(dict(zip(shots, split(half_target[half], SHOT_WEIGHT[half]))))
        shot_need.subtract(Counter(o["shot"] for o in mine))
        view_need = Counter(dict(zip(VIEWS, split(half_target[half], VIEW_WEIGHT[half]))))
        view_need.subtract(Counter(o["view"] for o in mine))
        pose_need = None
        if half == "body":
            pose_need = Counter(dict(zip(POSES, split(half_target[half], [1] * len(POSES)))))
            pose_need.subtract(Counter(o["pose"] for o in mine))
        shot_list = []
        for _ in range(gap):
            shot = max(shots, key=lambda s: (shot_need[s], -shots.index(s)))
            shot_need[shot] -= 1
            shot_list.append(shot)
        shot_list.sort(key=shots.index)
        used = Counter((o["shot"], o["view"]) for o in mine)
        for shot in shot_list:
            allowed = ALLOWED_VIEWS.get(shot, VIEWS)
            view = max(allowed, key=lambda v: (view_need[v] - 2 * used[(shot, v)], -VIEWS.index(v)))
            view_need[view] -= 1
            used[(shot, view)] += 1
            pose = ""
            if pose_need is not None:
                pose = max(POSES, key=lambda p: (pose_need[p], -POSES.index(p)))
                if shot == "upper body" and pose == "walking" and pose_need["standing"] > 0:
                    pose = "standing"
                pose_need[pose] -= 1
            cells.append({"half": half, "shot": shot, "view": view, "pose": pose})
    seen = Counter((o["shot"], o["view"], o["expression"]) for o in have)
    for cell in cells:
        options = ("neutral",) if cell["view"] == "back view" and cell["half"] == "body" else EXPRESSIONS
        expression = max(options, key=lambda e: (expr_need[e] - 3 * seen[(cell["shot"], cell["view"], e)],
                                                 -EXPRESSIONS.index(e)))
        expr_need[expression] -= 1
        seen[(cell["shot"], cell["view"], expression)] += 1
        cell["expression"] = expression
    taken = Counter()
    for index, cell in enumerate(cells):
        base = "cmp_" + "_".join(CODE[cell[k]] for k in ("shot", "view", "pose", "expression") if cell[k])
        taken[base] += 1
        cell["slot_id"] = base if taken[base] == 1 else f"{base}_{taken[base]}"
        plain = cell["shot"] == "close-up" or index % 5 in (0, 3)
        cell["background"] = PLAIN_BACKGROUND if plain else BACKGROUNDS[index % len(BACKGROUNDS)]
        cell["tags"] = tags_for(cell)
    return cells


def tags_for(cell):
    """The words a cell asks for: shot, view, pose, expression, background."""
    face = cell["half"] == "face"
    view = BACK_FACE_TAGS if face and cell["view"] == "back view" else VIEW_TAGS[cell["view"]]
    expression = "" if cell["view"] == "back view" and not face else EXPRESSION_TAGS[cell["expression"]]
    parts = (SHOT_TAGS[cell["shot"]], view, POSE_TAGS.get(cell["pose"], ""), expression, cell.get("background", ""))
    return ", ".join(p for p in parts if p)


def plan(project, target=None):
    """{"target", "originals", "cells", "totals"} for the project; `target` defaults to its settings.json."""
    target = int(target or settings.read(project)["target_images"])
    have = originals(project)
    if len(have) > target:
        raise CompositionError(f"{len(have)} originals already exceed the target of {target} pictures")
    cells = plan_cells(have, target)
    every = have + cells
    totals = {key: dict(Counter(c[key] for c in every)) for key in ("half", "shot", "view", "pose", "expression")}
    return {"target": target, "originals": have, "cells": cells, "totals": totals}


def enabled(card):
    return str(card.get("slot_plan") or "") == "composition"


def quotas(project, target=None):
    """{slot_id: 1} for every cell of the plan."""
    return {cell["slot_id"]: 1 for cell in plan(project, target)["cells"]}


def cell(project, slot_id, target=None):
    for row in plan(project, target)["cells"]:
        if row["slot_id"] == slot_id:
            return row
    raise CompositionError(f"slot {slot_id!r} is not a cell of this project's composition plan")


def is_cell(slot_id):
    return str(slot_id).startswith("cmp_")
