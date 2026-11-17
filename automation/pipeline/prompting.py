"""The words every Krea2 picture of the character is asked for: one builder for batch B, the epoch grid and M0.

The order is spec 3.4's, with the character's profile right after the style, before the slot's own words:

    trigger, caption_base, <who: age, adult proportions>, fixed_traits, <face_traits>, default outfit, slot wording

Why each piece is here (2026-11-17, batch B on the anime-tuned Turbo DiT):

* **profile** - `{"age": 28, "gender": "woman", "build": "..."}`. A prompt that names only the trigger leaves the age
  to the model, and an anime model draws a young girl with a big head on a small body. An adult age turns into
  "adult woman, 28 years old, mature face", and into adult body proportions on the shots that show the body. Variant
  (a) samples at cfg 1.0, so a negative prompt has no effect: the fix has to be said in the positive prompt. With no
  age the profile adds nothing - the builder never guesses one.
* **fixed_traits** - what every picture carries (hair, a scrunchie). As before.
* **face_traits** - small marks that only show on a large face (a mole under the eye). Asked for only on shots whose
  face is big enough to carry them; on a full body shot the words would only pull the camera in.

`caption_traits: omit` strips fixed_traits *and* face_traits from training captions (seed.trait_terms), so the
trigger learns them. The profile words stay out of that rule: they are never in a caption to begin with.
"""
from __future__ import annotations

ADULT_AGE = 18
# The shots that show enough of the body for its proportions to read.
BODY_SHOTS = ("full body", "cowboy shot", "upper body")
# The shots whose face is too small (or turned away) for a face trait to be drawn.
FACELESS_SHOTS = ("full body",)
ADULT_PROPORTIONS = "adult body proportions, small head in proportion to the body, long legs"
MATURE_FACE = "mature face"


def _clean(value):
    return str(value or "").strip()


def profile(card):
    """{"age": int or None, "gender": str, "build": str} from the card's `profile`; gender falls back to `subject`."""
    block = card.get("profile") or {}
    age = block.get("age")
    try:
        age = int(age) if age not in (None, "") and not isinstance(age, bool) else None
    except (TypeError, ValueError):
        age = None
    if age is not None and age <= 0:
        age = None
    gender = _clean(block.get("gender"))
    if not gender:
        subject = _clean(card.get("subject"))
        gender = subject[4:] if subject.lower().startswith("the ") else subject
    return {"age": age, "gender": gender, "build": _clean(block.get("build"))}


def subject_terms(card, *, shot=None):
    """Who is in the picture: the age, and adult proportions when the shot shows the body (any shot when None)."""
    who = profile(card)
    out = []
    if who["age"] is not None:
        if who["age"] >= ADULT_AGE:
            noun = f"adult {who['gender']}" if who["gender"] else "adult"
            out += [f"{noun}, {who['age']} years old", MATURE_FACE]
            if shot is None or shot in BODY_SHOTS:
                out.append(ADULT_PROPORTIONS)
        else:
            out.append(f"{who['gender'] + ', ' if who['gender'] else ''}{who['age']} years old")
    if who["build"]:
        out.append(who["build"])
    return out


def shows_face(cell):
    """Can a face trait be drawn in this cell? None (an eval prompt, a fixed slot) counts as yes."""
    if cell is None:
        return True
    if cell.get("shot") in FACELESS_SHOTS:
        return False
    return not (cell.get("half") == "body" and cell.get("view") == "back view")


def traits(card, key):
    return [_clean(t) for t in (card.get(key) or []) if _clean(t)]


def all_traits(card):
    """fixed_traits then face_traits: every trait word the caption policy is about."""
    return traits(card, "fixed_traits") + traits(card, "face_traits")


def build(card, words, cell=None):
    """The prompt for one picture: the card's words around `words`, the slot's (or eval prompt's) own wording.

    `cell` is a composition cell (`shot`, `view`, `half`) when there is one; it decides whether the body proportions
    and the face traits are asked for.
    """
    shot = (cell or {}).get("shot")
    parts = [card.get("trigger", ""), card.get("caption_base", "")]
    parts += subject_terms(card, shot=shot)
    parts += traits(card, "fixed_traits")
    if shows_face(cell):
        parts += traits(card, "face_traits")
    parts.append((card.get("outfits") or {}).get("default", ""))
    parts.append(words)
    return ", ".join(_clean(p) for p in parts if _clean(p))


def warnings(card):
    """What the card leaves to chance, in words a person can act on (planning logs these; nothing refuses)."""
    out = []
    if profile(card)["age"] is None:
        out.append("the card has no profile.age, so the prompts do not say how old the character is and an anime "
                   "model may draw someone younger, with a bigger head; set it (dataset tab: 나이)")
    return out


# ------------------------------------------------------------------------------------- the dataset tab's fields
class ProfileError(ValueError):
    """A typed age that is not a whole number of years."""


MAX_AGE = 150


def _join(items):
    return ", ".join(items)


def _split(text):
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def fields(card):
    """The card's age and traits as the dataset tab shows them: {"age": "28" or "", "fixed_traits", "face_traits"}."""
    age = profile(card)["age"]
    return {"age": "" if age is None else str(age), "fixed_traits": _join(traits(card, "fixed_traits")),
            "face_traits": _join(traits(card, "face_traits"))}


def apply(card, values):
    """A copy of `card` with the typed fields written in. An empty age removes it; traits are comma separated.

    Raises ProfileError for an age that is not a whole number from 1 to MAX_AGE. Only the keys given are touched.
    """
    out = dict(card)
    if "age" in values:
        text = str(values["age"] or "").strip()
        block = dict(out.get("profile") or {})
        if text:
            try:
                age = int(text)
            except ValueError:
                raise ProfileError(f"age {text!r} is not a whole number") from None
            if not 1 <= age <= MAX_AGE:
                raise ProfileError(f"age {age} is outside 1~{MAX_AGE}")
            block["age"] = age
        else:
            block.pop("age", None)
        out["profile"] = block
    for key in ("fixed_traits", "face_traits"):
        if key in values:
            out[key] = _split(values[key])
    return out
