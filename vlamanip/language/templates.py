"""Natural-language instructions for the two tasks.

``TaskSpec`` is the ground-truth meaning of an instruction (task type, which cube, which destination). It is
used to generate instructions, by the expert and by the evaluation metrics; the learned policy only ever sees
``TaskSpec.text``.

Phrasings are split into a training set and a held-out set, so evaluation can measure generalisation to
wording that never appeared in the demonstrations.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

PICK_PLACE = "pick_place"
STACK = "stack"

PICK_PLACE_TRAIN = (
    "Pick up the {obj} cube and place it in the {dst} bin.",
    "Put the {obj} cube into the {dst} bin.",
    "Move the {obj} cube into the {dst} bin.",
    "place the {obj} block in the {dst} bin",
    "drop the {obj} cube into the {dst} bin",
    "grab the {obj} cube and put it in the {dst} container",
    "put the {obj} block in the {dst} box",
    "pick the {obj} cube and drop it in the {dst} bin",
    # destination named first, so word order alone does not reveal which colour is the object
    "Use the {dst} bin for the {obj} cube.",
    "the {dst} container needs the {obj} block",
    "Target the {dst} bin and drop the {obj} cube in it.",
)
PICK_PLACE_HELDOUT = (
    "Transfer the {obj} cube to the {dst} bin.",
    "Could you put the {obj} block inside the {dst} bin?",
    "the {obj} cube goes in the {dst} bin",
)
STACK_TRAIN = (
    "Pick up the {obj} cube and place it on top of the {dst} cube.",
    "Stack the {obj} cube on the {dst} cube.",
    "Put the {obj} cube on the {dst} cube.",
    "place the {obj} block onto the {dst} block",
    "put the {obj} cube on top of the {dst} one",
    "stack the {obj} block on top of the {dst} block",
    "move the {obj} cube onto the {dst} cube",
    "Use the {dst} cube as the base for the {obj} cube.",
    "the {dst} block goes under the {obj} block",
    "With the {dst} cube as the base, stack the {obj} cube.",
)
STACK_HELDOUT = (
    "Set the {obj} cube on top of the {dst} cube.",
    "Could you stack the {obj} block onto the {dst} cube?",
    "build a tower with the {obj} cube on the {dst} cube",
)

# Stress test, never used for training: unseen phrasings that name the destination *before* the object, so a
# model that just takes "first colour = object" fails.
PICK_PLACE_REORDERED = (
    "Into the {dst} bin, put the {obj} cube.",
    "The {dst} bin should get the {obj} cube.",
    "Fill the {dst} bin with the {obj} cube.",
)
STACK_REORDERED = (
    "On top of the {dst} cube, place the {obj} cube.",
    "The {dst} cube should have the {obj} cube on it.",
    "Cover the {dst} cube with the {obj} cube.",
)

TEMPLATES = {
    (PICK_PLACE, "train"): PICK_PLACE_TRAIN,
    (PICK_PLACE, "heldout"): PICK_PLACE_HELDOUT,
    (PICK_PLACE, "reordered"): PICK_PLACE_REORDERED,
    (STACK, "train"): STACK_TRAIN,
    (STACK, "heldout"): STACK_HELDOUT,
    (STACK, "reordered"): STACK_REORDERED,
}


@dataclass(frozen=True)
class TaskSpec:
    task: str  # PICK_PLACE or STACK
    obj: str  # colour of the cube to move
    dst: str  # colour of the target bin (PICK_PLACE) or of the cube to stack on (STACK)
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def make_instruction(task: str, obj: str, dst: str, rng: random.Random, split: str = "train") -> TaskSpec:
    if task == STACK and obj == dst:
        raise ValueError("A cube cannot be stacked on itself.")
    template = rng.choice(TEMPLATES[(task, split)])
    return TaskSpec(task, obj, dst, template.format(obj=obj, dst=dst))


def sample_task(colors: list[str], rng: random.Random, split: str = "train", task: str | None = None) -> TaskSpec:
    """Random task over ``colors``: PICK_PLACE uses any cube/bin pair, STACK two different cubes."""
    task = task or rng.choice((PICK_PLACE, STACK))
    obj = rng.choice(colors)
    dst = rng.choice([c for c in colors if c != obj]) if task == STACK else rng.choice(colors)
    return make_instruction(task, obj, dst, rng, split)


def parse_instruction(text: str, colors: list[str]) -> TaskSpec | None:
    """Best-effort meaning of free text, used only to *score* a run (the policy reads the raw text).

    A bin-like word (bin, box, container, basket) means a bin destination, otherwise a cube (stacking). The
    destination is the colour followed by a bin word, or (stacking) the colour after "on/onto/of/cover the" or
    before "... as the base / should have / goes under"; the other colour is the cube to move. Without such a
    cue the first colour is the cube and the second the destination. Returns None when unclear.
    """
    lowered = text.lower()
    matches = list(re.finditer(r"\b(" + "|".join(map(re.escape, colors)) + r")\b", lowered))
    if len(matches) < 2:
        return None
    matches = matches[:2]
    bin_word = r"(bin|box|container|basket)\b"
    task = PICK_PLACE if re.search(r"\b" + bin_word, lowered) else STACK

    def is_destination(m: re.Match) -> bool:
        before, after = lowered[: m.start()], lowered[m.end():]
        if task == PICK_PLACE:
            return re.match(r"\s+" + bin_word, after) is not None
        return (
            re.search(r"\b(on|onto|of|cover)\s+the\s+$", before) is not None
            or re.match(r"\s+(cube|block|one)\s+(as the base|should have|goes under)\b", after) is not None
        )

    flags = [is_destination(m) for m in matches]
    first, second = matches[0].group(1), matches[1].group(1)
    obj, dst = (second, first) if flags == [True, False] else (first, second)
    if task == STACK and obj == dst:
        return None
    return TaskSpec(task, obj, dst, text)
