import itertools

from vlamanip.language.templates import PICK_PLACE, STACK, TEMPLATES, parse_instruction

COLORS = ["red", "blue", "green", "yellow"]


def meaning(text):
    spec = parse_instruction(text, COLORS)
    return None if spec is None else (spec.task, spec.obj, spec.dst)


def test_parse_examples():
    assert meaning("Pick up the red cube and place it in the blue bin.") == (PICK_PLACE, "red", "blue")
    assert meaning("Stack the yellow cube on the green cube") == (STACK, "yellow", "green")
    assert meaning("Put the green block into the green box") == (PICK_PLACE, "green", "green")
    assert meaning("Stack the red cube on the red cube") is None
    assert meaning("move it over there") is None


def test_parse_destination_first():
    assert meaning("Into the blue bin, put the red cube.") == (PICK_PLACE, "red", "blue")
    assert meaning("On top of the green cube, place the yellow cube.") == (STACK, "yellow", "green")
    assert meaning("The red cube should have the blue cube on it.") == (STACK, "blue", "red")


def test_parse_recovers_every_template():
    for (task, _), templates in TEMPLATES.items():
        for template, (obj, dst) in itertools.product(templates, [("red", "blue"), ("yellow", "green")]):
            assert meaning(template.format(obj=obj, dst=dst)) == (task, obj, dst), template
