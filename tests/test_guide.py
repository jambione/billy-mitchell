"""Walkthrough learning: FAQ ingestion (heuristic path), retrieval, and search-candidate bias."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from billy.knowledge.guide import GuideLibrary, GuideStep, heuristic_parse
from billy.systems.nes import controller as c

FAQ_SNIPPET = """
================================
  2. THE FIRST STEPS
================================

When you begin the game, you are on a screen with a cave.  Enter the cave
at the top of the screen and the old man inside will give you the wooden
sword.

Now exit the cave.  From this screen, head east eight screens until you
reach the coast.  Watch out for the Octoroks along the way, they shoot
rocks at you.

The history of Hyrule is long and storied.  Many years ago the land was
peaceful and its people prosperous.

To find the first dungeon, go north through the forest.  Follow the river
and cross the bridge, then enter the tree that looks like an eagle.
"""


def test_heuristic_parse_extracts_actionable_directional_steps():
    steps = heuristic_parse(FAQ_SNIPPET)
    texts = " | ".join(s.text.lower() for s in steps)
    assert any("east" in s.text.lower() for s in steps), "the east march must be captured"
    assert any(s.direction == "right" for s in steps), "east → logical right"
    assert any(s.direction == "up" for s in steps), "north → logical up"
    # Pure lore must not become a step.
    assert "history of hyrule" not in texts


def test_heuristic_parse_survives_decoration():
    steps = heuristic_parse("=== ****** ===\n\nGo north into the cave.\n\n¯¯¯¯¯¯")
    assert len(steps) == 1 and steps[0].direction == "up"


def test_persistence_roundtrip(tmp_path):
    lib = GuideLibrary(path=tmp_path / "g.jsonl")
    lib.steps = heuristic_parse(FAQ_SNIPPET)
    lib._save()
    lib2 = GuideLibrary(path=tmp_path / "g.jsonl")
    assert len(lib2) == len(lib) and lib2.steps[0].text == lib.steps[0].text


def test_retrieve_flat_fallback_without_embedder(tmp_path):
    lib = GuideLibrary(path=tmp_path / "g.jsonl")
    lib.steps = [GuideStep(order=i, text=f"step {i}", direction="right") for i in range(5)]
    got = lib.retrieve("anything", k=2)   # no embeddings anywhere → flat first-k
    assert [s.order for s in got] == [0, 1]


def test_direction_candidates_build_valid_plans(tmp_path):
    lib = GuideLibrary(path=tmp_path / "g.jsonl")
    lib.steps = [GuideStep(order=0, text="head east to the coast", direction="right"),
                 GuideStep(order=1, text="enter the cave", direction="enter")]
    plans = lib.direction_candidates("near a cave by the coast", c, k=2)
    assert plans, "directional steps must yield candidate plans"
    masks = {step.buttons for plan in plans for step in plan}
    assert c.BUTTON_BITS["right"] in masks
    assert c.BUTTON_BITS["up"] in masks or c.BUTTON_BITS["down"] in masks  # 'enter' probes
    # candidates only — every plan is a short bounded hold, nothing exotic
    assert all(1 <= step.frames <= 60 for plan in plans for step in plan)


def test_prompt_section_formats_lines(tmp_path):
    lib = GuideLibrary(path=tmp_path / "g.jsonl")
    lib.steps = [GuideStep(order=0, text="head east eight screens", direction="right")]
    section = lib.prompt_section("standing on the starting screen")
    assert "Walkthrough guidance" in section and "head east" in section


def test_empty_library_is_silent(tmp_path):
    lib = GuideLibrary(path=tmp_path / "g.jsonl")
    assert lib.retrieve("x") == []
    assert lib.prompt_section("x") == ""
    assert lib.direction_candidates("x", c) == []
