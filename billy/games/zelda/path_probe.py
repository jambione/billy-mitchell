"""Zelda path probe helpers — scene snapshots, plan I/O, milestone checks."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from ...abstractions import Plan, Step
from ...systems.nes import controller as c
from .perception import Scene
from .start_cave import interior_phase
from .walkthrough import current_phase


def scene_record(scene: Scene, *, t: int = 0, buttons: int = 0, note: str = "",
                 keyframe: bool = False) -> dict[str, Any]:
    """JSON-serializable snapshot for path logging."""
    visited = set(scene.visited_screens)
    return {
        "t": t,
        "frame": scene.frame,
        "screen": scene.map_location,
        "realm": scene.realm,
        "link": [scene.link_x, scene.link_y],
        "mode": scene.game_mode,
        "in_cave": scene.in_cave,
        "sword": scene.sword_level,
        "hearts": f"{scene.health}/{scene.max_hearts}",
        "rupees": scene.rupees,
        "progress": scene.objective_score(),
        "faq_phase": current_phase(
            map_location=scene.map_location,
            sword_level=scene.sword_level,
            max_hearts=scene.max_hearts,
            visited=visited,
            in_cave=scene.in_cave,
        ),
        "interior_phase": interior_phase(scene),
        "buttons": c.names_from_mask(buttons),
        "note": note,
        "keyframe": keyframe,
    }


def parse_script(spec: str) -> Plan:
    """Parse `RIGHT:120,DOWN:25` or `right:120,up+left:50` into a Plan."""
    if not spec.strip():
        raise ValueError("empty script")
    steps: list[Step] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"expected NAME:FRAMES, got {chunk!r}")
        names, frames_s = chunk.split(":", 1)
        frames = int(frames_s.strip())
        if frames < 1:
            raise ValueError(f"frames must be >= 1 in {chunk!r}")
        steps.append(Step(frames, c.mask_from_names(names.replace("+", ","))))
    if not steps:
        raise ValueError("script produced no steps")
    return steps


def plan_to_jsonable(plan: Plan) -> list[dict[str, Any]]:
    return [{"frames": s.frames, "buttons": c.names_from_mask(s.buttons)} for s in plan]


def plan_from_jsonable(data: list[dict[str, Any]]) -> Plan:
    out: list[Step] = []
    for item in data:
        out.append(Step(int(item["frames"]), c.mask_from_names(item.get("buttons", []))))
    return out


def load_plan(spec: str) -> Plan:
    """Load a plan from module:ATTR, a .json file, or a script string."""
    path = Path(spec)
    if path.suffix == ".json" and path.exists():
        raw = json.loads(path.read_text())
        if isinstance(raw, dict) and "steps" in raw:
            raw = raw["steps"]
        return plan_from_jsonable(raw)

    if ":" in spec and not spec.strip().endswith(".json"):
        mod_name, attr = spec.rsplit(":", 1)
        if "." in mod_name or mod_name.startswith("billy"):
            mod = importlib.import_module(mod_name)
            plan = getattr(mod, attr)
            return list(plan)

    return parse_script(spec)


def keyframe_reason(prev: dict[str, Any] | None, cur: dict[str, Any],
                    *, every_n: int = 0) -> str | None:
    """Return a keyframe tag when screen/phase changes or every N ticks."""
    if prev is None:
        return "start"
    if cur["screen"] != prev["screen"]:
        return f"screen_{cur['screen']}"
    if cur["faq_phase"] != prev["faq_phase"]:
        return f"faq_{cur['faq_phase']}"
    if cur["interior_phase"] != prev["interior_phase"]:
        return f"interior_{cur['interior_phase']}"
    if cur["sword"] != prev["sword"]:
        return f"sword_{cur['sword']}"
    if every_n > 0 and cur["t"] > 0 and cur["t"] % every_n == 0:
        return f"t{cur['t']}"
    return None


def check_milestones(obs, *, expect_screen: int | None, expect_sword: int | None,
                     expect_progress: int | None) -> list[str]:
    """Return list of failed expectation messages (empty = ok)."""
    fails: list[str] = []
    scene: Scene = obs.raw
    if expect_screen is not None and scene.map_location < expect_screen:
        fails.append(f"screen {scene.map_location} < {expect_screen}")
    if expect_sword is not None and scene.sword_level < expect_sword:
        fails.append(f"sword {scene.sword_level} < {expect_sword}")
    if expect_progress is not None and obs.progress < expect_progress:
        fails.append(f"progress {obs.progress} < {expect_progress}")
    return fails