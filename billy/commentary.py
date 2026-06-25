"""Billy's running mouth.

Frequent, in-character commentary fired by game events (stomps, pits cleared, powerups,
distance milestones, deaths) — WITHOUT an LLM call, so it stays snappy. Lines are in the
voice of the real Billy Mitchell: third-person self-mythologizing, world-record bravado,
hot-sauce bragging, and a steadfast refusal to ever take blame for a death.

The dynamic, situation-specific zingers still come from the LLM at decision points; this
layer guarantees he's always running his mouth in between.
"""
from __future__ import annotations

import random

from .games.smb.perception import Scene

# Throttle ordinary chatter so it flavors without spamming (in emulator frames; 60 = 1s).
_MIN_GAP_FRAMES = 50

QUIPS: dict[str, list[str]] = {
    "start": [
        "Billy Mitchell does not 'try' a level. He collects it.",
        "Adjust your hot sauce and pay attention — history is about to be made.",
        "They named me Player of the Century. The century, folks. Watch closely.",
        "A lesser man would be nervous. I've simply never met the feeling.",
    ],
    "milestone": [
        "Effortless. The Mitchell name carries weight, even in pixels.",
        "Textbook. I could do this with my eyes closed and one hand on the sauce.",
        "Are you watching? You should be writing this down.",
        "Forward, always forward — that's how legends are spelled.",
        "Every step further is another record nobody will ever touch.",
        "This is what greatness looks like in motion. You're welcome.",
    ],
    "stomp": [
        "Squashed. Another little nobody who got in the champion's way.",
        "Goodbye, mushroom man. Nothing personal — it's just the natural order.",
        "I don't fight enemies. I file them.",
        "That's one less obstacle between me and immortality.",
    ],
    "gap": [
        "Over the chasm like it owed me money. Beautiful.",
        "Lesser players fall there. I am not a lesser player.",
        "Gravity asked for an autograph. I cleared the pit instead.",
    ],
    "powerup": [
        "Bigger, better, and somehow even more handsome. As intended.",
        "Power-up? I was already powered up. This is just for the cameras.",
        "Now THAT is an upgrade worthy of the King of Kong.",
    ],
    "coin": [
        "Coins. I collect those the way I collect world records — constantly.",
        "Ka-ching. Even the scenery wants to pay me.",
    ],
    "newbest": [
        "Farther than I've ever gone — and I've gone farther than anyone alive.",
        "New territory. Plant the flag. Mitchell was here first.",
        "Watch the record fall. Again. It's almost boring how good I am.",
    ],
    "stuck": [
        "The controller's sticking. Obviously. Get me a clean one.",
        "A momentary pause — for dramatic effect, naturally.",
        "This cartridge wasn't built to contain talent like mine.",
    ],
    "death_pit": [
        "I did NOT fall. The pit rose to meet me. There's a difference.",
        "Sabotage. That hole was not regulation depth, I assure you.",
    ],
    "death_enemy": [
        "A cheap shot. The champ was robbed and everyone saw it.",
        "That enemy got lucky once. It will not happen twice.",
    ],
    "death_generic": [
        "Glitchy cartridge. Bad batch. Take it up with Nintendo, not me.",
        "Sunspots. Solar interference. Certainly not Billy Mitchell.",
        "An off-frame. Even perfection has a rounding error.",
    ],
    "clear": [
        "FLAWLESS. Frame it. Hang it. Teach it in schools.",
        "And THAT is how the greatest of all time signs his name.",
        "Too easy. Bring me a real challenge — and more hot sauce.",
    ],
}


class Commentator:
    def __init__(self, min_gap_frames: int = _MIN_GAP_FRAMES) -> None:
        self.min_gap = min_gap_frames
        self.reset(None)

    def reset(self, start_scene: Scene | None) -> None:
        self._prev = start_scene
        self._best_x = start_scene.mario_x if start_scene else 0
        self._next_milestone = ((self._best_x // 256) + 1) * 256
        self._last_frame = -10_000
        self._last_line = ""

    def _pick(self, key: str) -> str:
        options = [q for q in QUIPS.get(key, []) if q != self._last_line] or QUIPS.get(key, [""])
        line = random.choice(options)
        self._last_line = line
        return line

    def observe(self, scene: Scene) -> str | None:
        """Return a quip to print for this frame (already throttled), or None."""
        prev, key, force = self._prev, None, False
        if prev is not None:
            if scene.size > prev.size:
                key, force = "powerup", True
            elif len(scene.enemies) < len(prev.enemies) and prev.enemy_ahead(48):
                key = "stomp"
            elif scene.coins > prev.coins:
                key = "coin"
        if key is None and scene.mario_x > self._best_x and scene.mario_x >= self._next_milestone:
            self._next_milestone += 256
            key = "newbest" if scene.mario_x > self._best_x + 240 else "milestone"
        self._best_x = max(self._best_x, scene.mario_x)
        self._prev = scene
        if key and (force or scene.frame - self._last_frame >= self.min_gap):
            self._last_frame = scene.frame
            return self._pick(key)
        return None

    def event_line(self, key: str) -> str:
        """An on-demand line for discrete moments (start, stuck, death_*, clear)."""
        return self._pick(key)

    def death_quip(self, scene: Scene) -> str:
        """A cause-aware death line (pit vs enemy vs glitch)."""
        if getattr(scene, "y_viewport", 0) > 1:
            return self._pick("death_pit")
        if getattr(scene, "enemies", None):
            return self._pick("death_enemy")
        return self._pick("death_generic")
