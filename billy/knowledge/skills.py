"""The Skill layer — semantic, cross-game transfer of *abstract tactics*.

The SolutionCache transfers EXACT solutions within a game (deterministic replay). Skills transfer
GENERALISABLE tactics ACROSS games: "precise gap jump", "stomp an enemy from the approach",
"run-jump a tall obstacle". Each Skill carries an embedding of its description; on any game we
retrieve the skills whose description is most similar to the current situation (`obs.summary`) and
**instantiate** them into concrete candidate plans via the shared platformer builders
(games/common/platformer.py).

Crucially, skills only *widen the micro-search candidate set on a cache MISS* — they never replay
blind. So on a brand-new game (empty cache) Billy starts with sensible, transferable attempts
instead of a cold flailing search; but a wrong skill can only ever lose a search rollout, never
commit a bad action. This is the embedding half of transfer; the reflex primitives are the other.

Embeddings are best-effort (reuse the KB's `llm.embed`); if the embedder is offline the library
degrades to a flat candidate set (all skills), so search still benefits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..abstractions import Plan
from ..games.common import platformer
from ..games.common.platformer import PhysicsProfile
from .store import _cosine, _safe_embed

# Each skill kind maps to a builder that turns the current view+profile into candidate plans.
_BUILDERS = {
    "gap_jump": lambda view, p: platformer.gap_jumper(
        (view.gap_info() or (0, 2))[1], p),
    "stomp": lambda view, p: platformer.enemy_stomper(),
    "wall_jump": lambda view, p: platformer.wall_jumper(),
}

# The starter library distilled from SMB's proven tactics — the knowledge Billy carries forward.
STARTER_SKILLS = [
    ("precise gap jump", "a deadly pit ahead: launch near the edge and hold A scaled to the pit "
     "width to clear the gap and land safely", "gap_jump"),
    ("stomp from approach", "an enemy ahead at ground level: time a jump so you descend onto its "
     "head and stomp it instead of running into it", "stomp"),
    ("run-jump a tall obstacle", "a tall wall or pipe flush ahead: back up for runway, then run "
     "and jump to clear the obstacle that a standing jump can't", "wall_jump"),
]


@dataclass
class Skill:
    name: str
    description: str
    kind: str                       # dispatch tag into _BUILDERS (keeps the skill serialisable)
    embedding: list[float] = field(default_factory=list)
    # DISTILLED skills ("sequence" kind) carry a proven exact plan as payload:
    #   {"plan": [[frames, buttons], ...], "console": "nes", "sig": "<dedupe hash>",
    #    "gained": <px>, "source": "learn_section|demo|..."}
    # The plan is only ever offered as ONE MORE SEARCH CANDIDATE (verified on a clone before any
    # commit) — never blind-replayed — so a mismatched skill costs a rollout, never a death.
    payload: dict = field(default_factory=dict)

    def instantiate(self, view, profile: PhysicsProfile) -> list[Plan]:
        if self.kind == "sequence":
            steps = self.payload.get("plan") or []
            return [[_seq_step(f, b) for f, b in steps]] if steps else []
        builder = _BUILDERS.get(self.kind)
        return builder(view, profile) if builder else []


def _seq_step(frames: int, buttons: int):
    from ..abstractions import Step
    return Step(frames, buttons)


@dataclass
class SkillLibrary:
    """Embedding-retrieved abstract tactics, persisted to skills.jsonl. Reuses the KB's cosine +
    embedding helpers — no extra dependency."""
    path: Path = field(default_factory=lambda: config.SKILLS_FILE)
    skills: list[Skill] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    def add(self, name: str, description: str, kind: str,
            payload: dict | None = None) -> Skill:
        skill = Skill(name=name, description=description, kind=kind,
                      embedding=_safe_embed(f"{name}. {description}"),
                      payload=payload or {})
        self.skills.append(skill)
        self._save()
        return skill

    def has_signature(self, sig: str) -> bool:
        """Dedupe check for distilled sequence skills (same plan → same sig → skip)."""
        return any(s.payload.get("sig") == sig for s in self.skills)

    def seed_starter(self) -> "SkillLibrary":
        """Populate the standard transferable platformer tactics (idempotent by name)."""
        have = {s.name for s in self.skills}
        for name, desc, kind in STARTER_SKILLS:
            if name not in have:
                self.add(name, desc, kind)
        return self

    def retrieve(self, summary: str, k: int = 3) -> list[Skill]:
        if not self.skills:
            return []
        q = _safe_embed(summary)
        if not q:
            return self.skills[:k]   # embedder offline -> flat fallback
        return sorted(self.skills, key=lambda s: _cosine(q, s.embedding), reverse=True)[:k]

    def candidates(self, view, profile: PhysicsProfile, summary: str, k: int = 3,
                   console: str = "") -> list[Plan]:
        """Concrete candidate plans from the top-k situationally-relevant skills (for search).
        Distilled sequence skills are console-gated: their button masks only mean the same
        thing on the console they were recorded on (an NES mask is noise on a SNES pad)."""
        out: list[Plan] = []
        for s in self.retrieve(summary, k):
            if (s.kind == "sequence" and console
                    and s.payload.get("console", console) != console):
                continue
            out.extend(s.instantiate(view, profile))
        return out

    def __len__(self) -> int:
        return len(self.skills)

    # --- persistence --------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.skills.append(Skill(**json.loads(line)))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for s in self.skills:
                row = {"name": s.name, "description": s.description,
                       "kind": s.kind, "embedding": s.embedding}
                if s.payload:
                    row["payload"] = s.payload
                f.write(json.dumps(row) + "\n")
