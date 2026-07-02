"""The knowledge base — the literal substrate of Billy's learning.

Each lesson (situation -> tactic -> outcome) is embedded once (via LM Studio's embedding
model) and persisted to lessons.jsonl. Before a decision, the Director embeds the current
situation and pulls the top-K most similar lessons into Billy's prompt, so he reuses what
worked instead of re-reasoning from scratch. Near-duplicate situations are merged so the
store stays sharp rather than sprawling.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import config, llm

try:  # numpy just speeds up cosine search; everything works without it.
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

_MERGE_THRESHOLD = 0.93  # cosine above which a new lesson updates an existing one


@dataclass
class Lesson:
    situation: str
    tactic: str
    outcome: str
    world_stage: str = ""
    game: str = ""             # owning game (cli_name); "" = legacy/untagged
    uses: int = 0
    impact_score: float = 0.0  # cumulative progress gained from applying this lesson
    embedding: list[float] = field(default_factory=list)

    def prompt_line(self) -> str:
        return f"- At {self.situation}: {self.tactic} ({self.outcome})"

    def quality(self) -> float:
        """Lesson quality: impact per use. Higher = more valuable."""
        return self.impact_score / max(1, self.uses)

    def __hash__(self) -> int:
        """Hash based on content, not embedding or mutable fields."""
        return hash((self.situation, self.tactic))

    def __eq__(self, other: object) -> bool:
        """Equality based on situation+tactic (content), not metadata."""
        if not isinstance(other, Lesson):
            return NotImplemented
        return self.situation == other.situation and self.tactic == other.tactic


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    if _np is not None:
        va, vb = _np.asarray(a), _np.asarray(b)
        denom = (_np.linalg.norm(va) * _np.linalg.norm(vb)) or 1.0
        return float(va.dot(vb) / denom)
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class KnowledgeBase:
    def __init__(self, path: Path = config.LESSONS_FILE):
        self.path = Path(path)
        self.lessons: list[Lesson] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                self.lessons.append(Lesson(**json.loads(line)))

    def _save(self) -> None:
        config.ensure_dirs()
        with self.path.open("w") as f:
            for les in self.lessons:
                f.write(json.dumps(asdict(les)) + "\n")

    def add(self, situation: str, tactic: str, outcome: str, world_stage: str = "",
            game: str = "") -> Lesson:
        """Embed and store a lesson, merging into a near-duplicate if one exists.

        `game` scopes the lesson to its title (see retrieve): a merge also (re)tags the game,
        so legacy untagged lessons self-heal the next time they're seen during that game."""
        emb = _safe_embed(f"{situation} | {tactic}")
        if emb:
            for les in self.lessons:
                if _cosine(emb, les.embedding) >= _MERGE_THRESHOLD:
                    les.tactic, les.outcome, les.uses = tactic, outcome, les.uses + 1
                    if game:
                        les.game = game
                    self._save()
                    return les
        lesson = Lesson(situation=situation, tactic=tactic, outcome=outcome,
                        world_stage=world_stage, game=game, embedding=emb)
        self.lessons.append(lesson)
        self._save()
        return lesson

    def retrieve(self, situation: str, k: int = config.KB_TOP_K,
                 game: str = "") -> list[Lesson]:
        """Top-K lessons ranked by: similarity + quality (impact/use).
        High-impact lessons bubble up even if not perfectly similar.

        When `game` is given, restrict to that game's lessons — an SMB pit tactic must never
        surface during Zelda (lessons are advice, but cross-game noise pollutes the prompt).
        Legacy untagged lessons (game="") are dormant under this filter until they re-tag on
        their next in-game merge; that's fine — lessons are advice, never load-bearing, and a
        dormant lesson is strictly safer than a mis-scoped one."""
        pool = self.lessons
        if game:
            pool = [l for l in pool if l.game == game]
        if not pool:
            return []
        q = _safe_embed(situation)
        if not q:
            # fall back to most recent + highest quality if embedding unavailable
            return sorted(pool, key=lambda l: (l.impact_score, l.uses), reverse=True)[-k:]
        # Hybrid rank: similarity (70%) + quality (30%)
        ranked = sorted(
            pool,
            key=lambda l: (0.7 * _cosine(q, l.embedding) + 0.3 * min(1.0, l.quality() / 100)),
            reverse=True
        )
        return ranked[:k]

    def record_impact(self, lesson: Lesson, progress_gain: int) -> None:
        """Record that applying this lesson led to a progress gain."""
        if lesson in self.lessons:
            lesson.impact_score += progress_gain
            lesson.uses += 1
            self._save()


def _safe_embed(text: str) -> list[float]:
    """Embeddings are best-effort; the KB still works (recency fallback) if the model is down."""
    try:
        return llm.embed(text)
    except llm.LLMError:
        return []
