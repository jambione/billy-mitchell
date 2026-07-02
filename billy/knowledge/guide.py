"""Walkthrough learning — Billy reads the FAQ, not just his own scars.

Drop a text walkthrough at `walkthrough/<SYSTEM>/<game>` and Billy ingests it ONCE into an
ordered, embedded step list (`data/guides/<game>.jsonl`). Two uses, both invariant-safe:

  • SEARCH BIAS — on a cache miss, the steps most similar to the current situation
    contribute direction-hold candidate plans ("head north" → UP holds). Candidates only:
    micro-search still verifies on a clone before anything commits, so a wrong or misread
    instruction costs a rollout, never a death.
  • LLM CONTEXT — when Billy consults the LLM at a wall, the relevant walkthrough lines are
    retrieved into the prompt, so the local model improvises WITH the guide's knowledge.

Ingestion prefers the local LLM (chunked extraction of actionable steps); with the LLM
offline it degrades to a heuristic parser (directional-sentence scan) so the capability
never blocks on infrastructure. Embeddings are best-effort like the Skill layer: embedder
down → flat retrieval, still useful.

This layer is the general form of what games/zelda/walkthrough.py does by hand: that module
is a human-COMPILED route (precise, proven); this one is a machine-READ guide (approximate,
verified by search). Both feed the same engine; learning-from-performance remains the
authority on what actually works.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .. import config
from ..abstractions import Plan, Step
from .store import _cosine, _safe_embed

GUIDES_DIR = config.DATA_DIR / "guides"

_DIRECTIONS = {
    "north": "up", "up": "up", "top": "up",
    "south": "down", "down": "down", "bottom": "down",
    "east": "right", "right": "right",
    "west": "left", "left": "left",
    "enter": "enter", "inside": "enter",
}
_ACTION_VERBS = ("go", "head", "walk", "move", "continue", "enter", "take", "climb",
                 "cross", "follow", "push", "ride", "jump", "swim", "exit", "return")


@dataclass
class GuideStep:
    order: int
    text: str
    direction: str = ""          # "up/down/left/right/enter" or "" when purely informative
    embedding: list = field(default_factory=list)

    def prompt_line(self) -> str:
        return f"- {self.text}"


def heuristic_parse(text: str, max_steps: int = 400) -> list[GuideStep]:
    """LLM-free fallback: pull actionable, directional sentences out of a raw FAQ.

    FAQ text is hard-wrapped ASCII with decorative headers; we join paragraphs, split into
    sentences, and keep those that read like instructions (an action verb + ideally a
    direction). Approximate by design — search verifies anything acted on."""
    # Join hard-wrapped lines into paragraphs; drop decoration-heavy lines.
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(re.sub(r"[\w\s.,;:'\"!?()-]", "", line)) > len(line) * 0.2:
            lines.append("")   # decoration/blank ends a paragraph
        else:
            lines.append(line)
    paragraphs = [" ".join(p.split()) for p in "\n".join(lines).split("\n\n")]

    steps: list[GuideStep] = []
    for para in paragraphs:
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            s = sentence.strip()
            if not (20 <= len(s) <= 220):
                continue
            low = s.lower()
            if not any(re.search(rf"\b{v}\b", low) for v in _ACTION_VERBS):
                continue
            direction = ""
            for word, d in _DIRECTIONS.items():
                if re.search(rf"\b{word}\b", low):
                    direction = d
                    break
            if direction or any(re.search(rf"\b{v}\b", low) for v in _ACTION_VERBS[:8]):
                steps.append(GuideStep(order=len(steps), text=s, direction=direction))
            if len(steps) >= max_steps:
                return steps
    return steps


def llm_parse(text: str, *, chunk_chars: int = 4000, max_steps: int = 400) -> list[GuideStep]:
    """LLM extraction: each chunk of the FAQ becomes a few actionable steps with directions.
    Raises LLMError upward on total failure — the caller falls back to heuristic_parse."""
    from .. import llm

    steps: list[GuideStep] = []
    chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
    print(f"[guide] LLM-reading the walkthrough ({len(chunks)} chunks)...")
    for ci, chunk in enumerate(chunks):
        try:
            out = llm.chat_json([
                {"role": "system", "content":
                 "You extract ACTIONABLE game-walkthrough steps. Reply as JSON: "
                 '{"steps": [{"text": "<one imperative instruction, <=30 words>", '
                 '"direction": "up|down|left|right|enter|"}]}. Only concrete play '
                 "instructions (movement, items, fights); skip lore, credits, tables."},
                {"role": "user", "content": chunk},
            ], max_tokens=700)
        except llm.LLMError:
            continue   # one bad chunk shouldn't sink the read
        for s in out.get("steps", []) or []:
            txt = str(s.get("text", "")).strip()
            if 8 <= len(txt) <= 240:
                d = str(s.get("direction", "")).strip().lower()
                steps.append(GuideStep(order=len(steps), text=txt,
                                       direction=d if d in
                                       ("up", "down", "left", "right", "enter") else ""))
        if len(steps) >= max_steps:
            break
        if ci and ci % 5 == 0:
            print(f"[guide]   ...{ci}/{len(chunks)} chunks, {len(steps)} steps")
    if not steps:
        from ..llm import LLMError
        raise LLMError("no steps extracted")
    return steps


@dataclass
class GuideLibrary:
    """Persisted, embedded walkthrough steps for one game."""
    path: Path
    steps: list[GuideStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load()

    def ingest(self, text: str, *, use_llm: bool = True, max_steps: int = 400) -> int:
        """Parse a raw walkthrough into steps, embed, save.

        MERGES both readers: the LLM extraction is clean but lossy (a failed chunk silently
        drops its steps — the Zelda read lost 'head right 8 screens to the sea'), while the
        heuristic scan is noisy but thorough. LLM steps come first; heuristic steps join
        unless they near-duplicate one (word-overlap), so coverage never regresses below the
        heuristic floor."""
        llm_steps: list[GuideStep] = []
        if use_llm:
            try:
                llm_steps = llm_parse(text)
                print(f"[guide] LLM extracted {len(llm_steps)} steps")
            except Exception as e:
                print(f"[guide] LLM read unavailable ({type(e).__name__}) — heuristic only")
        heur = heuristic_parse(text)
        print(f"[guide] heuristic parser extracted {len(heur)} steps")

        def words(s: GuideStep) -> set:
            return set(re.findall(r"[a-z']+", s.text.lower()))

        steps = list(llm_steps)
        keys = [words(s) for s in steps]
        for h in heur:
            hw = words(h)
            if not hw:
                continue
            dup = any(len(hw & kw) / max(1, len(hw | kw)) >= 0.5 for kw in keys)
            if not dup:
                steps.append(h)
                keys.append(hw)
            if len(steps) >= max_steps:
                break
        for i, s in enumerate(steps):
            s.order = i
            s.embedding = _safe_embed(s.text)
        self.steps = steps
        self._save()
        print(f"[guide] merged guide: {len(steps)} steps "
              f"({len(llm_steps)} LLM + {len(steps) - len(llm_steps)} heuristic-only)")
        return len(steps)

    def retrieve(self, summary: str, k: int = 3) -> list[GuideStep]:
        if not self.steps:
            return []
        q = _safe_embed(summary)
        if not q:
            return self.steps[:k]   # embedder offline → flat fallback (early steps first)
        return sorted(self.steps, key=lambda s: _cosine(q, s.embedding), reverse=True)[:k]

    def prompt_section(self, summary: str, k: int = 3) -> str:
        got = self.retrieve(summary, k)
        if not got:
            return ""
        return "\nWalkthrough guidance (from the game's FAQ):\n" + \
            "\n".join(s.prompt_line() for s in got)

    def direction_candidates(self, summary: str, controller, k: int = 2) -> list[Plan]:
        """Direction-hold candidate plans from the most situation-relevant steps.
        Search-seed only — a misread direction loses a rollout, never a life."""
        bits = getattr(controller, "BUTTON_BITS", None) or getattr(controller, "buttons", {})
        plans: list[Plan] = []
        seen: set[str] = set()
        for s in self.retrieve(summary, k):
            d = s.direction
            if not d or d in seen:
                continue
            seen.add(d)
            if d == "enter":
                for name in ("up", "down"):
                    if name in bits:
                        plans.append([Step(32, bits[name])])
                continue
            if d in bits:
                plans.append([Step(24, bits[d])])
                plans.append([Step(48, bits[d])])
        return plans

    def __len__(self) -> int:
        return len(self.steps)

    # --- persistence --------------------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if line.strip():
                self.steps.append(GuideStep(**json.loads(line)))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for s in self.steps:
                f.write(json.dumps({"order": s.order, "text": s.text,
                                    "direction": s.direction,
                                    "embedding": s.embedding}) + "\n")


def load_guide_for(game, cli_name: str = "") -> GuideLibrary | None:
    """Find + (first run) ingest the walkthrough for a game; None when no FAQ file exists.

    Convention: `walkthrough/<SYSTEM>/<cli_name>` (e.g. walkthrough/NES/zelda). The parsed
    guide is cached at data/guides/<cli_name>.jsonl — delete it to re-ingest (e.g. after
    LM Studio comes up, for the better LLM read)."""
    name = cli_name or getattr(game, "cli_name", "") or ""
    if not name:
        return None
    sysname = getattr(game.system, "name", "")
    src = None
    for cand in (config.REPO_ROOT / "walkthrough" / sysname.upper() / name,
                 config.REPO_ROOT / "walkthrough" / sysname / name):
        if cand.is_file():
            src = cand
            break
    if src is None:
        return None
    lib = GuideLibrary(path=GUIDES_DIR / f"{name}.jsonl")
    if len(lib) == 0:
        from .. import llm
        print(f"[guide] first run: ingesting {src} ...")
        lib.ingest(src.read_text(errors="replace"), use_llm=llm.health())
    print(f"[guide] walkthrough loaded: {len(lib)} steps for {name}")
    return lib
