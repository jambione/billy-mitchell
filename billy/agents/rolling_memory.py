"""Rolling short/long-term memory for the LLM tier (B3 — ClaudePlayer pattern).

Advisory context only: never keys into SolutionCache or selects replay actions.
"""
from __future__ import annotations

from collections import deque

from .. import llm


class RollingGameMemory:
    """Keep a durable session summary + recent notes; roll up every N turns via LLM."""

    def __init__(self, *, rollup_every: int = 8, short_max: int = 12) -> None:
        self.short_term: deque[str] = deque(maxlen=short_max)
        self.long_summary: str = ""
        self.turn_count = 0
        self.rollup_every = max(1, rollup_every)

    def reset(self) -> None:
        self.short_term.clear()
        self.long_summary = ""
        self.turn_count = 0

    def note(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.short_term.append(text)
        self.turn_count += 1

    def should_rollup(self) -> bool:
        return bool(self.short_term) and self.turn_count % self.rollup_every == 0

    def rollup(self) -> None:
        """Compress recent notes into long_summary (off hot path — LLM call)."""
        if not self.short_term:
            return
        recent = "\n".join(f"- {e}" for e in self.short_term)
        prior = self.long_summary or "(none yet)"
        user = (
            f"PRIOR SUMMARY:\n{prior}\n\nRECENT EVENTS:\n{recent}\n\n"
            "Write a NEW compact summary (3-5 sentences) of Billy's run so far. "
            "Keep concrete facts: screens reached, items, deaths, what worked. JSON: "
            '{"summary": "..."}'
        )
        try:
            data = llm.chat_json(
                [{"role": "system", "content": "You compress game-play notes for a speedrunner."},
                 {"role": "user", "content": user}],
                temperature=0.2, max_tokens=200,
            )
            summary = str(data.get("summary", "")).strip()
            if summary:
                self.long_summary = summary
        except llm.LLMError:
            # Degrade gracefully — append recent to long_summary textually
            tail = "; ".join(list(self.short_term)[-4:])
            if tail:
                self.long_summary = (self.long_summary + " | " + tail).strip(" |")[-500:]
        self.short_term.clear()

    def prompt_section(self) -> str:
        parts: list[str] = []
        if self.long_summary:
            parts.append(f"SESSION MEMORY:\n{self.long_summary}")
        if self.short_term:
            parts.append("RECENT EVENTS: " + "; ".join(list(self.short_term)[-6:]))
        return "\n\n".join(parts)