"""Route memory — the discovered level topology of a game.

Every transition Billy OBSERVES becomes an edge: a level clear (flag), a screen/area change
(pipe warp, Zelda border), keyed by where on the source level it fired. This is the map a
strategist reads to route Billy toward "game complete" — most importantly, a discovered WARP
(an edge whose destination skips ahead of the natural next level) can be preferred over the
sequential grind. Like everything else in the knowledge stack the graph only records what
actually happened live; it never invents transitions.

Persistence is a tiny JSONL (one edge per line, hits aggregated) so route knowledge carries
across sessions alongside the solution cache and tapes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import config


@dataclass
class RouteEdge:
    src: tuple                  # level_key the transition fired FROM
    dst: tuple                  # level_key it landed ON
    kind: str                   # "clear" (level finished) | "screen" (area/screen change)
    at: int                     # progress on src where it fired (approach point)
    dst_label: str = ""         # human label of the destination ("1-3", "overworld #62")
    hits: int = 1               # times observed (confidence)

    def skips_ahead(self) -> bool:
        """A warp: the clear jumps MULTIPLE levels ahead (SMB's 1-2 → 4-1 warp zone), not the
        normal +1 progression. Crossing a world boundary (stage 3 → next world's stage 0) is
        ordinary, so a warp needs a ≥2-world jump, or a ≥2-stage jump within a world. Ordinal
        (world, stage, ...) keys only; non-ordinal keys (Zelda screens) are never warps."""
        if self.kind != "clear":
            return False
        s, d = tuple(self.src), tuple(self.dst)
        if not (len(s) >= 2 and len(d) >= 2
                and all(isinstance(v, int) for v in (s[0], s[1], d[0], d[1]))):
            return False
        world_jump = d[0] - s[0]
        stage_jump = d[1] - s[1]
        return world_jump >= 2 or (world_jump == 0 and stage_jump >= 2)


class RouteGraph:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else config.ROUTES_FILE
        self._edges: dict[tuple, RouteEdge] = {}   # (src, dst, kind) -> edge
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                e = RouteEdge(src=tuple(d["src"]), dst=tuple(d["dst"]), kind=d["kind"],
                              at=int(d["at"]), dst_label=d.get("dst_label", ""),
                              hits=int(d.get("hits", 1)))
            except (KeyError, ValueError, TypeError):
                continue   # skip a corrupt line, keep the rest of the map
            self._edges[(e.src, e.dst, e.kind)] = e

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"src": list(e.src), "dst": list(e.dst), "kind": e.kind,
                             "at": e.at, "dst_label": e.dst_label, "hits": e.hits})
                 for e in self._edges.values()]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""))

    def record(self, src: tuple, dst: tuple, kind: str, at: int, dst_label: str = "") -> None:
        """Record one observed transition (idempotent; repeat observations bump `hits`)."""
        if not src or not dst or src == dst:
            return
        k = (tuple(src), tuple(dst), kind)
        e = self._edges.get(k)
        if e is not None:
            e.hits += 1
            e.at = at or e.at
        else:
            self._edges[k] = RouteEdge(src=tuple(src), dst=tuple(dst), kind=kind, at=at,
                                       dst_label=dst_label)
        self._save()

    def edges_from(self, src: tuple) -> list[RouteEdge]:
        return sorted((e for e in self._edges.values() if e.src == tuple(src)),
                      key=lambda e: -e.hits)

    def warps(self) -> list[RouteEdge]:
        """Discovered skips — the strategist's shortcuts to game completion."""
        return [e for e in self._edges.values() if e.skips_ahead()]

    def __len__(self) -> int:
        return len(self._edges)

    def describe(self) -> str:
        """Compact map summary (for logs and LLM route prompts)."""
        parts = []
        for e in sorted(self._edges.values(), key=lambda e: (str(e.src), str(e.dst))):
            arrow = "⤳ WARP" if e.skips_ahead() else "→"
            parts.append(f"{e.src} {arrow} {e.dst} ({e.kind}@{e.at}, x{e.hits})")
        return "\n".join(parts)
