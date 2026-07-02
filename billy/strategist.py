"""Route strategist — turns the recorded route graph into a plan toward game completion.

`knowledge/routes.py` RECORDS every transition Billy observes; this layer DECIDES with it.
Given where Billy is, it plans the path over known edges that reaches the furthest level in the
fewest hops — so a discovered WARP (an edge that skips ahead, e.g. SMB's 1-2 warp zone) is
preferred over the sequential grind. It never invents edges: with no warp known the plan is
just the linear march Billy already walks, and the strategist's teeth (warp preference) engage
the moment a warp is recorded.

Advice, not authority — like the guide. The strategist names the next objective and feeds the
route map to the LLM; the reflex/cache/tape/search loop still owns how each level is actually
crossed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .knowledge.routes import RouteEdge, RouteGraph


def _default_rank(level_key: tuple) -> tuple:
    """How far toward completion a level is. Ordinal (world, stage, ...) keys rank by value;
    non-ordinal keys (Zelda realms) are unrankable here -> () (frontier-exploration falls back
    to hop count). Games can pass a custom rank for their own progress semantics."""
    if isinstance(level_key, tuple) and level_key and all(isinstance(v, int) for v in level_key):
        return level_key
    return ()


@dataclass
class Objective:
    kind: str            # "advance" (take the normal exit) | "warp" (a known skip-ahead) | "explore"
    target: tuple | None # the next level_key on the optimal path (None if unknown)
    target_label: str    # human label of the target
    via_warp: bool       # the next hop is a discovered warp

    def line(self) -> str:
        if self.target is None:
            return "🧭 objective: advance (no route learned yet — march forward)"
        tag = "⤳ WARP" if self.via_warp else "→"
        return f"🧭 objective: {tag} {self.target_label or self.target}"


class RouteStrategist:
    def __init__(self, routes: RouteGraph, rank=_default_rank) -> None:
        self.routes = routes
        self.rank = rank

    # --- planning over the known graph --------------------------------------------------
    def best_path(self, start: tuple) -> list[tuple]:
        """Path (list of level_keys, start first) over KNOWN edges that reaches the highest-rank
        level; ties broken by fewest hops, then by preferring a warp on the first hop. Returns
        [start] when nothing is known from here."""
        start = tuple(start)
        # BFS collecting the best path to every reachable node (fewest hops; warp-first tie-break).
        best: dict[tuple, list[tuple]] = {start: [start]}
        frontier = [start]
        while frontier:
            nxt: list[tuple] = []
            for node in frontier:
                edges = sorted(self.routes.edges_from(node),
                               key=lambda e: (not e.skips_ahead(), -e.hits))
                for e in edges:
                    if e.dst not in best or len(best[node]) + 1 < len(best[e.dst]):
                        best[e.dst] = best[node] + [e.dst]
                        nxt.append(e.dst)
            frontier = nxt
        # Target = reachable node with the max rank (warps win here — they reach a higher-ranked
        # level). Among EQUAL rank (e.g. non-ordinal Zelda keys, all unrankable), prefer the
        # DEEPEST node so the plan still points at the frontier instead of standing still.
        target = max(best, key=lambda k: (self.rank(k), len(best[k])))
        return best[target]

    def next_hop(self, cur: tuple) -> tuple | None:
        """The next level_key Billy should aim for from `cur` on the optimal path."""
        path = self.best_path(cur)
        return path[1] if len(path) > 1 else None

    def _edge(self, src: tuple, dst: tuple) -> RouteEdge | None:
        for e in self.routes.edges_from(src):
            if e.dst == tuple(dst):
                return e
        return None

    def objective(self, cur: tuple, cur_label: str = "") -> Objective:
        """What Billy should aim for from the current level."""
        nxt = self.next_hop(cur)
        if nxt is None:
            return Objective(kind="advance", target=None, target_label="", via_warp=False)
        edge = self._edge(cur, nxt)
        via_warp = bool(edge and edge.skips_ahead())
        label = edge.dst_label if edge else ""
        return Objective(kind="warp" if via_warp else "advance",
                         target=nxt, target_label=label, via_warp=via_warp)

    # --- surfacing for logs + the LLM ---------------------------------------------------
    def plan_labels(self, cur: tuple) -> list[str]:
        """Human labels along the optimal path (for logs / prompts)."""
        path = self.best_path(cur)
        labels = []
        for i in range(1, len(path)):
            e = self._edge(path[i - 1], path[i])
            arrow = "⤳" if (e and e.skips_ahead()) else "→"
            labels.append(f"{arrow} {e.dst_label if e and e.dst_label else path[i]}")
        return labels

    def prompt_section(self, cur: tuple) -> str:
        """Route plan for the LLM prompt (advice — the loop verifies everything)."""
        labels = self.plan_labels(cur)
        if not labels:
            return ""
        warps = self.routes.warps()
        head = "\nRoute plan (known map, warps preferred toward game completion):\n  " + \
               " ".join(labels)
        if warps:
            head += "\n  known warps: " + ", ".join(
                f"{w.src}⤳{w.dst_label or w.dst}" for w in warps)
        return head
