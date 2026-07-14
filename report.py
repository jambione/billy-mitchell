#!/usr/bin/env python3
"""Render data/metrics.jsonl (+ banked Remix demos) into a static learning-curve report.

Phase 6 of ROADMAP_REMIX.md: "one command answers 'is Billy learning faster than last
week?' with a graph, not an anecdote." No server, no external assets — the output is a
single self-contained HTML file; open it directly in a browser.

    .venv/bin/python report.py                # writes data/report.html
    .venv/bin/python report.py --open          # also opens it
"""
from __future__ import annotations

import argparse
import json
import re
import webbrowser
from pathlib import Path

from billy import config

# --- palette (billy-mitchell/.claude dataviz reference instance) --------------------------
SLOT = {
    "blue": ("#2a78d6", "#3987e5"),
    "aqua": ("#1baf7a", "#199e70"),
    "yellow": ("#eda100", "#c98500"),
    "violet": ("#4a3aa7", "#9085e9"),
}
GOOD, WARN = ("#0ca30c", "#0ca30c"), ("#fab219", "#fab219")


# --- data loading -----------------------------------------------------------------------

def load_attempts() -> list[dict]:
    if not config.METRICS_FILE.is_file():
        return []
    out = []
    with config.METRICS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_demos() -> dict[str, list[dict]]:
    """{game_id: [{level, world, mtime}, ...]} — one entry per banked Remix/teleop demo."""
    demos_dir = config.DATA_DIR / "rl" / "demos"
    out: dict[str, list[dict]] = {}
    if not demos_dir.is_dir():
        return out
    for game_dir in sorted(demos_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        entries = []
        for demo_path in sorted(game_dir.glob("*.demo.json")):
            m = re.match(r"^(.*)_x\d+\.demo\.json$", demo_path.name)
            slug = m.group(1) if m else demo_path.stem
            label = slug.replace("_", "-")
            entries.append({
                "level": label,
                "world": world_of(label),
                "mtime": demo_path.stat().st_mtime,
            })
        if entries:
            out[game_dir.name] = entries
    return out


def world_of(label: str) -> str:
    m = re.match(r"^(\d+)-", label)
    if m:
        return m.group(1)
    return re.split(r"[ #]", label, 1)[0] or label


# --- aggregation ------------------------------------------------------------------------

def group_by_game(attempts: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for a in attempts:
        groups.setdefault(a.get("game") or "legacy (untagged)", []).append(a)
    return groups


def top_levels(attempts: list[dict], n: int = 4) -> list[str]:
    counts: dict[str, int] = {}
    for a in attempts:
        lvl = a.get("world_stage") or "?"
        counts[lvl] = counts.get(lvl, 0) + 1
    return [lvl for lvl, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:n]]


# --- tiny SVG chart builders (thin marks, hairline grid, hover crosshair) ---------------

def _nice_ceiling(v: float) -> float:
    if v <= 0:
        return 1
    import math
    mag = 10 ** math.floor(math.log10(v))
    for step in (1, 2, 2.5, 5, 10):
        if v <= step * mag:
            return step * mag
    return 10 * mag


_CHART_SEQ = [0]


def line_chart(title: str, subtitle: str, x_labels: list[str],
                series: list[tuple[str, str, list[float]]], *,
                width: int = 640, height: int = 220) -> str:
    """series: list of (name, slot_key, values) — one line per series, single y-axis."""
    _CHART_SEQ[0] += 1
    cid = f"lc{_CHART_SEQ[0]}"
    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 24
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(x_labels)
    all_vals = [v for _, _, vs in series for v in vs] or [0]
    y_max = _nice_ceiling(max(all_vals))

    def xpos(i: int) -> float:
        return pad_l + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)

    def ypos(v: float) -> float:
        return pad_t + plot_h - (plot_h * v / y_max if y_max else 0)

    grid = "".join(
        f'<line class="gridline" x1="{pad_l}" x2="{width - pad_r}" '
        f'y1="{pad_t + plot_h * f:.1f}" y2="{pad_t + plot_h * f:.1f}"/>'
        f'<text class="tick" x="{pad_l - 6}" y="{pad_t + plot_h * f + 4:.1f}" '
        f'text-anchor="end">{int(y_max * (1 - f)):,}</text>'
        for f in (0.0, 0.5, 1.0)
    )

    lines, dots, legend = [], [], []
    for si, (name, slot, values) in enumerate(series):
        color_var = f"var(--series-{slot})"
        pts = " ".join(f"{xpos(i):.1f},{ypos(v):.1f}" for i, v in enumerate(values))
        lines.append(f'<polyline class="series-line" data-slot="{slot}" points="{pts}" '
                      f'style="stroke:{color_var}"/>')
        if values:
            ex, ey = xpos(len(values) - 1), ypos(values[-1])
            dots.append(f'<circle class="end-dot" cx="{ex:.1f}" cy="{ey:.1f}" r="4" '
                         f'style="fill:{color_var}"/>')
        legend.append(f'<span class="legend-item"><span class="legend-swatch" '
                       f'style="background:{color_var}"></span>{name}</span>')

    hit_dots = "".join(
        f'<circle class="hover-dot" data-series="{si}" r="5" style="fill:{f"var(--series-{slot})"}" '
        f'opacity="0"/>'
        for si, (_, slot, _) in enumerate(series))

    payload = json.dumps({
        "xLabels": x_labels,
        "series": [{"name": name, "values": vs} for name, _, vs in series],
        "xpos": [round(xpos(i), 1) for i in range(n)],
        "ys": [[round(ypos(v), 1) for v in vs] for _, _, vs in series],
    })

    legend_html = f'<div class="legend">{"".join(legend)}</div>' if len(series) > 1 else ""
    table_rows = "".join(
        f"<tr><td>{x_labels[i]}</td>" + "".join(
            f"<td>{vs[i]:,}</td>" for _, _, vs in series) + "</tr>"
        for i in range(n))
    table_head = "<th>attempt</th>" + "".join(f"<th>{name}</th>" for name, _, _ in series)

    return f"""
<div class="chart-card">
  <div class="chart-head"><h3>{title}</h3><p class="subtitle">{subtitle}</p></div>
  {legend_html}
  <svg class="line-chart" id="{cid}" viewBox="0 0 {width} {height}" data-payload='{payload}'>
    {grid}
    <line class="baseline" x1="{pad_l}" x2="{width - pad_r}" y1="{pad_t + plot_h}" y2="{pad_t + plot_h}"/>
    {"".join(lines)}
    {"".join(dots)}
    <line class="crosshair" x1="0" x2="0" y1="{pad_t}" y2="{pad_t + plot_h}" opacity="0"/>
    {hit_dots}
    <rect class="hit-layer" x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}"/>
  </svg>
  <div class="tooltip" id="{cid}-tip"></div>
  <details class="table-view"><summary>Table view</summary>
    <table><thead><tr>{table_head}</tr></thead><tbody>{table_rows}</tbody></table>
  </details>
</div>"""


def bar_chart(title: str, subtitle: str, categories: list[str], values: list[float],
              *, slot: str = "blue", width: int = 640, height: int = 200) -> str:
    _CHART_SEQ[0] += 1
    cid = f"bc{_CHART_SEQ[0]}"
    pad_l, pad_r, pad_t, pad_b = 40, 12, 12, 28
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = max(len(categories), 1)
    y_max = _nice_ceiling(max(values) if values else 1)
    band = plot_w / n
    bar_w = min(24, band * 0.6)

    grid = "".join(
        f'<line class="gridline" x1="{pad_l}" x2="{width - pad_r}" '
        f'y1="{pad_t + plot_h * f:.1f}" y2="{pad_t + plot_h * f:.1f}"/>'
        f'<text class="tick" x="{pad_l - 6}" y="{pad_t + plot_h * f + 4:.1f}" '
        f'text-anchor="end">{int(y_max * (1 - f)):,}</text>'
        for f in (0.0, 0.5, 1.0)
    )
    bars = []
    for i, (cat, v) in enumerate(zip(categories, values)):
        cx = pad_l + band * i + band / 2
        bh = plot_h * (v / y_max if y_max else 0)
        by = pad_t + plot_h - bh
        bars.append(
            f'<rect class="bar" data-label="{cat}" data-value="{v:,.0f}" '
            f'x="{cx - bar_w / 2:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(bh, 1):.1f}" '
            f'rx="4" style="fill:var(--series-{slot})"/>'
            f'<text class="tick cat-label" x="{cx:.1f}" y="{pad_t + plot_h + 16}" '
            f'text-anchor="middle">{cat}</text>')
    table_rows = "".join(f"<tr><td>{c}</td><td>{v:,.0f}</td></tr>" for c, v in zip(categories, values))

    return f"""
<div class="chart-card">
  <div class="chart-head"><h3>{title}</h3><p class="subtitle">{subtitle}</p></div>
  <svg class="bar-chart" id="{cid}" viewBox="0 0 {width} {height}">
    {grid}
    <line class="baseline" x1="{pad_l}" x2="{width - pad_r}" y1="{pad_t + plot_h}" y2="{pad_t + plot_h}"/>
    {"".join(bars)}
  </svg>
  <div class="tooltip" id="{cid}-tip"></div>
  <details class="table-view"><summary>Table view</summary>
    <table><thead><tr><th>world</th><th>demos taught</th></tr></thead>
    <tbody>{table_rows}</tbody></table>
  </details>
</div>"""


def stat_tile(label: str, value: str, trend: str = "") -> str:
    trend_html = f'<div class="stat-trend">{trend}</div>' if trend else ""
    return f'<div class="stat-tile"><div class="stat-label">{label}</div>' \
           f'<div class="stat-value">{value}</div>{trend_html}</div>'


# --- page assembly ------------------------------------------------------------------------

CSS = """
:root {
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --warn: #fab219;
  --series-blue: #2a78d6; --series-aqua: #1baf7a; --series-yellow: #eda100; --series-violet: #4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #ffffff; --text-secondary: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --good: #0ca30c; --warn: #fab219;
    --series-blue: #3987e5; --series-aqua: #199e70; --series-yellow: #c98500; --series-violet: #9085e9;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 24px 64px; background: var(--page); color: var(--text-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 40px 0 12px; color: var(--text-primary); }
h3 { font-size: 14px; margin: 0; font-weight: 600; }
.lede { color: var(--text-secondary); margin: 0 0 24px; max-width: 720px; }
.note { color: var(--muted); font-size: 12px; margin: -8px 0 20px; }
.wrap { max-width: 1040px; margin: 0 auto; }
.stat-row { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }
.stat-tile {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px; min-width: 140px; flex: 1 1 140px;
}
.stat-label { color: var(--text-secondary); font-size: 12px; }
.stat-value { font-size: 26px; font-weight: 600; margin-top: 2px; }
.stat-trend { font-size: 12px; margin-top: 4px; }
.stat-trend.good { color: var(--good); }
.stat-trend.warn { color: var(--warn); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.chart-card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 16px 16px 8px; position: relative;
}
.chart-head p.subtitle { margin: 2px 0 10px; color: var(--text-secondary); font-size: 12px; }
.legend { display: flex; gap: 14px; margin-bottom: 6px; font-size: 12px; color: var(--text-secondary); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
svg.line-chart, svg.bar-chart { width: 100%; height: auto; overflow: visible; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.baseline { stroke: var(--baseline); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 9px; }
.cat-label { font-size: 10px; }
.series-line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.end-dot { stroke: var(--surface-1); stroke-width: 2; }
.hover-dot { stroke: var(--surface-1); stroke-width: 2; pointer-events: none; }
.crosshair { stroke: var(--muted); stroke-width: 1; pointer-events: none; }
.hit-layer { fill: transparent; cursor: crosshair; }
.bar { transition: filter 0.1s; }
.bar:hover, .bar:focus { filter: brightness(1.12); outline: none; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--text-primary); color: var(--page);
  padding: 6px 10px; border-radius: 6px; font-size: 12px; opacity: 0; transform: translate(-50%, -110%);
  white-space: nowrap; transition: opacity 0.06s; z-index: 5;
}
.tooltip.show { opacity: 0.95; }
.tooltip b { font-variant-numeric: tabular-nums; }
.table-view { margin-top: 8px; font-size: 12px; }
.table-view summary { cursor: pointer; color: var(--text-secondary); padding: 6px 0; }
.table-view table { border-collapse: collapse; width: 100%; margin-bottom: 12px; }
.table-view td, .table-view th {
  text-align: right; padding: 3px 8px; border-bottom: 1px solid var(--grid);
  font-variant-numeric: tabular-nums;
}
.table-view th:first-child, .table-view td:first-child { text-align: left; }
"""

JS = """
function fmt(v) { return typeof v === 'number' ? v.toLocaleString() : v; }

document.querySelectorAll('svg.line-chart').forEach(function (svg) {
  var payload = JSON.parse(svg.dataset.payload);
  var tip = document.getElementById(svg.id + '-tip');
  var crosshair = svg.querySelector('.crosshair');
  var hoverDots = svg.querySelectorAll('.hover-dot');
  var hit = svg.querySelector('.hit-layer');

  function nearestIndex(clientX) {
    var rect = svg.getBoundingClientRect();
    var scale = svg.viewBox.baseVal.width / rect.width;
    var localX = (clientX - rect.left) * scale;
    var best = 0, bestDist = Infinity;
    payload.xpos.forEach(function (x, i) {
      var d = Math.abs(x - localX);
      if (d < bestDist) { bestDist = d; best = i; }
    });
    return best;
  }

  function show(clientX, clientY) {
    var i = nearestIndex(clientX);
    var x = payload.xpos[i];
    crosshair.setAttribute('x1', x); crosshair.setAttribute('x2', x); crosshair.setAttribute('opacity', 1);
    var rows = payload.series.map(function (s, si) {
      hoverDots[si].setAttribute('cx', x);
      hoverDots[si].setAttribute('cy', payload.ys[si][i]);
      hoverDots[si].setAttribute('opacity', 1);
      return '<div>' + s.name + ': <b>' + fmt(s.values[i]) + '</b></div>';
    }).join('');
    tip.innerHTML = '<div>' + payload.xLabels[i] + '</div>' + rows;
    var svgRect = svg.getBoundingClientRect();
    tip.style.left = (clientX - svgRect.left) + 'px';
    tip.style.top = (clientY - svgRect.top) + 'px';
    tip.classList.add('show');
  }
  function hide() {
    crosshair.setAttribute('opacity', 0);
    hoverDots.forEach(function (d) { d.setAttribute('opacity', 0); });
    tip.classList.remove('show');
  }
  hit.addEventListener('pointermove', function (e) { show(e.clientX, e.clientY); });
  hit.addEventListener('pointerleave', hide);
});

document.querySelectorAll('svg.bar-chart').forEach(function (svg) {
  var tip = document.getElementById(svg.id + '-tip');
  svg.querySelectorAll('.bar').forEach(function (bar) {
    bar.setAttribute('tabindex', '0');
    function show(clientX, clientY) {
      tip.innerHTML = '<div>' + bar.dataset.label + '</div><div><b>' + fmt(Number(bar.dataset.value)) + '</b> taught</div>';
      var svgRect = svg.getBoundingClientRect();
      tip.style.left = (clientX - svgRect.left) + 'px';
      tip.style.top = (clientY - svgRect.top) + 'px';
      tip.classList.add('show');
    }
    bar.addEventListener('pointermove', function (e) { show(e.clientX, e.clientY); });
    bar.addEventListener('pointerleave', function () { tip.classList.remove('show'); });
    bar.addEventListener('focus', function () {
      var r = bar.getBoundingClientRect();
      show(r.left + r.width / 2, r.top);
    });
    bar.addEventListener('blur', function () { tip.classList.remove('show'); });
  });
});
"""


def build_game_section(game_id: str, attempts: list[dict], demos: list[dict]) -> str:
    attempts_seq = list(enumerate(attempts, start=1))
    search_vals = [a.get("search_calls", 0) for a in attempts]
    replay_vals = [a.get("replay_calls", 0) for a in attempts]
    frontier_vals = [a.get("level_frontier") or a.get("frontier_x", 0) for a in attempts]
    x_labels = [str(i) for i, _ in attempts_seq]

    reached = max((a.get("max_x", 0) for a in attempts), default=0)
    stats = [stat_tile("attempts recorded", f"{len(attempts):,}")]
    charts: list[str] = []

    if not attempts:
        stats.append(stat_tile("demos taught", f"{len(demos):,}"))
        verdict_cls, verdict_txt = "warn", "no tagged attempts yet — re-run to start collecting"
    else:
        first_r, last_r = replay_vals[0], replay_vals[-1]
        first_f, last_f = frontier_vals[0], frontier_vals[-1]
        compounding = last_r >= first_r and last_f >= first_f
        verdict_cls = "good" if compounding else "warn"
        verdict_txt = "✅ compounding: replay↑ frontier↑" if compounding else "⚠️ not compounding yet"
        stats += [
            stat_tile("furthest reached (px)", f"{reached:,}"),
            stat_tile("demos taught", f"{len(demos):,}"),
        ]
        charts = [line_chart(
            "Search vs. replay", "Per attempt, in play order — search should fall, replay should rise",
            x_labels, [("search (new)", "blue", search_vals), ("replay (cached)", "aqua", replay_vals)],
        ), line_chart(
            "Solved frontier", "Furthest px banked as a verified crossing, per attempt",
            x_labels, [("frontier", "violet", frontier_vals)],
        )]
        for lvl in top_levels(attempts):
            lvl_attempts = [a for a in attempts if (a.get("world_stage") or "?") == lvl]
            lx = [str(i) for i in range(1, len(lvl_attempts) + 1)]
            ls = [a.get("search_calls", 0) for a in lvl_attempts]
            lr = [a.get("replay_calls", 0) for a in lvl_attempts]
            charts.append(line_chart(
                f"{lvl} — search vs. replay", f"{len(lvl_attempts)} attempts at this level",
                lx, [("search", "blue", ls), ("replay", "aqua", lr)], width=320, height=180,
            ))

    if demos:
        world_counts: dict[str, int] = {}
        for d in sorted(demos, key=lambda d: d["mtime"]):
            world_counts[d["world"]] = world_counts.get(d["world"], 0) + 1
        cats = sorted(world_counts.keys(), key=lambda w: (len(w), w))
        charts.append(bar_chart(
            "Demos taught per world", "One human crossing per bar — should fall as Billy outgrows the teacher",
            cats, [world_counts[c] for c in cats], slot="yellow", width=320, height=180,
        ))

    return f"""
<h2>{game_id} <span class="stat-trend {verdict_cls}">{verdict_txt}</span></h2>
<div class="stat-row">{stats}</div>
<div class="grid">{"".join(charts)}</div>
"""


def build_report() -> str:
    attempts = load_attempts()
    demos_by_game = load_demos()
    groups = group_by_game(attempts)

    all_games = (set(groups) | set(demos_by_game)) - {"legacy (untagged)"}
    order = sorted(all_games, key=lambda g: -(len(groups.get(g, [])) + len(demos_by_game.get(g, []))))
    if "legacy (untagged)" in groups:
        order.append("legacy (untagged)")
    sections = "".join(
        build_game_section(g, groups.get(g, []), demos_by_game.get(g, [])) for g in order)

    legacy_note = ""
    if "legacy (untagged)" in groups:
        legacy_note = ('<p class="note">"legacy (untagged)" is every attempt recorded before '
                        'AttemptResult carried a <code>game</code> field — those records predate '
                        'per-game tagging and can\'t be split out honestly. New attempts tag '
                        'themselves automatically.</p>')

    total_demos = sum(len(v) for v in demos_by_game.values())
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Billy Mitchell — learning curves</title>
<style>{CSS}</style>
</head><body>
<div class="wrap">
  <h1>Billy Mitchell — learning curves</h1>
  <p class="lede">Generated from data/metrics.jsonl ({len(attempts):,} attempts) and
  data/rl/demos ({total_demos:,} banked demos). Phase 6 of ROADMAP_REMIX.md — this replaces
  eyeballing the search↓/replay↑ table with a graph.</p>
  {legacy_note}
  {sections}
</div>
<script>{JS}</script>
</body></html>"""
    return html


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true", help="open the report after writing it")
    args = ap.parse_args()

    out = config.DATA_DIR / "report.html"
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report())
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
