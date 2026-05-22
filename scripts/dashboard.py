#!/usr/bin/env python3
"""Static experiment dashboard generator.

Scans experiments/<name>/<run>/ for metrics summaries + figures and renders a
single self-contained HTML (figures embedded as downscaled base64, so the file
is portable — open locally, no server needed). Optionally serve it.

Usage:
    python scripts/dashboard.py                      # writes dashboard/index.html
    python scripts/dashboard.py --serve 8888         # also serve on :8888
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"


def _thumb_b64(png: Path, max_side: int = 460) -> str | None:
    try:
        from PIL import Image

        im = Image.open(png).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def _load_summary(run: Path) -> dict:
    for cand in ("metrics_summary.json", "phase6_results.json", "results.json"):
        p = run / cand
        if p.exists():
            try:
                return json.load(open(p))
            except Exception:
                pass
    # fall back to any *results*.json in artifacts
    for p in (run / "artifacts").glob("*results*.json"):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}


def _flat(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            rows += _flat(v, key + ".")
        elif isinstance(v, list):
            rows.append((key, json.dumps(v)[:120]))
        else:
            rows.append((key, str(v)))
    return rows


def build() -> str:
    runs = []
    for name_dir in sorted(EXP.iterdir()):
        if not name_dir.is_dir():
            continue
        for run in sorted(name_dir.iterdir(), reverse=True):
            if not run.is_dir():
                continue
            figs = sorted((run / "artifacts").glob("*.png")) if (run / "artifacts").is_dir() else []
            summary = _load_summary(run)
            runs.append((name_dir.name, run.name, summary, figs))

    css = """body{background:#0d1117;color:#c9d1d9;font:14px/1.5 ui-monospace,Menlo,monospace;margin:0;padding:24px}
    h1{color:#58a6ff} .run{border:1px solid #30363d;border-radius:8px;margin:14px 0;padding:14px;background:#161b22}
    .run h2{margin:0 0 4px;color:#79c0ff;font-size:15px} .ts{color:#8b949e;font-size:12px}
    table{border-collapse:collapse;margin:8px 0;font-size:12px} td{border:1px solid #30363d;padding:2px 8px}
    td.k{color:#8b949e} .figs{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}
    .figs figure{margin:0} .figs img{border:1px solid #30363d;border-radius:6px;max-height:240px}
    .figs figcaption{color:#8b949e;font-size:11px} .badge{background:#1f6feb;color:#fff;border-radius:4px;padding:1px 6px;font-size:11px}"""

    parts = [f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=20>"
             f"<title>defense-lab dashboard</title><style>{css}</style>",
             "<h1>defense-lab &mdash; experiment dashboard</h1>",
             f"<p class=ts>SAM2 promptable segmentation pipeline &middot; {len(runs)} runs &middot; GH200 / bf16 "
             "&middot; class-agnostic scene segmentation (neutral scope)</p>"]

    for name, ts, summary, figs in runs:
        parts.append(f"<div class=run><h2>{html.escape(name)} <span class=badge>{len(figs)} figs</span></h2>"
                     f"<div class=ts>{html.escape(ts)}</div>")
        rows = _flat(summary)[:24]
        if rows:
            parts.append("<table>")
            for k, v in rows:
                parts.append(f"<tr><td class=k>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>")
            parts.append("</table>")
        if figs:
            parts.append("<div class=figs>")
            for f in figs[:8]:
                b64 = _thumb_b64(f)
                if b64:
                    parts.append(f"<figure><img src='data:image/jpeg;base64,{b64}'>"
                                 f"<figcaption>{html.escape(f.name)}</figcaption></figure>")
            parts.append("</div>")
        parts.append("</div>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dashboard" / "index.html"))
    ap.add_argument("--serve", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build())
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({size_kb:.0f} KB)")

    if args.serve:
        import http.server
        import os
        import socketserver

        os.chdir(out.parent)
        with socketserver.TCPServer(("0.0.0.0", args.serve), http.server.SimpleHTTPRequestHandler) as httpd:
            print(f"serving {out.parent} at http://0.0.0.0:{args.serve}  (Ctrl-C to stop)")
            httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
