"""Self-contained HTML case report: image, highlight overlay, regions, stream and ledger.

There are no plotting dependencies: PNGs are encoded with zlib and embedded as
data URIs, so a report is one file you can open anywhere.
"""
from __future__ import annotations

import base64
import html
import struct
import zlib

import numpy as np


def png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _heat(v: np.ndarray) -> np.ndarray:
    """Perceptually ordered black-red-yellow-white ramp for v in [0, 1]."""
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[0, 0, 0], [180, 20, 20], [250, 180, 30], [255, 255, 220]], float)
    x = v * (len(stops) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(stops) - 2)
    f = x - i
    return stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f


def render_panels(image: np.ndarray, highlight: np.ndarray, regions, threshold: float, scale: int = 5):
    g = np.repeat(np.clip(image, 0, 1)[..., None] * 255, 3, -1)
    heat = _heat(highlight / max(threshold * 2, 1e-6))
    alpha = np.clip(highlight / max(threshold, 1e-6) - 0.5, 0, 1)[..., None] * 0.75
    over = g * (1 - alpha) + heat * alpha
    for r in regions:
        y0, x0, y1, x1 = r.bbox
        h, w = highlight.shape  # pad tiny (single-cell) boxes so they stay visible
        y0, x0, y1, x1 = max(y0 - 2, 0), max(x0 - 2, 0), min(y1 + 2, h), min(x1 + 2, w)
        over[y0, x0:x1] = over[y1 - 1, x0:x1] = [0, 200, 255]
        over[y0:y1, x0] = over[y0:y1, x1 - 1] = [0, 200, 255]
    up = lambda a: np.kron(a, np.ones((scale, scale, 1)))
    return png_bytes(up(g)), png_bytes(up(over)), png_bytes(up(heat))


def _pct(p: float) -> str:
    return ">99.9%" if p > 0.999 else "<0.1%" if 0 < p < 0.001 else f"{100 * p:.1f}%"


def _uri(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--card:#fff;--line:#e4e2dc;--good:#1f7a4d;--bad:#b3261e;--warn:#a15c00}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--card:#1f1f1d;--line:#34332f;--good:#5ec28f;--bad:#f2837a;--warn:#e7a54a}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}
.imgs{display:flex;gap:12px;flex-wrap:wrap}.imgs figure{margin:0}.imgs img{width:min(280px,100%);image-rendering:pixelated;border-radius:6px}
figcaption{font-size:12px;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}
td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}th{color:var(--muted);font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}.pos{color:var(--good)}.neg{color:var(--bad)}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid currentColor}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>i{display:block;height:100%;background:var(--fg)}
pre{white-space:pre-wrap;font-size:12.5px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
.table-wrap{overflow-x:auto}
"""


def html_report(analysis, image: np.ndarray, disclaimer: str, truth: str | None = None) -> str:
    a = analysis
    e = html.escape
    raw, over, heat = render_panels(image, a.highlight, a.regions, a.extra["region_threshold"])
    p = a.probabilities[a.prediction]
    badge = ('<span class="badge pos">answered</span>' if a.answered
             else '<span class="badge neg">abstained: refer to specialist</span>')
    probs = "".join(f"<tr><td>{e(c)}</td><td class='num'>{_pct(v)}</td>"
                    f"<td style='width:40%'><div class='bar'><i style='width:{100 * v:.1f}%'></i></div></td></tr>"
                    for c, v in sorted(a.probabilities.items(), key=lambda kv: -kv[1]))
    base = a.component_probs["concept_evidence"][a.prediction]

    def role(r):
        drop = base - (base if r.p_top_without is None else r.p_top_without)
        return "drives conclusion" if drop > 0.2 else "contributes" if drop > 0.05 else "considered, dismissed"

    regions = "".join(
        f"<tr><td>{r.region_id}</td><td>rows {r.bbox[0]}-{r.bbox[2]}, cols {r.bbox[1]}-{r.bbox[3]}</td>"
        f"<td class='num'>{r.area_px}</td><td class='num'>{r.peak:.2f}x</td><td>{e(r.dominant_concept)}</td>"
        f"<td class='num'>{'' if r.p_top_without is None else _pct(r.p_top_without)}</td><td>{role(r)}</td></tr>"
        for r in a.regions) or "<tr><td colspan=7 class='muted'>No candidate region beyond the normal reference.</td></tr>"
    ev = "".join(
        f"<tr><td>{e(x.concept)}</td><td class='num'>{x.value:.2f}x</td>"
        f"<td class='num {'pos' if x.woe_db > 0 else 'neg'}'>{x.woe_db:+.1f}</td><td>{e(x.verdict)}</td>"
        f"<td class='muted'>{e(x.description)}</td></tr>" for x in a.evidence)
    stream = "".join(
        f"<tr><td class='num'>{s['step']}</td><td>({s['patch_row']}, {s['patch_col']})</td>"
        f"<td class='num'>{s['saliency']:.2f}x</td><td>{e(s['top'])}</td><td class='num'>{_pct(s['p_top'])}</td>"
        f"<td class='num'>{s['halt']:.2f}</td></tr>" for s in a.stream)
    truth_html = f" &middot; ground truth: <b>{e(truth)}</b>" if truth else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>OncoPattern case {e(a.case_id)}</title>
<style>{CSS}</style></head><body><main>
<h1>Case {e(a.case_id)}: {e(a.prediction)} {badge}</h1>
<div class="muted">calibrated probability {_pct(p)} &middot; plausible set: {e(', '.join(a.prediction_set))}
&middot; read {a.tokens_read}/{a.tokens_total} patches{truth_html}</div>
<h2>Where the model looked</h2>
<div class="imgs card"><figure><img src="{_uri(raw)}" alt="input"><figcaption>Input</figcaption></figure>
<figure><img src="{_uri(over)}" alt="overlay"><figcaption>Deviation overlay with flagged regions (boxes)</figcaption></figure>
<figure><img src="{_uri(heat)}" alt="heat"><figcaption>Deviation from normal reference</figcaption></figure></div>
<h2>Hypotheses</h2><div class="card table-wrap"><table>{probs}</table></div>
<h2>Candidate regions, ranked by counterfactual impact (P = concept evidence for {e(a.prediction)}, now {_pct(base)}, if the region looked normal)</h2>
<div class="card table-wrap"><table><tr><th>id</th><th>location</th><th class="num">px</th><th class="num">peak</th><th>dominant concept</th><th class="num">P without</th><th>role</th></tr>{regions}</table></div>
<h2>Evidence ledger: {e(a.prediction)} vs {e(a.rival)} (decibans)</h2>
<div class="card table-wrap"><table><tr><th>concept</th><th class="num">peak</th><th class="num">WoE</th><th>verdict</th><th>meaning</th></tr>{ev}</table></div>
<h2>Full-duplex stream (belief after each patch)</h2>
<div class="card table-wrap"><table><tr><th class="num">step</th><th>patch</th><th class="num">saliency</th><th>belief</th><th class="num">p</th><th class="num">halt</th></tr>{stream}</table></div>
<h2>Narrative (every sentence generated from the numbers above)</h2><pre>{e(a.narrative)}</pre>
<p class="muted">{e(disclaimer)}</p></main></body></html>"""
