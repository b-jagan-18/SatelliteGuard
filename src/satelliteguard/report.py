"""Build the technical report: template -> Markdown (numbers from reports/*.json) -> HTML -> PDF.

Every metric in the report is a placeholder such as ``{{ test.hgb.overall.pr_auc:.3f }}``,
resolved from the single source of truth (metrics.json, comparisons.json, leakage_check.json,
test_lock.json). An unresolved placeholder is an error, so report and results cannot drift.

    python -m satelliteguard.report              # -> reports/technical_report.{md,html,pdf}
    python -m satelliteguard.report --no-pdf     # Markdown + HTML only
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import mistune

from satelliteguard import PROJECT_ROOT, load_config

REPORTS = PROJECT_ROOT / "reports"
TEMPLATE = PROJECT_ROOT / "docs" / "report" / "technical_report.md"
CSS = PROJECT_ROOT / "docs" / "report" / "report.css"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*(?::\s*([^}]*?))?\s*\}\}")
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser", "msedge",
]


def context(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    m = read(cfg["paths"]["metrics"])
    return {"cv": m["cv"], "test": m["test"], "cmp": read(REPORTS / "comparisons.json"),
            "leak": read(REPORTS / "leakage_check.json"), "lock": read(REPORTS / "test_lock.json")}


def lookup(ctx: dict, path: str):
    node = ctx
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise KeyError(f"report placeholder '{path}': '{part}' not found")
    return node


def render_template(text: str, ctx: dict) -> str:
    """Replace every {{ path[:fmt] }}. fmt is a Python format spec; '%' multiplies by 100."""
    def sub(mt: re.Match) -> str:
        value, fmt = lookup(ctx, mt.group(1)), (mt.group(2) or "").strip()
        if fmt.endswith("%") and not fmt.endswith("%%"):
            return format(100 * value, fmt[:-1] + "f") + "%"
        return format(value, fmt) if fmt else str(value)
    out = PLACEHOLDER.sub(sub, text)
    if "{{" in out:
        raise ValueError("unresolved placeholder left in report")
    return out


def to_html(md: str, title: str) -> str:
    body = mistune.create_markdown(escape=False, plugins=["table", "footnotes"])(md)
    # Wrap "![alt](src)" + following "*Figure n. ...*" paragraph into a <figure>.
    body = re.sub(r"<p><img ([^>]*)/?>\s*</p>\s*<p><em>(Figure[^<]*(?:<[^/][^>]*>[^<]*</[^>]+>"
                  r"[^<]*)*)</em></p>",
                  r'<figure><img \1/><figcaption>\2</figcaption></figure>', body)
    css = CSS.read_text(encoding="utf-8")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
            f'<base href="{REPORTS.as_uri()}/"><style>{css}</style></head>'
            f"<body>{body}</body></html>")


# ------------------------------------------------------------------ PDF via Chrome DevTools
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise SystemExit("Chrome/Edge not found: install one or use --no-pdf")


async def _print_pdf(html_path: Path, pdf_path: Path, port: int, title: str) -> None:
    from tornado.websocket import websocket_connect

    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
    ws = await websocket_connect(next(t for t in info if t["type"] == "page")
                                 ["webSocketDebuggerUrl"], max_message_size=1 << 29)
    ids = iter(range(1, 10_000))

    async def call(method, **params):
        i = next(ids)
        await ws.write_message(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            msg = json.loads(await ws.read_message())
            if msg.get("id") == i:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    await call("Page.enable")
    await call("Page.navigate", url=html_path.as_uri())
    for _ in range(120):  # wait for images
        r = await call("Runtime.evaluate", returnByValue=True, expression=(
            "document.readyState === 'complete' && [...document.images].every(i => i.complete)"))
        if r["result"].get("value"):
            break
        await asyncio.sleep(0.5)
    small = "font-size:8px;color:#777;width:100%;padding:0 15mm;font-family:sans-serif"
    res = await call(
        "Page.printToPDF", printBackground=True, preferCSSPageSize=True,
        displayHeaderFooter=True,
        headerTemplate=f'<div style="{small};text-align:right">{title}</div>',
        footerTemplate=f'<div style="{small};text-align:center">'
                       '<span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    pdf_path.write_bytes(base64.b64decode(res["data"]))


def html_to_pdf(html_path: Path, pdf_path: Path, title: str) -> None:
    port, profile = _free_port(), tempfile.mkdtemp()
    chrome = subprocess.Popen(
        [_find_chrome(), "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
         f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2)
                break
            except OSError:
                time.sleep(0.5)
        asyncio.run(_print_pdf(html_path, pdf_path, port, title))
    finally:
        chrome.terminate()
        try:
            chrome.wait(10)
        except subprocess.TimeoutExpired:
            chrome.kill()
        shutil.rmtree(profile, ignore_errors=True)


def build(pdf: bool = True) -> dict[str, Path]:
    title = "SatelliteGuard — technical report"
    md = render_template(TEMPLATE.read_text(encoding="utf-8"), context())
    out = {"md": REPORTS / "technical_report.md", "html": REPORTS / "technical_report.html"}
    out["md"].write_text(md, encoding="utf-8")
    out["html"].write_text(to_html(md, title), encoding="utf-8")
    if pdf:
        out["pdf"] = REPORTS / "technical_report.pdf"
        html_to_pdf(out["html"], out["pdf"], title)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-pdf", action="store_true")
    for kind, path in build(pdf=not p.parse_args().no_pdf).items():
        print(f"{kind:<5} {path.relative_to(PROJECT_ROOT)}  {path.stat().st_size / 1e3:,.0f} kB")


if __name__ == "__main__":
    main()
