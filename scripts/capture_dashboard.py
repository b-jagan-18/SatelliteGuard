"""Capture full-height PNG screenshots of every dashboard page (for the technical report).

Starts its own light-themed Streamlit server, drives headless Chrome over the DevTools protocol,
waits until each page has finished running, then captures the whole page.

    python scripts/capture_dashboard.py [--chrome PATH]   # -> reports/figures/dashboard_*.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from tornado.websocket import websocket_connect

ROOT = Path(__file__).resolve().parents[1]
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT, DEBUG_PORT, WIDTH = _free_port(), _free_port(), 1400
# page -> number of Plotly charts it must have drawn before capture
PAGES = {"overview": 0, "review_queue": 0, "segment_explorer": 2, "model_comparison": 2,
         "data_health": 1}
URL_PATH = {"overview": ""}  # the default page is served at "/"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
]

# True once Streamlit has finished running the script and rendered the main block.
READY_JS = """(() => {
  const running = document.querySelector('[data-testid="stStatusWidget"]');
  const main = document.querySelector('[data-testid="stMainBlockContainer"]');
  const skeleton = document.querySelector('[data-testid="stSkeleton"]');
  return !!main && !running && !skeleton && main.innerText.length > 200;
})()"""
# Number of drawn Plotly charts, or -1 while any image is still loading.
DRAWN_JS = """(() => {
  const imgs = [...document.querySelectorAll('[data-testid="stMainBlockContainer"] img')];
  if (!imgs.every(i => i.complete)) return -1;
  return document.querySelectorAll('[data-testid="stPlotlyChart"] .main-svg').length;
})()"""
# Bottom edge of the rendered content (not the scroll container, which fills the viewport).
HEIGHT_JS = """(() => {
  const m = document.querySelector('[data-testid="stMainBlockContainer"]');
  return m ? Math.ceil(m.getBoundingClientRect().bottom + window.scrollY) : 900;
})()"""


class CDP:
    def __init__(self, ws):
        self.ws, self.ids = ws, itertools.count(1)

    async def call(self, method: str, **params):
        i = next(self.ids)
        await self.ws.write_message(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("id") == i:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg["result"]

    async def eval(self, js: str):
        return (await self.call("Runtime.evaluate", expression=js, returnByValue=True))[
            "result"].get("value")


async def capture(out_dir: Path) -> None:
    info = json.load(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list"))
    page = next(t for t in info if t["type"] == "page")
    cdp = CDP(await websocket_connect(page["webSocketDebuggerUrl"], max_message_size=1 << 28))
    await cdp.call("Page.enable")
    for name, n_charts in PAGES.items():
        await cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=900,
                       deviceScaleFactor=1, mobile=False)
        await cdp.call("Page.navigate",
                       url=f"http://localhost:{PORT}/{URL_PATH.get(name, name)}")
        await asyncio.sleep(2)
        t0, stable, last_h = time.time(), 0, -1
        # Grow the viewport to the page height (charts load lazily once visible) until the
        # script has finished, all charts are drawn and the height is stable for 3 polls.
        while stable < 3:
            if time.time() - t0 > 120:
                raise TimeoutError(f"{name}: page did not settle")
            await asyncio.sleep(1)
            h = int(await cdp.eval(HEIGHT_JS)) + 40
            await cdp.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=h,
                           deviceScaleFactor=1, mobile=False)
            done = await cdp.eval(READY_JS) and await cdp.eval(DRAWN_JS) >= n_charts
            stable = stable + 1 if done and h == last_h else 0
            last_h = h
        shot = await cdp.call("Page.captureScreenshot", format="png")
        path = out_dir / f"dashboard_{name}.png"
        path.write_bytes(base64.b64decode(shot["data"]))
        print(f"{path.relative_to(ROOT)}  {WIDTH}x{h}")
        for part in split_tall(path):
            print(f"  -> {part.relative_to(ROOT)}")


def split_tall(path: Path, max_ratio: float = 1.15) -> list[Path]:
    """Split a tall screenshot into parts at blank horizontal bands, for legible A4 printing.

    Cuts are placed at the blank row (uniform across the main content area) closest to each
    ideal cut position, so charts and tables are never sliced.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    n = max(1, round(h / (w * max_ratio) + 0.49))
    for old in path.parent.glob(f"{path.stem}_part*.png"):
        old.unlink()
    if n == 1:
        return []
    main = img.crop((320, 0, w - 20, h)).convert("L")  # skip the sidebar
    px = main.load()
    mw = main.size[0]
    blank = [all(abs(px[x, y] - px[0, y]) < 6 for x in range(0, mw, 4)) for y in range(h)]
    cuts = [0]
    for k in range(1, n):
        ideal = k * h // n
        y = min((y for y in range(cuts[-1] + 200, h - 200) if blank[y]),
                key=lambda y: abs(y - ideal), default=ideal)
        cuts.append(y)
    cuts.append(h)
    parts = []
    for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False), start=1):
        p = path.with_name(f"{path.stem}_part{i}.png")
        left = 0 if i == 1 else 330  # continuation parts: drop the (empty) sidebar
        img.crop((left, a, w, b)).save(p)
        parts.append(p)
    return parts


def find_chrome(given: str | None) -> str:
    for c in [given, *CHROME_CANDIDATES]:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    sys.exit("Chrome/Edge not found; pass --chrome PATH")


def wait_http(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(url)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--chrome")
    p.add_argument("--out", default=str(ROOT / "reports" / "figures"))
    args = p.parse_args()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
         "--server.headless", "true", "--server.port", str(PORT), "--theme.base", "light",
         "--client.toolbarMode", "minimal"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    profile = tempfile.mkdtemp()
    chrome = subprocess.Popen(
        [find_chrome(args.chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
         f"--remote-debugging-port={DEBUG_PORT}", f"--user-data-dir={profile}",
         f"--window-size={WIDTH},900", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(f"http://localhost:{PORT}/_stcore/health")
        wait_http(f"http://127.0.0.1:{DEBUG_PORT}/json/version")
        asyncio.run(capture(Path(args.out)))
    finally:
        for proc in (chrome, server):
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
