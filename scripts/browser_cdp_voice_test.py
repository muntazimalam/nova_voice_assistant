"""Drive headless Chrome against the real dashboard with a fake mic.

Spawns its own app server (run.py), launches Chrome headless with a fake
microphone fed a real speech WAV, clicks "Start Listening" in the real
app.js, then reports (a) the client terminal log and (b) the state chips.

Usage:
  python scripts/browser_cdp_voice_test.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
FAKE_MIC = os.path.join(tempfile.gettempdir(), "nova_fake_mic.wav")
DASHBOARD = "http://127.0.0.1:8000"
CDP_PORT = 9333


def wait_health(timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def get_page_ws_url():
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2) as r:
                targets = json.loads(r.read())
            for t in targets:
                if t.get("type") == "page":
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError("CDP target not available")


async def cdp(ws, msg_id, method, params=None):
    payload = {"id": msg_id, "method": method}
    if params:
        payload["params"] = params
    await ws.send(json.dumps(payload))
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if msg.get("id") == msg_id:
            return msg


async def main():
    import websockets

    # 1) Start the app server as our child so it always dies with us.
    server = subprocess.Popen(
        [PY, os.path.join(ROOT, "run.py")],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if not wait_health():
        print("server did not become healthy in 30s")
        server.kill()
        return 1
    print("[server] app running on 127.0.0.1:8000")

    user_data = tempfile.mkdtemp(prefix="nova-cdp-")
    args = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--mute-audio",
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={user_data}",
        "--use-fake-device-for-media-stream",
        "--use-fake-ui-for-media-stream",
        f"--use-file-for-fake-audio-capture={FAKE_MIC}",
        "--autoplay-policy=no-user-gesture-required",
        "about:blank",
    ]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        ws_url = get_page_ws_url()
        print(f"[cdp] attached to {ws_url}")
        async with websockets.connect(ws_url, max_size=None) as ws:
            console_logs = []

            async def _handle_events():
                # Drain events in the background: console + exceptions.
                async def _drain():
                    try:
                        while True:
                            msg = json.loads(await ws.recv())
                            m = msg.get("method")
                            if m == "Runtime.consoleAPICalled":
                                args = [a.get("value", "?") for a in msg.get("params", {}).get("args", [])]
                                console_logs.append(("console", " ".join(str(a) for a in args)))
                            elif m == "Runtime.exceptionThrown":
                                d = msg.get("params", {}).get("exceptionDetails", {})
                                console_logs.append(("exception", json.dumps(d, default=str)[:400]))
                            elif m == "Log.entryAdded":
                                e = msg.get("params", {}).get("entry", {})
                                console_logs.append(("log", f"{e.get('level')}: {e.get('text')}"))
                    except Exception:
                        pass

                drain_task = asyncio.create_task(_drain())

                await cdp(ws, 1, "Runtime.enable")
                await cdp(ws, 2, "Page.enable")
                await cdp(ws, 3, "Log.enable")
                return drain_task

            drain_task = await _handle_events()
            await asyncio.sleep(1.0)

            await cdp(ws, 10, "Page.navigate", {"url": DASHBOARD})
            await asyncio.sleep(2.5)

            async def evaluate(expression):
                result = await cdp(ws, 20, "Runtime.evaluate", {
                    "expression": expression, "returnByValue": True, "awaitPromise": True,
                })
                r = result.get("result", {})
                if r.get("exceptionDetails"):
                    print("[cdp] EXCEPTION:", json.dumps(r["exceptionDetails"], default=str)[:260])
                return r.get("result", {}).get("value")

            url = await evaluate("location.href")
            title = await evaluate("document.title")
            print(f"[cdp] location.href = {url!r}  title = {title!r}")
            btn = await evaluate("!!document.getElementById('btnStartListening')")
            print(f"[cdp] btnStartListening present = {btn!r}")
            if not btn:
                print("[cdp] page did not load the dashboard; aborting")
                return 2

            print("[cdp] clicking btnStartListening...")
            clicked = await evaluate("""(() => {
                const b = document.getElementById('btnStartListening');
                b.click();
                return 'CLICKED';
            })()""")
            print(f"[cdp] click result = {clicked!r}")
            await asyncio.sleep(3.0)

            terminal = await evaluate("document.getElementById('terminalOutput').innerText")
            status = await evaluate("document.getElementById('assistantStateChip').textContent")
            print("\n===== CLIENT TERMINAL AFTER CLICK (3s) =====")
            print(terminal or "(empty)")
            print(f"state chip: {status!r}\n")

            print("[cdp] waiting 15s for capture->STT->reply...")
            await asyncio.sleep(15)
            terminal2 = await evaluate("document.getElementById('terminalOutput').innerText")
            status2 = await evaluate("document.getElementById('assistantStateChip').textContent")
            print("===== CLIENT TERMINAL AFTER 15s =====")
            print(terminal2 or "(empty)")
            print(f"state chip: {status2!r}")

            print("\n===== BROWSER CONSOLE =====")
            for kind, text in console_logs:
                print(f"[{kind}] {text}")
            if not console_logs:
                print("(no console output captured)")
            print("===== END CONSOLE =====\n")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(asyncio.run(main()))