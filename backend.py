import os, json, base64, uuid
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote, urlencode
import urllib.request as _urlreq
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

BASE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "")) or BASE
DATA_DIR.mkdir(parents=True, exist_ok=True)
MEM_FILE = DATA_DIR / "memory.json"
INDEX_FILE = BASE / "index.html"
ENV_FILE = BASE / ".env"

# auto-load .env so double-click START.bat works without extra setup
if ENV_FILE.exists():
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# --- Config: FREE options first ---
# Set in env: AI_API_KEY, AI_BASE_URL, AI_MODEL
#  1) Gemini FREE (recommended, free key, no card, vision included):
#   AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
#   AI_MODEL=gemini-2.0-flash
#   AI_API_KEY=AIza... from https://aistudio.google.com/apikey
#  2) Groq FREE (fast): AI_BASE_URL=https://api.groq.com/openai/v1
#     AI_MODEL=llama-3.3-70b-versatile, key from https://console.groq.com
#  3) Ollama LOCAL (100% free, no key): AI_BASE_URL=http://localhost:11434/v1
#     AI_MODEL=llama3.1, AI_API_KEY=ollama
#  4) Meta Muse Spark (paid): AI_BASE_URL=https://api.meta.ai/v1, AI_MODEL=muse-spark-1.3
API_KEY = os.getenv("AI_API_KEY", "") or os.getenv("MODEL_API_KEY", "") or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
BASE_URL = os.getenv("AI_BASE_URL", "") or None
MODEL = os.getenv("AI_MODEL", "gemini-3-flash-preview")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

def need_auth(req: Request):
    if not APP_PASSWORD:
        return True
    pw = req.headers.get("x-app-password", "") or req.query_params.get("pw", "")
    return pw == APP_PASSWORD

def load_mem():
    if MEM_FILE.exists():
        try:
            return json.loads(MEM_FILE.read_text(encoding="utf-8"))
        except: pass
    return {"profile": {"name": "", "goals": [], "notes": ""}, "facts": [], "conversations": []}

CONTAINER_ID = uuid.uuid4().hex[:8]

def merge_mem(disk, incoming):
    """Union disk + incoming so a stale/parallel write can NEVER lose data.
    Same id/key: incoming (fresher) wins. Tombstoned ids stay deleted."""
    if not isinstance(disk, dict):
        return incoming
    tb = incoming.get("_tombstones", {}) or {}
    tconv = set(tb.get("conv", []))
    tfact = set(tuple(x) for x in tb.get("fact", []))
    tgoal = set(tb.get("goal", []))
    trem = set(tb.get("rem", []))
    tprof = set(tb.get("profile", []))

    def fkey(f):
        return (f.get("type", ""), f.get("text", ""), f.get("name", ""))

    convs, order = {}, []
    for c in disk.get("conversations", []) + incoming.get("conversations", []):
        cid = c.get("id") or (c.get("role", "") + "|" + str(c.get("content", ""))[:100])
        if cid in tconv:
            continue
        if cid not in convs:
            order.append(cid)
        convs[cid] = c
    facts = {}
    for f in disk.get("facts", []) + incoming.get("facts", []):
        k = fkey(f)
        if k not in tfact:
            facts[k] = f
    rems = {}
    for r in disk.get("reminders", []) + incoming.get("reminders", []):
        if r.get("id") not in trem:
            rems[r.get("id")] = r
    inss = {}
    for x in disk.get("insights", []) + incoming.get("insights", []):
        inss[x.get("id")] = x

    dp = disk.get("profile", {}) or {}
    ip = incoming.get("profile", {}) or {}
    prof = {}
    for k in ("name", "location", "notes"):
        if k in tprof:
            prof[k] = ip.get(k, "")
        else:
            prof[k] = ip.get(k) or dp.get(k, "")
    goals = [g for g in dict.fromkeys((dp.get("goals", []) or []) + (ip.get("goals", []) or []))
             if g not in tgoal]
    prof["goals"] = goals

    old_tb = disk.get("_tombstones", {}) or {}
    new_tb = {"conv": list(dict.fromkeys(
        (old_tb.get("conv", []) or []) + (tb.get("conv", []) or [])))[-300:],
        "fact": list(dict.fromkeys(
            [tuple(x) for x in (old_tb.get("fact", []) or [])] + list(tfact)))[-300:],
        "goal": list(dict.fromkeys(
            (old_tb.get("goal", []) or []) + list(tgoal)))[-300:],
        "rem": list(dict.fromkeys(
            (old_tb.get("rem", []) or []) + list(trem)))[-300:],
        "profile": list(dict.fromkeys(
            (old_tb.get("profile", []) or []) + list(tprof)))[-20:]}
    return {"profile": prof, "facts": list(facts.values()),
            "conversations": [convs[c] for c in order],
            "reminders": list(rems.values()), "insights": list(inss.values()),
            "_tombstones": new_tb}

def tomb(m, kind, val):
    tb = m.setdefault("_tombstones", {})
    lst = tb.setdefault(kind, [])
    if val not in lst:
        lst.append(val)
        tb[kind] = lst[-300:]

def save_mem(m, merge=True):
    if merge and MEM_FILE.exists():
        try:
            disk = json.loads(MEM_FILE.read_text(encoding="utf-8"))
            m = merge_mem(disk, m)
        except Exception:
            pass
    MEM_FILE.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    if os.getenv("DATA_DIR", ""):
        # cloud volume: commit NOW so no message is ever lost on container recycle
        try:
            import modal
            modal.Volume.from_name("agent010-data").commit()
        except Exception:
            pass

def ensure_ids(m):
    changed = False
    for c in m.get("conversations", []):
        if "id" not in c:
            c["id"] = uuid.uuid4().hex[:8]
            changed = True
    if changed:
        save_mem(m)
    return m

def clean_for_llm(convs):
    out = []
    for c in convs:
        if isinstance(c.get("content"), str):
            out.append({"role": c["role"], "content": c["content"]})
    return out

def get_client():
    # Ollama/local allows dummy key
    if not API_KEY and BASE_URL and "localhost" in BASE_URL:
        return OpenAI(api_key="ollama", base_url=BASE_URL)
    if not API_KEY:
        return None
    if BASE_URL:
        return OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return OpenAI(api_key=API_KEY)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatIn(BaseModel):
    message: str
    image_b64: str | None = None  # optional "data:image/jpeg;base64,..." from camera/upload

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_FILE.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

@app.get("/api/memory")
def get_memory(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return ensure_ids(load_mem())

@app.get("/api/diag")
def diag(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    try:
        mt = MEM_FILE.stat().st_mtime
    except Exception:
        mt = 0
    return {"container": CONTAINER_ID, "model": MODEL, "local_pc": LOCAL_PC,
            "messages": len(m.get("conversations", [])), "facts": len(m.get("facts", [])),
            "mem_mtime": mt, "mem_file": str(MEM_FILE)}

@app.get("/api/history")
def get_history(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = ensure_ids(load_mem())
    return {"conversations": m.get("conversations", []), "profile": m.get("profile", {}), "facts": m.get("facts", [])}

@app.post("/api/profile")
def save_profile(p: dict, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    m["profile"].update(p)
    save_mem(m)
    return {"ok": True, "profile": m["profile"]}

@app.post("/api/upload")
async def upload(req: Request, file: UploadFile = File(...)):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    data = await file.read()
    m = load_mem()
    name = file.filename.lower()
    if name.endswith(".pdf"):
        return JSONResponse({"error": "PDF reading is not supported yet. Please copy-paste the text into chat, or send a screenshot of the page as an image."}, status_code=400)
    if name.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return JSONResponse({"error": "Video understanding is not supported yet. Send a screenshot as an image instead and ask about it."}, status_code=400)
    if not name.endswith((".txt", ".md", ".csv", ".json", ".png", ".jpg", ".jpeg", ".webp")):
        return JSONResponse({"error": "That file type is not supported yet. I can read text files (.txt, .md, .csv, .json) and see images (.png, .jpg, .webp)."}, status_code=400)
    # everything kept: original saved to uploads/, content remembered in memory.json
    updir = DATA_DIR / "uploads"
    updir.mkdir(exist_ok=True)
    safename = datetime.now().strftime("%Y%m%d-%H%M%S-") + "".join(c for c in file.filename if c.isalnum() or c in "._-")[:80]
    (updir / safename).write_bytes(data)
    if name.endswith((".txt", ".md", ".csv", ".json")):
        text = data[:60000].decode("utf-8", errors="ignore")
    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        b64 = base64.b64encode(data).decode()
        m["facts"].append({"type": "image", "name": file.filename, "saved_as": safename, "at": datetime.now().isoformat(), "b64_len": len(b64)})
        save_mem(m)
        return {"ok": True, "note": "Image saved and remembered. Attach it with your next message (camera button) or ask about an image you sent."}
    m["facts"].append({"type": "file", "name": file.filename, "saved_as": safename, "at": datetime.now().isoformat(), "text": text[:60000]})
    save_mem(m)
    return {"ok": True, "extract": text[:2000]}

def user_now():
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(os.getenv("USER_TZ", "Africa/Casablanca")))
    except Exception:
        return datetime.now().astimezone()

def _http_get_json(url: str, timeout: int = 12):
    req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(_urlreq.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore"))

def tool_get_datetime(args: dict):
    now = user_now()
    return json.dumps({"local_time": now.strftime("%Y-%m-%d %H:%M (%A)"),
                       "timezone": str(now.tzinfo), "iso": now.isoformat()})

def tool_web_search(args: dict):
    q = (args.get("query") or "").strip()[:200]
    if not q:
        return json.dumps({"error": "empty query"})
    out = {"query": q, "results": []}
    try:
        ddg = _http_get_json("https://api.duckduckgo.com/?" + urlencode(
            {"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}))
        if ddg.get("AbstractText"):
            out["results"].append({"source": "DuckDuckGo", "text": ddg["AbstractText"][:1500],
                                   "url": ddg.get("AbstractURL", "")})
        if ddg.get("Answer"):
            out["results"].append({"source": "DuckDuckGo answer", "text": str(ddg["Answer"])[:500]})
    except Exception as e:
        out["ddg_error"] = str(e)[:200]
    try:
        op = _http_get_json("https://en.wikipedia.org/w/api.php?" + urlencode(
            {"action": "opensearch", "search": q, "limit": 3, "namespace": 0, "format": "json"}))
        for title in (op[1] if len(op) > 1 else [])[:2]:
            try:
                s = _http_get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                                   + quote(title.replace(" ", "_")))
                if s.get("extract"):
                    out["results"].append({"source": "Wikipedia: " + title,
                                           "text": s["extract"][:1500],
                                           "url": s.get("content_urls", {}).get("desktop", {}).get("page", "")})
            except Exception:
                pass
    except Exception as e:
        out["wiki_error"] = str(e)[:200]
    if not out["results"]:
        out["note"] = "No live results found — say so honestly instead of guessing."
    return json.dumps(out, ensure_ascii=False)[:6000]

def tool_browse(args: dict):
    url = (args.get("url") or "").strip()[:500]
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "give me a full link starting with http"})
    try:
        from playwright.sync_api import sync_playwright
        prof = BASE / ".browser-profile"
        prof.mkdir(exist_ok=True)
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(str(prof), headless=True)
            pg = ctx.new_page()
            pg.goto(url, timeout=25000, wait_until="domcontentloaded")
            title = pg.title()
            text = pg.inner_text("body")[:5000]
            ctx.close()
        if not text.strip():
            return json.dumps({"url": url, "title": title, "note": "page opened but no readable text found"})
        return json.dumps({"url": url, "title": title, "text": text}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": "could not open page: " + str(e)[:300]})

TOOL_HANDLERS = {"web_search": tool_web_search, "get_datetime": tool_get_datetime,
                  "browse_page": tool_browse, "save_insight": None}  # wired below

_pending_insights = []

def tool_save_insight(args: dict):
    t = (args.get("text") or "").strip()[:500]
    if t:
        _pending_insights.append(t)
    return json.dumps({"saved": True})

TOOL_HANDLERS["save_insight"] = tool_save_insight

LOCAL_PC = os.getenv("LOCAL_PC", "") == "1"
PC_PENDING = {}
PC_BLOCK = ("format ", "diskpart", "bcdedit", "reg delete", "cipher /w", "mkfs", "rm -rf /",
            "del /f /s /q c:\\windows", "rd /s /q c:\\windows")

def pc_guard(action: str, arg: str):
    if not LOCAL_PC:
        return "PC control only works on 010 running on the user's own laptop, not here."
    low = (action + " " + arg).lower()
    if any(b in low for b in PC_BLOCK):
        return "refused: destructive action, never allowed"
    return ""

def tool_pc_open(args: dict):
    g = pc_guard("open", args.get("target", ""))
    if g:
        return json.dumps({"error": g})
    target = (args.get("target") or "").strip()[:500]
    if not target:
        return json.dumps({"error": "nothing to open"})
    aid = uuid.uuid4().hex[:8]
    PC_PENDING[aid] = {"action": "open", "target": target, "at": datetime.now().isoformat()}
    return json.dumps({"pending_id": aid, "note": "Ask the user to tap Approve. It runs on their laptop."})

def tool_pc_run(args: dict):
    g = pc_guard("run", args.get("command", ""))
    if g:
        return json.dumps({"error": g})
    cmd = (args.get("command") or "").strip()[:2000]
    if not cmd:
        return json.dumps({"error": "empty command"})
    aid = uuid.uuid4().hex[:8]
    PC_PENDING[aid] = {"action": "run", "command": cmd, "at": datetime.now().isoformat()}
    return json.dumps({"pending_id": aid, "command": cmd,
                       "note": "Shell commands need the user's approval. Ask them to tap Approve."})

def tool_pc_file(args: dict):
    g = pc_guard("file", args.get("path", ""))
    if g:
        return json.dumps({"error": g})
    op, path = (args.get("op") or "list").lower(), (args.get("path") or "").strip()[:500]
    if not path:
        return json.dumps({"error": "no path given"})
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    low = str(p).lower()
    if low.startswith(("c:\\windows", "c:\\program files", "c:\\program files (x86)")):
        return json.dumps({"error": "system folders are off-limits"})
    try:
        if op == "list":
            if not p.is_dir():
                return json.dumps({"error": "not a folder"})
            return json.dumps({"path": str(p), "items": sorted(os.listdir(p))[:200]})
        if op == "read":
            data = p.read_bytes()
            return json.dumps({"path": str(p), "text": data[:20000].decode("utf-8", errors="ignore")})
        if op == "write":
            if str(p).lower().startswith(str((BASE / "workspace").lower())):
                (BASE / "workspace").mkdir(exist_ok=True)
                p.write_text(args.get("content", ""), encoding="utf-8")
                return json.dumps({"ok": True, "path": str(p)})
            aid = uuid.uuid4().hex[:8]
            PC_PENDING[aid] = {"action": "write", "path": str(p),
                               "content": args.get("content", "")[:20000],
                               "at": datetime.now().isoformat()}
            return json.dumps({"pending_id": aid, "note": "Writing outside 010's folder needs approval. Ask them to tap Approve."})
        return json.dumps({"error": "op must be list, read or write"})
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})

def tool_pc_screenshot(args: dict):
    g = pc_guard("screenshot", "")
    if g:
        return json.dumps({"error": g})
    try:
        from PIL import ImageGrab
        shot = ImageGrab.grab()
        updir = DATA_DIR / "uploads"
        updir.mkdir(exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S-shot.png")
        shot.save(updir / name)
        m = load_mem()
        m["facts"].append({"type": "image", "name": name, "saved_as": name,
                           "at": datetime.now().isoformat(), "note": "PC screenshot"})
        save_mem(m)
        return json.dumps({"ok": True, "saved_as": name,
                           "note": "Screenshot saved. Tell the user they can view it; describe you can't see it with this brain."})
    except Exception as e:
        return json.dumps({"error": "screenshot failed (needs Pillow: pip install pillow): " + str(e)[:200]})

def chrome_ctx():
    """Connect to the USER's own Chrome (must run with --remote-debugging-port=9222)."""
    from playwright.sync_api import sync_playwright
    p = sync_playwright().start()
    try:
        browser = p.chromium.connect_over_cdp("http://localhost:9222", timeout=8000)
    except Exception:
        p.stop()
        raise RuntimeError("your Chrome is not in debug mode — run ChromeDebug.bat first (close Chrome, then launch it)")
    return p, browser

def chrome_pages(browser):
    out = []
    for ctx in browser.contexts:
        out.extend([pg for pg in ctx.pages if not pg.url.startswith("chrome")])
    return out

def chrome_new_page(browser):
    return browser.contexts[0].new_page() if browser.contexts else None

def tool_chrome_tabs(args: dict):
    g = pc_guard("chrome", "")
    if g:
        return json.dumps({"error": g})
    try:
        p, browser = chrome_ctx()
        try:
            out = [{"i": i, "title": pg.title()[:120], "url": pg.url[:300]}
                   for i, pg in enumerate(chrome_pages(browser))]
            return json.dumps({"tabs": out, "note": "this is the USER's own logged-in Chrome"})
        finally:
            browser.close()
            p.stop()
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})

def tool_chrome_go(args: dict):
    g = pc_guard("chrome", "")
    if g:
        return json.dumps({"error": g})
    url = (args.get("url") or "").strip()[:500]
    if not url.startswith(("http://", "https://")):
        return json.dumps({"error": "give me a full link starting with http"})
    try:
        p, browser = chrome_ctx()
        try:
            pages = chrome_pages(browser)
            pg = pages[0] if pages else chrome_new_page(browser)
            if pg is None:
                return json.dumps({"error": "no Chrome window open"})
            pg.goto(url, timeout=25000, wait_until="domcontentloaded")
            return json.dumps({"ok": True, "title": pg.title()[:200], "url": pg.url[:300]})
        finally:
            browser.close()
            p.stop()
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})

def tool_chrome_read(args: dict):
    g = pc_guard("chrome", "")
    if g:
        return json.dumps({"error": g})
    try:
        p, browser = chrome_ctx()
        try:
            pages = chrome_pages(browser)
            if not pages:
                return json.dumps({"error": "no open tabs"})
            i = int(args.get("tab", 0))
            pg = pages[i] if 0 <= i < len(pages) else pages[0]
            return json.dumps({"title": pg.title()[:200], "url": pg.url[:300],
                               "text": pg.inner_text("body")[:6000]}, ensure_ascii=False)
        finally:
            browser.close()
            p.stop()
    except Exception as e:
        return json.dumps({"error": str(e)[:300]})

PC_TOOLS = [    {"type": "function", "function": {
        "name": "pc_open",
        "description": "Open something on the user's laptop: an app name (notepad, calculator, chrome), a URL, or a file/folder path. Runs after their approval.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "description": "app name, URL or path"}}, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "pc_run",
        "description": "Run a PowerShell command on the user's laptop. ALWAYS needs their approval first — tell them to tap Approve, then report the result.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "PowerShell command"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "pc_file",
        "description": "Files on the laptop. op=list/read are instant (no system folders). op=write outside 010's workspace folder needs approval.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "description": "list, read or write"},
            "path": {"type": "string", "description": "folder or file path"},
            "content": {"type": "string", "description": "text for op=write"}}, "required": ["op", "path"]}}},
    {"type": "function", "function": {
        "name": "pc_screenshot",
        "description": "Take a screenshot of the laptop screen and save it. You cannot see it with text-only brains — tell the user it's saved for them.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "chrome_tabs",
        "description": "List the USER's own open Chrome tabs (their real logged-in browser). Use to see what they have open or to find a tab.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "chrome_go",
        "description": "Open a URL in the USER's own Chrome (first tab). Their logins apply — good for YouTube, Facebook, Gmail, Canva.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "full http(s) URL"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "chrome_read",
        "description": "Read the text of a tab in the USER's own Chrome (e.g. Facebook messages). Tab 0 = first tab.",
        "parameters": {"type": "object", "properties": {
            "tab": {"type": "integer", "description": "tab number, default 0"}}}}},
]

if LOCAL_PC:
    TOOL_HANDLERS.update({"pc_open": tool_pc_open, "pc_run": tool_pc_run,
                           "pc_file": tool_pc_file, "pc_screenshot": tool_pc_screenshot,
                           "chrome_tabs": tool_chrome_tabs, "chrome_go": tool_chrome_go,
                           "chrome_read": tool_chrome_read})
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the live web (DuckDuckGo + Wikipedia) for current or external facts. Use it whenever the user asks about anything that changes over time or that you might not know: news, prices, scores, people, places, definitions.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_datetime",
        "description": "Get the current local date and time on the user's PC. Use it whenever the user asks about time, date, day, or scheduling.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "browse_page",
        "description": "Open any web link in a real Chrome browser and read its content. Use it when the user sends a link or asks you to check/open a specific page.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string", "description": "full http(s) URL to open"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "save_insight",
        "description": "Save something interesting you found while researching so you can tell the user about it later unprompted. Use sparingly — only genuinely useful finds tied to their goals.",
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "the interesting find, one or two sentences"}}, "required": ["text"]}}},
]

if LOCAL_PC:
    TOOLS.extend(PC_TOOLS)

import re as _re

def parse_reminder(text: str):
    """Understand 'remind me ... at 8pm / tomorrow / in 30 min / on friday / every day at 7am'.
    Returns (task, due_datetime, repeat) or (None, None, None)."""
    low = text.lower()
    i = low.find("remind me")
    if i < 0:
        return None, None, None
    task = text[i + len("remind me"):].strip()
    task = _re.sub(r"^(to|that|about)\s+", "", task, flags=_re.I).strip(" .")
    now = user_now()
    repeat = "daily" if _re.search(r"\bevery\s+day\b", low) else None
    due = None
    m = _re.search(r"in\s+(\d+)\s*(minute|min|hour|hr)s?", low)
    if m:
        n = int(m.group(1))
        due = now + timedelta(minutes=n if "min" in m.group(2) else n * 60)
        task = _re.sub(r"\s*in\s+\d+\s*(minute|min|hour|hr)s?", "", task, flags=_re.I).strip(" .")
    if due is None:
        m = _re.search(r"(tomorrow)(?:\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?)?", low)
        if m:
            h, mi, ap = int(m.group(2) or 9), int(m.group(3) or 0), (m.group(4) or "")
            if ap == "pm" and h < 12:
                h += 12
            due = (now + timedelta(days=1)).replace(hour=h % 24, minute=mi, second=0, microsecond=0)
            task = _re.sub(r"\s*tomorrow(\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?)?", "", task, flags=_re.I).strip(" .")
    if due is None:
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for di, d in enumerate(days):
            if _re.search(r"\b" + d + r"\b", low):
                delta = (di - now.weekday()) % 7 or 7
                due = (now + timedelta(days=delta)).replace(hour=9, minute=0, second=0, microsecond=0)
                mt = _re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
                if mt:
                    h, mi, ap = int(mt.group(1)), int(mt.group(2) or 0), (mt.group(3) or "")
                    if ap == "pm" and h < 12:
                        h += 12
                    due = due.replace(hour=h % 24, minute=mi)
                task = _re.sub(r"\s*(on\s+)?" + d + r"(\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?)?", "", task, flags=_re.I).strip(" .")
                break
    if due is None:
        m = _re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", low)
        if m:
            h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "")
            if ap == "pm" and h < 12:
                h += 12
            if ap == "am" and h == 12:
                h = 0
            due = now.replace(hour=h % 24, minute=mi, second=0, microsecond=0)
            if due <= now:
                due = due + timedelta(days=1)
            task = _re.sub(r"\s*at\s+\d{1,2}(:\d{2})?\s*(am|pm)?", "", task, flags=_re.I).strip(" .")
    if not task or due is None:
        return None, None, None
    task = _re.sub(r"\bevery\s+day\b", "", task, flags=_re.I).strip(" .")
    return task[:200] if task else None, due, repeat

def due_reminders(m):
    now = user_now()
    out = []
    for r in m.get("reminders", []):
        if not r.get("done"):
            try:
                if datetime.fromisoformat(r["due"]) <= now:
                    out.append(r)
            except Exception:
                pass
    return out

def upcoming_reminders(m):
    now = user_now()
    out = []
    for r in m.get("reminders", []):
        if not r.get("done"):
            try:
                dd = datetime.fromisoformat(r["due"])
                if now < dd <= now + timedelta(hours=24):
                    out.append(r)
            except Exception:
                pass
    return out

def unseen_insights(m):
    return [x for x in m.get("insights", []) if not x.get("seen")]

def complete_with_tools(client, model, msgs):
    """Run the chat with tool use (up to 3 rounds). Falls back to plain chat if tools unsupported."""
    try:
        resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=800,
                                              tools=TOOLS, tool_choice="auto")
        for _ in range(3):
            msg = resp.choices[0].message
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                return msg.content
            msgs.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                          "function": {"name": tc.function.name,
                                                       "arguments": tc.function.arguments}} for tc in calls]})
            for tc in calls:
                try:
                    res = TOOL_HANDLERS[tc.function.name](json.loads(tc.function.arguments or "{}"))
                except Exception as e:
                    res = json.dumps({"error": str(e)[:300]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": res})
            resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=800,
                                                  tools=TOOLS, tool_choice="auto")
        return resp.choices[0].message.content
    except Exception as e:
        if "tool" in str(e).lower():
            return client.chat.completions.create(model=model, messages=msgs,
                                                  max_tokens=800).choices[0].message.content
        raise

@app.post("/api/chat")
def chat(inp: ChatIn, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    client = get_client()
    if not client:
        return JSONResponse({"error": "No FREE key set. Get one at https://aistudio.google.com/apikey, paste into life-manager/.env as AI_API_KEY, restart."}, status_code=400)

    profile = m.get("profile", {})
    facts = m.get("facts", [])[-100:]  # last 100 facts — everything is kept in memory.json
    mem_text = json.dumps({"profile": profile, "facts": facts}, ensure_ascii=False)[:24000]

    now = user_now()
    due = due_reminders(m)
    fresh = unseen_insights(m)
    nudge_text = ""
    if due:
        nudge_text += "\nDUE NOW (lead with these like a present friend checking in): " + "; ".join(
            f"{r['text']} (was due {r['due'][:16].replace('T', ' ')})" for r in due[:5])
    if fresh:
        nudge_text += "\nTHINGS YOU SAVED TO TELL THEM (share naturally when relevant): " + "; ".join(
            x["text"] for x in fresh[-3:])
    system = f"""You are Personal Guide, a sharp friend who manages the user's life over text. Reply in ONE short message (1-3 sentences), contractions, no essays and no bullet lists unless asked. Only use /// to split into two texts on rare occasions when there are genuinely two separate thoughts.
Your name is Personal Guide. When asked who you are, say you are Personal Guide, their personal guide.
Current local time: {now.strftime("%Y-%m-%d %H:%M (%A)")}.
Help user fix their life and achieve goals. Be present: if something is due, check on them directly ("gym time — you locked in?").
Below you get the FULL conversation history plus MEMORY. Read the user's new message,
then read ALL past messages for context, then answer using everything you know.
You have live tools: web_search (current facts), browse_page (open links in Chrome), get_datetime (exact time) and save_insight (stash an interesting find to tell them later). Use them instead of guessing.
{"PC: you can act on the user's laptop with pc_open, pc_file and pc_screenshot (instant), pc_run and outside writes (need their Approve tap — always tell them to tap it). When they ask you to open/run/do something, CALL the tool immediately instead of asking for confirmation — approval happens via their Approve button. Paths: use %USERPROFILE% for home (e.g. %USERPROFILE%/Documents). chrome_tabs/chrome_go/chrome_read work inside THEIR real logged-in Chrome (needs ChromeDebug.bat running) — use them for YouTube, Facebook messages, Gmail." if LOCAL_PC else "IMPORTANT: you are the CLOUD copy — you cannot touch the laptop (no PC tools, no logged-in Chrome). If they ask to open/control anything on the laptop, tell them to use the laptop version (localhost:8000 with START.bat running)."}
MEMORY: {mem_text}{nudge_text}
Rules:
1. If user shares a durable fact (name, goal, habit, preference), acknowledge it briefly and it will be auto-saved.
2. One concrete next action max, never lectures.
3. One short question max when the goal is vague.
4. ONE message by default. Detail only when asked."""

    # auto-learn simple facts: "my name is X", "my goal is Y"
    msg_low = inp.message.lower()
    learned = None
    nm = extract_name(inp.message)
    if nm:
        profile["name"] = nm
        learned = f"Saved name: {nm}"
    loc = extract_location(inp.message)
    if loc:
        profile["location"] = loc
        learned = (learned + " " if learned else "") + f"Saved location: {loc}."
    for cert in extract_certs(inp.message):
        ctext = f"Has {cert}"
        if not any(f.get("text") == ctext for f in m.get("facts", [])):
            m.setdefault("facts", []).append({"type": "auto", "at": datetime.now().isoformat(), "text": ctext})
            learned = (learned + " " if learned else "") + f"Noted: {cert}."
    if "my goal is " in msg_low or "my goals are " in msg_low:
        g = inp.message.strip()[:300]
        if g not in profile.get("goals", []):
            profile.setdefault("goals", []).append(g)
            learned = "Saved to goals."

    # "remind me ..." → deterministic reminder, no AI needed
    if "remind me" in msg_low:
        task, due, repeat = parse_reminder(inp.message)
        if task and due:
            m.setdefault("reminders", []).append({"id": uuid.uuid4().hex[:8], "text": task,
                                                  "due": due.isoformat(), "repeat": repeat,
                                                  "done": False, "at": datetime.now().isoformat()})
            learned = (learned + " " if learned else "") + \
                f"⏰ Got it — I'll check on you: {task} ({due.strftime('%a %H:%M')})."
        elif not learned:
            learned = "When should I check on you? (e.g. at 8pm, tomorrow 9am, in 30 min, friday)"

    NO_VISION = ("llama-3.3-70b-versatile", "llama3.1", "openai/gpt-oss-120b", "openai/gpt-oss-20b")
    if inp.image_b64 and MODEL in NO_VISION:
        return JSONResponse({"error": "This brain (Llama) cannot see images. For photo questions, switch to Gemini or GPT in ⚙️ Brain, then ask again."}, status_code=400)

    # FULL history as context: read everything before answering.
    # Safety budget only matters at huge scale (400k chars ≈ 100k tokens, inside 1M context).
    BUDGET = 400000
    full = clean_for_llm(m.get("conversations", []))
    total = 0
    keep_from = 0
    for i in range(len(full) - 1, -1, -1):
        total += len(full[i].get("content", ""))
        if total > BUDGET:
            keep_from = i + 1
            break
    msgs = [{"role": "system", "content": system}]
    msgs.extend(full[keep_from:])
    if inp.image_b64:
        msgs.append({"role": "user", "content": [
            {"type": "text", "text": inp.message or "What do you see?"},
            {"type": "image_url", "image_url": {"url": inp.image_b64}}
        ]})
    else:
        msgs.append({"role": "user", "content": inp.message})

    try:
        reply = complete_with_tools(client, MODEL, msgs)
    except Exception as e:
        return JSONResponse({"error": f"AI call failed: {e}. Check AI_MODEL/BASE_URL/KEY."}, status_code=500)

    m.setdefault("conversations", []).append({"id": uuid.uuid4().hex[:8], "role": "user",
                                              "content": inp.message[:2000], "at": datetime.now().isoformat()})
    m["conversations"].append({"id": uuid.uuid4().hex[:8], "role": "assistant",
                               "content": (reply or "")[:4000], "at": datetime.now().isoformat()})
    # no truncation — everything is remembered in memory.json
    m["profile"] = profile
    # also persist learned facts as list
    if learned and learned not in [f.get("text","") for f in m["facts"][-5:]]:
        m["facts"].append({"type": "auto", "at": datetime.now().isoformat(), "text": inp.message[:500]})
    # persist insights the brain stashed via save_insight
    global _pending_insights
    for t in _pending_insights:
        m.setdefault("insights", []).append({"id": uuid.uuid4().hex[:8], "text": t,
                                              "at": datetime.now().isoformat(), "seen": False})
    _pending_insights = []
    save_mem(m)
    return {"reply": reply, "learned": learned}

class EditIn(BaseModel):
    content: str

def extract_name(text: str):
    low = text.lower()
    if "my name is " in low:
        idx = low.find("my name is ") + len("my name is ")
        return " ".join(text[idx:].strip().split()[0:2]).strip(",. ")
    return ""

def extract_location(text: str):
    m = _re.search(r"(?:i live in|i'm from|i am from|my city is|i'm in|based in)\s+([A-Za-z][\w\- ]*?)(?=\s+(?:and|with|but|because|since|for|as|which|that)\b|[,.]|$)",
                   text, flags=_re.I)
    if m:
        return m.group(1).strip(" .,").title()
    return ""

def extract_certs(text: str):
    found = []
    for m in _re.finditer(r"(?:have|got|hold|earned?)\s+(?:an?|my)?\s*([\w\- ]{2,40}?certificat\w*)",
                          text, flags=_re.I):
        found.append(m.group(1).strip().title())
    return found

def cascade_delete(m, content: str):
    """Wipe every trace of a deleted message: auto-facts, goals, learned name."""
    profile = m.get("profile", {})
    gone_facts = [f for f in m.get("facts", [])
                  if (f.get("type") == "auto" and f.get("text") == content[:500])]
    m["facts"] = [f for f in m.get("facts", []) if f not in gone_facts]
    for f in gone_facts:
        tomb(m, "fact", [f.get("type", ""), f.get("text", ""), f.get("name", "")])
    g = content.strip()[:300]
    if g in profile.get("goals", []):
        tomb(m, "goal", g)
    profile["goals"] = [x for x in profile.get("goals", []) if x != g]
    nm = extract_name(content)
    if nm and profile.get("name") == nm:
        profile["name"] = ""
        tomb(m, "profile", "name")
    loc = extract_location(content)
    if loc and profile.get("location") == loc:
        profile["location"] = ""
        tomb(m, "profile", "location")
    for cert in extract_certs(content):
        cut = [f for f in m.get("facts", []) if f.get("text") == f"Has {cert}"]
        m["facts"] = [f for f in m.get("facts", []) if f not in cut]
        for f in cut:
            tomb(m, "fact", [f.get("type", ""), f.get("text", ""), f.get("name", "")])

def cascade_edit(m, old: str, new: str):
    """Move every trace of an edited message to the new text."""
    profile = m.get("profile", {})
    for f in m.get("facts", []):
        if f.get("type") == "auto" and f.get("text") == old[:500]:
            tomb(m, "fact", [f.get("type", ""), f.get("text", ""), f.get("name", "")])
            f["text"] = new[:500]
    g_old, g_new = old.strip()[:300], new.strip()[:300]
    if g_old in profile.get("goals", []) and g_old != g_new:
        tomb(m, "goal", g_old)
    profile["goals"] = [g_new if x == g_old else x for x in profile.get("goals", [])]
    nm = extract_name(new)
    if nm:
        profile["name"] = nm
    nl = extract_location(new)
    if nl:
        profile["location"] = nl

@app.put("/api/message/{mid}")
def edit_message(mid: str, inp: EditIn, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    for c in m.get("conversations", []):
        if c.get("id") == mid:
            old = c.get("content", "")
            c["content"] = inp.content[:4000]
            cascade_edit(m, old, inp.content)
            save_mem(m)
            return {"ok": True}
    return JSONResponse({"error": "message not found"}, status_code=404)

@app.delete("/api/message/{mid}")
def del_message(mid: str, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    gone = next((c for c in m.get("conversations", []) if c.get("id") == mid), None)
    if not gone:
        return JSONResponse({"error": "message not found"}, status_code=404)
    m["conversations"] = [c for c in m.get("conversations", []) if c.get("id") != mid]
    tomb(m, "conv", mid)
    cascade_delete(m, gone.get("content", ""))
    save_mem(m)
    return {"ok": True, "wiped": True}

@app.get("/api/export")
def export_mem(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = ensure_ids(load_mem())
    return FileResponse(MEM_FILE, filename="personal-guide-memory.json", media_type="application/json")

@app.post("/api/import")
async def import_mem(req: Request, file: UploadFile = File(...)):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        d = json.loads((await file.read()).decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "not a valid personal-guide-memory.json file"}, status_code=400)
    if not isinstance(d, dict) or "conversations" not in d:
        return JSONResponse({"error": "not a valid personal-guide-memory.json file"}, status_code=400)
    m = {"profile": d.get("profile", {"name": "", "goals": [], "notes": ""}),
         "facts": d.get("facts", []), "conversations": d.get("conversations", [])}
    save_mem(m)
    ensure_ids(m)
    return {"ok": True, "messages": len(m["conversations"]), "facts": len(m["facts"])}

@app.get("/api/nudges")
def get_nudges(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    return {"due": due_reminders(m), "upcoming": upcoming_reminders(m),
            "insights": unseen_insights(m), "now": user_now().isoformat()}

class NudgeIn(BaseModel):
    id: str
    minutes: int = 30

@app.post("/api/nudge/done")
def nudge_done(inp: NudgeIn, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    for r in m.get("reminders", []):
        if r.get("id") == inp.id:
            if r.get("repeat") == "daily":
                r["due"] = (datetime.fromisoformat(r["due"]) + timedelta(days=1)).isoformat()
            else:
                r["done"] = True
            save_mem(m)
            return {"ok": True}
    for x in m.get("insights", []):
        if x.get("id") == inp.id:
            x["seen"] = True
            save_mem(m)
            return {"ok": True}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.post("/api/nudge/snooze")
def nudge_snooze(inp: NudgeIn, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    for r in m.get("reminders", []):
        if r.get("id") == inp.id:
            r["due"] = (user_now() + timedelta(minutes=inp.minutes)).isoformat()
            save_mem(m)
            return {"ok": True, "due": r["due"]}
    return JSONResponse({"error": "not found"}, status_code=404)

@app.delete("/api/nudge/{nid}")
def nudge_del(nid: str, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    m["reminders"] = [r for r in m.get("reminders", []) if r.get("id") != nid]
    tomb(m, "rem", nid)
    save_mem(m)
    return {"ok": True}

@app.get("/api/pc/pending")
def pc_pending(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not LOCAL_PC:
        return {"pending": [], "local": False}
    now = datetime.now()
    dead = [k for k, v in PC_PENDING.items()
            if (now - datetime.fromisoformat(v["at"])).total_seconds() > 600]
    for k in dead:
        PC_PENDING.pop(k, None)
    items = [{"id": k, **{kk: vv for kk, vv in v.items() if kk != "content"},
              "has_content": "content" in v} for k, v in PC_PENDING.items()]
    return {"pending": items, "local": True}

class PcDecide(BaseModel):
    id: str
    approve: bool

def pc_execute(p: dict):
    import subprocess
    a = p["action"]
    if a == "open":
        t = p["target"]
        if t.startswith(("http://", "https://")) or os.path.exists(t):
            os.startfile(t)
        else:
            subprocess.Popen(["powershell", "-NoProfile", "-Command", "Start-Process", t])
        return "opened " + t
    if a == "run":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", p["command"]],
                           capture_output=True, text=True, timeout=60)
        out = (r.stdout or "") + (("\nSTDERR:\n" + r.stderr) if r.stderr else "")
        return f"exit {r.returncode}\n" + out[:4000]
    if a == "write":
        Path(p["path"]).write_text(p.get("content", ""), encoding="utf-8")
        return "written to " + p["path"]
    return "unknown action"

@app.post("/api/pc/decide")
def pc_decide(inp: PcDecide, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    if not LOCAL_PC:
        return JSONResponse({"error": "PC control only runs on the laptop"}, status_code=400)
    p = PC_PENDING.pop(inp.id, None)
    if not p:
        return JSONResponse({"error": "action expired or not found"}, status_code=404)
    if not inp.approve:
        return {"ok": True, "denied": True}
    try:
        return {"ok": True, "result": pc_execute(p), "summary": p["action"] + ": " + p.get("target", p.get("command", p.get("path", "")))[:120]}
    except Exception as e:
        return JSONResponse({"error": "failed: " + str(e)[:300]}, status_code=500)

@app.post("/api/clear")
def clear_chat(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    try:
        (DATA_DIR / f"trash-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json").write_text(
            json.dumps(m, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    m["conversations"] = []
    save_mem(m, merge=False)
    return {"ok": True}

@app.delete("/api/fact/{idx}")
def del_fact(idx: int, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    try:
        m["facts"].pop(idx)
        save_mem(m)
        return {"ok": True}
    except IndexError:
        return JSONResponse({"error": "fact not found"}, status_code=404)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_CATALOG = [
    {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash — free", "base_url": GEMINI_URL,
     "env_key": "GEMINI_API_KEY", "key_from": "https://aistudio.google.com/apikey",
     "features": ["Free key, no credit card", "Sees images", "Fast", "Huge memory window"]},
    {"id": "gemini-flash-latest", "label": "Gemini Flash Latest — free", "base_url": GEMINI_URL,
     "env_key": "GEMINI_API_KEY", "key_from": "https://aistudio.google.com/apikey",
     "features": ["Free key, no credit card", "Sees images", "Can be busy at peak times"]},
    {"id": "openai/gpt-oss-120b", "label": "GPT-OSS 120B via Groq — free", "base_url": "https://api.groq.com/openai/v1",
     "env_key": "GROQ_API_KEY", "key_from": "https://console.groq.com",
     "features": ["Free key, very fast", "120B reasoning model", "131k memory window", "No image vision"]},
    {"id": "qwen/qwen3.8-27b", "label": "Qwen 3.8 27B via Groq — free", "base_url": "https://api.groq.com/openai/v1",
     "env_key": "GROQ_API_KEY", "key_from": "https://console.groq.com",
     "features": ["Free key, very fast", "Sees images", "131k memory window", "Smaller than 120B"]},
    {"id": "gpt-4o-mini", "label": "GPT-4o mini — cheap paid", "base_url": "",
     "env_key": "OPENAI_API_KEY", "key_from": "https://platform.openai.com",
     "features": ["A few dollars lasts months", "Sees images", "Very reliable"]},
    {"id": "gpt-4o", "label": "GPT-4o — smart paid", "base_url": "",
     "env_key": "OPENAI_API_KEY", "key_from": "https://platform.openai.com",
     "features": ["Smartest chat + vision", "Reads images + PDFs via chat", "Costs more per message"]},
    {"id": "muse-spark-1.3", "label": "Muse Spark 1.3 — genius paid", "base_url": "https://api.meta.ai/v1",
     "env_key": "MODEL_API_KEY", "key_from": "https://ai.developer.meta.com",
     "features": ["Deepest reasoning", "Text + image + PDF + video", "1M context window"]},
    {"id": "grok-4.3", "label": "Grok 4.3 (Elon Musk xAI) — paid", "base_url": "https://api.x.ai/v1",
     "env_key": "XAI_API_KEY", "key_from": "https://console.x.ai",
     "features": ["Witty, unfiltered personality", "Sees images (jpg/png)", "1M context window", "~$1.25 per 1M words in"]},
    {"id": "grok-4.7", "label": "Grok 4.7 flagship (Elon Musk xAI) — paid", "base_url": "https://api.x.ai/v1",
     "env_key": "XAI_API_KEY", "key_from": "https://console.x.ai",
     "features": ["Smartest Grok ever", "Sees images (jpg/png)", "500k context window", "~$2 per 1M words in"]},
    {"id": "llama3.1", "label": "Llama 3.1 via Ollama — offline free", "base_url": "http://localhost:11434/v1",
     "env_key": "", "key_from": "",
     "features": ["No key at all", "100% private, works offline", "Needs Ollama installed", "Weaker + no vision"]},
    {"id": "openai", "label": "Pollinations — no key (unreliable)", "base_url": "https://text.pollinations.ai/openai",
     "env_key": "", "key_from": "",
     "features": ["No signup", "Tiny free allowance, stops with 402 errors", "Use only as last resort"]},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o mini via GitHub — free", "base_url": "https://models.github.ai/inference",
     "env_key": "GITHUB_TOKEN", "key_from": "https://github.com/settings/tokens",
     "features": ["Free with any GitHub account", "Real GPT-4o mini + vision", "Rate-limited quotas"]},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "label": "Llama 3.3 via OpenRouter — free", "base_url": "https://openrouter.ai/api/v1",
     "env_key": "OPENROUTER_API_KEY", "key_from": "https://openrouter.ai/keys",
     "features": ["Free key, fast signup", "Strong open-weights model", "If ID fails, use Custom below"]},
]

# honest limits per brain (measured live where possible)
_BRAIN_LIMITS = {
    "gemini-3-flash-preview": "Free: ~20 requests/day — we hit this wall live, resets in hours",
    "gemini-flash-latest": "Free daily quota, but often overloaded (503) at peak times",
    "openai/gpt-oss-120b": "Free: ~1,000 requests per rolling window, refills in minutes (measured live on your key)",
    "qwen/qwen3.8-27b": "Free Groq quotas, rolling windows like the 120B",
    "gpt-4o-mini": "Paid: ~$0.15/1M words in — $5 lasts months, no daily cap",
    "gpt-4o": "Paid: ~$2.50/1M words in — $5 lasts weeks, no daily cap",
    "muse-spark-1.3": "Paid per word, no daily cap",
    "grok-4.3": "Paid: ~$1.25/1M words in, no daily cap",
    "grok-4.7": "Paid: ~$2/1M words in, no daily cap",
    "llama3.1": "Unlimited — runs on your own PC",
    "openai": "Shared free pool — dies with 402 errors when drained (happened live), back later",
    "openai/gpt-4o-mini": "Free GitHub quotas — modest daily cap, resets daily",
    "meta-llama/llama-3.3-70b-instruct:free": "Free key — modest daily cap (dozens of chats)",
}
for _e in MODEL_CATALOG:
    _e["limits"] = _BRAIN_LIMITS.get(_e["id"], "")

def _backfill_provider_keys():
    # keep per-brain keys so switching brains never loses a saved key
    for e in MODEL_CATALOG:
        if e["id"] == MODEL and e["env_key"] and not os.getenv(e["env_key"], "") \
                and API_KEY and API_KEY not in ("none", "ollama", "free"):
            os.environ[e["env_key"]] = API_KEY
            save_env({e["env_key"]: API_KEY})
def save_env(updates: dict):
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    out, seen = [], set()
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k = s.split("=", 1)[0].strip()
            if k in updates:
                out.append(f"{k}={updates[k]}")
                seen.add(k)
                continue
        out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    ENV_FILE.write_text("\n".join(out) + "\n", encoding="utf-8")

_backfill_provider_keys()

def key_preview():
    if not API_KEY:
        return "none"
    return "••••" + API_KEY[-4:]

@app.get("/api/settings")
def get_settings(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    cat = []
    for e in MODEL_CATALOG:
        ek = e["env_key"]
        cat.append({**e, "has_key": bool(os.getenv(ek, "")) if ek else True})
    return {"model": MODEL, "base_url": BASE_URL or "", "key_preview": key_preview(), "catalog": cat}

class SettingsIn(BaseModel):
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None

@app.post("/api/settings")
def post_settings(inp: SettingsIn, req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    global API_KEY, BASE_URL, MODEL
    updates = {}
    # first: stash the current brain's key so it is never lost by switching
    cur = next((e for e in MODEL_CATALOG if e["id"] == MODEL), None)
    if cur and cur["env_key"] and API_KEY and API_KEY not in ("none", "ollama", "free") \
            and not os.getenv(cur["env_key"], ""):
        os.environ[cur["env_key"]] = API_KEY
        updates[cur["env_key"]] = API_KEY
    entry = next((e for e in MODEL_CATALOG if e["id"] == inp.model), None) if inp.model else None
    if inp.model and not entry:
        # custom model id typed by user — accept it, server address decides the provider
        MODEL = inp.model.strip()
        updates["AI_MODEL"] = MODEL
        entry = None
    if entry:
        MODEL = entry["id"]
        updates["AI_MODEL"] = MODEL
        BASE_URL = entry["base_url"] or None
        updates["AI_BASE_URL"] = entry["base_url"]
    elif inp.base_url is not None:
        BASE_URL = inp.base_url or None
        updates["AI_BASE_URL"] = inp.base_url
    if inp.api_key:
        API_KEY = inp.api_key.strip()
        updates["AI_API_KEY"] = API_KEY
        if entry and entry["env_key"]:
            os.environ[entry["env_key"]] = API_KEY
            updates[entry["env_key"]] = API_KEY
    elif entry:
        # no new key typed: reuse this brain's saved key, or go keyless
        if entry["env_key"]:
            saved = os.getenv(entry["env_key"], "")
            if saved:
                API_KEY = saved
                updates["AI_API_KEY"] = saved
        else:
            API_KEY = "none"
            updates["AI_API_KEY"] = "none"
    os.environ["AI_MODEL"] = MODEL
    os.environ["AI_BASE_URL"] = BASE_URL or ""
    os.environ["AI_API_KEY"] = API_KEY
    save_env(updates)
    return {"ok": True, "model": MODEL, "base_url": BASE_URL or "", "key_preview": key_preview()}

@app.get("/api/ping")
def ping(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    client = get_client()
    if not client:
        return JSONResponse({"error": "no API key saved"}, status_code=400)
    try:
        r = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=200)
        return {"ok": True, "model": MODEL, "reply": (r.choices[0].message.content or "").strip()[:100]}
    except Exception as e:
        return JSONResponse({"error": f"{MODEL} failed: {e}"}, status_code=500)
