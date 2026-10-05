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

def save_mem(m):
    MEM_FILE.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    if os.getenv("DATA_DIR", ""):
        # cloud volume: sync in background so replies stay fast
        import threading

        def _commit():
            try:
                import modal
                modal.Volume.from_name("agent010-data").commit()
            except Exception:
                pass

        threading.Thread(target=_commit, daemon=True).start()

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
        with sync_playwright() as p:
            b = p.chromium.launch(headless=True)
            pg = b.new_page()
            pg.goto(url, timeout=25000, wait_until="domcontentloaded")
            title = pg.title()
            text = pg.inner_text("body")[:5000]
            b.close()
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
    system = f"""You are 010, a sharp friend who manages the user's life over text. Reply in ONE short message (1-3 sentences), contractions, no essays and no bullet lists unless asked. Only use /// to split into two texts on rare occasions when there are genuinely two separate thoughts.
Your name is 010. Current local time: {now.strftime("%Y-%m-%d %H:%M (%A)")}.
Help user fix their life and achieve goals. Be present: if something is due, check on them directly ("gym time — you locked in?").
Below you get the FULL conversation history plus MEMORY. Read the user's new message,
then read ALL past messages for context, then answer using everything you know.
You have live tools: web_search (current facts), browse_page (open links in Chrome), get_datetime (exact time) and save_insight (stash an interesting find to tell them later). Use them instead of guessing.
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

def cascade_delete(m, content: str):
    """Wipe every trace of a deleted message: auto-facts, goals, learned name."""
    profile = m.get("profile", {})
    m["facts"] = [f for f in m.get("facts", [])
                  if not (f.get("type") == "auto" and f.get("text") == content[:500])]
    g = content.strip()[:300]
    profile["goals"] = [x for x in profile.get("goals", []) if x != g]
    nm = extract_name(content)
    if nm and profile.get("name") == nm:
        profile["name"] = ""

def cascade_edit(m, old: str, new: str):
    """Move every trace of an edited message to the new text."""
    profile = m.get("profile", {})
    for f in m.get("facts", []):
        if f.get("type") == "auto" and f.get("text") == old[:500]:
            f["text"] = new[:500]
    g_old, g_new = old.strip()[:300], new.strip()[:300]
    profile["goals"] = [g_new if x == g_old else x for x in profile.get("goals", [])]
    nm = extract_name(new)
    if nm:
        profile["name"] = nm

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
    cascade_delete(m, gone.get("content", ""))
    save_mem(m)
    return {"ok": True, "wiped": True}

@app.get("/api/export")
def export_mem(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = ensure_ids(load_mem())
    return FileResponse(MEM_FILE, filename="010-memory.json", media_type="application/json")

@app.post("/api/import")
async def import_mem(req: Request, file: UploadFile = File(...)):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        d = json.loads((await file.read()).decode("utf-8"))
    except Exception:
        return JSONResponse({"error": "not a valid 010-memory.json file"}, status_code=400)
    if not isinstance(d, dict) or "conversations" not in d:
        return JSONResponse({"error": "not a valid 010-memory.json file"}, status_code=400)
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
    save_mem(m)
    return {"ok": True}

@app.post("/api/clear")
def clear_chat(req: Request):
    if not need_auth(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    m = load_mem()
    m["conversations"] = []
    save_mem(m)
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
