"""
app.py
------
Antigravity-style Stateful Agent Engine using OpenRouter API.
FastAPI server with WebSocket interview loop, session management,
background time-guard, token compaction, and analytics compilation.
"""

import asyncio
import json
import logging
import os
import time
import uuid
import hmac
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from dotenv import load_dotenv

from resume_processor import extract_resume_text
from analytics import TurnLog, compile_report, report_to_dict
from database import init_db, create_user, get_user_by_username, get_user_by_id, save_interview, get_user_interviews, get_interview_by_id, verify_password

load_dotenv(override=True)

# Initialize local SQLite DB on startup
init_db()

COOKIE_SECRET = b"antigravity_secret_key_12345"

def sign_value(val: str) -> str:
    signature = hmac.new(COOKIE_SECRET, val.encode('utf-8'), hashlib.sha256).hexdigest()
    return f"{val}:{signature}"

def verify_signed_value(signed_val: str) -> Optional[str]:
    try:
        val, signature = signed_val.split(":", 1)
        expected = hmac.new(COOKIE_SECRET, val.encode('utf-8'), hashlib.sha256).hexdigest()
        if hmac.compare_digest(signature, expected):
            return val
    except Exception:
        pass
    return None

def get_current_user(request: Request) -> Optional[dict]:
    cookie = request.cookies.get("session")
    if not cookie:
        return None
    user_id_str = verify_signed_value(cookie)
    if not user_id_str:
        return None
    return get_user_by_id(int(user_id_str))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & Templates
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates").replace("\\", "/"))

app = FastAPI(title="AI Mock Interview Platform", version="1.0.0")

# ---------------------------------------------------------------------------
# In-memory session store  {session_id: SessionData}
# ---------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Gemini client (lazy-init per session so API key is validated at runtime)
# ---------------------------------------------------------------------------

def _get_openai_client():
    from openai import AsyncOpenAI
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not configured on the server. Please check your .env file.")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a senior technical interviewer at a top-tier technology company.
You have been given the following candidate resume for context:

--- RESUME START ---
{resume_text}
--- RESUME END ---

STRICT RULES YOU MUST FOLLOW:
1. Your very first message must be exactly: "Welcome to your interview. To kick things off, please tell me a bit about yourself."
2. After that, ask exactly ONE short, direct, technical question per turn. Keep questions very concise (maximum 1 to 2 sentences). Avoid long explanations or preambles.
3. Structure the difficulty of your questions dynamically based on the conversation length:
   - First question (after candidate's intro): Ask an EASY, introductory question about a tool, framework, or project on their resume.
   - Next 2-3 questions: Ask MEDIUM difficulty questions focusing on practical implementation, design decisions, or simple troubleshooting.
   - Subsequent questions: Ask HARD/DEEP technical questions focusing on scalability, trade-offs, optimization, edge cases, or deep internal workings.
4. Base every question specifically on the candidate's actual projects, frameworks, tools, or achievements mentioned in the resume above.
5. Never repeat a question already asked in this conversation.
6. Do not give feedback or evaluation during the interview — only ask questions.
7. If the candidate's answer is vague, ask a short follow-up that digs deeper into a specific technical detail (matching the current difficulty level).
8. When you receive a signal that time is up, your final message must be exactly: "Thank you, that brings us to the end of our time today. You did great — results will be available shortly."
9. Maintain a professional, encouraging, but rigorous tone throughout.
"""

COMPACTION_PROMPT = """Below is a segment of a technical interview conversation. 
Summarise it concisely in bullet points, preserving all key technical topics discussed and answers given.
Keep it under 200 words.

{history_segment}
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/static/{filename}")
async def get_static_file(filename: str):
    filepath = BASE_DIR / "static" / filename
    if filepath.exists():
        return FileResponse(filepath)
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the auth landing page (login/register)."""
    current_user = get_current_user(request)
    if current_user:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "auth.html")


@app.post("/login")
async def login_post(
    username: str = Form(...),
    password: str = Form(...),
):
    """Authenticate credentials and set session cookie."""
    user = get_user_by_username(username)
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    
    response = JSONResponse({"message": "Successfully signed in!"})
    signed_cookie = sign_value(str(user["id"]))
    response.set_cookie("session", signed_cookie, httponly=True, max_age=86400)
    return response


@app.post("/register")
async def register_post(
    username: str = Form(...),
    password: str = Form(...),
):
    """Register a new user account."""
    if len(username.strip()) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    
    success = create_user(username, password)
    if not success:
        raise HTTPException(status_code=400, detail="Username already exists. Please choose another one.")
    
    return JSONResponse({"message": "Account created successfully! Please sign in."})


@app.get("/logout")
async def logout():
    """Clear session cookie and redirect to login."""
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    """Serve the configuration landing page, redirecting to login if needed."""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse("/login", status_code=303)
    
    past_interviews = get_user_interviews(current_user["id"])
    
    logger.info(f"[Route] GET / -- serving index.html for user: {current_user['username']}")
    return templates.TemplateResponse(request, "index.html", {
        "request": request,
        "user": current_user,
        "interviews": past_interviews
    })


@app.post("/start")
async def start_session(
    request: Request,
    resume: UploadFile = File(...),
    duration: int = Form(...),
):
    """Create a new interview session."""
    current_user = get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
        
    logger.info(f"[Route] POST /start — duration={duration}min, file={resume.filename}")

    # Validate file type
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await resume.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract resume text
    try:
        resume_text = extract_resume_text(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build session
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "session_id": session_id,
        "user_id": current_user["id"],
        "resume_text": resume_text,
        "duration_seconds": duration * 60,
        "start_time": None,           # set when WS connects
        "turns": [],                  # list of TurnLog
        "conversation_history": [],   # history
        "is_active": False,
        "report": None,
    }

    logger.info(f"[Session] ✅ Created session {session_id[:8]}… ({duration} min) for user_id={current_user['id']}")
    return JSONResponse({"session_id": session_id})


# ---------------------------------------------------------------------------
# Token compaction helper (pre_turn lifecycle hook equivalent)
# ---------------------------------------------------------------------------

async def _compact_history_if_needed(session: Dict, client, model: str):
    """
    If conversation history exceeds 6 candidate turns, summarise older turns
    using a temporary child chat call — isolating it from the main interview context.
    Locks the resume text and keeps last 3 turns intact.
    """
    candidate_turns = [t for t in session["turns"] if t.role == "candidate"]
    if len(candidate_turns) <= 6:
        return

    logger.info("[PreTurn] 🔄 Token compaction triggered — summarising older turns.")

    # Build history segment to compress (all but last 3 candidate exchanges)
    cutoff = len(session["conversation_history"]) - 6
    if cutoff <= 0:
        return

    old_segment = session["conversation_history"][1:cutoff] # Skip index 0 (system prompt)
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in old_segment
    )

    # Spawn sub-agent call (isolated single-shot OpenAI call)
    try:
        compaction_prompt = COMPACTION_PROMPT.format(history_segment=history_text)
        sub_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": compaction_prompt}]
        )
        summary = sub_response.choices[0].message.content.strip()
        logger.info(f"[PreTurn] ✅ Compacted {cutoff} messages into summary.")

        # Replace old history with summary block, preserve recent turns (and system prompt)
        summary_message = {
            "role": "user",
            "content": f"[INTERVIEW HISTORY SUMMARY]\n{summary}"
        }
        session["conversation_history"] = (
            [session["conversation_history"][0], summary_message] + session["conversation_history"][cutoff:]
        )
    except Exception as e:
        logger.warning(f"[PreTurn] ⚠️  Compaction failed (non-critical): {e}")


# ---------------------------------------------------------------------------
# WebSocket — Interview Engine
# ---------------------------------------------------------------------------

@app.websocket("/ws/{session_id}")
async def interview_websocket(websocket: WebSocket, session_id: str):
    """
    Bidirectional real-time interview channel.
    Manages the full Gemini stateful chat loop.
    """
    if session_id not in sessions:
        await websocket.close(code=4004)
        return

    cookie = websocket.cookies.get("session")
    if not cookie:
        await websocket.close(code=4001)
        return
    user_id_str = verify_signed_value(cookie)
    if not user_id_str:
        await websocket.close(code=4001)
        return

    user_id = int(user_id_str)
    session = sessions[session_id]
    if session.get("user_id") != user_id:
        await websocket.close(code=4003)
        return

    await websocket.accept()
    session["is_active"] = True
    session["start_time"] = time.time()

    resume_text = session["resume_text"]
    duration_secs = session["duration_seconds"]

    logger.info(f"[WS] 🔌 Connected — session {session_id[:8]}…")

    try:
        client = _get_openai_client()
    except Exception as e:
        await websocket.send_json({"type": "error", "message": f"Server Configuration Error: {e}"})
        await websocket.close()
        return

    model = "google/gemma-4-31b-it:free"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(resume_text=resume_text)

    # Initialize OpenAI chat history with system prompt
    session["conversation_history"] = [
        {"role": "system", "content": system_prompt}
    ]

    # -----------------------------------------------------------------------
    # Background time-guard coroutine
    # -----------------------------------------------------------------------
    async def time_guard():
        while session["is_active"]:
            await asyncio.sleep(5)
            elapsed = time.time() - session["start_time"]
            remaining = duration_secs - elapsed
            # Send countdown update every 5 s
            if session["is_active"]:
                try:
                    await websocket.send_json({
                        "type": "timer",
                        "elapsed": round(elapsed),
                        "remaining": max(0, round(remaining)),
                        "duration": duration_secs,
                    })
                except Exception:
                    break

            if elapsed >= duration_secs:
                logger.info(f"[TimeGuard] ⏰ Time limit reached for {session_id[:8]}…")
                if session["is_active"]:
                    session["is_active"] = False
                    try:
                        await websocket.send_json({"type": "time_up"})
                    except Exception:
                        pass
                break

    timer_task = asyncio.create_task(time_guard())

    # -----------------------------------------------------------------------
    # FIRST TURN — mandatory greeting
    # -----------------------------------------------------------------------
    try:
        greeting = "Welcome to your interview. To kick things off, please tell me a bit about yourself."
        session["turns"].append(TurnLog(role="interviewer", text=greeting, timestamp=time.time()))
        session["conversation_history"].append({
            "role": "assistant",
            "content": greeting
        })

        await websocket.send_json({
            "type": "ai_message",
            "text": greeting,
            "turn": 1,
        })
        logger.info(f"[WS] 🎙️ Greeting sent to {session_id[:8]}…")

    except Exception as e:
        logger.error(f"[WS] First turn failed: {e}")
        await websocket.close()
        timer_task.cancel()
        return

    # -----------------------------------------------------------------------
    # Main interview loop
    # -----------------------------------------------------------------------
    turn_number = 2

    try:
        while session["is_active"]:
            # Receive candidate message
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                logger.warning("[WS] Candidate silent for 5 min — closing session.")
                break
            except WebSocketDisconnect:
                logger.info("[WS] Client disconnected.")
                break

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"type": "candidate_message", "text": raw}

            if data.get("type") == "end_interview":
                logger.info(f"[WS] 🛑 Manual end requested — {session_id[:8]}…")
                session["is_active"] = False
                break

            candidate_text = data.get("text", "").strip()
            if not candidate_text:
                continue

            # Log candidate turn
            session["turns"].append(
                TurnLog(role="candidate", text=candidate_text, timestamp=time.time())
            )
            session["conversation_history"].append({
                "role": "user",
                "content": candidate_text
            })

            # Time check before generating next question
            elapsed = time.time() - session["start_time"]
            if elapsed >= duration_secs:
                session["is_active"] = False
                break

            # PRE-TURN: token compaction hook
            await _compact_history_if_needed(session, client, model)

            # Generate AI question
            max_retries = 3
            ai_text = None
            for attempt in range(max_retries):
                try:
                    ai_response = await client.chat.completions.create(
                        model=model,
                        messages=session["conversation_history"]
                    )
                    ai_text = ai_response.choices[0].message.content.strip()
                    break
                except Exception as e:
                    error_msg = str(e).lower()
                    if "429" in error_msg or "quota" in error_msg or "too many requests" in error_msg:
                        if attempt < max_retries - 1:
                            logger.warning(f"Rate limit hit. Retrying in 5s... (Attempt {attempt+1}/{max_retries})")
                            await websocket.send_json({"type": "error", "message": "Handling high traffic... retrying..."})
                            await asyncio.sleep(5)
                            continue
                        else:
                            user_msg = "Rate limit exceeded (Too many requests). Please wait a minute and try again."
                    else:
                        user_msg = "AI response failed. Please try again."
                    
                    logger.error(f"[WS] API error on turn {turn_number}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": user_msg,
                    })
                    break
            
            if not ai_text:
                continue

            # Log AI turn
            session["turns"].append(
                TurnLog(role="interviewer", text=ai_text, timestamp=time.time())
            )
            session["conversation_history"].append({
                "role": "assistant",
                "content": ai_text
            })

            await websocket.send_json({
                "type": "ai_message",
                "text": ai_text,
                "turn": turn_number,
            })

            logger.info(f"[WS] Turn {turn_number} complete — {session_id[:8]}…")
            turn_number += 1

    except WebSocketDisconnect:
        logger.info(f"[WS] WebSocket disconnected — {session_id[:8]}…")
    finally:
        # Cancel timer
        session["is_active"] = False
        timer_task.cancel()

        # Send closing statement
        closing = "Thank you, that brings us to the end of our time today. You did great — results will be available shortly."
        try:
            await websocket.send_json({"type": "ai_message", "text": closing, "turn": turn_number})
            session["turns"].append(TurnLog(role="interviewer", text=closing, timestamp=time.time()))
        except Exception:
            pass

        # Compile analytics report
        elapsed = time.time() - session["start_time"] if session["start_time"] else 1
        try:
            report = compile_report(
                turns=session["turns"],
                elapsed_seconds=elapsed,
                resume_text=resume_text,
            )
            session["report"] = report_to_dict(report)
            logger.info(f"[Session] 📊 Analytics compiled for {session_id[:8]}…")
            
            # Save completed interview to SQLite DB
            user_id = session.get("user_id")
            if user_id:
                save_interview(
                    session_id=session_id,
                    user_id=user_id,
                    duration_secs=session["duration_seconds"],
                    report=session["report"]
                )
                logger.info(f"[Session] 💾 Saved session {session_id[:8]}… to SQLite DB")
        except Exception as e:
            logger.error(f"[Session] Analytics failed: {e}")
            session["report"] = {"error": str(e)}

        # Signal client to navigate to dashboard
        try:
            await websocket.send_json({
                "type": "session_complete",
                "session_id": session_id,
            })
        except Exception:
            pass

        logger.info(f"[WS] 🔒 Session {session_id[:8]}… closed cleanly.")


# ---------------------------------------------------------------------------
# Results routes
# ---------------------------------------------------------------------------

@app.get("/results/{session_id}", response_class=HTMLResponse)
async def results_page(request: Request, session_id: str):
    """Serve the analytics dashboard HTML, verifying auth and ownership."""
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse("/login", status_code=303)
        
    # Check memory first, then DB
    session = sessions.get(session_id)
    if not session:
        db_session = get_interview_by_id(session_id)
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found.")
        owner_id = db_session["user_id"]
    else:
        owner_id = session.get("user_id")

    # Authorize ownership
    if owner_id != current_user["id"]:
        raise HTTPException(status_code=403, detail="Forbidden. You do not own this report.")

    return templates.TemplateResponse(request, "dashboard.html", {"session_id": session_id})


@app.get("/api/results/{session_id}")
async def results_api(request: Request, session_id: str):
    """Return raw JSON analytics payload for the dashboard to consume."""
    current_user = get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Check memory first, then DB
    session = sessions.get(session_id)
    if not session:
        db_session = get_interview_by_id(session_id)
        if not db_session:
            raise HTTPException(status_code=404, detail="Session not found.")
        if db_session["user_id"] != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        return JSONResponse(db_session["report"])
    else:
        if session.get("user_id") != current_user["id"]:
            raise HTTPException(status_code=403, detail="Forbidden")
        if session.get("report") is None:
            raise HTTPException(status_code=202, detail="Report not yet ready.")
        return JSONResponse(session["report"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting AI Mock Interview Platform...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
