import sqlite3
import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(override=True)

DB_PATH = Path(__file__).resolve().parent / "users.db"
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip()
    if DATABASE_URL.startswith("DATABASE_URL="):
        DATABASE_URL = DATABASE_URL.replace("DATABASE_URL=", "", 1).strip()
    if (DATABASE_URL.startswith('"') and DATABASE_URL.endswith('"')) or (DATABASE_URL.startswith("'") and DATABASE_URL.endswith("'")):
        DATABASE_URL = DATABASE_URL[1:-1].strip()
IS_POSTGRES = bool(DATABASE_URL)

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DBIntegrityError = psycopg2.IntegrityError
else:
    DBIntegrityError = sqlite3.IntegrityError

def get_db_connection():
    """Establish a connection to either PostgreSQL or SQLite depending on configuration."""
    if IS_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def init_db():
    """Create database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if IS_POSTGRES:
        # Create Users table (PostgreSQL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Interviews table (PostgreSQL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id VARCHAR(255) PRIMARY KEY,
                user_id INTEGER NOT NULL,
                duration_seconds INTEGER NOT NULL,
                overall_score REAL,
                overall_label VARCHAR(255),
                timestamp DOUBLE PRECISION NOT NULL,
                report_json TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
    else:
        # Create Users table (SQLite)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Interviews table (SQLite)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                duration_seconds INTEGER NOT NULL,
                overall_score REAL,
                overall_label TEXT,
                timestamp REAL NOT NULL,
                report_json TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
    conn.commit()
    conn.close()

# ---------------------------------------------------------------------------
# Password Hashing Helpers (PBKDF2-SHA256)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    """Generate a secure PBKDF2 hash of a password."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return f"{salt.hex()}:{key.hex()}"

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify a PBKDF2 hash against a password."""
    try:
        salt_hex, key_hex = hashed_password.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
        return new_key == key
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Database Queries
# ---------------------------------------------------------------------------

def create_user(username: str, password_raw: str) -> bool:
    """Create a new user. Returns True if successful, False if username exists."""
    hashed = hash_password(password_raw)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if IS_POSTGRES:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username.strip(), hashed)
            )
        else:
            cursor.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username.strip(), hashed)
            )
        conn.commit()
        return True
    except DBIntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user details by username."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if IS_POSTGRES:
        cursor.execute("SELECT * FROM users WHERE username = %s", (username.strip(),))
    else:
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.strip(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve user details by user ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if IS_POSTGRES:
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    else:
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def save_interview(session_id: str, user_id: int, duration_secs: int, report: Dict[str, Any]) -> None:
    """Save completed interview report associated with the user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    overall_score = report.get("overall_score", 0.0)
    overall_label = report.get("overall_label", "Needs Work")
    timestamp = report.get("turns", [{}])[0].get("timestamp", 0.0) if report.get("turns") else 0.0
    report_str = json.dumps(report)
    
    if IS_POSTGRES:
        cursor.execute("""
            INSERT INTO interviews (id, user_id, duration_seconds, overall_score, overall_label, timestamp, report_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                duration_seconds = EXCLUDED.duration_seconds,
                overall_score = EXCLUDED.overall_score,
                overall_label = EXCLUDED.overall_label,
                timestamp = EXCLUDED.timestamp,
                report_json = EXCLUDED.report_json
        """, (session_id, user_id, duration_secs, overall_score, overall_label, timestamp, report_str))
    else:
        cursor.execute("""
            INSERT OR REPLACE INTO interviews (id, user_id, duration_seconds, overall_score, overall_label, timestamp, report_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, duration_secs, overall_score, overall_label, timestamp, report_str))
    
    conn.commit()
    conn.close()

def get_user_interviews(user_id: int) -> List[Dict[str, Any]]:
    """Retrieve all interview reports for a user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if IS_POSTGRES:
        cursor.execute("""
            SELECT * FROM interviews WHERE user_id = %s ORDER BY timestamp DESC
        """, (user_id,))
    else:
        cursor.execute("""
            SELECT * FROM interviews WHERE user_id = ? ORDER BY timestamp DESC
        """, (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for r in rows:
        item = dict(r)
        if item.get("report_json"):
            item["report"] = json.loads(item["report_json"])
        results.append(item)
    return results

def get_interview_by_id(interview_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a single interview report by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if IS_POSTGRES:
        cursor.execute("SELECT * FROM interviews WHERE id = %s", (interview_id,))
    else:
        cursor.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        item = dict(row)
        if item.get("report_json"):
            item["report"] = json.loads(item["report_json"])
        return item
    return None
