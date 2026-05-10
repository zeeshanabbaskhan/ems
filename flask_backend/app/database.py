"""SQLite persistence for users, patients, sessions, alerts, fall incidents (dev-friendly)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask_backend.app.settings import repo_root

_lock = threading.Lock()
_db_path: Path | None = None


def use_mongo() -> bool:
    backend = os.environ.get("EMS_DB_BACKEND", "").strip().lower()
    return backend == "mongo" or bool(os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI"))


def get_db_path() -> Path:
    if use_mongo():
        raise RuntimeError("SQLite DB path is unavailable while MongoDB persistence is enabled")
    global _db_path
    if _db_path is None:
        raw = os.environ.get("EMS_DB_PATH") or os.environ.get("DATABASE_PATH")
        if raw:
            _db_path = Path(raw).expanduser().resolve()
            _db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            d = repo_root() / "data"
            d.mkdir(parents=True, exist_ok=True)
            _db_path = d / "elder_monitor.db"
    return _db_path


@contextmanager
def get_connection() -> Iterator[Any]:
    if use_mongo():
        from flask_backend.app.mongo_database import get_mongo_connection

        with get_mongo_connection() as conn:
            yield conn
        return

    path = get_db_path()
    with _lock:
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_schema() -> None:
    if use_mongo():
        from flask_backend.app.mongo_database import init_mongo_schema

        init_mongo_schema()
        return

    with get_connection() as conn:
        c = conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                username TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                full_name TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                age INTEGER,
                caregiver_id TEXT,
                elder_user_id TEXT,
                home_address TEXT,
                emergency_contact TEXT,
                notes TEXT,
                FOREIGN KEY (caregiver_id) REFERENCES users(id),
                FOREIGN KEY (elder_user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS caregiver_patient (
                caregiver_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                PRIMARY KEY (caregiver_id, patient_id),
                FOREIGN KEY (caregiver_id) REFERENCES users(id),
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
            CREATE TABLE IF NOT EXISTS devices (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                label TEXT,
                platform TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                device_id TEXT NOT NULL,
                status TEXT NOT NULL,
                sample_rate_hz REAL,
                started_at TEXT NOT NULL,
                stopped_at TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                device_id TEXT,
                session_id TEXT,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                score REAL,
                created_at TEXT NOT NULL,
                acknowledged_at TEXT,
                resolved_at TEXT,
                manually_triggered INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS fall_incidents (
                id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                session_id TEXT,
                stage TEXT NOT NULL,
                created_at TEXT NOT NULL,
                response_deadline_at TEXT,
                alarm_deadline_at TEXT,
                fall_probability REAL,
                fall_type_code TEXT,
                response TEXT,
                metadata_json TEXT
            );
            CREATE TABLE IF NOT EXISTS app_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS patient_live (
                patient_id TEXT PRIMARY KEY,
                patient_name TEXT,
                session_id TEXT,
                device_id TEXT,
                severity TEXT,
                score REAL,
                fall_probability REAL,
                predicted_activity_class TEXT,
                last_message TEXT,
                sample_rate_hz REAL,
                active_alert_ids TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )
        _migrate_schema(conn)


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Add columns on existing SQLite DBs (idempotent)."""
    c = conn.cursor()
    c.execute("PRAGMA table_info(patient_live)")
    existing = {row[1] for row in c.fetchall()}
    for col, decl in (
        ("latitude", "REAL"),
        ("longitude", "REAL"),
        ("location_accuracy_m", "REAL"),
        ("location_updated_at", "TEXT"),
        ("heading_degrees", "REAL"),
    ):
        if col not in existing:
            c.execute(f"ALTER TABLE patient_live ADD COLUMN {col} {decl}")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_default_admin() -> None:
    """Create admin@local if no admin row exists (``ADMIN_PASSWORD`` / ``ADMIN_EMAIL``)."""
    import os
    import uuid

    from flask_backend.app.auth_jwt import hash_password

    pwd = os.environ.get("ADMIN_PASSWORD", "admin123")
    email = os.environ.get("ADMIN_EMAIL", "admin@local")
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        if c.fetchone()[0] > 0:
            return
        uid = uuid.uuid4().hex
        c.execute(
            "INSERT INTO users (id, email, username, password_hash, role, full_name, created_at) VALUES (?,?,?,?,?,?,?)",
            (uid, email, None, hash_password(pwd), "admin", "Administrator", iso_now()),
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}
