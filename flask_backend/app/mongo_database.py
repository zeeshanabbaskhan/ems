"""MongoDB persistence adapter for the monitoring API.

The route layer historically used a small sqlite cursor API.  This adapter
implements the query shapes used by the app so routes can move to MongoDB
without a risky full rewrite in one pass.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_client: Any | None = None
_db: Any | None = None

TABLES = (
    "users",
    "patients",
    "caregiver_patient",
    "devices",
    "sessions",
    "alerts",
    "fall_incidents",
    "app_events",
    "patient_live",
)


class MongoRow(dict):
    def __init__(self, data: dict[str, Any], order: list[str] | None = None):
        clean = {k: v for k, v in data.items() if k != "_id"}
        super().__init__(clean)
        self._order = order or list(clean.keys())

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return dict.__getitem__(self, self._order[key])
        return dict.get(self, key)


def mongo_uri() -> str | None:
    return os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")


def mongo_db_name() -> str:
    return os.environ.get("MONGO_DB_NAME") or os.environ.get("MONGODB_DB") or "ems"


def get_mongo_db() -> Any:
    global _client, _db
    if _db is None:
        uri = mongo_uri()
        if not uri:
            raise RuntimeError("MongoDB is enabled but MONGO_URI/MONGODB_URI is not set")
        from pymongo import MongoClient

        _client = MongoClient(uri)
        _db = _client[mongo_db_name()]
    return _db


def init_mongo_schema() -> None:
    db = get_mongo_db()
    db.users.create_index("email", unique=True, sparse=True)
    db.users.create_index("username", unique=True, sparse=True)
    db.users.create_index("role")
    db.patients.create_index("caregiver_id")
    db.patients.create_index("elder_user_id")
    db.caregiver_patient.create_index(
        [("caregiver_id", 1), ("patient_id", 1)], unique=True
    )
    db.caregiver_patient.create_index("patient_id")
    db.devices.create_index("patient_id")
    db.sessions.create_index("patient_id")
    db.sessions.create_index("status")
    db.alerts.create_index([("patient_id", 1), ("status", 1)])
    db.alerts.create_index("created_at")
    db.fall_incidents.create_index([("patient_id", 1), ("stage", 1)])
    db.app_events.create_index("event_type")
    db.patient_live.create_index("patient_id", unique=True)


@contextmanager
def get_mongo_connection() -> Iterator["MongoConnection"]:
    yield MongoConnection(get_mongo_db())


class MongoConnection:
    def __init__(self, db: Any):
        self.db = db

    def cursor(self) -> "MongoCursor":
        return MongoCursor(self.db)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


class MongoCursor:
    def __init__(self, db: Any):
        self.db = db
        self._rows: list[MongoRow] = []

    def fetchone(self) -> MongoRow | None:
        if not self._rows:
            return None
        return self._rows.pop(0)

    def fetchall(self) -> list[MongoRow]:
        rows = self._rows
        self._rows = []
        return rows

    def execute(self, sql: str, params: Any = ()) -> "MongoCursor":
        params = tuple(params or ())
        raw = sql.strip()
        q = _norm(raw)
        if not q or q.startswith("create table") or q.startswith("pragma") or q.startswith("alter table"):
            self._rows = []
            return self
        if q.startswith("insert "):
            self._insert(raw, q, params)
        elif q.startswith("select "):
            self._select(raw, q, params)
        elif q.startswith("update "):
            self._update(raw, q, params)
        elif q.startswith("delete "):
            self._delete(raw, q, params)
        else:
            raise NotImplementedError(f"Unsupported Mongo SQL adapter query: {raw}")
        return self

    def executescript(self, _sql: str) -> "MongoCursor":
        init_mongo_schema()
        self._rows = []
        return self

    def _col(self, table: str) -> Any:
        return self.db[table]

    def _insert(self, raw: str, q: str, params: tuple[Any, ...]) -> None:
        if "on conflict(patient_id) do update" in q:
            cols = _columns(raw)
            doc = dict(zip(cols, params))
            doc["_id"] = doc["patient_id"]
            self._col("patient_live").replace_one(
                {"patient_id": doc["patient_id"]}, doc, upsert=True
            )
            self._rows = []
            return

        table = _insert_table(q)
        cols = _columns(raw)
        values = _values(raw)
        doc: dict[str, Any] = {}
        pi = 0
        for col, token in zip(cols, values):
            if token == "?":
                doc[col] = params[pi]
                pi += 1
            elif token.upper() == "NULL":
                doc[col] = None
            else:
                doc[col] = token.strip("'\"")
        if "id" in doc:
            doc["_id"] = doc["id"]
        elif table == "patient_live" and "patient_id" in doc:
            doc["_id"] = doc["patient_id"]
        elif table == "caregiver_patient":
            doc["_id"] = f"{doc.get('caregiver_id')}:{doc.get('patient_id')}"

        try:
            if "insert or ignore into" in q:
                self._col(table).update_one({"_id": doc["_id"]}, {"$setOnInsert": doc}, upsert=True)
            else:
                self._col(table).insert_one(doc)
        except Exception as exc:
            if "DuplicateKeyError" in exc.__class__.__name__:
                raise
            raise
        self._rows = []

    def _select(self, raw: str, q: str, params: tuple[Any, ...]) -> None:
        special = self._select_special(q, params)
        if special is not None:
            self._rows = special
            return

        if " count(*) " in f" {q} ":
            table = _from_table(q)
            filt = _where_filter(q, params)
            count = self._col(table).count_documents(filt)
            self._rows = [MongoRow({"COUNT(*)": count}, ["COUNT(*)"])]
            return

        table = _from_table(q)
        cols = _select_columns(raw)
        filt = _where_filter(q, params)
        docs = list(self._col(table).find(filt))
        docs = _sort_docs(docs, q)
        docs = _limit_docs(docs, q)
        self._rows = [_project(d, cols) for d in docs]

    def _select_special(self, q: str, params: tuple[Any, ...]) -> list[MongoRow] | None:
        if "from patients p" in q and "exists" in q:
            patient_id = str(params[0])
            caregiver_id = str(params[1])
            patient = self._col("patients").find_one({"id": patient_id})
            linked = patient and patient.get("caregiver_id") == caregiver_id
            if not linked:
                linked = self._col("caregiver_patient").find_one(
                    {"patient_id": patient_id, "caregiver_id": caregiver_id}
                )
            return [MongoRow({"id": patient_id}, ["id"])] if linked else []

        if "from patients p left join users u" in q:
            rows: list[MongoRow] = []
            patients = list(self._col("patients").find({}))
            patients.sort(key=lambda d: (d.get("full_name") or "").lower())
            for p in patients:
                elder = None
                if p.get("elder_user_id"):
                    elder = self._col("users").find_one(
                        {"id": p.get("elder_user_id"), "role": "elder"}
                    )
                row = {
                    "id": p.get("id"),
                    "full_name": p.get("full_name"),
                    "age": p.get("age"),
                    "caregiver_id": p.get("caregiver_id"),
                    "elder_user_id": p.get("elder_user_id"),
                    "elder_username": elder.get("username") if elder else None,
                }
                rows.append(MongoRow(row, list(row.keys())))
            return rows

        return None

    def _update(self, raw: str, q: str, params: tuple[Any, ...]) -> None:
        table = q.split()[1]
        set_part = re.search(r"set\s+(.+?)\s+where\s+", raw, flags=re.I | re.S)
        where_part = re.search(r"\swhere\s+(.+)$", raw, flags=re.I | re.S)
        if not set_part or not where_part:
            raise NotImplementedError(f"Unsupported UPDATE: {raw}")
        assignments = [a.strip() for a in set_part.group(1).split(",")]
        pi = 0
        update: dict[str, Any] = {}
        for assignment in assignments:
            col = assignment.split("=", 1)[0].strip()
            update[col] = params[pi]
            pi += 1
        filt = _where_filter("select * from x where " + where_part.group(1), params[pi:])
        self._col(table).update_many(filt, {"$set": update})
        self._rows = []

    def _delete(self, raw: str, q: str, params: tuple[Any, ...]) -> None:
        table = q.split()[2]
        where = re.search(r"\swhere\s+(.+)$", raw, flags=re.I | re.S)
        filt = _where_filter("select * from x where " + where.group(1), params) if where else {}
        self._col(table).delete_many(filt)
        self._rows = []


def _norm(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).lower()


def _insert_table(q: str) -> str:
    m = re.search(r"insert(?:\s+or\s+ignore)?\s+into\s+(\w+)", q)
    if not m:
        raise NotImplementedError(f"Cannot parse INSERT table: {q}")
    return m.group(1)


def _from_table(q: str) -> str:
    m = re.search(r"\sfrom\s+(\w+)", q)
    if not m:
        raise NotImplementedError(f"Cannot parse SELECT table: {q}")
    return m.group(1)


def _columns(sql: str) -> list[str]:
    m = re.search(r"\((.*?)\)\s*values", sql, flags=re.I | re.S)
    if not m:
        raise NotImplementedError(f"Cannot parse columns: {sql}")
    return [c.strip() for c in m.group(1).split(",")]


def _values(sql: str) -> list[str]:
    m = re.search(r"values\s*\((.*?)\)", sql, flags=re.I | re.S)
    if not m:
        raise NotImplementedError(f"Cannot parse values: {sql}")
    return [v.strip() for v in m.group(1).split(",")]


def _select_columns(sql: str) -> list[str] | None:
    m = re.search(r"select\s+(.+?)\s+from\s+", sql, flags=re.I | re.S)
    if not m:
        return None
    cols = m.group(1).strip()
    if cols == "*":
        return None
    return [c.strip().split()[-1] for c in cols.split(",")]


def _where_filter(q: str, params: tuple[Any, ...]) -> dict[str, Any]:
    if " where " not in q:
        return {}
    where = q.split(" where ", 1)[1]
    where = re.split(r" order by | limit ", where)[0]
    where = where.replace("1=1", "").strip()
    where = re.sub(r"^(and|or)\s+", "", where)
    if not where:
        return {}

    filt: dict[str, Any] = {}
    pi = 0
    for part in re.split(r"\s+and\s+", where):
        part = part.strip(" ()")
        if not part:
            continue
        in_lit = re.match(r"(?:\w+\.)?(\w+)\s+in\s*\(([^)]*)\)", part)
        if in_lit:
            col = in_lit.group(1)
            values = []
            for token in [t.strip() for t in in_lit.group(2).split(",") if t.strip()]:
                if token == "?":
                    values.append(params[pi])
                    pi += 1
                else:
                    values.append(token.strip("'\""))
            filt[col] = {"$in": values}
            continue
        eq_param = re.match(r"(?:\w+\.)?(\w+)\s*=\s*\?", part)
        if eq_param:
            filt[eq_param.group(1)] = params[pi]
            pi += 1
            continue
        eq_lit = re.match(r"(?:\w+\.)?(\w+)\s*=\s*'([^']*)'", part)
        if eq_lit:
            filt[eq_lit.group(1)] = eq_lit.group(2)
            continue
    return filt


def _project(doc: dict[str, Any], cols: list[str] | None) -> MongoRow:
    if cols is None:
        data = {k: v for k, v in doc.items() if k != "_id"}
        return MongoRow(data)
    data = {c: doc.get(c) for c in cols}
    return MongoRow(data, cols)


def _sort_docs(docs: list[dict[str, Any]], q: str) -> list[dict[str, Any]]:
    if "order by" not in q:
        return docs
    desc = " desc" in q
    m = re.search(r"order by\s+(\w+)", q)
    key = m.group(1) if m else "id"
    return sorted(docs, key=lambda d: d.get(key) or "", reverse=desc)


def _limit_docs(docs: list[dict[str, Any]], q: str) -> list[dict[str, Any]]:
    m = re.search(r"limit\s+(\d+)", q)
    if not m:
        return docs
    return docs[: int(m.group(1))]
