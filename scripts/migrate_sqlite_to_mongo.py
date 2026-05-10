"""Copy EMS SQLite data into MongoDB.

Example:
    python scripts/migrate_sqlite_to_mongo.py \
      --sqlite-path data/elder_monitor.db \
      --mongo-uri "mongodb+srv://USER:PASSWORD@HOST/ems" \
      --db ems
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from pymongo import MongoClient, ReplaceOne

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


def _doc_id(table: str, row: dict[str, Any]) -> str:
    if "id" in row and row["id"]:
        return str(row["id"])
    if table == "patient_live":
        return str(row["patient_id"])
    if table == "caregiver_patient":
        return f"{row['caregiver_id']}:{row['patient_id']}"
    raise ValueError(f"Cannot derive Mongo _id for {table}")


def migrate(sqlite_path: Path, mongo_uri: str, db_name: str) -> None:
    if not sqlite_path.is_file():
        raise FileNotFoundError(sqlite_path)

    source = sqlite3.connect(sqlite_path)
    source.row_factory = sqlite3.Row
    db = MongoClient(mongo_uri)[db_name]

    for table in TABLES:
        rows = [dict(row) for row in source.execute(f"SELECT * FROM {table}").fetchall()]
        if not rows:
            print(f"{table}: 0 rows")
            continue
        ops = []
        for row in rows:
            row["_id"] = _doc_id(table, row)
            ops.append(ReplaceOne({"_id": row["_id"]}, row, upsert=True))
        result = db[table].bulk_write(ops, ordered=False)
        print(f"{table}: {len(rows)} rows, upserted={result.upserted_count}, modified={result.modified_count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", type=Path, default=Path("data/elder_monitor.db"))
    parser.add_argument("--mongo-uri", required=True)
    parser.add_argument("--db", default="ems")
    args = parser.parse_args()
    migrate(args.sqlite_path, args.mongo_uri, args.db)


if __name__ == "__main__":
    main()
