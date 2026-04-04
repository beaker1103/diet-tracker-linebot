"""
資料庫模組：支援 SQLite（本機，未設 DATABASE_URL）與 PostgreSQL（Supabase 等）。
"""

import json
import os
import sqlite3
from datetime import date, datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = "diet_tracker.db"


class Database:
    def __init__(self, db_path: str = DB_PATH, database_url: str | None = None):
        self.db_path = db_path
        self._database_url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        self._pg = bool(self._database_url)

    def _adapt(self, sql: str) -> str:
        if self._pg:
            return sql.replace("?", "%s")
        return sql

    def _connect(self):
        if self._pg:
            import psycopg
            from psycopg.rows import dict_row

            return psycopg.connect(self._database_url, row_factory=dict_row)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self):
        if self._pg:
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_sqlite(self):
        conn = self._connect()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS meals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    calories REAL NOT NULL,
                    protein REAL NOT NULL,
                    food_description TEXT NOT NULL,
                    created_date TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    weight REAL,
                    body_fat_percentage REAL,
                    muscle_mass REAL,
                    bmr INTEGER,
                    daily_calorie_target INTEGER DEFAULT 2500,
                    daily_protein_target INTEGER DEFAULT 300,
                    last_inbody_date TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS cheat_days (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    created_at TEXT,
                    UNIQUE(user_id, date)
                );

                CREATE TABLE IF NOT EXISTS purchase_queries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    food_name TEXT NOT NULL,
                    decision TEXT,
                    calories REAL,
                    protein REAL,
                    overall_grade TEXT,
                    raw_data TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS weekly_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    week_start TEXT NOT NULL,
                    week_end TEXT NOT NULL,
                    overall_grade TEXT,
                    protein_grade TEXT,
                    calorie_grade TEXT,
                    regularity_grade TEXT,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_meals_user_date
                    ON meals(user_id, created_date);
                CREATE INDEX IF NOT EXISTS idx_cheat_user_date
                    ON cheat_days(user_id, date);
                CREATE INDEX IF NOT EXISTS idx_purchase_user
                    ON purchase_queries(user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_weekly_user
                    ON weekly_scores(user_id, week_start);
            """)
            conn.commit()
            logger.info("SQLite 資料庫初始化完成")
        finally:
            conn.close()

    def _init_postgres(self):
        stmts = [
            """CREATE TABLE IF NOT EXISTS meals (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                calories DOUBLE PRECISION NOT NULL,
                protein DOUBLE PRECISION NOT NULL,
                food_description TEXT NOT NULL,
                created_date TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                weight DOUBLE PRECISION,
                body_fat_percentage DOUBLE PRECISION,
                muscle_mass DOUBLE PRECISION,
                bmr INTEGER,
                daily_calorie_target INTEGER DEFAULT 2500,
                daily_protein_target INTEGER DEFAULT 300,
                last_inbody_date TEXT,
                created_at TEXT,
                updated_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS cheat_days (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                created_at TEXT,
                UNIQUE(user_id, date)
            )""",
            """CREATE TABLE IF NOT EXISTS purchase_queries (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                food_name TEXT NOT NULL,
                decision TEXT,
                calories DOUBLE PRECISION,
                protein DOUBLE PRECISION,
                overall_grade TEXT,
                raw_data TEXT,
                created_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS weekly_scores (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                week_start TEXT NOT NULL,
                week_end TEXT NOT NULL,
                overall_grade TEXT,
                protein_grade TEXT,
                calorie_grade TEXT,
                regularity_grade TEXT,
                created_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, created_date)",
            "CREATE INDEX IF NOT EXISTS idx_cheat_user_date ON cheat_days(user_id, date)",
            "CREATE INDEX IF NOT EXISTS idx_purchase_user ON purchase_queries(user_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_weekly_user ON weekly_scores(user_id, week_start)",
        ]
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                for s in stmts:
                    cur.execute(s)
            conn.commit()
            logger.info("PostgreSQL 資料庫初始化完成")
        finally:
            conn.close()

    def _row_to_dict(self, row):
        if row is None:
            return None
        if isinstance(row, dict):
            return row
        return dict(row)

    # ━━━ 餐食記錄 ━━━

    def add_meal(self, user_id: str, calories: float, protein: float,
                 description: str, created_date: str):
        sql = self._adapt(
            """INSERT INTO meals (user_id, timestamp, calories, protein,
               food_description, created_date)
               VALUES (?, ?, ?, ?, ?, ?)"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(
                        sql,
                        (user_id, datetime.now().isoformat(), calories, protein,
                         description, created_date),
                    )
            else:
                conn.execute(
                    sql,
                    (user_id, datetime.now().isoformat(), calories, protein,
                     description, created_date),
                )
            conn.commit()
        finally:
            conn.close()

    def get_daily_totals(self, user_id: str, date_str: str) -> dict:
        sql = self._adapt(
            """SELECT COALESCE(SUM(calories), 0) as calories,
                      COALESCE(SUM(protein), 0) as protein,
                      COUNT(*) as meal_count
               FROM meals WHERE user_id = ? AND created_date = ?"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, date_str))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, date_str)).fetchone()
            row = self._row_to_dict(row)
            return {
                "calories": row["calories"],
                "protein": row["protein"],
                "meal_count": row["meal_count"],
            }
        finally:
            conn.close()

    def get_meals_today(self, user_id: str, date_str: str) -> list[dict]:
        sql = self._adapt(
            """SELECT calories, protein, food_description, timestamp
               FROM meals WHERE user_id = ? AND created_date = ?
               ORDER BY timestamp"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, date_str))
                    rows = cur.fetchall()
            else:
                rows = conn.execute(sql, (user_id, date_str)).fetchall()
            return [self._row_to_dict(r) for r in rows]
        finally:
            conn.close()

    def clear_today(self, user_id: str, date_str: str) -> int:
        sql = self._adapt("DELETE FROM meals WHERE user_id = ? AND created_date = ?")
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, date_str))
                    n = cur.rowcount
            else:
                cur = conn.execute(sql, (user_id, date_str))
                n = cur.rowcount
            conn.commit()
            return n
        finally:
            conn.close()

    def get_active_users_today(self, date_str: str) -> list[str]:
        sql = self._adapt(
            "SELECT DISTINCT user_id FROM meals WHERE created_date = ?"
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (date_str,))
                    rows = cur.fetchall()
            else:
                rows = conn.execute(sql, (date_str,)).fetchall()
            return [self._row_to_dict(r)["user_id"] for r in rows]
        finally:
            conn.close()

    # ━━━ 使用者檔案 ━━━

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        sql = self._adapt("SELECT * FROM user_profiles WHERE user_id = ?")
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id,))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id,)).fetchone()
            return self._row_to_dict(row) if row else None
        finally:
            conn.close()

    def upsert_user_profile(self, user_id: str, weight=None, body_fat=None,
                            muscle_mass=None, bmr=None,
                            calorie_target=None, protein_target=None):
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            sel = self._adapt("SELECT user_id FROM user_profiles WHERE user_id = ?")
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sel, (user_id,))
                    existing = cur.fetchone()
            else:
                existing = conn.execute(sel, (user_id,)).fetchone()

            if existing:
                updates = []
                values = []
                for col, val in [
                    ("weight", weight),
                    ("body_fat_percentage", body_fat),
                    ("muscle_mass", muscle_mass),
                    ("bmr", bmr),
                    ("daily_calorie_target", calorie_target),
                    ("daily_protein_target", protein_target),
                ]:
                    if val is not None:
                        updates.append(f"{col} = ?")
                        values.append(val)
                updates.append("last_inbody_date = ?")
                values.append(date.today().isoformat())
                updates.append("updated_at = ?")
                values.append(now)
                values.append(user_id)

                sql = self._adapt(
                    f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?"
                )
                if self._pg:
                    with conn.cursor() as cur:
                        cur.execute(sql, values)
                else:
                    conn.execute(sql, values)
            else:
                ins = self._adapt(
                    """INSERT INTO user_profiles
                       (user_id, weight, body_fat_percentage, muscle_mass, bmr,
                        daily_calorie_target, daily_protein_target,
                        last_inbody_date, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                )
                params = (
                    user_id, weight, body_fat, muscle_mass, bmr,
                    calorie_target or 2500, protein_target or 300,
                    date.today().isoformat(), now, now,
                )
                if self._pg:
                    with conn.cursor() as cur:
                        cur.execute(ins, params)
                else:
                    conn.execute(ins, params)
            conn.commit()
        finally:
            conn.close()

    # ━━━ 欺騙日 ━━━

    def is_cheat_day(self, user_id: str, date_str: str) -> bool:
        sql = self._adapt(
            "SELECT 1 FROM cheat_days WHERE user_id = ? AND date = ? LIMIT 1"
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, date_str))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, date_str)).fetchone()
            return row is not None
        finally:
            conn.close()

    def activate_cheat_day(self, user_id: str, date_str: str):
        now = datetime.now().isoformat()
        conn = self._connect()
        try:
            if self._pg:
                sql = (
                    "INSERT INTO cheat_days (user_id, date, created_at) VALUES (%s, %s, %s) "
                    "ON CONFLICT (user_id, date) DO NOTHING"
                )
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, date_str, now))
            else:
                conn.execute(
                    """INSERT OR IGNORE INTO cheat_days (user_id, date, created_at)
                       VALUES (?, ?, ?)""",
                    (user_id, date_str, now),
                )
            conn.commit()
        finally:
            conn.close()

    def count_cheat_days_in_range(self, user_id: str,
                                  start_date: str, end_date: str) -> int:
        sql = self._adapt(
            """SELECT COUNT(*) as cnt FROM cheat_days
               WHERE user_id = ? AND date >= ? AND date <= ?"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, start_date, end_date))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, start_date, end_date)).fetchone()
            return self._row_to_dict(row)["cnt"]
        finally:
            conn.close()

    # ━━━ 購買查詢 ━━━

    def save_purchase_decision(self, user_id: str, analysis: dict, decision: str):
        now = datetime.now().isoformat()
        sql_pq = self._adapt(
            """INSERT INTO purchase_queries
               (user_id, timestamp, food_name, decision, calories, protein,
                overall_grade, raw_data, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        )
        params_pq = (
            user_id, now,
            analysis.get("name", "未知"),
            decision,
            analysis.get("calories"),
            analysis.get("protein"),
            analysis.get("grades", {}).get("overall"),
            json.dumps(analysis, ensure_ascii=False),
            now,
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql_pq, params_pq)
                    if decision == "purchased":
                        sql_m = self._adapt(
                            """INSERT INTO meals
                               (user_id, timestamp, calories, protein,
                                food_description, created_date)
                               VALUES (?, ?, ?, ?, ?, ?)"""
                        )
                        cur.execute(
                            sql_m,
                            (
                                user_id, now,
                                analysis.get("calories", 0),
                                analysis.get("protein", 0),
                                f"[購買] {analysis.get('name', '未知')}",
                                date.today().isoformat(),
                            ),
                        )
            else:
                conn.execute(sql_pq, params_pq)
                if decision == "purchased":
                    conn.execute(
                        self._adapt(
                            """INSERT INTO meals
                               (user_id, timestamp, calories, protein,
                                food_description, created_date)
                               VALUES (?, ?, ?, ?, ?, ?)"""
                        ),
                        (
                            user_id, now,
                            analysis.get("calories", 0),
                            analysis.get("protein", 0),
                            f"[購買] {analysis.get('name', '未知')}",
                            date.today().isoformat(),
                        ),
                    )
            conn.commit()
        finally:
            conn.close()

    # ━━━ 週積分 ━━━

    def save_weekly_score(self, user_id: str, week_start: str, week_end: str,
                          overall: str, protein: str, calorie: str,
                          regularity: str):
        conn = self._connect()
        try:
            del_sql = self._adapt(
                "DELETE FROM weekly_scores WHERE user_id = ? AND week_start = ?"
            )
            ins_sql = self._adapt(
                """INSERT INTO weekly_scores
                   (user_id, week_start, week_end, overall_grade,
                    protein_grade, calorie_grade, regularity_grade, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
            )
            ins_params = (
                user_id, week_start, week_end, overall,
                protein, calorie, regularity, datetime.now().isoformat(),
            )
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(del_sql, (user_id, week_start))
                    cur.execute(ins_sql, ins_params)
            else:
                conn.execute(del_sql, (user_id, week_start))
                conn.execute(ins_sql, ins_params)
            conn.commit()
        finally:
            conn.close()
