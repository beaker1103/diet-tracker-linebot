"""
資料庫模組 - SQLite 操作封裝
支援: 餐食記錄、使用者檔案、欺騙日、購買查詢、週積分
"""

import sqlite3
import json
from datetime import date, datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = "diet_tracker.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def init(self):
        """建立所有資料表。"""
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
            logger.info("資料庫初始化完成")
        finally:
            conn.close()

    # ━━━ 餐食記錄 ━━━

    def add_meal(self, user_id: str, calories: float, protein: float,
                 description: str, created_date: str):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO meals (user_id, timestamp, calories, protein,
                   food_description, created_date)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, datetime.now().isoformat(), calories, protein,
                 description, created_date),
            )
            conn.commit()
        finally:
            conn.close()

    def get_daily_totals(self, user_id: str, date_str: str) -> dict:
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT COALESCE(SUM(calories), 0) as calories,
                          COALESCE(SUM(protein), 0) as protein,
                          COUNT(*) as meal_count
                   FROM meals WHERE user_id = ? AND created_date = ?""",
                (user_id, date_str),
            ).fetchone()
            return {
                "calories": row["calories"],
                "protein": row["protein"],
                "meal_count": row["meal_count"],
            }
        finally:
            conn.close()

    def get_meals_today(self, user_id: str, date_str: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT calories, protein, food_description, timestamp
                   FROM meals WHERE user_id = ? AND created_date = ?
                   ORDER BY timestamp""",
                (user_id, date_str),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def clear_today(self, user_id: str, date_str: str) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "DELETE FROM meals WHERE user_id = ? AND created_date = ?",
                (user_id, date_str),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_active_users_today(self, date_str: str) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM meals WHERE created_date = ?",
                (date_str,),
            ).fetchall()
            return [r["user_id"] for r in rows]
        finally:
            conn.close()

    # ━━━ 使用者檔案 ━━━

    def get_user_profile(self, user_id: str) -> Optional[dict]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def upsert_user_profile(self, user_id: str, weight=None, body_fat=None,
                            muscle_mass=None, bmr=None,
                            calorie_target=None, protein_target=None):
        conn = self._connect()
        now = datetime.now().isoformat()
        try:
            existing = conn.execute(
                "SELECT user_id FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()

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

                conn.execute(
                    f"UPDATE user_profiles SET {', '.join(updates)} WHERE user_id = ?",
                    values,
                )
            else:
                conn.execute(
                    """INSERT INTO user_profiles
                       (user_id, weight, body_fat_percentage, muscle_mass, bmr,
                        daily_calorie_target, daily_protein_target,
                        last_inbody_date, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (user_id, weight, body_fat, muscle_mass, bmr,
                     calorie_target or 2500, protein_target or 300,
                     date.today().isoformat(), now, now),
                )
            conn.commit()
        finally:
            conn.close()

    # ━━━ 欺騙日 ━━━

    def is_cheat_day(self, user_id: str, date_str: str) -> bool:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM cheat_days WHERE user_id = ? AND date = ?",
                (user_id, date_str),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def activate_cheat_day(self, user_id: str, date_str: str):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO cheat_days (user_id, date, created_at)
                   VALUES (?, ?, ?)""",
                (user_id, date_str, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def count_cheat_days_in_range(self, user_id: str,
                                  start_date: str, end_date: str) -> int:
        conn = self._connect()
        try:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM cheat_days
                   WHERE user_id = ? AND date >= ? AND date <= ?""",
                (user_id, start_date, end_date),
            ).fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # ━━━ 購買查詢 ━━━

    def save_purchase_decision(self, user_id: str, analysis: dict, decision: str):
        conn = self._connect()
        now = datetime.now().isoformat()
        try:
            conn.execute(
                """INSERT INTO purchase_queries
                   (user_id, timestamp, food_name, decision, calories, protein,
                    overall_grade, raw_data, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, now,
                    analysis.get("name", "未知"),
                    decision,
                    analysis.get("calories"),
                    analysis.get("protein"),
                    analysis.get("grades", {}).get("overall"),
                    json.dumps(analysis, ensure_ascii=False),
                    now,
                ),
            )

            # 如果確認購買,同時記錄到 meals
            if decision == "purchased":
                conn.execute(
                    """INSERT INTO meals
                       (user_id, timestamp, calories, protein,
                        food_description, created_date)
                       VALUES (?, ?, ?, ?, ?, ?)""",
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
            # 避免重複: 同使用者同週只保留最新
            conn.execute(
                """DELETE FROM weekly_scores
                   WHERE user_id = ? AND week_start = ?""",
                (user_id, week_start),
            )
            conn.execute(
                """INSERT INTO weekly_scores
                   (user_id, week_start, week_end, overall_grade,
                    protein_grade, calorie_grade, regularity_grade, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, week_start, week_end, overall,
                 protein, calorie, regularity, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
