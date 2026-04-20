"""
資料庫模組：支援 SQLite（本機，未設 DATABASE_URL）與 PostgreSQL（Supabase 等）。
"""

import ipaddress
import json
import os
import re
import socket
import sqlite3
from datetime import date, datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = "diet_tracker.db"


def _should_force_ipv4_for_postgres() -> bool:
    """Render 等環境常無法連 Supabase 的 IPv6；設 DATABASE_FORCE_IPV4=0 可關閉。"""
    v = os.getenv("DATABASE_FORCE_IPV4", "").strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return os.getenv("RENDER", "").strip().lower() in ("true", "1", "yes")


def _ipv4_lookup(host: str) -> list:
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return []


def _conninfo_add_ipv4_hostaddr(params: dict) -> str | None:
    """有 IPv4 A 記錄時加 hostaddr；成功回傳 conninfo 字串，否則 None。"""
    from psycopg.conninfo import make_conninfo

    host = (params.get("host") or "").strip()
    if not host:
        return None
    infos = _ipv4_lookup(host)
    if not infos:
        return None
    merged = dict(params)
    merged["hostaddr"] = infos[0][4][0]
    try:
        return make_conninfo(**merged)
    except Exception as e:
        logger.warning("make_conninfo 失敗: %s", e)
        return None


def _supabase_direct_to_session_pooler(params: dict, ref: str) -> str | None:
    """
    db.<ref>.supabase.co（Direct）常僅有 IPv6。改連 Supavisor「Session pooler」（IPv4）：
    使用者 postgres.<ref>、主機 aws-0-<region>.pooler.supabase.com、port 5432。
    （勿把 6543 配在 pooler 主機上；6543 是 db.xxx.supabase.co 的 Transaction 模式，使用者為 postgres。）
    見: https://supabase.com/docs/guides/database/connecting-to-postgres
    """
    from psycopg.conninfo import make_conninfo

    region = (os.getenv("SUPABASE_REGION") or "").strip().lower().replace("_", "-")
    if not region:
        return None
    custom = (os.getenv("SUPABASE_POOLER_HOST") or "").strip()
    pooler_host = custom if custom else f"aws-0-{region}.pooler.supabase.com"
    merged = dict(params)
    merged["host"] = pooler_host
    merged["port"] = 5432
    merged["user"] = f"postgres.{ref}"
    merged.pop("hostaddr", None)
    try:
        return make_conninfo(**merged)
    except Exception as e:
        logger.warning("Pooler make_conninfo 失敗: %s", e)
        return None


def _supabase_pooler_aws1_fallback_conninfo(failed_conninfo: str) -> str | None:
    """部分專案掛在 aws-1-* pooler；Tenant not found 時改試 aws-1。"""
    if "aws-0-" not in failed_conninfo or ".pooler.supabase.com" not in failed_conninfo:
        return None
    alt = failed_conninfo.replace("aws-0-", "aws-1-", 1)
    return alt if alt != failed_conninfo else None


def _resolve_postgres_conninfo(uri: str) -> str:
    if not _should_force_ipv4_for_postgres():
        return uri

    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    try:
        params = conninfo_to_dict(uri)
    except Exception as e:
        logger.warning("DATABASE_URL 解析失敗: %s", e)
        return uri

    host = (params.get("host") or "").strip()
    if not host:
        return uri

    try:
        ipaddress.ip_address(host)
        return uri
    except ValueError:
        pass

    if params.get("hostaddr"):
        try:
            return make_conninfo(**dict(params))
        except Exception:
            return uri

    # Supabase Direct：常無 IPv4，只能走 pooler 或加 hostaddr（若有 A 記錄）
    m = re.fullmatch(r"db\.([a-z0-9]+)\.supabase\.co", host, re.I)
    if m:
        ref = m.group(1)
        ci = _conninfo_add_ipv4_hostaddr(params)
        if ci:
            logger.info("PostgreSQL 已套用 IPv4 hostaddr（%s）", host)
            return ci
        pool = _supabase_direct_to_session_pooler(params, ref)
        if pool:
            logger.warning(
                "Supabase Direct（%s）無 IPv4，已自動改用 Session pooler（5432，使用者 postgres.%s）。",
                host,
                ref,
            )
            return pool
        logger.error(
            "資料庫 %s 沒有 IPv4（Render 無法使用 IPv6）。"
            "請在 Render 設定 SUPABASE_REGION=你的區域（例 ap-south-1），"
            "或到 Supabase Connect 複製「Session pooler」URI 設為 DATABASE_URL。",
            host,
        )
        return uri

    # 其他主機：僅嘗試 hostaddr
    ci = _conninfo_add_ipv4_hostaddr(params)
    if ci:
        logger.info("PostgreSQL 已套用 IPv4 hostaddr")
        return ci
    logger.warning("無法解析 %s 的 IPv4，沿用原始 DATABASE_URL", host)
    return uri


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
            from psycopg import OperationalError
            from psycopg.rows import dict_row

            timeout = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "15"))
            conninfo = _resolve_postgres_conninfo(self._database_url)
            try:
                return psycopg.connect(
                    conninfo,
                    row_factory=dict_row,
                    connect_timeout=timeout,
                )
            except OperationalError as e:
                err = str(e)
                if "Tenant or user not found" in err:
                    alt = _supabase_pooler_aws1_fallback_conninfo(conninfo)
                    if alt:
                        logger.warning(
                            "Pooler 回傳 Tenant not found，改試 aws-1 pooler 主機"
                        )
                        return psycopg.connect(
                            alt,
                            row_factory=dict_row,
                            connect_timeout=timeout,
                        )
                raise
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
                    updated_at TEXT,
                    custom_quick_items TEXT
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

                CREATE TABLE IF NOT EXISTS user_message_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    at_utc TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_user_msg_log_user_at
                    ON user_message_log(user_id, at_utc);

                CREATE TABLE IF NOT EXISTS reminder_sent (
                    user_id TEXT NOT NULL,
                    local_date TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, local_date, slot)
                );
            """)
            conn.commit()
            logger.info("SQLite 資料庫初始化完成")
        finally:
            conn.close()
        self._migrate_user_profiles_quick_items()
        self._migrate_user_message_log_kind()

    def _migrate_user_message_log_kind(self):
        """補上 user_message_log.message_kind（text／image），供用餐提醒判斷是否已傳照片。"""
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE user_message_log "
                        "ADD COLUMN IF NOT EXISTS message_kind TEXT DEFAULT 'text'"
                    )
                    cur.execute(
                        "UPDATE user_message_log SET message_kind = 'text' "
                        "WHERE message_kind IS NULL"
                    )
                conn.commit()
            else:
                try:
                    conn.execute(
                        "ALTER TABLE user_message_log ADD COLUMN message_kind TEXT DEFAULT 'text'"
                    )
                    conn.commit()
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
                conn.execute(
                    "UPDATE user_message_log SET message_kind = 'text' "
                    "WHERE message_kind IS NULL"
                )
                conn.commit()
        finally:
            conn.close()

    def _migrate_user_profiles_quick_items(self):
        """舊資料庫補上 user_profiles.custom_quick_items。"""
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(
                        "ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS custom_quick_items TEXT"
                    )
                conn.commit()
            else:
                try:
                    conn.execute(
                        "ALTER TABLE user_profiles ADD COLUMN custom_quick_items TEXT"
                    )
                    conn.commit()
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
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
                updated_at TEXT,
                custom_quick_items TEXT
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
            """CREATE TABLE IF NOT EXISTS user_message_log (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                at_utc TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_user_msg_log_user_at ON user_message_log(user_id, at_utc)",
            """CREATE TABLE IF NOT EXISTS reminder_sent (
                user_id TEXT NOT NULL,
                local_date TEXT NOT NULL,
                slot TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, local_date, slot)
            )""",
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
        self._migrate_user_profiles_quick_items()
        self._migrate_user_message_log_kind()

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

    def update_custom_quick_items(self, user_id: str, items_json: str):
        """儲存使用者自訂快速記錄品項（JSON 字串）。"""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            sel = self._adapt("SELECT user_id FROM user_profiles WHERE user_id = ?")
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sel, (user_id,))
                    existing = cur.fetchone()
                    if existing:
                        upd = self._adapt(
                            """UPDATE user_profiles
                               SET custom_quick_items = ?, updated_at = ?
                               WHERE user_id = ?"""
                        )
                        cur.execute(upd, (items_json, now, user_id))
                    else:
                        ins = self._adapt(
                            """INSERT INTO user_profiles
                               (user_id, custom_quick_items, daily_calorie_target,
                                daily_protein_target, created_at, updated_at)
                               VALUES (?, ?, ?, ?, ?, ?)"""
                        )
                        cur.execute(
                            ins,
                            (
                                user_id,
                                items_json,
                                2500,
                                300,
                                now,
                                now,
                            ),
                        )
            else:
                existing = conn.execute(sel, (user_id,)).fetchone()
                if existing:
                    conn.execute(
                        self._adapt(
                            """UPDATE user_profiles
                               SET custom_quick_items = ?, updated_at = ?
                               WHERE user_id = ?"""
                        ),
                        (items_json, now, user_id),
                    )
                else:
                    conn.execute(
                        self._adapt(
                            """INSERT INTO user_profiles
                               (user_id, custom_quick_items, daily_calorie_target,
                                daily_protein_target, created_at, updated_at)
                               VALUES (?, ?, ?, ?, ?, ?)"""
                        ),
                        (user_id, items_json, 2500, 300, now, now),
                    )
            conn.commit()
        finally:
            conn.close()

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
        def _num(v, default: float = 0.0) -> float:
            if isinstance(v, (int, float)):
                return float(v)
            if v is None:
                return default
            s = str(v).strip().replace(",", "")
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            if not m:
                return default
            try:
                return float(m.group(0))
            except ValueError:
                return default

        if not isinstance(analysis, dict):
            analysis = {}
        grades = analysis.get("grades")
        if not isinstance(grades, dict):
            grades = {}

        calories = _num(analysis.get("calories"), 0.0)
        protein = _num(analysis.get("protein"), 0.0)
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
            calories,
            protein,
            grades.get("overall"),
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
                                calories,
                                protein,
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
                            calories,
                            protein,
                            f"[購買] {analysis.get('name', '未知')}",
                            date.today().isoformat(),
                        ),
                    )
            conn.commit()
        finally:
            conn.close()

    # ━━━ 用餐提醒（LINE 訊息時間軸）━━━

    def log_line_message(
        self,
        user_id: str,
        at_utc: str | None = None,
        message_kind: str = "text",
    ):
        """記錄使用者曾傳入訊息。message_kind: text／image（供用餐提醒判斷是否已傳照片）。"""
        ts = at_utc or datetime.now(timezone.utc).isoformat()
        kind = (message_kind or "text").strip().lower()
        if kind not in ("text", "image", "other"):
            kind = "other"
        sql = self._adapt(
            "INSERT INTO user_message_log (user_id, at_utc, message_kind) VALUES (?, ?, ?)"
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, ts, kind))
            else:
                conn.execute(sql, (user_id, ts, kind))
            conn.commit()
        finally:
            conn.close()

    def user_had_message_in_utc_range(
        self, user_id: str, start_iso: str, end_iso: str
    ) -> bool:
        """是否有訊息記錄落在 [start_iso, end_iso)（ISO 字串，建議 UTC）。"""
        sql = self._adapt(
            """SELECT 1 FROM user_message_log
               WHERE user_id = ? AND at_utc >= ? AND at_utc < ?
               LIMIT 1"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, start_iso, end_iso))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, start_iso, end_iso)).fetchone()
            return row is not None
        finally:
            conn.close()

    def user_had_meal_logged_in_utc_window(
        self, user_id: str, created_date: str, start_iso: str, end_iso: str
    ) -> bool:
        """該日已入帳的餐點中，是否有任一筆的 timestamp 落在 [start_iso, end_iso)（UTC ISO）。"""
        sql = self._adapt(
            """SELECT 1 FROM meals
               WHERE user_id = ? AND created_date = ?
               AND timestamp >= ? AND timestamp < ?
               LIMIT 1"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, created_date, start_iso, end_iso))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, created_date, start_iso, end_iso)).fetchone()
            return row is not None
        finally:
            conn.close()

    def user_had_photo_in_utc_range(
        self, user_id: str, start_iso: str, end_iso: str
    ) -> bool:
        """該時段內是否曾傳送圖片訊息（僅 message_kind=image）。"""
        sql = self._adapt(
            """SELECT 1 FROM user_message_log
               WHERE user_id = ? AND at_utc >= ? AND at_utc < ?
               AND COALESCE(message_kind, 'text') = 'image'
               LIMIT 1"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, start_iso, end_iso))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, start_iso, end_iso)).fetchone()
            return row is not None
        finally:
            conn.close()

    def get_user_ids_for_meal_reminders(self, meal_since_date: str) -> list[str]:
        """曾傳過訊息，或近期有餐點紀錄的使用者（去重）。"""
        conn = self._connect()
        try:
            if self._pg:
                sql = """
                    SELECT DISTINCT user_id FROM (
                        SELECT user_id FROM user_message_log
                        UNION
                        SELECT DISTINCT user_id FROM meals WHERE created_date >= %s
                    ) AS u
                """
                with conn.cursor() as cur:
                    cur.execute(sql, (meal_since_date,))
                    rows = cur.fetchall()
            else:
                sql = """
                    SELECT DISTINCT user_id FROM (
                        SELECT user_id FROM user_message_log
                        UNION
                        SELECT user_id FROM meals WHERE created_date >= ?
                    ) AS u
                """
                rows = conn.execute(sql, (meal_since_date,)).fetchall()
            return [self._row_to_dict(r)["user_id"] for r in rows]
        finally:
            conn.close()

    def get_user_ids_for_daily_summary(self, meal_since_date: str) -> list[str]:
        """每日總結推播對象：近期曾互動或有餐點紀錄者（與用餐提醒同一池，避免只吃到『當天有紀錄』）。"""
        return self.get_user_ids_for_meal_reminders(meal_since_date)

    def reminder_already_sent(self, user_id: str, local_date: str, slot: str) -> bool:
        sql = self._adapt(
            """SELECT 1 FROM reminder_sent
               WHERE user_id = ? AND local_date = ? AND slot = ? LIMIT 1"""
        )
        conn = self._connect()
        try:
            if self._pg:
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, local_date, slot))
                    row = cur.fetchone()
            else:
                row = conn.execute(sql, (user_id, local_date, slot)).fetchone()
            return row is not None
        finally:
            conn.close()

    def mark_reminder_sent(self, user_id: str, local_date: str, slot: str):
        now = datetime.now(timezone.utc).isoformat()
        conn = self._connect()
        try:
            if self._pg:
                sql = (
                    "INSERT INTO reminder_sent (user_id, local_date, slot, created_at) "
                    "VALUES (%s, %s, %s, %s) ON CONFLICT (user_id, local_date, slot) DO NOTHING"
                )
                with conn.cursor() as cur:
                    cur.execute(sql, (user_id, local_date, slot, now))
            else:
                conn.execute(
                    """INSERT OR IGNORE INTO reminder_sent
                       (user_id, local_date, slot, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (user_id, local_date, slot, now),
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
