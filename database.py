"""
資料庫模組 - 使用 SQLite 儲存用戶飲食記錄
"""

import aiosqlite
from datetime import datetime, date
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MealRecord:
    """餐點記錄資料類別"""
    id: int
    user_id: str
    timestamp: str
    calories: float
    protein: float
    food_description: str


class Database:
    """資料庫管理類別"""
    
    def __init__(self, db_path: str = "diet_tracker.db"):
        self.db_path = db_path
    
    async def init_db(self):
        """初始化資料庫,建立表格"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS meals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    calories REAL NOT NULL,
                    protein REAL NOT NULL,
                    food_description TEXT NOT NULL,
                    created_date TEXT NOT NULL
                )
            """)
            
            # 建立索引加速查詢
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_date 
                ON meals(user_id, created_date)
            """)
            
            await db.commit()
            logger.info("資料庫初始化完成")
    
    async def add_meal(
        self,
        user_id: str,
        calories: float,
        protein: float,
        food_description: str
    ) -> int:
        """新增餐點記錄"""
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        created_date = now.strftime("%Y-%m-%d")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO meals 
                (user_id, timestamp, calories, protein, food_description, created_date)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, timestamp, calories, protein, food_description, created_date))
            
            await db.commit()
            meal_id = cursor.lastrowid
            logger.info(f"新增餐點記錄: user={user_id}, id={meal_id}")
            return meal_id
    
    async def get_today_total(self, user_id: str) -> Dict[str, float]:
        """取得今日總計"""
        today = date.today().strftime("%Y-%m-%d")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT 
                    COALESCE(SUM(calories), 0) as total_calories,
                    COALESCE(SUM(protein), 0) as total_protein,
                    COUNT(*) as meal_count
                FROM meals
                WHERE user_id = ? AND created_date = ?
            """, (user_id, today))
            
            row = await cursor.fetchone()
            
            return {
                "calories": row[0],
                "protein": row[1],
                "meal_count": row[2]
            }
    
    async def get_today_meals(self, user_id: str) -> List[MealRecord]:
        """取得今日所有餐點記錄"""
        today = date.today().strftime("%Y-%m-%d")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, user_id, timestamp, calories, protein, food_description
                FROM meals
                WHERE user_id = ? AND created_date = ?
                ORDER BY timestamp ASC
            """, (user_id, today))
            
            rows = await cursor.fetchall()
            
            return [
                MealRecord(
                    id=row[0],
                    user_id=row[1],
                    timestamp=row[2],
                    calories=row[3],
                    protein=row[4],
                    food_description=row[5]
                )
                for row in rows
            ]
    
    async def delete_today_records(self, user_id: str) -> int:
        """刪除今日所有記錄"""
        today = date.today().strftime("%Y-%m-%d")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                DELETE FROM meals
                WHERE user_id = ? AND created_date = ?
            """, (user_id, today))
            
            await db.commit()
            deleted_count = cursor.rowcount
            logger.info(f"刪除今日記錄: user={user_id}, count={deleted_count}")
            return deleted_count
    
    async def get_active_users_today(self) -> List[str]:
        """取得今日有記錄的所有用戶ID"""
        today = date.today().strftime("%Y-%m-%d")
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT DISTINCT user_id
                FROM meals
                WHERE created_date = ?
            """, (today,))
            
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_weekly_protein_by_day(self, user_id: str) -> List[Dict]:
        """取得本週每日蛋白質總和（週一為第一天）"""
        from datetime import timedelta
        today = date.today()
        # 本週一
        weekday = today.weekday()  # 0=Monday, 6=Sunday
        monday = today - timedelta(days=weekday)
        days = []
        async with aiosqlite.connect(self.db_path) as db:
            for i in range(7):
                d = monday + timedelta(days=i)
                day_str = d.strftime("%Y-%m-%d")
                cursor = await db.execute("""
                    SELECT COALESCE(SUM(protein), 0)
                    FROM meals
                    WHERE user_id = ? AND created_date = ?
                """, (user_id, day_str))
                row = await cursor.fetchone()
                days.append({
                    "date": day_str,
                    "weekday": ["一", "二", "三", "四", "五", "六", "日"][i],
                    "protein": row[0] if row else 0.0,
                })
        return days
