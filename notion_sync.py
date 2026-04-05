"""
Notion 雙 Database 同步（每日飲食、InBody）。需設定 NOTION_TOKEN 與兩個 database id。
單人使用請設 NOTION_SYNC_USER_ID 為你的 LINE userId，避免多使用者寫入同一列「日期」衝突。
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _nv(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class NotionSync:
    """Notion 雙 Database 同步（同步 API，請由 asyncio.to_thread 呼叫）。"""

    def __init__(self):
        token = (os.getenv("NOTION_TOKEN") or "").strip()
        self.daily_db_id = (os.getenv("NOTION_DAILY_DB_ID") or "").strip()
        self.inbody_db_id = (os.getenv("NOTION_INBODY_DB_ID") or "").strip()
        self.sync_user_id = (os.getenv("NOTION_SYNC_USER_ID") or "").strip()
        self._client = None
        if token and self.daily_db_id and self.inbody_db_id:
            try:
                from notion_client import Client

                self._client = Client(auth=token)
            except Exception as e:
                logger.error("Notion Client 初始化失敗：%s", e)
        elif token or self.daily_db_id or self.inbody_db_id:
            logger.warning(
                "Notion 同步已停用：請同時設定 NOTION_TOKEN、NOTION_DAILY_DB_ID、NOTION_INBODY_DB_ID"
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def should_sync_line_user(self, line_user_id: str) -> bool:
        if not self.enabled:
            return False
        if not self.sync_user_id:
            logger.debug("未設定 NOTION_SYNC_USER_ID，略過 Notion 同步")
            return False
        return self.sync_user_id == line_user_id

    def sync_daily_nutrition(
        self,
        date_str: str,
        data: dict,
        protein_target: float,
    ) -> bool:
        """
        data: calories, protein, meal_count；可選 carbs, fat（本專案餐點未存碳水／脂肪時傳 0）。
        """
        if not self.enabled:
            return False
        try:
            cal = float(data.get("calories", 0))
            pro = float(data.get("protein", 0))
            meals = int(data.get("meal_count", 0))
            carbs = float(data.get("carbs", 0) or 0)
            fat = float(data.get("fat", 0) or 0)

            grade = _grade_from_protein(pro, protein_target)
            achieved = pro >= protein_target * 0.9 if protein_target > 0 else False

            properties = {
                "日期": {"title": [{"text": {"content": date_str}}]},
                "熱量": {"number": round(cal, 1)},
                "蛋白質": {"number": round(pro, 1)},
                "碳水化合物": {"number": round(carbs, 1)},
                "脂肪": {"number": round(fat, 1)},
                "評分": {"select": {"name": grade}},
                "餐數": {"number": meals},
                "達標": {"checkbox": achieved},
            }

            existing = self._find_daily_record(date_str)
            if existing:
                self._client.pages.update(page_id=existing["id"], properties=properties)
                logger.info("Notion 已更新每日記錄：%s", date_str)
            else:
                self._client.pages.create(
                    parent={"database_id": self.daily_db_id},
                    properties=properties,
                )
                logger.info("Notion 已建立每日記錄：%s", date_str)
            return True
        except Exception as e:
            logger.error("Notion 同步每日記錄失敗：%s", e, exc_info=True)
            return False

    def sync_inbody(self, inbody_data: dict) -> bool:
        if not self.enabled:
            return False
        try:
            w = _nv(inbody_data.get("weight"))
            bf = _nv(inbody_data.get("body_fat_percentage"))
            muscle = _nv(inbody_data.get("muscle_mass"))
            bmr = inbody_data.get("bmr")
            if w is None or bf is None:
                logger.warning("Notion InBody 同步略過：缺少體重或體脂率")
                return False

            test_date = inbody_data.get("test_date")
            if not test_date or test_date == "null":
                from datetime import date

                test_date = date.today().isoformat()

            previous = self._get_previous_inbody()
            weight_change = 0.0
            bf_change = 0.0
            if previous and previous.get("weight") is not None and previous.get("bf") is not None:
                weight_change = w - float(previous["weight"])
                bf_change = bf - float(previous["bf"])

            grade = _grade_inbody(weight_change, bf_change)

            bmr_num = int(bmr) if bmr is not None else None
            muscle_num = round(float(muscle), 1) if muscle is not None else None

            properties: dict = {
                "檢測日期": {"title": [{"text": {"content": str(test_date)}}]},
                "體重": {"number": round(w, 1)},
                "體脂率": {"number": round(bf, 1)},
                "骨骼肌": {"number": muscle_num},
                "基礎代謝": {"number": bmr_num},
                "體重變化": {"number": round(weight_change, 1)},
                "體脂變化": {"number": round(bf_change, 1)},
                "評估": {"select": {"name": grade}},
            }

            self._client.pages.create(
                parent={"database_id": self.inbody_db_id},
                properties=properties,
            )
            logger.info("Notion 已建立 InBody 記錄：%s", test_date)
            return True
        except Exception as e:
            logger.error("Notion 同步 InBody 失敗：%s", e, exc_info=True)
            return False

    def _find_daily_record(self, date_str: str) -> dict | None:
        try:
            results = self._client.databases.query(
                database_id=self.daily_db_id,
                filter={
                    "property": "日期",
                    "title": {"equals": date_str},
                },
            )
            rows = results.get("results") or []
            return rows[0] if rows else None
        except Exception as e:
            logger.error("Notion 查詢每日記錄失敗：%s", e)
            return None

    def _get_previous_inbody(self) -> dict | None:
        try:
            results = self._client.databases.query(
                database_id=self.inbody_db_id,
                sorts=[{"property": "檢測日期", "direction": "descending"}],
                page_size=1,
            )
            rows = results.get("results") or []
            if not rows:
                return None
            props = rows[0].get("properties") or {}
            w = _notion_number(props.get("體重"))
            bf = _notion_number(props.get("體脂率"))
            return {"weight": w, "bf": bf}
        except Exception as e:
            logger.error("Notion 讀取上一筆 InBody 失敗：%s", e)
            return None


def _notion_number(prop: dict | None) -> float | None:
    if not prop or prop.get("type") != "number":
        return None
    n = prop.get("number")
    return float(n) if n is not None else None


def _grade_from_protein(protein: float, target: float) -> str:
    if target <= 0:
        target = 300.0
    r = protein / target
    if r >= 1.0:
        return "ㄅ"
    if r >= 0.83:
        return "ㄆ"
    if r >= 0.67:
        return "ㄇ"
    if r >= 0.5:
        return "ㄈ"
    return "ㄉ"


def _grade_inbody(weight_change: float, bf_change: float) -> str:
    if bf_change <= -2 and weight_change <= 0:
        return "ㄅ"
    if bf_change <= -1:
        return "ㄆ"
    if bf_change <= 0.5 and weight_change <= 1:
        return "ㄇ"
    if bf_change <= 1.5:
        return "ㄈ"
    return "ㄉ"


_notion: NotionSync | None = None


def get_notion_sync() -> NotionSync:
    global _notion
    if _notion is None:
        _notion = NotionSync()
    return _notion
