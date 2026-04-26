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
        # 資料庫第一欄必為 Title；Notion 常顯示為 Name，可重新命名或在這裡指定 API 欄位名
        self.daily_title_prop = (os.getenv("NOTION_DAILY_TITLE_PROP") or "日期").strip() or "日期"
        self.inbody_title_prop = (os.getenv("NOTION_INBODY_TITLE_PROP") or "檢測日期").strip() or "檢測日期"
        self._client = None
        self._db_props: dict[str, dict[str, dict]] = {}
        self._missing_prop_warned: set[tuple[str, str]] = set()
        self._sync_user_missing_warned = False
        if token and self.daily_db_id and self.inbody_db_id:
            try:
                from notion_client import Client

                self._client = Client(auth=token)
                self._db_props[self.daily_db_id] = self._load_db_properties(self.daily_db_id, "每日飲食")
                self._db_props[self.inbody_db_id] = self._load_db_properties(self.inbody_db_id, "InBody")
                self.daily_title_prop = self._resolve_title_prop(
                    self.daily_db_id, self.daily_title_prop, "每日飲食"
                )
                self.inbody_title_prop = self._resolve_title_prop(
                    self.inbody_db_id, self.inbody_title_prop, "InBody"
                )
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
            if not self._sync_user_missing_warned:
                logger.warning("未設定 NOTION_SYNC_USER_ID，將允許所有使用者同步 Notion")
                self._sync_user_missing_warned = True
            return True
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
                self.daily_title_prop: {"title": [{"text": {"content": date_str}}]},
                "熱量": {"number": round(cal, 1)},
                "蛋白質": {"number": round(pro, 1)},
                "碳水化合物": {"number": round(carbs, 1)},
                "脂肪": {"number": round(fat, 1)},
                "評分": {"select": {"name": grade}},
                "餐數": {"number": meals},
                "達標": {"checkbox": achieved},
            }
            properties = self._filter_supported_properties(
                self.daily_db_id, properties, required=[self.daily_title_prop], db_label="每日飲食"
            )
            if self.daily_title_prop not in properties:
                logger.error("Notion 每日同步失敗：找不到可用的 Title 欄位")
                return False

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
                self.inbody_title_prop: {"title": [{"text": {"content": str(test_date)}}]},
                "體重": {"number": round(w, 1)},
                "體脂率": {"number": round(bf, 1)},
                "骨骼肌": {"number": muscle_num},
                "基礎代謝": {"number": bmr_num},
                "體重變化": {"number": round(weight_change, 1)},
                "體脂變化": {"number": round(bf_change, 1)},
                "評估": {"select": {"name": grade}},
            }
            properties = self._filter_supported_properties(
                self.inbody_db_id, properties, required=[self.inbody_title_prop], db_label="InBody"
            )
            if self.inbody_title_prop not in properties:
                logger.error("Notion InBody 同步失敗：找不到可用的 Title 欄位")
                return False

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
                    "property": self.daily_title_prop,
                    "title": {"equals": date_str},
                },
            )
            rows = results.get("results") or []
            return rows[0] if rows else None
        except Exception as e:
            logger.error("Notion 查詢每日記錄失敗：%s", e)
            return None

    def _load_db_properties(self, db_id: str, db_label: str) -> dict[str, dict]:
        try:
            db = self._client.databases.retrieve(database_id=db_id)
            props = db.get("properties") or {}
            if not isinstance(props, dict):
                return {}
            return props
        except Exception as e:
            logger.error("Notion 讀取 %s 資料庫欄位失敗：%s", db_label, e)
            return {}

    def _resolve_title_prop(self, db_id: str, preferred: str, db_label: str) -> str:
        props = self._db_props.get(db_id) or {}
        if preferred in props:
            return preferred
        for name, meta in props.items():
            if isinstance(meta, dict) and meta.get("type") == "title":
                logger.warning(
                    "Notion %s Title 欄位「%s」不存在，改用「%s」",
                    db_label,
                    preferred,
                    name,
                )
                return name
        return preferred

    def _filter_supported_properties(
        self,
        db_id: str,
        properties: dict[str, dict],
        required: list[str],
        db_label: str,
    ) -> dict[str, dict]:
        known = self._db_props.get(db_id) or {}
        if not known:
            return properties
        result: dict[str, dict] = {}
        for k, v in properties.items():
            if k in known:
                result[k] = v
            else:
                key = (db_id, k)
                if key not in self._missing_prop_warned:
                    logger.warning("Notion %s 缺少欄位「%s」，此欄位將略過同步", db_label, k)
                    self._missing_prop_warned.add(key)
        for req in required:
            if req in properties and req not in result:
                # required 欄位不存在時，讓上層流程能察覺並回報
                continue
        return result

    def _get_previous_inbody(self) -> dict | None:
        try:
            results = self._client.databases.query(
                database_id=self.inbody_db_id,
                sorts=[{"property": self.inbody_title_prop, "direction": "descending"}],
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
