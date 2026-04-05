#!/usr/bin/env python3
"""本機測試 Notion：讀取兩個 Database（不寫入）。請在專案根目錄建立 .env 後執行：
   python3 scripts/test_notion.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 專案根目錄
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv(ROOT / ".env.local", override=True)


def main() -> int:
    token = (os.getenv("NOTION_TOKEN") or "").strip()
    daily_id = (os.getenv("NOTION_DAILY_DB_ID") or "").strip()
    inbody_id = (os.getenv("NOTION_INBODY_DB_ID") or "").strip()
    uid = (os.getenv("NOTION_SYNC_USER_ID") or "").strip()
    daily_title = (os.getenv("NOTION_DAILY_TITLE_PROP") or "日期").strip() or "日期"
    inbody_title = (os.getenv("NOTION_INBODY_TITLE_PROP") or "檢測日期").strip() or "檢測日期"

    missing = [
        n
        for n, v in [
            ("NOTION_TOKEN", token),
            ("NOTION_DAILY_DB_ID", daily_id),
            ("NOTION_INBODY_DB_ID", inbody_id),
            ("NOTION_SYNC_USER_ID", uid),
        ]
        if not v
    ]
    if missing:
        print("缺少環境變數：", ", ".join(missing))
        print("請在專案根目錄的 .env 設定後再執行。")
        return 2

    from notion_client import Client

    c = Client(auth=token)

    daily_need = {
        daily_title,
        "熱量",
        "蛋白質",
        "碳水化合物",
        "脂肪",
        "評分",
        "餐數",
        "達標",
    }
    inbody_need = {
        inbody_title,
        "體重",
        "體脂率",
        "骨骼肌",
        "基礎代謝",
        "體重變化",
        "體脂變化",
        "評估",
    }

    for label, dbid, need in [
        ("每日飲食", daily_id, daily_need),
        ("InBody", inbody_id, inbody_need),
    ]:
        try:
            db = c.databases.retrieve(database_id=dbid)
        except Exception as e:
            print(f"FAIL [{label}] 無法讀取 database：{e}")
            return 1
        props = set((db.get("properties") or {}).keys())
        print(f"OK [{label}] 可讀取。Notion 欄位：{sorted(props)}")
        absent = sorted(need - props)
        if absent:
            print(f"  警告：程式預期但資料庫缺少欄位：{absent}")

    from notion_sync import get_notion_sync

    ns = get_notion_sync()
    print(f"NotionSync.enabled = {ns.enabled}")
    print(f"should_sync_line_user(NOTION_SYNC_USER_ID) = {ns.should_sync_line_user(uid)}")
    print("通過（僅 retrieve，未寫入任何列）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
