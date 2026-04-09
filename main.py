"""
減重監測 LINE Bot - 完整主程式
FastAPI + LINE Messaging API + OpenAI GPT Vision
"""

import os
import re
import json
import base64
import hmac
import logging
import asyncio
from io import BytesIO
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    AsyncApiClient,
    AsyncMessagingApi,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent, ImageMessageContent

from database import Database
from notion_sync import get_notion_sync

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_dotenv()

LINE_CHANNEL_SECRET = (os.getenv("LINE_CHANNEL_SECRET") or "").strip()
LINE_CHANNEL_ACCESS_TOKEN = (os.getenv("LINE_CHANNEL_ACCESS_TOKEN") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o").strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 常數與評分系統
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 注音評分 (僅用於食物查詢)
FOOD_GRADES = ["ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ"]

# 通用評分 (用於其他所有功能)
GENERAL_GRADES = ["S", "A", "B", "C", "D", "E"]

DEFAULT_PROTEIN_TARGET = 220  # g（無 InBody 時兜底；有體重時由 calculate_targets 重算）
DEFAULT_CALORIE_TARGET = 2500  # kcal

# 快速記錄（不呼叫 Vision，省 token）；可經「設定蛋白飲」自訂蛋白飲數值
DEFAULT_QUICK_ITEMS = {
    "蛋白飲": {"calories": 130, "protein": 25, "description": "乳清蛋白飲 一份"},
    "雞蛋": {"calories": 75, "protein": 6, "description": "水煮蛋 一顆"},
    "雞胸肉": {"calories": 165, "protein": 31, "description": "雞胸肉 100g"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 使用者狀態管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserState:
    IDLE = "idle"
    WAITING_PURCHASE_PHOTO = "waiting_purchase_photo"
    PURCHASE_REVIEWED = "purchase_reviewed"
    WAITING_INBODY_PHOTO = "waiting_inbody_photo"


# 記憶體內狀態 (生產環境可改用 Redis)
_user_states: dict[str, str] = {}
_user_contexts: dict[str, dict] = {}


def set_state(user_id: str, state: str, context: dict | None = None):
    _user_states[user_id] = state
    if context:
        _user_contexts[user_id] = context


def get_state(user_id: str) -> str:
    return _user_states.get(user_id, UserState.IDLE)


def get_context(user_id: str) -> dict:
    return _user_contexts.get(user_id, {})


def clear_state(user_id: str):
    _user_states.pop(user_id, None)
    _user_contexts.pop(user_id, None)


def leave_photo_wait_if_any(user_id: str) -> None:
    """執行其他功能時離開「等待傳照」狀態，避免下一張照片被誤判流程。"""
    s = get_state(user_id)
    if s in (UserState.WAITING_PURCHASE_PHOTO, UserState.WAITING_INBODY_PHOTO):
        clear_state(user_id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI Vision 呼叫
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _openai_extract_message_text(data: dict) -> str:
    """從 chat/completions JSON 取出 assistant 文字（相容 content 為 str、list 或 null）。"""
    try:
        choices = data.get("choices")
        if not choices:
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"]).strip()
            return ""
        msg = choices[0].get("message") or {}
        raw = msg.get("content")
        if raw is None:
            r = msg.get("refusal")
            return str(r).strip() if r else ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, list):
            parts: list[str] = []
            for item in raw:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    elif "text" in item:
                        parts.append(str(item["text"]))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts).strip()
        return str(raw).strip()
    except (IndexError, KeyError, TypeError) as e:
        logger.warning("OpenAI 回應結構無法解析: %s", e)
        return ""


def _try_parse_json_response(raw: str) -> dict | None:
    """從模型回傳文字解析 JSON 物件（含 markdown 包裹或前後贅字時盡力擷取）。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


class OpenAIUserNotice(str):
    """標記為應直接顯示給使用者的 OpenAI 連線／帳務錯誤說明（非模型回傳文字）。"""


def _openai_error_user_message(status: int, body: str) -> str | None:
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError:
        return None
    err = data.get("error")
    if not isinstance(err, dict):
        return None
    code = str(err.get("code") or "")
    typ = str(err.get("type") or "")
    if typ == "insufficient_quota" or code == "insufficient_quota":
        return (
            "目前無法使用 AI 分析，OpenAI 帳戶額度或方案已不足。"
            "請至 OpenAI 平台檢查方案與帳單，充值或升級後再試。"
        )
    if status == 401:
        return "無法連線 AI 服務，API 金鑰無效或已撤銷。請聯絡管理員檢查設定。"
    if status == 429:
        return "目前請求過於頻繁或服務忙碌，請稍後再試。"
    if status >= 500:
        return "AI 服務暫時無法使用，請稍後再試。"
    return None


async def call_openai_vision(
    system_prompt: str,
    user_prompt: str,
    image_url: str | None = None,
    image_base64: str | None = None,
    image_detail: str = "auto",
    response_json_object: bool = True,
) -> dict | str:
    """呼叫 OpenAI Vision API，回傳解析後的 JSON 或原始文字。"""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY 未設定")
        return OpenAIUserNotice("AI 服務未設定，請聯絡管理員檢查環境變數。")

    # 組裝 user content
    user_content = []
    if image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url, "detail": image_detail},
        })
    elif image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": image_detail},
        })
    user_content.append({"type": "text", "text": user_prompt})

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    if response_json_object:
        payload["response_format"] = {"type": "json_object"}

    timeout = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            logger.error(
                "OpenAI Vision HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:800],
            )
            um = _openai_error_user_message(resp.status_code, resp.text or "")
            if um:
                return OpenAIUserNotice(um)
        resp.raise_for_status()

    raw = _openai_extract_message_text(resp.json())
    if not raw:
        return ""

    parsed = _try_parse_json_response(raw)
    if parsed is not None:
        return parsed
    return raw


async def call_openai_text(system_prompt: str, user_prompt: str) -> dict | str:
    """呼叫 OpenAI 文字 API (無圖片)。"""
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY 未設定")
        return OpenAIUserNotice("AI 服務未設定，請聯絡管理員檢查環境變數。")

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }

    timeout = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if not resp.is_success:
            logger.error(
                "OpenAI Text HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:800],
            )
            um = _openai_error_user_message(resp.status_code, resp.text or "")
            if um:
                return OpenAIUserNotice(um)
        resp.raise_for_status()

    raw = _openai_extract_message_text(resp.json())
    if not raw:
        return ""
    parsed = _try_parse_json_response(raw)
    if parsed is not None:
        return parsed
    return raw


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LINE 圖片下載與壓縮
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def download_line_image_bytes(message_id: str) -> bytes:
    """從 LINE Message Content API 下載圖片原始位元組。"""
    url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            url,
            headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
        )
        resp.raise_for_status()
    return resp.content


def compress_image_bytes_to_jpeg_base64(data: bytes, max_side: int = 1600, quality: int = 85) -> str:
    """將圖片壓成 JPEG 再 base64，降低 Vision 請求體積。失敗時退回原始 base64。"""
    try:
        from PIL import Image

        im = Image.open(BytesIO(data))
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        logger.warning("圖片壓縮失敗，改以原始二進位送 Vision", exc_info=True)
        return base64.b64encode(data).decode("utf-8")


async def get_line_image_base64(message_id: str) -> str:
    """下載 LINE 圖片並壓縮為 JPEG base64，供 OpenAI Vision 使用。"""
    raw = await download_line_image_bytes(message_id)
    return compress_image_bytes_to_jpeg_base64(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 提示詞定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_MEAL_ANALYSIS = """你是專業、細心、有執照的極度資深營養師博士，嚴厲但也溫柔，專門為減脂客戶提供飲食分析。

核心原則：熱量寧可高估，不可低估（蛋白質維持合理、不偏高一併高估）。
- 若你判斷熱量範圍約 300～450 kcal，回報的單一數字應接近上限（例如 420～450），而非中位數
- 若份量模糊，假設較大份量；醬料、油脂、隱藏熱量一律算入
- 飲料的糖與加料按「合理範圍內較高」估算
- 外食餐廳份量通常比家常料理多約 20～30％，熱量一併反映
- 炸物、勾芡、焗烤等高油烹調，在合理範圍內額外多加約 15～20％ 熱量

估算步驟：
1. 先判斷食物種類與份量
2. 估算合理熱量區間（最低～最高）
3. 回報的 calories 取該區間約 75～85 百分位（偏上），不要取中位
4. 蛋白質以正常專業方式估算即可，不需刻意偏高

畫面若有手部或背景，僅為持握食物，請忽略；有包裝營養標示時，熱量仍以減脂保守原則為準（標示為參考，可因實際食用量略調高）。

不要使用任何 emoji。使用繁體中文。

請嚴格以下列 JSON 格式回覆，不要有其他文字：
{
  "calories": 數字,
  "protein": 數字,
  "description": "食物內容的簡要描述",
  "estimation_note": "簡短說明為何給這個熱量數字，例如：已含醬汁與用油、採區間上緣估算"
}"""

PROMPT_MEAL_FROM_TEXT = """你是專業、細心、有執照的極度資深營養師博士，嚴厲但也溫柔，專門為減脂客戶提供飲食分析。
使用者會「用中文文字」描述自己吃的內容與大概份量（沒有照片）。

核心原則與拍照分析相同：熱量寧可高估，不可低估（蛋白質維持合理、不刻意飆高）。
- 描述模糊時假設較大份量；醬料、用油、隱藏熱量一律算入
- 外食依餐廳份量特性在合理範圍內從寬估算

不要使用任何 emoji。使用繁體中文。

請嚴格以下列 JSON 格式回覆，不要有其他文字：
{
  "calories": 數字,
  "protein": 數字,
  "description": "食物內容的簡要描述（可整合使用者原意並補全份量假設）",
  "estimation_note": "簡短說明為何給這個熱量數字"
}"""

PROMPT_PURCHASE_QUERY = """你是專業的營養師與食品分析師，有長年的飲養輔助經驗、飲養分析研究、有執照。分析圖片中的食物或商品包裝，提供購買前的完整評估。

評級系統使用注音符號：
- ㄅ：非常適合減重／增肌目標，營養優良
- ㄆ：大致適合，偶爾可食用
- ㄇ：中性，需要控制份量
- ㄈ：不建議，高熱量或低營養價值
- ㄉ：極度不建議，對目標有嚴重負面影響

規則：
- 這是使用者自行拍攝的包裝與營養標示，屬於允許的分析情境。畫面中的手部、身體局部、桌面或背景請一律忽略，只專注讀取「包裝」上的品名、成分表、營養標示（每一份量、熱量、蛋白質、脂肪、碳水化合物、糖等）
- 只要營養標示或包裝文字有任何可讀片段，就必須據此填寫 JSON 數字欄位；略模糊時請合理估算，不要留空或填零敷衍
- 禁止以「無法識別圖片中的人物或物品」「無法辨識」等理由拒絕；禁止道歉、禁止要求使用者改傳文字、禁止閒聊；回覆必須且只能是一段合法 JSON
- 不要使用任何 emoji
- 使用繁體中文
- 評估要嚴格但公正
- 替代建議要具體且容易取得

請嚴格以下列 JSON 格式回覆，不要有其他文字：
{
  "name": "食物或商品名稱",
  "calories": 數字,
  "protein": 數字,
  "carbs": 數字,
  "fat": 數字,
  "sugar": 數字,
  "grades": {
    "overall": "ㄅ至ㄉ",
    "taste": "ㄅ至ㄉ",
    "fat_level": "ㄅ至ㄉ",
    "sugar_level": "ㄅ至ㄉ",
    "calorie_density": "ㄅ至ㄉ",
    "carb_quality": "ㄅ至ㄉ"
  },
  "timing": "適合的飲用或食用時機，例如健身前30分鐘、增肌期訓練後、減脂期避免等",
  "verdict": "一句話總結：建議購買或不建議，以及原因",
  "alternatives": ["更健康的替代選項1", "替代選項2", "替代選項3"]
}"""

PROMPT_INBODY_ANALYSIS = """你是專業的體適能教練。分析這張 InBody 體組成報告，提取關鍵數據。

規則：
- 若有手部或背景入鏡，請忽略，專注在報告紙張／螢幕上的數字與圖表
- 盡可能辨識所有數值
- 無法辨識的項目設為 null
- 不要使用任何 emoji
- 使用繁體中文

請嚴格以下列 JSON 格式回覆：
{
  "weight": 數字或null,
  "body_fat_percentage": 數字或null,
  "muscle_mass": 數字或null,
  "bmr": 數字或null,
  "body_water": 數字或null,
  "visceral_fat": 數字或null,
  "test_date": "YYYY-MM-DD或null",
  "summary": "一段簡要的身體組成分析與建議"
}"""

PROMPT_AI_COACH = """你是專業的健身營養教練，名叫「減重教練」。根據提供的數據，給出精準的個人化建議。

評分系統：
- S：超過100分的表現，極其出色
- A：優秀（85-100%）
- B：良好（70-84%）
- C：及格（55-69%）
- D：需改進（40-54%）
- E：需要大幅改進（<40%）

規則：
- 不要使用任何 emoji
- 使用繁體中文
- 語氣專業但友善，像一個嚴格但關心你的教練
- 建議要具體可執行，不要空泛
- 如果數據不足，誠實說明，不要編造

請嚴格以下列 JSON 格式回覆：
{
  "overall_grade": "S至E",
  "analysis": {
    "protein_adherence": {"grade": "S至E", "comment": "簡評"},
    "calorie_control": {"grade": "S至E", "comment": "簡評"},
    "consistency": {"grade": "S至E", "comment": "簡評"},
    "meal_balance": {"grade": "S至E", "comment": "簡評"}
  },
  "top_issues": ["最關鍵的問題1", "問題2", "問題3"],
  "action_plan": [
    {"task": "具體任務", "reason": "為什麼這很重要"},
    {"task": "具體任務", "reason": "原因"}
  ],
  "coach_note": "教練的個人化鼓勵或提醒，2-3句話"
}"""

PROMPT_CREATIVE_SUGGESTION = """你是一位專業的健身營養教練，有長年的飲養輔助經驗、有執照。用戶剛完成一餐紀錄，請根據他今天剩餘的蛋白質缺口，給出一個簡短、具體、有幫助的補充建議。

規則：
- 不要使用任何 emoji、表情符號、特殊符號
- 使用繁體中文
- 建議要具體到食物名稱和份量
- 控制在2-3行以內
- 語氣輕鬆但專業"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 功能處理函數
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

db = Database()


def build_progress_bar(current: float, target: float, width: int = 20) -> str:
    """產生文字進度條。"""
    ratio = min(current / target, 1.0) if target > 0 else 0
    filled = int(ratio * width)
    bar = ">" * filled + "-" * (width - filled)
    pct = ratio * 100
    return f"[{bar}] {pct:.0f}%"


def get_user_targets(user_id: str) -> dict:
    """取得使用者的每日目標。

    只要 profile 有體重，一律用 calculate_targets 依體重／體脂／BMR 重算，
    不再沿用 DB 裡可能過舊的 daily_*_target（例如先前寫入的 300g）。
    """
    profile = db.get_user_profile(user_id)
    if not profile:
        return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}
    w = profile.get("weight")
    if w is not None:
        try:
            wf = float(w)
            if wf > 0:
                return calculate_targets(
                    wf,
                    profile.get("body_fat_percentage"),
                    profile.get("bmr"),
                )
        except (TypeError, ValueError):
            pass
    cal = profile.get("daily_calorie_target")
    pro = profile.get("daily_protein_target")
    return {
        "calories": float(cal) if cal is not None else DEFAULT_CALORIE_TARGET,
        "protein": float(pro) if pro is not None else DEFAULT_PROTEIN_TARGET,
    }


def get_gap_filler(remaining_protein: float) -> str:
    """根據剩餘蛋白質量給建議。"""
    if remaining_protein <= 0:
        return "已達成今日蛋白質目標！保持下去。"
    elif remaining_protein <= 30:
        return f"只差 {remaining_protein:.0f}g——一份乳清蛋白即可達標。"
    elif remaining_protein <= 60:
        return f"還需 {remaining_protein:.0f}g——一塊雞胸肉（200g）約可補足。"
    elif remaining_protein <= 100:
        return f"還缺 {remaining_protein:.0f}g——可安排一份雞胸＋一份乳清。"
    else:
        return f"還缺 {remaining_protein:.0f}g——建議在剩餘餐次中優先選擇高蛋白食物。"


def _load_custom_quick_items(user_id: str) -> dict | None:
    profile = db.get_user_profile(user_id)
    if not profile:
        return None
    raw = profile.get("custom_quick_items")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


async def handle_quick_protein(user_id: str, item_name: str = "蛋白飲") -> str:
    """快速記錄固定品項，不呼叫 OpenAI。"""
    today_str = date.today().isoformat()
    custom = _load_custom_quick_items(user_id)

    if custom and item_name in custom:
        item = custom[item_name]
    elif item_name in DEFAULT_QUICK_ITEMS:
        item = DEFAULT_QUICK_ITEMS[item_name]
    else:
        return f"找不到「{item_name}」的快速記錄資料。"

    try:
        cal = float(item["calories"])
        pro = float(item["protein"])
    except (KeyError, TypeError, ValueError):
        return f"「{item_name}」資料格式異常，請用「設定蛋白飲」重新設定。"
    desc = str(item.get("description", item_name))

    db.add_meal(user_id, cal, pro, f"[快速記錄] {desc}", today_str)

    totals = db.get_daily_totals(user_id, today_str)
    targets = get_user_targets(user_id)
    remaining = targets["protein"] - totals["protein"]
    remaining_cal = targets["calories"] - totals["calories"]
    remaining_cal_line = (
        f"剩餘熱量：約 {remaining_cal:.0f} kcal"
        if remaining_cal >= 0
        else f"熱量狀態：已超過目標約 {-remaining_cal:.0f} kcal"
    )
    bar = build_progress_bar(totals["protein"], targets["protein"])
    gap = get_gap_filler(remaining)

    return (
        f"已記錄：{desc}\n"
        f"＋{cal:.0f} kcal／＋{pro:.0f} g 蛋白質\n"
        f"————\n"
        f"今日累計：\n"
        f"總熱量：{totals['calories']:.0f} kcal\n"
        f"{remaining_cal_line}\n"
        f"總蛋白質：{totals['protein']:.0f} g／{targets['protein']:.0f} g\n"
        f"{bar}\n\n"
        f"{gap}"
    )


def handle_set_quick_item(user_id: str, text: str) -> str:
    """解析「設定蛋白飲 130 25」並寫入自訂 JSON。"""
    numbers = re.findall(r"\d+", text)
    if len(numbers) < 2:
        return (
            "設定格式：\n"
            "  設定蛋白飲 [熱量] [蛋白質]\n\n"
            "範例：\n"
            "  設定蛋白飲 130 25\n"
            "  （一份 130 kcal、25 g 蛋白質）\n\n"
            "目前預設：130 kcal／25 g"
        )

    cal = int(numbers[0])
    pro = int(numbers[1])
    if cal > 2000 or pro > 200 or cal < 1 or pro < 1:
        return "數值異常，請確認單位為 kcal 與 g（熱量 1～2000，蛋白質 1～200）。"

    custom = _load_custom_quick_items(user_id) or {}
    custom["蛋白飲"] = {
        "calories": cal,
        "protein": pro,
        "description": "乳清蛋白飲 一份（自訂）",
    }
    db.update_custom_quick_items(
        user_id, json.dumps(custom, ensure_ascii=False)
    )

    return (
        f"蛋白飲數值已更新：\n"
        f"  熱量：{cal} kcal\n"
        f"  蛋白質：{pro} g\n\n"
        f"之後傳「加蛋白飲」會使用此數值記錄。"
    )


async def handle_meal_photo(user_id: str, message_id: str) -> str:
    """處理食物照片：分析＋記錄＋回傳摘要。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_MEAL_ANALYSIS,
        user_prompt="請分析這份餐點。目標為減脂，熱量請依提示採保守偏高估算；蛋白質正常估算。",
        image_base64=image_b64,
    )

    if isinstance(result, OpenAIUserNotice):
        return str(result)
    if isinstance(result, str):
        return f"分析失敗，請重新拍照。\n\n原始回應：\n{result[:200]}"

    cal = float(result.get("calories", 0))
    pro = float(result.get("protein", 0))
    desc = result.get("description", "無法辨識")
    note = (result.get("estimation_note") or "").strip()

    # 寫入 DB
    today_str = date.today().isoformat()
    db.add_meal(user_id, cal, pro, desc, today_str)

    # 今日累計
    totals = db.get_daily_totals(user_id, today_str)
    targets = get_user_targets(user_id)
    remaining = targets["protein"] - totals["protein"]
    remaining_cal = targets["calories"] - totals["calories"]
    remaining_cal_line = (
        f"剩餘熱量：約 {remaining_cal:.0f} kcal"
        if remaining_cal >= 0
        else f"熱量狀態：已超過目標約 {-remaining_cal:.0f} kcal"
    )
    bar = build_progress_bar(totals["protein"], targets["protein"])
    gap = get_gap_filler(remaining)

    lines = [
        "本餐分析結果",
        "————",
        f"食物內容：\n{desc}",
        "",
        "營養數據（減脂保守估計）：",
        f"熱量：{cal:.0f} kcal",
        f"蛋白質：{pro:.0f} g",
    ]
    if note:
        lines.append(f"說明：{note}")
    lines.extend([
        "",
        "————",
        "今日累計：",
        f"總熱量：{totals['calories']:.0f} kcal",
        remaining_cal_line,
        f"總蛋白質：{totals['protein']:.0f} g／{targets['protein']:.0f} g",
        f"{bar}",
        "",
        gap,
    ])
    return "\n".join(lines)


def _parse_meal_text_report(text: str) -> str | None:
    """若為文字報餐前綴，回傳去掉前綴後的描述；否則 None。"""
    t = text.strip()
    for prefix in ("記一筆", "記錄", "吃"):
        if t.startswith(prefix):
            desc = t[len(prefix) :].strip()
            if len(desc) >= 2:
                return desc
            return None
    return None


async def handle_meal_from_text(user_id: str, user_said: str) -> str:
    """依文字描述估算熱量與蛋白質並入帳（與拍照版同一套保守原則）。"""
    result = await call_openai_text(
        PROMPT_MEAL_FROM_TEXT,
        (
            "使用者原文如下，請估算並回傳 JSON：\n"
            f"{user_said.strip()}\n\n"
            "目標為減脂，熱量採保守偏高；蛋白質正常估算。"
        ),
    )

    if isinstance(result, OpenAIUserNotice):
        return str(result)
    if isinstance(result, str):
        return f"文字分析失敗，請寫清楚一點再試。\n\n原始回應：\n{result[:200]}"

    cal = float(result.get("calories", 0))
    pro = float(result.get("protein", 0))
    desc = result.get("description", "無法辨識")
    note = (result.get("estimation_note") or "").strip()

    today_str = date.today().isoformat()
    db.add_meal(user_id, cal, pro, f"[文字紀錄] {desc}", today_str)

    totals = db.get_daily_totals(user_id, today_str)
    targets = get_user_targets(user_id)
    remaining = targets["protein"] - totals["protein"]
    remaining_cal = targets["calories"] - totals["calories"]
    remaining_cal_line = (
        f"剩餘熱量：約 {remaining_cal:.0f} kcal"
        if remaining_cal >= 0
        else f"熱量狀態：已超過目標約 {-remaining_cal:.0f} kcal"
    )
    bar = build_progress_bar(totals["protein"], targets["protein"])
    gap = get_gap_filler(remaining)

    lines = [
        "本餐分析結果（文字紀錄）",
        "————",
        f"食物內容：\n{desc}",
        "",
        "營養數據（減脂保守估計）：",
        f"熱量：{cal:.0f} kcal",
        f"蛋白質：{pro:.0f} g",
    ]
    if note:
        lines.append(f"說明：{note}")
    lines.extend([
        "",
        "————",
        "今日累計：",
        f"總熱量：{totals['calories']:.0f} kcal",
        remaining_cal_line,
        f"總蛋白質：{totals['protein']:.0f} g／{targets['protein']:.0f} g",
        f"{bar}",
        "",
        gap,
    ])
    return "\n".join(lines)


async def handle_purchase_query_photo(user_id: str, message_id: str) -> str:
    """處理購買查詢照片。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_PURCHASE_QUERY,
        user_prompt=(
            "請只分析商品包裝與營養標示（忽略手持與背景），"
            "依標示填寫熱量與巨量營養素並完成購買前評估，回傳 JSON。"
        ),
        image_base64=image_b64,
        image_detail="high",
    )

    if isinstance(result, OpenAIUserNotice):
        clear_state(user_id)
        return str(result)
    if isinstance(result, str):
        clear_state(user_id)
        return f"分析失敗，請重新拍照。\n\n原始回應：\n{result[:200]}"

    # 儲存分析結果到 context，等使用者決定
    set_state(user_id, UserState.PURCHASE_REVIEWED, context=result)

    g = result.get("grades", {})
    if not isinstance(g, dict):
        g = {}

    lines = [
        f"購買前分析報告",
        f"{'=' * 24}",
        f"品項：{result.get('name', '未知')}",
        "",
        f"營養數據（估算）：",
        f"  熱量：{result.get('calories', '?')} kcal",
        f"  蛋白質：{result.get('protein', '?')} g",
        f"  碳水化合物：{result.get('carbs', '?')} g",
        f"  脂肪：{result.get('fat', '?')} g",
        f"  糖：{result.get('sugar', '?')} g",
        "",
        f"等級評定：",
        f"  綜合評價：{g.get('overall', '?')}",
        f"  美味度：{g.get('taste', '?')}",
        f"  脂肪含量：{g.get('fat_level', '?')}",
        f"  糖含量：{g.get('sugar_level', '?')}",
        f"  熱量密度：{g.get('calorie_density', '?')}",
        f"  碳水品質：{g.get('carb_quality', '?')}",
        "",
        f"適合時機：",
        f"  {result.get('timing', '無特別建議')}",
        "",
        f"{'=' * 24}",
        f"結論：{result.get('verdict', '')}",
        "",
        f"更健康的替代選項：",
    ]

    for i, alt in enumerate(result.get("alternatives", []), 1):
        lines.append(f"  {i}. {alt}")

    lines.extend([
        "",
        "-----",
        "回覆「買了」記錄此筆購買",
        "回覆「不買」放棄購買",
        "回覆其他文字則取消查詢",
    ])

    return "\n".join(lines)


async def handle_inbody_photo(user_id: str, message_id: str) -> str:
    """處理 InBody 照片：OCR＋更新目標。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_INBODY_ANALYSIS,
        user_prompt="請分析這張 InBody 體組成報告，提取所有可辨識的數據。",
        image_base64=image_b64,
    )

    if isinstance(result, OpenAIUserNotice):
        clear_state(user_id)
        return str(result)
    if isinstance(result, str):
        clear_state(user_id)
        return f"InBody 分析失敗，請確認照片清晰度。\n\n{result[:200]}"

    # 更新使用者檔案
    weight = result.get("weight")
    bf = result.get("body_fat_percentage")
    muscle = result.get("muscle_mass")
    bmr = result.get("bmr")

    # 根據數據重新計算目標
    new_targets = calculate_targets(weight, bf, bmr)

    db.upsert_user_profile(
        user_id=user_id,
        weight=weight,
        body_fat=bf,
        muscle_mass=muscle,
        bmr=bmr,
        calorie_target=new_targets["calories"],
        protein_target=new_targets["protein"],
    )

    clear_state(user_id)

    notion = get_notion_sync()
    if notion.should_sync_line_user(user_id):
        td = result.get("test_date")
        if not td or str(td).lower() == "null":
            td = date.today().isoformat()
        inbody_payload = {
            "test_date": str(td),
            "weight": weight,
            "body_fat_percentage": bf,
            "muscle_mass": muscle,
            "bmr": bmr,
        }
        try:
            await asyncio.to_thread(notion.sync_inbody, inbody_payload)
        except Exception as ne:
            logger.error("Notion InBody 同步失敗（仍會回覆 LINE）: %s", ne)

    lines = [
        "InBody 數據已更新",
        "=" * 24,
        "",
        "辨識結果：",
        f"  體重：{weight or '?'} kg",
        f"  體脂率：{bf or '?'} %",
        f"  骨骼肌：{muscle or '?'} kg",
        f"  基礎代謝：{bmr or '?'} kcal",
        f"  體水分：{result.get('body_water', '?')} kg",
        f"  內臟脂肪：{result.get('visceral_fat', '?')}",
        f"  檢測日期：{result.get('test_date', '?')}",
        "",
        "=" * 24,
        "目標已自動調整：",
        f"  每日熱量目標：{new_targets['calories']:.0f} kcal",
        f"  每日蛋白質目標：{new_targets['protein']:.0f} g",
        "",
        result.get("summary", ""),
    ]

    return "\n".join(lines)


def calculate_targets(weight, body_fat, bmr) -> dict:
    """
    每日熱量／蛋白質目標（減脂情境為主）。

    蛋白質：Helms et al. (2014) 熱量赤字下約 2.3–3.1 g/kg 去脂體重（LBM），
    並以 Morton et al. (2018) 約 1.62 g/kg 體重作為額外效益有限的上緣參考（取與 LBM 上限的較小值）。
    熱量：以 Katch–McArdle（有體脂時）估算 BMR，避免 InBody OCR 異常偏高的數字直接放大 TDEE；
    TDEE = BMR × PAL，赤字比例依體脂調整，並遵守「約不超過 TDEE 的 25–30%」與絕對赤字上限（ACSM／Hall 類安全區）。
    """
    if not weight:
        return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}

    w = float(weight)
    bf = float(body_fat) if body_fat is not None else None

    # 去脂體重（kg）
    if bf is not None and 3 <= bf <= 60:
        lbm = w * (1.0 - bf / 100.0)
        lbm = max(lbm, w * 0.45)
    else:
        lbm = w * 0.82

    # BMR：有可靠體脂時優先 Katch–McArdle，否則用 InBody 數值並做合理裁剪
    if bf is not None and 3 <= bf <= 60:
        bmr_used = 370.0 + 21.6 * lbm
    else:
        if not bmr:
            return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}
        raw = float(bmr)
        bmr_used = max(900.0, min(raw, min(3800.0, 24.0 * w + 450.0)))

    # PAL：阻力訓練＋日常活動（減脂仍不宜用過低 PAL，否則 TDEE 失真）
    if bf is not None and bf < 15:
        pal = 1.62
    elif bf is not None and bf >= 30:
        pal = 1.62
    elif w >= 110:
        pal = 1.58
    elif bf is None or bf >= 22:
        pal = 1.52
    else:
        pal = 1.48
    tdee = bmr_used * pal

    # 赤字：以 TDEE 的 22–28% 為主；體脂高者不超過約 30–35% 理論上限，並受絕對赤字約束
    if bf is None:
        d_pct = 0.22
    elif bf >= 32:
        d_pct = 0.25
    elif bf >= 25:
        d_pct = 0.25
    elif bf >= 18:
        d_pct = 0.22
    else:
        d_pct = 0.18

    deficit_rel = tdee * d_pct
    deficit_cap = min(1100.0, 0.35 * tdee)
    deficit = min(deficit_rel, deficit_cap)
    cal = tdee - deficit

    # 安全邊界（勿用過高的「每公斤最低熱量」，否則會抵銷赤字）
    cal = max(1400.0, cal)
    cal = min(cal, tdee - 350.0, 4800.0)

    # 蛋白質（g）：Helms 區間與 Morton 上緣
    morton_cap = 1.62 * w
    helms_lo = 2.3 * lbm
    helms_hi_lbm = 3.1 * lbm
    helms_hi = min(helms_hi_lbm, morton_cap)
    if helms_hi >= helms_lo:
        pro = (helms_lo + helms_hi) / 2.0
    else:
        pro = min(morton_cap, max(helms_lo, 2.05 * lbm))
    # 文獻實務上多落在約 210–250 g（高 LBM 減脂）；上緣與 Morton 1.62 g/kg 對齊
    pro = int(max(120, min(250, round(pro))))

    return {"calories": int(round(cal)), "protein": pro}


async def handle_weekly_score(user_id: str) -> str:
    """產生本週飲食積分卡。"""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())  # 本週一

    days_data = []
    for i in range(7):
        d = week_start + timedelta(days=i)
        if d > today:
            break
        totals = db.get_daily_totals(user_id, d.isoformat())
        days_data.append({"date": d.isoformat(), **totals})

    if not days_data or all(d["meal_count"] == 0 for d in days_data):
        return "本週尚無飲食紀錄，開始記錄後才能產生積分卡。"

    targets = get_user_targets(user_id)
    active_days = [d for d in days_data if d["meal_count"] > 0]
    total_days = len(active_days)

    if total_days == 0:
        return "本週尚無飲食紀錄。"

    # 蛋白質達標率
    protein_met = sum(1 for d in active_days if d["protein"] >= targets["protein"] * 0.9)
    protein_rate = protein_met / total_days

    # 平均熱量偏差
    avg_cal = sum(d["calories"] for d in active_days) / total_days
    cal_diff_pct = abs(avg_cal - targets["calories"]) / targets["calories"]

    # 紀錄規律性 (有紀錄的天數比例)
    elapsed_days = (today - week_start).days + 1
    regularity_rate = total_days / elapsed_days

    # 每日餐次穩定度
    meal_counts = [d["meal_count"] for d in active_days]
    avg_meals = sum(meal_counts) / len(meal_counts) if meal_counts else 0

    def grade_general(rate):
        if rate >= 0.95:
            return "S"
        elif rate >= 0.85:
            return "A"
        elif rate >= 0.70:
            return "B"
        elif rate >= 0.55:
            return "C"
        elif rate >= 0.40:
            return "D"
        else:
            return "E"

    def grade_calorie(diff):
        if diff <= 0.05:
            return "S"
        elif diff <= 0.10:
            return "A"
        elif diff <= 0.15:
            return "B"
        elif diff <= 0.25:
            return "C"
        elif diff <= 0.35:
            return "D"
        else:
            return "E"

    protein_grade = grade_general(protein_rate)
    calorie_grade = grade_calorie(cal_diff_pct)
    regularity_grade = grade_general(regularity_rate)

    # 綜合
    grade_values = {"S": 6, "A": 5, "B": 4, "C": 3, "D": 2, "E": 1}
    avg_score = (
        grade_values[protein_grade]
        + grade_values[calorie_grade]
        + grade_values[regularity_grade]
    ) / 3

    overall = "S" if avg_score >= 5.5 else \
              "A" if avg_score >= 4.5 else \
              "B" if avg_score >= 3.5 else \
              "C" if avg_score >= 2.5 else \
              "D" if avg_score >= 1.5 else "E"

    # 生成評語
    comments = []
    if protein_grade in ("D", "E"):
        comments.append(f"蛋白質達標率偏低（{protein_met}/{total_days} 天），需要在每餐加入高蛋白食物。")
    elif protein_grade in ("S", "A"):
        comments.append(f"蛋白質攝取表現出色（{protein_met}/{total_days} 天達標）。")

    if calorie_grade in ("D", "E"):
        direction = "超出" if avg_cal > targets["calories"] else "不足"
        comments.append(f"平均熱量{direction}目標 {abs(avg_cal - targets['calories']):.0f} kcal，需要調整。")
    elif calorie_grade in ("S", "A"):
        comments.append("熱量控制非常精準，繼續保持。")

    if regularity_grade in ("D", "E"):
        comments.append(f"本週只有 {total_days}/{elapsed_days} 天有紀錄，請養成每餐拍照的習慣。")

    # 儲存本週積分
    db.save_weekly_score(
        user_id=user_id,
        week_start=week_start.isoformat(),
        week_end=today.isoformat(),
        overall=overall,
        protein=protein_grade,
        calorie=calorie_grade,
        regularity=regularity_grade,
    )

    lines = [
        f"本週飲食積分卡",
        f"({week_start.isoformat()} ~ {today.isoformat()})",
        "=" * 24,
        "",
        f"  綜合評分：{overall}",
        "",
        f"  蛋白質達標：{protein_grade}",
        f"    （{protein_met}/{total_days} 天達到 90% 以上）",
        "",
        f"  熱量控制：{calorie_grade}",
        f"    （平均 {avg_cal:.0f}／目標 {targets['calories']:.0f} kcal）",
        "",
        f"  紀錄規律：{regularity_grade}",
        f"    （{total_days}/{elapsed_days} 天有紀錄）",
        "",
        "=" * 24,
        "教練評語：",
    ]

    for c in comments:
        lines.append(f"  {c}")

    return "\n".join(lines)


async def handle_cheat_day(user_id: str) -> str:
    """啟動或查詢欺騙日。"""
    today_str = date.today().isoformat()

    if db.is_cheat_day(user_id, today_str):
        # 已經是欺騙日
        totals = db.get_daily_totals(user_id, today_str)
        targets = get_user_targets(user_id)
        cheat_cal = targets["calories"] * 1.3

        return (
            f"今天已經是欺騙日模式\n"
            f"{'=' * 24}\n"
            f"正常熱量上限：{targets['calories']:.0f} kcal\n"
            f"欺騙日上限：{cheat_cal:.0f} kcal\n"
            f"目前已攝取：{totals['calories']:.0f} kcal\n\n"
            f"提醒：欺騙日不代表無限制，\n"
            f"蛋白質目標依然需要達成。"
        )

    # 檢查本週是否已使用
    week_start = date.today() - timedelta(days=date.today().weekday())
    if db.count_cheat_days_in_range(user_id, week_start.isoformat(), today_str) > 0:
        return (
            "本週已使用過欺騙日。\n"
            "建議每週最多一次，以免影響整體進度。\n\n"
            "如果仍要啟動，請輸入「強制欺騙日」。"
        )

    # 啟動欺騙日
    db.activate_cheat_day(user_id, today_str)
    targets = get_user_targets(user_id)
    cheat_cal = targets["calories"] * 1.3

    return (
        f"欺騙日模式已啟動\n"
        f"{'=' * 24}\n"
        f"今日熱量上限放寬至：{cheat_cal:.0f} kcal\n"
        f"（正常：{targets['calories']:.0f} kcal，＋{cheat_cal - targets['calories']:.0f}）\n\n"
        f"規則：\n"
        f"  1. 蛋白質目標不變\n"
        f"  2. 盡量安排在訓練日使用\n"
        f"  3. 不會對今日的食物評語過於嚴格\n\n"
        f"享受美食，明天繼續努力。"
    )


async def handle_ai_coach(user_id: str) -> str:
    """AI 教練分析。"""
    today = date.today()
    targets = get_user_targets(user_id)

    # 收集過去 14 天數據
    days_data = []
    for i in range(14):
        d = today - timedelta(days=i)
        totals = db.get_daily_totals(user_id, d.isoformat())
        if totals["meal_count"] > 0:
            days_data.append({"date": d.isoformat(), **totals})

    if len(days_data) < 3:
        return (
            "數據不足，無法進行有效分析。\n"
            f"目前只有 {len(days_data)} 天的紀錄，\n"
            "建議至少累積 3 天以上再使用 AI 教練功能。"
        )

    # 組裝數據摘要
    profile = db.get_user_profile(user_id)
    data_summary = {
        "days_recorded": len(days_data),
        "avg_daily_calories": sum(d["calories"] for d in days_data) / len(days_data),
        "avg_daily_protein": sum(d["protein"] for d in days_data) / len(days_data),
        "days_meeting_protein_90pct": sum(
            1 for d in days_data if d["protein"] >= targets["protein"] * 0.9
        ),
        "calorie_target": targets["calories"],
        "protein_target": targets["protein"],
        "daily_details": days_data[:7],  # 只送最近 7 天明細
    }

    if profile:
        data_summary["weight"] = profile.get("weight")
        data_summary["body_fat"] = profile.get("body_fat_percentage")

    result = await call_openai_text(
        system_prompt=PROMPT_AI_COACH,
        user_prompt=f"以下是使用者的飲食數據，請進行分析：\n{json.dumps(data_summary, ensure_ascii=False, indent=2)}",
    )

    if isinstance(result, OpenAIUserNotice):
        return str(result)
    if isinstance(result, str):
        return f"AI 教練分析完成：\n\n{result}"

    # 格式化輸出
    lines = [
        "AI 教練分析報告",
        "=" * 24,
        "",
        f"綜合評分：{result.get('overall_grade', '?')}",
        "",
        "各項評分：",
    ]

    analysis = result.get("analysis", {})
    labels = {
        "protein_adherence": "蛋白質遵守",
        "calorie_control": "熱量控制",
        "consistency": "一致性",
        "meal_balance": "營養均衡",
    }
    for key, label in labels.items():
        item = analysis.get(key, {})
        lines.append(f"  {label}：{item.get('grade', '?')}－{item.get('comment', '')}")

    lines.extend(["", "=" * 24, "主要問題："])
    for i, issue in enumerate(result.get("top_issues", []), 1):
        lines.append(f"  {i}. {issue}")

    lines.extend(["", "行動計畫："])
    for item in result.get("action_plan", []):
        lines.append(f"  - {item.get('task', '')}")
        lines.append(f"    原因：{item.get('reason', '')}")

    coach_note = result.get("coach_note", "")
    if coach_note:
        lines.extend(["", "-----", f"教練的話：{coach_note}"])

    return "\n".join(lines)


async def handle_today_summary(user_id: str) -> str:
    """今日飲食總結。"""
    today_str = date.today().isoformat()
    totals = db.get_daily_totals(user_id, today_str)
    targets = get_user_targets(user_id)
    meals = db.get_meals_today(user_id, today_str)

    if totals["meal_count"] == 0:
        return "今天還沒有任何飲食紀錄。\n拍一張食物照片開始記錄吧。"

    remaining_protein = targets["protein"] - totals["protein"]
    bar = build_progress_bar(totals["protein"], targets["protein"])

    is_cheat = db.is_cheat_day(user_id, today_str)
    cal_target = targets["calories"] * 1.3 if is_cheat else targets["calories"]
    remaining_cal = cal_target - totals["calories"]
    if remaining_cal >= 0:
        remaining_cal_line = f"剩餘熱量：約 {remaining_cal:.0f} kcal"
    else:
        remaining_cal_line = f"熱量狀態：已超過今日目標約 {-remaining_cal:.0f} kcal"

    lines = [
        f"今日飲食總結 ({today_str})",
        f"{'[欺騙日模式]' if is_cheat else ''}",
        "————",
        "",
        f"總熱量：{totals['calories']:.0f}／{cal_target:.0f} kcal",
        remaining_cal_line,
        f"總蛋白質：{totals['protein']:.0f}／{targets['protein']:.0f} g",
        f"{bar}",
        "",
        f"共 {totals['meal_count']} 餐：",
    ]

    for i, meal in enumerate(meals, 1):
        lines.append(f"{i}. {meal['food_description']}")
        lines.append(f"{meal['calories']:.0f} kcal／{meal['protein']:.0f}g 蛋白質")

    lines.append("")
    lines.append(get_gap_filler(remaining_protein))

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定時推播（建議由 GitHub Actions 呼叫 /cron/daily-summary）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def execute_daily_summary_push() -> dict:
    """推播每日總結（可重複呼叫）：對象為近期曾互動／有紀錄者；無餐點也會收到提示。"""
    today_str = date.today().isoformat()
    meal_since = (date.today() - timedelta(days=400)).isoformat()
    user_ids = db.get_user_ids_for_daily_summary(meal_since)
    ok, fail = 0, 0
    notion = get_notion_sync()
    logger.info("每日總結：預計推播 %s 位使用者", len(user_ids))
    for uid in user_ids:
        try:
            if notion.should_sync_line_user(uid):
                totals = db.get_daily_totals(uid, today_str)
                if totals.get("meal_count", 0) > 0:
                    targets = get_user_targets(uid)
                    try:
                        await asyncio.to_thread(
                            notion.sync_daily_nutrition,
                            today_str,
                            {
                                **totals,
                                "carbs": 0.0,
                                "fat": 0.0,
                            },
                            float(targets["protein"]),
                        )
                    except Exception as ne:
                        logger.error(
                            "Notion 每日同步失敗（仍會推播 LINE）%s: %s",
                            uid[:8],
                            ne,
                        )
            summary = await handle_today_summary(uid)
            await push_line_text_with_retry(uid, summary)
            ok += 1
            logger.info("已推播每日總結給 %s...", uid[:8])
        except Exception as e:
            fail += 1
            logger.error("推播失敗 %s: %s", uid[:8], e)
    return {"date": today_str, "users": len(user_ids), "pushed_ok": ok, "pushed_fail": fail}


def _utc_range_local_midnight_to(
    local_day: date, end_local: time, tz: ZoneInfo
) -> tuple[str, str]:
    """當地日期的 00:00 起至 end_local（不含）止，轉成 UTC ISO 區間 [start, end)。"""
    start_local = datetime.combine(local_day, time(0, 0), tzinfo=tz)
    end_local = datetime.combine(local_day, end_local, tzinfo=tz)
    return (
        start_local.astimezone(timezone.utc).isoformat(),
        end_local.astimezone(timezone.utc).isoformat(),
    )


def _utc_range_local_window(
    local_day: date, start_local: time, end_local: time, tz: ZoneInfo
) -> tuple[str, str]:
    """當地同日 start_local（含）至 end_local（不含）轉成 UTC ISO 區間 [start, end)。"""
    a = datetime.combine(local_day, start_local, tzinfo=tz)
    b = datetime.combine(local_day, end_local, tzinfo=tz)
    return (
        a.astimezone(timezone.utc).isoformat(),
        b.astimezone(timezone.utc).isoformat(),
    )


def _next_meal_reminder_fire_utc(now_utc: datetime, tz: ZoneInfo) -> tuple[datetime, str]:
    """下次觸發：當地 13:00 午餐提醒／20:30 晚餐提醒。"""
    now_local = now_utc.astimezone(tz)
    d = now_local.date()
    noon_local = datetime.combine(d, time(13, 0), tzinfo=tz)
    eve_local = datetime.combine(d, time(20, 30), tzinfo=tz)
    cands: list[tuple[datetime, str]] = []
    if now_local < noon_local:
        cands.append((noon_local.astimezone(timezone.utc), "noon"))
    if now_local < eve_local:
        cands.append((eve_local.astimezone(timezone.utc), "evening"))
    if not cands:
        d2 = d + timedelta(days=1)
        n2 = datetime.combine(d2, time(13, 0), tzinfo=tz)
        cands.append((n2.astimezone(timezone.utc), "noon"))
    return min(cands, key=lambda x: x[0])


async def execute_meal_reminder_push(slot: str) -> dict:
    """13:00／20:30：若該時段內未傳圖、且無任何入帳餐點，才推提醒（文字報餐／快速加蛋白等入帳也算已吃過）。"""
    tzname = (os.getenv("BOT_TIMEZONE") or "Asia/Taipei").strip()
    tz = ZoneInfo(tzname)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    local_date = now_local.date()
    local_date_s = local_date.isoformat()

    if slot == "noon":
        start_iso, end_iso = _utc_range_local_window(
            local_date, time(11, 0), time(13, 0), tz
        )
        text = "中午吃什麼～"
    else:
        start_iso, end_iso = _utc_range_local_window(
            local_date, time(17, 0), time(20, 30), tz
        )
        text = "晚餐吃什麼～"

    meal_since = (local_date - timedelta(days=400)).isoformat()
    user_ids = db.get_user_ids_for_meal_reminders(meal_since)
    ok, skip, fail = 0, 0, 0
    configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    async with AsyncApiClient(configuration) as api_client:
        line_api = AsyncMessagingApi(api_client)
        for uid in user_ids:
            try:
                if db.user_had_photo_in_utc_range(uid, start_iso, end_iso):
                    skip += 1
                    continue
                if db.user_had_meal_logged_in_utc_window(
                    uid, local_date_s, start_iso, end_iso
                ):
                    skip += 1
                    continue
                if db.reminder_already_sent(uid, local_date_s, slot):
                    skip += 1
                    continue
                await line_api.push_message(
                    PushMessageRequest(
                        to=uid,
                        messages=[TextMessage(text=text)],
                    )
                )
                db.mark_reminder_sent(uid, local_date_s, slot)
                ok += 1
                logger.info("已推播用餐提醒（%s）給 %s...", slot, uid[:8])
            except Exception as e:
                fail += 1
                logger.error("用餐提醒推播失敗 %s: %s", uid[:8], e)
    return {
        "slot": slot,
        "date": local_date_s,
        "users": len(user_ids),
        "pushed_ok": ok,
        "skipped": skip,
        "pushed_fail": fail,
    }


async def meal_reminder_job_internal():
    """進程內排程：依 BOT_TIMEZONE 每日 13:00、20:30 觸發。"""
    tzname = (os.getenv("BOT_TIMEZONE") or "Asia/Taipei").strip()
    tz = ZoneInfo(tzname)
    logger.info("用餐提醒排程時區：%s", tzname)
    while True:
        now = datetime.now(timezone.utc)
        fire_utc, slot = _next_meal_reminder_fire_utc(now, tz)
        wait_sec = (fire_utc - now).total_seconds()
        logger.info(
            "用餐提醒：下次觸發 %s（約 %.0f 秒後）",
            slot,
            max(0, wait_sec),
        )
        await asyncio.sleep(max(1.0, wait_sec))
        try:
            await execute_meal_reminder_push(slot)
        except Exception:
            logger.exception("用餐提醒批次失敗")
        await asyncio.sleep(3)


async def daily_summary_job_internal():
    """僅在 ENABLE_INTERNAL_DAILY_CRON=1 時啟用：進程內每晚當地 22:00 推播（勿用伺服器本地時區）。"""
    tzname = (os.getenv("BOT_TIMEZONE") or "Asia/Taipei").strip()
    tz = ZoneInfo(tzname)
    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(tz)
        target_local = datetime.combine(now_local.date(), time(22, 0), tzinfo=tz)
        if now_local >= target_local:
            target_local += timedelta(days=1)
        fire_utc = target_local.astimezone(timezone.utc)
        wait_seconds = max(1.0, (fire_utc - now_utc).total_seconds())
        logger.info(
            "內建每日總結：下次當地 22:00 推播在 %.0f 秒後（%s）",
            wait_seconds,
            target_local.isoformat(),
        )
        await asyncio.sleep(wait_seconds)
        await execute_daily_summary_push()
        await asyncio.sleep(60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 應用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期：初始化 DB；可選內建當地 22:00 總結與用餐提醒排程。"""
    db.init()
    bg_tasks: list[asyncio.Task] = []
    if os.getenv("ENABLE_INTERNAL_DAILY_CRON") == "1":
        bg_tasks.append(asyncio.create_task(daily_summary_job_internal()))
        logger.info("已啟用進程內每日總結（BOT_TIMEZONE 22:00）")
    if os.getenv("ENABLE_INTERNAL_MEAL_REMINDERS", "1") != "0":
        bg_tasks.append(asyncio.create_task(meal_reminder_job_internal()))
        logger.info("已啟用進程內用餐提醒（13:00／20:30，BOT_TIMEZONE）")
    logger.info("Bot 啟動完成")
    yield
    for t in bg_tasks:
        t.cancel()


app = FastAPI(title="Diet Tracker LINE Bot", lifespan=lifespan)

parser = WebhookParser(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)


def truncate_line_text(text: str) -> str:
    if len(text) > 5000:
        return text[:4950] + "\n\n（訊息過長，已截斷。）"
    return text


async def push_line_text(user_id: str, text: str):
    """以 Push API 傳送文字（用於圖片分析完成後，不依賴 reply_token）。"""
    text = truncate_line_text(text)
    cfg = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    async with AsyncApiClient(cfg) as api_client:
        api = AsyncMessagingApi(api_client)
        await api.push_message(
            PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
        )


async def push_line_text_with_retry(user_id: str, text: str, attempts: int = 3):
    """Push 失敗時短暫重試，降低網路抖動造成的漏訊。"""
    last_err: Exception | None = None
    for i in range(attempts):
        try:
            await push_line_text(user_id, text)
            return
        except Exception as e:
            last_err = e
            logger.warning("push 嘗試 %s/%s 失敗: %s", i + 1, attempts, e)
            await asyncio.sleep(0.8 * (i + 1))
    if last_err:
        raise last_err


async def _analyze_image_by_state(user_id: str, message_id: str, state_at_receive: str) -> str:
    if state_at_receive == UserState.WAITING_PURCHASE_PHOTO:
        return await handle_purchase_query_photo(user_id, message_id)
    if state_at_receive == UserState.WAITING_INBODY_PHOTO:
        return await handle_inbody_photo(user_id, message_id)
    return await handle_meal_photo(user_id, message_id)


def _log_image_background_task(task: asyncio.Task) -> None:
    """create_task 預設吞掉例外；在此記錄，避免圖片分析靜默失敗。"""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logger.error("背景圖片分析任務失敗: %s", exc, exc_info=exc)


async def run_image_analysis_and_push(user_id: str, message_id: str, state_at_receive: str):
    """背景執行：下載、壓縮、Vision、寫入 DB，完成後 push 結果。"""
    timeout_sec = float(os.getenv("IMAGE_ANALYSIS_TIMEOUT_SEC", "150"))
    logger.info(
        "圖片分析開始 user=%s msg=%s state=%s timeout=%.0fs",
        user_id[:8],
        message_id,
        state_at_receive,
        timeout_sec,
    )
    try:
        body = await asyncio.wait_for(
            _analyze_image_by_state(user_id, message_id, state_at_receive),
            timeout=timeout_sec,
        )
        logger.info("圖片分析完成 user=%s msg=%s", user_id[:8], message_id)
    except asyncio.TimeoutError:
        logger.error("圖片分析逾時 user=%s msg=%s", user_id[:8], message_id)
        body = (
            "本次圖片分析耗時過長，已自動中止。\n"
            "請重新傳一次照片（盡量清晰、只拍重點），我會立即重跑。"
        )
    except Exception as e:
        logger.error("背景圖片分析失敗: %s", e, exc_info=True)
        body = "分析過程發生錯誤，請稍後再試或重新傳送照片。"
    try:
        await push_line_text_with_retry(user_id, body, attempts=3)
        logger.info("圖片分析結果已推送 user=%s msg=%s", user_id[:8], message_id)
    except Exception as e:
        logger.error("Push 分析結果失敗: %s", e, exc_info=True)


HELP_TEXT = (
    "減重監測 Bot 使用說明\n"
    "=" * 24 + "\n\n"
    "基本操作：\n"
    "  拍照傳食物，將自動分析並記錄（熱量為減脂保守估計）。\n"
    "  文字報餐：以「記錄」「記一筆」或「吃」開頭＋描述（例：記錄 雞腿飯半個＋燙青菜）。\n\n"
    "文字指令：\n"
    "  「今日」：查看今日總結\n"
    "  「清除今日」：刪除今日所有紀錄\n"
    "  「本週積分」「週報」「積分卡」等：本週飲食成績\n"
    "  「設定蛋白飲 熱量 蛋白質」：自訂快速記錄數值（例：設定蛋白飲 130 25）\n"
    "  「說明」：顯示此說明\n\n"
    "圖文選單：\n"
    "  「食物查詢」或「購買查詢」：買前先查熱量等級\n"
    "  「加蛋白飲」：一鍵記錄一份蛋白飲（不耗 AI）\n"
    "  「上傳InBody」：更新體組成\n"
    "  「欺騙日」「AI教練」「今日總結」：同字面\n\n"
    "快速記錄（不耗 token）：\n"
    "  「加蛋白飲」「蛋白飲」「＋蛋白」\n"
    "  「加雞蛋」「＋雞蛋」\n"
    "  「加雞胸肉」「＋雞胸肉」"
)


@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    async with AsyncApiClient(configuration) as api_client:
        line_api = AsyncMessagingApi(api_client)

        for event in events:
            if not isinstance(event, MessageEvent):
                continue

            user_id = event.source.user_id
            if isinstance(event.message, ImageMessageContent):
                msg_kind = "image"
            elif isinstance(event.message, TextMessageContent):
                msg_kind = "text"
            else:
                msg_kind = "other"
            try:
                db.log_line_message(
                    user_id,
                    datetime.now(timezone.utc).isoformat(),
                    message_kind=msg_kind,
                )
            except Exception as e:
                logger.warning("log_line_message 失敗（略過）: %s", e)

            reply_token = event.reply_token
            state = get_state(user_id)

            try:
                if isinstance(event.message, ImageMessageContent):
                    logger.info(
                        "收到圖片訊息 user=%s msg=%s state=%s",
                        user_id[:8],
                        event.message.id,
                        state,
                    )
                    reply_text = "已收到照片，正在分析中…"
                    snap_state = get_state(user_id)
                    bg = asyncio.create_task(
                        run_image_analysis_and_push(
                            user_id, event.message.id, snap_state
                        )
                    )
                    bg.add_done_callback(_log_image_background_task)
                else:
                    reply_text = await route_message(event, user_id, state)
            except Exception as e:
                logger.error(f"處理訊息失敗: {e}", exc_info=True)
                reply_text = "處理時發生錯誤，請稍後再試。"

            if reply_text:
                reply_text = truncate_line_text(reply_text)

                try:
                    await line_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=reply_token,
                            messages=[TextMessage(text=reply_text)],
                        )
                    )
                except Exception as e:
                    logger.warning(
                        "reply_message 失敗（常見於 Reply Token 過期或主機冷啟動過慢）: %s，改以 push 傳送",
                        e,
                    )
                    try:
                        await push_line_text(user_id, reply_text)
                    except Exception as e2:
                        logger.error("push 備援亦失敗: %s", e2, exc_info=True)

    return JSONResponse(content={"status": "ok"})


async def route_message(event: MessageEvent, user_id: str, state: str) -> str:
    """路由訊息到對應的處理函數（圖片改由 webhook 立即回覆後於背景分析並 push）。"""
    if isinstance(event.message, ImageMessageContent):
        return ""

    # ── 文字訊息 ──
    if isinstance(event.message, TextMessageContent):
        text = event.message.text.replace("\u3000", " ").strip()

        # 狀態內的文字回應
        if state == UserState.PURCHASE_REVIEWED:
            if text in ("買了", "確定", "購買"):
                ctx = get_context(user_id)
                db.save_purchase_decision(user_id, ctx, "purchased")
                clear_state(user_id)
                return (
                    f"已記錄購買：{ctx.get('name', '未知')}\n"
                    f"熱量 {ctx.get('calories', '?')} kcal 已計入今日統計。"
                )
            elif text in ("不買", "取消", "放棄"):
                ctx = get_context(user_id)
                db.save_purchase_decision(user_id, ctx, "cancelled")
                clear_state(user_id)
                return "明智的選擇。繼續保持紀律。"
            else:
                clear_state(user_id)
                return "查詢已取消。"

        # ── 一般指令（優先於「等待傳照」提示，才能隨時切換購買查詢／InBody 等）──
        if text in ("說明", "幫助", "help", "Help", "HELP"):
            leave_photo_wait_if_any(user_id)
            return HELP_TEXT

        if text in ("今日", "今日總計", "今日總結", "總計", "今天"):
            leave_photo_wait_if_any(user_id)
            return await handle_today_summary(user_id)

        if text == "清除今日":
            leave_photo_wait_if_any(user_id)
            today_str = date.today().isoformat()
            count = db.clear_today(user_id, today_str)
            return f"已清除今日 {count} 筆紀錄。"

        if text.startswith("設定蛋白飲"):
            leave_photo_wait_if_any(user_id)
            return handle_set_quick_item(user_id, text)

        if text in ("加蛋白飲", "蛋白飲", "+蛋白飲", "+蛋白", "＋蛋白飲", "＋蛋白"):
            leave_photo_wait_if_any(user_id)
            return await handle_quick_protein(user_id, "蛋白飲")

        if text in ("加雞蛋", "+雞蛋", "＋雞蛋"):
            leave_photo_wait_if_any(user_id)
            return await handle_quick_protein(user_id, "雞蛋")

        if text in ("加雞胸肉", "+雞胸肉", "＋雞胸肉"):
            leave_photo_wait_if_any(user_id)
            return await handle_quick_protein(user_id, "雞胸肉")

        if text in ("購買查詢", "食物查詢", "查詢", "買之前"):
            set_state(user_id, UserState.WAITING_PURCHASE_PHOTO)
            return (
                "購買前熱量查詢已啟動\n"
                "-----\n"
                "請拍攝食物或商品包裝照片傳送。\n"
                "我會分析營養數據並給予等級評定。\n\n"
                "輸入「取消」可結束查詢。"
            )

        if text in ("本週積分", "積分", "積分卡", "本週", "週報"):
            leave_photo_wait_if_any(user_id)
            return await handle_weekly_score(user_id)

        if text in ("上傳InBody", "InBody", "inbody", "INBODY"):
            set_state(user_id, UserState.WAITING_INBODY_PHOTO)
            return (
                "InBody 上傳模式已啟動\n"
                "-----\n"
                "請拍攝或傳送 InBody 體組成報告照片。\n"
                "我會自動辨識數據並更新你的營養目標。\n\n"
                "輸入「取消」可結束。"
            )

        if text in ("欺騙日", "cheat day", "Cheat Day"):
            leave_photo_wait_if_any(user_id)
            return await handle_cheat_day(user_id)

        if text == "強制欺騙日":
            leave_photo_wait_if_any(user_id)
            today_str = date.today().isoformat()
            db.activate_cheat_day(user_id, today_str)
            targets = get_user_targets(user_id)
            cheat_cal = targets["calories"] * 1.3
            return (
                f"欺騙日已強制啟動\n"
                f"今日熱量上限：{cheat_cal:.0f} kcal\n\n"
                f"注意：頻繁使用欺騙日會影響整體進度。"
            )

        if text in ("AI教練", "教練", "ai教練", "AI 教練"):
            leave_photo_wait_if_any(user_id)
            return await handle_ai_coach(user_id)

        if text in ("目標", "我的目標", "查看目標"):
            leave_photo_wait_if_any(user_id)
            targets = get_user_targets(user_id)
            profile = db.get_user_profile(user_id)
            lines = [
                "目前設定的目標",
                "=" * 24,
                f"每日熱量：{targets['calories']:.0f} kcal",
                f"每日蛋白質：{targets['protein']:.0f} g",
            ]
            if profile:
                lines.extend([
                    "",
                    "身體數據：",
                    f"  體重：{profile.get('weight', '?')} kg",
                    f"  體脂率：{profile.get('body_fat_percentage', '?')} %",
                    f"  上次 InBody：{profile.get('last_inbody_date', '未上傳')}",
                ])
            return "\n".join(lines)

        if state == UserState.WAITING_PURCHASE_PHOTO:
            if text in ("取消", "結束"):
                clear_state(user_id)
                return "購買查詢已取消。"
            return "請傳送食物或商品包裝的照片。\n或輸入「取消」結束查詢。"

        if state == UserState.WAITING_INBODY_PHOTO:
            if text in ("取消", "結束"):
                clear_state(user_id)
                return "InBody 上傳已取消。"
            return "請傳送 InBody 報告的照片。\n或輸入「取消」結束。"

        meal_txt = _parse_meal_text_report(text)
        if meal_txt:
            leave_photo_wait_if_any(user_id)
            return await handle_meal_from_text(user_id, meal_txt)

        # 未知指令 → 當作食物文字描述? 或提示使用說明
        return (
            f"無法辨識指令「{text[:20]}」。\n\n"
            "拍照傳食物可自動分析記錄；\n"
            "或用「記錄 …」「吃 …」文字報餐。\n"
            "輸入「說明」查看所有功能。"
        )

    return ""


@app.get("/ping")
async def ping():
    """給 UptimeRobot 等監控每幾分鐘 GET，避免 Render 閒置休眠。"""
    return {"status": "ok"}


@app.head("/ping")
async def ping_head():
    """允許 HEAD 探活，避免 405 噪音。"""
    return JSONResponse(content=None, status_code=200)


@app.post("/cron/daily-summary")
async def cron_daily_summary(request: Request):
    """給 GitHub Actions 定時 POST；標頭 X-Cron-Secret 須與環境變數 CRON_SECRET 相同。"""
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="CRON_SECRET 未設定")
    got = request.headers.get("X-Cron-Secret", "")
    ga, gb = got.encode("utf-8"), secret.encode("utf-8")
    if len(ga) != len(gb) or not hmac.compare_digest(ga, gb):
        raise HTTPException(status_code=401, detail="Unauthorized")
    result = await execute_daily_summary_push()
    return JSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
