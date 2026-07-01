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
import unicodedata
from io import BytesIO
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from contextlib import asynccontextmanager
import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request, HTTPException
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
from linebot.v3.webhooks import FollowEvent, MessageEvent, TextMessageContent, ImageMessageContent

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
JITAI_MODEL = (os.getenv("JITAI_MODEL") or "gpt-4o-mini").strip()

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

FITNESS_GOALS = ("增肌", "減脂", "增肌減脂", "維持", "增肥")
FITNESS_GOAL_ALIASES = {
    "增肌": "增肌",
    "減脂": "減脂",
    "增肌減脂": "增肌減脂",
    "增肌＋減脂": "增肌減脂",
    "增肌+減脂": "增肌減脂",
    "recomp": "增肌減脂",
    "維持": "維持",
    "維持體重": "維持",
    "增肥": "增肥",
    "增重": "增肥",
}

ONBOARDING_WELCOME_TEXT = (
    "歡迎使用飲食追蹤 Bot\n"
    "================\n\n"
    "開始使用前，請先完成兩步驟：\n"
    "1. 上傳一張 InBody 體組成報告照片\n"
    "2. 回覆你的健身目標（擇一）：\n"
    "   增肌／減脂／增肌減脂／維持／增肥\n\n"
    "請先傳 InBody 報告照片。"
)

FITNESS_GOAL_PROMPT = (
    "請回覆你的健身目標（擇一）：\n"
    "  增肌\n"
    "  減脂\n"
    "  增肌減脂\n"
    "  維持\n"
    "  增肥"
)

ONBOARDING_NEED_INBODY_TEXT = (
    "尚未完成初始設定。\n\n"
    "請先上傳 InBody 體組成報告照片，完成後再選擇健身目標。"
)

ONBOARDING_BLOCKED_TEXT = (
    "請先完成初始設定（InBody ＋ 健身目標），才能使用此功能。\n\n"
    + FITNESS_GOAL_PROMPT
)

# 快速記錄（不呼叫 Vision，省 token）；可經「設定蛋白飲」自訂蛋白飲數值
DEFAULT_QUICK_ITEMS = {
    "蛋白飲": {"calories": 130, "protein": 25, "description": "乳清蛋白飲 一份"},
    "雞蛋": {"calories": 75, "protein": 6, "description": "水煮蛋 一顆"},
    "雞胸肉": {"calories": 165, "protein": 31, "description": "雞胸肉 100g"},
    "碳水": {"calories": 130, "protein": 2, "description": "地瓜 一份（約 150g）"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 使用者狀態管理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class UserState:
    IDLE = "idle"
    WAITING_PURCHASE_PHOTO = "waiting_purchase_photo"
    PURCHASE_REVIEWED = "purchase_reviewed"
    WAITING_INBODY_PHOTO = "waiting_inbody_photo"
    ONBOARDING_WAITING_GOAL = "onboarding_waiting_goal"


# 記憶體內狀態 (生產環境可改用 Redis)
_user_states: dict[str, str] = {}
_user_contexts: dict[str, dict] = {}
_pending_notes: dict[str, dict] = {}
_pending_notes_lock = asyncio.Lock()


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


def parse_fitness_goal(text: str) -> str | None:
    t = _compact_command(text)
    for alias, goal in FITNESS_GOAL_ALIASES.items():
        if t == _compact_command(alias):
            return goal
    for key in FITNESS_GOALS:
        if t == _compact_command(key):
            return key
    return None


def _compact_command(text: str) -> str:
    """指令比對用：全形轉半形、移除所有空白。"""
    t = unicodedata.normalize("NFKC", (text or "").replace("\u3000", " "))
    return re.sub(r"\s+", "", t.strip())


JITAI_ON_COMMANDS = {
    _compact_command(x)
    for x in (
        "開啟體醒", "開啟智能體醒", "開啟JITAI", "開啟jitai體醒",
        "開啟提醒", "開啟智能提醒", "開啟jitai提醒",
    )
}
JITAI_OFF_COMMANDS = {
    _compact_command(x)
    for x in (
        "關閉體醒", "關閉智能體醒", "關閉JITAI", "關閉jitai體醒",
        "關閉提醒", "關閉智能提醒",
    )
}
JITAI_STATUS_COMMANDS = {
    _compact_command(x)
    for x in ("體醒狀態", "查詢體醒", "智能體醒狀態", "提醒狀態", "查詢提醒")
}


def _profile_fitness_goal(profile: dict | None) -> str:
    if profile and profile.get("fitness_goal") in FITNESS_GOALS:
        return profile["fitness_goal"]
    return "減脂"


def leave_photo_wait_if_any(user_id: str) -> None:
    """執行其他功能時離開「等待傳照」狀態，避免下一張照片被誤判流程。"""
    s = get_state(user_id)
    if s in (UserState.WAITING_PURCHASE_PHOTO, UserState.WAITING_INBODY_PHOTO):
        clear_state(user_id)


def _photo_note_window_sec() -> float:
    return float(os.getenv("PHOTO_NOTE_WINDOW_SEC", "10"))


def _extract_note_text(text: str) -> tuple[str | None, bool]:
    """解析「備註／備注／秤重／克數 xxx」，回傳 (內容, 是否為秤重前綴)。"""
    t = text.strip()
    scale_prefixes = ("秤重", "克數", "餐食克數", "餐時克數")
    normal_prefixes = ("備註", "備注", "备注", "註記", "注記")
    for p in scale_prefixes:
        if t.startswith(p):
            body = t[len(p):].lstrip("：: ").strip()
            return (body or None), True
    for p in normal_prefixes:
        if t.startswith(p):
            body = t[len(p):].lstrip("：: ").strip()
            return (body or None), False
    return None, False


_SCALE_NOTE_KEYWORDS = ("秤重", "克數", "餐食克數", "餐時克數")
_SCALE_SKIP_NAMES = frozenset({
    "備註", "備注", "秤重", "克數", "餐食克數", "餐時克數", "約", "大概", "半份", "一份",
    "總重量", "總重", "總共",
})


def _parse_total_weight_from_note(note: str) -> float | None:
    """解析備註中的全餐一次秤重總熟重（公克）。"""
    if not (note or "").strip():
        return None
    m = re.search(
        r"總(?:重量|重|共)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(?:g|克|公克)?",
        note,
        re.I,
    )
    if not m:
        return None
    grams = float(m.group(1))
    if grams <= 0 or grams > 5000:
        return None
    return grams


def _parse_scale_weights_from_note(
    note: str, *, force_scale: bool = False,
) -> list[dict[str, str | float]]:
    """從備註解析廚房秤實測克數，回傳 [{"name": "雞胸", "grams": 150.0}, ...]。"""
    if not (note or "").strip():
        return []
    has_scale_hint = force_scale or any(kw in note for kw in _SCALE_NOTE_KEYWORDS)
    has_unit_hint = bool(re.search(r"(?:\d\s*(?:g|克|公克)\b)", note, re.I))

    items: list[dict[str, str | float]] = []
    seen: set[str] = set()

    def _add(name: str, grams: float) -> None:
        name = name.strip().strip("、，,;； ")
        if not name or name in _SCALE_SKIP_NAMES:
            return
        if re.match(r"^總(?:重量|重|共)", name):
            return
        if re.match(r"^[兩三四五六七八九十半\d]+[個片塊碗盤份杯]", name):
            return
        if grams <= 0 or grams > 5000:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        items.append({"name": name, "grams": grams})

    for m in re.finditer(
        r"([^\d、,，;；\n\r]{1,24}?)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(g|克|公克)?",
        note,
        re.I,
    ):
        name = m.group(1)
        unit = (m.group(3) or "").strip()
        grams = float(m.group(2))
        if unit or has_scale_hint or has_unit_hint:
            _add(name, grams)

    if not items and (has_scale_hint or "、" in note or "，" in note):
        for part in re.split(r"[、,，]", note):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(.+?)\s+(\d+(?:\.\d+)?)\s*(?:g|克|公克)?$", part, re.I)
            if m:
                _add(m.group(1), float(m.group(2)))

    return items


def _format_scale_weights_summary(weights: list[dict[str, str | float]]) -> str:
    return "、".join(f'{w["name"]} {w["grams"]:.0f}g' for w in weights)


def _format_note_ack(note: str, *, force_scale: bool = False) -> str:
    """組裝備註附加成功的回覆文案。"""
    weights = _parse_scale_weights_from_note(note, force_scale=force_scale)
    total = _parse_total_weight_from_note(note)
    lines: list[str] = []
    if weights:
        lines.append(f"已附加秤重備註：{_format_scale_weights_summary(weights)}")
    if total is not None:
        lines.append(f"總重量：{total:.0f}g")
    if weights or total is not None:
        lines.append("（分析將優先採用實測克數）")
        return "\n".join(lines)
    return f"已附加備註：{note}"


def _build_meal_photo_note_prompt(
    user_note: str, *, force_scale: bool = False,
) -> tuple[str, list[dict[str, str | float]], float | None]:
    """組裝照片分析的備註／秤重提示詞。"""
    note = (user_note or "").strip()
    if not note:
        return "", [], None
    total_weight = _parse_total_weight_from_note(note)
    weights = _parse_scale_weights_from_note(note, force_scale=force_scale)
    has_scale_data = bool(weights) or total_weight is not None
    limit = 500 if has_scale_data else 300
    clipped = note[:limit]
    if has_scale_data:
        blocks = [f"使用者備註原文：\n{clipped}"]
        if weights:
            weight_lines = "\n".join(f"- {w['name']}：{w['grams']:.0f} g" for w in weights)
            blocks.append(f"【各品項實測熟重（公克）— 必須採用】\n{weight_lines}")
        if total_weight is not None:
            blocks.append(
                f"【全餐一次秤重總熟重（公克）— 必須採用】\n"
                f"- 總重量：{total_weight:.0f} g"
            )
        rules = [
            "秤重規則：",
            "- 以上為使用者廚房秤實測值，禁止用視覺估份量覆寫。",
        ]
        if weights:
            rules.append("- 各品項克數已給定者，熱量與蛋白質必須依該熟重計算；food_breakdown 標「依據：秤重」。")
        if total_weight is not None:
            if weights:
                rules.append(
                    f"- 總重量 {total_weight:.0f}g 亦為實測值；各品項克數加總應與總重一致"
                    "（必要時僅微調未單獨秤重者）。"
                )
            else:
                rules.append(
                    "- 使用者僅提供全餐總重：依照片辨識品項後，按盤面體積／占比分配各項熟重，"
                    "加總須等於總重量；food_breakdown 標「依據：總重分配」。"
                )
        rules.append("- 照片僅用於核對品項與烹調方式。")
        blocks.append("\n".join(rules))
        return "\n\n" + "\n\n".join(blocks), weights, total_weight
    return f"\n\n使用者補充備註（高優先參考）：\n{clipped}", [], None


async def create_pending_note_window(user_id: str, message_id: str, window_sec: float) -> None:
    """建立此圖片的備註等待視窗。"""
    expire_at = datetime.now(timezone.utc) + timedelta(seconds=window_sec)
    async with _pending_notes_lock:
        _pending_notes[user_id] = {
            "message_id": message_id,
            "expire_at": expire_at,
            "note": "",
            "force_scale": False,
            "event": asyncio.Event(),
        }


async def add_pending_note_if_open(
    user_id: str, note: str, *, force_scale: bool = False,
) -> tuple[bool, str]:
    """若仍在等待視窗內，將備註綁到最近一張圖片。"""
    now = datetime.now(timezone.utc)
    async with _pending_notes_lock:
        item = _pending_notes.get(user_id)
        if not item:
            return False, "目前沒有可附加備註的待分析照片。請先上傳照片。"
        if now > item["expire_at"]:
            _pending_notes.pop(user_id, None)
            sec = int(_photo_note_window_sec())
            return False, f"這張照片的備註視窗已超過 {sec} 秒，請重新上傳照片後再補備註。"
        item["note"] = note
        item["force_scale"] = force_scale
        ev = item.get("event")
        if isinstance(ev, asyncio.Event):
            ev.set()
    return True, _format_note_ack(note, force_scale=force_scale)


async def wait_and_take_pending_note(
    user_id: str, message_id: str, window_sec: float,
) -> tuple[str, bool]:
    """等待最多 window_sec 秒接收備註，回傳 (備註內容, 是否為秤重前綴)。"""
    async with _pending_notes_lock:
        item = _pending_notes.get(user_id)
        if not item or item.get("message_id") != message_id:
            return "", False
        expire_at = item["expire_at"]
        ev = item.get("event")
    remaining = max(0.0, (expire_at - datetime.now(timezone.utc)).total_seconds())
    wait_for = min(window_sec, remaining)
    if wait_for > 0 and isinstance(ev, asyncio.Event):
        try:
            await asyncio.wait_for(ev.wait(), timeout=wait_for)
        except asyncio.TimeoutError:
            pass
    async with _pending_notes_lock:
        latest = _pending_notes.get(user_id)
        if not latest or latest.get("message_id") != message_id:
            return "", False
        note = str(latest.get("note") or "").strip()
        force_scale = bool(latest.get("force_scale"))
        _pending_notes.pop(user_id, None)
        return note, force_scale


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


class UserFacingError(Exception):
    """可直接顯示給使用者的流程錯誤。"""


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

    timeout_sec = float(os.getenv("OPENAI_TIMEOUT_SEC", "90"))
    max_retries = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    timeout = httpx.Timeout(timeout_sec, connect=20.0)

    resp: httpx.Response | None = None
    for i in range(max_retries):
        try:
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
                if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                    if i < max_retries - 1:
                        await asyncio.sleep(0.8 * (i + 1))
                        continue
                    raise UserFacingError("AI 服務目前忙碌或不穩定，請稍後再試一次。")
            resp.raise_for_status()
            break
        except httpx.TimeoutException:
            logger.warning("OpenAI Vision 呼叫逾時（第 %s/%s 次）", i + 1, max_retries)
            if i < max_retries - 1:
                await asyncio.sleep(0.8 * (i + 1))
                continue
            raise UserFacingError("圖片分析等待過久，請重新傳一次照片後再試。")
        except httpx.RequestError as e:
            logger.warning("OpenAI Vision 網路錯誤（第 %s/%s 次）: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                await asyncio.sleep(0.8 * (i + 1))
                continue
            raise UserFacingError("目前與 AI 服務連線不穩，請稍後再試。")

    if resp is None:
        raise UserFacingError("AI 服務暫時無回應，請稍後再試。")

    raw = _openai_extract_message_text(resp.json())
    if not raw:
        return ""

    parsed = _try_parse_json_response(raw)
    if parsed is not None:
        return parsed
    return raw


PROMPT_JITAI_NUDGE = """你是減脂／增肌飲食教練，撰寫「此刻可執行」的 JITAI 智能提醒訊息。

原則：
- 不要只播報數字；必須給 2～3 個可馬上採行的食物組合（每項一行，含約略蛋白質與熱量）
- 優先便利、超商／外食可取得、高蛋白密度（例：希臘優格＋花生醬、酪梨＋水煮蛋、乳清＋香蕉、雞胸即食包＋地瓜）
- 結合動機與下一步：簡短點出處境 → 具體選項 → 一句可執行動作
- 口吻專業、冷靜、一針見血；不使用 emoji；繁體中文
- 總長不超過 380 字；勿輸出 JSON 或標題符號"""


async def call_openai_jitai_nudge(user_prompt: str) -> str:
    """以較輕量模型產生可執行的提醒文案（純文字）。"""
    if not OPENAI_API_KEY:
        return ""
    payload = {
        "model": JITAI_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT_JITAI_NUDGE},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 600,
        "temperature": 0.55,
    }
    timeout = httpx.Timeout(45.0, connect=15.0)
    try:
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
            logger.warning("JITAI OpenAI HTTP %s", resp.status_code)
            return ""
        resp.raise_for_status()
        return _openai_extract_message_text(resp.json())
    except Exception as e:
        logger.warning("JITAI OpenAI 呼叫失敗: %s", e)
        return ""


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

    timeout_sec = float(os.getenv("OPENAI_TIMEOUT_SEC", "90"))
    max_retries = max(1, int(os.getenv("OPENAI_MAX_RETRIES", "3")))
    timeout = httpx.Timeout(timeout_sec, connect=20.0)

    resp: httpx.Response | None = None
    for i in range(max_retries):
        try:
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
                if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                    if i < max_retries - 1:
                        await asyncio.sleep(0.8 * (i + 1))
                        continue
                    raise UserFacingError("AI 服務目前忙碌或不穩定，請稍後再試一次。")
            resp.raise_for_status()
            break
        except httpx.TimeoutException:
            logger.warning("OpenAI Text 呼叫逾時（第 %s/%s 次）", i + 1, max_retries)
            if i < max_retries - 1:
                await asyncio.sleep(0.8 * (i + 1))
                continue
            raise UserFacingError("目前文字分析等待過久，請稍後再試。")
        except httpx.RequestError as e:
            logger.warning("OpenAI Text 網路錯誤（第 %s/%s 次）: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                await asyncio.sleep(0.8 * (i + 1))
                continue
            raise UserFacingError("目前與 AI 服務連線不穩，請稍後再試。")

    if resp is None:
        raise UserFacingError("AI 服務暫時無回應，請稍後再試。")

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
    timeout = httpx.Timeout(float(os.getenv("LINE_IMAGE_TIMEOUT_SEC", "20")), connect=10.0)
    max_retries = max(1, int(os.getenv("LINE_IMAGE_MAX_RETRIES", "3")))
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"},
                )
            if resp.status_code == 404:
                raise UserFacingError("圖片已過期或無法取得，請重新傳送一張照片。")
            if resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500:
                logger.warning(
                    "LINE 圖片下載暫時失敗 HTTP %s（第 %s/%s 次）",
                    resp.status_code,
                    i + 1,
                    max_retries,
                )
                if i < max_retries - 1:
                    await asyncio.sleep(0.6 * (i + 1))
                    continue
            resp.raise_for_status()
            return resp.content
        except httpx.TimeoutException:
            logger.warning("LINE 圖片下載逾時（第 %s/%s 次）", i + 1, max_retries)
            if i < max_retries - 1:
                await asyncio.sleep(0.6 * (i + 1))
                continue
            raise UserFacingError("下載圖片逾時，請重新傳送照片。")
        except httpx.RequestError as e:
            logger.warning("LINE 圖片下載網路錯誤（第 %s/%s 次）: %s", i + 1, max_retries, e)
            if i < max_retries - 1:
                await asyncio.sleep(0.6 * (i + 1))
                continue
            raise UserFacingError("目前無法下載圖片，請稍後再試。")
    raise UserFacingError("目前無法取得圖片，請重新傳送照片。")


def compress_image_bytes_to_jpeg_base64(data: bytes, max_side: int = 1280, quality: int = 80) -> str:
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
    compress_timeout = float(os.getenv("IMAGE_COMPRESS_TIMEOUT_SEC", "30"))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(compress_image_bytes_to_jpeg_base64, raw),
            timeout=compress_timeout,
        )
    except asyncio.TimeoutError as e:
        raise UserFacingError("圖片處理耗時過長，請重新傳送照片。") from e


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 提示詞定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_MEAL_ANALYSIS_CORE = """# 角色定位
你是擁有營養學博士學位與執照的極度資深臨床營養師。作風嚴謹、專業、實事求是，專為減脂客戶提供不容一絲誤差的飲食精準分析。
溝通風格：口吻專業、冷靜、一針見血，不使用任何 emoji，一律使用繁體中文。

# 核心原則
1. 熱量（Calories）— 減脂保守上緣：推導出區間（Minimum ~ Maximum）後，輸出【必須取 Maximum 上緣】（或 ≥ 上緣 95%）。嚴禁中位數、平均、折衷。
2. 蛋白質（Protein）— 精準中立：逐項依食材與熟重（或包裝標示）專業估算後加總；取合理最佳估計值即可，【不要】刻意高估或採區間上緣。依計算結果如實輸出即可，無需刻意湊成 5 或 10 的倍數，精度至少 1 g（必要時 0.1 g）。
3. 份量與隱形熱量（僅影響熱量）：份量模糊一律從大；外食 +25%；醬汁、勾芡、用油、隱藏糖從嚴計入。炒菜類熱量再加約 20% 用油係數。

# 分析步驟（內部必須依序完成，不可跳步）

## 步驟一｜視覺描述（先描述、再推論）
逐一列出畫面中每個容器／包裝／盤裝食物，各寫：
- 外觀顏色、形狀、質地（勿用籠統詞如「滷味」「炒菜」代替描述）
- 是否可見包裝文字或品牌
- 烹調狀態：生／熟／冷藏未烹調／已烹調
禁止在未描述前就斷定食物名稱。深色炒物可能是豆干菇類，不一定是香腸滷味；袋裝葉菜可能是生鮮而非燙青菜。

## 步驟二｜包裝文字辨識（有包裝則優先）
若畫面有食品包裝，優先讀取並採用：
- 品名、淨重（g）、每份或每 100g 營養標示（熱量、蛋白質）
包裝可讀數據優先於純視覺猜測；蛋白質依標示與實際食用量精準計算；熱量在包裝或估算基礎上依減脂原則從嚴上修。

## 步驟三｜分類、份量與生熟（What + How Much）
- 依步驟一、二推論品項與熟重（g）或份數；標註每項信心（高／中／低）。
- 生鮮冷藏、仍在原包裝、色澤偏生（如粉白雞胸）→ 標記「未烹調／待烹調」；若無法確認使用者已食用，在 breakdown 註明「假設已烹調並計入」或列入 user_confirm_prompt。
- 有照片時以餐具為比例尺估體積；僅文字時依描述，缺漏從大。

## 步驟四｜烹調與隱形熱量（How Cooked）
- Level 1（無油／清蒸）：+0g 油
- Level 2（家常炒）：+3~5g 油
- Level 3（外食大火炒／勾芡）：+8~12g 油，或熱量 +20% 用油係數
- Level 4（油炸／酥皮）：食材總重 15%~20% 併入油脂熱量

## 步驟五｜加總與輸出
- 逐項估算後加總：calories 取總熱量區間 Maximum；protein 取逐項精準加總（非上緣）。

# 輸出格式（僅 JSON，無其他文字）
{
  "calories": 數字,
  "protein": 數字,
  "description": "一句話餐點摘要（含生熟假設若適用）",
  "food_breakdown": "Markdown 簡潔版：每項「名稱｜約XXg｜生/熟｜依據：包裝/視覺」",
  "recognition_confidence": "高或中或低",
  "uncertain_items": "不確定項目與採用的保守假設，無則空字串",
  "user_confirm_prompt": "建議使用者確認的一句話，無則空字串",
  "estimation_note": "熱量採區間上緣、蛋白質精準估算等關鍵假設"
}"""

PROMPT_MEAL_ANALYSIS = (
    PROMPT_MEAL_ANALYSIS_CORE
    + """

# 本任務：照片餐點分析
使用者提供食物照片。必須先完成步驟一（視覺描述）與步驟二（包裝文字），再分類；忽略手部與無關背景。

# 秤重備註（若 user 訊息含實測克數或總重量）
若使用者提供各品項克數或「總重量／總重」全餐一次秤重，份量【僅能】採用實測值；
僅有總重量時，依照片品項占比分配各項熟重，加總須等於總重。視覺估計不得覆寫秤重。"""
)

PROMPT_MEAL_FROM_TEXT = (
    PROMPT_MEAL_ANALYSIS_CORE
    + """

# 本任務：文字餐點分析
使用者以中文描述餐點（無照片）。跳過步驟一視覺與步驟二包裝，從步驟三依文字執行；缺漏份量從大、外食 +25%。"""
)

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

PROMPT_INBODY_ANALYSIS = """你是 InBody 體組成報告的數據提取專家。分析圖片並精確提取數據。

規則：
- 若有手部或背景入鏡，請忽略，專注在報告紙張／螢幕上的數字與圖表
- 盡可能辨識所有數值
- 無法辨識、或不符合驗證規則的欄位，設為 null
- 不要使用任何 emoji
- 使用繁體中文
- 只回傳 JSON，不要任何其他說明文字

【重要欄位定義，務必區分】
- weight：體重（kg），例如 136.8
- body_fat_percentage：體脂率（%），例如 34.7
- smm：骨骼肌重 SMM（kg）
  - 在「肌肉脂肪分析」區塊的「骨骼肌重」列
  - 常見約 30~70 kg，通常不應超過體重的 50%
- lbm：除脂體重 LBM（kg）
  - 在「研究參數」區塊的「除脂體重／除脂體體重」
  - 常見約 60~100 kg
- bmr（基礎代謝率）：右側「研究參數」區塊，標籤「基礎代謝率／BMR」，單位 kcal，常見 1200~3500
- tdee（建議的熱量攝取）— 極易讀錯，請嚴格遵守：
  - 僅填標籤為「建議的熱量攝取」或「Recommended Calorie Intake」的那一欄
  - 該欄位在「研究參數」區塊【最下方／最後一行】，通常是該區塊【最大的 kcal 數字】
  - 必須明顯大於 BMR（常見 ≥ BMR × 1.5，例如 BMR 2298 時 TDEE 常為 3500~5500）
  - 勿將「1日エネルギー消費量」、活動量中間值、或其他非「建議的熱量攝取」的 kcal 誤填為 tdee
  - 若同區塊有多個 kcal，請全部列入 research_kcal_values，tdee 填其中最大且 > BMR × 1.4 者
- body_water：體水分（kg）
- visceral_fat（內臟脂肪級別）— 極易讀錯，請嚴格遵守：
  - 讀「內臟脂肪級別／Visceral Fat Level」【文字標籤同一行、標籤右側】的數字（例：20）
  - 該數字通常印在標籤旁，與長條圖【末端數值】一致；若末端在 20 附近，必須填 20
  - 禁止讀取：長條圖左/下方刻度軸（5、10、15…）、網格線數字、顏色區間標記
  - 常見錯誤：把刻度軸上的「10」誤當成級別；若體脂偏高且條形接近高風險區，級別通常 ≥ 15
- waist_hip_ratio：腰臀比（小數，例如 1.01）
- test_date：檢測日期（YYYY-MM-DD）

【版面與生理合理性】
- 若讀出 bmr > tdee，代表欄位可能對調，請重新對照報告後再填
- research_kcal_values 必須完整列出研究參數區所有 kcal（例：[2298, 4453]），不可遺漏最大者

【驗證規則，違反時設為 null】
1. smm 與 lbm 皆存在時，smm 必須 < lbm，且 smm < weight × 0.5
2. weight 與 body_fat_percentage 與 lbm 皆存在時，lbm 應接近 weight × (1 - body_fat_percentage/100)，誤差超過 ±3 kg 視為可疑
3. bmr 與 tdee 皆存在時，tdee 必須 > bmr × 1.4（建議熱量通常至少為 BMR 的 1.5 倍）
4. bmr 約 900~3800；tdee 約 1100~7500；visceral_fat 為 1~30 的整數
5. research_kcal_values 須列出研究參數區塊內所有可見 kcal 數字（由小到大）

請嚴格以下列 JSON 格式回覆：
{
  "weight": 數字或null,
  "body_fat_percentage": 數字或null,
  "smm": 數字或null,
  "lbm": 數字或null,
  "muscle_mass": 數字或null,
  "bmr": 數字或null,
  "tdee": 數字或null,
  "research_kcal_values": [數字,...] 或 [],
  "body_water": 數字或null,
  "visceral_fat": 數字或null,
  "waist_hip_ratio": 數字或null,
  "test_date": "YYYY-MM-DD或null"
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


def _format_meal_macro_g(value: float) -> str:
    """蛋白質顯示：保留 1 位小數（若為整數則不顯示 .0）。"""
    if abs(value - round(value)) < 0.05:
        return f"{round(value):.0f}"
    return f"{value:.1f}"


def _meal_text_field(result: dict, key: str) -> str:
    """安全取出餐點分析 JSON 文字欄位（模型偶爾回傳非字串）。"""
    val = result.get(key)
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    return str(val).strip()


def _meal_analysis_detail_lines(
    result: dict,
    cal: float,
    pro: float,
    desc: str,
    scale_weights: list[dict[str, str | float]] | None = None,
    total_weight_g: float | None = None,
) -> list[str]:
    """組裝 AI 回傳的 Markdown 分析區塊與數值摘要。"""
    note = _meal_text_field(result, "estimation_note")
    breakdown = _meal_text_field(result, "food_breakdown")
    confidence = _meal_text_field(result, "recognition_confidence")
    uncertain = _meal_text_field(result, "uncertain_items")
    confirm = _meal_text_field(result, "user_confirm_prompt")

    lines = [
        f"食物內容：\n{desc}",
        "",
    ]
    if scale_weights:
        lines.append(f"已採用秤重：{_format_scale_weights_summary(scale_weights)}")
    if total_weight_g is not None:
        lines.append(f"總重量：{total_weight_g:.0f}g（一次秤重）")
    if scale_weights or total_weight_g is not None:
        lines.append("")
    if breakdown:
        lines.extend(["食物拆解明細（簡潔版）", breakdown, ""])
    if confidence:
        lines.append(f"辨識信心：{confidence}")
    if uncertain:
        lines.append(f"不確定項目：{uncertain}")
    if confirm:
        lines.append(f"建議確認：{confirm}")
    lines.extend([
        "",
        "營養數據（熱量：減脂保守上緣；蛋白質：精準估算）：",
        f"熱量：{cal:.0f} kcal",
        f"蛋白質：{_format_meal_macro_g(pro)} g",
    ])
    if note:
        lines.append(f"說明：{note}")
    return lines


def get_user_targets(user_id: str) -> dict:
    """取得使用者的每日目標。

    只要 profile 有體重，一律用 calculate_targets 依體重／體脂／BMR／TDEE 重算，
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
                    profile.get("tdee"),
                    fitness_goal=_profile_fitness_goal(profile),
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


def _safe_float(v, default: float = 0.0) -> float:
    """容錯轉數字：可處理 950、約950、950 kcal、?。"""
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


def _optional_positive_float(val) -> float | None:
    """將 JSON／字串轉成正數浮點；無效或 missing 時為 None。"""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "null", "none", "?"):
            return None
    if isinstance(val, (int, float)):
        x = float(val)
    else:
        s = str(val).strip().replace(",", "")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if not m:
            return None
        try:
            x = float(m.group(0))
        except ValueError:
            return None
    if x <= 0:
        return None
    return x


def _normalize_inbody_visceral_fat(vf, body_fat_pct=None) -> float | None:
    """內臟脂肪級別（1~30），過濾刻度誤讀；高體脂時偏低值視為可疑。"""
    v = _optional_positive_float(vf)
    if v is None:
        return None
    v_int = int(round(v))
    if not (1 <= v_int <= 30):
        logger.warning("內臟脂肪級別超出合理範圍（1-30），棄用：%s", vf)
        return None
    bf = _optional_positive_float(body_fat_pct)
    if bf is not None and bf >= 28.0 and v_int <= 12:
        logger.warning(
            "高體脂(%.1f%%)但內臟脂肪級別偏低(%s)，疑似讀到圖表刻度，棄用",
            bf,
            v_int,
        )
        return None
    return float(v_int)


def _infer_bmr_tdee_from_kcal_candidates(
    candidates: set[float],
) -> tuple[float | None, float | None]:
    """從研究參數區多個 kcal 推斷 BMR（較小）與 TDEE（較大）。"""
    vals = sorted(c for c in candidates if 900.0 <= c <= 7500.0)
    if len(vals) < 2:
        return (vals[0], None) if len(vals) == 1 else (None, None)
    t = vals[-1]
    below = [v for v in vals[:-1] if v < t / 1.35]
    b = max(below) if below else vals[-2]
    if t > b * 1.35:
        return b, t
    return None, None


def _build_inbody_summary(
    *,
    weight,
    bf,
    muscle,
    bmr: int | None,
    tdee: int | None,
    visceral_fat: float | None,
    used_pal_fallback: bool,
) -> str:
    """依驗證後數值組摘要，不使用 LLM 自由文字。"""
    parts: list[str] = []
    if weight is not None:
        parts.append(f"體重 {weight} kg")
    if bf is not None:
        parts.append(f"體脂率 {bf} %")
    if muscle is not None:
        parts.append(f"骨骼肌 {muscle} kg")
    if bmr is not None and tdee is not None:
        parts.append(f"基礎代謝 {bmr} kcal，建議每日攝取 {tdee} kcal")
    elif used_pal_fallback:
        parts.append("BMR/TDEE 未能可靠辨識，每日熱量目標已改用 PAL 推算")
    elif bmr is not None:
        parts.append(f"基礎代謝 {bmr} kcal（建議攝取未能可靠辨識）")
    elif tdee is not None:
        parts.append(f"建議每日攝取 {tdee} kcal（基礎代謝未能可靠辨識）")
    if visceral_fat is not None:
        parts.append(f"內臟脂肪級別 {int(visceral_fat)}")
    if not parts:
        return "部分數值未能可靠辨識，請確認照片清晰後重新上傳。"
    return "。".join(parts) + "。"


def _resolve_inbody_tdee(
    bmr,
    tdee,
    kcal_candidates,
    weight_kg: float | None = None,
) -> tuple[float | None, float | None]:
    """從 BMR、OCR TDEE 與研究參數區所有 kcal 候選值，解析可信 BMR/TDEE。"""
    b = _optional_positive_float(bmr)
    t_ocr = _optional_positive_float(tdee)

    candidates: set[float] = set()
    if b is not None:
        candidates.add(b)
    if t_ocr is not None:
        candidates.add(t_ocr)
    raw_list = kcal_candidates if isinstance(kcal_candidates, list) else []
    for item in raw_list:
        c = _optional_positive_float(item)
        if c is not None and 900.0 <= c <= 7500.0:
            candidates.add(c)

    w = _optional_positive_float(weight_kg)
    min_ratio = 1.4

    if b is not None:
        valid_tdee = [c for c in candidates if c > b * min_ratio]
        if valid_tdee:
            best = max(valid_tdee)
            if t_ocr is None or best > t_ocr * 1.05:
                if t_ocr is not None and best > t_ocr * 1.08:
                    logger.warning(
                        "TDEE 修正：OCR tdee=%s → 採研究參數區最大合理值=%s（bmr=%s）",
                        t_ocr,
                        best,
                        b,
                    )
                t_ocr = best

    ib, it = _infer_bmr_tdee_from_kcal_candidates(candidates)
    if ib is not None and b is None:
        logger.warning("BMR 由 research_kcal_values 推斷：%s", ib)
        b = ib
    if it is not None:
        if t_ocr is None or t_ocr < (b or ib or 0) * min_ratio:
            logger.warning(
                "TDEE 由 research_kcal_values 推斷：%s（原 tdee=%s）",
                it,
                t_ocr,
            )
            t_ocr = it

    b_out, t_out = _normalize_inbody_bmr_tdee(b, t_ocr, weight_kg=w)
    return b_out, t_out


def _normalize_inbody_bmr_tdee(
    bmr,
    tdee,
    weight_kg: float | None = None,
) -> tuple[float | None, float | None]:
    """清理 BMR/TDEE：對調偵測、合理範圍、比例與體重交叉驗證。"""
    b = _optional_positive_float(bmr)
    t = _optional_positive_float(tdee)

    if b is not None and t is not None and b > t:
        logger.warning("BMR/TDEE 疑似對調，自動交換：bmr=%s tdee=%s", b, t)
        b, t = t, b

    if b is not None and not (900.0 <= b <= 3800.0):
        logger.warning("BMR 超出合理範圍，棄用：%s", b)
        b = None
    if t is not None and not (1100.0 <= t <= 7500.0):
        logger.warning("TDEE 超出合理範圍，棄用：%s", t)
        t = None

    if b is not None and t is not None:
        ratio = t / b
        if ratio < 1.4:
            logger.warning(
                "TDEE/BMR 比例過低(%.2f)，疑似誤讀中間欄位，棄用 TDEE：bmr=%s tdee=%s",
                ratio,
                b,
                t,
            )
            t = None
        elif ratio > 2.8:
            logger.warning(
                "TDEE/BMR 比例過高(%.2f)，棄用 TDEE：bmr=%s tdee=%s",
                ratio,
                b,
                t,
            )
            t = None

    w = _optional_positive_float(weight_kg)
    if w is not None:
        bmr_floor = max(1200.0, 12.0 * w)
        tdee_floor = max(2000.0, 26.0 * w)
        if b is not None and w >= 100 and b < bmr_floor:
            logger.warning(
                "BMR 與體重不匹配，棄用：bmr=%s weight=%s（預期至少約 %.0f）",
                b,
                w,
                bmr_floor,
            )
            b = None
        if t is not None and w >= 90:
            if b is not None:
                tdee_floor = max(tdee_floor, b * 1.45)
            if t < tdee_floor:
                logger.warning(
                    "TDEE 與體重/BMR 不匹配，棄用：tdee=%s weight=%s（預期至少約 %.0f）",
                    t,
                    w,
                    tdee_floor,
                )
                t = None

    return b, t


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
            "目前預設：130 kcal／20 g"
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


async def handle_meal_photo(
    user_id: str, message_id: str, user_note: str = "", *, force_scale: bool = False,
) -> str:
    """處理食物照片：分析＋記錄＋回傳摘要。"""
    image_b64 = await get_line_image_base64(message_id)
    note_prompt, scale_weights, total_weight_g = _build_meal_photo_note_prompt(
        user_note, force_scale=force_scale,
    )

    result = await call_openai_vision(
        system_prompt=PROMPT_MEAL_ANALYSIS,
        user_prompt=(
            "請分析這份餐點照片。目標為減脂：先描述畫面再推論品項；"
            "有包裝則優先讀標示；區分生/熟；calories 取區間 Maximum，"
            "protein 精準中立勿上緣；填寫 food_breakdown 與信心欄位。"
            f"{note_prompt}"
        ),
        image_base64=image_b64,
        image_detail="high",
    )

    if isinstance(result, OpenAIUserNotice):
        return str(result)
    if isinstance(result, str):
        return f"分析失敗，請重新拍照。\n\n原始回應：\n{result[:200]}"

    cal = _safe_float(result.get("calories"), 0.0)
    pro = _safe_float(result.get("protein"), 0.0)
    desc = result.get("description", "無法辨識")

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
        *_meal_analysis_detail_lines(
            result, cal, pro, desc,
            scale_weights=scale_weights or None,
            total_weight_g=total_weight_g,
        ),
    ]
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
            "使用者原文如下，請依步驟三～五推導並回傳 JSON：\n"
            f"{user_said.strip()}\n\n"
            "目標為減脂：calories 取區間 Maximum，protein 精準中立勿上緣；"
            "填寫 food_breakdown 與信心欄位。"
        ),
    )

    if isinstance(result, OpenAIUserNotice):
        return str(result)
    if isinstance(result, str):
        return f"文字分析失敗，請寫清楚一點再試。\n\n原始回應：\n{result[:200]}"

    cal = _safe_float(result.get("calories"), 0.0)
    pro = _safe_float(result.get("protein"), 0.0)
    desc = result.get("description", "無法辨識")

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
        *_meal_analysis_detail_lines(result, cal, pro, desc),
    ]
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

    # 容錯正規化，避免後續「買了」寫 DB 時因欄位格式炸掉
    cleaned = dict(result)
    cleaned["calories"] = _safe_float(cleaned.get("calories"), 0.0)
    cleaned["protein"] = _safe_float(cleaned.get("protein"), 0.0)
    cleaned["carbs"] = _safe_float(cleaned.get("carbs"), 0.0)
    cleaned["fat"] = _safe_float(cleaned.get("fat"), 0.0)
    cleaned["sugar"] = _safe_float(cleaned.get("sugar"), 0.0)
    if not isinstance(cleaned.get("grades"), dict):
        cleaned["grades"] = {}

    # 儲存分析結果到 context，等使用者決定
    set_state(user_id, UserState.PURCHASE_REVIEWED, context=cleaned)

    g = cleaned.get("grades", {})
    if not isinstance(g, dict):
        g = {}

    lines = [
        f"購買前分析報告",
        f"{'=' * 24}",
        f"品項：{cleaned.get('name', '未知')}",
        "",
        f"營養數據（估算）：",
        f"  熱量：{cleaned.get('calories', '?')} kcal",
        f"  蛋白質：{cleaned.get('protein', '?')} g",
        f"  碳水化合物：{cleaned.get('carbs', '?')} g",
        f"  脂肪：{cleaned.get('fat', '?')} g",
        f"  糖：{cleaned.get('sugar', '?')} g",
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
        f"  {cleaned.get('timing', '無特別建議')}",
        "",
        f"{'=' * 24}",
        f"結論：{cleaned.get('verdict', '')}",
        "",
        f"更健康的替代選項：",
    ]

    for i, alt in enumerate(cleaned.get("alternatives", []), 1):
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
        user_prompt=(
            "請分析這張 InBody 體組成報告，提取所有可辨識的數據。\n"
            "【TDEE】只填「建議的熱量攝取」；research_kcal_values 須列出該區【全部】kcal（含最大者）。\n"
            "【內臟脂肪】只讀標籤同行右側數字（非圖表刻度）；例：標籤旁印 20 就填 20。"
        ),
        image_base64=image_b64,
        image_detail="high",
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
    smm_raw = _optional_positive_float(result.get("smm"))
    lbm_raw = _optional_positive_float(result.get("lbm"))
    legacy_muscle_raw = _optional_positive_float(result.get("muscle_mass"))

    # 優先使用 SMM；若缺失則回退舊欄位 muscle_mass
    muscle = smm_raw if smm_raw is not None else legacy_muscle_raw

    # SMM/LBM 合理性檢查，避免把 LBM 當成骨骼肌重
    weight_f = _optional_positive_float(weight)
    if muscle is not None and lbm_raw is not None and muscle > lbm_raw * 0.6:
        logger.warning("SMM 疑似誤讀為 LBM：smm=%s lbm=%s", muscle, lbm_raw)
        muscle = None
    if muscle is not None and weight_f is not None and muscle > weight_f * 0.5:
        logger.warning("SMM 超過體重 50%%，視為誤讀：smm=%s weight=%s", muscle, weight_f)
        muscle = None

    bf_f = _optional_positive_float(bf)
    visceral_fat = _normalize_inbody_visceral_fat(result.get("visceral_fat"), bf_f)

    bmr_f, tdee_f = _resolve_inbody_tdee(
        result.get("bmr"),
        result.get("tdee"),
        result.get("research_kcal_values"),
        weight_kg=weight_f,
    )
    bmr = int(round(bmr_f)) if bmr_f is not None else None
    tdee = int(round(tdee_f)) if tdee_f is not None else None
    used_pal_fallback = bmr is None or tdee is None
    summary = _build_inbody_summary(
        weight=weight,
        bf=bf,
        muscle=muscle,
        bmr=bmr,
        tdee=tdee,
        visceral_fat=visceral_fat,
        used_pal_fallback=used_pal_fallback,
    )
    vf_display = int(visceral_fat) if visceral_fat is not None else "?"

    was_onboarded = db.is_onboarded(user_id)
    profile_before = db.get_user_profile(user_id)
    goal_for_calc = _profile_fitness_goal(profile_before) if was_onboarded else "減脂"
    new_targets = calculate_targets(weight, bf, bmr, tdee, fitness_goal=goal_for_calc)

    db.upsert_user_profile(
        user_id=user_id,
        weight=weight,
        body_fat=bf,
        muscle_mass=muscle,
        bmr=bmr,
        tdee=tdee,
        calorie_target=new_targets["calories"] if was_onboarded else None,
        protein_target=new_targets["protein"] if was_onboarded else None,
        onboarding_complete=1 if was_onboarded else 0,
    )

    pending_goal = db.needs_fitness_goal(user_id)
    if pending_goal:
        set_state(user_id, UserState.ONBOARDING_WAITING_GOAL)
    else:
        clear_state(user_id)
        db.upsert_user_profile(
            user_id=user_id,
            calorie_target=new_targets["calories"],
            protein_target=new_targets["protein"],
        )

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
            "tdee": tdee,
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
        f"  基礎代謝（BMR）：{bmr or '?'} kcal",
        f"  建議攝取／TDEE：{tdee or '?'} kcal",
        f"  體水分：{result.get('body_water', '?')} kg",
        f"  內臟脂肪：{vf_display}",
        f"  檢測日期：{result.get('test_date', '?')}",
        "",
        summary,
    ]
    if pending_goal:
        lines.extend([
            "",
            "=" * 24,
            "InBody 已記錄。請完成最後一步：",
            FITNESS_GOAL_PROMPT,
        ])
    else:
        lines.extend([
            "",
            "=" * 24,
            "目標已自動調整：",
            f"  健身目標：{goal_for_calc}",
            f"  每日熱量目標：{new_targets['calories']:.0f} kcal"
            + ("（PAL 推算）" if used_pal_fallback else ""),
            f"  每日蛋白質目標：{new_targets['protein']:.0f} g",
        ])

    return "\n".join(lines)


def calculate_targets(
    weight,
    body_fat,
    bmr,
    tdee_ocr=None,
    fitness_goal: str = "減脂",
) -> dict:
    """
    每日熱量／蛋白質目標。依 fitness_goal 調整熱量赤字或盈余；蛋白質依 LBM 估算。
    """
    if not weight:
        return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}

    w = float(weight)
    bf = float(body_fat) if body_fat is not None else None
    goal = fitness_goal if fitness_goal in FITNESS_GOALS else "減脂"

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

    # PAL：阻力訓練＋日常活動（僅在無報告 TDEE 時用於推算 TDEE）
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

    tdee_from_pal = bmr_used * pal
    use_device_tdee = False
    tdee = tdee_from_pal
    t_ocr = _optional_positive_float(tdee_ocr)
    bmr_ocr = _optional_positive_float(bmr)
    if t_ocr is not None:
        ref_bmr = bmr_used
        if bmr_ocr is not None and 900.0 <= bmr_ocr <= 3800.0:
            ref_bmr = bmr_ocr
        low = max(1100.0, ref_bmr * 1.45)
        high = min(7500.0, ref_bmr * 1.8)
        if t_ocr > ref_bmr * 1.4 and low <= t_ocr <= high:
            tdee = t_ocr
            use_device_tdee = True
        else:
            logger.warning(
                "Device TDEE 未通過驗證，改用 PAL：t_ocr=%s ref_bmr=%.1f "
                "bmr_used=%.1f low=%.1f high=%.1f",
                t_ocr,
                ref_bmr,
                bmr_used,
                low,
                high,
            )

    morton_cap = 1.62 * w
    helms_lo = 2.3 * lbm
    helms_hi_lbm = 3.1 * lbm
    helms_hi = min(helms_hi_lbm, morton_cap)
    if helms_hi >= helms_lo:
        pro_mid = (helms_lo + helms_hi) / 2.0
        pro_high = helms_hi
    else:
        pro_mid = min(morton_cap, max(helms_lo, 2.05 * lbm))
        pro_high = pro_mid

    if w >= 120:
        abs_deficit_cap = 1200.0
    elif w >= 90:
        abs_deficit_cap = 1000.0
    else:
        abs_deficit_cap = 800.0

    if goal == "減脂":
        if use_device_tdee:
            if bf is None:
                d_pct = 0.25
            elif bf >= 30:
                d_pct = 0.33
            elif bf >= 25:
                d_pct = 0.30
            elif bf >= 18:
                d_pct = 0.27
            else:
                d_pct = 0.22
        else:
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
        if use_device_tdee:
            deficit_cap = min(0.35 * tdee, abs_deficit_cap)
        else:
            deficit_cap = min(1100.0, 0.35 * tdee)
        deficit = min(deficit_rel, deficit_cap)
        cal = tdee - deficit
        pro = pro_mid
    elif goal == "增肌":
        surplus = min(tdee * 0.12, 450.0)
        cal = tdee + max(200.0, surplus)
        pro = pro_high
    elif goal == "增肌減脂":
        deficit = min(tdee * 0.12, abs_deficit_cap * 0.85)
        cal = tdee - max(250.0, deficit)
        pro = pro_high
    elif goal == "維持":
        cal = tdee
        pro = pro_mid
    elif goal == "增肥":
        surplus = min(tdee * 0.18, 650.0)
        cal = tdee + max(280.0, surplus)
        pro = pro_mid
    else:
        cal = tdee
        pro = pro_mid

    cal = max(1400.0, cal)
    cal = min(cal, 5500.0)
    if goal in ("減脂", "增肌減脂"):
        cal = min(cal, tdee - 200.0)

    pro = int(max(120, min(250, round(pro))))
    return {"calories": int(round(cal)), "protein": pro}


async def handle_fitness_goal_selection(user_id: str, goal: str) -> str:
    """完成 onboarding：寫入健身目標並重算營養目標。"""
    profile = db.get_user_profile(user_id)
    if not profile or not profile.get("last_inbody_date"):
        set_state(user_id, UserState.IDLE)
        return ONBOARDING_NEED_INBODY_TEXT

    w = profile.get("weight")
    targets = calculate_targets(
        w,
        profile.get("body_fat_percentage"),
        profile.get("bmr"),
        profile.get("tdee"),
        fitness_goal=goal,
    )
    db.complete_onboarding(
        user_id,
        goal,
        targets["calories"],
        targets["protein"],
    )
    clear_state(user_id)

    return (
        f"初始設定完成\n"
        f"================\n\n"
        f"健身目標：{goal}\n"
        f"每日熱量目標：{targets['calories']:.0f} kcal\n"
        f"每日蛋白質目標：{targets['protein']:.0f} g\n\n"
        "之後可直接拍照記錄餐點，或輸入「說明」查看指令。"
    )


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
# JITAI 智能提醒（需手動開啟；門檻觸發 + 窗格收尾最後通牒）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

JITAI_CHECKPOINTS = {
    "lunch": {
        "hour": 14,
        "minute": 0,
        "slot": "jitai_lunch",
        "label": "午間檢查",
        "protein_progress_min": 0.40,
    },
    "afternoon": {
        "hour": 18,
        "minute": 0,
        "slot": "jitai_afternoon",
        "label": "傍晚檢查",
        "protein_progress_min": 0.65,
    },
}


def _bot_timezone() -> ZoneInfo:
    return ZoneInfo((os.getenv("BOT_TIMEZONE") or "Asia/Taipei").strip())


def _jitai_window_end_local(d: date, tz: ZoneInfo) -> datetime:
    end_h = int(os.getenv("JITAI_WINDOW_END_HOUR", "22"))
    end_m = int(os.getenv("JITAI_WINDOW_END_MINUTE", "0"))
    return datetime.combine(d, time(end_h, end_m), tzinfo=tz)


def _jitai_final_fire_local(d: date, tz: ZoneInfo) -> datetime:
    mins_before = int(os.getenv("JITAI_FINAL_MINUTES_BEFORE", "75"))
    end = _jitai_window_end_local(d, tz)
    return end - timedelta(minutes=mins_before)


def _jitai_recent_meal_minutes() -> int:
    return int(os.getenv("JITAI_RECENT_MEAL_MINUTES", "45"))


def _jitai_user_behind(
    totals: dict,
    targets: dict,
    protein_progress_min: float,
) -> bool:
    t_pro = float(targets.get("protein") or 0)
    if t_pro <= 0:
        return False
    cur = float(totals.get("protein") or 0)
    if cur >= t_pro:
        return False
    remaining = t_pro - cur
    if remaining < 12:
        return False
    return cur < t_pro * protein_progress_min


def _jitai_rule_based_options(remaining_pro: float, urgent: bool) -> str:
    """AI 失敗時的備援：仍給可執行選項。"""
    lines = ["可立即補充的選項："]
    opts: list[str] = []
    if remaining_pro >= 35:
        opts.extend([
            "乳清蛋白飲 1 份（約 +25g 蛋白）",
            "即食雞胸 150g（約 +35g 蛋白）",
            "希臘優格 200g ＋ 花生醬 1 湯匙（約 +25g 蛋白）",
        ])
    elif remaining_pro >= 18:
        opts.extend([
            "水煮蛋 2 顆（約 +12g 蛋白）",
            "乳清蛋白飲 1 份（約 +25g 蛋白）",
            "即食雞胸 100g（約 +23g 蛋白）",
        ])
    else:
        opts.extend([
            "水煮蛋 1 顆（約 +6g 蛋白）",
            "無糖豆漿 450ml（約 +15g 蛋白）",
            "希臘優格 150g（約 +15g 蛋白）",
        ])
    for i, o in enumerate(opts[:3], 1):
        lines.append(f"{i}. {o}")
    if urgent:
        lines.append("\n今日紀錄窗格即將關閉，請選一項最快能拿到的先補上。")
    else:
        lines.append("\n選一項最順手的先補，補完拍張照或傳「加蛋白飲」記錄。")
    return "\n".join(lines)


async def _build_jitai_nudge_message(
    user_id: str,
    local_date_s: str,
    checkpoint: str,
    *,
    urgent: bool = False,
) -> str:
    totals = db.get_daily_totals(user_id, local_date_s)
    targets = get_user_targets(user_id)
    profile = db.get_user_profile(user_id) or {}
    goal = _profile_fitness_goal(profile)
    remaining_pro = max(0.0, float(targets["protein"]) - float(totals["protein"]))
    remaining_cal = float(targets["calories"]) - float(totals["calories"])

    tz = _bot_timezone()
    now_local = datetime.now(timezone.utc).astimezone(tz)
    window_end = _jitai_window_end_local(now_local.date(), tz)
    mins_left = max(0, int((window_end - now_local).total_seconds() // 60))

    meals = db.get_meals_today(user_id, local_date_s)
    meal_lines = []
    for m in meals[-4:]:
        meal_lines.append(
            f"- {m.get('food_description', '?')} "
            f"（{float(m.get('protein') or 0):.0f}g 蛋白）"
        )
    meals_ctx = "\n".join(meal_lines) if meal_lines else "（今日尚無紀錄）"

    cp_label = JITAI_CHECKPOINTS.get(checkpoint, {}).get("label", checkpoint)
    urgency = (
        "這是今日紀錄窗格關閉前的最後提醒，請聚焦最快能補上的組合，語氣可更緊迫。"
        if urgent
        else "僅在進度落後時觸發；請給務實、可馬上執行的建議。"
    )
    user_prompt = (
        f"檢查點：{cp_label}\n"
        f"健身目標：{goal}\n"
        f"今日已攝取蛋白質：{totals['protein']:.0f} g／目標 {targets['protein']:.0f} g\n"
        f"尚缺蛋白質：約 {remaining_pro:.0f} g\n"
        f"今日已攝取熱量：{totals['calories']:.0f} kcal／目標 {targets['calories']:.0f} kcal\n"
        f"熱量剩餘空間：約 {remaining_cal:.0f} kcal\n"
        f"距離今日窗格結束還有約 {mins_left} 分鐘\n"
        f"{urgency}\n\n"
        f"今日已記錄餐點：\n{meals_ctx}"
    )

    ai_text = await call_openai_jitai_nudge(user_prompt)
    ai_text = (ai_text or "").strip()
    if ai_text:
        header = "【智能提醒"
        if urgent:
            header += "｜今日最後補充窗口"
        header += "】\n"
        return truncate_line_text(header + ai_text)

    fallback_intro = (
        f"蛋白質尚缺約 {remaining_pro:.0f} g，距離今日窗格結束約 {mins_left} 分鐘。\n\n"
        if remaining_pro > 0
        else f"今日蛋白質已達標；距離窗格結束約 {mins_left} 分鐘，維持記錄節奏即可。\n\n"
    )
    return truncate_line_text(
        "【智能提醒" + ("｜今日最後補充窗口" if urgent else "") + "】\n"
        + fallback_intro
        + _jitai_rule_based_options(remaining_pro, urgent=urgent)
    )


def handle_jitai_toggle(user_id: str, enabled: bool) -> str:
    if not db.is_onboarded(user_id):
        return ONBOARDING_BLOCKED_TEXT
    db.set_jitai_nudges_enabled(user_id, enabled)
    if enabled:
        tz = _bot_timezone()
        end = _jitai_window_end_local(date.today(), tz)
        final = _jitai_final_fire_local(date.today(), tz)
        return (
            "已開啟智能提醒（JITAI）。\n\n"
            "運作方式：\n"
            "· 僅在你進度真的落後時才推送（不會每天固定時間唸數字）\n"
            "· 內容為可馬上執行的補充組合，而非單純報告缺口\n"
            f"· 每日 {final.strftime('%H:%M')} 有一次最後補充提醒（窗格 {end.strftime('%H:%M')} 前）\n\n"
            "傳「關閉提醒」可隨時關閉。"
        )
    return "已關閉智能提醒，不會再收到進度提醒推播。"


def handle_jitai_status(user_id: str) -> str:
    if not db.is_onboarded(user_id):
        return ONBOARDING_BLOCKED_TEXT
    on = db.jitai_nudges_enabled(user_id)
    tz = _bot_timezone()
    end = _jitai_window_end_local(date.today(), tz)
    final = _jitai_final_fire_local(date.today(), tz)
    return (
        "智能提醒狀態\n"
        "————\n"
        f"目前：{'已開啟' if on else '已關閉（預設）'}\n\n"
        "檢查點（僅落後時推送）：\n"
        "· 14:00 午間\n"
        "· 18:00 傍晚\n"
        f"最後補充：{final.strftime('%H:%M')}（窗格 {end.strftime('%H:%M')} 前，必定推送）\n\n"
        "指令：開啟提醒／關閉提醒"
    )


async def execute_jitai_nudge_push(checkpoint: str) -> dict:
    """
    JITAI 智能提醒批次推播。
    checkpoint: lunch｜afternoon｜final
    """
    if checkpoint not in (*JITAI_CHECKPOINTS.keys(), "final"):
        return {"error": f"unknown_checkpoint: {checkpoint}"}

    tz = _bot_timezone()
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz)
    local_date_s = now_local.date().isoformat()
    urgent = checkpoint == "final"

    user_ids = db.get_user_ids_with_jitai_enabled()
    ok, skip, fail = 0, 0, 0
    configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    recent_min = _jitai_recent_meal_minutes()

    async with AsyncApiClient(configuration) as api_client:
        line_api = AsyncMessagingApi(api_client)
        for uid in user_ids:
            try:
                slot = (
                    JITAI_CHECKPOINTS[checkpoint]["slot"]
                    if checkpoint in JITAI_CHECKPOINTS
                    else "jitai_final"
                )
                if db.reminder_already_sent(uid, local_date_s, slot):
                    skip += 1
                    continue

                totals = db.get_daily_totals(uid, local_date_s)
                targets = get_user_targets(uid)

                if not urgent:
                    prog_min = JITAI_CHECKPOINTS[checkpoint]["protein_progress_min"]
                    if not _jitai_user_behind(totals, targets, prog_min):
                        skip += 1
                        continue
                    if db.user_had_meal_in_recent_minutes(
                        uid, local_date_s, recent_min, end_utc_iso=now_utc.isoformat()
                    ):
                        skip += 1
                        continue

                text = await _build_jitai_nudge_message(
                    uid, local_date_s, checkpoint, urgent=urgent,
                )
                await line_api.push_message(
                    PushMessageRequest(
                        to=uid,
                        messages=[TextMessage(text=text)],
                    )
                )
                db.mark_reminder_sent(uid, local_date_s, slot)
                ok += 1
                logger.info("已推播 JITAI 提醒（%s）給 %s...", checkpoint, uid[:8])
            except Exception as e:
                fail += 1
                logger.error("JITAI 提醒推播失敗 %s: %s", uid[:8], e)

    return {
        "checkpoint": checkpoint,
        "date": local_date_s,
        "users_enabled": len(user_ids),
        "pushed_ok": ok,
        "skipped": skip,
        "pushed_fail": fail,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定時推播（建議由 GitHub Actions 呼叫 /cron/daily-summary）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def execute_daily_summary_push() -> dict:
    """推播每日總結（可重複呼叫）：對象為近期曾互動／有紀錄者；無餐點也會收到提示。"""
    today_str = date.today().isoformat()
    meal_since = (date.today() - timedelta(days=400)).isoformat()
    try:
        user_ids = db.get_user_ids_for_daily_summary(meal_since)
    except Exception as e:
        logger.error("每日總結：讀取目標使用者失敗: %s", e, exc_info=True)
        return {
            "date": today_str,
            "users": 0,
            "pushed_ok": 0,
            "pushed_fail": 0,
            "ok": False,
            "error": f"load_users_failed: {e}",
        }
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
    return {
        "date": today_str,
        "users": len(user_ids),
        "pushed_ok": ok,
        "pushed_fail": fail,
        "ok": True,
    }


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
                text_to_send = text
                if slot == "evening":
                    totals_today = db.get_daily_totals(uid, local_date_s)
                    if totals_today.get("meal_count", 0) == 0:
                        text_to_send = "今天還沒有任何飲食紀錄。\n拍一張食物照片開始記錄吧。"
                await line_api.push_message(
                    PushMessageRequest(
                        to=uid,
                        messages=[TextMessage(text=text_to_send)],
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


async def _analyze_image_by_state(
    user_id: str,
    message_id: str,
    state_at_receive: str,
    user_note: str = "",
    force_scale: bool = False,
) -> str:
    if state_at_receive == UserState.WAITING_PURCHASE_PHOTO:
        return await handle_purchase_query_photo(user_id, message_id)
    if state_at_receive in (
        UserState.WAITING_INBODY_PHOTO,
        UserState.ONBOARDING_WAITING_GOAL,
    ):
        return await handle_inbody_photo(user_id, message_id)
    if db.needs_inbody(user_id):
        return await handle_inbody_photo(user_id, message_id)
    if not db.is_onboarded(user_id):
        return ONBOARDING_BLOCKED_TEXT
    return await handle_meal_photo(
        user_id, message_id, user_note=user_note, force_scale=force_scale,
    )


async def run_image_analysis_and_push(user_id: str, message_id: str, state_at_receive: str):
    """背景執行：下載、壓縮、Vision、寫入 DB，完成後 push 結果。"""
    timeout_sec = float(os.getenv("IMAGE_ANALYSIS_TIMEOUT_SEC", "180"))
    note_wait_sec = _photo_note_window_sec()
    analyze_timeout = max(60.0, timeout_sec - note_wait_sec)
    body = ""
    logger.info(
        "圖片分析開始 user=%s msg=%s state=%s total=%.0fs analyze=%.0fs",
        user_id[:8],
        message_id,
        state_at_receive,
        timeout_sec,
        analyze_timeout,
    )
    try:
        user_note = ""
        force_scale = False
        skip_note = state_at_receive in (
            UserState.WAITING_PURCHASE_PHOTO,
            UserState.WAITING_INBODY_PHOTO,
            UserState.ONBOARDING_WAITING_GOAL,
        ) or db.needs_inbody(user_id)
        if not skip_note:
            user_note, force_scale = await wait_and_take_pending_note(
                user_id, message_id, note_wait_sec,
            )
            if user_note:
                logger.info("圖片分析附加使用者備註 user=%s msg=%s", user_id[:8], message_id)
        body = await asyncio.wait_for(
            _analyze_image_by_state(
                user_id, message_id, state_at_receive,
                user_note=user_note, force_scale=force_scale,
            ),
            timeout=analyze_timeout,
        )
        logger.info("圖片分析完成 user=%s msg=%s", user_id[:8], message_id)
    except asyncio.TimeoutError:
        logger.error("圖片分析逾時 user=%s msg=%s", user_id[:8], message_id)
        body = (
            "本次圖片分析耗時過長，已自動中止。\n"
            "請重新傳一次照片（盡量清晰、只拍重點），我會立即重跑。"
        )
    except asyncio.CancelledError:
        logger.error("圖片分析任務被取消 user=%s msg=%s", user_id[:8], message_id)
        if not body.strip():
            body = "分析尚未完成（可能因服務重啟），請重新傳送照片。"
        raise
    except UserFacingError as e:
        logger.warning("圖片分析可恢復錯誤 user=%s msg=%s err=%s", user_id[:8], message_id, e)
        body = str(e)
    except Exception as e:
        logger.error("背景圖片分析失敗: %s", e, exc_info=True)
        body = "分析過程發生錯誤，請稍後再試或重新傳送照片。"
    finally:
        if not (body or "").strip():
            body = "分析完成但未產生結果，請重新傳送照片。"
        try:
            await push_line_text_with_retry(user_id, body, attempts=3)
            logger.info("圖片分析結果已推送 user=%s msg=%s", user_id[:8], message_id)
        except Exception as e:
            logger.error("Push 分析結果失敗 user=%s msg=%s: %s", user_id[:8], message_id, e, exc_info=True)


HELP_TEXT = (
    "減重監測 Bot 使用說明\n"
    "=" * 24 + "\n\n"
    "新使用者：請先上傳 InBody 報告，再回覆健身目標"
    "（增肌／減脂／增肌減脂／維持／增肥）。\n\n"
    "基本操作：\n"
    "  拍照傳食物，將自動分析並記錄（熱量為減脂保守估計）。\n"
    "  文字報餐：以「記錄」「記一筆」或「吃」開頭＋描述（例：記錄 雞腿飯半個＋燙青菜）。\n\n"
    "  傳完照片後 10 秒內可補備註／備注，例如：\n"
    "    備註：這是雞胸肉不是豬排\n"
    "    秤重：雞胸150g、米飯180g、總重量：450g\n"
    "    備注：炸雞、高麗菜、泡麵，總重量：520g（僅一次秤重時寫總重量）\n"
    "    克數 雞胸 150、泡麵 50（品名＋克數，建議加 g 或克）\n\n"
    "文字指令：\n"
    "  「今日」：查看今日總結\n"
    "  「清除今日」：刪除今日所有紀錄\n"
    "  「本週積分」「週報」「積分卡」等：本週飲食成績\n"
    "  「設定蛋白飲 熱量 蛋白質」：自訂快速記錄數值（例：設定蛋白飲 130 25）\n"
    "  「Notion狀態」：檢查目前是否會同步到 Notion\n"
    "  「我的ID」：顯示你的 LINE userId（可用於比對 NOTION_SYNC_USER_ID）\n"
    "  「說明」：顯示此說明\n\n"
    "智能提醒 JITAI（預設關閉，需手動開啟）：\n"
    "  「開啟提醒」：僅進度落後時推送可執行補充建議\n"
    "  「關閉提醒」：停止提醒推播\n"
    "  「提醒狀態」：查看是否已開啟與檢查時段\n"
    "  （舊別名「開啟體醒」等仍可使用）\n\n"
    "圖文選單：\n"
    "  「食物查詢」或「購買查詢」：買前先查熱量等級\n"
    "  「加蛋白飲」：一鍵記錄一份蛋白飲（不耗 AI）\n"
    "  「上傳InBody」：更新體組成\n"
    "  「欺騙日」「今日總結」：同字面；「加一份碳水」：同「加碳水」快速記錄\n"
    "  「AI教練」：仍可打字使用\n\n"
    "快速記錄（不耗 token）：\n"
    "  「加蛋白飲」「蛋白飲」「＋蛋白」\n"
    "  「加雞蛋」「＋雞蛋」\n"
    "  「加雞胸肉」「＋雞胸肉」\n"
    "  「加碳水」「加一份碳水」「加地瓜」「＋碳水」"
)


@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks):
    signature = request.headers.get("X-Line-Signature", "")
    body = (await request.body()).decode("utf-8")

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    async with AsyncApiClient(configuration) as api_client:
        line_api = AsyncMessagingApi(api_client)

        for event in events:
            if isinstance(event, FollowEvent):
                user_id = event.source.user_id
                logger.info("新使用者加入好友 user=%s", user_id[:8])
                welcome = truncate_line_text(ONBOARDING_WELCOME_TEXT)
                try:
                    await line_api.reply_message(
                        ReplyMessageRequest(
                            reply_token=event.reply_token,
                            messages=[TextMessage(text=welcome)],
                        )
                    )
                except Exception as e:
                    logger.warning("Follow 歡迎訊息 reply 失敗: %s", e)
                continue

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
                    if snap_state in (
                        UserState.WAITING_INBODY_PHOTO,
                        UserState.ONBOARDING_WAITING_GOAL,
                    ) or db.needs_inbody(user_id):
                        reply_text = "已收到 InBody 照片，正在分析中…"
                    if snap_state not in (
                        UserState.WAITING_PURCHASE_PHOTO,
                        UserState.WAITING_INBODY_PHOTO,
                    ) and not db.needs_inbody(user_id):
                        await create_pending_note_window(
                            user_id, event.message.id, window_sec=_photo_note_window_sec(),
                        )
                    background_tasks.add_task(
                        run_image_analysis_and_push,
                        user_id,
                        event.message.id,
                        snap_state,
                    )
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
        note_text, is_scale_prefix = _extract_note_text(text)
        if note_text is not None:
            ok, msg = await add_pending_note_if_open(
                user_id, note_text, force_scale=is_scale_prefix,
            )
            if ok:
                return msg

        onboarding_allowed = {
            "說明", "幫助", "help", "Help", "HELP",
            "我的ID", "我的id", "my id", "My ID", "userid", "user id",
        }

        goal = parse_fitness_goal(text)
        if goal and (
            state == UserState.ONBOARDING_WAITING_GOAL or db.needs_fitness_goal(user_id)
        ):
            return await handle_fitness_goal_selection(user_id, goal)

        if not db.is_onboarded(user_id):
            if text in onboarding_allowed:
                pass
            elif db.needs_inbody(user_id):
                return ONBOARDING_NEED_INBODY_TEXT
            elif db.needs_fitness_goal(user_id):
                return ONBOARDING_BLOCKED_TEXT

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

        if _compact_command(text) in JITAI_ON_COMMANDS:
            leave_photo_wait_if_any(user_id)
            return handle_jitai_toggle(user_id, True)

        if _compact_command(text) in JITAI_OFF_COMMANDS:
            leave_photo_wait_if_any(user_id)
            return handle_jitai_toggle(user_id, False)

        if _compact_command(text) in JITAI_STATUS_COMMANDS:
            leave_photo_wait_if_any(user_id)
            return handle_jitai_status(user_id)

        if text in ("今日", "今日總計", "今日總結", "總計", "今天"):
            leave_photo_wait_if_any(user_id)
            return await handle_today_summary(user_id)

        if text in ("我的ID", "我的id", "my id", "My ID", "userid", "user id"):
            leave_photo_wait_if_any(user_id)
            return f"你的 LINE userId：\n{user_id}"

        if text in ("Notion狀態", "notion狀態", "Notion 狀態", "notion status", "Notion status"):
            leave_photo_wait_if_any(user_id)
            notion = get_notion_sync()
            sync_ok = notion.should_sync_line_user(user_id)
            uid_set = "已設定" if notion.sync_user_id else "未設定"
            uid_match = "符合" if sync_ok else "不符合"
            return (
                "Notion 同步檢查\n"
                "--------------------\n"
                f"Notion 啟用：{'是' if notion.enabled else '否'}\n"
                f"NOTION_SYNC_USER_ID：{uid_set}\n"
                f"目前 userId 是否可同步：{uid_match}\n\n"
                "若顯示不符合，請先傳「我的ID」，並把該值填入 Render 的 NOTION_SYNC_USER_ID。"
            )

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

        if text in (
            "加碳水",
            "加一份碳水",
            "碳水",
            "+碳水",
            "＋碳水",
            "加地瓜",
            "+地瓜",
            "＋地瓜",
        ):
            leave_photo_wait_if_any(user_id)
            return await handle_quick_protein(user_id, "碳水")

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
            goal_label = _profile_fitness_goal(profile)
            lines = [
                "目前設定的目標",
                "=" * 24,
                f"健身目標：{goal_label}",
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
    _verify_cron_secret_or_401(request)
    try:
        result = await execute_daily_summary_push()
        return JSONResponse(content=result)
    except Exception as e:
        logger.error("cron daily-summary 執行失敗: %s", e, exc_info=True)
        return JSONResponse(
            content={
                "ok": False,
                "error": f"cron_daily_summary_failed: {e}",
            }
        )


def _verify_cron_secret_or_401(request: Request):
    """驗證 cron 共享密鑰。"""
    secret = os.getenv("CRON_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="CRON_SECRET 未設定")
    got = request.headers.get("X-Cron-Secret", "")
    ga, gb = got.encode("utf-8"), secret.encode("utf-8")
    if len(ga) != len(gb) or not hmac.compare_digest(ga, gb):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/cron/meal-reminder")
async def cron_meal_reminder(request: Request):
    """
    給外部排程觸發用餐提醒：
    - /cron/meal-reminder?slot=noon
    - /cron/meal-reminder?slot=evening
    - 未給 slot 時同次執行兩個時段
    """
    _verify_cron_secret_or_401(request)
    slot = (request.query_params.get("slot") or "").strip().lower()
    try:
        if slot in ("noon", "evening"):
            result = await execute_meal_reminder_push(slot)
            return JSONResponse(content={"ok": True, "slot": slot, "result": result})
        noon = await execute_meal_reminder_push("noon")
        evening = await execute_meal_reminder_push("evening")
        return JSONResponse(
            content={"ok": True, "slot": "both", "noon": noon, "evening": evening}
        )
    except Exception as e:
        logger.error("cron meal-reminder 執行失敗: %s", e, exc_info=True)
        return JSONResponse(
            content={
                "ok": False,
                "error": f"cron_meal_reminder_failed: {e}",
                "slot": slot or "both",
            }
        )


@app.post("/cron/jitai-nudge")
async def cron_jitai_nudge(request: Request):
    """
    智能提醒 JITAI（僅已開啟者）：
    - /cron/jitai-nudge?checkpoint=lunch
    - /cron/jitai-nudge?checkpoint=afternoon
    - /cron/jitai-nudge?checkpoint=final
    """
    _verify_cron_secret_or_401(request)
    checkpoint = (request.query_params.get("checkpoint") or "").strip().lower()
    if checkpoint not in ("lunch", "afternoon", "final"):
        raise HTTPException(
            status_code=400,
            detail="checkpoint 須為 lunch、afternoon 或 final",
        )
    try:
        result = await execute_jitai_nudge_push(checkpoint)
        return JSONResponse(content={"ok": True, "checkpoint": checkpoint, "result": result})
    except Exception as e:
        logger.error("cron jitai-nudge 執行失敗: %s", e, exc_info=True)
        return JSONResponse(
            content={
                "ok": False,
                "error": f"cron_jitai_nudge_failed: {e}",
                "checkpoint": checkpoint,
            }
        )


@app.post("/cron/db-keepalive")
async def cron_db_keepalive(request: Request):
    """給外部排程觸發：執行最小 DB 讀取，避免長期閒置。"""
    _verify_cron_secret_or_401(request)
    today_str = date.today().isoformat()
    probe_uid = os.getenv("DB_KEEPALIVE_PROBE_USER_ID", "__keepalive__")
    totals = db.get_daily_totals(probe_uid, today_str)
    return JSONResponse(
        content={
            "status": "ok",
            "date": today_str,
            "probe_user": probe_uid,
            "db_mode": "postgres" if db._pg else "sqlite",
            "totals_read": totals,
        }
    )


@app.get("/health")
async def health():
    commit = (
        os.getenv("RENDER_GIT_COMMIT")
        or os.getenv("GIT_COMMIT")
        or "local"
    )[:12]
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "features": ["jitai_nudge", "scale_note", "onboarding"],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
