"""
減重監測 LINE Bot - 完整主程式
FastAPI + LINE Messaging API + OpenAI GPT Vision
"""

import os
import json
import base64
import logging
import asyncio
from io import BytesIO
from datetime import date, datetime, timedelta
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_dotenv()

LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 常數與評分系統
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 注音評分 (僅用於食物查詢)
FOOD_GRADES = ["ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ"]

# 通用評分 (用於其他所有功能)
GENERAL_GRADES = ["S", "A", "B", "C", "D", "E"]

DEFAULT_PROTEIN_TARGET = 300  # g
DEFAULT_CALORIE_TARGET = 2500  # kcal


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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI Vision 呼叫
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def call_openai_vision(
    system_prompt: str,
    user_prompt: str,
    image_url: str | None = None,
    image_base64: str | None = None,
) -> dict | str:
    """呼叫 OpenAI Vision API，回傳解析後的 JSON 或原始文字。"""

    # 組裝 user content
    user_content = []
    if image_url:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url, "detail": "high"},
        })
    elif image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "high"},
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

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # 嘗試解析 JSON
    try:
        # 移除可能的 markdown 包裹
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        return json.loads(cleaned)
    except (json.JSONDecodeError, IndexError):
        return raw


async def call_openai_text(system_prompt: str, user_prompt: str) -> dict | str:
    """呼叫 OpenAI 文字 API (無圖片)。"""
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2000,
        "temperature": 0.4,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    try:
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned.strip())
    except (json.JSONDecodeError, IndexError):
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
        logger.warning("圖片壓縮失敗,改以原始二進位送 Vision", exc_info=True)
        return base64.b64encode(data).decode("utf-8")


async def get_line_image_base64(message_id: str) -> str:
    """下載 LINE 圖片並壓縮為 JPEG base64，供 OpenAI Vision 使用。"""
    raw = await download_line_image_bytes(message_id)
    return compress_image_bytes_to_jpeg_base64(raw)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 提示詞定義
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROMPT_MEAL_ANALYSIS = """你是專業的營養師。分析圖片中的食物,精確估算營養數據。

規則:
- 仔細觀察食物份量、容器大小、食材組成
- 考慮烹調方式對熱量的影響 (油炸 > 煎 > 烤 > 蒸 > 水煮)
- 如果是包裝食品,盡量辨識品牌與標示
- 不要使用任何 emoji
- 使用繁體中文

請嚴格以下列 JSON 格式回覆,不要有其他文字:
{
  "calories": 數字,
  "protein": 數字,
  "description": "食物內容的簡要描述"
}"""

PROMPT_PURCHASE_QUERY = """你是專業的營養師與食品分析師。分析圖片中的食物或商品包裝,提供購買前的完整評估。

評級系統使用注音符號:
- ㄅ: 非常適合減重/增肌目標,營養優良
- ㄆ: 大致適合,偶爾可食用
- ㄇ: 中性,需要控制份量
- ㄈ: 不建議,高熱量或低營養價值
- ㄉ: 極度不建議,對目標有嚴重負面影響

規則:
- 不要使用任何 emoji
- 使用繁體中文
- 評估要嚴格但公正
- 替代建議要具體且容易取得

請嚴格以下列 JSON 格式回覆,不要有其他文字:
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
  "timing": "適合的飲用或食用時機,例如健身前30分鐘、增肌期訓練後、減脂期避免等",
  "verdict": "一句話總結: 建議購買或不建議,以及原因",
  "alternatives": ["更健康的替代選項1", "替代選項2", "替代選項3"]
}"""

PROMPT_INBODY_ANALYSIS = """你是專業的體適能教練。分析這張 InBody 體組成報告,提取關鍵數據。

規則:
- 盡可能辨識所有數值
- 無法辨識的項目設為 null
- 不要使用任何 emoji
- 使用繁體中文

請嚴格以下列 JSON 格式回覆:
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

PROMPT_AI_COACH = """你是專業的健身營養教練,名叫「減重教練」。根據提供的數據,給出精準的個人化建議。

評分系統:
- S: 超過100分的表現,極其出色
- A: 優秀 (85-100%)
- B: 良好 (70-84%)
- C: 及格 (55-69%)
- D: 需改進 (40-54%)
- E: 需要大幅改進 (<40%)

規則:
- 不要使用任何 emoji
- 使用繁體中文
- 語氣專業但友善,像一個嚴格但關心你的教練
- 建議要具體可執行,不要空泛
- 如果數據不足,誠實說明,不要編造

請嚴格以下列 JSON 格式回覆:
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
  "coach_note": "教練的個人化鼓勵或提醒,2-3句話"
}"""

PROMPT_CREATIVE_SUGGESTION = """你是一位有創意的健身營養教練。用戶剛完成一餐紀錄,請根據他今天剩餘的蛋白質缺口,給出一個簡短、具體、有幫助的補充建議。

規則:
- 不要使用任何 emoji
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
    """取得使用者的每日目標 (從 DB 或使用預設值)。"""
    profile = db.get_user_profile(user_id)
    if profile:
        return {
            "calories": profile.get("daily_calorie_target", DEFAULT_CALORIE_TARGET),
            "protein": profile.get("daily_protein_target", DEFAULT_PROTEIN_TARGET),
        }
    return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}


def get_gap_filler(remaining_protein: float) -> str:
    """根據剩餘蛋白質量給建議。"""
    if remaining_protein <= 0:
        return "已達成今日蛋白質目標! 保持下去。"
    elif remaining_protein <= 30:
        return f"只差 {remaining_protein:.0f}g -- 一份乳清蛋白即可達標。"
    elif remaining_protein <= 60:
        return f"還需 {remaining_protein:.0f}g -- 一塊雞胸肉 (200g) 約可補足。"
    elif remaining_protein <= 100:
        return f"還缺 {remaining_protein:.0f}g -- 可安排一份雞胸 + 一份乳清。"
    else:
        return f"還缺 {remaining_protein:.0f}g -- 建議在剩餘餐次中優先選擇高蛋白食物。"


async def handle_meal_photo(user_id: str, message_id: str) -> str:
    """處理食物照片: 分析 + 記錄 + 回傳摘要。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_MEAL_ANALYSIS,
        user_prompt="請分析這份餐點的熱量與蛋白質。",
        image_base64=image_b64,
    )

    if isinstance(result, str):
        return f"分析失敗,請重新拍照。\n\n原始回應:\n{result[:200]}"

    cal = float(result.get("calories", 0))
    pro = float(result.get("protein", 0))
    desc = result.get("description", "無法辨識")

    # 寫入 DB
    today_str = date.today().isoformat()
    db.add_meal(user_id, cal, pro, desc, today_str)

    # 今日累計
    totals = db.get_daily_totals(user_id, today_str)
    targets = get_user_targets(user_id)
    remaining = targets["protein"] - totals["protein"]

    bar = build_progress_bar(totals["protein"], targets["protein"])
    gap = get_gap_filler(remaining)

    return (
        f"本餐分析結果\n"
        f"{'=' * 24}\n"
        f"食物內容:\n{desc}\n\n"
        f"營養數據:\n"
        f"  熱量: {cal:.0f} kcal\n"
        f"  蛋白質: {pro:.0f} g\n\n"
        f"{'=' * 24}\n"
        f"今日累計:\n"
        f"  總熱量: {totals['calories']:.0f} kcal\n"
        f"  總蛋白質: {totals['protein']:.0f} g / {targets['protein']:.0f} g\n"
        f"  {bar}\n\n"
        f"{gap}"
    )


async def handle_purchase_query_photo(user_id: str, message_id: str) -> str:
    """處理購買查詢照片。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_PURCHASE_QUERY,
        user_prompt="請分析這個食物或商品,提供完整的購買前評估。",
        image_base64=image_b64,
    )

    if isinstance(result, str):
        clear_state(user_id)
        return f"分析失敗,請重新拍照。\n\n原始回應:\n{result[:200]}"

    # 儲存分析結果到 context,等使用者決定
    set_state(user_id, UserState.PURCHASE_REVIEWED, context=result)

    g = result.get("grades", {})

    lines = [
        f"購買前分析報告",
        f"{'=' * 24}",
        f"品項: {result.get('name', '未知')}",
        "",
        f"營養數據 (估算):",
        f"  熱量: {result.get('calories', '?')} kcal",
        f"  蛋白質: {result.get('protein', '?')} g",
        f"  碳水化合物: {result.get('carbs', '?')} g",
        f"  脂肪: {result.get('fat', '?')} g",
        f"  糖: {result.get('sugar', '?')} g",
        "",
        f"等級評定:",
        f"  綜合評價: {g.get('overall', '?')}",
        f"  美味度: {g.get('taste', '?')}",
        f"  脂肪含量: {g.get('fat_level', '?')}",
        f"  糖含量: {g.get('sugar_level', '?')}",
        f"  熱量密度: {g.get('calorie_density', '?')}",
        f"  碳水品質: {g.get('carb_quality', '?')}",
        "",
        f"適合時機:",
        f"  {result.get('timing', '無特別建議')}",
        "",
        f"{'=' * 24}",
        f"結論: {result.get('verdict', '')}",
        "",
        f"更健康的替代選項:",
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
    """處理 InBody 照片: OCR + 更新目標。"""
    image_b64 = await get_line_image_base64(message_id)

    result = await call_openai_vision(
        system_prompt=PROMPT_INBODY_ANALYSIS,
        user_prompt="請分析這張 InBody 體組成報告,提取所有可辨識的數據。",
        image_base64=image_b64,
    )

    if isinstance(result, str):
        clear_state(user_id)
        return f"InBody 分析失敗,請確認照片清晰度。\n\n{result[:200]}"

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

    lines = [
        "InBody 數據已更新",
        "=" * 24,
        "",
        "辨識結果:",
        f"  體重: {weight or '?'} kg",
        f"  體脂率: {bf or '?'} %",
        f"  骨骼肌: {muscle or '?'} kg",
        f"  基礎代謝: {bmr or '?'} kcal",
        f"  體水分: {result.get('body_water', '?')} kg",
        f"  內臟脂肪: {result.get('visceral_fat', '?')}",
        f"  檢測日期: {result.get('test_date', '?')}",
        "",
        "=" * 24,
        "目標已自動調整:",
        f"  每日熱量目標: {new_targets['calories']:.0f} kcal",
        f"  每日蛋白質目標: {new_targets['protein']:.0f} g",
        "",
        result.get("summary", ""),
    ]

    return "\n".join(lines)


def calculate_targets(weight, body_fat, bmr) -> dict:
    """根據 InBody 數據計算每日目標。"""
    if not bmr or not weight:
        return {"calories": DEFAULT_CALORIE_TARGET, "protein": DEFAULT_PROTEIN_TARGET}

    if body_fat and body_fat > 20:
        # 減脂: TDEE * 0.85
        cal = bmr * 1.5 * 0.85
        pro = weight * 2.2
    elif body_fat and body_fat < 15:
        # 增肌: TDEE * 1.1
        cal = bmr * 1.6 * 1.1
        pro = weight * 2.0
    else:
        # 維持
        cal = bmr * 1.55
        pro = weight * 2.0

    return {"calories": round(cal), "protein": round(pro)}


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
        return "本週尚無飲食紀錄,開始記錄後才能產生積分卡。"

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
        comments.append(f"蛋白質達標率偏低 ({protein_met}/{total_days} 天),需要在每餐加入高蛋白食物。")
    elif protein_grade in ("S", "A"):
        comments.append(f"蛋白質攝取表現出色 ({protein_met}/{total_days} 天達標)。")

    if calorie_grade in ("D", "E"):
        direction = "超出" if avg_cal > targets["calories"] else "不足"
        comments.append(f"平均熱量{direction}目標 {abs(avg_cal - targets['calories']):.0f} kcal,需要調整。")
    elif calorie_grade in ("S", "A"):
        comments.append("熱量控制非常精準,繼續保持。")

    if regularity_grade in ("D", "E"):
        comments.append(f"本週只有 {total_days}/{elapsed_days} 天有紀錄,請養成每餐拍照的習慣。")

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
        f"  綜合評分: {overall}",
        "",
        f"  蛋白質達標: {protein_grade}",
        f"    ({protein_met}/{total_days} 天達到 90% 以上)",
        "",
        f"  熱量控制: {calorie_grade}",
        f"    (平均 {avg_cal:.0f} / 目標 {targets['calories']:.0f} kcal)",
        "",
        f"  紀錄規律: {regularity_grade}",
        f"    ({total_days}/{elapsed_days} 天有紀錄)",
        "",
        "=" * 24,
        "教練評語:",
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
            f"正常熱量上限: {targets['calories']:.0f} kcal\n"
            f"欺騙日上限: {cheat_cal:.0f} kcal\n"
            f"目前已攝取: {totals['calories']:.0f} kcal\n\n"
            f"提醒: 欺騙日不代表無限制,\n"
            f"蛋白質目標依然需要達成。"
        )

    # 檢查本週是否已使用
    week_start = date.today() - timedelta(days=date.today().weekday())
    if db.count_cheat_days_in_range(user_id, week_start.isoformat(), today_str) > 0:
        return (
            "本週已使用過欺騙日。\n"
            "建議每週最多一次,以免影響整體進度。\n\n"
            "如果仍要啟動,請輸入「強制欺騙日」。"
        )

    # 啟動欺騙日
    db.activate_cheat_day(user_id, today_str)
    targets = get_user_targets(user_id)
    cheat_cal = targets["calories"] * 1.3

    return (
        f"欺騙日模式已啟動\n"
        f"{'=' * 24}\n"
        f"今日熱量上限放寬至: {cheat_cal:.0f} kcal\n"
        f"(正常: {targets['calories']:.0f} kcal, +{cheat_cal - targets['calories']:.0f})\n\n"
        f"規則:\n"
        f"  1. 蛋白質目標不變\n"
        f"  2. 盡量安排在訓練日使用\n"
        f"  3. 不會對今日的食物評語過於嚴格\n\n"
        f"享受美食,明天繼續努力。"
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
            "數據不足,無法進行有效分析。\n"
            f"目前只有 {len(days_data)} 天的紀錄,\n"
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
        user_prompt=f"以下是使用者的飲食數據,請進行分析:\n{json.dumps(data_summary, ensure_ascii=False, indent=2)}",
    )

    if isinstance(result, str):
        return f"AI 教練分析完成:\n\n{result}"

    # 格式化輸出
    lines = [
        "AI 教練分析報告",
        "=" * 24,
        "",
        f"綜合評分: {result.get('overall_grade', '?')}",
        "",
        "各項評分:",
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
        lines.append(f"  {label}: {item.get('grade', '?')} - {item.get('comment', '')}")

    lines.extend(["", "=" * 24, "主要問題:"])
    for i, issue in enumerate(result.get("top_issues", []), 1):
        lines.append(f"  {i}. {issue}")

    lines.extend(["", "行動計畫:"])
    for item in result.get("action_plan", []):
        lines.append(f"  - {item.get('task', '')}")
        lines.append(f"    原因: {item.get('reason', '')}")

    coach_note = result.get("coach_note", "")
    if coach_note:
        lines.extend(["", "-----", f"教練的話: {coach_note}"])

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

    lines = [
        f"今日飲食總結 ({today_str})",
        f"{'[欺騙日模式]' if is_cheat else ''}",
        "=" * 24,
        "",
        f"總熱量: {totals['calories']:.0f} / {cal_target:.0f} kcal",
        f"總蛋白質: {totals['protein']:.0f} / {targets['protein']:.0f} g",
        f"{bar}",
        "",
        f"共 {totals['meal_count']} 餐:",
    ]

    for i, meal in enumerate(meals, 1):
        lines.append(f"  {i}. {meal['food_description']}")
        lines.append(f"     {meal['calories']:.0f} kcal / {meal['protein']:.0f}g 蛋白質")

    lines.append("")
    lines.append(get_gap_filler(remaining_protein))

    return "\n".join(lines)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 定時推播 (每晚 23:00)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def daily_summary_job():
    """每晚 23:00 推播當日總結。"""
    while True:
        now = datetime.now()
        # 計算到今晚 23:00 的秒數
        target = now.replace(hour=23, minute=0, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now).total_seconds()

        logger.info(f"下次推播在 {wait_seconds:.0f} 秒後 ({target})")
        await asyncio.sleep(wait_seconds)

        # 取得所有今日有紀錄的使用者
        today_str = date.today().isoformat()
        user_ids = db.get_active_users_today(today_str)

        configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
        async with AsyncApiClient(configuration) as api_client:
            line_api = AsyncMessagingApi(api_client)
            for uid in user_ids:
                try:
                    summary = await handle_today_summary(uid)
                    await line_api.push_message(
                        PushMessageRequest(
                            to=uid,
                            messages=[TextMessage(text=summary)],
                        )
                    )
                    logger.info(f"已推播每日總結給 {uid[:8]}...")
                except Exception as e:
                    logger.error(f"推播失敗 {uid[:8]}: {e}")

        # 等一分鐘避免重複觸發
        await asyncio.sleep(60)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI 應用
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用生命週期: 啟動背景任務。"""
    db.init()
    task = asyncio.create_task(daily_summary_job())
    logger.info("Bot 啟動完成")
    yield
    task.cancel()


app = FastAPI(title="Diet Tracker LINE Bot", lifespan=lifespan)

parser = WebhookParser(LINE_CHANNEL_SECRET)
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)


def truncate_line_text(text: str) -> str:
    if len(text) > 5000:
        return text[:4950] + "\n\n(訊息過長,已截斷)"
    return text


async def push_line_text(user_id: str, text: str):
    """以 Push API 傳送文字 (用於圖片分析完成後,不依賴 reply_token)。"""
    text = truncate_line_text(text)
    cfg = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
    async with AsyncApiClient(cfg) as api_client:
        api = AsyncMessagingApi(api_client)
        await api.push_message(
            PushMessageRequest(to=user_id, messages=[TextMessage(text=text)])
        )


async def run_image_analysis_and_push(user_id: str, message_id: str, state_at_receive: str):
    """背景執行: 下載、壓縮、Vision、寫入 DB,完成後 push 結果。"""
    try:
        if state_at_receive == UserState.WAITING_PURCHASE_PHOTO:
            body = await handle_purchase_query_photo(user_id, message_id)
        elif state_at_receive == UserState.WAITING_INBODY_PHOTO:
            body = await handle_inbody_photo(user_id, message_id)
        else:
            body = await handle_meal_photo(user_id, message_id)
    except Exception as e:
        logger.error("背景圖片分析失敗: %s", e, exc_info=True)
        body = "分析過程發生錯誤,請稍後再試或重新傳送照片。"
    try:
        await push_line_text(user_id, body)
    except Exception as e:
        logger.error("Push 分析結果失敗: %s", e, exc_info=True)


HELP_TEXT = (
    "減重監測 Bot 使用說明\n"
    "=" * 24 + "\n\n"
    "基本操作:\n"
    "  拍照傳食物 -> 自動分析記錄\n\n"
    "文字指令:\n"
    "  「今日」- 查看今日總結\n"
    "  「清除今日」- 刪除今日所有紀錄\n"
    "  「說明」- 顯示此說明\n\n"
    "圖文選單功能:\n"
    "  「購買查詢」- 買之前先查熱量等級\n"
    "  「本週積分」- 本週飲食控制成績\n"
    "  「上傳InBody」- 更新體組成數據\n"
    "  「欺騙日」- 啟動欺騙日模式\n"
    "  「AI教練」- 取得個人化建議\n"
    "  「今日」- 今日飲食總結"
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
            reply_token = event.reply_token
            state = get_state(user_id)

            try:
                if isinstance(event.message, ImageMessageContent):
                    reply_text = "已收到照片，正在分析中…"
                    snap_state = get_state(user_id)
                    asyncio.create_task(
                        run_image_analysis_and_push(user_id, event.message.id, snap_state)
                    )
                else:
                    reply_text = await route_message(event, user_id, state)
            except Exception as e:
                logger.error(f"處理訊息失敗: {e}", exc_info=True)
                reply_text = "處理時發生錯誤,請稍後再試。"

            if reply_text:
                reply_text = truncate_line_text(reply_text)

                await line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=reply_token,
                        messages=[TextMessage(text=reply_text)],
                    )
                )

    return JSONResponse(content={"status": "ok"})


async def route_message(event: MessageEvent, user_id: str, state: str) -> str:
    """路由訊息到對應的處理函數 (圖片改由 webhook 立即回覆後於背景分析並 push)。"""
    if isinstance(event.message, ImageMessageContent):
        return ""

    # ── 文字訊息 ──
    if isinstance(event.message, TextMessageContent):
        text = event.message.text.strip()

        # 狀態內的文字回應
        if state == UserState.PURCHASE_REVIEWED:
            if text in ("買了", "確定", "購買"):
                ctx = get_context(user_id)
                db.save_purchase_decision(user_id, ctx, "purchased")
                clear_state(user_id)
                return (
                    f"已記錄購買: {ctx.get('name', '未知')}\n"
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

        # ── 指令路由 ──
        if text in ("說明", "幫助", "help", "Help", "HELP"):
            return HELP_TEXT

        if text in ("今日", "今日總計", "總計", "今天"):
            return await handle_today_summary(user_id)

        if text == "清除今日":
            today_str = date.today().isoformat()
            count = db.clear_today(user_id, today_str)
            return f"已清除今日 {count} 筆紀錄。"

        if text in ("購買查詢", "查詢", "買之前"):
            set_state(user_id, UserState.WAITING_PURCHASE_PHOTO)
            return (
                "購買前熱量查詢已啟動\n"
                "-----\n"
                "請拍攝食物或商品包裝照片傳送。\n"
                "我會分析營養數據並給予等級評定。\n\n"
                "輸入「取消」可結束查詢。"
            )

        if text in ("本週積分", "積分", "積分卡", "本週"):
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
            return await handle_cheat_day(user_id)

        if text == "強制欺騙日":
            today_str = date.today().isoformat()
            db.activate_cheat_day(user_id, today_str)
            targets = get_user_targets(user_id)
            cheat_cal = targets["calories"] * 1.3
            return (
                f"欺騙日已強制啟動\n"
                f"今日熱量上限: {cheat_cal:.0f} kcal\n\n"
                f"注意: 頻繁使用欺騙日會影響整體進度。"
            )

        if text in ("AI教練", "教練", "ai教練", "AI 教練"):
            return await handle_ai_coach(user_id)

        if text in ("目標", "我的目標", "查看目標"):
            targets = get_user_targets(user_id)
            profile = db.get_user_profile(user_id)
            lines = [
                "目前設定的目標",
                "=" * 24,
                f"每日熱量: {targets['calories']:.0f} kcal",
                f"每日蛋白質: {targets['protein']:.0f} g",
            ]
            if profile:
                lines.extend([
                    "",
                    "身體數據:",
                    f"  體重: {profile.get('weight', '?')} kg",
                    f"  體脂率: {profile.get('body_fat_percentage', '?')} %",
                    f"  上次 InBody: {profile.get('last_inbody_date', '未上傳')}",
                ])
            return "\n".join(lines)

        # 未知指令 → 當作食物文字描述? 或提示使用說明
        return (
            f"無法辨識指令「{text[:20]}」。\n\n"
            "拍照傳食物可自動分析記錄,\n"
            "或輸入「說明」查看所有功能。"
        )

    return ""


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
