
"""
高強度減重監測 LINE Bot
功能:
1. 接收食物照片,分析熱量與蛋白質
2. 記錄每日營養數據
3. 晚上23:00推播每日總結
"""

import os
import asyncio
import base64
import json
import re
import uuid
from datetime import datetime, time
from typing import Optional
import logging

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    ReplyMessageRequest,
    PushMessageRequest,
    TextMessage,
    ImageMessage,
)
from linebot.v3.webhooks import (
    MessageEvent,
    TextMessageContent,
    ImageMessageContent
)
from openai import AsyncOpenAI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pydantic_settings import BaseSettings

from database import Database, MealRecord

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """環境變數設定"""
    LINE_CHANNEL_SECRET: str
    LINE_CHANNEL_ACCESS_TOKEN: str
    OPENAI_API_KEY: str
    BASE_URL: str = "https://your-app.fly.dev"  # 用於週報圖表等對外網址
    
    class Config:
        env_file = ".env"


# 初始化設定
settings = Settings()
app = FastAPI(title="Diet Tracker LINE Bot")

# LINE Bot 設定
configuration = Configuration(access_token=settings.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(settings.LINE_CHANNEL_SECRET)

# OpenAI 客戶端
openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# 資料庫（雲端部署時可設 DATABASE_PATH，例如 Fly volume: /data/diet_tracker.db）
db = Database(os.environ.get("DATABASE_PATH", "diet_tracker.db"))

# 排程器
scheduler = AsyncIOScheduler()

# 週報圖表快取（token -> PNG bytes，取用後刪除）
chart_cache: dict = {}


@app.on_event("startup")
async def startup_event():
    """應用啟動時初始化"""
    await db.init_db()
    
    # 設定每日 23:00 推播
    scheduler.add_job(
        daily_summary_push,
        'cron',
        hour=23,
        minute=0,
        id='daily_summary'
    )
    scheduler.start()
    logger.info("Bot 啟動成功,每日推播已設定")


@app.get("/")
async def root():
    """健康檢查端點"""
    return {"status": "ok", "message": "Diet Tracker Bot is running"}


@app.get("/chart/{token}")
async def get_chart_image(token: str):
    """提供週報圖表圖片（一次性 URL，供 LINE 顯示）"""
    if token not in chart_cache:
        raise HTTPException(status_code=404, detail="圖表不存在或已過期")
    data = chart_cache.pop(token)
    return Response(content=data, media_type="image/png")


@app.post("/webhook")
async def webhook(request: Request):
    """LINE Webhook 端點"""
    signature = request.headers.get('X-Line-Signature', '')
    body = await request.body()
    
    try:
        handler.handle(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    return JSONResponse(content={"status": "ok"})


@handler.add(MessageEvent, message=TextMessageContent)
def handle_text_message(event):
    """處理文字訊息"""
    user_id = event.source.user_id
    text = event.message.text.strip()
    
    # 指令處理
    if text in ["今日", "今日總計", "總計"]:
        summary = asyncio.run(get_daily_summary(user_id))
        reply_text = summary
    elif text in ["說明", "幫助", "help"]:
        reply_text = get_help_message()
    elif text == "清除今日":
        asyncio.run(db.delete_today_records(user_id))
        reply_text = "已清除今日所有記錄"
    elif text == "週報":
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            token, chart_url, fallback_text = asyncio.run(prepare_weekly_chart(user_id))
            if token and chart_url:
                base = settings.BASE_URL.rstrip("/")
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[
                            TextMessage(text=fallback_text),
                            ImageMessage(
                                originalContentUrl=chart_url,
                                previewImageUrl=chart_url,
                            ),
                        ],
                    )
                )
            else:
                line_bot_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=fallback_text)],
                    )
                )
        return
    else:
        reply_text = "請傳送食物照片讓我分析，或輸入「今日」查看總計、「週報」查看本週圖表。"
    
    # 回覆訊息
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)]
            )
        )


@handler.add(MessageEvent, message=ImageMessageContent)
def handle_image_message(event):
    """處理圖片訊息 - 分析食物"""
    user_id = event.source.user_id
    message_id = event.message.id
    
    try:
        # 下載圖片（須使用 MessagingApiBlob 取得二進位內容）
        with ApiClient(configuration) as api_client:
            blob_api = MessagingApiBlob(api_client)
            image_data = blob_api.get_message_content(message_id)
        if not image_data or len(image_data) == 0:
            raise ValueError("無法取得圖片內容")
        
        # 轉換為 base64（支援 bytearray / bytes）
        image_base64 = base64.b64encode(bytes(image_data)).decode("utf-8")
        
        # 呼叫 GPT-5 Vision 分析
        analysis = asyncio.run(analyze_food_image(image_base64))
        
        # 儲存到資料庫
        asyncio.run(db.add_meal(
            user_id=user_id,
            calories=analysis['calories'],
            protein=analysis['protein'],
            food_description=analysis['description']
        ))
        
        # 取得今日總計
        today_total = asyncio.run(db.get_today_total(user_id))
        
        # 組合回覆訊息
        reply_text = format_analysis_reply(analysis, today_total)
        
        # 回覆
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply_text)]
                )
            )
    
    except Exception as e:
        logger.exception("圖片分析失敗")
        reply = build_error_message("分析失敗", str(e))
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            line_bot_api.reply_message(
                ReplyMessageRequest(
                    reply_token=event.reply_token,
                    messages=[TextMessage(text=reply)]
                )
            )


async def analyze_food_image(image_base64: str) -> dict:
    """使用 GPT-5 Vision 分析食物圖片"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": """你是專業的營養師。分析圖片中的食物,估算:
1. 總熱量(kcal)
2. 蛋白質含量(g)
3. 食物描述

請以 JSON 格式回覆:
{
  "calories": 數字,
  "protein": 數字,
  "description": "詳細食物內容"
}

估算要準確,考慮份量大小。"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "請分析這份餐點的熱量與蛋白質"
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        # 解析回應（可能被包在 ```json ... ``` 中）
        result_text = (response.choices[0].message.content or "").strip()
        if "```" in result_text:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", result_text)
            if match:
                result_text = match.group(1).strip()
        result = json.loads(result_text)
        
        return {
            "calories": float(result.get("calories", 0)),
            "protein": float(result.get("protein", 0)),
            "description": result.get("description", "未知食物")
        }
    
    except Exception as e:
        logger.error(f"GPT 分析錯誤: {e}")
        raise


def format_analysis_reply(analysis: dict, today_total: dict) -> str:
    """格式化分析回覆訊息（結構化、易讀）"""
    protein_progress = (today_total["protein"] / 300) * 100
    
    message = f"""【本餐已記錄】

━━━ 本餐內容 ━━━
{analysis['description']}

━━━ 本餐營養 ━━━
  熱量    {analysis['calories']:.0f} kcal
  蛋白質  {analysis['protein']:.1f} g

━━━ 今日累計 ━━━
  總熱量    {today_total['calories']:.0f} kcal
  總蛋白質  {today_total['protein']:.1f} g / 300 g
  達成率    {protein_progress:.1f}%

{get_progress_bar(protein_progress)}

{get_quick_tip(today_total['protein'])}"""
    
    return message


def get_progress_bar(percentage: float) -> str:
    """產生進度條"""
    filled = int(percentage / 10)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}]"


def get_quick_tip(current_protein: float) -> str:
    """根據當前蛋白質提供建議"""
    remaining = 300 - current_protein
    
    if remaining <= 0:
        return "太棒了!已達成今日目標!"
    elif remaining <= 50:
        return f"再補充 {remaining:.0f}g 就達標囉!"
    elif remaining <= 100:
        return f"還需要 {remaining:.0f}g,可以吃一份雞胸肉(約50g蛋白質)"
    else:
        return f"還缺 {remaining:.0f}g,建議增加高蛋白食物攝取"


async def get_daily_summary(user_id: str) -> str:
    """取得每日總結（結構化、易讀）"""
    today_total = await db.get_today_total(user_id)
    meals = await db.get_today_meals(user_id)
    
    if not meals:
        return """【今日尚無記錄】

開始拍照上傳食物，我會幫你記錄熱量與蛋白質。"""
    
    protein_progress = (today_total["protein"] / 300) * 100
    
    meals_list = "\n".join([
        f"  {i+1}. {m.food_description}\n     蛋白質 {m.protein:.1f}g · 熱量 {m.calories:.0f}kcal"
        for i, m in enumerate(meals)
    ])
    
    summary = f"""【今日營養總結】

━━━ 今日餐點 ━━━
{meals_list}

━━━ 今日總計 ━━━
  總熱量    {today_total['calories']:.0f} kcal
  總蛋白質  {today_total['protein']:.1f} g / 300 g
  達成率    {protein_progress:.1f}%

{get_progress_bar(protein_progress)}

{generate_gap_filler(today_total['protein'])}"""
    
    return summary


def generate_gap_filler(current_protein: float) -> str:
    """產生 Gap Filler 建議"""
    remaining = 300 - current_protein
    
    if remaining <= 0:
        return "恭喜達標!明天繼續保持!"
    
    suggestions = []
    
    # 雞胸肉 (每100g約31g蛋白質)
    if remaining >= 50:
        chicken_amount = int((remaining / 31) * 100)
        suggestions.append(f"雞胸肉 {chicken_amount}g (約{remaining/31*100/100:.0f}份)")
    
    # 乳清蛋白 (每份約25g)
    whey_servings = int(remaining / 25)
    if whey_servings > 0:
        suggestions.append(f"乳清蛋白 {whey_servings} 份")
    
    # 雞蛋 (每顆約6g)
    eggs = int(remaining / 6)
    if eggs > 0 and eggs <= 10:
        suggestions.append(f"雞蛋 {eggs} 顆")
    
    tip = f"""Gap Filler 建議 (還缺 {remaining:.0f}g):

可選擇以下任一補充:
{chr(10).join(f"  - {s}" for s in suggestions)}

睡前記得補充!"""
    
    return tip


async def daily_summary_push():
    """每日 23:00 推播總結給所有用戶"""
    try:
        # 取得所有今日有記錄的用戶
        active_users = await db.get_active_users_today()
        
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            
            for user_id in active_users:
                summary = await get_daily_summary(user_id)
                
                try:
                    line_bot_api.push_message(
                        PushMessageRequest(
                            to=user_id,
                            messages=[TextMessage(text=f"晚間總結\n\n{summary}")]
                        )
                    )
                    logger.info(f"推播成功: {user_id}")
                except Exception as e:
                    logger.error(f"推播失敗 {user_id}: {e}")
        
        logger.info(f"每日推播完成,共 {len(active_users)} 位用戶")
    
    except Exception as e:
        logger.error(f"每日推播錯誤: {e}")


def generate_weekly_chart_image(weekly_data: list) -> bytes:
    """產生本週每日蛋白質長條圖，回傳 PNG bytes"""
    if not HAS_MATPLOTLIB:
        raise RuntimeError("未安裝 matplotlib")
    dates = [f"{d['date'][5:]}\n週{d['weekday']}" for d in weekly_data]
    proteins = [float(d["protein"]) for d in weekly_data]
    target = 300.0
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(dates, proteins, color="#4CAF50", edgecolor="#2E7D32", linewidth=1.2)
    ax.axhline(y=target, color="#F44336", linestyle="--", linewidth=1.5, label=f"目標 {target}g")
    ax.set_ylabel("蛋白質 (g)", fontsize=11)
    ax.set_title("本週每日蛋白質攝取量", fontsize=13)
    ax.set_ylim(0, max(max(proteins) * 1.2 if proteins else 1, target * 1.1))
    ax.legend(loc="upper right", fontsize=9)
    for i, (bar, val) in enumerate(zip(bars, proteins)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{val:.0f}g", ha="center", fontsize=9)
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def prepare_weekly_chart(user_id: str) -> tuple:
    """取得本週資料、產生圖表、存入快取。回傳 (token, chart_url, text)；失敗時 token 與 chart_url 為 None。"""
    if not HAS_MATPLOTLIB:
        return (None, None, "週報圖表功能尚未啟用（需安裝 matplotlib）。")
    base = settings.BASE_URL.rstrip("/")
    try:
        weekly_data = await db.get_weekly_protein_by_day(user_id)
        if not weekly_data:
            return (None, None, "本週尚無記錄，先拍照記錄幾餐再來看週報吧！")
        total = sum(d["protein"] for d in weekly_data)
        if total == 0:
            return (None, None, "本週尚無營養記錄，開始拍照記錄後再試「週報」。")
        png_bytes = generate_weekly_chart_image(weekly_data)
        token = str(uuid.uuid4())
        chart_cache[token] = png_bytes
        chart_url = f"{base}/chart/{token}"
        summary_line = f"本週總蛋白質：{total:.0f}g（目標 300g/日）"
        return (token, chart_url, summary_line)
    except Exception as e:
        logger.exception("週報圖表產生失敗")
        return (None, None, f"週報產生失敗，請稍後再試。\n（{str(e)[:80]}）")


def build_error_message(title: str, detail: str = "") -> str:
    """產生結構化、易讀的錯誤訊息"""
    lines = [
        "【" + title + "】",
        "",
        "可能原因：",
        "  照片模糊或非食物畫面",
        "  網路不穩，請稍後再試",
        "  服務忙碌，請再傳一次",
        "",
        "若持續失敗，請稍後再試或換一張照片。",
    ]
    if detail and "timeout" in detail.lower():
        lines.insert(-1, "  本次為連線逾時")
    return "\n".join(lines)


def get_help_message() -> str:
    """說明訊息（結構化、易理解）"""
    return """【飲控機器人 · 使用說明】

━━━ 主要功能 ━━━

  1. 拍照上傳食物
     自動分析熱量與蛋白質並記錄

  2. 輸入「今日」或「總計」
     查看今日營養數據與達成率

  3. 輸入「週報」
     查看本週每日蛋白質攝取圖表

  4. 每晚 23:00
     自動推播每日總結與建議

━━━ 其他指令 ━━━

  清除今日  刪除今日所有記錄
  說明/help  顯示此訊息

━━━━━━━━━━━━━━━━━━━━━━
  目標：每日蛋白質 300g
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
