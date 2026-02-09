# 高強度減重監測 LINE Bot

## 專案概述
使用 FastAPI + GPT-5 Vision 建立的智能飲食追蹤 LINE Bot,支援:
- 拍照辨識食物熱量與蛋白質
- 每日數據追蹤(目標蛋白質 300g)
- 晚上 23:00 自動推播總結
- 智能營養建議

## 技術堆疊
- 後端: Python 3.11 + FastAPI
- AI: OpenAI GPT-5 Vision API
- 訊息: LINE Messaging API
- 資料庫: SQLite
- 部署: Render / Fly.io

## 快速開始

### 1. 環境準備
```bash

### 2. 設定環境變數
```bash
### 3. 本地測試
```bash
cat > README.md << 'EOF'
# 高強度減重監測 LINE Bot

## 專案概述
使用 FastAPI + GPT-5 Vision 建立的智能飲食追蹤 LINE Bot,支援:
- 拍照辨識食物熱量與蛋白質
- 每日數據追蹤(目標蛋白質 300g)
- 晚上 23:00 自動推播總結
- 智能營養建議

## 技術堆疊
- 後端: Python 3.11 + FastAPI
- AI: OpenAI GPT-5 Vision API
- 訊息: LINE Messaging API
- 資料庫: SQLite
- 部署: Render / Fly.io

## 快速開始

### 1. 環境準備
```bash
# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安裝套件
pip install -r requirements.txt
```

### 2. 設定環境變數
```bash
cp .env.example .env
# 編輯 .env 填入你的 API keys
```

### 3. 本地測試
```bash
# 啟動服務
python main.py

# 開新終端,使用 ngrok 建立公開網址
ngrok http 8000
```

### 4. 設定 LINE Webhook
1. 前往 LINE Developers Console
2. 建立 Messaging API Channel
3. 設定 Webhook URL: https://你的ngrok網址/webhook
4. 關閉自動回覆,啟用 Webhook

## 每月成本估算
- OpenAI GPT-5 API: 約 $0.64/月
- LINE API: 免費 (500則/月內)
- 雲端託管: 免費
**總計: 約 NT$20/月**

## 功能說明
- 傳送食物照片 → 自動分析
- 輸入「今日」→ 查看總計
- 輸入「說明」→ 使用說明
- 輸入「清除今日」→ 刪除記錄
- 每晚 23:00 自動推播

## 授權
MIT License
