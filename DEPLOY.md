# 免費上線：Render 部署 LINE Bot（完整操作步驟）

本指南用 **Render 免費方案** 把 LINE Bot 放上雲端，不需信用卡，每月 750 小時足夠 24 小時運行。

---

## 三步驟總覽

1. **程式推到 GitHub**：在專案目錄執行 `git push`，讓 Render 能讀到你的程式。
2. **Render 建立 Web Service**：登入 [render.com](https://render.com) → New + → Web Service → 連你的 GitHub repo → 填 Build/Start 指令與環境變數（LINE、OpenAI、BASE_URL）→ Create。
3. **LINE 改 Webhook**：到 LINE Developers Console → 你的 Channel → Messaging API → Webhook URL 改成 `https://你的Render網址/webhook` → Verify → 關閉自動回覆、開啟 Webhook。

完成後 Bot 就在雲端跑。建議再設定「保持喚醒」（第四節），避免冷啟動延遲。

---

## 免費方案須知

| 項目 | 說明 |
|------|------|
| 費用 | 完全免費，不需信用卡 |
| 運行時間 | 每月 750 小時，單一服務 24/7 足夠 |
| 冷啟動 | 約 15 分鐘沒人使用會休眠，下次請求約 30–50 秒才回應；可透過「保持喚醒」避免 |
| 資料 | 免費方案重啟或重新部署後，SQLite 記錄會清空（僅試用/個人用可接受） |

---

## 一、事前準備（約 5 分鐘）

### 1.1 申請 / 登入 GitHub

1. 打開 [https://github.com](https://github.com)
2. 若沒有帳號：點 **Sign up** 註冊
3. 登入後備用

### 1.2 把專案推到 GitHub

在終端機（Terminal）執行（路徑請改成你的專案目錄）：

```bash
cd /Users/beaker1103/Desktop/diet-tracker-linebot

# 若還沒初始化 git
git init
git add .
git commit -m "Initial commit for Render deploy"

# 在 GitHub 網頁先建一個「空的」repo（不要勾選 README）
# 然後執行（請把 你的帳號 換成你的 GitHub 使用者名稱）：
git remote add origin https://github.com/你的帳號/diet-tracker-linebot.git
git branch -M main
git push -u origin main
```

若本來就有 `origin` 且已 push 過，只要確保最新程式有 push 即可：

```bash
git add .
git commit -m "Update for Render free deploy"
git push
```

---

## 二、在 Render 建立免費 Web Service（約 5 分鐘）

### 2.1 登入 Render

1. 打開 [https://render.com](https://render.com)
2. 點 **Get Started for Free**
3. 選 **Sign in with GitHub**，授權 Render 讀取你的 GitHub

### 2.2 建立 Web Service

1. 登入後在 **Dashboard** 點 **New +**
2. 選 **Web Service**
3. **Connect a repository**：
   - 若列表沒有你的 repo，點 **Configure account** 勾選 `diet-tracker-linebot`（或你 repo 名稱）的讀取權限
   - 選 **diet-tracker-linebot**（或你的 repo 名稱），再點 **Connect**

### 2.3 填寫設定（照抄即可）

| 欄位 | 請填寫 |
|------|--------|
| **Name** | `diet-tracker-linebot`（或自訂名稱，之後會變成網址的一部分） |
| **Region** | **Singapore**（或任選一區） |
| **Branch** | `main` |
| **Runtime** | **Python 3** |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

**不要**勾選付費方案，維持免費。

### 2.4 設定環境變數（必填）

在 **Environment Variables** 區塊點 **Add Environment Variable**，一筆一筆加：

| Key | Value |
|-----|--------|
| `LINE_CHANNEL_SECRET` | 你的 LINE Channel Secret（從 LINE Developers 複製） |
| `LINE_CHANNEL_ACCESS_TOKEN` | 你的 LINE Channel Access Token |
| `OPENAI_API_KEY` | 你的 OpenAI API Key（開頭通常是 `sk-...`） |
| `BASE_URL` | 先填 `https://diet-tracker-linebot.onrender.com`（若 Name 有改，網址會是 `https://你的Name.onrender.com`） |

**說明：**  
- `BASE_URL` 用來產生週報圖表連結，部署成功後會得到真實網址，再回來改一次即可（見下方 2.6）。

### 2.5 建立服務

1. 檢查上面欄位與環境變數都填好
2. 點 **Create Web Service**
3. 等 2–5 分鐘，看 **Logs** 出現類似 `Uvicorn running on ...` 即代表啟動成功

### 2.6 記下網址並修正 BASE_URL

1. 頁面頂部會顯示 **Your service is live at** 底下的網址，例如：  
   `https://diet-tracker-linebot.onrender.com`
2. 複製這個網址
3. 到 **Environment** 分頁，找到 `BASE_URL`，把 Value 改成**剛剛複製的網址**（不要加 `/` 結尾）
4. 存檔後 Render 會自動重新部署（等 1–2 分鐘）

---

## 三、設定 LINE Webhook（約 2 分鐘）

1. 打開 [LINE Developers Console](https://developers.line.biz/console/)
2. 選你的 **Provider** → 點進你的 **Messaging API Channel**
3. 切到 **Messaging API** 分頁
4. 找到 **Webhook URL**：
   - 點 **Edit**
   - 填上：`https://你的Render網址/webhook`  
     例如：`https://diet-tracker-linebot.onrender.com/webhook`
   - 按 **Update**
5. 點 **Verify**，若成功會顯示「Success」
6. 同一頁往下：
   - **Auto-reply messages** 設為 **Disabled**
   - **Webhook** 設為 **Enabled**

完成後，用手機對 Bot 傳一句「說明」或傳一張照片測試即可。

---

## 四、避免冷啟動（可選，建議做）

免費方案約 **15 分鐘沒有請求** 會休眠，下次傳訊息可能要等 30–50 秒 Bot 才回。若希望一傳就回，可讓外部定期呼叫你的網址「保持喚醒」。

### 4.1 用 cron-job.org（免費）

1. 打開 [https://cron-job.org](https://cron-job.org)，註冊一個帳號
2. 登入後 **Create cronjob**：
   - **Title**：`LINE Bot 喚醒`（隨意）
   - **URL**：`https://你的Render網址/`  
     例如：`https://diet-tracker-linebot.onrender.com/`
   - **Schedule**：每 10 分鐘一次，例如選 **Every 10 minutes**
3. 儲存後，約每 10 分鐘會打一次你的首頁，服務就不會進入休眠，LINE 傳訊息時幾乎不會遇到冷啟動。

### 4.2 用 UptimeRobot（免費）

1. 打開 [https://uptimerobot.com](https://uptimerobot.com)，註冊
2. **Add New Monitor**：
   - **Monitor Type**：HTTP(s)
   - **Friendly Name**：`LINE Bot`
   - **URL**：`https://你的Render網址/`
   - **Monitoring Interval**：選 5 分鐘
3. 儲存後，每 5 分鐘會檢查一次，同樣能減少冷啟動。

---

## 五、操作檢查清單

| 步驟 | 確認項目 |
|------|----------|
| 1 | 程式已 push 到 GitHub（含 `requirements.txt`、`main.py`、`render.yaml`） |
| 2 | Render 已建立 Web Service，Build / Start 成功（Logs 無紅字錯誤） |
| 3 | 環境變數 `LINE_CHANNEL_SECRET`、`LINE_CHANNEL_ACCESS_TOKEN`、`OPENAI_API_KEY`、`BASE_URL` 都已填 |
| 4 | `BASE_URL` 與瀏覽器打開的 Render 網址一致（可開 `https://你的網址/` 看到 `{"status":"ok",...}`） |
| 5 | LINE Developers 的 Webhook URL 為 `https://你的網址/webhook`，且 Verify 成功 |
| 6 | LINE 後台「Auto-reply」關閉、「Webhook」開啟 |
| 7 | （可選）cron-job.org 或 UptimeRobot 已設定每 5–10 分鐘打一次 `https://你的網址/` |

---

## 六、之後要更新程式怎麼做？

1. 在本機改程式
2. 終端機執行：
   ```bash
   git add .
   git commit -m "更新說明"
   git push
   ```
3. 到 Render Dashboard 點進你的 **diet-tracker-linebot** 服務，會自動偵測到 push 並重新部署（看 **Logs** 等 Build / Start 完成即可）

---

## 七、若之後想用付費方案或換平台

- **Render 付費**：可加 **Persistent Disk** 讓 SQLite 資料在重啟後保留。
- **Fly.io**：專案內有 `fly.toml`，可照 [Fly 官方文件](https://fly.io/docs/) 部署，並用 Volume 存資料。
- **DEPLOY.md** 原本的 Fly.io 章節仍可當作付費/進階部署參考。

目前用免費 Render 照上述步驟做完，LINE Bot 就會在雲端 24 小時運行，不開筆電也能用。
