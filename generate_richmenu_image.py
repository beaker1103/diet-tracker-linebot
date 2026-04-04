"""
產生圖文選單圖片（2500 x 1686）
使用 Pillow 繪製，不需外部圖片素材。
"""

from PIL import Image, ImageDraw, ImageFont
import os

WIDTH = 2500
HEIGHT = 1686
COLS = 3
ROWS = 2
CELL_W = WIDTH // COLS
CELL_H = HEIGHT // ROWS

# 深色系配色
BG_COLOR = (18, 18, 24)
GRID_COLOR = (45, 45, 60)
ACCENT_COLORS = [
    (255, 183, 77),   # 食物查詢 - 琥珀
    (129, 199, 132),  # 本週積分 - 綠
    (100, 181, 246),  # InBody - 藍
    (240, 98, 146),   # 欺騙日 - 粉紅
    (186, 104, 200),  # AI教練 - 紫
    (255, 255, 255),  # 今日總結 - 白
]

LABELS = [
    ("食物查詢", "拍照查熱量等級"),
    ("本週積分", "飲食控制成績"),
    ("InBody", "上傳體組成報告"),
    ("欺騙日", "啟動放鬆模式"),
    ("AI教練", "個人化建議"),
    ("今日總結", "查看今日數據"),
]

ICONS = [
    "?",   # 搜尋
    "S",   # Score
    "B",   # Body
    "!",   # 驚嘆
    "AI",  # AI
    "#",   # 數據
]


def generate():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # 嘗試載入字體
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]

    title_font = None
    sub_font = None
    icon_font = None

    for fp in font_paths:
        if os.path.exists(fp):
            try:
                title_font = ImageFont.truetype(fp, 64)
                sub_font = ImageFont.truetype(fp, 36)
                icon_font = ImageFont.truetype(fp, 120)
                break
            except Exception:
                continue

    if not title_font:
        # Fallback: 使用預設字體
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
            sub_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            icon_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 120)
        except Exception:
            title_font = ImageFont.load_default()
            sub_font = title_font
            icon_font = title_font

    for idx in range(6):
        row = idx // COLS
        col = idx % COLS
        x0 = col * CELL_W
        y0 = row * CELL_H

        accent = ACCENT_COLORS[idx]
        label, sublabel = LABELS[idx]
        icon = ICONS[idx]

        # 格子背景 (微漸層效果)
        for y in range(y0, y0 + CELL_H):
            ratio = (y - y0) / CELL_H
            r = int(BG_COLOR[0] + (accent[0] - BG_COLOR[0]) * ratio * 0.08)
            g = int(BG_COLOR[1] + (accent[1] - BG_COLOR[1]) * ratio * 0.08)
            b = int(BG_COLOR[2] + (accent[2] - BG_COLOR[2]) * ratio * 0.08)
            draw.line([(x0, y), (x0 + CELL_W, y)], fill=(r, g, b))

        # 邊框
        draw.rectangle([x0, y0, x0 + CELL_W - 1, y0 + CELL_H - 1],
                       outline=GRID_COLOR, width=2)

        # 頂部裝飾線
        draw.rectangle([x0 + 40, y0 + 20, x0 + CELL_W - 40, y0 + 24],
                       fill=(*accent, ))

        cx = x0 + CELL_W // 2
        cy = y0 + CELL_H // 2

        # 圖示圓圈
        circle_r = 80
        circle_y = cy - 80
        draw.ellipse(
            [cx - circle_r, circle_y - circle_r,
             cx + circle_r, circle_y + circle_r],
            outline=accent, width=4,
        )

        # 圖示文字
        icon_bbox = draw.textbbox((0, 0), icon, font=icon_font if len(icon) <= 2 else sub_font)
        iw = icon_bbox[2] - icon_bbox[0]
        ih = icon_bbox[3] - icon_bbox[1]
        draw.text(
            (cx - iw // 2, circle_y - ih // 2 - 10),
            icon,
            fill=accent,
            font=icon_font if len(icon) <= 2 else sub_font,
        )

        # 標題
        title_bbox = draw.textbbox((0, 0), label, font=title_font)
        tw = title_bbox[2] - title_bbox[0]
        draw.text(
            (cx - tw // 2, cy + 60),
            label,
            fill=(255, 255, 255),
            font=title_font,
        )

        # 副標題
        sub_bbox = draw.textbbox((0, 0), sublabel, font=sub_font)
        sw = sub_bbox[2] - sub_bbox[0]
        draw.text(
            (cx - sw // 2, cy + 140),
            sublabel,
            fill=(160, 160, 180),
            font=sub_font,
        )

    img.save("richmenu_image.png", "PNG")
    print(f"圖片已產生: richmenu_image.png ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    generate()
