"""
LINE 圖文選單建立腳本
執行一次即可，會建立 6 格選單並設為預設。
圖片需要另外準備（2500 x 1686 px）。
"""

import os
import sys
from dotenv import load_dotenv

from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    RichMenuRequest,
    RichMenuSize,
    RichMenuArea,
    RichMenuBounds,
    MessageAction,
)

load_dotenv()
TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")


def create_rich_menu():
    configuration = Configuration(access_token=TOKEN)

    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        blob_api = MessagingApiBlob(api_client)

        menu = RichMenuRequest(
            size=RichMenuSize(width=2500, height=1686),
            selected=True,
            name="飲食監測選單",
            chat_bar_text="點擊開啟功能選單",
            areas=[
                # ┌───────────┬───────────┬───────────┐
                # │  購買查詢  │ 加蛋白飲   │  InBody   │
                # ├───────────┼───────────┼───────────┤
                # │  欺騙日   │ 加一份碳水 │  今日總計  │
                # └───────────┴───────────┴───────────┘

                # 左上: 食物查詢（與 main.py「購買查詢」流程相同）
                RichMenuArea(
                    bounds=RichMenuBounds(x=0, y=0, width=833, height=843),
                    action=MessageAction(label="食物查詢", text="食物查詢"),
                ),
                # 中上: 快速加蛋白飲（本週積分改為文字指令）
                RichMenuArea(
                    bounds=RichMenuBounds(x=834, y=0, width=833, height=843),
                    action=MessageAction(label="加蛋白飲", text="加蛋白飲"),
                ),
                # 右上: InBody
                RichMenuArea(
                    bounds=RichMenuBounds(x=1667, y=0, width=833, height=843),
                    action=MessageAction(label="上傳InBody", text="上傳InBody"),
                ),
                # 左下: 欺騙日
                RichMenuArea(
                    bounds=RichMenuBounds(x=0, y=843, width=833, height=843),
                    action=MessageAction(label="欺騙日", text="欺騙日"),
                ),
                # 中下: 加一份碳水（與 main.py「加碳水」快速記錄相同邏輯）
                RichMenuArea(
                    bounds=RichMenuBounds(x=834, y=843, width=833, height=843),
                    action=MessageAction(label="加一份碳水", text="加一份碳水"),
                ),
                # 右下: 今日總結（與 main.py「今日」相同）
                RichMenuArea(
                    bounds=RichMenuBounds(x=1667, y=843, width=833, height=843),
                    action=MessageAction(label="今日總結", text="今日總結"),
                ),
            ],
        )

        # 建立選單
        result = api.create_rich_menu(rich_menu_request=menu)
        menu_id = result.rich_menu_id
        print(f"選單已建立: {menu_id}")

        # 上傳圖片
        image_path = "richmenu_image.png"
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                blob_api.set_rich_menu_image(
                    rich_menu_id=menu_id,
                    body=f.read(),
                    _content_type="image/png",
                )
            print("圖片已上傳")
        else:
            print(f"警告: 找不到 {image_path}")
            print("請準備 2500x1686 的 PNG 圖片後手動上傳")

        # 設為預設
        api.set_default_rich_menu(rich_menu_id=menu_id)
        print("已設為預設選單")

        return menu_id


def list_rich_menus():
    configuration = Configuration(access_token=TOKEN)
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        menus = api.get_rich_menu_list()
        for m in menus.richmenus:
            print(f"  ID: {m.rich_menu_id}  名稱: {m.name}")


def delete_all_rich_menus():
    configuration = Configuration(access_token=TOKEN)
    with ApiClient(configuration) as api_client:
        api = MessagingApi(api_client)
        menus = api.get_rich_menu_list()
        for m in menus.richmenus:
            api.delete_rich_menu(rich_menu_id=m.rich_menu_id)
            print(f"已刪除: {m.rich_menu_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "list":
            list_rich_menus()
        elif cmd == "delete":
            delete_all_rich_menus()
        elif cmd == "create":
            create_rich_menu()
        else:
            print("用法: python create_richmenu.py [create|list|delete]")
    else:
        create_rich_menu()
