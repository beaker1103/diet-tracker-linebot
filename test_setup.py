"""
環境檢查腳本 - 確認所有設定都正確
執行: python test_setup.py
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """檢查 Python 版本"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print("✅ Python 版本正確:", f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print("❌ Python 版本過舊,需要 3.11+,目前:", f"{version.major}.{version.minor}.{version.micro}")
        return False

def check_env_file():
    """檢查 .env 檔案"""
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env 檔案存在")
        
        with open(env_path) as f:
            content = f.read()
            
        required_vars = [
            "LINE_CHANNEL_SECRET",
            "LINE_CHANNEL_ACCESS_TOKEN",
            "OPENAI_API_KEY"
        ]
        
        missing = []
        for var in required_vars:
            if var not in content or f"{var}=your" in content or f"{var}=sk-your" in content:
                missing.append(var)
        
        if missing:
            print(f"⚠️  請在 .env 設定以下變數: {', '.join(missing)}")
            return False
        else:
            print("✅ 所有環境變數都已設定")
            return True
    else:
        print("❌ .env 檔案不存在,請執行: cp .env.example .env")
        return False

def check_dependencies():
    """檢查套件是否安裝"""
    required_packages = [
        "fastapi",
        "uvicorn",
        "linebot",
        "openai",
        "dotenv",
        "aiosqlite",
        "apscheduler"
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == "dotenv":
                __import__("dotenv")
            elif package == "linebot":
                __import__("linebot.v3")
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ 缺少套件,請執行: pip install -r requirements.txt")
        return False
    else:
        print("✅ 所有套件都已安裝")
        return True

def main():
    """執行所有檢查"""
    print("=" * 50)
    print("🔍 開始環境檢查...\n")
    
    results = []
    
    print("1️⃣ 檢查 Python 版本")
    results.append(check_python_version())
    print()
    
    print("2️⃣ 檢查套件安裝")
    results.append(check_dependencies())
    print()
    
    print("3️⃣ 檢查環境變數檔案")
    results.append(check_env_file())
    print()
    
    print("=" * 50)
    if all(results):
        print("🎉 所有檢查通過!可以開始使用了")
        print("\n下一步:")
        print("1. 本地測試: python main.py")
        print("2. 使用 ngrok: ngrok http 8000")
        print("3. 設定 LINE Webhook URL")
    else:
        print("⚠️  請修正上述問題後再試")
    print("=" * 50)

if __name__ == "__main__":
    main()
