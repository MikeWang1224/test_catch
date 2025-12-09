import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
from datetime import datetime
import json
import os

# 🔐 讀取 Firebase 服務帳戶金鑰（環境變數方式）
key_dict = json.loads(os.environ["FIREBASE"])
cred = credentials.Certificate(key_dict)
firebase_admin.initialize_app(cred)
db = firestore.client()

# 📌 只抓光寶科 (2301.TW)
ticker_symbol = "2301.TW"
liteon = yf.Ticker(ticker_symbol)
df_liteon = liteon.history(period="6mo")   # 可改成 1y, 3mo, max 等

# 📈 計算技術指標
def calculate_indicators(df):
    # SMA
    df['SMA_5'] = df['Close'].rolling(window=5).mean().round(5)
    df['SMA_10'] = df['Close'].rolling(window=10).mean().round(5)
    df['SMA_50'] = df['Close'].rolling(window=50).mean().round(5)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=20).mean()
    avg_loss = loss.rolling(window=20).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = (100 - (100 / (1 + rs))).round(5)

    # KD
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    df['K'] = (100 * (df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14'])).round(5)
    df['D'] = df['K'].rolling(window=3).mean().round(5)

    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (df['EMA_12'] - df['EMA_26']).round(5)
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean().round(5)

    return df

df_liteon = calculate_indicators(df_liteon)

# 🔎 要儲存的欄位
selected_columns = ['Close', 'MACD', 'RSI', 'K', 'D', 'Volume']

# 🔥 Firebase Collection 名稱
collection_name = "NEW_stock_data_liteon"

# 💾 儲存到 Firestore（以日期為 doc id）
def save_data():
    batch = db.batch()
    count = 0

    for idx, row in df_liteon.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        data = {col: round(float(row[col]), 5) for col in selected_columns if not pd.isna(row[col])}

        doc_ref = db.collection(collection_name).document(date_str)
        batch.set(doc_ref, {"2301.TW": data})
        count += 1

        if count >= 300:  # 批次寫入避免 timeout
            batch.commit()
            print(f"批次寫入 {count} 筆")
            batch = db.batch()
            count = 0

    if count > 0:
        batch.commit()
        print(f"剩餘 {count} 筆已寫入")

    print(" 🎉 光寶科股票數據已成功寫入 Firestore！")

# ▶️ 執行儲存
save_data()
