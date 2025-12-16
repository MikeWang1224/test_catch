# -*- coding: utf-8 -*-
"""
個股資料抓取 + 技術指標計算 + Firestore 更新與寫回
✅ 今日 Close 先覆寫，再重新計算指標（一致性修正版）
➕ 加入加權指數 / 電子指數（Close only）
不含模型、不含預測、不含繪圖
"""

import os
import json
import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------- Firebase 初始化 ----------------
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    print("⚠️ FIREBASE 未設定，Firestore 寫入將略過")

# ---------------- 技術指標計算（全集中） ----------------
def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(20).mean()
    avg_loss = loss.rolling(20).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    low14 = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    denom = high14 - low14
    df["K"] = np.where(denom == 0, 50.0, 100 * (df["Close"] - low14) / denom)
    df["D"] = df["K"].rolling(3).mean()

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["SignalLine"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["RET_1"] = df["Close"].pct_change()
    df["LOG_RET_1"] = np.log(df["Close"] / df["Close"].shift(1))

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_mid"] = mid
    df["BB_upper"] = mid + 2 * std
    df["BB_lower"] = mid - 2 * std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / mid

    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_SMA_20"] = df["OBV"].rolling(20).mean()

    df["Vol_SMA_5"] = df["Volume"].rolling(5).mean()
    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()

    return df.dropna()

# ---------------- Firestore 覆寫今日 Close ----------------
def overwrite_today_close(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if db is None:
        return df

    today = datetime.now().strftime("%Y-%m-%d")
    try:
        doc = db.collection("NEW_stock_data_liteon").document(today).get()
        if doc.exists:
            payload = doc.to_dict().get(ticker, {})
            if "Close" in payload:
                ts = pd.Timestamp(today)
                if ts in df.index:
                    df.loc[ts, "Close"] = float(payload["Close"])
                    print(f"✔ Firestore 覆寫今日 Close：{payload['Close']}")
    except Exception as e:
        print(f"⚠️ 今日 Close 覆寫失敗：{e}")

    return df

# ---------------- 抓個股 ----------------
def fetch_prepare_recalc(ticker="2301.TW", period="12mo") -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    df = overwrite_today_close(df, ticker)
    df = add_all_indicators(df)
    return df

# ---------------- Firestore 寫個股 ----------------
def save_to_firestore(df: pd.DataFrame, ticker="2301.TW", collection="NEW_stock_data_liteon"):
    if db is None:
        return

    batch = db.batch()
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        payload = {
            "Open": float(row["Open"]),
            "High": float(row["High"]),
            "Low": float(row["Low"]),
            "Close": float(row["Close"]),
            "Volume": float(row["Volume"]),
            "MACD": float(row["MACD"]),
            "RSI": float(row["RSI"]),
            "K": float(row["K"]),
            "D": float(row["D"]),
            "ATR_14": float(row["ATR_14"]),
        }
        doc_ref = db.collection(collection).document(date_str)
        batch.set(doc_ref, {ticker: payload}, merge=True)

    batch.commit()
    print(f"🔥 Firestore 寫入完成：{ticker}")

# ---------------- ➕ 指數抓取（Close only） ----------------
def save_index_close(ticker: str, alias: str, period="12mo"):
    if db is None:
        return

    df = yf.Ticker(ticker).history(period=period)
    for idx, row in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        doc_ref = db.collection("NEW_stock_data_liteon").document(date_str)
        doc_ref.set({alias: {"Close": float(row["Close"])}}, merge=True)

    print(f"🔥 指數寫入完成：{alias}")

# ---------------- Main ----------------
if __name__ == "__main__":
    df = fetch_prepare_recalc("2301.TW")
    save_to_firestore(df, "2301.TW")

    # ➕ 加權指數 / 電子指數
    save_index_close("^TWII", "TAIEX")
    save_index_close("^TWTE", "ELECTRONICS")

    print(df.tail())
