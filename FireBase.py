# -*- coding: utf-8 -*-
"""
FireBase_Attention_LSTM_Direction.py
- Attention-LSTM
- Multi-task: Return path + Direction
- 小資料友善、無 leakage
- 回測圖：昨日預測 vs 今日實際（折線）
"""

import os, json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Softmax, Lambda
from tensorflow.keras.callbacks import EarlyStopping

import firebase_admin
from firebase_admin import credentials, firestore

# ================= Firebase 初始化 =================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(cred)
    db = firestore.client()

# ================= Firestore 讀取 =================
def load_df_from_firestore(ticker, collection="NEW_stock_data_liteon", days=500):
    rows = []
    for doc in db.collection(collection).stream():
        p = doc.to_dict().get(ticker)
        if p:
            rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ Firestore 無資料")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").tail(days).set_index("date")

# ================= 假日補今天 =================
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    if df.index.max() < today:
        df.loc[today] = df.iloc[-1]
    return df.sort_index()

# ================= Feature =================
def add_features(df):
    df["Volume"] = np.log1p(df["Volume"].astype(float))
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence =================
def create_sequences(df, features, steps=5, window=40):
    X, y_ret, y_dir, idx = [], [], [], []
    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(window, len(df) - steps):
        x = feat[i - window:i]
        r = logret.iloc[i:i + steps].values
        if np.any(np.isnan(x)) or np.any(np.isnan(r)):
            continue
        X.append(x)
        y_ret.append(r)
        y_dir.append(float(r.sum() > 0))
        idx.append(df.index[i])

    return np.array(X), np.array(y_ret), np.array(y_dir), np.array(idx)

# ================= Model =================
def build_attention_lstm(input_shape, steps, max_daily_logret=0.06):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inp)
    x = Dropout(0.2)(x)

    score = Dense(1)(x)
    weights = Softmax(axis=1)(score)
    context = Lambda(lambda t: tf.reduce_sum(t[0] * t[1], axis=1))([x, weights])

    raw = Dense(steps, activation="tanh")(context)
    out_ret = Lambda(lambda t: t * max_daily_logret, name="return")(raw)
    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(7e-4),
        loss={"return": tf.keras.losses.Huber(), "direction": "binary_crossentropy"},
        loss_weights={"return": 1.0, "direction": 0.4},
        metrics={"direction": ["accuracy"]}
    )
    return model

# ================= 預測圖（不變） =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)
    x_hist = np.arange(len(hist))
    x_future = np.arange(len(hist), len(hist) + len(future_df))

    plt.figure(figsize=(18,8))
    plt.plot(x_hist, hist["Close"], label="Close")
    plt.plot(x_hist, hist["SMA5"], label="SMA5")
    plt.plot(x_hist, hist["SMA10"], label="SMA10")

    today_x = x_hist[-1]
    today_y = hist["Close"].iloc[-1]
    plt.scatter(today_x, today_y, s=120, marker="*", label="Today")

    plt.plot(
        np.concatenate([[today_x], x_future]),
        [today_y] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    plt.legend()
    plt.grid(True)
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png", dpi=300)
    plt.close()

# ================= 回測圖：昨日預測 vs 今日實際（折線） =================
def plot_yesterday_forecast_vs_today(df):
    today = df.index[-1]
    today_close = float(df["Close"].iloc[-1])

    yday = today - BDay(1)
    fc_path = f"results/{yday:%Y-%m-%d}_forecast.csv"
    if not os.path.exists(fc_path):
        print("⚠️ 找不到昨日 forecast.csv")
        return

    fc = pd.read_csv(fc_path)
    pred_price = float(fc.iloc[0]["Pred_Close"])
    pred_date = pd.to_datetime(fc.iloc[0]["date"])

    plt.figure(figsize=(8,5))
    plt.plot(
        [pred_date, today],
        [pred_price, today_close],
        "o-",
        label="Yesterday Forecast → Today Actual"
    )

    plt.text(pred_date, pred_price, f"{pred_price:.2f}", ha="center", va="bottom")
    plt.text(today, today_close, f"{today_close:.2f}", ha="center", va="bottom")

    plt.title("Backtest: Yesterday Forecast vs Today Actual")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{today:%Y-%m-%d}_backtest.png", dpi=300)
    plt.close()

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER)
    df = ensure_today_row(df)
    df = add_features(df)

    FEATURES = ["Close", "Volume", "RSI", "MACD", "K", "D", "ATR_14"]
    df = df.dropna()

    X, y_ret, y_dir, idx = create_sequences(df, FEATURES, STEPS, LOOKBACK)
    split = int(len(X) * 0.85)

    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_dir_tr = y_ret[:split], y_dir[:split]

    scaler = MinMaxScaler()
    scaler.fit(df.loc[:idx[split-1], FEATURES].values)

    def scale(X):
        n,t,f = X.shape
        return scaler.transform(X.reshape(-1,f)).reshape(n,t,f)

    X_tr_s = scale(X_tr)
    X_te_s = scale(X_te)

    model = build_attention_lstm((LOOKBACK, len(FEATURES)), STEPS)
    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=80,
        batch_size=16,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=2
    )

    pred_ret, pred_dir = model.predict(X_te_s, verbose=0)
    raw = pred_ret[-1]

    price = df["Close"].iloc[-1]
    prices = []
    for r in raw:
        price *= np.exp(r)
        prices.append(price)

    seq = df["Close"].iloc[-10:].tolist()
    future = []
    for p in prices:
        seq.append(p)
        future.append({
            "Pred_Close": p,
            "Pred_MA5": np.mean(seq[-5:]),
            "Pred_MA10": np.mean(seq[-10:])
        })

    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(df.index[-1] + BDay(1), periods=STEPS)

    os.makedirs("results", exist_ok=True)
    future_df.to_csv(f"results/{datetime.now():%Y-%m-%d}_forecast.csv",
                     index=False, encoding="utf-8-sig")

    plot_and_save(df, future_df)
    plot_yesterday_forecast_vs_today(df)
