# -*- coding: utf-8 -*-
"""
改良版：光寶科（2301.TW）多步 LSTM -> 預測未來 10 個交易日 Close，再計算 MA5/MA10
已加入：
 - 基本技術指標：SMA_5/SMA_10/SMA_50, RSI, K, D, MACD, SignalLine
 - 將歷史股票資料（Close, Volume, MACD, RSI, K, D）寫回 Firestore (collection: NEW_stock_data_liteon)
流程：
 - 抓資料 -> 計算指標 -> 寫回 Firestore -> LSTM 訓練/預測 -> 畫圖 -> 寫入預測到 Firestore
新增：
 - baseline 評估（last-close, last-SMA5 fallback, simple random-walk returns）
 - 圖片上傳至 Firebase Storage，並把 image_url 寫回 Firestore
"""
import os, json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math
import random

# ---------------- Firebase 初始化（含 Storage） ----------------
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
bucket = None
storage_client = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        # initialize_app with storageBucket ensures credentials scoped for storage operations
        firebase_admin.initialize_app(cred, {"storageBucket": f"{key_dict.get('project_id')}.appspot.com"})
    db = firestore.client()
    try:
        # google-cloud-storage uses application default credentials; the service account JSON loaded above
        storage_client = storage.Client.from_service_account_info(key_dict)
        bucket = storage_client.bucket(f"{key_dict.get('project_id')}.appspot.com")
    except Exception as e:
        print("⚠️ Storage client 初始化失敗，Storage 功能停用:", e)
        bucket = None
else:
    print("⚠️ FIREBASE env 未設定 — 會略過上傳步驟")

# ---------------- 傳統技術指標：SMA / RSI / KD / MACD ----------------
def add_basic_indicators(df):
    df = df.copy()

    # --- SMA ---
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # --- RSI (20) ---
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=20).mean()
    avg_loss = loss.rolling(window=20).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = (100 - (100 / (1 + rs)))

    # --- KD (14,3) ---
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    denom = (df['Highest_14'] - df['Lowest_14'])
    # avoid division by zero
    df['K'] = np.where(denom == 0, 50.0, 100 * (df['Close'] - df['Lowest_14']) / denom)
    df['D'] = df['K'].rolling(window=3).mean()

    # --- MACD (12,26,9) ---
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['SignalLine'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # round for neatness (optional)
    for col in ['SMA_5','SMA_10','SMA_50','RSI','K','D','MACD','SignalLine']:
        if col in df.columns:
            df[col] = df[col].round(5)

    return df

# ---------------- 其他特徵工程函式 ----------------
def add_technical_features(df):
    df = df.copy()
    # SMA (already computed in basic but keep for compatibility)
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()

    # returns & log returns
    df['RET_1'] = df['Close'].pct_change().fillna(0)
    df['LOG_RET_1'] = np.log(df['Close'] / df['Close'].shift(1)).fillna(0)

    # SMA diffs
    df['Close_minus_SMA5'] = df['Close'] - df['SMA_5']
    df['SMA5_minus_SMA10'] = df['SMA_5'] - df['SMA_10']

    # ATR
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(20).mean()
    df['BB_std'] = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - 2 * df['BB_std']
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_mid']

    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_SMA_20'] = df['OBV'].rolling(20).mean()

    # Volume MA
    df['Vol_SMA_5'] = df['Volume'].rolling(5).mean()
    df['Vol_SMA_20'] = df['Volume'].rolling(20).mean()

    df = df.dropna()
    return df

# ---------------- 取得資料並計指標 ----------------
def fetch_and_prepare(ticker="2301.TW", period="12mo"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    # first compute original technical features
    df = add_technical_features(df)
    # then add basic indicators (SMA/RSI/KD/MACD)
    df = add_basic_indicators(df)
    # drop any remaining NaN
    df = df.dropna()
    return df

# ---------------- 更新今天 Close 從 Firestore（若有） ----------------
def update_today_from_firestore(df):
    if db is None:
        return df
    today_str = datetime.now().strftime("%Y-%m-%d")
    try:
        doc_ref = db.collection("NEW_stock_data_liteon").document(today_str)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict().get("2301.TW", {})
            if "Close" in data:
                try:
                    df.loc[pd.Timestamp(today_str), 'Close'] = float(data["Close"])
                except Exception:
                    pass
    except Exception:
        # 若連線或讀取失敗，不影響後續流程
        pass
    df = df.dropna()
    return df

# ---------------- 寫入股票資料回 Firestore（歷史資料） ----------------
def save_stock_data_to_firestore(df, ticker="2301.TW", collection_name="NEW_stock_data_liteon"):
    if db is None:
        print("⚠️ Firebase 未啟用，略過寫入股票資料")
        return

    batch = db.batch()
    count = 0
    try:
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d")
            # construct payload; only include required fields
            payload = {}
            try:
                payload = {
                    "Close": float(row["Close"]),
                    "Volume": float(row["Volume"]),
                    "MACD": float(row["MACD"]),
                    "RSI": float(row["RSI"]),
                    "K": float(row["K"]),
                    "D": float(row["D"])
                }
            except Exception:
                # 若某欄位缺失，跳過該日
                continue

            doc_ref = db.collection(collection_name).document(date_str)
            batch.set(doc_ref, {ticker: payload})
            count += 1

            if count >= 300:
                batch.commit()
                batch = db.batch()
                count = 0

        if count > 0:
            batch.commit()

        print(f"🔥 歷史股票資料已寫入 Firestore （collection: {collection_name}）")
    except Exception as e:
        print("❌ 寫入 Firestore 發生錯誤：", e)

# ---------------- 建資料集 ----------------
def create_sequences(df, features, target_steps=10, window=60):
    X, y = [], []
    closes = df['Close'].values
    data = df[features].values
    for i in range(window, len(df) - target_steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+target_steps])
    return np.array(X), np.array(y)

# ---------------- 建模型 ----------------
def build_lstm_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mae')
    return model

# ---------------- 時序 split ----------------
def time_series_split(X, y, test_ratio=0.15):
    n = len(X)
    test_n = int(n * test_ratio)
    split_idx = n - test_n
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# ---------------- MA 計算 ----------------
def compute_pred_ma_from_pred_closes(last_known_closes, pred_closes):
    closes_seq = list(last_known_closes)
    results = []
    for pc in pred_closes:
        closes_seq.append(pc)
        ma5 = np.mean(closes_seq[-5:]) if len(closes_seq) >= 5 else np.mean(closes_seq)
        ma10 = np.mean(closes_seq[-10:]) if len(closes_seq) >= 10 else np.mean(closes_seq)
        results.append((pc, ma5, ma10))
    return results

# ---------------- 繪圖 + 上傳 Storage（修正版） ----------------
def plot_and_upload_to_storage(df_real, df_future, bucket_obj=None, hist_days=60):
    """
    畫圖並上傳至 Firebase Storage（如果 bucket_obj 提供）。
    回傳 public image url 或 None。
    修正：
      - 確保預測序列的日期與 labels 對齊（包含起始已知日期）
      - pred_table 與 labels 都使用同一份 df_future_plot（包含起始點）
    """
    df_real_plot = df_real.copy().tail(10)  # 顯示最近 10 日

    if df_real_plot.empty:
        print("⚠️ df_real_plot 為空，無法繪圖")
        return None

    df_future = df_future.copy().reset_index(drop=True)

    # 建立一個包含最後一個歷史日期 + future dates 的 df，用於繪圖 labels
    last_hist_date = df_real_plot.index[-1]
    start_row = {
        "date": last_hist_date,
        "Pred_Close": df_real_plot['Close'].iloc[-1],
        "Pred_MA5": df_real_plot['SMA_5'].iloc[-1] if 'SMA_5' in df_real_plot.columns else df_real_plot['Close'].iloc[-1],
        "Pred_MA10": df_real_plot['SMA_10'].iloc[-1] if 'SMA_10' in df_real_plot.columns else df_real_plot['Close'].iloc[-1]
    }
    df_future_plot = pd.concat([pd.DataFrame([start_row]), df_future], ignore_index=True)

    plt.figure(figsize=(16,8))

    # 歷史：畫近 10 日的 Close / SMA5 / SMA10
    x_real = list(range(len(df_real_plot)))
    plt.plot(x_real, df_real_plot['Close'].values, label="Close")
    if 'SMA_5' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_5'].values, label="SMA5")
    if 'SMA_10' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_10'].values, label="SMA10")

    # 預測（從最後一個歷史索引開始）
    offset = len(df_real_plot) - 1
    x_future = [offset + i for i in range(len(df_future_plot))]
    plt.plot(x_future, df_future_plot['Pred_Close'].values, linestyle=':', marker='o', label="Pred Close")
    plt.plot(x_future, df_future_plot['Pred_MA5'].values, linestyle='--', label="Pred MA5")
    plt.plot(x_future, df_future_plot['Pred_MA10'].values, linestyle='--', label="Pred MA10")

    # X 軸標籤
    # Build labels: history except last (since last is start_row), then all df_future_plot dates
    labels = []
    for d in df_real_plot.index[:-1]:
        labels.append(pd.Timestamp(d).strftime('%m-%d'))
    for d in df_future_plot['date']:
        labels.append(pd.Timestamp(d).strftime('%m-%d'))

    ticks = list(range(len(labels)))
    # Ensure ticks cover plotted range
    plt.xticks(ticks=ticks, labels=labels, rotation=45)
    plt.xlim(0, max(ticks))

    plt.legend()
    plt.title("2301.TW 歷史 + 預測（近 10 日 + 未來 10 日）")
    plt.xlabel("Date")
    plt.ylabel("Price")

    os.makedirs("results", exist_ok=True)
    file_name = f"{datetime.now().strftime('%Y-%m-%d')}_future_trade_days.png"
    file_path = os.path.join("results", file_name)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("📌 圖片已儲存：", file_path)

    # 若提供 bucket，則上傳並回傳 URL（try/catch）
    if bucket_obj is not None:
        try:
            blob = bucket_obj.blob(f"LSTM_Pred_Images/{file_name}")
            blob.upload_from_filename(file_path)
            public_url = None
            try:
                blob.make_public()
                public_url = blob.public_url
            except Exception:
                # 無法 make public（多為 storage policy）；仍回傳 blob.public_url 若可取得
                try:
                    public_url = blob.public_url
                except Exception:
                    public_url = None
            print("🔥 圖片已上傳至 Storage：", public_url)
            return public_url
        except Exception as e:
            print("❌ 上傳 Storage 發生錯誤：", e)
            return None

    return None

# ---------------- Baseline / MA helper functions ----------------
def compute_metrics(y_true, y_pred):
    maes = []
    rmses = []
    for step in range(y_true.shape[1]):
        maes.append(mean_absolute_error(y_true[:, step], y_pred[:, step]))
        rmses.append(math.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step])))
    return np.array(maes), np.array(rmses)


def compute_ma_from_predictions(last_known_window_closes, y_pred_matrix, ma_period=5):
    n_samples, window = last_known_window_closes.shape
    steps = y_pred_matrix.shape[1]
    preds_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_known_window_closes[i])  # copy
        for t in range(steps):
            seq.append(y_pred_matrix[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            preds_ma[i, t] = np.mean(look)
    return preds_ma


def compute_true_ma(last_window, y_true, ma_period=5):
    n_samples, window = last_window.shape
    steps = y_true.shape[1]
    true_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_window[i])
        for t in range(steps):
            seq.append(y_true[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            true_ma[i, t] = np.mean(look)
    return true_ma

# ---------------- 主流程 ----------------
if __name__ == "__main__":
    TICKER = "2301.TW"
    LOOKBACK = 60
    PRED_STEPS = 10
    PERIOD = "18mo"
    TEST_RATIO = 0.15

    # 1) 取得資料並計指標（包括基本技術指標）
    df = fetch_and_prepare(ticker=TICKER, period=PERIOD)

    # 2) 若 Firestore 有今天 close，可用來更新（選擇性）
    df = update_today_from_firestore(df)

    # 3) 先把歷史資料（含技術指標）寫回 Firestore （再給 LSTM 跑）
    save_stock_data_to_firestore(df, ticker=TICKER)

    # 4) 準備訓練資料（保持你原本的 features）
    features = ['Close', 'Volume', 'RET_1', 'LOG_RET_1', 'Close_minus_SMA5',
                'SMA5_minus_SMA10', 'ATR_14', 'BB_width', 'OBV', 'OBV_SMA_20',
                'Vol_SMA_5']
    df_features = df[features].dropna()

    X, y = create_sequences(df_features, features, target_steps=PRED_STEPS, window=LOOKBACK)
    print("X shape:", X.shape, "y shape:", y.shape)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_ratio=TEST_RATIO)

    # Scaler
    nsamples, tw, nfeatures = X_train.shape
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train.reshape((nsamples*tw, nfeatures)))
    def scale_X(X_raw):
        s = X_raw.reshape((-1, X_raw.shape[-1]))
        return scaler_x.transform(s).reshape(X_raw.shape)
    X_train_s, X_test_s = scale_X(X_train), scale_X(X_test)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train_s, y_test_s = scaler_y.transform(y_train), scaler_y.transform(y_test)

    # Build & train model
    model = build_lstm_multi_step(input_shape=(LOOKBACK, nfeatures), output_steps=PRED_STEPS)
    model.summary()

    os.makedirs("models", exist_ok=True)
    ckpt_path = f"models/{TICKER}_best.h5"
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)

    history = model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
                        epochs=80, batch_size=32, callbacks=[es, mc], verbose=2)

    # Predict & inverse scale
    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(pred_s)

    # ========== 你的原始評估（每步 MAE / RMSE） ==========
    maes, rmses = [], []
    for step in range(PRED_STEPS):
        y_true, y_pred = y_test[:, step], pred[:, step]
        maes.append(mean_absolute_error(y_true, y_pred))
        rmses.append(math.sqrt(mean_squared_error(y_true, y_pred)))
    print("MAE per step (model):", np.round(maes,4))
    print("RMSE per step (model):", np.round(rmses,4))
    print("Avg MAE (model):", np.round(np.mean(maes),4))

    # 取最後一個測試 sample 的已知 closes（作為起始序列），並用最後一筆預測計算 Pred MA5/MA10
    last_known_window = X_test[-1]
    last_known_closes = list(last_known_window[:,0])
    results = compute_pred_ma_from_pred_closes(last_known_closes, pred[-1])

    # 建立未來交易日日期
    today = pd.Timestamp(datetime.now().date())
    first_bday = (today + BDay(1)).date()
    business_days = pd.bdate_range(start=first_bday, periods=PRED_STEPS)
    df_future = pd.DataFrame({
        "date": business_days,
        "Pred_Close": [r[0] for r in results],
        "Pred_MA5": [r[1] for r in results],
        "Pred_MA10": [r[2] for r in results]
    })

    # 繪圖（改為上傳 Storage）
    image_url = plot_and_upload_to_storage(df, df_future, bucket_obj=bucket)
    print("Image URL:", image_url)

    print(df_future)

    # ---------------- Baseline 評估（整合區塊） ----------------
    print("\n===== Baseline 評估開始 =====")

    # Baseline A: last known close repeated
    last_known_closes_all = X_test[:, -1, 0]  # 每個測試 sample 最後一個已知 close
    baselineA = np.vstack([last_known_closes_all for _ in range(pred.shape[1])]).T  # (n_samples, steps)

    # Baseline B: 使用 df 的最後一個 SMA_5 值作為保守 baseline（若有）
    try:
        if 'SMA_5' in df.columns and not df['SMA_5'].dropna().empty:
            last_sma5_val = df['SMA_5'].dropna().iloc[-1]
            last_known_sma5_all = np.array([last_sma5_val] * X_test.shape[0])
            baselineB = np.vstack([last_known_sma5_all for _ in range(pred.shape[1])]).T
        else:
            baselineB = baselineA.copy()
    except Exception:
        baselineB = baselineA.copy()

    # Baseline C: simple random-walk on returns
    last_ret_1 = X_test[:, -1, features.index('RET_1')] if 'RET_1' in features else None
    if last_ret_1 is not None:
        baselineC = np.zeros_like(baselineA)
        for i in range(baselineC.shape[0]):
            price = last_known_closes_all[i]
            r = last_ret_1[i]
            for t in range(baselineC.shape[1]):
                price = price * (1 + r)
                baselineC[i, t] = price
    else:
        baselineC = baselineA.copy()

    # 計算 metrics（每 step）
    maes_model, rmses_model = compute_metrics(y_test, pred)
    maes_bA, rmses_bA = compute_metrics(y_test, baselineA)
    maes_bB, rmses_bB = compute_metrics(y_test, baselineB)
    maes_bC, rmses_bC = compute_metrics(y_test, baselineC)

    print("=== Per-step MAE (model) ===\n", np.round(maes_model,4))
    print("=== Per-step RMSE (model) ===\n", np.round(rmses_model,4))
    print("=== Per-step MAE (Baseline A: last close) ===\n", np.round(maes_bA,4))
    print("=== Per-step MAE (Baseline B: last SMA5/fallback) ===\n", np.round(maes_bB,4))
    print("=== Per-step MAE (Baseline C: simple returns) ===\n", np.round(maes_bC,4))

    print("\nAvg MAE model:", np.round(maes_model.mean(),4),
          "baselineA:", np.round(maes_bA.mean(),4), "baselineB:", np.round(maes_bB.mean(),4),
          "baselineC:", np.round(maes_bC.mean(),4))
    print("Avg RMSE model:", np.round(rmses_model.mean(),4),
          "baselineA:", np.round(rmses_bA.mean(),4))

    # Evaluate effect on MA5 / MA10
    last_closes_window = X_test[:, :, 0]  # shape (n_samples, LOOKBACK)

    model_MA5 = compute_ma_from_predictions(last_closes_window, pred, ma_period=5)
    model_MA10 = compute_ma_from_predictions(last_closes_window, pred, ma_period=10)

    bA_MA5 = compute_ma_from_predictions(last_closes_window, baselineA, ma_period=5)
    bA_MA10 = compute_ma_from_predictions(last_closes_window, baselineA, ma_period=10)

    bB_MA5 = compute_ma_from_predictions(last_closes_window, baselineB, ma_period=5)
    bB_MA10 = compute_ma_from_predictions(last_closes_window, baselineB, ma_period=10)

    true_MA5 = compute_true_ma(last_closes_window, y_test, ma_period=5)
    true_MA10 = compute_true_ma(last_closes_window, y_test, ma_period=10)

    mae_model_MA5 = np.mean(np.abs(model_MA5 - true_MA5))
    mae_bA_MA5 = np.mean(np.abs(bA_MA5 - true_MA5))
    mae_bB_MA5 = np.mean(np.abs(bB_MA5 - true_MA5))

    mae_model_MA10 = np.mean(np.abs(model_MA10 - true_MA10))
    mae_bA_MA10 = np.mean(np.abs(bA_MA10 - true_MA10))
    mae_bB_MA10 = np.mean(np.abs(bB_MA10 - true_MA10))

    print("\nMAE on derived MA5 -> model:", np.round(mae_model_MA5,4),
          "baselineA:", np.round(mae_bA_MA5,4), "baselineB:", np.round(mae_bB_MA5,4))
    print("MAE on derived MA10 -> model:", np.round(mae_model_MA10,4),
          "baselineA:", np.round(mae_bA_MA10,4), "baselineB:", np.round(mae_bB_MA10,4))

    print("===== Baseline 評估結束 =====\n")

    # 寫入預測到 Firestore（如啟用）
    if db is not None:
        for i, row in df_future.iterrows():
            try:
                db.collection("NEW_stock_data_liteon_preds").document(row['date'].strftime("%Y-%m-%d")).set({
                    "2301.TW": {
                        "Pred_Close": float(row['Pred_Close']),
                        "Pred_MA5": float(row['Pred_MA5']),
                        "Pred_MA10": float(row['Pred_MA10'])
                    }
                })
            except Exception as e:
                print("寫入預測到 Firestore 發生錯誤：", e)
        # 同時寫入 metadata doc（包含 image_url）
        try:
            pred_table_serialized = []
            for _, r in df_future.reset_index(drop=True).iterrows():
                rec = {
                    "date": pd.Timestamp(r['date']).strftime("%Y-%m-%d"),
                    "Pred_Close": float(r['Pred_Close']),
                    "Pred_MA5": float(r['Pred_MA5']),
                    "Pred_MA10": float(r['Pred_MA10'])
                }
                pred_table_serialized.append(rec)

            meta_doc = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "image_url": image_url,
                "pred_table": pred_table_serialized,
                "update_time": datetime.now().isoformat()
            }
            db.collection("NEW_stock_data_liteon_preds_meta").document(datetime.now().strftime("%Y-%m-%d")).set(meta_doc)
        except Exception as e:
            print("寫入預測 metadata 到 Firestore 發生錯誤：", e)

        print("🔥 預測寫入 Firestore 完成")
