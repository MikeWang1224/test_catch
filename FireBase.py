# -*- coding: utf-8 -*-
"""
Refactored FireBase.py
完全從 Firestore 讀取 df，已移除抓 Yahoo / 歷史寫回
"""
import os
import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Firebase
import json
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage

# ================= Firebase 初始化 =================
key_dict = json.loads(os.environ.get("FIREBASE", "{}"))
db = None
bucket = None

if key_dict:
    cred = credentials.Certificate(key_dict)
    try:
        firebase_admin.get_app()
    except Exception:
        firebase_admin.initialize_app(
            cred, {"storageBucket": f"{key_dict.get('project_id')}.appspot.com"}
        )
    db = firestore.client()
    try:
        storage_client = storage.Client.from_service_account_info(key_dict)
        bucket = storage_client.bucket(f"{key_dict.get('project_id')}.appspot.com")
    except Exception:
        bucket = None

# ================= 從 Firestore 讀取 df =================
def load_df_from_firestore(ticker="2301.TW", collection="NEW_stock_data_liteon", days=400):
    if db is None:
        raise RuntimeError("Firestore 未初始化")

    docs = (
        db.collection(collection)
        .order_by("__name__", direction=firestore.Query.DESCENDING)
        .limit(days)
        .stream()
    )

    rows = []
    for doc in docs:
        payload = doc.to_dict().get(ticker)
        if not payload:
            continue
        rows.append({"date": doc.id, **payload})

    if not rows:
        raise RuntimeError("Firestore 無任何股價資料")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df

# ================= Dataset helpers =================
def create_sequences(df, features, target_steps=10, window=60):
    X, y = [], []
    closes = df['Close'].values
    data = df[features].values
    for i in range(window, len(df) - target_steps + 1):
        X.append(data[i-window:i])
        y.append(closes[i:i+target_steps])
    return np.array(X), np.array(y)


def time_series_split(X, y, test_ratio=0.15):
    n = len(X)
    test_n = int(n * test_ratio)
    split_idx = n - test_n
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

# ================= Model =================
def build_lstm_multi_step(input_shape, output_steps=10):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(output_steps))
    model.compile(optimizer='adam', loss='mae')
    return model

# ================= MA / metrics helpers =================
def compute_pred_ma_from_pred_closes(last_known_closes, pred_closes):
    closes_seq = list(last_known_closes)
    results = []
    for pc in pred_closes:
        closes_seq.append(pc)
        ma5 = np.mean(closes_seq[-5:]) if len(closes_seq) >= 5 else np.mean(closes_seq)
        ma10 = np.mean(closes_seq[-10:]) if len(closes_seq) >= 10 else np.mean(closes_seq)
        results.append((pc, ma5, ma10))
    return results

def compute_metrics(y_true, y_pred):
    maes, rmses = [], []
    for step in range(y_true.shape[1]):
        maes.append(mean_absolute_error(y_true[:, step], y_pred[:, step]))
        rmses.append(math.sqrt(mean_squared_error(y_true[:, step], y_pred[:, step])))
    return np.array(maes), np.array(rmses)

def compute_ma_from_predictions(last_known_window_closes, y_pred_matrix, ma_period=5):
    n_samples, _ = last_known_window_closes.shape
    steps = y_pred_matrix.shape[1]
    preds_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_known_window_closes[i])
        for t in range(steps):
            seq.append(y_pred_matrix[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            preds_ma[i, t] = np.mean(look)
    return preds_ma

def compute_true_ma(last_window, y_true, ma_period=5):
    n_samples, _ = last_window.shape
    steps = y_true.shape[1]
    true_ma = np.zeros((n_samples, steps))
    for i in range(n_samples):
        seq = list(last_window[i])
        for t in range(steps):
            seq.append(y_true[i, t])
            look = seq[-ma_period:] if len(seq) >= ma_period else seq
            true_ma[i, t] = np.mean(look)
    return true_ma

# ================= Plot + upload =================
def plot_and_upload_to_storage(df_real, df_future, bucket_obj=None):
    df_real_plot = df_real.tail(10)
    if df_real_plot.empty:
        return None

    last_hist_date = df_real_plot.index[-1]
    start_row = {
        'date': last_hist_date,
        'Pred_Close': df_real_plot['Close'].iloc[-1],
        'Pred_MA5': df_real_plot.get('SMA_5', df_real_plot['Close']).iloc[-1],
        'Pred_MA10': df_real_plot.get('SMA_10', df_real_plot['Close']).iloc[-1],
    }

    df_future_plot = pd.concat([pd.DataFrame([start_row]), df_future], ignore_index=True)

    plt.figure(figsize=(16, 8))

    x_real = range(len(df_real_plot))
    plt.plot(x_real, df_real_plot['Close'], label='Close')
    if 'SMA_5' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_5'], label='SMA5')
    if 'SMA_10' in df_real_plot.columns:
        plt.plot(x_real, df_real_plot['SMA_10'], label='SMA10')

    offset = len(df_real_plot) - 1
    x_future = [offset + i for i in range(len(df_future_plot))]
    plt.plot(x_future, df_future_plot['Pred_Close'], 'r:o', label='Pred Close')

    for xf, val in zip(x_future, df_future_plot['Pred_Close']):
        plt.annotate(f"{val:.2f}", (xf, val), xytext=(6, 6), textcoords='offset points',
                     fontsize=8, bbox=dict(fc='white', alpha=0.7))

    plt.plot(x_future, df_future_plot['Pred_MA5'], '--', label='Pred MA5')
    plt.plot(x_future, df_future_plot['Pred_MA10'], '--', label='Pred MA10')

    labels = [d.strftime('%m-%d') for d in df_real_plot.index[:-1]] + \
             [d.strftime('%m-%d') for d in df_future_plot['date']]
    plt.xticks(range(len(labels)), labels, rotation=45)

    plt.legend()
    plt.title('2301.TW 預測')

    os.makedirs('results', exist_ok=True)
    fname = f"{datetime.now().strftime('%Y-%m-%d')}_pred.png"
    fpath = os.path.join('results', fname)
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()

    if bucket_obj is not None:
        blob = bucket_obj.blob(f"LSTM_Pred_Images/{fname}")
        blob.upload_from_filename(fpath)
        try:
            blob.make_public()
            return blob.public_url
        except Exception:
            return blob.public_url

    return None

# ================= Main =================
if __name__ == '__main__':
    TICKER = '2301.TW'
    LOOKBACK = 60
    PRED_STEPS = 10
    TEST_RATIO = 0.15

    # ---------------- 從 Firestore 讀 df ----------------
    df = load_df_from_firestore(ticker=TICKER)
    print("🔥 從 Firestore 讀取資料筆數:", len(df))

    # ---------------- 準備特徵（LSTM） ----------------
    features = ['Close', 'Volume', 'RET_1', 'LOG_RET_1', 'Close_minus_SMA5',
                'SMA5_minus_SMA10', 'ATR_14', 'BB_width', 'OBV', 'OBV_SMA_20',
                'Vol_SMA_5']

    # 確認 df 包含這些欄位
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"缺少必要欄位: {missing}")

    df_features = df[features].dropna()

    X, y = create_sequences(df_features, features, target_steps=PRED_STEPS, window=LOOKBACK)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_ratio=TEST_RATIO)

    # Scaler
    nsamples, tw, nfeatures = X_train.shape
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train.reshape((nsamples*tw, nfeatures)))
    X_train_s = scaler_x.transform(X_train.reshape((-1, nfeatures))).reshape(X_train.shape)
    X_test_s = scaler_x.transform(X_test.reshape((-1, nfeatures))).reshape(X_test.shape)

    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    y_train_s = scaler_y.transform(y_train)
    y_test_s = scaler_y.transform(y_test)

    # ---------------- LSTM 訓練 ----------------
    model = build_lstm_multi_step(input_shape=(LOOKBACK, nfeatures), output_steps=PRED_STEPS)
    ckpt_path = f"models/{TICKER}_best.h5"
    os.makedirs('models', exist_ok=True)
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)

    model.fit(X_train_s, y_train_s, validation_data=(X_test_s, y_test_s),
              epochs=80, batch_size=32, callbacks=[es, mc], verbose=2)

    # ---------------- 預測 ----------------
    pred_s = model.predict(X_test_s)
    pred = scaler_y.inverse_transform(pred_s)

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

    # ---------------- 繪圖 + 上傳 Storage ----------------
    image_url = plot_and_upload_to_storage(df, df_future, bucket_obj=bucket)
    print("Image URL:", image_url)

    # ---------------- Baseline 評估 ----------------
    last_known_closes_all = X_test[:, -1, 0]
    baselineA = np.vstack([last_known_closes_all for _ in range(pred.shape[1])]).T

    maes_model, rmses_model = compute_metrics(y_test, pred)
    maes_bA, rmses_bA = compute_metrics(y_test, baselineA)

    print("Avg MAE model:", np.round(maes_model.mean(),4), "baselineA:", np.round(maes_bA.mean(),4))

    # ---------------- 寫入預測結果到 Firestore ----------------
    if db is not None:
        for i, row in df_future.iterrows():
            db.collection("NEW_stock_data_liteon_preds").document(row['date'].strftime("%Y-%m-%d")).set({
                TICKER: {
                    "Pred_Close": float(row['Pred_Close']),
                    "Pred_MA5": float(row['Pred_MA5']),
                    "Pred_MA10": float(row['Pred_MA10'])
                }
            })
        # metadata
        meta_doc = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "image_url": image_url,
            "pred_table": df_future.to_dict('records'),
            "update_time": datetime.now().isoformat()
        }
        db.collection("NEW_stock_data_liteon_preds_meta").document(datetime.now().strftime("%Y-%m-%d")).set(meta_doc)
        print("🔥 預測寫入 Firestore 完成")
