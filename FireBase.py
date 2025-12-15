# -*- coding: utf-8 -*-
"""
FireBase_Transformer_Direction.py
- Transformer Encoder (MultiHeadAttention)
- Multi-task: Return path + Direction
- ✅ 小資料友善版：更穩、更不容易亂噴
  1) LOOKBACK=40, STEPS=5
  2) Transformer 變小（d_model=32, heads=2, depth=1, dropout=0.2）
  3) Walk-forward：folds=2 + 固定 val_len=25
  4) features 減肥：log_ret / hl_range / gap / Volume / ATR_14
  5) ✅ 核心修正：Return head 加 tanh 限幅（避免預測爆炸）
  6) ✅ Volume 做 log1p（小資料更穩）
- 圖表輸出完全不變（保留 Today 標記）
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
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Lambda,
    LayerNormalization, MultiHeadAttention, Add, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

# Firebase
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
    if db:
        for doc in db.collection(collection).stream():
            p = doc.to_dict().get(ticker)
            if p:
                rows.append({"date": doc.id, **p})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("⚠️ Firestore 無資料")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").tail(days).set_index("date")
    return df

# ================= 假日補今天 =================
def ensure_today_row(df):
    today = pd.Timestamp(datetime.now().date())
    last_date = df.index.max()
    if last_date < today:
        df.loc[today] = df.loc[last_date]
        print(f"⚠️ 今日無資料，使用 {last_date.date()} 補今日")
    return df.sort_index()

# ================= Feature Engineering =================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # ✅ Volume 尺度穩定（很重要：小資料避免極端值主宰）
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].astype(float))

    # 相對/報酬特徵
    df["log_ret"]  = np.log(df["Close"]).diff()
    df["oc_ret"]   = np.log(df["Close"] / df["Open"])
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["gap"]      = np.log(df["Open"] / df["Close"].shift(1))

    # 圖表用均線（保持不變）
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA10"] = df["Close"].rolling(10).mean()
    return df

# ================= Sequence（避免錯位） =================
def create_sequences(df, features, steps=5, window=40):
    """
    X: t-window ~ t-1
    y_ret: t ~ t+steps-1 的 log return
    y_dir: 未來 steps 天累積方向
    """
    X, y_ret, y_dir = [], [], []

    close = df["Close"].astype(float)
    logret = np.log(close).diff()
    feat = df[features].values

    for i in range(window, len(df) - steps):
        x_seq = feat[i - window:i]
        future_ret = logret.iloc[i:i + steps].values
        if np.any(np.isnan(future_ret)) or np.any(np.isnan(x_seq)):
            continue
        X.append(x_seq)
        y_ret.append(future_ret)
        y_dir.append(1.0 if future_ret.sum() > 0 else 0.0)

    return np.array(X), np.array(y_ret), np.array(y_dir)

# ================= Transformer Encoder Block =================
def transformer_block(x, d_model, num_heads, ff_dim, dropout=0.2):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff_out = Dense(ff_dim, activation="relu")(x)
    ff_out = Dropout(dropout)(ff_out)
    ff_out = Dense(d_model)(ff_out)
    x = Add()([x, ff_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x

# ================= Model（小資料友善 + ✅ return 限幅避免爆炸） =================
def build_transformer_model(
    input_shape, steps,
    d_model=32, num_heads=2, ff_dim=64, depth=1, dropout=0.2,
    max_daily_logret=0.06  # ✅ 單日 log-return 最大幅度（可調：0.04~0.08 常見）
):
    inp = Input(shape=input_shape)
    x = Dense(d_model)(inp)

    for _ in range(depth):
        x = transformer_block(x, d_model=d_model, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)

    context = GlobalAveragePooling1D()(x)
    context = Dropout(dropout)(context)

    # ✅ 核心：把 return head 限在 [-max_daily_logret, +max_daily_logret]
    raw = Dense(steps, activation="tanh")(context)  # [-1, 1]
    out_ret = Lambda(lambda t: t * max_daily_logret, name="return")(raw)

    out_dir = Dense(1, activation="sigmoid", name="direction")(context)

    model = Model(inp, [out_ret, out_dir])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=7e-4),
        loss={
            "return": tf.keras.losses.Huber(),
            "direction": "binary_crossentropy"
        },
        loss_weights={
            "return": 1.0,
            "direction": 0.4
        },
        metrics={
            "direction": [tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")]
        }
    )
    return model

# ================= 原預測圖（完全不動：含 Today 標記） =================
def plot_and_save(df_hist, future_df):
    hist = df_hist.tail(10)
    hist_dates = hist.index.strftime("%Y-%m-%d").tolist()
    future_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()

    all_dates = hist_dates + future_dates
    x_hist = np.arange(len(hist_dates))
    x_future = np.arange(len(hist_dates), len(all_dates))

    plt.figure(figsize=(18,8))
    ax = plt.gca()

    ax.plot(x_hist, hist["Close"], label="Close")
    ax.plot(x_hist, hist["SMA5"], label="SMA5")
    ax.plot(x_hist, hist["SMA10"], label="SMA10")

    today_x = x_hist[-1]
    today_y = float(hist["Close"].iloc[-1])
    ax.scatter([today_x], [today_y], marker="*", s=160, label="Today Close")
    ax.text(today_x, today_y + 0.3, f"Today {today_y:.2f}",
            fontsize=10, ha="center")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["Close"].iloc[-1]] + future_df["Pred_Close"].tolist(),
        "r:o", label="Pred Close"
    )

    for i, price in enumerate(future_df["Pred_Close"]):
        ax.text(x_future[i], price + 0.3, f"{price:.2f}",
                color="red", fontsize=9, ha="center")

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA5"].iloc[-1]] + future_df["Pred_MA5"].tolist(),
        "g--o", label="Pred MA5"
    )

    ax.plot(
        np.concatenate([[x_hist[-1]], x_future]),
        [hist["SMA10"].iloc[-1]] + future_df["Pred_MA10"].tolist(),
        "b--o", label="Pred MA10"
    )

    ax.set_xticks(np.arange(len(all_dates)))
    ax.set_xticklabels(all_dates, rotation=45, ha="right")
    ax.legend()
    ax.set_title("2301.TW Attention-LSTM 預測")

    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{datetime.now():%Y-%m-%d}_pred.png",
                dpi=300, bbox_inches="tight")
    plt.close()

# ================= 回測誤差圖（不動） =================
def plot_backtest_error(df, X_te_s, y_te, model, steps):
    X_last = X_te_s[-1:]
    y_true = y_te[-1]

    pred_ret, _ = model.predict(X_last, verbose=0)
    pred_ret = pred_ret[0]

    dates = df.index[-steps:]
    start_price = df.loc[dates[0] - BDay(1), "Close"]

    true_prices, pred_prices = [], []
    p_true = p_pred = start_price

    for r_t, r_p in zip(y_true, pred_ret):
        p_true *= np.exp(r_t)
        p_pred *= np.exp(r_p)
        true_prices.append(p_true)
        pred_prices.append(p_pred)

    mae = np.mean(np.abs(np.array(true_prices) - np.array(pred_prices)))
    rmse = np.sqrt(np.mean((np.array(true_prices) - np.array(pred_prices)) ** 2))

    plt.figure(figsize=(12,6))
    plt.plot(dates, true_prices, label="Actual Close")
    plt.plot(dates, pred_prices, "--o", label="Pred Close")
    plt.title(f"Backtest | MAE={mae:.2f}, RMSE={rmse:.2f}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    os.makedirs("results", exist_ok=True)
    plt.savefig(
        f"results/{datetime.now():%Y-%m-%d}_backtest.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# ================= Walk-forward（小資料友善：少 folds + 固定 val_len） =================
def walk_forward_evaluate(
    X, y_ret, y_dir,
    features_df_for_scaler, features,
    lookback, steps,
    folds=2, val_len=25, train_min=0.70, seed=42
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n = len(X)
    if n < (val_len * 3):
        print("⚠️ 可用序列真的偏少，walk-forward 只做極少 fold 會比較合理。")

    start_train_end = max(int(n * train_min), lookback + 10)
    max_train_end = n - val_len - 1
    if max_train_end <= start_train_end:
        start_train_end = max(n - val_len - 1, lookback + 10)

    train_ends = np.linspace(start_train_end, max_train_end, folds).astype(int)

    fold_metrics = []
    for k, train_end in enumerate(train_ends, start=1):
        tr_slice = slice(0, train_end)
        va_slice = slice(train_end, train_end + val_len)

        X_tr, X_va = X[tr_slice], X[va_slice]
        y_ret_tr, y_ret_va = y_ret[tr_slice], y_ret[va_slice]
        y_dir_tr, y_dir_va = y_dir[tr_slice], y_dir[va_slice]

        fit_end = min(train_end + lookback, len(features_df_for_scaler))
        sx = MinMaxScaler()
        sx.fit(features_df_for_scaler[features].iloc[:fit_end])

        def scale_X_block(Xb):
            nb, t, f = Xb.shape
            return sx.transform(Xb.reshape(-1, f)).reshape(nb, t, f)

        X_tr_s = scale_X_block(X_tr)
        X_va_s = scale_X_block(X_va)

        model = build_transformer_model((lookback, len(features)), steps)

        model.fit(
            X_tr_s,
            {"return": y_ret_tr, "direction": y_dir_tr},
            epochs=60,
            batch_size=16,
            verbose=0,
            callbacks=[EarlyStopping(patience=8, restore_best_weights=True)]
        )

        pred_ret, pred_dir = model.predict(X_va_s, verbose=0)

        dir_prob = pred_dir.reshape(-1)
        dir_pred = (dir_prob >= 0.5).astype(int)
        dir_true = y_dir_va.astype(int)
        acc = (dir_pred == dir_true).mean()

        true_cum = y_ret_va.sum(axis=1)
        pred_cum = pred_ret.sum(axis=1)
        mae_cum = np.mean(np.abs(true_cum - pred_cum))
        rmse_cum = np.sqrt(np.mean((true_cum - pred_cum) ** 2))

        fold_metrics.append((acc, mae_cum, rmse_cum))
        print(f"[WF Fold {k}/{folds}] dir_acc={acc:.3f} | cumRet_MAE={mae_cum:.4f} | cumRet_RMSE={rmse_cum:.4f}")

    accs = [m[0] for m in fold_metrics]
    maes = [m[1] for m in fold_metrics]
    rmses = [m[2] for m in fold_metrics]
    print(f"\n[WF Summary] dir_acc={np.mean(accs):.3f}±{np.std(accs):.3f} | cumRet_MAE={np.mean(maes):.4f} | cumRet_RMSE={np.mean(rmses):.4f}\n")

# ================= Main =================
if __name__ == "__main__":
    TICKER = "2301.TW"

    LOOKBACK = 40
    STEPS = 5

    df = load_df_from_firestore(TICKER, days=500)
    df = ensure_today_row(df)
    df = add_features(df)

    FEATURES = [
        "log_ret", "hl_range", "gap",
        "Volume", "ATR_14"
    ]

    df = df.dropna()

    X, y_ret, y_dir = create_sequences(df, FEATURES, steps=STEPS, window=LOOKBACK)
    print(f"df rows: {len(df)} | X samples: {len(X)}")

    if len(X) < 40:
        raise ValueError("⚠️ 可用序列太少（<40）。建議：降低 LOOKBACK/STEPS，或檢查資料是否缺欄位/過多 NaN。")

    # ========= Walk-forward =========
    walk_forward_evaluate(
        X, y_ret, y_dir,
        features_df_for_scaler=df,
        features=FEATURES,
        lookback=LOOKBACK,
        steps=STEPS,
        folds=2,
        val_len=25,
        train_min=0.70
    )

    # ========= 最終模型：用最後 15% 當 test =========
    split = int(len(X) * 0.85)
    X_tr, X_te = X[:split], X[split:]
    y_ret_tr, y_ret_te = y_ret[:split], y_ret[split:]
    y_dir_tr, y_dir_te = y_dir[:split], y_dir[split:]

    sx = MinMaxScaler()
    sx.fit(df[FEATURES].iloc[:min(split + LOOKBACK, len(df))])

    def scale_X(Xb):
        n, t, f = Xb.shape
        return sx.transform(Xb.reshape(-1, f)).reshape(n, t, f)

    X_tr_s = scale_X(X_tr)
    X_te_s = scale_X(X_te)

    # ✅ 這裡可調 max_daily_logret：想更保守就 0.04~0.05
    model = build_transformer_model(
        (LOOKBACK, len(FEATURES)), STEPS,
        d_model=32, num_heads=2, ff_dim=64, depth=1, dropout=0.2,
        max_daily_logret=0.06
    )

    model.fit(
        X_tr_s,
        {"return": y_ret_tr, "direction": y_dir_tr},
        epochs=80,
        batch_size=16,
        verbose=2,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
    )

    pred_ret, pred_dir = model.predict(X_te_s, verbose=0)
    raw_returns = pred_ret[-1]  # ✅ 已被結構性限幅，不會爆炸

    print(f"📈 預測方向機率（看漲）: {pred_dir[-1][0]:.2%}")

    asof_date = df.index.max()
    last_close = float(df.loc[asof_date, "Close"])

    prices = []
    price = last_close
    for r in raw_returns:
        price *= np.exp(r)
        prices.append(price)

    seq = df.loc[:asof_date, "Close"].iloc[-10:].tolist()
    future = []
    for p in prices:
        seq.append(p)
        future.append({
            "Pred_Close": float(p),
            "Pred_MA5": float(np.mean(seq[-5:])),
            "Pred_MA10": float(np.mean(seq[-10:]))
        })

    future_df = pd.DataFrame(future)
    future_df["date"] = pd.bdate_range(
        start=df.index.max() + BDay(1),
        periods=STEPS
    )

    plot_and_save(df, future_df)
    plot_backtest_error(df, X_te_s, y_ret_te, model, STEPS)
