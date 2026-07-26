# app.py — Bitcoin LSTM web app (Streamlit)
# Loads a model PACKAGE (model + scalers + config) at runtime. To update the
# model later you only replace the files in ./models and edit config.json —
# no change to this file is needed as long as the input/output interface is the same.

import os
import json
import numpy as np
import pandas as pd
import streamlit as st

from features import build_features, last_window

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Bitcoin LSTM Forecast", page_icon="₿", layout="wide")


# ---------- loading (cached so it runs once) ----------
@st.cache_resource(show_spinner=False)
def load_package():
    import keras
    import joblib
    cfg = json.load(open(os.path.join(MODELS_DIR, "config.json")))
    model = keras.models.load_model(os.path.join(MODELS_DIR, cfg["model_file"]))
    fscaler = joblib.load(os.path.join(MODELS_DIR, cfg["feature_scaler"]))
    tscaler = None
    if cfg.get("task") == "regression" and cfg.get("target_scaler"):
        tpath = os.path.join(MODELS_DIR, cfg["target_scaler"])
        if os.path.exists(tpath):
            tscaler = joblib.load(tpath)
    return cfg, model, fscaler, tscaler


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(start="2014-09-17"):
    import yfinance as yf
    raw = yf.download("BTC-USD", start=start, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return raw[["Close", "Volume"]].copy()


def scale_window(win, fscaler):
    n = win.shape[2]
    flat = fscaler.transform(win.reshape(-1, n))
    return flat.reshape(win.shape)


# ---------- UI ----------
st.title("₿ Bitcoin LSTM Forecast")
st.caption("Next-day forecast from an LSTM model. Educational thesis demo — not financial advice.")

try:
    cfg, model, fscaler, tscaler = load_package()
except Exception as e:
    st.error(
        "Could not load the model package from ./models. "
        "Make sure model_*.keras, the *.pkl scalers and config.json are present."
    )
    st.exception(e)
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write(f"**Version:** {cfg.get('version', '?')}")
    st.write(f"**Task:** {cfg.get('task')}")
    st.write(f"**Look-back:** {cfg.get('look_back')} days")
    st.write(f"**Horizon:** +{cfg.get('horizon')} day(s)")
    st.write(f"**Features:** {', '.join(cfg.get('features', []))}")
    st.write(f"**Trained on:** {cfg.get('trained_on', '?')}")
    if st.button("🔄 Refresh price data"):
        fetch_prices.clear()
        st.rerun()

# data + features
with st.spinner("Fetching latest BTC-USD data…"):
    prices = fetch_prices()
feat = build_features(prices)

look_back = int(cfg["look_back"])
features = cfg["features"]
win = last_window(feat, features, look_back)
win_s = scale_window(win, fscaler)

latest_price = float(feat["Close"].iloc[-1])
pred_raw = model.predict(win_s, verbose=0)

col1, col2, col3 = st.columns(3)

if cfg.get("task") == "regression":
    if tscaler is not None:
        pred_price = float(tscaler.inverse_transform(pred_raw.reshape(-1, 1)).ravel()[0])
    else:
        pred_price = float(np.ravel(pred_raw)[0])
    change = pred_price - latest_price
    pct = change / latest_price * 100
    signal = "BUY ▲" if change > 0 else ("SELL ▼" if change < 0 else "HOLD ▬")
    col1.metric("Latest close", f"${latest_price:,.0f}")
    col2.metric(f"Predicted (+{cfg['horizon']}d)", f"${pred_price:,.0f}", f"{pct:+.2f}%")
    col3.metric("Signal", signal)
    future_val = pred_price
else:  # classification
    probs = np.ravel(pred_raw)
    classes = cfg.get("classes", [str(i) for i in range(len(probs))])
    idx = int(np.argmax(probs))
    col1.metric("Latest close", f"${latest_price:,.0f}")
    col2.metric("Signal", classes[idx])
    col3.metric("Confidence", f"{probs[idx]*100:.1f}%")
    st.write({c: f"{p*100:.1f}%" for c, p in zip(classes, probs)})
    future_val = None

# chart: last 90 days + predicted point
st.subheader("Price — last 90 days")
hist = feat[["Close"]].tail(90).reset_index(drop=True).rename(columns={"Close": "Actual"})
if future_val is not None:
    hist = pd.concat([hist, pd.DataFrame({"Actual": [np.nan]})], ignore_index=True)
    # a short Forecast line joining today's price to the predicted point
    hist["Forecast"] = [np.nan] * (len(hist) - 2) + [latest_price, future_val]
st.line_chart(hist)

st.info(
    "How updates work: replace the files in **models/** with a new version and update "
    "**config.json**. If the features / look-back / outputs are unchanged, this app needs "
    "no edits — just redeploy. | لتحديث النموذج: استبدل ملفات مجلد models وحدّث config.json. "
    "إذا بقيت الخصائص والنافذة والمخرجات كما هي، لا يحتاج الموقع أي تعديل — فقط أعد النشر."
)
