# app.py — Bitcoin LSTM web app (Streamlit) + candlestick/line chart (Plotly)
# Config-driven: replace files in ./models and edit config.json to update the model.
 
import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
 
from features import build_features, last_window
 
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
st.set_page_config(page_title="Bitcoin LSTM Forecast", page_icon="₿", layout="wide")
 
SIGNAL_COLOR = {"Buy": "#16c784", "Sell": "#ea3943", "Hold": "#f0b90b"}
 
 
# ---------- load model package (once) ----------
@st.cache_resource(show_spinner=False)
def load_package():
    import keras
    import joblib
    cfg = json.load(open(os.path.join(MODELS_DIR, "config.json")))
    model = keras.models.load_model(os.path.join(MODELS_DIR, cfg["model_file"]))
    fscaler = joblib.load(os.path.join(MODELS_DIR, cfg["feature_scaler"]))
    tscaler = None
    if cfg.get("task") == "regression" and cfg.get("target_scaler"):
        tp = os.path.join(MODELS_DIR, cfg["target_scaler"])
        if os.path.exists(tp):
            tscaler = joblib.load(tp)
    return cfg, model, fscaler, tscaler
 
 
# ---------- robust price fetch (with retries + fallback) ----------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices():
    import yfinance as yf
    cols = ["Open", "High", "Low", "Close", "Volume"]
 
    def _clean(raw):
        if raw is None or len(raw) == 0:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.reset_index()
        raw.columns = [str(c) for c in raw.columns]
        if "Date" not in raw.columns:
            raw = raw.rename(columns={raw.columns[0]: "Date"})
        keep = [c for c in cols if c in raw.columns]
        out = raw[["Date"] + keep].dropna().reset_index(drop=True)
        return out
 
    # try 1: download period=max
    for _ in range(3):
        try:
            out = _clean(yf.download("BTC-USD", period="max", interval="1d",
                                     auto_adjust=True, progress=False))
            if len(out) > 0:
                return out
        except Exception:
            time.sleep(1)
 
    # try 2: Ticker().history fallback
    try:
        h = yf.Ticker("BTC-USD").history(period="max", interval="1d", auto_adjust=True)
        return _clean(h)
    except Exception:
        return pd.DataFrame()
 
 
def scale_window(win, fscaler):
    n = win.shape[2]
    return fscaler.transform(win.reshape(-1, n)).reshape(win.shape)
 
 
# ---------- UI ----------
st.title("₿ Bitcoin LSTM Forecast")
st.caption("Next-move signal from an LSTM model. Educational thesis demo — not financial advice.")
 
try:
    cfg, model, fscaler, tscaler = load_package()
except Exception as e:
    st.error("Could not load the model package from ./models "
             "(need model_*.keras, *.pkl scaler, config.json).")
    st.exception(e)
    st.stop()
 
with st.sidebar:
    st.header("Model")
    st.write(f"**Version:** {cfg.get('version', '?')}")
    st.write(f"**Task:** {cfg.get('task')}")
    st.write(f"**Look-back:** {cfg.get('look_back')} days")
    st.write(f"**Horizon:** +{cfg.get('horizon')} day(s)")
    st.write(f"**Trained on:** {cfg.get('trained_on', '?')}")
    days = st.slider("Chart days", 30, 365, 120, step=10)
    if st.button("🔄 Refresh data"):
        fetch_prices.clear()
        st.rerun()
 
with st.spinner("Fetching latest BTC-USD data…"):
    prices = fetch_prices()
 
if prices is None or len(prices) == 0:
    st.error("Price data came back empty from the data source. "
             "This is usually a temporary rate-limit — press **🔄 Refresh data** "
             "in the sidebar in a moment.")
    st.stop()
 
feat = build_features(prices)
look_back = int(cfg["look_back"])
features = cfg["features"]
 
if len(feat) < look_back:
    st.error(f"Not enough usable data yet (need {look_back} rows after building "
             f"features, have {len(feat)}). Press 🔄 Refresh data shortly.")
    st.stop()
 
win = last_window(feat, features, look_back)
win_s = scale_window(win, fscaler)
latest_price = float(prices["Close"].iloc[-1])
pred_raw = model.predict(win_s, verbose=0)
 
# ---------- prediction + signal ----------
c1, c2, c3 = st.columns(3)
future_val = None
if cfg.get("task") == "regression":
    if tscaler is not None:
        pred_price = float(tscaler.inverse_transform(pred_raw.reshape(-1, 1)).ravel()[0])
    else:
        pred_price = float(np.ravel(pred_raw)[0])
    change = pred_price - latest_price
    pct = change / latest_price * 100
    signal = "Buy" if change > 0 else ("Sell" if change < 0 else "Hold")
    c1.metric("Latest close", f"${latest_price:,.0f}")
    c2.metric(f"Predicted (+{cfg['horizon']}d)", f"${pred_price:,.0f}", f"{pct:+.2f}%")
    c3.metric("Signal", signal)
    future_val = pred_price
else:
    probs = np.ravel(pred_raw)
    classes = cfg.get("classes", [str(i) for i in range(len(probs))])
    idx = int(np.argmax(probs))
    signal = classes[idx]
    c1.metric("Latest close", f"${latest_price:,.0f}")
    c2.metric(f"Signal (+{cfg['horizon']}d)", signal)
    c3.metric("Confidence", f"{probs[idx]*100:.1f}%")
 
# colored signal badge
sc = SIGNAL_COLOR.get(signal, "#888")
st.markdown(
    f"<div style='display:inline-block;padding:6px 18px;border-radius:20px;"
    f"background:{sc};color:white;font-weight:700;font-size:18px;'>Signal: {signal}</div>",
    unsafe_allow_html=True,
)
 
# ---------- chart: candlestick / line toggle ----------
st.subheader("Price chart")
view = st.radio("View", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
 
dfp = prices.tail(int(days)).copy()
fig = go.Figure()
if view == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=dfp["Date"], open=dfp["Open"], high=dfp["High"],
        low=dfp["Low"], close=dfp["Close"], name="BTC-USD",
        increasing_line_color="#16c784", decreasing_line_color="#ea3943"))
else:
    fig.add_trace(go.Scatter(
        x=dfp["Date"], y=dfp["Close"], mode="lines",
        line=dict(color="#f0b90b", width=2), name="Close"))
 
# marker for the model's signal on the last day
fig.add_trace(go.Scatter(
    x=[dfp["Date"].iloc[-1]], y=[latest_price], mode="markers+text",
    marker=dict(size=13, color=sc, line=dict(color="white", width=1.5)),
    text=[signal], textposition="top center",
    textfont=dict(color=sc, size=13), name="Signal", showlegend=False))
 
if future_val is not None:
    fig.add_trace(go.Scatter(
        x=[dfp["Date"].iloc[-1]], y=[future_val], mode="markers",
        marker=dict(size=11, color=sc, symbol="diamond"),
        name="Forecast", showlegend=False))
 
fig.update_layout(
    template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=False, hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
 
st.info("Update the model any time: replace the files in **models/** and edit "
        "**config.json** — no code change needed. | لتحديث النموذج: استبدل ملفات "
        "models وحدّث config.json، بدون تعديل الكود.")
 
