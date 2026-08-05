# app.py — Bitcoin LSTM web app (Streamlit) — reads a bundled CSV (no live yfinance).
# Candlestick / line chart via Plotly. Config-driven model package.

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from features import last_window  # feature builder not needed: CSV already has features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "btc_dataset_with_sentiment.csv")

st.set_page_config(page_title="Bitcoin LSTM Forecast", page_icon="₿", layout="wide")
SIGNAL_COLOR = {"Buy": "#16c784", "Sell": "#ea3943", "Hold": "#f0b90b"}


@st.cache_resource(show_spinner=False)
def load_package():
    import keras
    import joblib
    cfg = json.load(open(os.path.join(MODELS_DIR, "config.json")))
    model = keras.models.load_model(os.path.join(MODELS_DIR, cfg["model_file"]))
    fscaler = joblib.load(os.path.join(MODELS_DIR, cfg["feature_scaler"]))
    return cfg, model, fscaler


@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def scale_window(win, fscaler):
    n = win.shape[2]
    return fscaler.transform(win.reshape(-1, n)).reshape(win.shape)


# ---------- UI ----------
st.title("₿ Bitcoin LSTM Forecast")
st.caption("Next-move signal from an LSTM model. Educational thesis demo — not financial advice.")

try:
    cfg, model, fscaler = load_package()
except Exception as e:
    st.error("Could not load the model package from ./models "
             "(need model_*.keras, *.pkl scaler, config.json).")
    st.exception(e)
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error("Could not read data/btc_dataset_with_sentiment.csv. "
             "Make sure the CSV is uploaded inside the app's data/ folder.")
    st.exception(e)
    st.stop()

look_back = int(cfg["look_back"])
features = cfg["features"]

with st.sidebar:
    st.header("Model")
    st.write(f"**Version:** {cfg.get('version', '?')}")
    st.write(f"**Task:** {cfg.get('task')}")
    st.write(f"**Look-back:** {look_back} days")
    st.write(f"**Horizon:** +{cfg.get('horizon')} day(s)")
    st.write(f"**Trained on:** {cfg.get('trained_on', '?')}")
    st.write(f"**Rows in data:** {len(df)}")
    st.write(f"**Latest date:** {df['Date'].iloc[-1].date()}")
    days = st.slider("Chart days", 30, 365, 120, step=10)

if len(df) < look_back:
    st.error(f"Data has {len(df)} rows but the model needs {look_back}.")
    st.stop()

# ---------- predict on the most recent window ----------
win = df[features].tail(look_back).values.astype("float32").reshape(1, look_back, len(features))
win_s = scale_window(win, fscaler)
latest_price = float(df["Close"].iloc[-1])
pred_raw = model.predict(win_s, verbose=0)

probs = np.ravel(pred_raw)
classes = cfg.get("classes", [str(i) for i in range(len(probs))])
idx = int(np.argmax(probs))
signal = classes[idx]

c1, c2, c3 = st.columns(3)
c1.metric("Latest close", f"${latest_price:,.0f}")
c2.metric(f"Signal (+{cfg['horizon']}d)", signal)
c3.metric("Confidence", f"{probs[idx]*100:.1f}%")

sc = SIGNAL_COLOR.get(signal, "#888")
st.markdown(
    f"<div style='display:inline-block;padding:6px 18px;border-radius:20px;"
    f"background:{sc};color:white;font-weight:700;font-size:18px;'>Signal: {signal}</div>",
    unsafe_allow_html=True,
)

# probability breakdown
pcols = st.columns(len(classes))
for i, (cl, p) in enumerate(zip(classes, probs)):
    pcols[i].progress(min(float(p), 1.0), text=f"{cl}: {p*100:.1f}%")

# ---------- chart: candlestick / line ----------
st.subheader("Price chart")
view = st.radio("View", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")

dfp = df.tail(int(days)).copy()
fig = go.Figure()
has_ohlc = all(c in dfp.columns for c in ["Open", "High", "Low", "Close"])
if view == "Candlestick" and has_ohlc:
    fig.add_trace(go.Candlestick(
        x=dfp["Date"], open=dfp["Open"], high=dfp["High"],
        low=dfp["Low"], close=dfp["Close"], name="BTC-USD",
        increasing_line_color="#16c784", decreasing_line_color="#ea3943"))
else:
    fig.add_trace(go.Scatter(
        x=dfp["Date"], y=dfp["Close"], mode="lines",
        line=dict(color="#f0b90b", width=2), name="Close"))

fig.add_trace(go.Scatter(
    x=[dfp["Date"].iloc[-1]], y=[latest_price], mode="markers+text",
    marker=dict(size=13, color=sc, line=dict(color="white", width=1.5)),
    text=[signal], textposition="top center",
    textfont=dict(color=sc, size=13), showlegend=False))

fig.update_layout(
    template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=False, hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

st.info("This demo reads a fixed dataset bundled with the app (reliable for the defense). "
        "To refresh with newer prices, re-export the CSV from Colab and replace it in data/. | "
        "يقرأ العرض بيانات ثابتة مرفقة مع التطبيق (موثوقة للمناقشة). لتحديثها، صدّر الملف من Colab واستبدله.")
