# app.py — Bitcoin LSTM web app (Streamlit)
# Tries LIVE data (yfinance) first; falls back to the bundled CSV if that fails.
# A clear badge shows which source is in use + the date of the data.

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_CSV = os.path.join(os.path.dirname(__file__), "data", "btc_dataset_with_sentiment.csv")

st.set_page_config(page_title="Bitcoin LSTM Forecast", page_icon="₿", layout="wide")
SIGNAL_COLOR = {"Buy": "#16c784", "Sell": "#ea3943", "Hold": "#f0b90b"}


@st.cache_resource(show_spinner=False)
def load_package():
    import keras, joblib
    cfg = json.load(open(os.path.join(MODELS_DIR, "config.json")))
    model = keras.models.load_model(os.path.join(MODELS_DIR, cfg["model_file"]))
    fscaler = joblib.load(os.path.join(MODELS_DIR, cfg["feature_scaler"]))
    return cfg, model, fscaler


# ---------- STATIC source: the bundled CSV (already has features + OHLC) ----------
@st.cache_data(show_spinner=False)
def load_static():
    df = pd.read_csv(DATA_CSV)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").reset_index(drop=True)


# ---------- LIVE source: yfinance -> build the same 8 features ----------
@st.cache_data(ttl=1800, show_spinner=False)
def load_live():
    import yfinance as yf, ta
    raw = yf.download("BTC-USD", period="max", interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    d = raw.reset_index()
    d.columns = [str(c) for c in d.columns]
    if "Date" not in d.columns:
        d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"])
    close, vol = d["Close"], d["Volume"]
    d["ret_1"] = close.pct_change(1)
    d["ret_5"] = close.pct_change(5)
    d["rsi"] = ta.momentum.RSIIndicator(close, 14).rsi() / 100.0
    e12 = ta.trend.EMAIndicator(close, 12).ema_indicator()
    e26 = ta.trend.EMAIndicator(close, 26).ema_indicator()
    d["close_ema12"] = close / e12 - 1
    d["close_ema26"] = close / e26 - 1
    d["ema_cross"] = e12 / e26 - 1
    d["vol_chg"] = np.log(vol / vol.rolling(20).mean())
    d["volatility"] = d["ret_1"].rolling(10).std()
    feats = ["ret_1", "ret_5", "rsi", "close_ema12", "close_ema26",
             "ema_cross", "vol_chg", "volatility"]
    d = d.dropna(subset=feats).reset_index(drop=True)
    return d


def scale_window(win, fscaler):
    n = win.shape[2]
    return fscaler.transform(win.reshape(-1, n)).reshape(win.shape)


# ---------- UI ----------
st.title("₿ Bitcoin LSTM Forecast")
st.caption("Next-move signal from an LSTM model. Educational thesis demo — not financial advice.")

try:
    cfg, model, fscaler = load_package()
except Exception as e:
    st.error("Could not load the model package from ./models.")
    st.exception(e); st.stop()

look_back = int(cfg["look_back"])
features = cfg["features"]

with st.sidebar:
    st.header("Data source")
    use_live = st.toggle("Try live data (yfinance)", value=True,
                         help="If live fetch fails, the app falls back to the bundled dataset.")
    if st.button("🔄 Refresh"):
        load_live.clear(); st.rerun()

# pick source
df, source = None, None
if use_live:
    try:
        live = load_live()
        if live is not None and len(live) >= look_back:
            df, source = live, "live"
    except Exception:
        df = None
if df is None:
    df, source = load_static(), "static"

latest_date = df["Date"].iloc[-1].date()

# ---------- SOURCE BADGE (this tells you live vs static) ----------
if source == "live":
    st.success(f"🟢 **LIVE data** — latest market date: **{latest_date}** "
               f"(fetched from Yahoo Finance)")
else:
    st.warning(f"🟡 **STATIC dataset** — fixed snapshot up to **{latest_date}**. "
               f"Live fetch was off or unavailable, so the app is using the bundled file.")

with st.sidebar:
    st.divider()
    st.write(f"**Source:** {'🟢 Live' if source=='live' else '🟡 Static file'}")
    st.write(f"**Signal date:** {latest_date}")
    st.write(f"**Model:** {cfg.get('version','?')} · {cfg.get('task')}")
    st.write(f"**Look-back:** {look_back} d · **Horizon:** +{cfg.get('horizon')} d")
    st.write(f"**Rows:** {len(df)}")
    days = st.slider("Chart days", 30, 365, 120, step=10)

if len(df) < look_back:
    st.error(f"Not enough data: need {look_back} rows, have {len(df)}."); st.stop()

# ---------- predict on the most recent window ----------
win = df[features].tail(look_back).values.astype("float32").reshape(1, look_back, len(features))
win_s = scale_window(win, fscaler)
latest_price = float(df["Close"].iloc[-1])
probs = np.ravel(model.predict(win_s, verbose=0))
classes = cfg.get("classes", [str(i) for i in range(len(probs))])
idx = int(np.argmax(probs)); signal = classes[idx]
sc = SIGNAL_COLOR.get(signal, "#888")

c1, c2, c3 = st.columns(3)
c1.metric("Latest close", f"${latest_price:,.0f}")
c2.metric(f"Signal (+{cfg['horizon']}d)", signal)
c3.metric("Confidence", f"{probs[idx]*100:.1f}%")

st.markdown(
    f"<div style='display:inline-block;padding:6px 18px;border-radius:20px;"
    f"background:{sc};color:white;font-weight:700;font-size:18px;'>"
    f"Signal for {latest_date}: {signal}</div>", unsafe_allow_html=True)

pcols = st.columns(len(classes))
for i, (cl, pr) in enumerate(zip(classes, probs)):
    pcols[i].progress(min(float(pr), 1.0), text=f"{cl}: {pr*100:.1f}%")

# ---------- chart ----------
st.subheader("Price chart")
view = st.radio("View", ["Candlestick", "Line"], horizontal=True, label_visibility="collapsed")
dfp = df.tail(int(days)).copy()
has_ohlc = all(c in dfp.columns for c in ["Open", "High", "Low", "Close"])
fig = go.Figure()
if view == "Candlestick" and has_ohlc:
    fig.add_trace(go.Candlestick(x=dfp["Date"], open=dfp["Open"], high=dfp["High"],
        low=dfp["Low"], close=dfp["Close"], name="BTC-USD",
        increasing_line_color="#16c784", decreasing_line_color="#ea3943"))
else:
    fig.add_trace(go.Scatter(x=dfp["Date"], y=dfp["Close"], mode="lines",
        line=dict(color="#f0b90b", width=2), name="Close"))
fig.add_trace(go.Scatter(x=[dfp["Date"].iloc[-1]], y=[latest_price],
    mode="markers+text", marker=dict(size=13, color=sc, line=dict(color="white", width=1.5)),
    text=[signal], textposition="top center", textfont=dict(color=sc, size=13), showlegend=False))
fig.update_layout(template="plotly_dark", height=520, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_rangeslider_visible=False, hovermode="x unified",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)
