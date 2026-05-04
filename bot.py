import telebot
import schedule
import threading
import time
import logging
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
API_TOKEN              = "8307000040:AAEoGT1cBoYXK_ed_mfTffvxXG8yb4TokD8"
CHAT_IDS               = [7094045595]
SIGNAL_INTERVAL_MINUTES = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

bot = telebot.TeleBot(API_TOKEN, parse_mode="Markdown")

# ─────────────────────────────────────────────
#  ASSET MAP  (OTC label → Yahoo Finance ticker)
# ─────────────────────────────────────────────
ASSET_MAP = {
    "EUR/USD-OTC": "EURUSD=X",
    "GBP/USD-OTC": "GBPUSD=X",
    "USD/JPY-OTC": "JPY=X",
    "AUD/USD-OTC": "AUDUSD=X",
    "USD/CHF-OTC": "CHF=X",
}

signal_count = 0


# ══════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ══════════════════════════════════════════════

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def compute_macd(series: pd.Series):
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return round(float(macd.iloc[-1]), 6), round(float(signal.iloc[-1]), 6), round(float(hist.iloc[-1]), 6)


def compute_bollinger(series: pd.Series, period: int = 20):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    price = float(series.iloc[-1])
    return round(float(upper.iloc[-1]), 6), round(float(lower.iloc[-1]), 6), round(price, 6)


def compute_ema(series: pd.Series, period: int = 9) -> float:
    return round(float(series.ewm(span=period, adjust=False).mean().iloc[-1]), 6)


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14):
    lowest  = low.rolling(k).min()
    highest = high.rolling(k).max()
    stoch_k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    stoch_d = stoch_k.rolling(3).mean()
    return round(float(stoch_k.iloc[-1]), 2), round(float(stoch_d.iloc[-1]), 2)


# ══════════════════════════════════════════════
#  AI-STYLE SCORING ENGINE
#  Each indicator votes UP / DOWN / NEUTRAL
#  Weighted confidence → final prediction
# ══════════════════════════════════════════════

def ai_predict(rsi, macd_val, macd_sig, macd_hist,
               bb_upper, bb_lower, price,
               ema9, ema21,
               stoch_k, stoch_d,
               prev_close):
    """
    Rule-based weighted voting system that mimics supervised
    classification (UP / DOWN) with a confidence score.
    Weights are tuned for 1-minute binary-option style signals.
    """
    score = 0.0        # positive = UP, negative = DOWN
    total_weight = 0.0

    # ── RSI  (weight 2.0) ──────────────────────
    w = 2.0
    if rsi < 30:
        score += w        # oversold → UP
    elif rsi > 70:
        score -= w        # overbought → DOWN
    elif 30 <= rsi < 45:
        score += w * 0.5
    elif 55 < rsi <= 70:
        score -= w * 0.5
    total_weight += w

    # ── MACD histogram cross  (weight 2.5) ─────
    w = 2.5
    if macd_hist > 0 and macd_val > macd_sig:
        score += w        # bullish cross
    elif macd_hist < 0 and macd_val < macd_sig:
        score -= w        # bearish cross
    else:
        score += w * 0.1 * np.sign(macd_hist)
    total_weight += w

    # ── Bollinger Band position  (weight 1.5) ──
    w = 1.5
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        bb_pct = (price - bb_lower) / bb_range   # 0..1
        if bb_pct < 0.2:
            score += w      # near lower band → bounce UP
        elif bb_pct > 0.8:
            score -= w      # near upper band → reversal DOWN
        else:
            score += w * (0.5 - bb_pct) * 0.5
    total_weight += w

    # ── EMA 9 vs EMA 21  (weight 1.5) ──────────
    w = 1.5
    if ema9 > ema21:
        score += w
    else:
        score -= w
    total_weight += w

    # ── Stochastic  (weight 1.5) ────────────────
    w = 1.5
    if stoch_k < 20 and stoch_k > stoch_d:
        score += w        # oversold + crossover
    elif stoch_k > 80 and stoch_k < stoch_d:
        score -= w        # overbought + crossover
    elif stoch_k < 50:
        score += w * 0.3
    else:
        score -= w * 0.3
    total_weight += w

    # ── Momentum (price vs prev close)  (weight 1.0) ─
    w = 1.0
    momentum = (price - prev_close) / (prev_close + 1e-10)
    score += w * np.sign(momentum) * min(abs(momentum) * 1000, 1)
    total_weight += w

    # ── Confidence ──────────────────────────────
    confidence = abs(score) / total_weight  # 0..1
    direction  = "UP" if score >= 0 else "DOWN"
    confidence_pct = round(min(confidence * 100 + 50, 98), 1)   # scale to 50–98%

    return direction, confidence_pct, round(score, 3)


# ══════════════════════════════════════════════
#  FETCH LIVE DATA + ANALYSE
# ══════════════════════════════════════════════

def fetch_and_analyse(otc_label: str) -> dict | None:
    ticker_sym = ASSET_MAP.get(otc_label)
    if not ticker_sym:
        return None
    try:
        df = yf.download(ticker_sym, period="2d", interval="1m",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            log.warning("Not enough data for %s", otc_label)
            return None

        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()

        rsi                         = compute_rsi(close)
        macd_val, macd_sig, macd_h  = compute_macd(close)
        bb_upper, bb_lower, price   = compute_bollinger(close)
        ema9                        = compute_ema(close, 9)
        ema21                       = compute_ema(close, 21)
        stoch_k, stoch_d            = compute_stochastic(high, low, close)
        prev_close                  = float(close.iloc[-2])

        direction, confidence, raw_score = ai_predict(
            rsi, macd_val, macd_sig, macd_h,
            bb_upper, bb_lower, price,
            ema9, ema21, stoch_k, stoch_d, prev_close
        )

        return {
            "asset":      otc_label,
            "price":      price,
            "direction":  direction,
            "confidence": confidence,
            "rsi":        rsi,
            "macd_hist":  macd_h,
            "ema9":       ema9,
            "ema21":      ema21,
            "stoch_k":    stoch_k,
            "bb_upper":   bb_upper,
            "bb_lower":   bb_lower,
        }
    except Exception as e:
        log.error("Analysis error for %s: %s", otc_label, e)
        return None


# ══════════════════════════════════════════════
#  SIGNAL MESSAGE BUILDER
# ══════════════════════════════════════════════

def build_signal_message(data: dict) -> str:
    global signal_count
    signal_count += 1

    d         = data["direction"]
    conf      = data["confidence"]
    trend     = "📈" if d == "UP" else "📉"
    arrow     = "✅ CALL (BUY)" if d == "UP" else "🔴 PUT (SELL)"
    now       = datetime.now().strftime("%H:%M:%S")
    date      = datetime.now().strftime("%d %b %Y")

    # Strength label
    if conf >= 85:
        strength = "🔥 STRONG"
    elif conf >= 70:
        strength = "💪 MEDIUM"
    else:
        strength = "⚠️ WEAK"

    # RSI label
    rsi = data["rsi"]
    if rsi < 30:
        rsi_label = "Oversold 🟢"
    elif rsi > 70:
        rsi_label = "Overbought 🔴"
    else:
        rsi_label = "Neutral ⚪"

    # MACD trend
    macd_trend = "Bullish 📈" if data["macd_hist"] > 0 else "Bearish 📉"

    # EMA trend
    ema_trend  = "Bullish 📈" if data["ema9"] > data["ema21"] else "Bearish 📉"

    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *QUTEX AI SIGNAL BOT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{trend} Asset       : `{data['asset']}`\n"
        f"💲 Live Price  : `{data['price']}`\n"
        f"🎯 Signal      : *{arrow}*\n"
        f"⏱ Duration    : `1 MIN`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🧠 *AI ANALYSIS*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 RSI         : `{rsi}` — {rsi_label}\n"
        f"📉 MACD        : {macd_trend}\n"
        f"📈 EMA 9/21    : {ema_trend}\n"
        f"📉 Stochastic  : `{data['stoch_k']}`\n"
        f"📐 BB Upper    : `{data['bb_upper']}`\n"
        f"📐 BB Lower    : `{data['bb_lower']}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔮 AI Confidence : *{conf}%* {strength}\n"
        f"🕐 Time          : `{now}`\n"
        f"📅 Date          : `{date}`\n"
        f"🔢 Signal #      : `{signal_count}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Analysis: QUTEX AI Engine_\n"
    )


def build_no_signal_message(asset: str) -> str:
    global signal_count
    signal_count += 1
    now  = datetime.now().strftime("%H:%M:%S")
    date = datetime.now().strftime("%d %b %Y")
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *QUTEX AI SIGNAL BOT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ Asset : `{asset}`\n"
        f"❌ *কোনো স্পষ্ট সিগনাল নেই*\n"
        f"💤 Market এখন sideways চলছে\n"
        f"🕐 Time  : `{now}`\n"
        f"📅 Date  : `{date}`\n"
        f"🔢 Signal # : `{signal_count}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Waiting for clear trend..._\n"
    )


# ══════════════════════════════════════════════
#  BEST ASSET PICKER
#  Analyses all assets, picks highest-confidence
# ══════════════════════════════════════════════

def pick_best_signal() -> tuple[str, dict | None]:
    """Return (message, data_or_None) for the best asset."""
    best_data = None
    best_conf = 0

    for otc_label in ASSET_MAP.keys():
        data = fetch_and_analyse(otc_label)
        if data and data["confidence"] > best_conf:
            best_conf = data["confidence"]
            best_data = data

    if best_data and best_conf >= 60:          # minimum confidence threshold
        return build_signal_message(best_data), best_data
    else:
        asset = list(ASSET_MAP.keys())[0]
        return build_no_signal_message(asset), None


# ══════════════════════════════════════════════
#  SEND SIGNALS
# ══════════════════════════════════════════════

def send_signal_to_all():
    msg, data = pick_best_signal()
    for chat_id in CHAT_IDS:
        try:
            bot.send_message(chat_id, msg)
            if data:
                log.info("Signal #%d sent | %s | %s | conf=%.1f%%",
                         signal_count, data["asset"], data["direction"], data["confidence"])
            else:
                log.info("No-signal message sent (#%d)", signal_count)
        except Exception as e:
            log.error("Failed to send to %s: %s", chat_id, e)


# ══════════════════════════════════════════════
#  BOT COMMANDS
# ══════════════════════════════════════════════

@bot.message_handler(commands=["start"])
def cmd_start(message):
    bot.reply_to(message,
        "✅ *Qutex AI Signal Bot চালু!*\n\n"
        "🧠 Live Market Analysis সহ প্রতি *1 মিনিটে* সিগনাল আসবে।\n\n"
        "📌 *Commands:*\n"
        "/signal — এখনই AI সিগনাল নাও\n"
        "/analyze EUR/USD-OTC — নির্দিষ্ট asset বিশ্লেষণ\n"
        "/status — বটের অবস্থা\n"
        "/assets — সব asset দেখো"
    )


@bot.message_handler(commands=["signal"])
def cmd_signal(message):
    bot.reply_to(message, "⏳ _AI বিশ্লেষণ চলছে..._")
    msg, _ = pick_best_signal()
    bot.reply_to(message, msg)


@bot.message_handler(commands=["analyze"])
def cmd_analyze(message):
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.reply_to(message, "⚠️ Usage: `/analyze EUR/USD-OTC`")
        return
    asset = parts[1].strip().upper()
    if asset not in ASSET_MAP:
        bot.reply_to(message,
            f"❌ `{asset}` পাওয়া যায়নি।\n"
            f"Available: {', '.join(f'`{a}`' for a in ASSET_MAP)}")
        return
    bot.reply_to(message, f"⏳ _{asset} বিশ্লেষণ চলছে..._")
    data = fetch_and_analyse(asset)
    if data:
        bot.reply_to(message, build_signal_message(data))
    else:
        bot.reply_to(message, f"❌ `{asset}` এর data পাওয়া যায়নি। পরে চেষ্টা করুন।")


@bot.message_handler(commands=["status"])
def cmd_status(message):
    bot.reply_to(message,
        f"✅ *Bot চালু আছে*\n"
        f"🧠 AI Engine: `Active`\n"
        f"📡 Data Source: `Yahoo Finance`\n"
        f"📊 Indicators: `RSI, MACD, BB, EMA, Stochastic`\n"
        f"🔢 মোট সিগনাল: `{signal_count}`\n"
        f"⏱ Interval: `{SIGNAL_INTERVAL_MINUTES} min`"
    )


@bot.message_handler(commands=["assets"])
def cmd_assets(message):
    asset_list = "\n".join(f"• `{a}`" for a in ASSET_MAP)
    bot.reply_to(message, f"📋 *Available Assets:*\n{asset_list}")


# ══════════════════════════════════════════════
#  SCHEDULER THREAD
# ══════════════════════════════════════════════

def run_scheduler():
    schedule.every(SIGNAL_INTERVAL_MINUTES).minutes.do(send_signal_to_all)
    while True:
        schedule.run_pending()
        time.sleep(1)


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    log.info("🚀 Qutex AI Signal Bot starting...")
    send_signal_to_all()                                      # immediate first signal

    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()

    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=20)
        except Exception as e:
            log.error("Polling error: %s — restarting in 10s", e)
            time.sleep(10)
