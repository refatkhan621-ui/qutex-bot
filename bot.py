import telebot
import schedule
import threading
import time
import random
import logging
from datetime import datetime

API_TOKEN = "8307000040:AAEoGT1cBoYXK_ed_mfTffvxXG8yb4TokD8"
CHAT_IDS = [7094045595]
SIGNAL_INTERVAL_MINUTES = 1

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

bot = telebot.TeleBot(API_TOKEN, parse_mode="Markdown")

ASSETS = ["EUR/USD-OTC","GBP/USD-OTC","USD/JPY-OTC",
          "AUD/USD-OTC","USD/BDT-OTC","BTC/USD-OTC"]
SIGNALS = [("UP","✅ CALL (BUY)"),("DOWN","🔴 PUT (SELL)")]
ACCURACY = ["78%","82%","75%","80%","85%"]
signal_count = 0

def generate_signal():
    global signal_count
    signal_count += 1
    asset = random.choice(ASSETS)
    direction, arrow = random.choice(SIGNALS)
    acc = random.choice(ACCURACY)
    now = datetime.now().strftime("%H:%M:%S")
    date = datetime.now().strftime("%d %b %Y")
    trend = "📈" if direction == "UP" else "📉"
    return (
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🤖 *QUTEX SIGNAL BOT*\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"{trend} Asset    : `{asset}`\n"
        f"🎯 Signal   : *{arrow}*\n"
        f"⏱ Duration : `1 MIN`\n"
        f"📊 Accuracy : `{acc}`\n"
        f"🕐 Time     : `{now}`\n"
        f"📅 Date     : `{date}`\n"
        f"🔢 Signal # : `{signal_count}`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"_Analysis: Trader Rifat_\n"
        f"⚠️ _Demo only_"
    )

def send_signal_to_all():
    msg = generate_signal()
    for chat_id in CHAT_IDS:
        try:
            bot.send_message(chat_id, msg)
            log.info("Signal #%d sent", signal_count)
        except Exception as e:
            log.error("Failed: %s", e)

@bot.message_handler(commands=["start"])
def cmd_start(message):
    bot.reply_to(message,
        f"✅ *Qutex Signal Bot চালু!*\n"
        f"প্রতি *1 মিনিটে* সিগনাল আসবে।\n\n"
        f"/signal — এখনই সিগনাল\n"
        f"/status — বটের অবস্থা")

@bot.message_handler(commands=["signal"])
def cmd_signal(message):
    bot.reply_to(message, generate_signal())

@bot.message_handler(commands=["status"])
def cmd_status(message):
    bot.reply_to(message,
        f"✅ *Bot চালু আছে*\n"
        f"🔢 মোট সিগনাল: `{signal_count}`")

def run_scheduler():
    schedule.every(SIGNAL_INTERVAL_MINUTES).minutes.do(send_signal_to_all)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    send_signal_to_all()
    t = threading.Thread(target=run_scheduler, daemon=True)
    t.start()
    while True:
        try:
            bot.infinity_polling(timeout=30, long_polling_timeout=20)
        except Exception as e:
            log.error("Polling error: %s", e)
            time.sleep(10)
