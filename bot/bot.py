import threading

from dotenv import load_dotenv

load_dotenv()

import os

from apscheduler.schedulers.background import BackgroundScheduler
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent import run_agent
from config import ADMIN_USER_IDS, ALLOWED_MODEL_PATTERNS, DEFAULT_MODEL, is_model_allowed
from news_digest import run_news_digest
from store import get_config, get_model, init_db, set_config, set_model

app = App(token=os.getenv("SLACK_BOT_TOKEN"))

init_db()


@app.event("app_mention")
def handle_mention(event, client):
    channel = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    text = event["text"]

    model = get_model(channel) or DEFAULT_MODEL

    file_info = None
    if files := event.get("files"):
        f = files[0]
        file_info = {"url": f["url_private"], "filename": f["name"]}

    threading.Thread(
        target=run_agent,
        args=(text, model, client, channel, thread_ts, file_info),
        daemon=True,
    ).start()


@app.command("/register")
def handle_register(ack, command, client):
    ack()

    user_id = command["user_id"]
    channel = command["channel_id"]
    text = command["text"].strip()

    if user_id not in ADMIN_USER_IDS:
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text="⛔ このコマンドは管理者のみ使用できます。",
        )
        return

    if not text:
        pattern_list = "\n".join(f"• `{p.pattern}`" for p in ALLOWED_MODEL_PATTERNS)
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text=f"使い方: `/register <モデル名>`\n\n許可パターン（正規表現）:\n{pattern_list}",
        )
        return

    if not is_model_allowed(text):
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text=f"⚠️ `{text}` は使用できません。\n`/register` で一覧を確認してください。",
        )
        return

    set_model(channel_id=channel, model=text, updated_by=user_id)
    client.chat_postMessage(
        channel=channel,
        text=f"✅ このチャンネルのモデルを `{text}` に設定しました。（<@{user_id}>）",
    )


@app.command("/subscribe-news")
def handle_subscribe_news(ack, command, client):
    ack()
    user_id = command["user_id"]
    channel = command["channel_id"]

    if user_id not in ADMIN_USER_IDS:
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text="⛔ このコマンドは管理者のみ使用できます。",
        )
        return

    set_config("news_channel", channel)
    client.chat_postMessage(
        channel=channel,
        text=f"✅ このチャンネルをデイリーニュース配信先に設定しました。（<@{user_id}>）",
    )


@app.command("/news-test")
def handle_news_test(ack, command, client):
    ack()
    user_id = command["user_id"]
    channel = command["channel_id"]

    if user_id not in ADMIN_USER_IDS:
        client.chat_postEphemeral(
            channel=channel,
            user=user_id,
            text="⛔ このコマンドは管理者のみ使用できます。",
        )
        return

    client.chat_postEphemeral(
        channel=channel,
        user=user_id,
        text="⏳ ニュースダイジェストのテスト実行を開始しました...",
    )
    threading.Thread(target=run_news_digest, args=(app.client,), daemon=True).start()


if __name__ == "__main__":
    news_hour = int(os.getenv("NEWS_CRON_HOUR", "6"))
    news_minute = int(os.getenv("NEWS_CRON_MINUTE", "0"))
    news_tz = os.getenv("NEWS_CRON_TZ", "Asia/Tokyo")

    scheduler = BackgroundScheduler(timezone=news_tz)
    scheduler.add_job(
        lambda: run_news_digest(app.client),
        trigger="cron",
        hour=news_hour,
        minute=news_minute,
    )
    scheduler.start()
    print(f"[scheduler] News digest scheduled at {news_hour:02d}:{news_minute:02d} {news_tz}", flush=True)

    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
