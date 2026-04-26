import threading

from dotenv import load_dotenv

load_dotenv()

import os

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agent import run_agent
from config import ADMIN_USER_IDS, ALLOWED_MODEL_PATTERNS, DEFAULT_MODEL, is_model_allowed
from store import get_model, init_db, set_model

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


if __name__ == "__main__":
    handler = SocketModeHandler(app, os.getenv("SLACK_APP_TOKEN"))
    handler.start()
