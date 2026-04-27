import json
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests
from openai import RateLimitError  # type: ignore[import-untyped]

from agent import _create_with_retry, _md_to_mrkdwn
from config import DEFAULT_MODEL
from store import get_config, get_model
from tools import fetch_url, web_search

NEWS_DIGEST_MAX_LOOPS = 8

NEWS_TOPICS = [
    "国際情勢（主要な国際ニュース・外交・紛争など）",
    "日本のトップニュース（政治・社会・経済など）",
    "エネルギー関連ニュース（石油・ガス・再生可能エネルギー・電力市場など）",
    "経済・株式マーケットのトップニュース（日本・米国・世界の市場動向）",
]

# RSS seed URLs pre-fetched per topic — bypasses unreliable model search-query generation
_TOPIC_SEED_URLS: dict[str, list[str]] = {
    "国際情勢（主要な国際ニュース・外交・紛争など）": [
        "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "https://www3.nhk.or.jp/rss/news/cat6.xml",
    ],
    "日本のトップニュース（政治・社会・経済など）": [
        "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "https://www3.nhk.or.jp/rss/news/cat1.xml",
    ],
    "エネルギー関連ニュース（石油・ガス・再生可能エネルギー・電力市場など）": [
        "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "https://www3.nhk.or.jp/rss/news/cat2.xml",
    ],
    "経済・株式マーケットのトップニュース（日本・米国・世界の市場動向）": [
        "https://www3.nhk.or.jp/rss/news/cat0.xml",
        "https://www3.nhk.or.jp/rss/news/cat2.xml",
    ],
}

_NEWS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "URLのページ内容を取得する",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Web検索して最新情報を取得する（ニュース一覧に情報が不足している場合のみ使う）",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_TOPIC_SYSTEM_PROMPT = """\
あなたはニュースキュレーターです。提供されたニュース一覧から指定トピックに関連する記事を選び、fetch_url で本文を取得してダイジェストを作成します。

## 手順（必ず順番通りに実行）
1. 提供されたニュース一覧の中から、指定トピックに関連しそうな記事URLを3件以上特定する
2. それらのURLを fetch_url で取得して本文を確認する
3. 本文が空・短すぎる・無関係な場合は一覧の別のURLを試す
4. 一覧にトピック関連記事が少ない場合のみ web_search を追加で使う
   - web_search のクエリに年号・日付を絶対に含めない（カレンダーページが返るため）
   - 「最新」「速報」「latest」「breaking」を使う
5. 3件以上の記事本文が揃ったら出力する

## 出力形式（Slack mrkdwn）
- *トピック名* をヘッダーとして使う
- 対象日前後に報道された個別ニュースを **最低3件** 箇条書きで記載
- 各項目: 1〜2文 + ソースURL（<URL|タイトル> 形式）
- 前置き・後置き・挨拶は不要（ダイジェスト本文のみ出力）

## 絶対禁止
- 提供一覧のURLを fetch_url で1件も取得せずに終了すること
- 月次・週次まとめのみを情報源にすること
- 2件以下で出力を終了すること
"""


def _fetch_rss_headlines(url: str, max_items: int = 25) -> list[dict]:
    """Fetch RSS and return list of {title, link, description, pubDate}."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            desc = (item.findtext("description") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if not link:
                guid = item.find("guid")
                if guid is not None and guid.get("isPermaLink") != "false":
                    link = (guid.text or "").strip()
            if title and link:
                items.append({
                    "title": title,
                    "link": link,
                    "description": desc[:150] if desc else "",
                    "pubDate": pub_date,
                })
        return items[:max_items]
    except Exception as e:
        print(f"[news_digest] RSS fetch failed for {url}: {e}", flush=True)
        return []


def _build_headlines_context(topic: str) -> str:
    """Pre-fetch RSS headlines for a topic and return formatted string."""
    seed_urls = _TOPIC_SEED_URLS.get(topic, [])
    seen: set[str] = set()
    items: list[dict] = []
    for url in seed_urls:
        for item in _fetch_rss_headlines(url):
            if item["link"] not in seen:
                seen.add(item["link"])
                items.append(item)

    if not items:
        return ""

    lines = [f"## 本日取得ニュース一覧（{len(items)}件）"]
    for item in items:
        lines.append(f"- {item['title']}")
        lines.append(f"  URL: {item['link']}")
        if item["description"]:
            lines.append(f"  概要: {item['description']}")
        if item["pubDate"]:
            lines.append(f"  日時: {item['pubDate']}")
    return "\n".join(lines)


def _execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(args["query"])
    if name == "fetch_url":
        return fetch_url(args["url"])
    return "不明なツールです"


def _create_topic_digest(model: str, topic: str, yesterday: str) -> str:
    """Run a single-topic news agent and return the mrkdwn summary."""
    headlines_context = _build_headlines_context(topic)

    if headlines_context:
        user_message = (
            f"以下は本日取得した最新ニュース一覧です。\n\n"
            f"{headlines_context}\n\n"
            f"トピック「{topic}」に関連するニュースを上記から3件以上選び、"
            f"各記事のURLを fetch_url で取得して本文を確認してください。"
            f"対象日（{yesterday} 前後）のニュースをダイジェストにまとめてください。"
        )
    else:
        user_message = (
            f"トピック「{topic}」について、{yesterday} に報道されたニュースを調べてダイジェストを作成してください。\n"
            "【重要】検索クエリに日付・年号を含めないこと。「最新」「速報」「latest」を使うこと。\n"
            "web_search → fetch_url の順に情報を収集し、最低3件のニュースを挙げてください。"
        )

    messages: list = [
        {
            "role": "system",
            "content": (
                _TOPIC_SYSTEM_PROMPT
                + f"\n\n今日の日付: {date.today().isoformat()}"
                + f"\n対象日（昨日）: {yesterday}"
            ),
        },
        {"role": "user", "content": user_message},
    ]

    for _ in range(NEWS_DIGEST_MAX_LOOPS):
        response = _create_with_retry(model, messages, tools=_NEWS_TOOLS)
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)

        if choice.finish_reason == "stop":
            return msg.content or "（応答なし）"

        tool_calls = msg.tool_calls or []
        for tc in tool_calls:
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return f"*{topic}*\n（情報収集がループ上限に達しました）"


def _create_news_digest(model: str) -> str:
    """Run one sub-agent per topic in parallel and combine results."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    topic_results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=len(NEWS_TOPICS)) as executor:
        futures = {
            executor.submit(_create_topic_digest, model, topic, yesterday): i
            for i, topic in enumerate(NEWS_TOPICS)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                topic_results[idx] = future.result()
            except Exception as e:
                topic_results[idx] = f"*{NEWS_TOPICS[idx]}*\n（エラー: {e}）"

    return "\n\n---\n\n".join(topic_results[i] for i in range(len(NEWS_TOPICS)))


def run_news_digest(slack_client) -> None:
    channel_id = get_config("news_channel") or os.getenv("NEWS_CHANNEL_ID", "").strip()
    if not channel_id:
        print("[news_digest] No news channel configured. Set it with /news-channel.", flush=True)
        return

    model = get_model(channel_id) or os.getenv("NEWS_MODEL", "").strip() or DEFAULT_MODEL
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    header = f":newspaper: *デイリーニュースダイジェスト* ({yesterday})"

    print(f"[news_digest] Starting digest for {yesterday} → #{channel_id}", flush=True)
    try:
        content = _create_news_digest(model)
        mrkdwn = _md_to_mrkdwn(content)

        blocks: list = [
            {"type": "section", "text": {"type": "mrkdwn", "text": header}},
            {"type": "divider"},
        ]
        for i in range(0, len(mrkdwn), 3000):
            blocks.append(
                {"type": "section", "text": {"type": "mrkdwn", "text": mrkdwn[i : i + 3000]}}
            )

        slack_client.chat_postMessage(
            channel=channel_id,
            blocks=blocks,
            text=f"{header}\n{content}",
        )
        print("[news_digest] Digest posted successfully.", flush=True)

    except RateLimitError:
        slack_client.chat_postMessage(
            channel=channel_id,
            text=":hourglass: ニュースダイジェストの生成中にレート制限が発生しました。モデルを変更するか、しばらくしてから再試行してください。",
        )
    except Exception as e:
        print(f"[news_digest] Error: {e}", flush=True)
        slack_client.chat_postMessage(
            channel=channel_id,
            text=f":x: ニュースダイジェストの生成に失敗しました: {e}",
        )
