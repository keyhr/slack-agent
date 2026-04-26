import io
import os
import threading
from datetime import date

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify

from config import SEARXNG_BASE_URL

_search_semaphore = threading.Semaphore(1)


def _brave_search(query: str, api_key: str, num_results: int) -> str:
    resp = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": num_results},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("web", {}).get("results", [])[:num_results]
    if not results:
        return "検索結果が見つかりませんでした。"
    return "\n".join(
        f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('description', '')}"
        for r in results
    )


def _searxng_search(query: str, num_results: int) -> str:
    resp = requests.get(
        f"{SEARXNG_BASE_URL}/search",
        params={"q": query, "format": "json"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])[:num_results]
    if not results:
        return "検索結果が見つかりませんでした。"
    return "\n".join(
        f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('content', '')}"
        for r in results
    )


def web_search(query: str, num_results: int = 5) -> str:
    with _search_semaphore:
        try:
            api_key = os.getenv("BRAVE_SEARCH_API_KEY", "")
            if api_key:
                return _brave_search(query, api_key, num_results)
            return _searxng_search(query, num_results)
        except Exception as e:
            return f"検索に失敗しました: {e}"


def fetch_url(url: str, max_chars: int = 8000) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        md = markdownify(str(soup), heading_style="ATX")
        return md[:max_chars]
    except Exception as e:
        return f"URLの取得に失敗しました: {e}"


def slack_search(query: str, num_results: int = 10) -> str:
    user_token = os.getenv("SLACK_USER_TOKEN", "")
    if not user_token:
        return "SLACK_USER_TOKEN が設定されていないため、Slackワークスペース検索は利用できません。"
    try:
        resp = requests.get(
            "https://slack.com/api/search.messages",
            headers={"Authorization": f"Bearer {user_token}"},
            params={"query": query, "count": num_results, "sort": "timestamp", "sort_dir": "desc"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            return f"Slack検索エラー: {data.get('error', '不明なエラー')}"
        matches = data.get("messages", {}).get("matches", [])
        if not matches:
            return "該当するメッセージが見つかりませんでした。"
        lines = []
        for m in matches:
            ts = m.get("ts", "")
            username = m.get("username") or m.get("user", "unknown")
            channel_name = m.get("channel", {}).get("name", "unknown")
            text = m.get("text", "").replace("\n", " ")[:200]
            permalink = m.get("permalink", "")
            lines.append(f"- [{channel_name}] {username}: {text}\n  {permalink}")
        return "\n".join(lines)
    except Exception as e:
        return f"Slack検索に失敗しました: {e}"


def read_file(url: str, filename: str) -> str:
    slack_token = os.getenv("SLACK_BOT_TOKEN", "")
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {slack_token}"},
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.content
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if ext in ("txt", "md", "py", "js", "ts", "json", "yaml", "yml", "toml", "sh", "rb", "go", "rs"):
            return content.decode("utf-8", errors="replace")

        if ext == "pdf":
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)

        if ext == "docx":
            import docx
            doc = docx.Document(io.BytesIO(content))
            return "\n".join(p.text for p in doc.paragraphs)

        if ext == "csv":
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            return df.head(50).to_string()

        return f"対応していないファイル形式です: .{ext}"
    except Exception as e:
        return f"ファイルの読み込みに失敗しました: {e}"


def _slack_search_tool_def():
    return {
        "type": "function",
        "function": {
            "name": "slack_search",
            "description": "Slackワークスペース内のメッセージを全文検索する。過去の会話・決定事項・情報を探すときに使う。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


TOOLS_DEFINITION = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Web検索して最新情報を取得する。最新情報・時事・リアルタイムデータが必要な場合に使う。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "URLのページ内容を取得して要約する",
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
            "name": "read_file",
            "description": "添付ファイルの内容を読み込む",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["url", "filename"],
            },
        },
    },
]

def get_tools_definition() -> list:
    tools = list(TOOLS_DEFINITION)
    if os.getenv("SLACK_USER_TOKEN"):
        tools.append(_slack_search_tool_def())
    return tools


_SYSTEM_PROMPT_BASE = "あなたはSlack上のアシスタントです。"
_SYSTEM_PROMPT_SUFFIX = (
    "ツールが不要な質問（コード説明、数学、一般知識など）は直接回答して構いません。\n\n"
    "## 最新情報を調べる際の必須手順\n\n"
    "最新情報が必要な場合、以下を**必ず**実行してください。途中で諦めることは禁止です。\n\n"
    "**Step 1: web_search で検索する**\n"
    "- クエリに「2026年」「2025年」などの年号を含めない。年号があると検索エンジンがカレンダーページを返す。\n"
    "- 代わりに「最新」「today」「latest」「速報」などを使う。\n"
    "- 例: NG→「2026年4月 株価ニュース」 / OK→「株価 最新ニュース 速報」\n\n"
    "**Step 2: 検索結果のURLを fetch_url で取得する（必須・省略不可）**\n"
    "- 検索結果に含まれるURLを必ず1件以上 fetch_url で取得すること。\n"
    "- ニュース・記事・公式サイトのURLを優先する。カレンダーや辞書サイトは除外する。\n"
    "- 1件目が役に立たなければ2件目・3件目も試す。\n"
    "- 「取得できなかった」「結果がなかった」と判断するのは fetch_url を最低1回実行した後のみ。\n\n"
    "**Step 3: 取得した本文を根拠として回答する**\n"
    "- fetch_url の結果が空または短すぎる場合は別のURLを試す。\n"
    "- 情報が古い・不十分な場合は追加で web_search → fetch_url を繰り返す。\n\n"
    "**絶対禁止**: 検索結果のスニペットだけを見て「情報が見つかりませんでした」と返答すること。"
)


def build_system_prompt() -> str:
    tool_lines = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']}"
        for t in get_tools_definition()
    )
    today = date.today().isoformat()
    return (
        f"{_SYSTEM_PROMPT_BASE}\n\n今日の日付: {today}\n\n"
        f"必要に応じて以下のツールを使ってください：\n{tool_lines}\n\n{_SYSTEM_PROMPT_SUFFIX}"
    )
