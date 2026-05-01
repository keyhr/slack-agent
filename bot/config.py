import os
import re

DEFAULT_MODEL = "google/gemma-4-31b-it:free"
MAX_AGENT_LOOPS = 8
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "https://searx.be")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

# モデル名の許可パターン（正規表現リスト）。いずれか1つにマッチすれば許可。
ALLOWED_MODEL_PATTERNS: list[re.Pattern] = [
    re.compile(r".+:free$"),  # デフォルト: :free 末尾のモデルのみ許可
]


def is_model_allowed(model: str) -> bool:
    return any(p.fullmatch(model) for p in ALLOWED_MODEL_PATTERNS)


ADMIN_USER_IDS = [uid for uid in os.getenv("ADMIN_USER_IDS", "").split(",") if uid]
