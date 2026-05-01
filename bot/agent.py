import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, RateLimitError

from config import LLM_API_KEY, LLM_BASE_URL, MAX_AGENT_LOOPS
from tools import build_system_prompt, fetch_url, get_tools_definition, read_file, slack_search, web_search

openrouter_client = OpenAI(
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
)


_LATEX_SYMBOLS: dict[str, str] = {
    # Arrows
    r"\rightarrow": "→", r"\to": "→", r"\leftarrow": "←", r"\gets": "←",
    r"\uparrow": "↑", r"\downarrow": "↓", r"\leftrightarrow": "↔",
    r"\Rightarrow": "⇒", r"\Leftarrow": "⇐", r"\Leftrightarrow": "⇔",
    r"\nearrow": "↗", r"\searrow": "↘", r"\mapsto": "↦",
    # Greek lowercase
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\varepsilon": "ε", r"\epsilon": "ε", r"\zeta": "ζ", r"\eta": "η",
    r"\vartheta": "θ", r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ",
    r"\lambda": "λ", r"\mu": "μ", r"\nu": "ν", r"\xi": "ξ",
    r"\varpi": "π", r"\pi": "π", r"\varrho": "ρ", r"\rho": "ρ",
    r"\varsigma": "ς", r"\sigma": "σ", r"\tau": "τ", r"\upsilon": "υ",
    r"\varphi": "φ", r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    # Greek uppercase
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ",
    r"\Xi": "Ξ", r"\Pi": "Π", r"\Sigma": "Σ", r"\Upsilon": "Υ",
    r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω",
    # Operators & relations
    r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\cdot": "·", r"\circ": "∘", r"\bullet": "•",
    r"\leq": "≤", r"\le": "≤", r"\geq": "≥", r"\ge": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\sim": "∼", r"\simeq": "≃", r"\propto": "∝",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫", r"\oint": "∮",
    r"\sqrt": "√", r"\therefore": "∴", r"\because": "∵",
    # Sets & logic
    r"\in": "∈", r"\notin": "∉", r"\ni": "∋",
    r"\subset": "⊂", r"\supset": "⊃", r"\subseteq": "⊆", r"\supseteq": "⊇",
    r"\cup": "∪", r"\cap": "∩", r"\emptyset": "∅", r"\varnothing": "∅",
    r"\forall": "∀", r"\exists": "∃", r"\nexists": "∄",
    r"\neg": "¬", r"\lnot": "¬", r"\land": "∧", r"\wedge": "∧",
    r"\lor": "∨", r"\vee": "∨",
    r"\oplus": "⊕", r"\otimes": "⊗", r"\odot": "⊙",
    # Brackets & misc
    r"\langle": "⟨", r"\rangle": "⟩", r"\lfloor": "⌊", r"\rfloor": "⌋",
    r"\lceil": "⌈", r"\rceil": "⌉",
    r"\ldots": "…", r"\cdots": "⋯", r"\vdots": "⋮", r"\ddots": "⋱",
    r"\hbar": "ℏ", r"\ell": "ℓ", r"\Re": "ℜ", r"\Im": "ℑ",
    # Common fractions
    r"\frac{1}{2}": "½", r"\frac{1}{3}": "⅓", r"\frac{2}{3}": "⅔",
    r"\frac{1}{4}": "¼", r"\frac{3}{4}": "¾",
}

# Superscript/subscript digit maps
_SUPERSCRIPTS = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPTS = str.maketrans("0123456789+-=()aeinoruvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢₙₒᵣᵤᵥₓ")


def _apply_latex_symbols(s: str) -> str:
    for cmd in sorted(_LATEX_SYMBOLS, key=len, reverse=True):
        s = re.sub(re.escape(cmd) + r"(?![a-zA-Z])", _LATEX_SYMBOLS[cmd], s)
    # Simple superscripts: ^{n} or ^n (single char)
    s = re.sub(r"\^\{([0-9+\-=()n]+)\}", lambda m: m.group(1).translate(_SUPERSCRIPTS), s)
    s = re.sub(r"\^([0-9n])", lambda m: m.group(1).translate(_SUPERSCRIPTS), s)
    # Simple subscripts: _{n} or _n (single char)
    s = re.sub(r"_\{([0-9+\-=()aeinoruvx]+)\}", lambda m: m.group(1).translate(_SUBSCRIPTS), s)
    s = re.sub(r"_([0-9])", lambda m: m.group(1).translate(_SUBSCRIPTS), s)
    return s


def _process_latex(text: str) -> str:
    # Block math $$...$$ → code block
    def _block(m: re.Match) -> str:
        inner = _apply_latex_symbols(m.group(1).strip())
        return f"```\n{inner}\n```"
    text = re.sub(r"\$\$(.+?)\$\$", _block, text, flags=re.DOTALL)

    # Inline math $...$ → unicode; backtick if still complex
    def _inline(m: re.Match) -> str:
        inner = _apply_latex_symbols(m.group(1))
        return f"`{inner}`" if ("\\" in inner or "{" in inner) else inner
    text = re.sub(r"\$([^$\n]+?)\$", _inline, text)

    return text


def _convert_tables(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        if re.match(r"^\|.+\|", lines[i]):
            block: list[str] = []
            while i < len(lines) and re.match(r"^\|.+\|", lines[i]):
                block.append(lines[i])
                i += 1
            data = [l for l in block if not re.match(r"^\|[\s\-|:]+\|$", l.strip())]
            if data:
                out.append("```")
                for row in data:
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    out.append("  |  ".join(cells))
                out.append("```")
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def _md_to_mrkdwn(text: str) -> str:
    # LaTeX → Unicode / code block
    text = _process_latex(text)
    # Tables → code block
    text = _convert_tables(text)
    # Horizontal rules → blank line
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Headers → bold (placeholder prevents italic pass from clobbering)
    text = re.sub(r"^#{1,6} (.+)$", lambda m: f"\x00B\x00{m.group(1)}\x00E\x00", text, flags=re.MULTILINE)
    # Bold **text** → placeholder
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: f"\x00B\x00{m.group(1)}\x00E\x00", text, flags=re.DOTALL)
    # Unordered list "* item" → "• item" (before italic pass)
    text = re.sub(r"^(\s*)\* ", r"\1• ", text, flags=re.MULTILINE)
    # Italic *text* → _text_ (safe now: no **bold** markers remain)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"_\1_", text)
    # Restore bold placeholders
    text = text.replace("\x00B\x00", "*").replace("\x00E\x00", "*")
    # Links [text](url) → <url|text>
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    # Strikethrough ~~text~~ → ~text~
    text = re.sub(r"~~(.+?)~~", r"~\1~", text, flags=re.DOTALL)
    return text


def _tool_label(name: str, args: dict) -> str:
    if name == "web_search":
        return f"🔍 検索中: `{args.get('query')}`"
    if name == "fetch_url":
        return f"🌐 取得中: {args.get('url')}"
    if name == "read_file":
        return f"📄 読み込み中: `{args.get('filename')}`"
    if name == "slack_search":
        return f"💬 Slack検索中: `{args.get('query')}`"
    return f"⚙️ {name} 実行中"


def _tool_label_done(name: str, args: dict, result: str) -> str:
    if name == "web_search":
        count = result.count("\n- [") + (1 if result.startswith("- [") else 0)
        return f"🔍 検索完了: `{args.get('query')}` ({count}件)"
    if name == "fetch_url":
        return f"🌐 取得完了: {args.get('url')} ({len(result):,}文字)"
    if name == "read_file":
        return f"📄 読み込み完了: `{args.get('filename')}` ({len(result):,}文字)"
    if name == "slack_search":
        count = result.count("\n- [") + (1 if result.startswith("- [") else 0)
        return f"💬 Slack検索完了: `{args.get('query')}` ({count}件)"
    return f"⚙️ {name} 完了"


def _execute_tool(name: str, args: dict) -> str:
    if name == "web_search":
        return web_search(args["query"])
    if name == "fetch_url":
        return fetch_url(args["url"])
    if name == "read_file":
        return read_file(args["url"], args["filename"])
    if name == "slack_search":
        return slack_search(args["query"])
    return "不明なツールです"


def _create_with_retry(model: str, messages: list, max_retries: int = 3, tools: list | None = None):
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(max_retries):
        try:
            return openrouter_client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools is not None else get_tools_definition(),  # type: ignore[arg-type]
            )
        except RateLimitError as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)  # 1s, 2s
    raise last_exc


def _run_tool_calls_parallel(tool_calls, slack_client, channel, thread_ts) -> dict[str, str]:
    parsed = [(tc, tc.function.name, json.loads(tc.function.arguments)) for tc in tool_calls]

    label_ts: dict[str, str] = {}
    for tc, name, args in parsed:
        resp = slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=_tool_label(name, args),
        )
        label_ts[tc.id] = resp["ts"]

    def _exec(item):
        tc, name, args = item
        result = _execute_tool(name, args)
        slack_client.chat_update(
            channel=channel,
            ts=label_ts[tc.id],
            text=_tool_label_done(name, args, result),
        )
        return tc.id, result

    with ThreadPoolExecutor(max_workers=len(parsed)) as executor:
        return dict(executor.map(_exec, parsed))


def run_agent(
    user_message: str,
    model: str,
    slack_client,
    channel: str,
    thread_ts: str,
    file_info: dict | None = None,
) -> None:
    if file_info:
        user_message += (
            f"\n\n[添付ファイル]\nURL: {file_info['url']}\n"
            f"ファイル名: {file_info['filename']}"
        )

    messages: list = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_message},
    ]
    thinking_msg = slack_client.chat_postMessage(
        channel=channel, thread_ts=thread_ts, text="⌛ 考え中..."
    )

    def _update_thinking(text: str) -> None:
        slack_client.chat_update(channel=channel, ts=thinking_msg["ts"], text=text)

    def _post_answer(content: str) -> None:
        mrkdwn = _md_to_mrkdwn(content)
        blocks: list = [{"type": "divider"}]
        for i in range(0, len(mrkdwn), 3000):
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": mrkdwn[i : i + 3000]}})
        slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            blocks=blocks,
            text=content,
        )

    try:
        for step in range(1, MAX_AGENT_LOOPS + 1):
            _update_thinking(f"⌛ 考え中... (ステップ {step})")
            response = _create_with_retry(model, messages)
            choice = response.choices[0]
            msg = choice.message
            messages.append(msg)

            if choice.finish_reason == "stop":
                _update_thinking("✅ 完了")
                _post_answer(msg.content or "（応答なし）")
                return

            tool_calls = msg.tool_calls or []
            if tool_calls:
                labels = " / ".join(
                    _tool_label(tc.function.name, json.loads(tc.function.arguments))
                    for tc in tool_calls
                )
                _update_thinking(f"🔧 ツール実行中 (ステップ {step}): {labels}")
                results = _run_tool_calls_parallel(tool_calls, slack_client, channel, thread_ts)
                for tc in tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": results[tc.id],
                        }
                    )

        _update_thinking("⚠️ 処理が長くなりすぎたため中断しました。")
    except RateLimitError:
        _update_thinking(
            f"⏳ モデル `{model}` がレート制限中です。しばらく待つか、`/register` で別のモデルに切り替えてください。"
        )
    except Exception as e:
        _update_thinking(f"❌ エラーが発生しました: {e}")
