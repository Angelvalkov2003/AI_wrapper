"""
Anthropic Claude API – пълна функционалност.
- Messages: create, stream
- Count tokens
- Всички параметри: model, max_tokens, temperature, top_p, top_k, stop_sequences,
  system prompt, metadata, tools, stream
- Vision (images), tool use
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from .config import ANTHROPIC_API_KEY
except ImportError:
    try:
        from config import ANTHROPIC_API_KEY
    except ImportError:
        ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

from anthropic import Anthropic


def get_client() -> Anthropic:
    """Връща Anthropic клиент с API ключ от config."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "Задай ANTHROPIC_API_KEY в .env файл в тази папка или в променлива на средата."
        )
    return Anthropic(api_key=ANTHROPIC_API_KEY)


# ---------- Messages (create) ----------


def message_create(
    messages: list[dict],
    *,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    system: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    stop_sequences: list[str] | None = None,
    metadata: dict | None = None,
    tools: list | None = None,
    stream: bool = False,
    **kwargs,
):
    """
    Изпраща съобщения към Claude.
    messages: [{"role": "user", "content": "..."}] или content като списък от блокове (текст/image).
    """
    client = get_client()
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        params["system"] = system
    if temperature is not None:
        params["temperature"] = temperature
    if top_p is not None:
        params["top_p"] = top_p
    if top_k is not None:
        params["top_k"] = top_k
    if stop_sequences:
        params["stop_sequences"] = stop_sequences
    if metadata:
        params["metadata"] = metadata
    if tools:
        params["tools"] = tools
    params["stream"] = stream
    params.update({k: v for k, v in kwargs.items() if v is not None})
    return client.messages.create(**params)


def message_simple(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    **kwargs,
) -> str:
    """Прост еднократен отговор."""
    r = message_create(
        [{"role": "user", "content": prompt}],
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
    return (
        next((b.text for b in r.content if hasattr(b, "text") and b.text), "")
        if r.content
        else ""
    )


def message_stream(
    prompt: str,
    *,
    model: str = "claude-sonnet-4-20250514",
    system: str | None = None,
    max_tokens: int = 1024,
    content_blocks: list | None = None,
    **kwargs,
):
    """Стрийминг на отговора. content_blocks: за vision – [{"type":"text","text":...}, {"type":"image",...}]."""
    client = get_client()
    if content_blocks is not None:
        messages = [{"role": "user", "content": content_blocks}]
    else:
        messages = [{"role": "user", "content": prompt}]
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        **kwargs,
    ) as stream:
        for text in stream.text_stream:
            yield text


# ---------- Count tokens ----------


def count_tokens(
    messages: list[dict],
    *,
    model: str = "claude-sonnet-4-20250514",
    **kwargs,
) -> int:
    """Брой токени за дадени съобщения (без изпращане на заявка за генериране)."""
    client = get_client()
    r = client.messages.count_tokens(model=model, messages=messages, **kwargs)
    return r.input_tokens  # или r.input_tokens + изходни; API връща input_tokens


# ---------- Message с изображение (vision) ----------


def message_vision(
    prompt: str,
    image_path: str | Path,
    *,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    media_type: str = "image/jpeg",
    **kwargs,
) -> str:
    """Vision: текст + изображение."""
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
            ],
        }
    ]
    r = message_create(messages, model=model, max_tokens=max_tokens, **kwargs)
    return next((b.text for b in r.content if hasattr(b, "text") and b.text), "")


# ---------- Тест ----------


def main():
    print("=== Anthropic API тест ===\n")
    if not ANTHROPIC_API_KEY:
        print("Задай ANTHROPIC_API_KEY в .env за тестове.")
        return
    print("Message (кратък отговор):")
    try:
        text = message_simple("В един изречение: какво е Claude?", max_tokens=80)
        print("Отговор:", text)
    except Exception as e:
        print("Грешка:", e)
    print("\nCount tokens:")
    try:
        n = count_tokens([{"role": "user", "content": "Hello, world!"}])
        print("Токени:", n)
    except Exception as e:
        print("Грешка:", e)
    print("\nStream (първите 100 символа):")
    try:
        out = []
        for chunk in message_stream("Кажи едно изречение за AI.", max_tokens=50):
            out.append(chunk)
        print("".join(out)[:100])
    except Exception as e:
        print("Грешка:", e)


if __name__ == "__main__":
    main()
