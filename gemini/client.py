"""
Gemini API – пълна функционалност.
- Генериране на съдържание (текст, стрийминг)
- Списък модели
- Safety settings
- Function calling / tools
- Embeddings (ако е налично в SDK)
- Токени (count)
- Файлове (upload)
- Чатове (chat sessions)
- Конфигурация: temperature, top_p, top_k, max_output_tokens, stop_sequences, и др.
"""
from __future__ import annotations

import os
from pathlib import Path

# Зареждане на config от тази папка, за да се чете gemini/.env
try:
    from .config import GEMINI_API_KEY  # когато се импортира като gemini.client (напр. от gui)
except ImportError:
    try:
        from config import GEMINI_API_KEY  # когато се стартира client.py от папка gemini/
    except ImportError:
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

from google import genai
from google.genai import types


def get_client() -> genai.Client:
    """Връща Gemini клиент с API ключ от config."""
    if not GEMINI_API_KEY:
        raise ValueError(
            "Задай GEMINI_API_KEY в .env файл в тази папка или в променлива на средата."
        )
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------- Генериране на съдържание ----------


def generate_content(
    prompt: str,
    *,
    model: str = "gemini-2.0-flash",
    system_instruction: str | None = None,
    temperature: float = 0.7,
    top_p: float | None = 0.95,
    top_k: int | None = 40,
    max_output_tokens: int | None = 2048,
    stop_sequences: list[str] | None = None,
    safety_settings: list[dict] | None = None,
    response_mime_type: str | None = None,
    image_parts: list[tuple[bytes, str]] | None = None,
    **kwargs,
) -> str:
    """
    Генерира текст чрез Gemini. image_parts: списък от (bytes, mime_type) за снимки (vision).
    """
    client = get_client()
    config_dict = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_output_tokens": max_output_tokens,
        "system_instruction": system_instruction,
    }
    if stop_sequences:
        config_dict["stop_sequences"] = stop_sequences
    if safety_settings:
        config_dict["safety_settings"] = [
            types.SafetySetting(category=s.get("category"), threshold=s.get("threshold"))
            for s in safety_settings
        ]
    if response_mime_type:
        config_dict["response_mime_type"] = response_mime_type
    config_dict.update(kwargs)
    config = types.GenerateContentConfig(**{k: v for k, v in config_dict.items() if v is not None})

    contents = _build_multimodal_contents(prompt, image_parts) if image_parts else prompt
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    return response.text or ""


def _build_multimodal_contents(prompt: str, image_parts: list[tuple[bytes, str]]):
    """Сглобява contents за API: снимки + текст (за vision/multimodal)."""
    parts = []
    for img_bytes, mime_type in image_parts:
        blob = types.Blob(mime_type=mime_type or "image/jpeg", data=img_bytes)
        parts.append(types.Part(inline_data=blob))
    parts.append(types.Part.from_text(text=prompt))
    return [types.Content(role="user", parts=parts)]


def generate_content_stream(
    prompt: str,
    *,
    model: str = "gemini-2.0-flash",
    system_instruction: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int | None = 2048,
    image_parts: list[tuple[bytes, str]] | None = None,
    **kwargs,
):
    """Стрийминг на отговора токен по токен. image_parts за vision (снимки)."""
    client = get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
        **{k: v for k, v in kwargs.items() if v is not None},
    )
    contents = _build_multimodal_contents(prompt, image_parts) if image_parts else prompt
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=config,
    ):
        if chunk.text:
            yield chunk.text


# ---------- Модели ----------


def list_models(page_size: int = 20):
    """Списък на наличните Gemini модели. Връща списък (не итератор), за да не се затваря клиентът."""
    client = get_client()
    return list(client.models.list(config={"page_size": page_size}))


# ---------- Embeddings ----------


def embed_content(
    contents: str | list[str],
    *,
    model: str = "gemini-embedding-001",
    task_type: str | None = None,
    output_dimensionality: int | None = None,
    **kwargs,
) -> list[list[float]]:
    """
    Генерира текстови embeddings. Връща списък от вектори (всеки е list[float]).
    contents: един текст или списък от текстове.
    task_type: SEMANTIC_SIMILARITY, RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CLASSIFICATION, CLUSTERING, и др.
    output_dimensionality: 128–3072, препоръчително 768, 1536 или 3072.
    """
    client = get_client()
    if isinstance(contents, str):
        contents = [contents]
    config_dict = {}
    if task_type:
        config_dict["task_type"] = task_type
    if output_dimensionality is not None:
        config_dict["output_dimensionality"] = output_dimensionality
    config_dict.update(kwargs)
    config = types.EmbedContentConfig(**config_dict) if config_dict else None
    result = client.models.embed_content(
        model=model,
        contents=contents,
        config=config,
    )
    embeddings = getattr(result, "embeddings", result)
    out = []
    for e in embeddings:
        vals = getattr(e, "values", None) or list(e) if hasattr(e, "__iter__") and not isinstance(e, str) else []
        out.append(vals if isinstance(vals, list) else list(vals))
    return out


# ---------- Токени ----------


def count_tokens(model: str, contents: str | list) -> int:
    """Брой токени за дадено съдържание."""
    client = get_client()
    result = client.models.count_tokens(model=model, contents=contents)
    return result.total_tokens or 0


# ---------- Файлове ----------


def upload_file(file_path: str | Path) -> types.File:
    """Качва файл към Gemini и връща обект File."""
    client = get_client()
    return client.files.upload(file=str(file_path))


def list_files():
    """Списък качени файлове."""
    client = get_client()
    return list(client.files.list())


# ---------- Chat (многократен разговор с история) ----------


def chat_with_history(
    messages: list[dict],
    *,
    model: str = "gemini-2.0-flash",
    system_instruction: str | None = None,
    temperature: float = 0.7,
    max_output_tokens: int = 2048,
) -> str:
    """
    Многократен разговор: messages е списък от {"role": "user"|"model", "parts": [{"text": "..."}]}
    или по-прост формат [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}].
    """
    client = get_client()
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content") or (m.get("parts", [{}])[0].get("text") if m.get("parts") else "")
        if not text:
            continue
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(model=model, contents=contents, config=config)
    return response.text or ""


# ---------- Генериране на изображения ----------


def generate_image(
    prompt: str,
    *,
    model: str = "gemini-2.5-flash-image",
    aspect_ratio: str = "1:1",
) -> bytes:
    """
    Генерира изображение от текстов опис. Връща PNG като bytes.
    aspect_ratio: "1:1", "9:16", "16:9", "4:3", "3:4" и др.
    """
    import io
    client = get_client()
    try:
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        )
    except Exception:
        config = None
    if config:
        response = client.models.generate_content(
            model=model, contents=prompt, config=config
        )
    else:
        response = client.models.generate_content(model=model, contents=prompt)
    parts = getattr(response, "parts", None) or (
        response.candidates[0].content.parts if response.candidates else []
    )
    for part in parts:
        if hasattr(part, "inline_data") and getattr(part, "inline_data", None):
            return part.inline_data.data
        if hasattr(part, "as_image"):
            img = part.as_image()
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
    raise ValueError("Моделът не върна изображение.")


# ---------- Safety ----------


def generate_with_safety(
    prompt: str,
    *,
    model: str = "gemini-2.0-flash",
    harm_block: str = "BLOCK_ONLY_HIGH",  # BLOCK_NONE, BLOCK_LOW_AND_ABOVE, BLOCK_MEDIUM_AND_ABOVE, BLOCK_ONLY_HIGH
):
    """Генериране с зададени safety настройки."""
    client = get_client()
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    safety = [types.SafetySetting(category=c, threshold=harm_block) for c in categories]
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(safety_settings=safety),
    )
    return response.text or ""


# ---------- Function calling (tools) ----------


def generate_with_tools(
    prompt: str,
    tools: list,
    *,
    model: str = "gemini-2.0-flash",
    automatic_function_calling: bool = True,
):
    """
    Генериране с инструменти (function calling).
    tools може да е списък от Python функции с docstring или types.Tool.
    """
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=not automatic_function_calling,
            ),
        ),
    )
    return response.text or ""


# ---------- Тестови скрипт ----------


def main():
    """Тест на основни функции (без API ключ само при list_models ако е зададен)."""
    print("=== Gemini API тест ===\n")

    try:
        models = list_models(page_size=5)
        print("Налични модели (първи 5):")
        for m in models:
            print(f"  - {m.name}")
    except Exception as e:
        print("Грешка при list_models:", e)

    if not GEMINI_API_KEY:
        print("\nЗадай GEMINI_API_KEY в .env за пълни тестове.")
        return

    print("\nГенериране на кратък отговор...")
    try:
        text = generate_content("В един изречение: защо небето е синьо?", max_output_tokens=100)
        print("Отговор:", text)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            print("Грешка: лимит/квота (429). Изчакай малко и опитай отново, или провери квотата в Google AI Studio.")
        else:
            print("Грешка:", e)

    print("\nСтрийминг (първите 200 символа)...")
    try:
        out = []
        for chunk in generate_content_stream("Кажи едно изречение за морето.", max_output_tokens=50):
            out.append(chunk)
        print("".join(out)[:200])
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            print("Грешка: лимит/квота (429). Изчакай малко и опитай отново.")
        else:
            print("Грешка:", e)


if __name__ == "__main__":
    main()
