"""
OpenAI API – пълна функционалност.
- Chat Completions (с всички параметри: temperature, top_p, frequency_penalty, presence_penalty, stop, и др.)
- Completions (legacy)
- Embeddings
- Images (DALL-E – генериране, вариации, редакция)
- Audio (Whisper – транскрипция, превод)
- Moderations
- Models (списък, get)
- Files (upload, list, retrieve, delete) – за fine-tuning / assistants
- Assistants API (create, threads, messages, runs)
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

from openai import OpenAI


def get_client() -> OpenAI:
    """Връща OpenAI клиент с API ключ от config."""
    if not OPENAI_API_KEY:
        raise ValueError(
            "Задай OPENAI_API_KEY в .env файл в тази папка или в променлива на средата."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


# ---------- Chat Completions ----------


def chat_completion(
    messages: list[dict],
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int | None = 1024,
    top_p: float | None = 1.0,
    frequency_penalty: float | None = 0.0,
    presence_penalty: float | None = 0.0,
    stop: str | list[str] | None = None,
    stream: bool = False,
    response_format: dict | None = None,
    seed: int | None = None,
    **kwargs,
):
    """
    Chat completion с всички опции.
    messages: [{"role": "user"|"system"|"assistant", "content": "..."}]
    """
    client = get_client()
    params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "stream": stream,
    }
    if stop is not None:
        params["stop"] = stop
    if response_format is not None:
        params["response_format"] = response_format
    if seed is not None:
        params["seed"] = seed
    params.update({k: v for k, v in kwargs.items() if v is not None})
    return client.chat.completions.create(**params)


def chat_simple(prompt: str, *, model: str = "gpt-4o-mini", system: str | None = None, **kwargs) -> str:
    """Прост еднократен отговор."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    r = chat_completion(messages, model=model, **kwargs)
    return r.choices[0].message.content or ""


def chat_stream(prompt: str, *, model: str = "gpt-4o-mini", **kwargs):
    """Стрийминг на chat completion."""
    r = chat_completion(
        [{"role": "user", "content": prompt}],
        model=model,
        stream=True,
        **kwargs,
    )
    for chunk in r:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ---------- Embeddings ----------


def embeddings_create(
    input_text: str | list[str],
    *,
    model: str = "text-embedding-3-small",
    dimensions: int | None = None,
    **kwargs,
):
    """Създава embeddings за текст/списък от текстове."""
    client = get_client()
    params = {"model": model, "input": input_text}
    if dimensions is not None:
        params["dimensions"] = dimensions
    params.update(kwargs)
    return client.embeddings.create(**params)


# ---------- Images ----------


def images_generate(
    prompt: str,
    *,
    model: str = "dall-e-3",
    size: str = "1024x1024",
    quality: str = "standard",
    n: int = 1,
    response_format: str = "url",
    style: str | None = None,
    **kwargs,
):
    """Генерира изображения с DALL-E."""
    client = get_client()
    params = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": n,
        "response_format": response_format,
    }
    if style and model == "dall-e-3":
        params["style"] = style  # "vivid" | "natural"
    params.update(kwargs)
    return client.images.generate(**params)


def image_edit(image_path: str | Path, prompt: str, *, mask_path: str | Path | None = None, **kwargs):
    """Редактиране на изображение (DALL-E 2)."""
    client = get_client()
    with open(image_path, "rb") as f:
        image = f.read()
    params = {"model": "dall-e-2", "image": image, "prompt": prompt, **kwargs}
    if mask_path:
        with open(mask_path, "rb") as f:
            params["mask"] = f.read()
    return client.images.edit(**params)


def image_variation(image_path: str | Path, *, n: int = 1, size: str = "1024x1024", **kwargs):
    """Вариации на изображение (DALL-E 2)."""
    client = get_client()
    with open(image_path, "rb") as f:
        image = f.read()
    return client.images.create_variation(image=image, n=n, size=size, **kwargs)


# ---------- Audio (Whisper) ----------


def audio_transcribe(
    file_path: str | Path,
    *,
    model: str = "whisper-1",
    language: str | None = None,
    response_format: str = "json",
    temperature: float | None = None,
    **kwargs,
):
    """Транскрипция на аудио файл."""
    client = get_client()
    with open(file_path, "rb") as f:
        return client.audio.transcriptions.create(
            file=f,
            model=model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if v is not None},
        )


def audio_translate(file_path: str | Path, *, model: str = "whisper-1", **kwargs):
    """Превод на аудио на английски."""
    client = get_client()
    with open(file_path, "rb") as f:
        return client.audio.translations.create(model=model, file=f, **kwargs)


# ---------- Moderations ----------


def moderations_create(input_text: str | list[str], *, model: str | None = None, **kwargs):
    """Проверка за съдържание (модерация)."""
    client = get_client()
    params = {"input": input_text}
    if model:
        params["model"] = model
    params.update(kwargs)
    return client.moderations.create(**params)


# ---------- Models ----------


def models_list():
    """Списък на налични модели."""
    client = get_client()
    return list(client.models.list())


def models_retrieve(model_id: str):
    """Детайли за един модел."""
    client = get_client()
    return client.models.retrieve(model_id)


# ---------- Files ----------


def files_upload(file_path: str | Path, *, purpose: str = "assistants"):
    """Качва файл (за assistants, fine-tuning и др.)."""
    client = get_client()
    with open(file_path, "rb") as f:
        return client.files.create(file=f, purpose=purpose)


def files_list():
    """Списък качени файлове."""
    client = get_client()
    return list(client.files.list())


def files_retrieve(file_id: str):
    """Метаданни за файл."""
    client = get_client()
    return client.files.retrieve(file_id)


def files_delete(file_id: str):
    """Изтриване на файл."""
    client = get_client()
    return client.files.delete(file_id)


# ---------- Assistants ----------


def assistants_create(
    name: str,
    *,
    model: str = "gpt-4o-mini",
    instructions: str | None = None,
    tools: list | None = None,
    file_ids: list[str] | None = None,
    **kwargs,
):
    """Създава асистент."""
    client = get_client()
    params = {"name": name, "model": model}
    if instructions:
        params["instructions"] = instructions
    if tools:
        params["tools"] = tools
    if file_ids:
        params["file_ids"] = file_ids
    params.update(kwargs)
    return client.beta.assistants.create(**params)


def assistants_list(*, limit: int = 20):
    """Списък асистенти."""
    client = get_client()
    return list(client.beta.assistants.list(limit=limit))


def thread_create(*, metadata: dict | None = None):
    """Създава разговорна нишка."""
    client = get_client()
    return client.beta.threads.create(metadata=metadata or {})


def thread_message_create(thread_id: str, content: str, *, role: str = "user", **kwargs):
    """Добавя съобщение в нишка."""
    client = get_client()
    return client.beta.threads.messages.create(thread_id=thread_id, role=role, content=content, **kwargs)


def run_create(thread_id: str, assistant_id: str, **kwargs):
    """Стартира run на асистент върху нишка."""
    client = get_client()
    return client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id, **kwargs)


def run_retrieve(thread_id: str, run_id: str):
    """Статус на run."""
    client = get_client()
    return client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)


# ---------- Тест ----------


def main():
    print("=== OpenAI API тест ===\n")
    if not OPENAI_API_KEY:
        print("Задай OPENAI_API_KEY в .env за тестове.")
        return
    try:
        models = models_list()
        print("Налични модели (първи 5):")
        for m in models[:5]:
            print(f"  - {m.id}")
    except Exception as e:
        print("Грешка при models_list:", e)
    print("\nChat (кратък отговор):")
    try:
        text = chat_simple("В един изречение: какво е Python?", max_tokens=80)
        print("Отговор:", text)
    except Exception as e:
        print("Грешка:", e)
    print("\nEmbeddings (първи вектор, първи 5 стойности):")
    try:
        emb = embeddings_create("Hello world", model="text-embedding-3-small")
        vec = emb.data[0].embedding[:5]
        print(vec)
    except Exception as e:
        print("Грешка:", e)
    print("\nModerations:")
    try:
        mod = moderations_create("This is a normal sentence.")
        print("Flagged:", mod.results[0].flagged)
    except Exception as e:
        print("Грешка:", e)


if __name__ == "__main__":
    main()
