# AI APIs – Gemini, OpenAI, Anthropic

Три отделни проекта за тестване и настройка на API-тата на **Google Gemini**, **OpenAI** и **Anthropic** (Claude). Кодът е на Python с пълна поддръжка на възможностите на всяко API.

## Структура

```
ai_apis/
├── gemini/          # Google Gemini API
├── openai/          # OpenAI API (Chat, DALL-E, Whisper, Assistants, …)
├── anthropic/       # Anthropic Claude API
└── README.md
```

## Настройка на API ключове

Във всяка папка (`gemini`, `openai`, `anthropic`) има файл **`.env.example`**. Копирай го като **`.env`** и сложи съответния API ключ:

| Проект    | Файл             | Променлива          | Откъде да вземеш ключ                                   |
| --------- | ---------------- | ------------------- | ------------------------------------------------------- |
| Gemini    | `gemini/.env`    | `GEMINI_API_KEY`    | [Google AI Studio](https://aistudio.google.com/apikey)  |
| OpenAI    | `openai/.env`    | `OPENAI_API_KEY`    | [OpenAI Platform](https://platform.openai.com/api-keys) |
| Anthropic | `anthropic/.env` | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/)     |

Пример за `gemini/.env`:

```
GEMINI_API_KEY=тук_твоят_ключ
```

След това можеш да тестваш и да настройваш всички опции за съответното API.

---

## 1. Gemini (`gemini/`)

- **Инсталация:** `pip install -r gemini/requirements.txt`
- **Тест:** `python gemini/client.py`

**Възможности в `client.py`:**

- `generate_content()` – текст с опции: `temperature`, `top_p`, `top_k`, `max_output_tokens`, `stop_sequences`, `system_instruction`, `safety_settings`, `response_mime_type`
- `generate_content_stream()` – стрийминг
- `list_models()` – списък модели
- `count_tokens()` – брой токени
- `upload_file()`, `list_files()` – файлове
- `chat_with_history()` – разговор с история
- `generate_with_safety()` – safety настройки
- `generate_with_tools()` – function calling / tools

---

## 2. OpenAI (`openai/`)

- **Инсталация:** `pip install -r openai/requirements.txt`
- **Тест:** `python openai/client.py`

**Възможности в `client.py`:**

- **Chat:** `chat_completion()`, `chat_simple()`, `chat_stream()` – с `temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`, `stop`, `response_format`, `seed`
- **Embeddings:** `embeddings_create()` – с опция `dimensions`
- **Images:** `images_generate()` (DALL-E), `image_edit()`, `image_variation()`
- **Audio:** `audio_transcribe()`, `audio_translate()` (Whisper)
- **Moderations:** `moderations_create()`
- **Models:** `models_list()`, `models_retrieve()`
- **Files:** `files_upload()`, `files_list()`, `files_retrieve()`, `files_delete()`
- **Assistants:** `assistants_create()`, `assistants_list()`, `thread_create()`, `thread_message_create()`, `run_create()`, `run_retrieve()`

---

## 3. Anthropic (`anthropic/`)

- **Инсталация:** `pip install -r anthropic/requirements.txt`
- **Тест:** `python anthropic/client.py`

**Възможности в `client.py`:**

- **Messages:** `message_create()`, `message_simple()`, `message_stream()` – с `model`, `max_tokens`, `system`, `temperature`, `top_p`, `top_k`, `stop_sequences`, `metadata`, `tools`
- `count_tokens()` – брой токени за съобщения
- `message_vision()` – текст + изображение (vision)

---

## Общо между проектите

- Всеки проект има свой **`config.py`**, който чете ключа от **`.env`** (или от променливи на средата).
- В **`client.py`** са събрани основните функции за съответното API; можеш да ги извикваш от свои скриптове или да разшириш с още опции.
- За да тестваш всички възможности за дадено API, задай ключа в съответния `.env` и стартирай `python <проект>/client.py` – в края на всеки `client.py` има функция `main()` за бърз тест.

Ако искаш да добавим още конкретни функции или опции за някое от трите API-та, кажи за кой проект и какво да поддържа (например конкретни параметри или endpoints).
