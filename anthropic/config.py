"""Конфигурация за Anthropic API - зарежда API ключ от .env или променливи на средата."""
import os
from pathlib import Path

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
