"""Константы и настройки приложения локального AI-чата.

Все настраиваемые значения вынесены сюда, чтобы избежать хардкода
в UI-логике (``app.py``) и в сервисном модуле (``llm_service.py``).

URL сервера Bionic может быть переопределен через переменную
окружения ``BIONIC_BASE_URL`` (см. README.md).
"""

from __future__ import annotations

import os

# --- Сервер Bionic -----------------------------------------------------------
# OpenAI-совместимый endpoint локального сервера Bionic.
DEFAULT_BIONIC_BASE_URL: str = "http://localhost:1235/v1"
BIONIC_BASE_URL: str = os.getenv("BIONIC_BASE_URL", DEFAULT_BIONIC_BASE_URL)

# Bionic не требует авторизации, но клиент openai обязан получить непустой ключ.
DUMMY_API_KEY: str = "not-needed"

# Таймаут подключения к серверу (секунды), чтобы UI не "зависал" навсегда.
REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("BIONIC_REQUEST_TIMEOUT", "30"))

# --- Параметры модели по умолчанию -------------------------------------------
DEFAULT_TEMPERATURE: float = 0.7
MIN_TEMPERATURE: float = 0.0
MAX_TEMPERATURE: float = 1.0
TEMPERATURE_STEP: float = 0.1

# Значение, используемое при выборе модели, если список пуст или сервер недоступен.
NO_MODEL_PLACEHOLDER: str = "(нет доступных моделей)"

# --- Промпты -------------------------------------------------------------------
DEFAULT_SYSTEM_PROMPT: str = "Ты полезный AI-ассистент."

WELCOME_MESSAGE: str = (
    "Привет! Я локальная модель, развернутая на сервере **Bionic**. "
    "Задайте мне вопрос — ответы передаются потоком (streaming)."
)

ERROR_BIONIC_UNAVAILABLE: str = (
    "Не удалось подключиться к серверу Bionic. "
    "Убедитесь, что сервер запущен и доступен по указанному URL."
)

# --- Ключи st.session_state ------------------------------------------------------
MESSAGES_KEY: str = "messages"
SYSTEM_PROMPT_KEY: str = "system_prompt"
MODEL_KEY: str = "model"
TEMPERATURE_KEY: str = "temperature"
