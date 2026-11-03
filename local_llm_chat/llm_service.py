"""Модуль работы с локальным сервером Bionic через OpenAI-совместимый API.

Инкапсулирует все сетевые вызовы (перечисление моделей, генерация ответа)
за пределами UI-логики Streamlit, чтобы ``app.py`` занимался только состоянием
и отображением.
"""

from __future__ import annotations

from typing import Generator

import openai

import config


class BionicConnectionError(RuntimeError):
    """Ошибка подключения/доступности сервера Bionic.

    Attributes:
        message: Человекочитаемое описание проблемы для показа в UI.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def create_client() -> openai.OpenAI:
    """Создает экземпляр OpenAI-клиента, указывающего на локальный Bionic.

    Возвращает:
        openai.OpenAI: Настроенный клиент (base_url и api_key берутся из ``config``).
    """
    return openai.OpenAI(
        base_url=config.BIONIC_BASE_URL,
        api_key=config.DUMMY_API_KEY,  # Заглушка: Bionic не требует авторизации.
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )


def get_available_models(client: openai.OpenAI | None = None) -> list[str]:
    """Запрашивает у Bionic список доступных локальных моделей.

    Args:
        client: Необязательный переиспользуемый экземпляр клиента; если не передан,
            создается новый через :func:`create_client`.

    Returns:
        list[str]: Список ID моделей (``[model.id for model in response.data]``).

    Raises:
        BionicConnectionError: Если сервер недоступен или вернул ошибку API.
    """
    if client is None:
        client = create_client()

    try:
        response = client.models.list()
    except openai.APIConnectionError as exc:
        raise BionicConnectionError(config.ERROR_BIONIC_UNAVAILABLE) from exc
    except openai.APIStatusError as exc:
        raise BionicConnectionError(
            f"Bionic ответил с ошибкой {exc.status_code}: {exc.message}"
        ) from exc

    return [model.id for model in response.data]


def generate_stream(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    client: openai.OpenAI | None = None,
) -> Generator[str, None, None]:
    """Генерирует ответ ассистента потоком (SSE-чанки от Bionic).

    Args:
        messages: История диалога в формате OpenAI Chat Completions API.
        model: ID модели, которую нужно использовать для генерации.
        temperature: Температура сэмплирования (0.0–1.0).
        client: Необязательный переиспользуемый экземпляр клиента.

    Yields:
        str: Не пустые текстовые фрагменты ответа (по одному за итерацию).

    Raises:
        BionicConnectionError: Если сервер недоступен или вернул ошибку API.
    """
    if client is None:
        client = create_client()

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]  # dict -> ChatCompletionMessageParam
            temperature=temperature,
            stream=True,
        )
    except openai.APIConnectionError as exc:
        raise BionicConnectionError(config.ERROR_BIONIC_UNAVAILABLE) from exc
    except openai.APIStatusError as exc:
        raise BionicConnectionError(
            f"Bionic ответил с ошибкой {exc.status_code}: {exc.message}"
        ) from exc

    for chunk in stream:
        # У финальных чанков (и чанков со служебными полями) choices может не быть,
        # а delta.content может отсутствовать (None) — такие чанки пропускаем.
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content
