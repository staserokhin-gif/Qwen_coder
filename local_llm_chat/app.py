"""Локальный AI-чат на Streamlit поверх сервера Bionic (OpenAI-совместимый API).

Точка входа приложения: отвечает за инициализацию st.session_state,
боковую панель с настройками, отображение истории диалога и стриминг ответов.
Сетевые вызовы инкапсулированы в ``llm_service.py``, константы — в ``config.py``.
"""

from __future__ import annotations

from typing import Generator

import streamlit as st

import config
import llm_service


def init_session_state() -> None:
    """Инициализирует ключи st.session_state значениями по умолчанию.

    Вызывается один раз в самом начале работы скрипта; существующие значения
    (например, накопленная история сообщений) не перезаписываются.
    """
    if config.MESSAGES_KEY not in st.session_state:
        # Формат элементов: {"role": "system"|"user"|"assistant", "content": "..."}
        st.session_state[config.MESSAGES_KEY] = []
    if config.SYSTEM_PROMPT_KEY not in st.session_state:
        st.session_state[config.SYSTEM_PROMPT_KEY] = config.DEFAULT_SYSTEM_PROMPT


def load_models() -> tuple[list[str], str | None]:
    """Динамически подгружает список моделей с сервера Bionic.

    Возвращает:
        tuple[list[str], str | None]: Список ID моделей и текст ошибки
        (``None``, если подключение прошло успешно).
    """
    try:
        return llm_service.get_available_models(), None
    except llm_service.BionicConnectionError as exc:
        return [], exc.message


def build_prompt_messages(system_prompt: str) -> list[dict[str, str]]:
    """Формирует payload для Chat Completions из сохраненной истории.

    Активный системный промпт всегда подставляется первым, поэтому в историю
    он не дублируется (см. требование 5: messages хранит user/assistant реплики).

    Args:
        system_prompt: Текущий текст системного промпта из боковой панели.

    Returns:
        list[dict[str, str]]: Список сообщений для OpenAI-совместимого API.
    """
    prompt_messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    prompt_messages.extend(
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state[config.MESSAGES_KEY]
    )
    return prompt_messages


def render_sidebar(available_models: list[str]) -> tuple[str, float, str]:
    """Отображает боковую панель с настройками чата.

    Args:
        available_models: Список ID моделей, полученных от Bionic.

    Returns:
        tuple[str, float, str]: Выбранная модель, температура и системный промпт.
    """
    with st.sidebar:
        st.header("Настройки")

        model_options: list[str] = available_models or [config.NO_MODEL_PLACEHOLDER]
        selected_model: str = st.selectbox(
            "Модель",
            options=model_options,
            index=0,
            key=config.MODEL_KEY,
        )

        temperature: float = st.slider(
            "Температура",
            min_value=config.MIN_TEMPERATURE,
            max_value=config.MAX_TEMPERATURE,
            step=config.TEMPERATURE_STEP,
            value=config.DEFAULT_TEMPERATURE,
            key=config.TEMPERATURE_KEY,
        )

        st.text_area(
            "Системный промпт",
            height=120,
            key=config.SYSTEM_PROMPT_KEY,
        )

        if st.button("Очистить историю", use_container_width=True):
            st.session_state[config.MESSAGES_KEY] = []
            st.rerun()

    return selected_model, temperature, st.session_state[config.SYSTEM_PROMPT_KEY]


def stream_assistant_reply(
    token_generator: Generator[str, None, None],
) -> str:
    """Стримит токены ответа в пустое сообщение ассистента через ``st.write_stream``.

    Args:
        token_generator: Генератор текстовых фрагментов от
            :func:`llm_service.generate_stream`.

    Returns:
        str: Собранный текст ответа; пустая строка, если генерация не началась.

    Note:
        При обрыве соединения на середине стрима частичный ответ сохраняется
        в истории, а UI показывает ошибку вместо падения (требование 7.3).
    """
    collected: str = ""
    with st.chat_message("assistant"):
        try:
            collected = st.write_stream(token_generator) or ""
        except Exception as exc:  # noqa: BLE001 - не роняем UI при обрыве стриминга
            st.error(f"Генерация прервана: {exc}")
    return collected


def main() -> None:
    """Главная функция приложения: конфигурация страницы, состояние, цикл чата."""
    st.set_page_config(page_title="Локальный AI-чат (Bionic)", page_icon="🤖")
    st.title("🤖 Локальный AI-чат на Bionic")

    init_session_state()

    models, connection_error = load_models()
    selected_model, temperature, system_prompt = render_sidebar(models)

    if connection_error:
        st.error(config.ERROR_BIONIC_UNAVAILABLE)
        st.info(f"URL сервера: `{config.BIONIC_BASE_URL}`")
        return  # Блокируем чат, пока Bionic недоступен.

    # История воспроизводится при каждом реруне целиком (Streamlit перестраивает страницу).
    for message in st.session_state[config.MESSAGES_KEY]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not st.session_state[config.MESSAGES_KEY]:
        with st.chat_message("assistant"):
            st.markdown(config.WELCOME_MESSAGE)

    user_input: str | None = st.chat_input("Введите сообщение...")

    if user_input is None:
        return

    stripped_input: str = user_input.strip()
    if not stripped_input:
        return  # Игнорируем пустой ввод.

    st.session_state[config.MESSAGES_KEY].append(
        {"role": "user", "content": stripped_input}
    )
    with st.chat_message("user"):
        st.markdown(stripped_input)

    prompt_messages = build_prompt_messages(system_prompt)
    token_generator = llm_service.generate_stream(
        messages=prompt_messages,
        model=selected_model,
        temperature=temperature,
    )

    reply = stream_assistant_reply(token_generator)
    if reply:
        st.session_state[config.MESSAGES_KEY].append(
            {"role": "assistant", "content": reply}
        )


if __name__ == "__main__":
    main()
