"""Hace la misma pregunta a varios modelos y mide el tiempo de cada uno."""

import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class ModelAnswer:
    model: str
    answer: str
    seconds: float
    error: str | None = None


def ask_all(question: str, models: dict[str, BaseChatModel]) -> list[ModelAnswer]:
    results = []
    for name, llm in models.items():
        start = time.perf_counter()
        try:
            response = llm.invoke(question)
            results.append(
                ModelAnswer(name, response.text, time.perf_counter() - start)
            )
        except Exception as e:  # un modelo caído no debe tumbar a los demás
            results.append(ModelAnswer(name, "", time.perf_counter() - start, str(e)))
    return results
