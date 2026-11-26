"""Modelos disponibles para comparar y modelo juez."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

COMMON = {"temperature": 0, "timeout": 60, "max_retries": 1}


def get_models() -> dict[str, BaseChatModel]:
    """Devuelve un diccionario {nombre visible: modelo}."""
    return {
        "Gemini Flash Lite (Google)": ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest", **COMMON
        ),
        "GPT-OSS 120B (OpenAI vía Groq)": ChatGroq(
            model="openai/gpt-oss-120b", **COMMON
        ),
        "Qwen 3.8 27B (Alibaba vía Groq)": ChatGroq(
            model="qwen/qwen3.8-27b", reasoning_format="hidden", **COMMON
        ),
    }


def get_judge() -> BaseChatModel:
    """Modelo que califica las respuestas."""
    return ChatGroq(model="openai/gpt-oss-120b", **COMMON)
