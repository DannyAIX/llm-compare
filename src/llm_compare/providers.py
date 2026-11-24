"""Modelos disponibles para comparar."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def get_models() -> dict[str, BaseChatModel]:
    """Devuelve un diccionario {nombre visible: modelo}."""
    return {
        "Gemini Flash (Google)": ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest", temperature=0
        ),
        "GPT-OSS 120B (OpenAI vía Groq)": ChatGroq(
            model="openai/gpt-oss-120b", temperature=0
        ),
        "Qwen 3.8 27B (Alibaba vía Groq)": ChatGroq(
            model="qwen/qwen3.8-27b", temperature=0, reasoning_format="hidden"
        ),
    }
