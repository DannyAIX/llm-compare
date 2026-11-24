"""llm-compare: compara respuestas de varios LLMs."""

import sys

from llm_compare.compare import ask_all
from llm_compare.providers import get_models


def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        or "Explica qué es un índice en una base de datos en 3 líneas."
    )
    print(f"Pregunta: {question}")
    for r in ask_all(question, get_models()):
        print(f"\n=== {r.model} ({r.seconds:.1f}s) ===")
        print(f"ERROR: {r.error}" if r.error else r.answer)
