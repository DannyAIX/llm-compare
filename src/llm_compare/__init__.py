"""llm-compare: compara respuestas de varios LLMs y las califica con un juez."""

import sys

from llm_compare.compare import ask_all
from llm_compare.judge import judge
from llm_compare.providers import get_judge, get_models


def main() -> None:
    question = (
        " ".join(sys.argv[1:])
        or "Explica qué es un índice en una base de datos en 3 líneas."
    )
    print(f"Pregunta: {question}")

    answers = []
    for r in ask_all(question, get_models()):
        answers.append(r)
        print(f"\n=== {r.model} ({r.seconds:.1f}s) ===")
        print(f"ERROR: {r.error}" if r.error else r.answer)

    if sum(1 for a in answers if not a.error) < 2:
        print("\nSe necesitan al menos 2 respuestas válidas para juzgar.")
        return

    print("\n... el juez está evaluando", flush=True)
    verdict, names = judge(question, answers, get_judge())

    print("\n=== VEREDICTO DEL JUEZ ===")
    ranked = sorted(
        verdict.scores,
        key=lambda s: s.accuracy + s.completeness + s.clarity,
        reverse=True,
    )
    for s in ranked:
        total = s.accuracy + s.completeness + s.clarity
        print(
            f"{names.get(s.label, s.label)}: precisión {s.accuracy} | "
            f"completitud {s.completeness} | claridad {s.clarity} | total {total}/30"
        )
        print(f"   Problemas: {s.issues}")

    rationale = verdict.rationale
    for letter, name in names.items():
        rationale = rationale.replace(f"respuesta {letter}", name)
        rationale = rationale.replace(f"Answer {letter}", name)

    print(f"\nGanador: {names.get(verdict.winner, verdict.winner)}")
    print(f"Por qué: {rationale}")
