"""Modelo juez: califica respuestas anónimas con salida estructurada."""

import random
import string

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from llm_compare.compare import ModelAnswer


class Score(BaseModel):
    label: str = Field(description="Letter of the answer being scored, e.g. 'A'")
    accuracy: int = Field(ge=1, le=10, description="Factual correctness")
    completeness: int = Field(ge=1, le=10, description="Covers what was asked")
    clarity: int = Field(ge=1, le=10, description="Clear and well organized")
    issues: str = Field(
        description="Factual errors or weaknesses found; 'none' if none"
    )


class Verdict(BaseModel):
    scores: list[Score]
    winner: str = Field(description="Letter of the best answer")
    rationale: str = Field(description="One or two sentences explaining the choice")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict, impartial evaluator. You will see a question and several "
            "anonymous answers labeled with letters. Check facts carefully and penalize "
            "errors and exaggerations more than style. Do not favor longer answers. "
            "Write 'issues' and 'rationale' in the same language as the question.",
        ),
        ("human", "Question:\n{question}\n\nAnswers:\n{answers}"),
    ]
)


def judge(
    question: str, answers: list[ModelAnswer], judge_llm: BaseChatModel
) -> tuple[Verdict, dict[str, str]]:
    """Devuelve el veredicto y el mapa {letra: nombre del modelo}."""
    valid = [a for a in answers if not a.error]
    random.shuffle(valid)  # evita el sesgo de posición
    labeled = dict(zip(string.ascii_uppercase, valid))
    text = "\n\n".join(f"--- Answer {k} ---\n{a.answer}" for k, a in labeled.items())

    chain = PROMPT | judge_llm.with_structured_output(Verdict)
    verdict = chain.invoke({"question": question, "answers": text})
    return verdict, {k: a.model for k, a in labeled.items()}
