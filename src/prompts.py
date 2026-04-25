"""Prompt templates for the legal Q&A pipeline."""
from __future__ import annotations

from src.vector_store import RetrievedChunk


SYSTEM_PROMPT = """You are Nyaya-Bodh, a legal assistant grounded in the Bharatiya Nyaya Sanhita (BNS) 2023.

Hard rules:
1. Use ONLY the BNS context provided below. If the answer is not present, reply that the provided context does not cover this question and suggest consulting a qualified legal professional.
2. Always cite the BNS sections you used, formatted like [BNS Section 303].
3. Reply in the same language as the user's question. If the question is in Hindi (Devanagari or Romanised), reply in Hindi.
4. Be concise: 4 to 8 sentences. End with a one-line plain-language summary.
5. Do NOT show your reasoning, planning, or thinking process. Do NOT use phrases like "Okay,", "Let me check", "First, I need to". Output ONLY the final answer directly.
6. Never invent section numbers, punishments, or fines that are not in the context.
7. Add this disclaimer at the end, in the user's language:
   English: "This is general information, not legal advice."
   Hindi: "Yeh saamaanya jaankari hai, kanooni salaah nahi."
"""


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return f"Question: {question}\n\nContext: (no relevant BNS sections retrieved)"

    context_blocks = []
    for chunk in retrieved:
        header = f"[{chunk.section_id} | {chunk.title}]"
        context_blocks.append(f"{header}\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)

    return (
        f"Context (BNS extracts):\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in the same language as the question, citing BNS sections used."
    )


JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring a legal answer.

Rules:
1. Score CORRECTNESS on a 0 to 5 scale where 5 is fully correct, 0 is wrong or hallucinated.
2. Score CITATION on a 0 to 1 scale: 1 if the cited BNS section matches the expected one, 0 otherwise.
3. Reply ONLY with a JSON object: {"correctness": int, "citation": int, "reason": "short explanation"}.
"""


def build_judge_prompt(question: str, expected_section: str, answer: str) -> str:
    return (
        f"Question: {question}\n"
        f"Expected BNS section: {expected_section}\n"
        f"Candidate answer:\n{answer}\n"
    )
