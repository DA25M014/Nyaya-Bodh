"""Gradio UI for Nyaya-Bodh.

Two tabs:
1. Ask a Question: free-text legal Q&A grounded in BNS, with cited sources and matching schemes.
2. IPC vs BNS: side-by-side comparison of legacy IPC sections to BNS replacements.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr

from src.ipc_bns_compare import compare_by_ipc
from src.pipeline import NyayaBodhPipeline


PIPELINE: NyayaBodhPipeline | None = None


def get_pipeline() -> NyayaBodhPipeline:
    global PIPELINE
    if PIPELINE is None:
        PIPELINE = NyayaBodhPipeline()
    return PIPELINE


def _format_citations(response) -> str:
    if not response.citations:
        return "_No BNS sections retrieved._"
    lines = []
    for c in response.citations:
        snippet = c.text[:280].replace("\n", " ")
        lines.append(f"- **{c.section_id} ({c.title})** — score {c.score:.3f}\n  > {snippet}...")
    return "\n".join(lines)


def _format_schemes(response) -> str:
    if not response.schemes:
        return "_No closely matching schemes found._"
    lines = []
    for s in response.schemes:
        lines.append(
            f"- **{s.name}** ({s.ministry})\n"
            f"  - Eligibility: {s.eligibility}\n"
            f"  - {s.description}\n"
            f"  - [More info]({s.url})"
        )
    return "\n".join(lines)


def ask_question(question: str):
    question = (question or "").strip()
    if not question:
        return "Please type a question.", "", ""
    pipeline = get_pipeline()
    response = pipeline.answer(question)
    answer_md = (
        f"{response.answer}\n\n"
        f"_Latency: {response.latency_ms} ms_"
    )
    return answer_md, _format_citations(response), _format_schemes(response)


def compare_ipc(ipc_section: str):
    ipc_section = (ipc_section or "").strip()
    if not ipc_section:
        return "Enter an IPC section number to compare."
    result = compare_by_ipc(ipc_section)
    if result is None:
        return f"No mapping found for IPC section {ipc_section}. Try 302, 304, 378, 379, 415, 420, or 503."
    return (
        f"### IPC Section {result.ipc_section}: {result.ipc_title}\n\n"
        f"**Maps to** BNS Section {result.bns_section}: {result.bns_title}\n\n"
        f"**What changed:** {result.summary_of_change}"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Nyaya-Bodh") as demo:
        gr.Markdown(
            "# Nyaya-Bodh\n"
            "A multilingual legal assistant grounded in the Bharatiya Nyaya Sanhita 2023. "
            "Ask in English or Hindi. Answers are cited from the BNS text."
        )

        with gr.Tab("Ask a Question"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_box = gr.Textbox(
                        label="Your question",
                        placeholder="e.g. What is the punishment for theft under the new BNS?",
                        lines=3,
                    )
                    submit = gr.Button("Ask", variant="primary")
                    examples = gr.Examples(
                        examples=[
                            ["What is the punishment for theft under the new BNS?"],
                            ["BNS के तहत हत्या की सजा क्या है?"],
                            ["What does the BNS say about criminal intimidation?"],
                            ["What is organised crime under BNS?"],
                            ["BNS में चोरी की सजा क्या है?"],
                        ],
                        inputs=[question_box],
                    )
                with gr.Column(scale=3):
                    answer_box = gr.Markdown(label="Answer")

            with gr.Accordion("Cited BNS sections", open=True):
                citations_box = gr.Markdown()
            with gr.Accordion("Related government schemes", open=False):
                schemes_box = gr.Markdown()

            submit.click(ask_question, inputs=[question_box], outputs=[answer_box, citations_box, schemes_box])
            question_box.submit(ask_question, inputs=[question_box], outputs=[answer_box, citations_box, schemes_box])

        with gr.Tab("IPC vs BNS"):
            gr.Markdown(
                "Enter an IPC section number to see its BNS replacement and a summary of what changed."
            )
            with gr.Row():
                ipc_box = gr.Textbox(label="IPC section number", placeholder="e.g. 420", scale=1)
                compare_btn = gr.Button("Compare", variant="primary", scale=0)
            comparison_box = gr.Markdown()
            gr.Examples(
                examples=[["302"], ["304"], ["378"], ["379"], ["415"], ["420"], ["503"]],
                inputs=[ipc_box],
            )
            compare_btn.click(compare_ipc, inputs=[ipc_box], outputs=[comparison_box])
            ipc_box.submit(compare_ipc, inputs=[ipc_box], outputs=[comparison_box])

        gr.Markdown(
            "---\n"
            "Disclaimer: Nyaya-Bodh is a research prototype for the Bharat Bricks Hacks. "
            "It is general legal information, not legal advice."
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
