"""Smoke tests for components that do not require the OpenAI API."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import chunk_records, chunk_text
from src.ingestion import parse_bns_text


def test_chunk_text_respects_token_budget():
    text = "A" * 10000
    chunks = chunk_text(text, chunk_tokens=128, overlap=16)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_chunk_text_handles_empty():
    assert chunk_text("", 128, 16) == []


def test_chunk_records_emits_ids():
    records = [("BNS_1", "hello world. " * 100)]
    rows = chunk_records(records, chunk_tokens=64, overlap=8)
    assert rows
    assert rows[0]["section_id"] == "BNS_1"
    assert rows[0]["chunk_id"].startswith("BNS_1::chunk_")
    assert rows[0]["chunk_index"] == 0


def test_parse_bns_text_extracts_sections():
    sample = (
        "Section 303. Theft.\n"
        "Whoever commits theft shall be punished.\n"
        "Section 318. Cheating.\n"
        "Whoever cheats shall be punished.\n"
    )
    sections = parse_bns_text(sample)
    ids = [s["section_id"] for s in sections]
    assert "BNS_303" in ids
    assert "BNS_318" in ids


def test_parse_bns_text_handles_no_headers():
    sections = parse_bns_text("just some prose without section headers")
    assert len(sections) == 1
    assert sections[0]["section_id"] == "BNS_FULL"


def test_real_bns_file_parses():
    path = ROOT / "data" / "bns_sections.txt"
    sections = parse_bns_text(path.read_text(encoding="utf-8"))
    ids = {s["section_id"] for s in sections}
    for required in ["BNS_103", "BNS_303", "BNS_318", "BNS_351", "BNS_111", "BNS_113"]:
        assert required in ids, f"Missing {required} from parsed sections"
