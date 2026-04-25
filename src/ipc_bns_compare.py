"""Side-by-side IPC to BNS comparison driven by the mapping table."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import CONFIG


@dataclass
class ClauseComparison:
    ipc_section: str
    ipc_title: str
    bns_section: str
    bns_title: str
    summary_of_change: str


_CACHE: pd.DataFrame | None = None


def _load() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        _CACHE = pd.read_parquet(CONFIG.ipc_bns_parquet_path)
        _CACHE["ipc_section"] = _CACHE["ipc_section"].astype(str).str.strip()
    return _CACHE


def compare_by_ipc(ipc_section: str) -> ClauseComparison | None:
    df = _load()
    needle = str(ipc_section).strip().lstrip("0") or "0"
    matches = df[df["ipc_section"].str.lstrip("0").str.casefold() == needle.casefold()]
    if matches.empty:
        return None
    row = matches.iloc[0]
    return ClauseComparison(
        ipc_section=str(row["ipc_section"]),
        ipc_title=str(row["ipc_title"]),
        bns_section=str(row["bns_section"]),
        bns_title=str(row["bns_title"]),
        summary_of_change=str(row["summary_of_change"]),
    )


def list_all() -> list[ClauseComparison]:
    df = _load()
    return [
        ClauseComparison(
            ipc_section=str(r["ipc_section"]),
            ipc_title=str(r["ipc_title"]),
            bns_section=str(r["bns_section"]),
            bns_title=str(r["bns_title"]),
            summary_of_change=str(r["summary_of_change"]),
        )
        for _, r in df.iterrows()
    ]
