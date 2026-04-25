"""Parse BNS source documents, chunk them with Spark, write Delta tables.

Runs in two modes:
- Spark mode (Databricks or local Spark): writes Delta tables under the `legal`
  database.
- Pandas fallback (no Spark available): writes Parquet files to the artifacts
  directory so the local Gradio app still works end to end.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from config import CONFIG
from src.chunking import chunk_records, chunk_text


_SECTION_HEADER = re.compile(r"^Section\s+(\d+[A-Z]?)\.\s+(.*)$", re.MULTILINE)


def parse_bns_text(raw_text: str) -> list[dict]:
    """Split the BNS source text into one record per section.

    Expects sections in the form:
        Section 103. Murder.
        <body text...>
    """
    matches = list(_SECTION_HEADER.finditer(raw_text))
    if not matches:
        return [{"section_id": "BNS_FULL", "title": "BNS Full Text", "text": raw_text.strip()}]

    sections: list[dict] = []
    for i, match in enumerate(matches):
        section_id = f"BNS_{match.group(1)}"
        title = match.group(2).strip().rstrip(".")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        full = f"Section {match.group(1)}. {title}.\n\n{body}"
        sections.append({"section_id": section_id, "title": title, "text": full})
    return sections


def parse_bns_pdf(pdf_path: Path) -> list[dict]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    return parse_bns_text(raw)


def load_bns_records(data_dir: Path) -> list[dict]:
    pdf_path = data_dir / "bns.pdf"
    if pdf_path.exists():
        return parse_bns_pdf(pdf_path)

    txt_path = data_dir / "bns_sections.txt"
    if not txt_path.exists():
        raise FileNotFoundError(
            f"Neither {pdf_path} nor {txt_path} exists. Drop the BNS source into data/."
        )
    return parse_bns_text(txt_path.read_text(encoding="utf-8"))


def _try_spark():
    try:
        from pyspark.sql import SparkSession  # noqa: F401

        from delta import configure_spark_with_delta_pip

        builder = (
            __import__("pyspark.sql", fromlist=["SparkSession"])
            .SparkSession.builder.appName("nyaya-bodh-ingestion")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
        return configure_spark_with_delta_pip(builder).getOrCreate()
    except Exception:
        return None


def ingest_with_spark(
    bns_records: list[dict],
    schemes_csv: Path,
    ipc_bns_csv: Path,
    chunk_tokens: int,
    overlap: int,
    database: str = "legal",
) -> None:
    spark = _try_spark()
    if spark is None:
        raise RuntimeError("Spark is not available. Use ingest_with_pandas instead.")

    from pyspark.sql.functions import explode, udf
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    chunk_schema = ArrayType(
        StructType(
            [
                StructField("chunk_id", StringType(), False),
                StructField("chunk_index", StringType(), False),
                StructField("text", StringType(), False),
            ]
        )
    )

    def _chunk_udf(section_id: str, text: str):
        out = []
        for idx, chunk in enumerate(chunk_text(text, chunk_tokens, overlap)):
            out.append(
                {
                    "chunk_id": f"{section_id}::chunk_{idx:04d}",
                    "chunk_index": str(idx),
                    "text": chunk,
                }
            )
        return out

    chunk_udf = udf(_chunk_udf, chunk_schema)

    spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")

    sections_df = spark.createDataFrame(bns_records)
    chunked = (
        sections_df.withColumn("chunks", chunk_udf("section_id", "text"))
        .select("section_id", "title", explode("chunks").alias("c"))
        .selectExpr(
            "section_id",
            "title",
            "c.chunk_id as chunk_id",
            "cast(c.chunk_index as int) as chunk_index",
            "c.text as text",
        )
    )
    chunked.write.mode("overwrite").format("delta").saveAsTable(f"{database}.bns_chunks")

    schemes_df = spark.read.option("header", "true").csv(str(schemes_csv))
    schemes_df.write.mode("overwrite").format("delta").saveAsTable(f"{database}.schemes")

    ipc_df = spark.read.option("header", "true").csv(str(ipc_bns_csv))
    ipc_df.write.mode("overwrite").format("delta").saveAsTable(f"{database}.ipc_bns_map")


def ingest_with_pandas(
    bns_records: list[dict],
    schemes_csv: Path,
    ipc_bns_csv: Path,
    chunk_tokens: int,
    overlap: int,
) -> None:
    """Local fallback. Writes Parquet to the artifacts directory."""
    pairs: Iterable[tuple[str, str]] = (
        (rec["section_id"], rec["text"]) for rec in bns_records
    )
    chunk_rows = chunk_records(pairs, chunk_tokens, overlap)

    title_by_section = {rec["section_id"]: rec["title"] for rec in bns_records}
    for row in chunk_rows:
        row["title"] = title_by_section.get(row["section_id"], "")

    chunks_df = pd.DataFrame(chunk_rows)
    chunks_df.to_parquet(CONFIG.chunks_parquet_path, index=False)

    schemes_df = pd.read_csv(schemes_csv)
    schemes_df.to_parquet(CONFIG.schemes_parquet_path, index=False)

    ipc_df = pd.read_csv(ipc_bns_csv)
    ipc_df.to_parquet(CONFIG.ipc_bns_parquet_path, index=False)


def run_ingestion(data_dir: Path, prefer_spark: bool = False) -> None:
    bns_records = load_bns_records(data_dir)
    schemes_csv = data_dir / "schemes.csv"
    ipc_bns_csv = data_dir / "ipc_bns_mapping.csv"

    if prefer_spark and _try_spark() is not None:
        ingest_with_spark(
            bns_records, schemes_csv, ipc_bns_csv, CONFIG.chunk_tokens, CONFIG.chunk_overlap
        )
        return

    ingest_with_pandas(
        bns_records, schemes_csv, ipc_bns_csv, CONFIG.chunk_tokens, CONFIG.chunk_overlap
    )


if __name__ == "__main__":
    run_ingestion(Path(__file__).resolve().parent.parent / "data", prefer_spark=False)
    print(f"Ingestion complete. Artifacts under {CONFIG.artifacts_dir}")
