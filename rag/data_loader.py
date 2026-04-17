"""
data_loader.py  -  PART A: Data Engineering & Preparation
CS4241 - Introduction to Artificial Intelligence
Student Name:  Daniel Kingsley Bright Amusah
Index Number:  10012300036

Loads and cleans both data sources:
  1. 2025 Ghana Budget Statement PDF (252 pages)
  2. Ghana Election Results CSV

No external RAG frameworks used - all extraction is manual.
"""

import re
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PDF LOADER
# ─────────────────────────────────────────────

def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from every page of the Budget PDF.
    Returns a list of page dicts: {page_num, text, source}.

    Cleaning applied:
      - Strip headers / footers (short repeated lines at edges)
      - Collapse excess whitespace
      - Remove page-number-only lines
      - Drop blank pages
    """
    pages = []
    pdf_path = Path(pdf_path)

    logger.info(f"Loading PDF: {pdf_path.name}")

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            cleaned = _clean_pdf_page(raw, page_num)
            if cleaned.strip():
                pages.append({
                    "page_num": page_num,
                    "text": cleaned,
                    "source": f"{pdf_path.name} | page {page_num}",
                    "doc_type": "budget_pdf",
                })

    logger.info(f"Extracted {len(pages)} non-empty pages from PDF")
    return pages


def _clean_pdf_page(text: str, page_num: int) -> str:
    """Clean a single PDF page's raw text."""
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip bare page numbers
        if re.fullmatch(r"\d+", line):
            continue

        # Skip short repeated headers / footers (< 5 chars)
        if len(line) < 5:
            continue

        # Collapse internal multi-spaces
        line = re.sub(r" {2,}", " ", line)

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# ─────────────────────────────────────────────
# CSV LOADER
# ─────────────────────────────────────────────

def load_election_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load and clean the Ghana Election Results CSV.
    Each row is converted to a natural-language sentence chunk
    so it can be embedded like regular text.

    Cleaning applied:
      - Strip BOM / non-breaking spaces from values
      - Normalise region names (remove NBSP)
      - Convert percentage strings to floats
      - Drop rows with missing critical fields
    """
    csv_path = Path(csv_path)
    logger.info(f"Loading CSV: {csv_path.name}")

    df = pd.read_csv(str(csv_path), encoding="utf-8-sig")

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    # Clean string columns - remove non-breaking spaces
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    # Convert Votes to int (remove commas if present)
    df["Votes"] = (
        df["Votes"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

    # Clean percentage column
    df["Votes(%)"] = (
        df["Votes(%)"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
    )
    df["Votes(%)"] = pd.to_numeric(df["Votes(%)"], errors="coerce")

    # Drop rows missing critical fields
    critical = ["Year", "Candidate", "Party", "Votes"]
    df.dropna(subset=critical, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Convert each row to a natural-language document
    records = []
    for _, row in df.iterrows():
        text = _row_to_text(row)
        records.append({
            "text": text,
            "source": f"Ghana_Election_Result.csv | {row['Year']} | {row.get('New Region', row.get('Old Region', ''))}",
            "doc_type": "election_csv",
            "year": str(row["Year"]),
            "region": str(row.get("New Region", row.get("Old Region", ""))),
            "candidate": str(row["Candidate"]),
            "party": str(row["Party"]),
            "votes": int(row["Votes"]) if pd.notna(row["Votes"]) else 0,
        })

    logger.info(f"Loaded {len(records)} cleaned election records")
    return records


def _row_to_text(row: pd.Series) -> str:
    """Convert a single election row into a readable sentence."""
    pct = f"{row['Votes(%)']:.2f}%" if pd.notna(row["Votes(%)"]) else "unknown %"
    votes = f"{int(row['Votes']):,}" if pd.notna(row["Votes"]) else "unknown"

    return (
        f"In the {row['Year']} Ghana presidential election, "
        f"{row['Candidate']} of the {row['Party']} party received "
        f"{votes} votes ({pct}) in the "
        f"{row.get('New Region', row.get('Old Region', 'unknown'))} region "
        f"(formerly {row.get('Old Region', 'N/A')})."
    )


# ─────────────────────────────────────────────
# COMBINED LOADER
# ─────────────────────────────────────────────

def load_all_documents(data_dir: str) -> List[Dict[str, Any]]:
    """Load both data sources and return a unified document list."""
    data_dir = Path(data_dir)

    pdf_docs = load_pdf(str(data_dir / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"))
    csv_docs = load_election_csv(str(data_dir / "Ghana_Election_Result.csv"))

    all_docs = pdf_docs + csv_docs
    logger.info(f"Total raw documents loaded: {len(all_docs)}")
    return all_docs
