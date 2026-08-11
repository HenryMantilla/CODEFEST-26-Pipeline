"""
inventory.py — Load the organizers' official file inventory.

Indice_Datos_Codefest.xlsx, sheet "Inventario de Archivos", holds 1826 rows of
  Fenomeno | Observatorio | Codigo | DOC_ID | Nombre estandarizado | Carpeta | Tipo
with DOC_IDs like F1-AIINDEX-001. Using it removes all guesswork about which
phenomenon a document belongs to, and gives doc_ids that match the organizers'
own naming.

Usage:
    python inventory.py --xlsx "CORPUS.../Indice_Datos_Codefest.xlsx" \
        --corpus "CORPUS CODEFEST AD ASTRA 2026" --out inventory.json

Requires: pip install pandas openpyxl
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

SHEET = "Inventario de Archivos"


def _normalize(name: str) -> str:
    """Collapse a file name so inventory and disk spellings compare equal."""
    name = unicodedata.normalize("NFC", str(name)).strip().lower()
    name = re.sub(r"[_\s]+", "-", name)
    return name


def load_inventory(xlsx: Path, corpus_root: Path) -> dict[str, dict]:
    """
    Map relative path on disk -> {"doc_id", "phenomenon", "observatory"}.

    Matching is done on the normalized file name rather than the recorded
    folder, because the inventory stores standardized names while the disk
    carries a `<CODE>_` prefix and hyphenated casing.
    """
    import pandas as pd

    df = pd.read_excel(xlsx, sheet_name=SHEET, dtype=str).fillna("")
    columns = {c.lower().strip(): c for c in df.columns}

    def column(*candidates: str) -> str | None:
        for candidate in candidates:
            for key, original in columns.items():
                if candidate in key:
                    return original
        return None

    col_doc = column("doc_id")
    col_phen = column("fenómeno", "fenomeno")
    col_name = column("nombre estandarizado", "nombre")
    col_obs = column("observatorio")

    if not (col_doc and col_name):
        raise SystemExit(f"Unexpected columns in {SHEET}: {list(df.columns)}")

    by_name: dict[str, dict] = {}
    for _, row in df.iterrows():
        phenomenon = 0
        match = re.search(r"[123]", str(row.get(col_phen, "")))
        if match:
            phenomenon = int(match.group())
        by_name[_normalize(row[col_name])] = {
            "doc_id": row[col_doc].strip(),
            "phenomenon": phenomenon,
            "observatory": row.get(col_obs, ""),
        }

    mapping: dict[str, dict] = {}
    unmatched: list[str] = []

    for path in corpus_root.rglob("*"):
        if not path.is_file():
            continue
        relative = str(path.relative_to(corpus_root))
        name = _normalize(path.name)

        record = by_name.get(name)
        if record is None:
            # Disk names carry a "<CODE>_" prefix the inventory lacks
            stripped = name.split("_", 1)[-1] if "_" in name else name
            record = by_name.get(stripped)
        if record is None:
            unmatched.append(relative)
            continue
        mapping[relative] = record

    print(f"matched {len(mapping)} files, {len(unmatched)} unmatched")
    for item in unmatched[:10]:
        print("  unmatched:", item)
    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("inventory.json"))
    args = parser.parse_args()

    mapping = load_inventory(args.xlsx, args.corpus)
    args.out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"Wrote {args.out}")