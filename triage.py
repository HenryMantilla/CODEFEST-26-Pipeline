"""
triage.py — Classify every PDF in the corpus before extraction.

Answers, per folder: which files have a real text layer, which are scanned and
need OCR, which are slide decks whose text layer is present but incomplete, and
which pages are multi-column.

Routing decision per file:
    digital   -> PyMuPDF text layer, no OCR
    scanned   -> full-page rasterize + Tesseract (per-image OCR finds nothing:
                 the whole page is one image)
    sparse    -> text layer exists but is thin (slides, poster-style pages);
                 extract text AND OCR the page images, then deduplicate
    empty     -> no text, no images: genuinely blank

Usage:
    python triage.py --corpus "CORPUS CODEFEST AD ASTRA 2026" --out triage.json

Requires: pip install pymupdf
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import fitz

from layout import detect_columns, document_is_two_up

# Thresholds, in characters of extracted text per page.
SCANNED_MAX_CHARS = 100      # below this there is effectively no text layer
SPARSE_MAX_CHARS = 400       # slide decks land here: some text, not the whole page


def long_path(path: os.PathLike | str) -> str:
    r"""
    Windows refuses paths over 260 characters unless prefixed with \\?\.
    321 files in this corpus exceed 255 characters, concentrated in
    Atlantic_Council, CEEEP, INPE and CSIS_Aerospace. os.walk lists them
    happily and then open() fails with WinError 3, which looks like a
    corrupt-file problem but is not.
    """
    resolved = os.path.abspath(str(path))
    if os.name == "nt" and len(resolved) > 240 and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


@dataclass
class PdfProfile:
    path: str
    folder: str
    pages: int
    chars: int
    chars_per_page: float
    images_per_page: float
    multicolumn_pages: int
    sampled_pages: int
    two_up: bool
    klass: str                  # digital | scanned | sparse | empty | error
    ocr_needed: bool
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def profile_pdf(path: Path, corpus_root: Path, sample: int = 12) -> PdfProfile:
    """
    Profile one PDF. Pages are sampled evenly rather than read in order: many
    reports open with a scanned cover and continue with digital text, and
    reading only the first pages would misclassify the whole file.
    """
    folder = str(path.parent.relative_to(corpus_root))
    try:
        doc = fitz.open(long_path(path))
    except Exception as exc:
        return PdfProfile(str(path), folder, 0, 0, 0.0, 0.0, 0, 0, False,
                          "error", False, note=str(exc)[:120])

    n_pages = doc.page_count
    if n_pages == 0:
        doc.close()
        return PdfProfile(str(path), folder, 0, 0, 0.0, 0.0, 0, 0, False,
                          "empty", False, note="no pages")

    step = max(1, n_pages // sample)
    indices = list(range(0, n_pages, step))[:sample]

    chars = images = multicolumn = 0
    for i in indices:
        try:
            page = doc[i]
        except Exception:
            continue
        chars += len(page.get_text("text").strip())
        images += len(page.get_images(full=True))
        try:
            if detect_columns(page) > 1:
                multicolumn += 1
        except Exception:
            pass

    sampled = len(indices)

    # Two logical pages per sheet. Recorded here, once, so build_index can
    # ROUTE on it: these files need either the geometric split or Docling, and
    # plain extraction reads straight across the fold, splicing the first line
    # of the left page onto the first line of the right one. That corruption
    # is invisible in the output -- it is fluent Spanish, just not the
    # sentence anyone wrote.
    try:
        two_up = document_is_two_up(doc)
    except Exception:
        two_up = False

    doc.close()

    per_page = chars / max(sampled, 1)
    images_per_page = images / max(sampled, 1)

    if per_page < SCANNED_MAX_CHARS:
        if images_per_page >= 0.5:
            klass, ocr, note = "scanned", True, "no text layer, page images present"
        else:
            klass, ocr, note = "empty", False, "no text and no images"
    elif per_page < SPARSE_MAX_CHARS:
        klass, ocr = "sparse", images_per_page >= 0.5
        note = "thin text layer, likely slides or poster layout"
    else:
        klass, ocr, note = "digital", False, ""

    return PdfProfile(
        path=str(path.relative_to(corpus_root)), folder=folder, pages=n_pages,
        chars=chars, chars_per_page=round(per_page, 1),
        images_per_page=round(images_per_page, 2),
        multicolumn_pages=multicolumn, sampled_pages=sampled, two_up=two_up,
        klass=klass, ocr_needed=ocr, note=note,
    )


def scan_corpus(corpus_root: Path, sample: int = 12) -> list[PdfProfile]:
    pdfs = sorted(p for p in corpus_root.rglob("*.pdf") if p.is_file())
    profiles = []
    for i, path in enumerate(pdfs, 1):
        profiles.append(profile_pdf(path, corpus_root, sample))
        if i % 25 == 0:
            print(f"  ... {i}/{len(pdfs)}")
    return profiles


def folder_report(profiles: list[PdfProfile]) -> list[dict]:
    """Aggregate per folder — this is the table that decides where OCR runs."""
    buckets: dict[str, list[PdfProfile]] = defaultdict(list)
    for p in profiles:
        buckets[p.folder].append(p)

    rows = []
    for folder, items in buckets.items():
        counts = defaultdict(int)
        for item in items:
            counts[item.klass] += 1
        ocr_files = [i for i in items if i.ocr_needed]
        rows.append({
            "folder": folder,
            "pdfs": len(items),
            "digital": counts["digital"],
            "scanned": counts["scanned"],
            "sparse": counts["sparse"],
            "empty": counts["empty"] + counts["error"],
            "ocr_files": len(ocr_files),
            "ocr_pages": sum(i.pages for i in ocr_files),
            "multicolumn_files": sum(1 for i in items if i.multicolumn_pages > 0),
            "two_up_files": sum(1 for i in items if i.two_up),
        })
    rows.sort(key=lambda r: -r["ocr_pages"])
    return rows


def print_report(rows: list[dict]) -> None:
    header = f"{'folder':44} {'pdfs':>5} {'digi':>5} {'scan':>5} {'sprs':>5} {'ocrPg':>6} {'multiCol':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        flag = "  <-- OCR" if r["ocr_pages"] else ""
        print(f"{r['folder'][:44]:44} {r['pdfs']:5} {r['digital']:5} "
              f"{r['scanned']:5} {r['sparse']:5} {r['ocr_pages']:6} "
              f"{r['multicolumn_files']:9}{flag}")

    total_ocr = sum(r["ocr_pages"] for r in rows)
    print("-" * len(header))
    print(f"{'TOTAL':44} {sum(r['pdfs'] for r in rows):5} "
          f"{sum(r['digital'] for r in rows):5} {sum(r['scanned'] for r in rows):5} "
          f"{sum(r['sparse'] for r in rows):5} {total_ocr:6}")


# Tesseract language packs by dominant folder language. Stacking every language
# into one call slows OCR down and lowers accuracy, so route per folder.
# Tesseract only. RapidOCR (the default backend) does NOT need this: its
# `latin` recognition head is one script model covering Spanish, English AND
# Portuguese at once, so no per-folder routing is required and a document that
# mixes languages -- a Spanish report quoting an English treaty -- is read
# correctly without anyone declaring that in advance. Chinese and Russian are
# NOT Latin script and would need a different head; the only such files here
# are translations of documents that also exist in Spanish, and build_index
# drops those before extraction.
OCR_LANGS = {
    "Alertas_Tempranas": "spa",
    "MAPP_OEA": "spa",
    "CEEEP": "spa",
    "ILIA_Latam": "spa",
    "RutaN_GEIAL": "spa",
    "CENIA": "spa+eng",
    "INPE": "por",
    "CEOBS": "eng",
    "SIPRI": "eng+spa",
    "UNOOSA": "eng+spa",
    "Amazon_Underworld": "spa+por",
    "RESDAL": "eng+spa",
}
DEFAULT_OCR_LANG = "eng+spa"


def ocr_language_for(folder: str) -> str:
    for key, lang in OCR_LANGS.items():
        if key in folder:
            return lang
    return DEFAULT_OCR_LANG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("triage.json"))
    parser.add_argument("--sample", type=int, default=12,
                        help="pages sampled per PDF")
    args = parser.parse_args()

    print(f"Scanning {args.corpus} ...")
    profiles = scan_corpus(args.corpus, args.sample)
    rows = folder_report(profiles)
    print_report(rows)

    payload = {
        "folders": rows,
        "files": [p.to_dict() for p in profiles],
        "ocr_queue": [
            {"path": p.path, "pages": p.pages, "lang": ocr_language_for(p.folder)}
            for p in profiles if p.ocr_needed
        ],
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\nWrote {args.out}  "
          f"({len(payload['ocr_queue'])} files queued for OCR)")


if __name__ == "__main__":
    main()