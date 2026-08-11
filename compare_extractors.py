"""
compare_extractors.py — PyMuPDF+XY-cut vs Docling, measured on YOUR PDFs.

THE METRIC: EXCESS COLUMN ALTERNATIONS
    Cluster every extracted text item by the x-centre of its bounding box, so
    a page resolves into columns. Walk the items in the order the extractor
    produced them and count how often the column changes. A correct reading
    order changes column exactly (ncols - 1) times: down column one, down
    column two, done. Anything beyond that is the extractor reading ACROSS
    instead of DOWN. Zero is perfect.

    Both extractors expose bounding boxes -- PyMuPDF blocks, Docling
    provenance -- so the same measurement applies to both with no ground
    truth and no per-file annotation.

WHY NOT A TEXT HEURISTIC
    The first version of this script counted "line ends mid-sentence and the
    next starts with a capital". It scored the FIXED output slightly worse
    than the broken one on a file where the reading order provably changed.
    Sentence-shape heuristics measure prose style, not order. Geometry
    measures order. If a metric cannot detect a change you have already
    verified by eye, it is not a weak metric, it is the wrong one.

Usage:
    python tools/compare_extractors.py --pdf "CORPUS.../ILIA_2023.pdf"
    python tools/compare_extractors.py --corpus "CORPUS..." --triage triage.json
    python tools/compare_extractors.py --pdf X.pdf --sweep-gutter
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Item = tuple[int, float, float, str]        # page, x0, x1, text


# ------------------------------------------------------------- the metric

def _column_bounds(items: list[Item], page_width: float,
                   min_gap_frac: float = 0.04) -> list[float]:
    centres = sorted((i[1] + i[2]) / 2 for i in items)
    if not centres:
        return []
    gap = page_width * min_gap_frac
    return [(a + b) / 2 for a, b in zip(centres, centres[1:]) if b - a > gap]


def excess_alternations(items: list[Item], page_width: float) -> tuple[int, int]:
    """(alternations beyond the minimum, number of columns found)."""
    bounds = _column_bounds(items, page_width)
    if not bounds:
        return 0, 1
    sequence = [sum(1 for b in bounds if (i[1] + i[2]) / 2 > b) for i in items]
    changes = sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    return max(0, changes - len(bounds)), len(bounds) + 1


def score(items_by_page: dict[int, list[Item]], widths: dict[int, float]
          ) -> tuple[int, int, float, int]:
    """(excess alternations, pages scored, mean columns, characters kept)."""
    total = pages = columns = characters = 0
    for page, items in items_by_page.items():
        characters += sum(len(i[3]) for i in items)
        if len(items) < 3:
            continue
        excess, ncols = excess_alternations(items, widths.get(page, 612.0))
        total += excess
        columns += ncols
        pages += 1
    return total, pages, (columns / pages if pages else 0.0), characters


# ------------------------------------------------------------- extractors

# THE UNIT OF MEASUREMENT IS ONE LOGICAL PAGE, FOR BOTH EXTRACTORS.
#
# Getting this wrong produced a Docling score of 1.01 on ILIA_2023 against
# PyMuPDF's 0.26, and none of the gap was real:
#   * Docling saw the SPLIT file, so its x ran 0..624 on both halves, while
#     PyMuPDF saw the original, x running 0..1247;
#   * split pages were remapped back onto one original page number, so
#     left-half and right-half items shared a bucket and their x-ranges
#     overlapped -- two different columns binned as one;
#   * the column-gap threshold used the original 1247pt width against 624pt
#     coordinates, demanding a 50pt gap where the real gutters are 10pt.
# Every fold crossing then counted as a column change. Comparing extractors
# means comparing extractors, not comparing coordinate systems.


def _to_logical_pages(items: list[Item], page: int, width: float, two_up: bool
                      ) -> tuple[dict[int, list[Item]], dict[int, float]]:
    """Split one physical page into logical pages, normalising x on the right."""
    if not two_up:
        return {page: items}, {page: width}
    mid = width / 2
    left = [i for i in items if (i[1] + i[2]) / 2 < mid]
    right = [(i[0], i[1] - mid, i[2] - mid, i[3]) for i in items
             if (i[1] + i[2]) / 2 >= mid]
    return ({page * 2 - 1: left, page * 2: right},
            {page * 2 - 1: mid, page * 2: mid})


def pymupdf_items(path: Path, reorder: bool = True
                  ) -> tuple[dict[int, list[Item]], dict[int, float]]:
    import pymupdf

    from layout import _page_blocks, document_is_two_up, order_blocks

    doc = pymupdf.open(path)
    two_up = document_is_two_up(doc)
    out: dict[int, list[Item]] = {}
    widths: dict[int, float] = {}

    for n, page in enumerate(doc, 1):
        blocks = _page_blocks(page)
        if reorder:
            blocks = order_blocks(blocks, page.rect.width, page.rect.height,
                                  two_up=two_up, trust_input_order=True)
        else:
            blocks = [b for b in page.get_text("blocks")
                      if b[6] == 0 and b[4].strip()]
        items = [(n, b[0], b[2], b[4]) for b in blocks]
        spread = two_up and page.rect.width > page.rect.height * 1.25
        pages, page_widths = _to_logical_pages(items, n, page.rect.width, spread)
        out.update(pages)
        widths.update(page_widths)

    doc.close()
    return out, widths


def docling_items(path: Path, device: str = "cuda"
                  ) -> tuple[dict[int, list[Item]], dict[int, float]]:
    import pymupdf

    from docling_extract import DoclingEngine
    from layout import split_spreads

    target = path
    split = split_spreads(path)
    if split:
        target = Path(split[0])

    # Widths of the file DOCLING ACTUALLY SEES. After a split that is the
    # logical page, which is the same unit pymupdf_items reports.
    handle = pymupdf.open(target)
    widths = {n + 1: page.rect.width for n, page in enumerate(handle)}
    handle.close()

    engine = DoclingEngine(device=device, do_ocr=False)
    document = engine.converter.convert(str(target)).document

    out: dict[int, list[Item]] = {}
    for item, _level in document.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        provenance = getattr(item, "prov", None)
        if not text or not provenance:
            continue
        box = provenance[0].bbox
        # Keep the SPLIT page numbering: one bucket per logical page.
        page = int(provenance[0].page_no)
        out.setdefault(page, []).append((page, float(box.l), float(box.r), text))

    if split:
        Path(split[0]).unlink(missing_ok=True)
    return out, widths


# ------------------------------------------------------------- driver

def compare(path: Path, device: str) -> dict:
    row: dict = {"file": path.name}

    for label, fn in (("raw", lambda p: pymupdf_items(p, reorder=False)),
                      ("pymupdf", lambda p: pymupdf_items(p, reorder=True)),
                      ("docling", lambda p: docling_items(p, device))):
        started = time.time()
        try:
            items, widths = fn(path)
            total, pages, cols, characters = score(items, widths)
            row[label] = {"excess": total, "pages": pages,
                          "per_page": round(total / max(pages, 1), 2),
                          "cols": round(cols, 1), "chars": characters,
                          "seconds": round(time.time() - started, 1)}
        except Exception as exc:
            row[label] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
    return row


def show(rows: list[dict]) -> None:
    # `keep` is the share of the raw text layer an extractor actually emits.
    #
    # WITHOUT IT THE ORDER METRIC IS GAMEABLE. An extractor scores a perfect 0
    # excess alternations by emitting one paragraph per page, and Docling
    # genuinely discards regions it labels table, picture, chart, header or
    # footer. A tool that reads 60% of a document in perfect order is worse
    # for retrieval than one that reads all of it imperfectly: missing text
    # cannot be retrieved at any rank.
    print(f"\n{'file':32s} {'raw':>7s} {'pymupdf':>16s} {'docling':>16s} "
          f"{'doc s':>6s}")
    print(f"{'':32s} {'ord':>7s} {'ord':>7s} {'keep':>8s} "
          f"{'ord':>7s} {'keep':>8s}")
    print("-" * 84)
    totals: dict[str, list[float]] = {"raw": [], "pymupdf": [], "docling": []}

    for row in rows:
        base = row.get("raw", {}).get("chars") or 0
        cells = []
        for key in ("raw", "pymupdf", "docling"):
            cell = row.get(key, {})
            if "error" in cell:
                cells.append(("    err", "     -"))
                continue
            keep = (cell.get("chars", 0) / base) if base else 0.0
            cells.append((f"{cell['per_page']:7.2f}",
                          f"{100*keep:7.0f}%" + ("!" if keep < 0.85 else " ")))
            totals[key].append(cell["per_page"])
            row.setdefault("keep", {})[key] = keep
        print(f"{row['file'][:32]:32s} {cells[0][0]} "
              f"{cells[1][0]} {cells[1][1]} {cells[2][0]} {cells[2][1]}"
              f"{row.get('docling', {}).get('seconds', 0):6}")

    means = {k: (sum(v) / len(v) if v else None) for k, v in totals.items()}
    print("-" * 84)
    print(f"{'MEAN excess alternations / page':32s} "
          + " ".join(f"{means[k]:7.2f}" if means[k] is not None else "    n/a"
                     for k in ("raw", "pymupdf", "docling")))

    thin = [r["file"] for r in rows
            if r.get("keep", {}).get("docling", 1.0) < 0.85]
    if thin:
        print(f"\n  ! Docling emitted under 85% of the raw text on "
              f"{len(thin)} file(s): {thin[:3]}\n"
              f"    Its order score is NOT comparable there -- text it never "
              f"emitted cannot be\n    mis-ordered. Check what _DROP is "
              f"discarding before routing those files to it.")

    print("\n  0 = reads each column top to bottom exactly once. Higher = "
          "reads across\n  columns. Compare the columns against each other, "
          "not against zero.")
    if means["pymupdf"] is not None and means["docling"] is not None:
        gap = means["pymupdf"] - means["docling"]
        verdict = ("within noise -- keep --docling off, it costs a GPU pass "
                   "and a\n  TableFormer paragraph in the informe for nothing"
                   if abs(gap) < 0.15 else
                   f"Docling is {gap:.2f}/page better -- worth it for these files"
                   if gap > 0 else
                   f"PyMuPDF is {-gap:.2f}/page better -- Docling reads these worse")
        print(f"\n  Difference {gap:+.2f}: {verdict}.")
    if means["raw"] is not None and means["pymupdf"] is not None \
            and means["raw"] < means["pymupdf"] - 0.15:
        print("\n  NOTE: raw content-stream order beats the XY-cut here. That "
              "means the\n  gutters are narrower than GUTTER_FRACTION and the "
              "cut never fires --\n  try --sweep-gutter before blaming the "
              "algorithm.")


def sweep_gutter(path: Path) -> None:
    import layout
    original = layout.GUTTER_FRACTION
    print(f"\n{path.name}: gutter sweep (excess alternations per page)")
    for frac in (0.035, 0.025, 0.018, 0.012, 0.008, 0.005):
        layout.GUTTER_FRACTION = frac
        items, widths = pymupdf_items(path, reorder=True)
        total, pages, cols, _chars = score(items, widths)
        print(f"  GUTTER_FRACTION={frac:<6} {total/max(pages,1):6.2f}   "
              f"({cols:.1f} columns detected on average)")
    layout.GUTTER_FRACTION = original
    print("  Lower is better, but a fraction that is too small starts reading "
          "table\n  cells as columns. Check a rendered page before adopting one.")



def decide(corpus: Path, triage: Path, out: Path, device: str,
           sample_per_folder: int, margin: float,
           report: Path | None = None) -> None:
    """
    Produce the routing list for `build_index.py --docling list`.

    WHY A LIST AND NOT A RULE
        Docling is not uniformly better. Measured: it won by 9.73/page on the
        AI Index economy chapter and LOST by 4.72 on SWF Counterspace and by
        0.48 on ILIA_2023. `--docling complex` sends the losers to it too. A
        rule that is right 60% of the time is worse than a measurement, and
        the measurement is a one-off cost.

    TWO GRANULARITIES, AND THE COST IS THE WHOLE DIFFERENCE
        sample_per_folder > 0  measures a few files per folder and applies the
            verdict to the folder. Cheap, and it exploits the fact that one
            observatory publishes on one template: all four AI Index chapters
            won, both SWF files lost.
        sample_per_folder = 0  measures EVERY complex file. Strictly better
            decisions -- and it costs a full Docling pass, roughly what the
            build itself will cost, so you pay for Docling twice. Worth it
            only if the folder mode reports many SPLIT folders.

    A folder that splits is left on the cheaper path rather than guessed at.
    """
    profiles = json.loads(triage.read_text(encoding="utf-8"))["files"]
    complex_files = [f for f in profiles
                     if f.get("two_up")
                     or f.get("multicolumn_pages", 0)
                     / max(f.get("sampled_pages", 1), 1) >= 0.3]

    by_folder: dict[str, list[dict]] = {}
    for f in complex_files:
        by_folder.setdefault(f.get("folder") or "?", []).append(f)

    per_file = sample_per_folder <= 0
    print(f"{len(complex_files)} complex-layout files in {len(by_folder)} "
          f"folders; mode = {'PER FILE' if per_file else f'{sample_per_folder} per folder'}")
    if per_file:
        print("  every file is measured with Docling, so this pass costs "
              "about as much as\n  the build will. Budget for it.\n")
    else:
        print()

    # Fail fast on a corpus-root mismatch. triage.json stores paths relative
    # to the root triage.py was given; if --corpus here is a different root,
    # every path misses and every folder reports "no measurement" -- which
    # looks like Docling failing rather than a wrong argument.
    probe = [corpus / f["path"] for f in complex_files[:20]]
    if probe and not any(p.exists() for p in probe):
        raise SystemExit(
            f"None of the first {len(probe)} triage paths exist under "
            f"{corpus.resolve()}.\n"
            f"  e.g. {probe[0]}\n"
            f"  triage.json stores paths RELATIVE to the corpus root it was "
            f"built with.\n  Pass the same --corpus you passed to triage.py.")

    chosen: list[str] = []
    measurements: list[dict] = []
    missing = 0
    errors: list[str] = []

    for folder, members in sorted(by_folder.items()):
        sample = members if per_file else members[:sample_per_folder]
        results = []
        for f in sample:
            path = corpus / f["path"]
            if not path.exists():
                missing += 1
                continue
            row = compare(path, device)
            try:
                best_py = min(row["raw"]["per_page"], row["pymupdf"]["per_page"])
                delta = best_py - row["docling"]["per_page"]
            except (KeyError, TypeError):
                delta = None
                # Say WHICH extractor failed and why. "no measurement" on its
                # own is indistinguishable from a missing file, a broken
                # import and a Docling crash, and those need different fixes.
                for label in ("raw", "pymupdf", "docling"):
                    problem = row.get(label, {}).get("error")
                    if problem:
                        errors.append(f"{label}: {problem}  [{path.name}]")
            results.append((f["path"], delta, row))
            measurements.append({"path": f["path"], "folder": folder,
                                 "delta": delta, "detail": row})

        usable = [d for _p, d, _r in results if d is not None]
        if not usable:
            print(f"  {folder[:52]:52s} no measurement -> pymupdf")
            continue

        if per_file:
            winners = [p for p, d, _r in results if d is not None and d > margin]
            chosen += winners
            print(f"  {folder[:52]:52s} {len(winners)}/{len(results)} files "
                  f"-> docling")
            continue

        wins = sum(1 for d in usable if d > margin)
        losses = sum(1 for d in usable if d < -margin / 2)
        mean = sum(usable) / len(usable)
        if wins and not losses:
            chosen += [f["path"] for f in members]
            verdict = f"DOCLING  (+{mean:.2f}/page, {len(members)} files)"
        elif losses and not wins:
            verdict = f"pymupdf  ({mean:+.2f}/page)"
        else:
            verdict = f"SPLIT {wins}w/{losses}l -> pymupdf (measure per file)"
        print(f"  {folder[:52]:52s} {verdict}")

    if missing or errors:
        print(f"\n  {missing} sampled files not found on disk, "
              f"{len(errors)} measurement failures.")
        seen: set[str] = set()
        for problem in errors:
            key = problem.split("[")[0]
            if key not in seen:
                seen.add(key)
                print(f"    {problem}")
        if errors and all("docling" in e.split(":")[0] for e in errors):
            print("    Only Docling failed -- the routing question cannot be "
                  "answered until it runs.")
        elif errors:
            print("    A PyMuPDF-side failure usually means layout.py is out "
                  "of date:\n    order_blocks() and document_is_two_up() must "
                  "both be importable from it.")

    out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(f"\nWrote {out}: {len(chosen)} files routed to Docling "
          f"({100*len(chosen)/max(len(complex_files),1):.0f}% of the complex "
          f"set, {100*len(chosen)/max(len(profiles),1):.0f}% of all PDFs)")

    if report:
        report.write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        print(f"Wrote {report}: per-file measurements, for the informe")

    print(f"\nEverything not listed stays on its current path:")
    print(f"  digital + text layer   -> PyMuPDF + order_blocks   (Phase A)")
    print(f"  no usable text layer   -> RapidOCR + order_blocks  (Phase B)")
    print(f"  listed above           -> Docling layout model     (Phase C)")
    print(f"\n  build_index.py --docling list --docling-list {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, action="append", default=[])
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--triage", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sweep-gutter", action="store_true")
    parser.add_argument("--decide", type=Path, metavar="OUT.txt",
                        help="measure per folder and write the Docling routing "
                             "list for build_index.py --docling list")
    parser.add_argument("--sample-per-folder", type=int, default=3,
                        help="files measured per folder; 0 measures EVERY "
                             "complex file and decides per file (costs a full "
                             "Docling pass)")
    parser.add_argument("--decide-report", type=Path, default=None,
                        help="write per-file measurements as JSON")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="excess-alternations-per-page advantage Docling "
                             "must show before a folder is routed to it")
    args = parser.parse_args()

    if args.decide:
        if not (args.corpus and args.triage):
            raise SystemExit("--decide needs --corpus and --triage")
        decide(args.corpus, args.triage, args.decide, args.device,
               args.sample_per_folder, args.margin, args.decide_report)
        return

    paths = list(args.pdf)
    if args.triage and args.corpus:
        triage = json.loads(args.triage.read_text(encoding="utf-8"))
        picked = [f for f in triage["files"]
                  if f.get("two_up")
                  or f.get("multicolumn_pages", 0)
                  / max(f.get("sampled_pages", 1), 1) >= 0.3]

        # ROUND-ROBIN ACROSS FOLDERS, not top-N by column count.
        #
        # Sorting by multicolumn_pages picks the most extreme files, and those
        # cluster: the first ten were nine files scoring a perfect 12/12 drawn
        # from five observatories, while ILIA_2023 -- a two-up spread, clearly
        # complex at 8/12 -- sat at rank 94 and was never measured. A sample
        # that misses whole publishers cannot decide a per-publisher question.
        by_folder: dict[str, list[dict]] = {}
        for f in sorted(picked, key=lambda f: -f.get("multicolumn_pages", 0)):
            by_folder.setdefault(f.get("folder") or "?", []).append(f)

        ordered, depth = [], 0
        while len(ordered) < args.limit and depth < 50:
            added = False
            for members in by_folder.values():
                if depth < len(members):
                    ordered.append(members[depth])
                    added = True
                    if len(ordered) >= args.limit:
                        break
            if not added:
                break
            depth += 1

        paths += [args.corpus / f["path"] for f in ordered]
        print(f"triage: {len(picked)} complex-layout PDFs in {len(by_folder)} "
              f"folders; testing {len(ordered)}, one per folder before "
              f"a second from any")
        if not any("two_up" in f for f in triage["files"]):
            print("  WARNING: this triage.json predates the two_up field. "
                  "Spreads whose halves are\n           single-column will "
                  "not be in the complex set at all. Re-run triage.py.")

    paths = [p for p in paths if p.exists()][:args.limit]
    if not paths:
        raise SystemExit("No PDFs. Pass --pdf, or --corpus with --triage.")

    if args.sweep_gutter:
        for path in paths:
            sweep_gutter(path)
        return

    show([compare(p, args.device) for p in paths])


if __name__ == "__main__":
    main()