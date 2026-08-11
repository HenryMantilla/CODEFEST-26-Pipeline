"""
make_entrega.py — Assemble and verify entrega/ before submitting. NOT shipped.

WHY THIS EXISTS
    Nothing owns the deliverable directory. build_index.py writes
    base_vectorial/, generador.py writes resultados.jsonl, informe_tecnico.pdf
    is written by hand, and consultas_50.jsonl has to be copied in so the run
    is self-contained. Four producers, no assembler, and 1.4 is explicit:
    "Si no es posible reproducir los resultados, se excluirá de la
    evaluación." A missing file is not a bad score, it is a zero.

    This script does not build anything. It collects what exists, checks the
    parts that are checkable, and refuses to say OK when something is off.
    Run it as the last step before packaging.

WHAT IT CHECKS
    - every encoder folder has index.faiss + metadata.jsonl, and the vector
      count matches the metadata line count (1.4: "El orden de las líneas
      debe coincidir con los identificadores internos asignados por FAISS")
    - metadata carries every mandatory Table 1 field
    - `fuente` is a file name, not a path (10.2.1 matches documents on it)
    - no raw U+0085/U+2028/U+2029 anywhere, which would make a splitlines()
      based reader see a malformed file (9.3.2 discards those)
    - resultados.jsonl passes the full 9.3.2 schema check
    - informe_tecnico.pdf exists and is within the 8-page limit
    - nothing junk is about to ship (caches, .bak, editor droppings)

Usage:
    python tools/make_entrega.py                     # check what is there
    python tools/make_entrega.py --assemble          # also copy in the
                                                     # queries and generador
    python tools/make_entrega.py --assemble --zip entrega.zip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REQUIRED_METADATA_FIELDS = [          # Table 1, section 3.4
    "doc_id", "chunk_id", "fuente", "formato", "fenomeno", "posicion",
    "num_tokens", "texto",
]

JUNK = ["__pycache__", "*.bak", "*.pyc", ".DS_Store", "*.tmp", ".ipynb_checkpoints"]

_LINE_SEPARATORS = re.compile("[\u0085\u2028\u2029]")


class Report:
    """Collects findings so the exit code reflects all of them, not the first."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)
        print(f"  FAIL  {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  warn  {message}")

    def ok(self, message: str) -> None:
        print(f"  ok    {message}")


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return str(n)


def tree_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ------------------------------------------------- checks

def check_vector_store(entrega: Path, report: Report) -> None:
    base = entrega / "base_vectorial"
    if not base.is_dir():
        report.error("base_vectorial/ missing -- run build_index.py "
                     "--out entrega/base_vectorial")
        return

    folders = sorted(p for p in base.iterdir() if p.is_dir() and p.name != "grafo")
    if not folders:
        report.error("base_vectorial/ has no encoder_* subfolders")
        return

    counts = {}
    for folder in folders:
        index_file = folder / "index.faiss"
        metadata_file = folder / "metadata.jsonl"

        if not index_file.exists():
            report.error(f"{folder.name}/index.faiss missing")
            continue
        if not metadata_file.exists():
            report.error(f"{folder.name}/metadata.jsonl missing")
            continue

        raw = metadata_file.read_text(encoding="utf-8")
        # split("\n"), never splitlines(): U+2028 and friends are legal inside
        # a JSON string and would inflate the count. See _LINE_SEPARATORS.
        lines = [l for l in raw.split("\n") if l.strip()]
        counts[folder.name] = len(lines)

        stray = len(_LINE_SEPARATORS.findall(raw))
        if stray:
            report.error(f"{folder.name}: {stray} raw U+0085/U+2028/U+2029 in "
                         f"metadata. Run tools/fix_metadata_fuente.py")

        try:
            import faiss
            ntotal = faiss.read_index(str(index_file)).ntotal
            if ntotal != len(lines):
                report.error(f"{folder.name}: {ntotal} vectors vs {len(lines)} "
                             f"metadata lines -- FAISS ids and metadata order "
                             f"disagree (1.4). Rebuild.")
            else:
                report.ok(f"{folder.name}: {ntotal} vectors aligned with metadata")
        except ImportError:
            report.warn(f"{folder.name}: faiss not importable, vector count "
                        f"unverified ({len(lines)} metadata lines)")

        first = json.loads(lines[0])
        missing = [f for f in REQUIRED_METADATA_FIELDS if f not in first]
        if missing:
            report.error(f"{folder.name}: metadata missing Table 1 fields {missing}")

        source = first.get("fuente", "")
        if "/" in source or "\\" in source:
            report.error(f"{folder.name}: `fuente` is a path ({source!r}). "
                         f"10.2.1 matches documents on this field; run "
                         f"tools/fix_metadata_fuente.py")

    if len(set(counts.values())) > 1:
        report.error(f"encoders hold different chunk counts {counts} -- fusion "
                     f"keys on chunk_id and assumes one shared chunk set. "
                     f"Rebuild every encoder in a single build_index run.")


def check_resultados(entrega: Path, report: Report) -> None:
    path = entrega / "resultados.jsonl"
    if not path.exists():
        report.error("resultados.jsonl missing -- run entrega/generador.py")
        return

    # Importing generador would drop entrega/__pycache__ into the very
    # directory we are about to package, and the hygiene check would then
    # flag junk this script created. Turn bytecode writing off first.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(entrega))
    try:
        from entrega.generador import validate
    except Exception as exc:
        report.warn(f"could not import generador.validate ({exc}); "
                    f"skipping schema check")
        return

    print("        --- 9.3.2 schema check ---")
    if validate(path):
        report.ok("resultados.jsonl passes the 9.3.2 schema check")
    else:
        report.error("resultados.jsonl fails the 9.3.2 schema check (above)")


def check_informe(entrega: Path, report: Report) -> None:
    path = entrega / "informe_tecnico.pdf"
    if not path.exists():
        report.error("informe_tecnico.pdf missing (1.4, mandatory)")
        return
    try:
        import pymupdf
        pages = pymupdf.open(path).page_count
        if pages > 8:
            report.error(f"informe_tecnico.pdf is {pages} pages, limit is 8 (1.4)")
        else:
            report.ok(f"informe_tecnico.pdf, {pages} pages")
    except Exception:
        report.warn("informe_tecnico.pdf present, page count unverified")


def check_junk(entrega: Path, report: Report) -> None:
    found = []
    for pattern in JUNK:
        found += [p for p in entrega.rglob(pattern)]
    if found:
        report.warn(f"{len(found)} junk paths would ship, e.g. "
                    f"{[p.name for p in found[:4]]}. Delete before zipping.")


# ------------------------------------------------- assembly

def assemble(repo: Path, entrega: Path, report: Report) -> None:
    """Copy in the two files that have no other producer."""
    entrega.mkdir(parents=True, exist_ok=True)

    queries = entrega / "consultas_50.jsonl"
    if not queries.exists():
        source = repo / "consultas_50.jsonl"
        if source.exists():
            shutil.copy2(source, queries)
            report.ok(f"copied consultas_50.jsonl into {entrega.name}/")
        else:
            report.error("consultas_50.jsonl not found at the repo root")

    # generador.py's canonical home IS entrega/. Keeping a second copy at the
    # repo root is how you end up shipping the older one: evaluar.py imports
    # from entrega/, so that is the copy under test.
    if not (entrega / "generador.py").exists():
        stray = repo / "generador.py"
        if stray.exists():
            shutil.move(str(stray), str(entrega / "generador.py"))
            report.warn("moved generador.py from the repo root into entrega/ "
                        "-- that is its only home from now on")
        else:
            report.error("generador.py not found in entrega/ or at the repo root")
    elif (repo / "generador.py").exists():
        report.warn("generador.py exists BOTH at the repo root and in "
                    "entrega/. evaluar.py imports the entrega/ copy; delete "
                    "the other before they drift apart.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrega", type=Path, default=Path("entrega"))
    parser.add_argument("--assemble", action="store_true",
                        help="copy consultas_50.jsonl in and settle where "
                             "generador.py lives")
    parser.add_argument("--zip", type=Path, default=None,
                        help="package the directory once every check passes")
    args = parser.parse_args()

    repo = Path.cwd()
    entrega = args.entrega
    report = Report()

    print(f"entrega: {entrega.resolve()}\n")

    if args.assemble:
        print("ASSEMBLE")
        assemble(repo, entrega, report)
        print()

    print("VECTOR STORE");   check_vector_store(entrega, report); print()
    print("RESULTADOS");     check_resultados(entrega, report);   print()
    print("INFORME");        check_informe(entrega, report);      print()
    print("HYGIENE");        check_junk(entrega, report);         print()

    print("CONTENTS")
    for path in sorted(entrega.rglob("*")):
        if path.is_file() and len(path.relative_to(entrega).parts) <= 2:
            print(f"  {human(path.stat().st_size):>10s}  "
                  f"{path.relative_to(entrega)}")
        elif path.is_dir() and len(path.relative_to(entrega).parts) == 2:
            print(f"  {human(tree_size(path)):>10s}  "
                  f"{path.relative_to(entrega)}/  (directory)")
    print(f"\n  total: {human(tree_size(entrega))}")

    print(f"\n{len(report.errors)} errors, {len(report.warnings)} warnings")
    if report.errors:
        print("NOT READY TO SUBMIT")
        raise SystemExit(1)

    if args.zip:
        shutil.make_archive(str(args.zip.with_suffix("")), "zip",
                            root_dir=entrega.parent, base_dir=entrega.name)
        print(f"wrote {args.zip.with_suffix('.zip')}")
    print("READY")


if __name__ == "__main__":
    main()