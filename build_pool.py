"""
build_pool.py — Turn 7 usable queries into 50 by judging your own output.

THE PROBLEM
    The organisers' sample ground truth covers 7 of the 50 queries with 12
    fragments. F1@3 on it moves in steps of 1/7, so every configuration
    change lands on the same handful of values and most real improvements are
    invisible. Tuning against it past a certain point is fitting noise.

THE STANDARD ANSWER: POOLED RELEVANCE JUDGMENTS
    This is how TREC builds ground truth and it works here for the same
    reason. Run several DIFFERENT configurations over all 50 queries, take
    the union of their top-k fragments, and judge each unique fragment ONCE
    by hand. Every configuration can then be scored against the pool --
    including ones invented later, as long as their results fall inside it.

    Pooling across diverse configurations is what makes it fair. If the pool
    came from one configuration, that configuration would score perfectly by
    construction, because everything it returned would have been judged and
    everything it missed would be invisible. Diversity in the pool is not a
    nicety, it is the thing that makes the numbers mean anything.

    Unjudged is treated as NOT relevant. That biases against configurations
    whose results fall outside the pool, which is why the pool should be
    built once, wide, and reused.

THE COST, HONESTLY
    50 queries x ~25 unique fragments is around 1,200 judgments. At ten
    seconds each that is three or four hours, and it splits cleanly across a
    team because each row is independent. That is the single highest-value
    use of an afternoon left in this project: it converts every subsequent
    experiment from a coin flip into a measurement.

JUDGING SCALE (graded, matching 10.2.1's r_i)
    2  answers the question directly
    1  on topic and useful context, does not answer it
    0  wrong topic, boilerplate, table of contents, reference list

Usage:
    python tools/build_pool.py --index-dir entrega/base_vectorial \
        --queries entrega/consultas_50.jsonl --out pool.xlsx
    # ... judge pool.xlsx by hand, fill the `relevancia` column ...
    python eval.py --judgments pool.xlsx --index-dir entrega/base_vectorial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "entrega"))

from entrega.generador import (GraphIndex, LexicalIndex, RetrievalConfig,  # noqa: E402
                       load_stores, read_queries, retrieve)


POOL_CONFIGS: list[tuple[str, dict]] = [
    ("dense+bm25+rerank", {}),
    ("no reranker",       {"reranker": ""}),
    ("no bm25",           {"bm25_weight": 0.0}),
    ("bm25 only",         {"bm25_weight": 1.0, "dense_only_off": True}),
    ("no phenomenon",     {"phenomenon_boost": 0.0, "phenomenon_boost_doc": 0.0}),
    ("graph weight 2",     {"graph_weight": 2.0}),   # only config with the
                                                     # graph channel emphasised
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path,
                        default=Path("entrega/base_vectorial"))
    parser.add_argument("--queries", type=Path,
                        default=Path("entrega/consultas_50.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("pool.xlsx"))
    parser.add_argument("--depth", type=int, default=10,
                        help="fragments pooled per configuration per query")
    parser.add_argument("--doc-encoder", default="")
    args = parser.parse_args()

    import openpyxl

    stores = load_stores(args.index_dir, args.doc_encoder)
    queries = read_queries(args.queries)
    print(f"{len(queries)} queries, {len(POOL_CONFIGS)} configurations")

    lexical = LexicalIndex.build(stores[0].metadata, [q for _i, q in queries])
    graph = GraphIndex.load(args.index_dir / "grafo" / "grafo.graphml")
    print(f"Grafo: {'off (not built)' if graph is None else f'{len(graph.entities)} entities'}")

    # chunk_id -> row, per query. Judged once, reused by every configuration.
    pooled: dict[str, dict[str, dict]] = {}

    for name, overrides in POOL_CONFIGS:
        cfg = RetrievalConfig()
        dense_off = overrides.pop("dense_only_off", False)
        for key, value in overrides.items():
            setattr(cfg, key, value)
        print(f"  {name} ...", flush=True)

        for query_id, text in queries:
            result = retrieve(stores, text, query_id, cfg,
                              lexical if cfg.bm25_weight > 0 else None,
                              graph if cfg.graph_weight > 0 else None)
            ranking = result.unique
            if dense_off and lexical is not None:
                ranking = lexical.search(text, args.depth)
            bucket = pooled.setdefault(query_id, {})
            for rank, (meta, _score) in enumerate(ranking[:args.depth], 1):
                row = bucket.setdefault(meta["chunk_id"], {
                    "query_id": query_id, "query": text,
                    "chunk_id": meta["chunk_id"], "doc_id": meta["doc_id"],
                    "fuente": meta.get("fuente", ""),
                    "contexto": meta.get("contexto", ""),
                    "texto": meta.get("texto", ""),
                    "best_rank": rank, "found_by": set()})
                row["best_rank"] = min(row["best_rank"], rank)
                row["found_by"].add(name)

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "juicios"
    sheet.append(["query_id", "consulta", "relevancia", "chunk_id", "doc_id",
                  "fuente", "contexto", "texto", "mejor_rango", "hallado_por"])
    sheet.freeze_panes = "A2"
    for width, column in zip((9, 46, 11, 26, 18, 30, 30, 90, 11, 24), "ABCDEFGHIJ"):
        sheet.column_dimensions[column].width = width

    total = 0
    for query_id in sorted(pooled):
        # Best rank first: the fragments most likely to be relevant come
        # first, so a judge who runs out of time has still covered what
        # matters most.
        rows = sorted(pooled[query_id].values(), key=lambda r: r["best_rank"])
        for row in rows:
            sheet.append([row["query_id"], row["query"], "", row["chunk_id"],
                          row["doc_id"], Path(row["fuente"]).name,
                          row["contexto"][:200], row["texto"][:1500],
                          row["best_rank"], ", ".join(sorted(row["found_by"]))])
            total += 1

    workbook.save(args.out)
    per_query = total / max(len(pooled), 1)
    print(f"\nWrote {args.out}: {total} unique fragments across "
          f"{len(pooled)} queries ({per_query:.1f} each)")
    print(f"  Estimated judging time at 10s per row: "
          f"{total * 10 / 3600:.1f} hours -- split it across the team.")
    print("  Fill column C (`relevancia`) with 2 / 1 / 0. Leave blank to mean 0.")
    print("     2  answers the question directly")
    print("     1  on topic, useful context, does not answer it")
    print("     0  wrong topic, boilerplate, index, reference list")
    print(f"\n  Then: python eval.py --judgments {args.out}")


if __name__ == "__main__":
    main()