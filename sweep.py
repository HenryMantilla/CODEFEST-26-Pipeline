"""
sweep.py — Run many generador configurations against one loaded index.

WHY THIS EXISTS
    Every `eval.py` run reloads two encoders and rebuilds BM25 over 140k
    chunks: minutes of setup for seconds of scoring. Fifteen configurations
    that way is an hour of waiting, which is why knobs go untested. Here the
    index, the encoders and the lexical channel are loaded ONCE and every
    configuration reuses them.

WHAT TO READ
    F1@3 and nDCG@10 are the official metrics but move in steps of 1/N, and
    with the 7-query sample N is seven. MRR over the full rankings is
    continuous and responds to a document moving 40 -> 5, which the official
    metrics cannot see. Sort by MRR while tuning; report F1@3 and nDCG@10.

    With --judgments (tools/build_pool.py) N becomes 50 and the official
    metrics become usable on their own.

Usage:
    python tools/sweep.py --gt "FASE ORDENADA CODEFEST.xlsx"
    python tools/sweep.py --judgments pool.xlsx --grid doc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "entrega"))

import eval as  evaluar
from entrega.generador import (GraphIndex, LexicalIndex, RetrievalConfig,  # noqa: E402
                       aggregate_documents, load_stores, retrieve)

# Grids, grouped so a run answers one question rather than mixing several.
GRIDS: dict[str, list[tuple[str, dict]]] = {
    "graph": [
        ("off",                  {"graph_weight": 0.0}),
        ("weight .5",            {"graph_weight": 0.5}),
        ("weight 1.0 (default)", {}),
        ("weight 2.0",           {"graph_weight": 2.0}),
        ("weight 1, neighbour .2", {"graph_neighbour": 0.2}),
        ("weight 1, neighbour .7", {"graph_neighbour": 0.7}),
    ],
    # gt_doc_rank showed correct documents at deep rank 3-5 that did NOT
    # survive --doc-pool 30. This is the most direct lever on F1@3.
    # NOTE: --doc-pool alone does nothing under mode="max" -- enlarging the
    # pool can only admit lower-scoring documents, which never displace the
    # top 3. Verified: 30/60/150/400 gave identical F1@3. The pool only
    # becomes live once aggregation stops being pure max.
    "doc": [
        ("baseline (max, 30)",  {}),
        ("sum3, pool 150",      {"doc_agg": "sum", "doc_pool": 150}),
        ("sum3, pool 400",      {"doc_agg": "sum", "doc_pool": 400}),
        ("sum5, pool 400",      {"doc_agg": "sum", "doc_top_m": 5, "doc_pool": 400}),
        ("mean3, pool 400",     {"doc_agg": "mean", "doc_pool": 400}),
        ("rrf3, pool 400",      {"doc_agg": "rrf", "doc_pool": 400}),
        ("sum3 400, bonus 0",   {"doc_agg": "sum", "doc_pool": 400,
                                 "doc_hit_bonus": 0.0}),
        ("combsum + sum3 400",  {"doc_score": "combsum", "doc_agg": "sum",
                                 "doc_pool": 400}),
        ("no phen boost doc",   {"phenomenon_boost_doc": 0.0}),
    ],
    # q002's ground truth includes an F3 document answering an F1 query, so
    # the phenomenon prior is provably wrong at least once.
    "phenomenon": [
        ("baseline",       {}),
        ("frag 0",         {"phenomenon_boost": 0.0}),
        ("doc 0",          {"phenomenon_boost_doc": 0.0}),
        ("both 0",         {"phenomenon_boost": 0.0, "phenomenon_boost_doc": 0.0}),
        ("frag .16",       {"phenomenon_boost": 0.16}),
        ("doc .6",         {"phenomenon_boost_doc": 0.6}),
    ],
    "rerank": [
        ("off",            {"reranker": ""}),
        ("blend .3",       {"rerank_blend": 0.3}),
        ("blend .5",       {"rerank_blend": 0.5}),
        ("blend .7",       {"rerank_blend": 0.7}),
        ("blend 1.0",      {"rerank_blend": 1.0}),
        ("depth 400",      {"rerank_depth": 400}),
        ("no context",     {"rerank_context": False}),
    ],
    # Neither of these is in any FAQ question, which is the point: the whole
    # field is converging on dense + BM25 + cross-encoder, so those are table
    # stakes. Coverage and expansion are where a lead comes from.
    "expansion": [
        ("baseline",        {}),
        ("rm3 5 terms",     {"rm3_terms": 5}),
        ("rm3 10 terms",    {"rm3_terms": 10}),
        ("rm3 10, fb 20",   {"rm3_terms": 10, "rm3_feedback": 20}),
        ("rm3 20 terms",    {"rm3_terms": 20}),
        ("rm3 10, orig .8", {"rm3_terms": 10, "rm3_original_weight": 0.8}),
    ],
    "diversity": [
        ("baseline",        {}),
        ("mmr .9",          {"mmr_lambda": 0.9}),
        ("mmr .8",          {"mmr_lambda": 0.8}),
        ("mmr .7",          {"mmr_lambda": 0.7}),
        ("mmr .5",          {"mmr_lambda": 0.5}),
        ("mmr .8 + dedupe off", {"mmr_lambda": 0.8, "dedupe_threshold": 0.0}),
    ],
    "fragment": [
        ("baseline",       {}),
        ("dedupe .30",     {"dedupe_threshold": 0.30}),
        ("dedupe .60",     {"dedupe_threshold": 0.60}),
        ("dedupe off",     {"dedupe_threshold": 0.0}),
        ("bm25 0",         {"bm25_weight": 0.0}),
        ("bm25 2",         {"bm25_weight": 2.0}),
        ("depth 3000",     {"depth": 3000}),
    ],
}


def score(gt, stores, cfg, lexical, graph, frag_threshold: float,
          doc_k: int, frag_k: int) -> dict:
    """One configuration -> the four numbers, quietly."""
    f1 = nd = mrr_doc = mrr_frag = 0.0
    for entry in gt:
        result = retrieve(stores, entry["query"], entry.get("query_id", ""),
                          cfg, lexical if cfg.bm25_weight > 0 else None,
                          graph if cfg.graph_weight > 0 else None)
        grades = entry.get("grades")

        ranked = aggregate_documents(result.doc_ranking, doc_k, cfg.doc_pool,
                                     cfg.doc_hit_bonus, cfg.doc_hit_cap,
                                     cfg.doc_agg, cfg.doc_top_m)
        deep = aggregate_documents(result.doc_ranking, None,
                                   len(result.doc_ranking),
                                   cfg.doc_hit_bonus, cfg.doc_hit_cap,
                                   cfg.doc_agg, cfg.doc_top_m)
        by_doc = {m["doc_id"]: m for m, _ in result.doc_ranking}
        wanted = [evaluar.norm_doc(d) for d in entry["documents"] if d]

        def hit(doc_id: str) -> bool:
            if grades is not None:
                return doc_id in entry["documents"]
            source = evaluar.norm_doc(by_doc.get(doc_id, {}).get("fuente", ""))
            return any(source.endswith(w) or w.endswith(source)
                       for w in wanted if w)

        doc_hits = [hit(d) for d in ranked]
        f1 += evaluar.f1_at_k(doc_hits, len(wanted), doc_k)
        best = next((i for i, d in enumerate(deep, 1) if hit(d)), None)
        mrr_doc += 1.0 / best if best else 0.0

        if grades is not None:
            frag_hits = [grades.get(m["chunk_id"], 0) > 0
                         for m, _s in result.unique[:frag_k]]
            n_rel = sum(1 for g in grades.values() if g > 0)
        else:
            gt_grams = [evaluar.ngrams(f["text"]) for f in entry["fragments"]]
            frag_hits, covered = [], set()
            for meta, _s in result.unique[:frag_k]:
                chunk = evaluar.ngrams(meta["texto"])
                fresh = {i for i, g in enumerate(gt_grams)
                         if i not in covered
                         and evaluar.coverage(g, chunk) >= frag_threshold}
                frag_hits.append(bool(fresh))
                covered |= fresh
            n_rel = len(gt_grams)
        nd += evaluar.ndcg(frag_hits, frag_k, n_rel)
        mrr_frag += evaluar.reciprocal_rank(frag_hits)

    n = len(gt) or 1
    return {"f1": f1 / n, "ndcg": nd / n,
            "mrr_doc": mrr_doc / n, "mrr_frag": mrr_frag / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--index-dir", type=Path,
                        default=Path("entrega/base_vectorial"))
    parser.add_argument("--queries", type=Path, default=Path("consultas_50.jsonl"))
    parser.add_argument("--grid", default="all",
                        choices=[*GRIDS, "all"])
    parser.add_argument("--doc-encoder", default="")
    parser.add_argument("--frag-threshold", type=float, default=0.30)
    parser.add_argument("--doc-k", type=int, default=3)
    parser.add_argument("--frag-k", type=int, default=10)
    args = parser.parse_args()

    if args.judgments:
        gt = evaluar.load_judgments(args.judgments)
    elif args.gt:
        gt = evaluar.load_ground_truth(args.gt)
        if args.queries.exists():
            evaluar.resolve_ids(gt, args.queries)
            gt = [q for q in gt if q.get("query_id")]
    else:
        raise SystemExit("Pass --gt or --judgments.")

    stores = load_stores(args.index_dir, args.doc_encoder, "cosine")
    lexical = LexicalIndex.build(stores[0].metadata,
                                 [q["query"] for q in gt])
    graph = GraphIndex.load(args.index_dir / "grafo" / "grafo.graphml")
    print(f"Grafo: {'off (not built)' if graph is None else f'{len(graph.entities)} entities'}")
    print(f"{len(gt)} queries; resolution of F1@3 is 1/{len(gt)} = "
          f"{1/len(gt):.3f}\n")

    grids = list(GRIDS) if args.grid == "all" else [args.grid]
    for name in grids:
        print(f"=== {name} ===")
        print(f"  {'config':22s} {'F1@3':>7s} {'nDCG@10':>8s} "
              f"{'MRRdoc':>8s} {'MRRfrag':>8s}")
        results = []
        for label, overrides in GRIDS[name]:
            cfg = RetrievalConfig()
            for key, value in overrides.items():
                setattr(cfg, key, value)
            row = score(gt, stores, cfg, lexical, graph, args.frag_threshold,
                        args.doc_k, args.frag_k)
            results.append((label, row))
            print(f"  {label:22s} {row['f1']:7.3f} {row['ndcg']:8.3f} "
                  f"{row['mrr_doc']:8.3f} {row['mrr_frag']:8.3f}", flush=True)

        best = max(results, key=lambda r: r[1]["mrr_doc"] + r[1]["mrr_frag"])
        print(f"  -> best by MRR: {best[0]}\n")

    print("Read MRR while tuning; report F1@3 and nDCG@10. On a 7-query "
          "sample a change\nworth under ~0.05 in F1@3 is noise -- MRR is the "
          "one that moves honestly.")


if __name__ == "__main__":
    main()