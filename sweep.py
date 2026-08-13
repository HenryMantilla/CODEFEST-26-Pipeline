from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "entrega"))

import eval as evaluar                                            
from entrega.generador import (GraphIndex, LexicalIndex,          
                               RetrievalConfig, load_stores, retrieve)

GRIDS: dict[str, list[tuple[str, dict]]] = {
    "graph": [
        ("off",                       {"graph_weight": 0.0}),
        ("weight .5 (default)",       {}),
        ("weight 2.0",                {"graph_weight": 2.0}),
        ("weight .5, neighbour .2",   {"graph_neighbour": 0.2}),
        ("weight .5, neighbour .7",   {"graph_neighbour": 0.7}),
        ("ABLATION weight 1.0 (pre-fix default)", {"graph_weight": 1.0}),
    ],

    "doc": [
        ("rankdecay, decay .85 (default)",       {}),
        ("rankdecay, decay .95 (flatter)",       {"doc_decay": 0.95}),
        ("rankdecay, decay .70 (concentrated)",  {"doc_decay": 0.70}),
        ("rankdecay, decay .50 (top-heavy)",     {"doc_decay": 0.50}),
        ("rankdecay, pool 60",                   {"doc_pool": 60}),
        ("rankdecay, pool 100",                  {"doc_pool": 100}),
        ("rankdecay, pool 15 (tighter)",         {"doc_pool": 15}),
        ("ABLATION cosine + max (pre-fix default)",
         {"doc_score": "cosine"}),
        ("ABLATION combsum + sum3, pool 400",
         {"doc_score": "combsum", "doc_agg": "sum", "doc_pool": 400}),
        ("ABLATION combsum + mean3, pool 400",
         {"doc_score": "combsum", "doc_agg": "mean", "doc_pool": 400}),
        ("ABLATION combsum + rrf3, pool 400",
         {"doc_score": "combsum", "doc_agg": "rrf", "doc_pool": 400}),
    ],

    "phenomenon": [
        ("baseline (add .20, default)", {}),
        ("frag 0",                      {"phenomenon_boost": 0.0}),
        ("doc 0",                       {"phenomenon_boost_doc": 0.0}),
        ("both 0",                      {"phenomenon_boost": 0.0,
                                         "phenomenon_boost_doc": 0.0}),
        ("frag add .10",                {"phenomenon_boost": 0.10}),
        ("frag add .30",                {"phenomenon_boost": 0.30}),
        ("doc .6",                      {"phenomenon_boost_doc": 0.6}),
        ("ABLATION frag multiply .08 (pre-fix default)",
         {"phenomenon_boost": 0.08, "phenomenon_mode": "multiply"}),
        ("ABLATION frag multiply .20",
         {"phenomenon_boost": 0.20, "phenomenon_mode": "multiply"}),
    ],
    "rerank": [
        ("off",            {"reranker": ""}),
        ("blend .3",       {"rerank_blend": 0.3}),
        ("blend .5 (default)", {}),
        ("blend .7",       {"rerank_blend": 0.7}),
        ("blend 1.0",      {"rerank_blend": 1.0}),
        ("depth 400",      {"rerank_depth": 400}),
        ("no context",     {"rerank_context": False}),
    ],

    "expansion": [
        ("baseline",        {}),
        ("rm3 5 terms",     {"rm3_terms": 5}),
        ("rm3 10 terms",    {"rm3_terms": 10}),
        ("rm3 10, fb 20",   {"rm3_terms": 10, "rm3_feedback": 20}),
        ("rm3 20 terms",    {"rm3_terms": 20}),
        ("rm3 10, orig .8", {"rm3_terms": 10, "rm3_original_weight": 0.8}),
    ],
    "diversity": [
        ("baseline",            {}),
        ("mmr .9",               {"mmr_lambda": 0.9}),
        ("mmr .8",               {"mmr_lambda": 0.8}),
        ("mmr .7",               {"mmr_lambda": 0.7}),
        ("mmr .5",               {"mmr_lambda": 0.5}),
        ("mmr .8 + dedupe off",  {"mmr_lambda": 0.8, "dedupe_threshold": 0.0}),
    ],

    "fragment": [
        ("baseline (bm25 .5, dedupe .45)", {}),
        ("dedupe .30",     {"dedupe_threshold": 0.30}),
        ("dedupe .60",     {"dedupe_threshold": 0.60}),
        ("dedupe off",     {"dedupe_threshold": 0.0}),
        ("bm25 0",         {"bm25_weight": 0.0}),
        ("bm25 2",         {"bm25_weight": 2.0}),
        ("depth 3000",     {"depth": 3000}),
        ("ABLATION bm25 1.0 (pre-fix default)", {"bm25_weight": 1.0}),
    ],

    "final_check": [
        ("baseline (graph .5, depth 150 -- current code defaults)", {}),
        ("graph 1.0 alone",                  {"graph_weight": 1.0}),
        ("depth 400 alone",                  {"rerank_depth": 400}),
        ("graph 1.0 + depth 400 (main candidate)",
         {"graph_weight": 1.0, "rerank_depth": 400}),
        ("graph .75 alone (midpoint)",       {"graph_weight": 0.75}),
        ("depth 300 alone (midpoint)",       {"rerank_depth": 300}),
        ("graph 1.0 + depth 300",
         {"graph_weight": 1.0, "rerank_depth": 300}),
        ("graph .75 + depth 400",
         {"graph_weight": 0.75, "rerank_depth": 400}),
        ("main candidate + doc_decay .70",
         {"graph_weight": 1.0, "rerank_depth": 400, "doc_decay": 0.70}),
        ("main candidate + bm25 1.0 (re-test joint decision)",
         {"graph_weight": 1.0, "rerank_depth": 400, "bm25_weight": 1.0}),
    ],
}

_ABLATION_PREFIX = "ABLATION"


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score(gt, stores, cfg: RetrievalConfig, lexical, graph,
          frag_threshold: float, doc_k: int, frag_k: int) -> dict:
    """One configuration -> the four numbers, quietly.

    Document aggregation goes through evaluar._aggregate(), NOT
    aggregate_documents() directly -- see the module docstring. This is the
    same function eval.py's evaluate() calls, so a row scored here and the
    same configuration scored via `python eval.py` cannot silently disagree.
    """
    f1 = nd = mrr_doc = mrr_frag = 0.0
    for entry in gt:
        result = retrieve(stores, entry["query"], entry.get("query_id", ""),
                          cfg, lexical if cfg.bm25_weight > 0 else None,
                          graph if cfg.graph_weight > 0 else None)
        grades = entry.get("grades")

        ranked = evaluar._aggregate(result.doc_ranking, doc_k, cfg.doc_pool, cfg)
        deep = evaluar._aggregate(result.doc_ranking, None,
                                  len(result.doc_ranking), cfg)
        by_doc = {m["doc_id"]: m for m, _ in result.doc_ranking}
        wanted = [evaluar.norm_doc(d) for d in entry["documents"] if d]

        def hit(doc_id: str) -> bool:
            if grades is not None:
                return doc_id in entry["documents"]
            meta = by_doc.get(doc_id, {})
            source = evaluar.norm_doc(meta.get("nombre_archivo")
                                      or meta.get("fuente", ""))
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


def _rank_key(row: dict) -> float:
    return row["mrr_doc"] + row["mrr_frag"]


def run_row(gt, stores, overrides: dict, lexical, graph,
           frag_threshold: float, doc_k: int, frag_k: int) -> dict:
    """
    score() wrapped for fault isolation and timing.

    A single bad combination (an unsupported doc_score/doc_agg pairing, a
    reranker OOM at a large --rerank-depth) used to take the whole sweep
    down, discarding every row already computed -- expensive when a run
    covers seven grids over a loaded 140k-chunk index. Failures are
    reported per-row instead, and the sweep keeps going.
    """
    cfg = RetrievalConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)

    started = time.perf_counter()
    try:
        row = score(gt, stores, cfg, lexical, graph, frag_threshold,
                    doc_k, frag_k)
        row["error"] = None
    except Exception as exc:  # noqa: BLE001
        row = {"f1": None, "ndcg": None, "mrr_doc": None, "mrr_frag": None,
               "error": f"{type(exc).__name__}: {exc}"}
    row["seconds"] = time.perf_counter() - started
    row["overrides"] = overrides
    return row


def print_row(label: str, row: dict) -> None:
    if row["error"]:
        print(f"  {label:34s} FAILED: {row['error'][:70]}")
        return
    print(f"  {label:34s} {row['f1']:7.3f} {row['ndcg']:8.3f} "
          f"{row['mrr_doc']:8.3f} {row['mrr_frag']:8.3f}  "
          f"{row['seconds']:6.1f}s")

    
def combine_and_score(gt, stores, lexical, graph, frag_threshold: float,
                      doc_k: int, frag_k: int,
                      winners: dict[str, tuple[str, dict, dict]]) -> None:
    """
    Merge the best-by-MRR row's overrides from each grid actually run this
    session (ABLATION rows excluded -- they exist for comparison, not to be
    shipped) and score the combination.

    Single-axis grids cannot see interaction effects by construction: a knob
    that helps in isolation can help less, more, or even hurt once another
    winning knob is already active. This is the closest this tool gets to
    answering "what should I actually ship" rather than "does this one
    thing help".
    """
    if len(winners) < 2:
        return

    merged: dict = {}
    collisions: list[str] = []
    for grid_name, (label, overrides, _row) in winners.items():
        for key, value in overrides.items():
            if key in merged and merged[key] != value:
                collisions.append(
                    f"    {key}: {grid_name!r}'s winner wants {value!r}, "
                    f"already set to {merged[key]!r} by an earlier grid "
                    f"-- keeping the earlier value")
                continue
            merged[key] = value

    print(f"\n=== combined ===")
    print("  merging the best-by-MRR row from each grid run this session "
          "(ABLATION rows excluded):")
    for grid_name, (label, _o, row) in winners.items():
        print(f"    {grid_name:12s} -> {label}  "
              f"(mrr_doc={row['mrr_doc']:.3f} mrr_frag={row['mrr_frag']:.3f})")
    if collisions:
        print("  field collisions (first grid to touch a field wins; "
              "shown so this is visible, not silent):")
        for line in collisions:
            print(line)
    print(f"  merged overrides: {merged}\n")

    row = run_row(gt, stores, merged, lexical, graph, frag_threshold,
                  doc_k, frag_k)
    print_row("combined", row)

    baseline = run_row(gt, stores, {}, lexical, graph, frag_threshold,
                       doc_k, frag_k)
    print_row("plain baseline (for reference)", baseline)

    if row["error"] is None and baseline["error"] is None:
        delta_doc = row["mrr_doc"] - baseline["mrr_doc"]
        delta_frag = row["mrr_frag"] - baseline["mrr_frag"]
        print(f"\n  combined vs plain baseline: "
              f"MRRdoc {delta_doc:+.3f}  MRRfrag {delta_frag:+.3f}")
        best_single = max(winners.values(), key=lambda w: _rank_key(w[2]))
        if _rank_key(row) < _rank_key(best_single[2]):
            print(f"  NOTE: combined scores BELOW its best single "
                  f"contributor ({best_single[0]!r} from "
                  f"{[k for k, v in winners.items() if v is best_single][0]}). "
                  f"That is a real negative interaction between two winning "
                  f"knobs, not noise -- worth checking which pairing is at "
                  f"fault before shipping the full combination.")

    return {"overrides": merged, "sources": {g: label for g, (label, _o, _r)
                                             in winners.items()},
           "collisions": collisions, "result": row, "baseline": baseline}



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--judgments", type=Path)
    parser.add_argument("--index-dir", type=Path,
                        default=Path("entrega/base_vectorial"))
    parser.add_argument("--queries", type=Path, default=Path("consultas_50.jsonl"))
    parser.add_argument("--grid", nargs="+", default=["all"],
                        choices=[*GRIDS, "all"],
                        help="one or more grid names, or 'all' (default). "
                             "Running more than one triggers the combined "
                             "pass unless --no-combine is given.")
    parser.add_argument("--no-combine", action="store_true",
                        help="skip the combined pass even with multiple grids")
    parser.add_argument("--doc-encoder", default="")
    parser.add_argument("--frag-threshold", type=float, default=0.30)
    parser.add_argument("--doc-k", type=int, default=3)
    parser.add_argument("--frag-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None,
                        help="write every row's metrics, timing and any "
                             "error, plus the combined pass, to this JSON "
                             "file -- a sweep session survives past the "
                             "terminal scrollback")
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


    stores = load_stores(args.index_dir, args.doc_encoder, "")
    lexical = LexicalIndex.build(stores[0].metadata,
                                 [q["query"] for q in gt])
    graph = GraphIndex.load(args.index_dir / "grafo" / "grafo.graphml")
    print(f"Grafo: {'off (not built)' if graph is None else f'{len(graph.entities)} entities'}")
    print(f"{len(gt)} queries; resolution of F1@3 is 1/{len(gt)} = "
          f"{1/len(gt):.3f}")

    print(f"query_ids used: {sorted(q.get('query_id', '?') for q in gt)}\n")

    grid_names = list(GRIDS) if "all" in args.grid else args.grid
    export: dict = {"grids": {}, "combined": None}
    winners: dict[str, tuple[str, dict, dict]] = {}

    for name in grid_names:
        print(f"=== {name} ===")
        print(f"  {'config':34s} {'F1@3':>7s} {'nDCG@10':>8s} "
              f"{'MRRdoc':>8s} {'MRRfrag':>8s}  {'time':>6s}")
        results = []
        for label, overrides in GRIDS[name]:
            row = run_row(gt, stores, overrides, lexical, graph,
                         args.frag_threshold, args.doc_k, args.frag_k)
            results.append((label, row))
            print_row(label, row)

        export["grids"][name] = [
            {"label": label, **row} for label, row in results]

        ok = [(label, row) for label, row in results
             if row["error"] is None and not label.startswith(_ABLATION_PREFIX)]
        if ok:
            best_label, best_row = max(ok, key=lambda lr: _rank_key(lr[1]))
            print(f"  -> best by MRR: {best_label}\n")
            winners[name] = (best_label, best_row["overrides"], best_row)
        else:
            print(f"  -> no successful non-ablation row to select a "
                  f"winner from\n")

    if not args.no_combine and len(winners) >= 2:
        combined = combine_and_score(
            gt, stores, lexical, graph, args.frag_threshold,
            args.doc_k, args.frag_k, winners)
        export["combined"] = combined

    print("Read MRR while tuning; report F1@3 and nDCG@10. On a small "
          "sample a change\nworth under ~0.05 in F1@3 is noise -- MRR is "
          "the one that moves honestly.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(export, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()