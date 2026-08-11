"""
check_graph_linking.py — Is the graph channel firing at all?

WHY THIS EXISTS
    weight 1.0 gave results identical to weight 0 (off) down to three decimal
    places, but weight 2.0 moved nDCG@10 down. Those two facts together are
    ambiguous: they are consistent with EITHER

      (a) the graph links entities on most queries and its ranking mostly
          agrees with the other channels -- so it rides along unnoticed at
          weight 1, and only overrides them, for the worse, at weight 2; or

      (b) the graph finds NO entities on most of these 7-8 queries, so it
          contributes nothing at any reasonable weight, and weight 2 hurt
          because of the one or two queries where it did fire.

    Those call for opposite responses. (a) means the graph's ranking signal
    is weak evidence -- worth keeping at a low weight, not worth chasing
    higher. (b) means the ENTITY EXTRACTION is the bottleneck, not the fusion
    weight, and no amount of --graph-weight tuning will fix it.

    This prints entity linking per query so the difference is visible
    directly, instead of inferred from how a downstream metric moved.

Usage:
    python tools/check_graph_linking.py --index-dir entrega/base_vectorial \
        --gt "FASE ORDENADA CODEFEST.xlsx"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "entrega"))

import eval  as evaluar
from entrega.generador import GraphIndex, read_queries  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path,
                        default=Path("entrega/base_vectorial"))
    parser.add_argument("--gt", type=Path,
                        help="score only the queries in the sample GT, "
                             "for comparison against the sweep")
    parser.add_argument("--queries", type=Path,
                        default=Path("entrega/consultas_50.jsonl"))
    args = parser.parse_args()

    graph = GraphIndex.load(args.index_dir / "grafo" / "grafo.graphml")
    if graph is None:
        raise SystemExit(f"No graph at {args.index_dir / 'grafo' / 'grafo.graphml'}")
    print(f"Grafo: {len(graph.entities)} entities, "
          f"{sum(len(v) for v in graph.neighbours.values()) // 2} relations\n")

    if args.gt:
        gt = evaluar.load_ground_truth(args.gt)
        evaluar.resolve_ids(gt, args.queries)
        pairs = [(e["query_id"], e["query"]) for e in gt if e.get("query_id")]
        pairs.sort()
        label = f"the {len(pairs)}-query sample"
    else:
        pairs = read_queries(args.queries)
        label = "all 50 queries"

    linked = 0
    print(f"{'query':8s} {'entities linked':50s} hits")
    for query_id, text in pairs:
        entities = graph.link(text)
        hit_chunks = set()
        for entity in entities:
            hit_chunks |= set(graph.entities.get(entity, {}).get("chunks", ()))
        linked += bool(entities)
        shown = ", ".join(entities) if entities else "(none)"
        print(f"{query_id:8s} {shown[:50]:50s} {len(hit_chunks)}")

    print(f"\n{linked}/{len(pairs)} queries link at least one entity "
          f"({label})")
    if linked < len(pairs) * 0.4:
        print("\n  Fewer than 40% of queries link an entity. --graph-weight "
              "cannot help the\n  rest -- the channel contributes nothing to "
              "them at any weight. Before\n  tuning weight further, check "
              "WHY: is --min-entity-freq filtering too\n  aggressively, does "
              "the heuristic extractor miss multi-word names, or do\n  these "
              "queries genuinely name few specific entities (plausible -- "
              "they are\n  abstractions like '¿Cómo contribuyen las "
              "economías ilícitas...?').")
    else:
        print("\n  Most queries link something. If weight 1.0 still looked "
              "like a no-op\n  in the sweep, the graph's ranking mostly "
              "AGREES with the other channels\n  rather than being silent -- "
              "it is riding along, not contributing distinctly.")


if __name__ == "__main__":
    main()