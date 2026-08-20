"""Generic, pluggable weight-aggregation logic for scoring a report against
an RCB checklist -- kept separate from claude_scorer.py's actual
ground-truth rubric/prompt/Claude-calling implementation, so the two can be
read and reasoned about independently: this module is pure aggregation math
with no ground-truth content of its own.

`score_report_text_items` takes a `score_fn(report_text, item, instructions)`
callback and applies RCB's own weighted-aggregation logic on top -- currently
called only with claude_scorer.score_item_with_claude (host-side, via
domains/research/harness.py's compute_node_utility).
"""

from concurrent.futures import ThreadPoolExecutor


def score_report_text_items(report_text, checklist, instructions, score_fn, max_workers=8):
    """Score every text-type item in `checklist` with `score_fn(report_text, item, instructions)`.

    Returns (per_item_results, weighted_score): per_item_results is a list of
    {"index", "content", "weight", "score", "reasoning"} dicts (text items
    only, in checklist order); weighted_score is RCB's own weight-averaged
    0-100 aggregate over those items (0 if there are none).
    """
    text_items = [(i, item) for i, item in enumerate(checklist) if item.get("type", "text") == "text"]

    def score_one(indexed_item):
        i, item = indexed_item
        sr = score_fn(report_text, item, instructions)
        return {
            "index": i,
            "content": item.get("content", "")[:200],
            "weight": float(item.get("weight", 1.0)),
            "score": sr["score"],
            "reasoning": sr["reasoning"],
        }

    if not text_items:
        return [], 0.0

    with ThreadPoolExecutor(max_workers=min(len(text_items), max_workers)) as executor:
        results = list(executor.map(score_one, text_items))
    results.sort(key=lambda r: r["index"])

    total_weighted = sum(r["score"] * r["weight"] for r in results)
    total_weight = sum(r["weight"] for r in results)
    weighted_score = (total_weighted / total_weight) if total_weight > 0 else 0.0
    return results, weighted_score
