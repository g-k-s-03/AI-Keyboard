"""GLEU (Grammar BLEU) -- an n-gram precision/recall metric for grammar
correction tasks. For each n-gram order we take the minimum of precision
and recall (so the score penalizes both over- and under-generation, unlike
plain BLEU which only looks at precision), then combine orders 1-4 with a
geometric mean and average across references.

Pure Python, no external dependency.
"""
from collections import Counter

MAX_ORDER = 4


def _tokenize(text: str) -> list:
    return text.strip().split()


def _ngrams(tokens: list, n: int):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _order_score(hyp_tokens: list, ref_tokens: list, n: int):
    """Precision/recall of n-grams of this order, or None if neither
    sequence is long enough to contain an n-gram of this order (so that
    order is skipped rather than forced to zero -- short sentences would
    otherwise always score 0 once n exceeds their length)."""
    if len(hyp_tokens) < n and len(ref_tokens) < n:
        return None

    hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
    ref_ngrams = Counter(_ngrams(ref_tokens, n))
    overlap = sum((hyp_ngrams & ref_ngrams).values())

    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    precision = overlap / hyp_total if hyp_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    return precision, recall


def _gleu_against_one_reference(hyp_tokens: list, ref_tokens: list) -> float:
    scores = []
    for n in range(1, MAX_ORDER + 1):
        result = _order_score(hyp_tokens, ref_tokens, n)
        if result is None:
            continue
        precision, recall = result
        scores.append(min(precision, recall))

    if not scores or any(s <= 0 for s in scores):
        return 0.0

    product = 1.0
    for s in scores:
        product *= s
    return product ** (1.0 / len(scores))


def calculate_gleu(hypothesis: str, references: list) -> float:
    """Average GLEU of `hypothesis` against each reference, as a 0-100 float."""
    if not hypothesis or not references:
        return 0.0

    hyp_tokens = _tokenize(hypothesis)
    if not hyp_tokens:
        return 0.0

    per_reference_scores = []
    for reference in references:
        if not reference:
            continue
        ref_tokens = _tokenize(reference)
        if not ref_tokens:
            continue
        per_reference_scores.append(_gleu_against_one_reference(hyp_tokens, ref_tokens))

    if not per_reference_scores:
        return 0.0

    avg = sum(per_reference_scores) / len(per_reference_scores)
    return round(avg * 100, 2)
