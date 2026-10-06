import sacrebleu

LANG_BLEU_TOKENIZER = {
    "hinglish_to_english": "13a",
    "english_grammar": "13a",
    "professional_rewrite": "13a",
    "error_correction": "13a",
    "meaning_preservation": "13a",
    "protected_tokens": "13a",
    "hindi_correction": "none",
    "russian_correction": "none",
    "portuguese_correction": "flores101",
}

LANG_ROUGE_STEMMER = {
    "english_grammar": True,
    "professional_rewrite": True,
    "error_correction": True,
    "meaning_preservation": True,
    "protected_tokens": True,
    "hinglish_to_english": False,
    "hindi_correction": False,
    "russian_correction": False,
    "portuguese_correction": False,
}


def calculate_bleu(hypothesis: str, gold_references: list, category: str = "english_grammar") -> float:
    if not hypothesis or not gold_references:
        return 0.0
    tokenize = LANG_BLEU_TOKENIZER.get(category, "13a")
    try:
        result = sacrebleu.corpus_bleu(
            [hypothesis],
            [[ref] for ref in gold_references],
            tokenize=tokenize,
            use_effective_order=True,
        )
        return round(result.score, 2)
    except Exception:
        return 0.0


def calculate_chrf(hypothesis: str, gold_references: list) -> float:
    if not hypothesis or not gold_references:
        return 0.0
    try:
        result = sacrebleu.corpus_chrf(
            [hypothesis],
            [[ref] for ref in gold_references],
        )
        return round(result.score, 2)
    except Exception:
        return 0.0


def calculate_rouge(hypothesis: str, gold_references: list, category: str = "english_grammar") -> dict:
    try:
        from rouge_score import rouge_scorer
        use_stemmer = LANG_ROUGE_STEMMER.get(category, False)
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=use_stemmer,
        )
        r1 = max(scorer.score(ref, hypothesis)["rouge1"].fmeasure for ref in gold_references)
        r2 = max(scorer.score(ref, hypothesis)["rouge2"].fmeasure for ref in gold_references)
        rl = max(scorer.score(ref, hypothesis)["rougeL"].fmeasure for ref in gold_references)
        return {
            "rouge1": round(r1 * 100, 2),
            "rouge2": round(r2 * 100, 2),
            "rougeL": round(rl * 100, 2),
        }
    except Exception:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_exact_match(hypothesis: str, gold_references: list) -> bool:
    hyp_norm = hypothesis.strip().lower()
    return any(hyp_norm == ref.strip().lower() for ref in gold_references)


def calculate_all_metrics(hypothesis: str, gold_references: list, category: str) -> dict:
    return {
        "bleu": calculate_bleu(hypothesis, gold_references, category),
        "chrf": calculate_chrf(hypothesis, gold_references),
        "rouge": calculate_rouge(hypothesis, gold_references, category),
        "exact_match": calculate_exact_match(hypothesis, gold_references),
    }
