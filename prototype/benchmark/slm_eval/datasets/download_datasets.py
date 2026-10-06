"""Download real NLP datasets from HuggingFace and merge them into
eval_dataset.json as extra prompts for the matching category.

Run as: python -m slm_eval.datasets.download_datasets
"""
import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "eval_dataset.json"

# (huggingface dataset id, split, category id, id prefix, sample count)
DOWNLOAD_SPECS = [
    ("dcavar/jfleg", "validation", "english_grammar", "en_ext", 15),
    ("nilc-nlp/assin2", "validation", "portuguese_correction", "pt_ext", 10),
    ("l3cube-pune/HinglishSenti", "train", "hinglish_to_english", "hi_ext", 10),
]

CATEGORY_LANG_META = {
    "english_grammar": {"input_lang": "en", "output_lang": "en", "script_expected": "latin"},
    "portuguese_correction": {"input_lang": "pt", "output_lang": "pt", "script_expected": "latin"},
    "hinglish_to_english": {"input_lang": "hi-Latn", "output_lang": "en", "script_expected": "latin"},
}


def _pair(values):
    """Return exactly 2 gold references, duplicating the first if needed."""
    values = [v for v in values if v]
    if not values:
        return ["", ""]
    if len(values) == 1:
        return [values[0], values[0]]
    return values[:2]


def _build_jfleg_prompts(ds, n, prefix):
    lang_meta = CATEGORY_LANG_META["english_grammar"]
    prompts = []
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        prompts.append({
            "id": f"{prefix}_{i + 1}",
            "instruction": "Fix the grammar in this sentence.",
            "input": row["sentence"],
            "gold": _pair(row["corrections"]),
            "min_output_tokens": 5,
            "already_correct": False,
            "source": "dcavar/jfleg",
            **lang_meta,
        })
    return prompts


def _build_assin2_prompts(ds, n, prefix):
    lang_meta = CATEGORY_LANG_META["portuguese_correction"]
    prompts = []
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        prompts.append({
            "id": f"{prefix}_{i + 1}",
            "instruction": "Corrija o português nesta frase.",
            "input": row["premise"],
            "gold": _pair([row["hypothesis"], row["hypothesis"]]),
            "min_output_tokens": 5,
            "already_correct": False,
            "source": "nilc-nlp/assin2",
            **lang_meta,
        })
    return prompts


def _build_hinglish_prompts(ds, n, prefix):
    lang_meta = CATEGORY_LANG_META["hinglish_to_english"]
    prompts = []
    for i, row in enumerate(ds.select(range(min(n, len(ds))))):
        prompts.append({
            "id": f"{prefix}_{i + 1}",
            "instruction": "Translate this Hinglish text to English.",
            "input": row["text"],
            "gold": _pair([row["text"], row["text"]]),
            "min_output_tokens": 5,
            "already_correct": False,
            "source": "l3cube-pune/HinglishSenti",
            "gold_is_placeholder": True,
            **lang_meta,
        })
    return prompts


BUILDERS = {
    "dcavar/jfleg": _build_jfleg_prompts,
    "nilc-nlp/assin2": _build_assin2_prompts,
    "l3cube-pune/HinglishSenti": _build_hinglish_prompts,
}


def _load_data():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_data(data):
    total = sum(len(cat["prompts"]) for cat in data["categories"])
    data["total_prompts"] = total
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_downloads() -> dict:
    """Download the configured HuggingFace datasets, convert them to our
    prompt format, and merge them into eval_dataset.json without
    duplicating prompts by input text. Never raises: failures are
    collected and reported instead so one bad dataset doesn't block
    the others."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: the 'datasets' package is not installed. "
              "Run: pip install datasets")
        return {"added": {}, "errors": ["datasets package not installed"]}

    data = _load_data()
    categories_by_id = {cat["id"]: cat for cat in data["categories"]}

    added_counts = {}
    errors = []

    for dataset_id, split, category_id, prefix, n_samples in DOWNLOAD_SPECS:
        category = categories_by_id.get(category_id)
        if category is None:
            errors.append(f"{dataset_id}: category '{category_id}' not found in eval_dataset.json")
            continue

        try:
            ds = load_dataset(dataset_id, split=split)
        except Exception as e:
            print(f"WARNING: failed to load '{dataset_id}' ({split}): {e}")
            errors.append(f"{dataset_id}: {e}")
            continue

        try:
            builder = BUILDERS[dataset_id]
            existing_inputs = {p["input"] for p in category["prompts"]}
            existing_prefixed_ids = {p["id"] for p in category["prompts"] if p["id"].startswith(prefix)}
            next_index = len(existing_prefixed_ids) + 1

            candidate_prompts = builder(ds, n_samples, prefix)

            added = 0
            for prompt in candidate_prompts:
                if prompt["input"] in existing_inputs:
                    continue
                prompt["id"] = f"{prefix}_{next_index}"
                next_index += 1
                category["prompts"].append(prompt)
                existing_inputs.add(prompt["input"])
                added += 1

            added_counts[category_id] = added_counts.get(category_id, 0) + added

        except Exception as e:
            print(f"WARNING: failed to convert/merge '{dataset_id}': {e}")
            errors.append(f"{dataset_id}: {e}")
            continue

    if added_counts:
        _save_data(data)

    print("\nDataset download summary:")
    if not added_counts:
        print("  No new prompts were added.")
    for category_id, count in added_counts.items():
        print(f"  {category_id}: +{count} prompts")
    if errors:
        print("\nErrors encountered:")
        for e in errors:
            print(f"  - {e}")

    return {"added": added_counts, "errors": errors}


if __name__ == "__main__":
    run_downloads()
