import json
from pathlib import Path

DATASET_PATH = Path(__file__).parent.parent / "slm_eval" / "datasets" / "eval_dataset.json"


def test_dataset_loads():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    assert "categories" in data
    assert len(data["categories"]) > 0


def test_all_prompts_have_required_fields():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    required = ["id", "input", "gold", "instruction",
                "input_lang", "output_lang", "script_expected"]
    for cat in data["categories"]:
        for prompt in cat["prompts"]:
            for field in required:
                assert field in prompt, f"{cat['id']}/{prompt['id']} missing {field}"


def test_all_prompts_have_two_gold_references():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for cat in data["categories"]:
        for prompt in cat["prompts"]:
            assert len(prompt["gold"]) >= 2, \
                f"{cat['id']}/{prompt['id']} needs 2+ gold references"


def test_total_prompt_count():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    total = sum(len(cat["prompts"]) for cat in data["categories"])
    assert total == data["total_prompts"]


def test_already_correct_prompts_exist():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    already_correct = [
        prompt
        for cat in data["categories"]
        for prompt in cat["prompts"]
        if prompt.get("already_correct") is True
    ]
    assert len(already_correct) >= 3


def test_source_field_exists_on_extended_prompts():
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    sourced = [
        prompt
        for cat in data["categories"]
        for prompt in cat["prompts"]
        if "source" in prompt
    ]
    for prompt in sourced:
        assert isinstance(prompt["source"], str) and prompt["source"] != ""
