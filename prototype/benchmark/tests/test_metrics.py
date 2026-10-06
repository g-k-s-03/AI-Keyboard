from slm_eval.metrics.bleu import (
    calculate_bleu, calculate_chrf, calculate_exact_match,
)
from slm_eval.utils.reproducibility import get_benchmark_env
from slm_eval.validation.output_validator import sanitize_output, validate_output


def test_bleu_perfect():
    assert calculate_bleu("hello world", ["hello world"], "english_grammar") == 100.0


def test_bleu_empty():
    assert calculate_bleu("", ["hello world"], "english_grammar") == 0.0


def test_chrf_basic():
    score = calculate_chrf("hello world", ["hello world"])
    assert score == 100.0


def test_exact_match_true():
    assert calculate_exact_match("Hello World", ["hello world"]) is True


def test_exact_match_false():
    assert calculate_exact_match("goodbye", ["hello world"]) is False


def test_sanitize_output_removes_prefix():
    output = "Sure! Here is the corrected text: I will be there."
    cleaned = sanitize_output(output)
    assert "Sure" not in cleaned
    assert "I will be there" in cleaned


def test_validate_output_hindi_success():
    prompt = {
        "script_expected": "devanagari",
        "min_output_tokens": 2,
        "input": "test",
        "category_id": "hindi_correction",
        "protected_values": [],
    }
    result = validate_output("मेरी रिपोर्ट कल तक नहीं भेजी गई थी।", prompt)
    assert result["script_ok"] is True
    assert result["echo_detected"] is False


def test_validate_output_echo_detected():
    prompt = {
        "script_expected": "latin",
        "min_output_tokens": 2,
        "input": "hello world",
        "category_id": "english_grammar",
        "protected_values": [],
    }
    result = validate_output("hello world", prompt)
    assert result["echo_detected"] is True
    assert result["task_success"] is False


def test_validate_protected_token_missing():
    prompt = {
        "script_expected": "latin",
        "min_output_tokens": 2,
        "input": "send to govind@example.com",
        "category_id": "protected_tokens",
        "protected_values": ["govind@example.com"],
    }
    result = validate_output("Please send the email.", prompt)
    assert result["protected_ok"] is False
    assert "govind@example.com" in result["missing_protected_tokens"]


def test_get_benchmark_env_keys():
    env = get_benchmark_env()
    assert isinstance(env, dict)
    for key in ("python_version", "torch_version", "transformers_version", "platform"):
        assert key in env
