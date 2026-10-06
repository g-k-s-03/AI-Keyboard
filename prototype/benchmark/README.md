# SLM Eval - On-Device Language Model Benchmark

A CLI tool for benchmarking Small Language Models (SLMs) on both
PC and Android phones. Built for the AOSSIE AI Keyboard project to
evaluate multilingual grammar correction and translation quality
under real mobile hardware constraints.

## Install

```
pip install -e ".[dev]"
```

## Quick Start

```
# Verify dependencies
slm-eval doctor

# Check your device
slm-eval device-info

# Inspect the eval dataset
slm-eval dataset
slm-eval validate-dataset

# Test one model
slm-eval test --model Qwen/Qwen2.5-0.5B-Instruct --device-name "MyPC"

# Test on a specific language subset
slm-eval test --model Qwen/Qwen2.5-0.5B-Instruct --lang hindi

# Compare multiple models
slm-eval benchmark Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct --device-name "Redmi12"

# List recommended models
slm-eval list-models
```

## What It Tests

31 prompts across 9 categories (`slm_eval/datasets/eval_dataset.json`, v1.1.0):

- Hinglish to English translation
- Hindi same-language correction (Devanagari output)
- Portuguese same-language correction
- Russian same-language correction (Cyrillic output)
- English grammar fix
- Professional rewrite
- Wrong input / spelling error correction
- Meaning preservation (names, times, dates must survive correction)
- Protected token preservation (emails, URLs, OTPs, @mentions)

This is a pilot evaluation set. Automated metrics (BLEU, chrF, ROUGE)
cannot fully judge rewriting quality on their own — results should be
supplemented with human evaluation before making model decisions.

## Metrics

- BLEU, chrF and ROUGE-1/2/L against gold references
- Exact match
- Latency: average, p50, p95 across multiple runs per prompt (with a warm-up pass)
- RAM usage and cold-start time
- Tokens per second
- Output validation: script correctness (Latin/Devanagari/Cyrillic),
  echo detection, contamination detection, protected-token preservation
- Device profile (budget/midrange/flagship mobile, or desktop)
- Battery and CPU temperature where readable (Android, best-effort)

## Android Setup

See [docs/ANDROID_SETUP.md](docs/ANDROID_SETUP.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for running tests, adding
models/prompts, and submitting device results.
