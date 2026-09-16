# On-Device SLM Benchmark Guide

This guide explains how to run `model_benchmark.py`, what the numbers mean,
how to add a new model, and how to submit your results back to the project.

## 1. Install dependencies

```bash
pip install transformers torch psutil pandas
```

Works on Windows, macOS, and Linux with a standard Python 3.9+ install.
No GPU is required (the script runs on CPU with `float32`, which is closer
to what an on-device keyboard would actually experience).

## 2. Run the benchmark

```bash
cd prototype
python model_benchmark.py
```

The first run will download each model from HuggingFace (can take a few
minutes depending on model size and connection speed). Progress is printed
to the console as each model loads and as each prompt is tested.

Results are written to:

- `results/benchmark_results.json` — full detail, including every prompt's
  input/output text (useful for developers/debugging).
- `results/benchmark_results.csv` — condensed `Model, Latency(ms), RAM(MB),
  Tokens/sec` table, matching the format used in Prithvi's existing
  comparison sheet so rows can be pasted directly alongside his results.

If a model fails to download or load (missing repo, gated access, out of
disk space, etc.), the script logs the error, marks that model as
`"status": "failed"` in the JSON output, and continues with the next model
instead of crashing.

## 3. What each metric means

| Metric | Meaning |
|---|---|
| **Cold start load (ms)** | Time for `from_pretrained()` to load the model the first time in this run. |
| **Warm start load (ms)** | Time to load the same model again immediately after, once its files are in the OS disk cache. This measures disk-cache warmth, not a model kept resident in memory — see note below. |
| **Avg TTFT (ms)** | Average "time to first token" — how long until the model produces its very first output token. Approximated here via a 1-token `generate()` call. This is the number that most affects how "instant" the keyboard feels. |
| **Avg / Min / Max latency (ms)** | Time to generate a full response (up to 64 tokens) for a prompt. Averaged, plus the fastest/slowest prompt. |
| **Tokens/sec** | Generation throughput: total tokens generated across all prompts divided by total generation time. |
| **RAM before/after load (MB)** | Process resident memory (`psutil`) measured right before and right after the model is loaded. |
| **Model RAM (MB)** | `RAM after - RAM before` — the memory footprint attributable to the model itself. |
| **Model size on disk (MB)** | Size of the model's HuggingFace cache directory. Approximates the app/keyboard extension bundle size impact. |
| **Quality score (/100)** | A simple heuristic, *not* a substitute for human eval: +1/3 if the response isn't more than 2x the input length, +1/3 if it doesn't just repeat the prompt, +1/3 if the output is mostly English text. |

**Note on cold vs. warm start:** a real on-device keyboard loads its model
once per process lifetime, so "warm" here does not mean "already resident in
RAM" — it means "the files are warm in the OS page/disk cache," which is the
realistic best-case for a second cold process launch on a phone (e.g. after
the user reopens the keyboard). If you need true resident-memory latency
(model never unloaded), that is what `avg_latency_ms` per prompt already
reflects, since the model stays loaded across all prompts in a single run.

## 4. Adding a new model to test

Open `model_benchmark.py` and add an entry to `MODEL_CONFIGS`:

```python
MODEL_CONFIGS = [
    # ...existing entries...
    {
        "label": "MyNewModel-1B",
        "candidates": [
            "org/MyNewModel-1B-Instruct",   # tried first
            "org/MyNewModel-1B",            # fallback if the first 404s or is gated
        ],
    },
]
```

- `label` is the display name used in the console table and JSON/CSV output.
- `candidates` is a list of HuggingFace repo ids tried in order. This is
  useful when a model's exact release name isn't confirmed yet (as with
  Qwen3.5-0.8B at the time this script was written) — the script will fall
  back automatically and record which repo id actually worked as
  `resolved_model_id`.

No other code changes are required — the model is automatically included
in prompt testing, the comparison table, the device compatibility table,
and the JSON/CSV exports.

## 5. Submitting results as a PR

1. Run the benchmark on your machine and confirm `results/benchmark_results.json`
   and `results/benchmark_results.csv` were generated.
2. Note your machine specs in the PR description (CPU, RAM, OS) — these
   numbers are hardware-dependent and not directly comparable across
   different machines without that context.
3. Commit only your results files (don't commit large model files or
   HuggingFace cache directories — they should already be excluded by
   `.gitignore`/`.cache` conventions).
4. Open a PR against the `AI-Keyboard` repo with:
   - Title: `benchmark: results for <model name(s)> on <device/OS>`
   - A short summary of what you tested and any surprising findings
     (e.g. a model that's fast but scores poorly on Hinglish prompts).
5. Tag Bhavik and Prithvi for review so results can be merged into the
   shared comparison sheet.
