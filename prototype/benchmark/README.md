# SLM Eval — Mobile Keyboard Benchmark

## 1. What this is

SLM Eval is a mobile-first benchmark for evaluating small language models
(SLMs) on the kinds of tasks an AI keyboard actually needs: grammar
correction, tone/professional rewriting, Hinglish-to-English translation,
and same-language correction in Hindi, Portuguese, and Russian. It's built
for the AOSSIE AI Keyboard project and designed to run anywhere a keyboard
would actually ship — Android under Termux, a resource-constrained
VirtualBox VM, or a normal desktop — so model choices are measured against
real device constraints (RAM, latency, offline-first) instead of just
leaderboard scores.

## 2. Metrics

| Metric | What it measures | Good score |
|---|---|---|
| BLEU | N-gram precision of the output against gold references | Higher is better, 0–100 |
| chrF | Character n-gram F-score against gold references (more tolerant of small morphological differences than BLEU) | Higher is better, 0–100 |
| GLEU | Precision *and* recall of n-grams (1–4), penalizing both under- and over-generation — more suited to correction tasks than plain BLEU | Higher is better, 0–100 |
| WER | Word Error Rate: edit distance to the reference divided by reference word count | Lower is better, 0.0 = perfect, can exceed 1.0 for very bad output |
| exact_match | Whether the output exactly matches a gold reference (case-insensitive) | Boolean; rare to hit for free-form rewriting, more meaningful for already-correct pass-through prompts |
| task_success | Composite pass/fail: correct script (Latin/Devanagari/Cyrillic), long enough, not an echo of the input, no chat-assistant preamble, protected tokens (emails/URLs/OTPs/@mentions) intact | Boolean; this is the metric to watch first — it catches failures BLEU/chrF alone can miss |

ROUGE-1/2/L are also computed per prompt (see `slm_eval/metrics/bleu.py`)
but aren't part of the headline set above.

## 3. Dataset

168 prompts across 11 categories (`slm_eval/datasets/eval_dataset.json`):

| Category | Prompts | Description |
|---|---|---|
| `english_grammar` | 99 | English grammar and spelling correction |
| `hindi_keyboard` | 20 | Real Hindi text, used as already-correct pass-through input |
| `hinglish_to_english` | 14 | Romanized Hindi+English mixed text translated to English |
| `portuguese_paraphrase` | 10 | Natural paraphrases of the same scene (semantic similarity, not grammar errors) |
| `error_correction` | 6 | Real-world typos, autocorrect mistakes, and SMS abbreviations |
| `protected_tokens` | 4 | Emails, URLs, OTPs, and @mentions that must survive correction unchanged |
| `hindi_correction` | 3 | Romanized Hindi corrected and returned in Devanagari script |
| `portuguese_correction` | 3 | Informal Portuguese corrected to proper Portuguese |
| `russian_correction` | 3 | Romanized Russian corrected and returned in Cyrillic script |
| `professional_rewrite` | 3 | Casual messages rewritten in a professional tone |
| `meaning_preservation` | 3 | Grammar fixes that must not alter names, times, dates, or amounts |

Most prompts are handwritten; five categories are supplemented with real
data pulled from HuggingFace via `slm-eval download-datasets`
(`slm_eval/datasets/download_datasets.py`):

- **JFLEG** (`jhu-clsp/jfleg`) and **wi_locness** (`martinsr/wi_locness`) → `english_grammar`
- **findnitai/english-to-hinglish** → `hinglish_to_english`
- **wikimedia/wikipedia** (Hindi, `20231101.hi`) → `hindi_keyboard`
- **ASSIN2** (`nilc-nlp/assin2`) → `portuguese_paraphrase`

This is a pilot evaluation set. Automated metrics can't fully judge
rewriting quality on their own — results should be supplemented with
human evaluation before making model decisions.

## 4. How to run (desktop)

**Prerequisites:** Python 3.9+ (see `setup.py`)

```
pip install -e ".[dev]"
```

**Commands:**

```
slm-eval test --model Qwen/Qwen2.5-0.5B-Instruct
slm-eval test --model Qwen/Qwen2.5-0.5B-Instruct --lang hindi
slm-eval test --model Qwen/Qwen2.5-0.5B-Instruct --category english_grammar
python -m pytest tests/ -v
```

Other useful commands: `slm-eval doctor` (dependency check), `slm-eval
device-info`, `slm-eval dataset` / `validate-dataset`, `slm-eval
download-datasets`, `slm-eval benchmark <model1> <model2> ...` (side-by-side
comparison), `slm-eval list-models` (verified-working model suggestions).

## 5. How to run (Android / Termux)

One-liner (detects the environment, installs missing dependencies, sets up
a venv, and runs the benchmark for you):

```
pkg update && pkg upgrade -y
pkg install python git
curl -fsSL https://raw.githubusercontent.com/g-k-s-03/AI-Keyboard/prototype/model-benchmark/prototype/benchmark/termux_benchmark.sh | bash
```

> That URL points at this development branch on the fork it was built on.
> Once this work merges into `AOSSIE-Org/AI-Keyboard`, switch it to
> `https://raw.githubusercontent.com/AOSSIE-Org/AI-Keyboard/main/prototype/benchmark/termux_benchmark.sh`.

`termux_benchmark.sh` also runs unmodified on a plain Linux VM (e.g.
VirtualBox with Ubuntu 22.04, **4GB RAM minimum**) — it detects it isn't
Termux and falls back to `apt` instead of `pkg`. See
[docs/ANDROID_SETUP.md](docs/ANDROID_SETUP.md) for the full manual
walkthrough, VirtualBox setup, and known limitations.

## 6. Output

Each run writes a JSON result with:

- `device_info` — environment (termux/virtualbox/linux/desktop), CPU, RAM, Android version where applicable
- `model_id` — the HuggingFace model that was benchmarked
- `quantization` — `4bit` (when `bitsandbytes` is available) or `float32`
- `per_prompt_results` — one entry per prompt: input, output, gold references, metrics, latency breakdown (`prefill_ms`, `gen_ms`, `ms_per_token`)
- `summary` — aggregated scores:
  - `bleu`, `chrf`, `gleu`, `wer`
  - `task_success_rate`
  - `avg_ms_per_token`
  - `p50_latency_ms` (and `p95_latency_ms`)

Desktop runs via `slm-eval test` save to `results/`; Termux/VirtualBox runs
via `termux_benchmark.sh` save to `~/slm_eval_results/`.

## 7. Adding a new model

For the `slm-eval` CLI (this package), there's no config list to edit —
just point it at any HuggingFace model id:

```
slm-eval test --model <hf-org>/<hf-model>
```

1. Check the model is a text-only `CausalLM` (not multimodal/vision) —
   `slm-eval doctor` and a quick `huggingface_hub.HfApi().model_info(...)`
   tag check will tell you.
2. Run it: `slm-eval test --model <hf-org>/<hf-model> --category english_grammar`
3. If it's a good general recommendation, add it to the table in
   `list_models()` in `slm_eval/cli.py`.

> Note: `prototype/model_benchmark.py` is a separate, standalone legacy
> benchmark script (not part of the `slm_eval` package) that *does* use a
> `MODEL_CONFIGS` list — if you're working on that script specifically, add
> an entry there and verify the Hub id exists via `HfApi` before running it.

## 8. Project structure

```
prototype/benchmark/
├── slm_eval/
│   ├── benchmarks/
│   │   └── slm_benchmark.py       # desktop benchmark runner
│   ├── datasets/
│   │   ├── eval_dataset.json      # the 168-prompt dataset
│   │   └── download_datasets.py   # pulls in real HF data per category
│   ├── metrics/
│   │   ├── bleu.py                # BLEU, chrF, ROUGE, exact_match
│   │   ├── gleu.py                # GLEU
│   │   └── wer.py                 # WER
│   ├── models/
│   │   └── loader.py              # model/tokenizer loading, chat templates
│   ├── validation/
│   │   └── output_validator.py    # task_success, echo/contamination checks
│   ├── utils/
│   │   └── reproducibility.py     # seeding, environment capture
│   ├── termux_runner.py           # mobile-friendly runner (timeout, quantization)
│   └── cli.py                     # `slm-eval` entry point
├── tests/
├── termux_benchmark.sh            # Termux/VirtualBox setup + run script
└── Makefile
```
