# prototype/

Feasibility scripts for choosing the on-device small language model (SLM)
that will power the AI Keyboard's grammar-fix, Hinglish-translation, and
tone-rewrite features.

This folder is **not** part of the shipped Flutter app — it's a Python-based
testbed for evaluating candidate models on a desktop/CI machine before any
model is committed to for on-device (Android/iOS) integration.

## Contents

- `model_benchmark.py` — benchmarks candidate SLMs (currently
  Qwen2.5-0.5B-Instruct and Qwen3.5-0.8B, with a fallback if the latter
  isn't published on HuggingFace yet) against real keyboard-style prompts,
  measuring latency, RAM, disk size, throughput, and a heuristic quality
  score.
- `BENCHMARK_GUIDE.md` — how to install, run, extend, and submit benchmark
  results.
- `results/` — output directory for `benchmark_results.json` and
  `benchmark_results.csv`. Kept in git via `.gitkeep`; generated result
  files are expected to be added per-PR when contributors submit their
  numbers (see the guide for what to include, like machine specs).

## Why this exists

Model choice for the keyboard needs to balance latency, memory footprint
(especially for the 50MB iOS keyboard extension limit), and output quality
for Hinglish/grammar/tone use cases — this can't be judged from spec sheets
alone. This benchmark gives the team comparable, reproducible numbers to
inform that decision, building on the manual testing Prithvi already did on
MicroLlama-300M and the SmolLM2 family.
