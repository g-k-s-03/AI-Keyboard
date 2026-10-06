# Contributing to SLM Eval

## Running Tests
```
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Adding a New Model
1. Verify the model is text-only CausalLM (not multimodal).
2. Test with: `slm-eval test --model YOUR/MODEL --lang english`
3. If it works, add it to `configs/keyboard_usecase.yaml`.
4. Add it to the `list-models` command in `cli.py`.

## Adding Dataset Prompts
1. Edit `slm_eval/datasets/eval_dataset.json`.
2. Add all required fields including `input_lang`, `output_lang`,
   `script_expected`, `min_output_tokens`, `protected_values`.
3. Add at least 2 gold reference answers.
4. Run: `slm-eval validate-dataset`
5. Run: `python -m pytest tests/test_dataset.py`

## Submitting Device Results
1. Run the full benchmark on your device.
2. Copy `results/*.csv` and `results/*.json`.
3. Submit as a PR with the device name in the title.

## Human Evaluation Note
Automated metrics (BLEU, chrF, ROUGE) cannot fully judge
text quality for rewriting tasks. Results should be
supplemented with human evaluation before making final
model decisions. See the README for evaluation methodology.
