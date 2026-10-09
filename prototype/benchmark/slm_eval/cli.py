import io
import sys

import click
from rich.console import Console
from rich.table import Table

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

console = Console()


@click.group()
@click.version_option("0.2.0", prog_name="slm-eval")
def main():
    """SLM Eval - Benchmark Small Language Models on PC and Android."""
    pass


@main.command()
def doctor():
    """Check that all dependencies are installed and working."""
    console.print("\n[bold cyan]SLM Eval Doctor[/bold cyan]\n")
    checks = []

    try:
        import torch
        checks.append(("torch", True, torch.__version__))
    except ImportError:
        checks.append(("torch", False, "not installed - run: pip install torch"))

    try:
        import transformers
        checks.append(("transformers", True, transformers.__version__))
    except ImportError:
        checks.append(("transformers", False, "not installed"))

    try:
        import sacrebleu
        checks.append(("sacrebleu", True, sacrebleu.__version__))
    except ImportError:
        checks.append(("sacrebleu", False, "not installed"))

    try:
        import rouge_score  # noqa: F401
        checks.append(("rouge-score", True, "ok"))
    except ImportError:
        checks.append(("rouge-score", False, "not installed"))

    try:
        import psutil
        checks.append(("psutil", True, psutil.__version__))
    except ImportError:
        checks.append(("psutil", False, "not installed"))

    try:
        import pandas
        checks.append(("pandas", True, pandas.__version__))
    except ImportError:
        checks.append(("pandas", False, "not installed"))

    table = Table(title="Dependency Check")
    table.add_column("Package", style="cyan")
    table.add_column("Status")
    table.add_column("Info", style="dim")

    all_ok = True
    for name, ok, info in checks:
        status = "[green]OK[/green]" if ok else "[red]MISSING[/red]"
        if not ok:
            all_ok = False
        table.add_row(name, status, info)

    console.print(table)
    if all_ok:
        console.print("\n[green]All dependencies OK. Ready to benchmark.[/green]")
    else:
        console.print("\n[red]Fix missing dependencies before running benchmarks.[/red]")
        console.print("Run: pip install -r requirements.txt")


@main.command()
@click.option("--device-name", default="unknown")
def device_info(device_name):
    """Show current device information."""
    from slm_eval.metrics.device_metrics import get_device_info
    info = get_device_info(device_name)
    table = Table(title="Device Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    for key, value in info.items():
        table.add_row(str(key), str(value) if value is not None else "N/A")
    console.print(table)


@main.command()
def dataset():
    """Show dataset summary."""
    import json
    from pathlib import Path
    dataset_path = Path(__file__).parent / "datasets" / "eval_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"\n[bold]Dataset v{data['version']}[/bold]")
    console.print(f"Total prompts: [cyan]{data['total_prompts']}[/cyan]\n")
    console.print(f"[dim]{data.get('note', '')}[/dim]\n")

    table = Table(title="Categories")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Prompts", style="green")

    for cat in data["categories"]:
        table.add_row(cat["id"], cat["name"], str(len(cat["prompts"])))
    console.print(table)


@main.command()
@click.option("--model", default=None, help="HuggingFace model ID")
@click.option("--lang", default="all",
              type=click.Choice(["all", "hindi", "english", "portuguese", "russian", "hinglish"]),
              help="Language filter")
@click.option("--device-name", default="unknown", help="Your device name")
@click.option("--runs", default=3, help="Inference runs per prompt for reliable latency")
@click.option("--warmup", default=1, help="Warm-up runs before measuring")
@click.option("--max-new-tokens", default=100)
@click.option("--output-dir", default="results")
@click.option("--category", default=None, help="Restrict to a single dataset category id")
@click.option("--verbose", is_flag=True, default=False, help="Print input/output/gold per prompt")
@click.option("--all-models", is_flag=True, default=False, help="Run every model in RECOMMENDED_MODELS")
@click.option("--input-text", default=None, help="Run on this text instead of the dataset")
@click.option("--input-file", default=None, type=click.Path(exists=True, dir_okay=False),
              help="Run on each non-empty line of this file instead of the dataset")
def test(model, lang, device_name, runs, warmup, max_new_tokens, output_dir, category, verbose,
         all_models, input_text, input_file):
    """Test a model on the benchmark dataset (or custom input)."""
    from slm_eval.metrics.device_metrics import get_device_info
    from slm_eval.benchmarks.slm_benchmark import (
        RECOMMENDED_MODELS,
        prompt_model_selection,
        print_comparison_table,
        run_all_models_benchmark,
        run_custom_inputs,
        run_slm_benchmark,
    )
    from slm_eval.reporters.csv_reporter import save_csv
    from slm_eval.reporters.json_reporter import save_json

    if input_text and input_file:
        raise click.UsageError("--input-text and --input-file are mutually exclusive.")

    if input_text or input_file:
        if not model:
            raise click.UsageError("--model is required when using --input-text/--input-file.")

        if input_text:
            texts = [input_text]
        else:
            with open(input_file, encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            if not texts:
                raise click.UsageError(f"No non-empty lines found in {input_file}.")

        console.print(f"\n[bold cyan]SLM Eval[/bold cyan] - Custom input on [yellow]{model}[/yellow]")
        with console.status(f"Running {model} on {len(texts)} input(s)..."):
            result = run_custom_inputs(model, texts, max_new_tokens=max_new_tokens)

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            return

        if len(texts) == 1:
            console.print(f"\n[green]Output:[/green] {result['outputs'][0]}")
        else:
            console.print()
            for i, (inp, out) in enumerate(zip(result["inputs"], result["outputs"]), start=1):
                console.print(f"[dim]{i}. Input:[/dim]  {inp}")
                console.print(f"   [green]Output:[/green] {out}\n")
        return

    if not model and not all_models:
        selection = prompt_model_selection()
        if selection is None:
            raise click.UsageError(
                "No model specified. Pass --model <hf-id> or --all-models "
                "(interactive selection requires a terminal)."
            )
        kind, value = selection
        if kind == "all":
            all_models = True
        else:
            model = value

    if all_models:
        console.print(f"\n[bold cyan]SLM Eval[/bold cyan] - Running {len(RECOMMENDED_MODELS)} models in sequence\n")
        outcome = run_all_models_benchmark(
            lang_filter=lang,
            category_filter=category,
            runs=runs,
            warmup_runs=warmup,
            max_new_tokens=max_new_tokens,
            device_name=device_name,
            output_dir=output_dir,
        )
        console.print(f"[green]Saved combined comparison:[/green] {outcome['combined_path']}")
        return

    console.print(f"\n[bold cyan]SLM Eval[/bold cyan] - Testing [yellow]{model}[/yellow]")
    device = get_device_info(device_name)
    console.print(
        f"Device: [green]{device['device_profile']}[/green] | "
        f"RAM: [green]{device['total_ram_gb']}GB[/green] | "
        f"Platform: [green]{device['platform']}[/green] | "
        f"Runs/prompt: [green]{runs}[/green]\n"
    )

    with console.status(f"Loading and benchmarking {model}..."):
        result = run_slm_benchmark(
            model,
            lang_filter=lang,
            category_filter=category,
            runs=runs,
            warmup_runs=warmup,
            max_new_tokens=max_new_tokens,
            device_name=device_name,
        )

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    if verbose:
        for r in result["prompt_results"]:
            console.print(f"\n[dim]--- {r.get('prompt_id')} ({r.get('category')}) ---[/dim]")
            console.print(f"[cyan]Input:[/cyan] {r.get('input')}")
            console.print(f"[green]Output:[/green] {r.get('sanitized_output', r.get('error', ''))}")
            gold = r.get("gold") or []
            console.print(f"[yellow]Gold:[/yellow] {gold[0] if gold else ''}")

    table = Table(title=f"Results: {model}")
    table.add_column("Category", style="cyan")
    table.add_column("BLEU", style="green")
    table.add_column("chrF", style="blue")
    table.add_column("GLEU", style="green")
    table.add_column("WER", style="red")
    table.add_column("Success%", style="magenta")

    for cat, metrics in result["per_category_metrics"].items():
        table.add_row(
            cat,
            f"{metrics['avg_bleu']:.1f}",
            f"{metrics['avg_chrf']:.1f}",
            f"{metrics['avg_gleu']:.1f}",
            f"{metrics['avg_wer']:.2f}",
            f"{metrics['success_rate']:.0f}%",
        )
    table.add_row(
        "[bold]OVERALL[/bold]",
        f"[bold]{result['avg_bleu']:.1f}[/bold]",
        f"[bold]{result['avg_chrf']:.1f}[/bold]",
        f"[bold]{result['avg_gleu']:.1f}[/bold]",
        f"[bold]{result['avg_wer']:.2f}[/bold]",
        f"[bold]{result['task_success_rate']:.0f}%[/bold]",
    )
    console.print(table)

    console.print(
        f"\nRAM: [yellow]{result['model_ram_mb']}MB[/yellow] | "
        f"Cold start: [yellow]{result['cold_start_ms']}ms[/yellow] | "
        f"Avg latency: [yellow]{result['avg_latency_ms']}ms[/yellow]"
    )

    csv_file = save_csv(result, device, output_dir)
    json_file = save_json(result, device, output_dir)
    console.print(f"\n[green]Saved CSV:[/green] {csv_file}")
    console.print(f"[green]Saved JSON:[/green] {json_file}")


@main.command()
@click.argument("models", nargs=-1, required=True)
@click.option("--lang", default="all")
@click.option("--device-name", default="unknown")
@click.option("--runs", default=3)
@click.option("--warmup", default=1)
@click.option("--max-new-tokens", default=100)
@click.option("--output-dir", default="results")
def benchmark(models, lang, device_name, runs, warmup, max_new_tokens, output_dir):
    """Benchmark and compare multiple models."""
    from slm_eval.metrics.device_metrics import get_device_info
    from slm_eval.benchmarks.slm_benchmark import run_slm_benchmark
    from slm_eval.reporters.csv_reporter import save_csv
    from slm_eval.reporters.json_reporter import save_json

    device = get_device_info(device_name)
    all_results = []

    for model in models:
        console.print(f"\n[cyan]Testing: {model}[/cyan]")
        result = run_slm_benchmark(
            model,
            lang_filter=lang,
            runs=runs,
            warmup_runs=warmup,
            max_new_tokens=max_new_tokens,
            device_name=device_name,
        )
        if "error" not in result:
            all_results.append(result)
            save_csv(result, device, output_dir)
            save_json(result, device, output_dir)
        else:
            console.print(f"[red]Failed: {result['error']}[/red]")

    if not all_results:
        console.print("[red]No successful results.[/red]")
        return

    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("BLEU", style="green")
    table.add_column("chrF", style="blue")
    table.add_column("Success%", style="magenta")
    table.add_column("Avg Latency", style="yellow")
    table.add_column("RAM", style="red")

    for r in sorted(all_results, key=lambda x: x["avg_chrf"], reverse=True):
        table.add_row(
            r["model_id"].split("/")[-1],
            f"{r['avg_bleu']:.1f}",
            f"{r['avg_chrf']:.1f}",
            f"{r['task_success_rate']:.0f}%",
            f"{r['avg_latency_ms']}ms",
            f"{r['model_ram_mb']}MB",
        )
    console.print(table)


@main.command("list-models")
def list_models():
    """Show recommended models to test."""
    models = [
        ("Qwen/Qwen2.5-0.5B-Instruct", "~400MB", "Verified working"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "~1.5GB", "Verified working"),
        ("HuggingFaceTB/SmolLM2-360M-Instruct", "~200MB", "Verified working"),
        ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "~1.5GB", "Verified working"),
        ("microsoft/Phi-3-mini-4k-instruct", "~2.4GB", "Verified working"),
    ]
    table = Table(title="Recommended Models (text-only CausalLM)")
    table.add_column("Model ID", style="cyan")
    table.add_column("Approx Size", style="yellow")
    table.add_column("Notes", style="green")
    for model, size, notes in models:
        table.add_row(model, size, notes)

    console.print(table)
    console.print("\n[dim]Multimodal / vision-language models are not supported.[/dim]")
    console.print("[dim]Use text-only CausalLM models only.[/dim]")


@main.command("download-datasets")
def download_datasets():
    """Download real NLP datasets from HuggingFace and merge them into the eval dataset."""
    from slm_eval.datasets.download_datasets import run_downloads
    run_downloads()


@main.command("validate-dataset")
def validate_dataset():
    """Validate the eval dataset for required fields."""
    import json
    from pathlib import Path

    dataset_path = Path(__file__).parent / "datasets" / "eval_dataset.json"
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    required_fields = ["id", "input", "gold", "instruction", "input_lang",
                        "output_lang", "script_expected", "min_output_tokens"]
    errors = []

    for cat in data["categories"]:
        for prompt in cat["prompts"]:
            for field in required_fields:
                if field not in prompt:
                    errors.append(f"{cat['id']}/{prompt['id']} missing: {field}")
            if len(prompt.get("gold", [])) < 2:
                errors.append(f"{cat['id']}/{prompt['id']}: needs 2+ gold references")

    actual_total = sum(len(cat["prompts"]) for cat in data["categories"])
    if actual_total != data.get("total_prompts"):
        errors.append(
            f"total_prompts metadata ({data.get('total_prompts')}) "
            f"does not match actual count ({actual_total})"
        )

    if errors:
        console.print("[red]Dataset validation FAILED:[/red]")
        for e in errors:
            console.print(f"  [red]- {e}[/red]")
    else:
        console.print(
            f"[green]Dataset valid! {data['total_prompts']} prompts "
            f"across {len(data['categories'])} categories.[/green]"
        )


if __name__ == "__main__":
    main()
