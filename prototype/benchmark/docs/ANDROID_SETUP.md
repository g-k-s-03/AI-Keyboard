# Running SLM Eval on Android (Termux)

## Prerequisites
- Android phone with at least 3GB RAM
- F-Droid app installed
- Good internet for initial model download

## Quick Start: One-Liner

If you just want a benchmark run with no manual setup, `termux_benchmark.sh`
detects the environment, installs missing dependencies, creates a venv, and
runs the benchmark for you.

```
pkg update && pkg upgrade -y
pkg install python git
curl -sL https://raw.githubusercontent.com/g-k-s-03/AI-Keyboard/prototype/model-benchmark/prototype/benchmark/termux_benchmark.sh | bash
```

> The URL above points at this branch on the development fork. Once this
> work merges into `AOSSIE-Org/AI-Keyboard`, update it to:
> `https://raw.githubusercontent.com/AOSSIE-Org/AI-Keyboard/main/prototype/benchmark/termux_benchmark.sh`

To choose a model, category, or per-prompt timeout instead of the defaults,
clone the repo and run the script directly with arguments:

```
git clone https://github.com/AOSSIE-Org/AI-Keyboard
cd AI-Keyboard/prototype/benchmark
./termux_benchmark.sh Qwen/Qwen2.5-0.5B-Instruct english_grammar 60
```

The rest of this guide walks through what that script automates, step by
step, for anyone who wants to understand or customize the setup.

## Step 1: Install Termux from F-Droid
Do NOT install from Google Play Store (outdated version).
https://f-droid.org/packages/com.termux/

## Step 2: Install System Packages
Open Termux and run:
```
pkg update && pkg upgrade -y
pkg install python git
```

## Step 3: Install PyTorch via Termux Repository
`pip install torch` does NOT work on Android ARM.
Use the official Termux repository instead:
```
pkg install tur-repo
pkg install python-torch python-numpy python-pandas
```

## Step 4: Install Remaining Dependencies
```
pip install transformers sacrebleu rouge-score click rich psutil
```

## Step 5: Clone and Install
```
git clone https://github.com/AOSSIE-Org/AI-Keyboard
cd AI-Keyboard/prototype/benchmark
pip install -e . --no-deps
```

## Step 6: Set Model Cache to External Storage (Optional)
If your phone has limited internal storage:
```
export HF_HOME=/sdcard/huggingface
```
Add to `~/.bashrc` to make permanent.

## Step 7: Verify Setup
```
slm-eval doctor
slm-eval device-info --device-name "YourPhone"
```

## Step 8: Run Benchmark
```
slm-eval test \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --device-name "Redmi12" \
  --runs 3 \
  --warmup 1
```

## Known Limitations on Android
- Battery and CPU temperature readings require root on Android 10+
  (returns None if unavailable, this is expected)
- First model download requires internet (~400MB for the 0.5B model)
- After download, benchmark runs fully offline
- Results are saved to `prototype/benchmark/results/`

## Step 9: Submit Your Results
Copy the `results/` folder contents and submit as a PR
to share your device benchmark with the community.

## Running on VirtualBox / a Linux VM

If you don't have an Android device handy, `termux_benchmark.sh` also runs
unmodified on a plain Linux VM — it detects it isn't Termux and falls back
to `apt` instead of `pkg`.

1. Create an Ubuntu 22.04 VM in VirtualBox (**4GB RAM minimum** — less than
   that and the OS and model loading will fight over memory and the run
   will thrash or OOM).
2. Inside the VM:
   ```
   sudo apt-get update && sudo apt-get install -y python3 git curl
   git clone https://github.com/AOSSIE-Org/AI-Keyboard
   cd AI-Keyboard/prototype/benchmark
   ./termux_benchmark.sh Qwen/Qwen2.5-0.5B-Instruct english_grammar 60
   ```
3. Results are written the same way, under `~/slm_eval_results/`.

This is a convenient way to sanity-check the benchmark pipeline itself (and
get a desktop-class latency baseline) before testing on real phone hardware.

## Expected Latency (community-contributed)

These numbers are **not yet measured** — contribute a run via Step 9 (or the
VirtualBox section above) to help fill this in.

| Model | Device | RAM | P50 latency | chrF |
|-------|--------|-----|-------------|------|
| Qwen2.5-0.5B | Termux (Android) | TBD | TBD | TBD |
| Qwen2.5-0.5B | VirtualBox Ubuntu | TBD | TBD | TBD |
| Qwen2.5-1.5B | Termux (Android) | TBD | TBD | TBD |

## Known Limitations
- iOS not tested (50MB RAM limit makes on-device SLM impractical)
- Human evaluation pending native speaker review (Bruno for Portuguese/Russian, Keshav for Hindi/Hinglish)
- Multiple device testing pending community contributions
- GPU benchmarking not implemented
- End-to-end ASR+SLM pipeline latency not yet measured
