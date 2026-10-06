# Running SLM Eval on Android (Termux)

## Prerequisites
- Android phone with at least 3GB RAM
- F-Droid app installed
- Good internet for initial model download

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
