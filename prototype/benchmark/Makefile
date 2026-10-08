MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
DEVICE ?= local

.PHONY: test benchmark termux download-data

test:
	pytest tests/ -v

benchmark:
	slm-eval test --model $(MODEL) --device-name "$(DEVICE)"

termux:
	@echo "Termux install instructions:"
	@echo "  pkg update && pkg upgrade -y"
	@echo "  pkg install python git"
	@echo "  curl -sL https://raw.githubusercontent.com/g-k-s-03/AI-Keyboard/prototype/model-benchmark/prototype/benchmark/termux_benchmark.sh | bash"
	@echo ""
	@echo "Or clone the repo and run directly:"
	@echo "  ./termux_benchmark.sh <model> <category> <timeout_per_prompt>"

download-data:
	slm-eval download-datasets
