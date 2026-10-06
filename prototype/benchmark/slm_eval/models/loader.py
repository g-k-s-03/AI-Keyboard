from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

SUPPORTED_ARCHITECTURES = [
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "GPTNeoXForCausalLM",
    "PhiForCausalLM",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "SmolLMForCausalLM",
    "OLMoForCausalLM",
]


def load_model_and_tokenizer(model_id: str):
    """
    Load a HuggingFace model safely.
    Returns (tokenizer, model, error_message).
    error_message is None on success.
    Rejects multimodal/incompatible architectures.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model, None

    except Exception as e:
        error = str(e)
        if "multimodal" in error.lower() or "vision" in error.lower() or "Conditional" in error:
            return None, None, f"INCOMPATIBLE: {model_id} appears to be a multimodal model. Use a text-only CausalLM model."
        return None, None, f"LOAD ERROR: {error}"


def build_prompt(tokenizer, system_prompt: str, instruction: str, input_text: str) -> torch.Tensor:
    """
    Build prompt using chat template if available,
    fall back to raw string format if not.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{instruction}\n\nText: {input_text}"},
    ]
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            return inputs
    except Exception:
        pass

    raw = (
        f"{system_prompt}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Text: {input_text}\n\nOutput:"
    )
    return tokenizer(raw, return_tensors="pt")["input_ids"]
