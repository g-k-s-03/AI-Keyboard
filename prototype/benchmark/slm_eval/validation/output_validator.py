CONTAMINATION_PREFIXES = [
    "sure", "here is", "here's", "certainly", "of course",
    "sure!", "here you go", "the corrected", "translation:",
    "corrected text:", "output:", "answer:",
]


def sanitize_output(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    lower = text.lower()
    for prefix in CONTAMINATION_PREFIXES:
        if lower.startswith(prefix):
            colon_idx = text.find(":")
            newline_idx = text.find("\n")
            if colon_idx != -1 and colon_idx < 80:
                text = text[colon_idx + 1:].strip()
            elif newline_idx != -1 and newline_idx < 80:
                text = text[newline_idx + 1:].strip()
            break
    return text.strip()


def get_script_ratio(text: str, script: str) -> float:
    if not text:
        return 0.0
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return 0.0
    if script == "devanagari":
        count = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    elif script == "cyrillic":
        count = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    elif script == "latin":
        count = sum(1 for c in text if c.isascii() and c.isalpha())
    else:
        return 1.0
    return count / total_alpha


def validate_output(output: str, prompt: dict) -> dict:
    script = prompt.get("script_expected", "latin")
    protected = prompt.get("protected_values", [])

    script_ratio = get_script_ratio(output, script)
    script_ok = script_ratio >= 0.6

    length_ok = len(output.split()) >= prompt.get("min_output_tokens", 1)

    echo_detected = output.strip().lower() == prompt["input"].strip().lower()

    contamination_detected = any(
        output.lower().startswith(p) for p in CONTAMINATION_PREFIXES
    )

    protected_ok = True
    missing_protected = []
    for token in protected:
        if token.lower() not in output.lower():
            protected_ok = False
            missing_protected.append(token)

    task_success = (
        script_ok
        and length_ok
        and not echo_detected
        and not contamination_detected
        and protected_ok
    )

    return {
        "task_success": task_success,
        "script_ratio": round(script_ratio, 3),
        "script_ok": script_ok,
        "length_ok": length_ok,
        "echo_detected": echo_detected,
        "contamination_detected": contamination_detected,
        "protected_ok": protected_ok,
        "missing_protected_tokens": missing_protected,
    }
