"""Word Error Rate: (substitutions + deletions + insertions) / reference
word count, computed via a standard Levenshtein dynamic program over words.
"""


def calculate_wer(hypothesis: str, reference: str) -> float:
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    n = len(ref_words)
    if n == 0:
        return 1.0

    m = len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion from the reference
                    dp[i][j - 1],      # insertion into the hypothesis
                    dp[i - 1][j - 1],  # substitution
                )

    distance = dp[n][m]
    return distance / n


def calculate_wer_against_references(hypothesis: str, references: list) -> float:
    """Best-case (minimum) WER of `hypothesis` against any one of references."""
    if not references:
        return calculate_wer(hypothesis, "")
    return min(calculate_wer(hypothesis, ref) for ref in references)
