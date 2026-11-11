"""Last \\boxed{} answer, compared with the gold answer (planning.md, step 4).

Gold answers are strings: integers for GSM8K and AIME, LaTeX for MATH.
A cleaned exact match is tried first; otherwise math-verify decides
(fractions vs decimals, reordered expressions, intervals, tuples).
"""
import re

_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")
# words in \text{...}: units like "dollars", "cm", "inches" (but not \text{(A)})
_UNIT_TEXT = re.compile(r"\\(?:text|textbf|mathrm|mbox)\s*\{\s*[A-Za-z][A-Za-z ]*\}")
_NOISE = re.compile(r"\\[,;!: ]|\\\$|\\%|%|\^\s*\\circ|\^\{\\circ\}|\\left|\\right|[\s$]")
_NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def last_boxed(text):
    """Content of the last complete \\boxed{...} in text, or None."""
    if text is None:
        return None
    for m in reversed(list(_BOXED.finditer(text))):
        depth, start = 1, m.end()
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
        # unclosed box (truncated output): fall back to the previous one
    return None


def clean(s):
    """Answer string without units, dollar/percent/degree signs, spacing and \\left/\\right."""
    without_units = _UNIT_TEXT.sub("", s)
    if without_units.strip():
        s = without_units
    else:  # the whole answer is text, e.g. \text{Evelyn}: keep the word itself
        s = re.sub(r"\\(?:text|textbf|mathrm|mbox)\s*\{([^{}]*)\}", r"\1", s)
    s = _NOISE.sub("", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.rstrip(".")
    if _NUMBER.fullmatch(s):  # 1,000 -> 1000 and 18.00 -> 18 (not for tuples like (1,2))
        s = s.replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    return s


def is_equiv(pred, gold):
    if pred is None:
        return False
    p, g = clean(pred), clean(gold)
    if not p:
        return False
    if p == g:
        return True
    from math_verify import parse, verify  # slow import: only when needed
    try:
        return bool(verify(parse(f"${g}$"), parse(f"${p}$")))
    except Exception:  # math-verify gives up on some LaTeX; count as wrong
        return False


def normalise_gsm8k_gold(answer):
    """GSM8K solution text -> the integer after '####', as a string."""
    gold = answer.split("####")[-1].strip().replace(",", "")
    if not re.fullmatch(r"-?\d+", gold):
        raise ValueError(f"GSM8K gold is not an integer: {gold!r}")
    return str(int(gold))


def grade(text, gold):
    """Return (pred, correct) for an answer text."""
    pred = last_boxed(text)
    return pred, is_equiv(pred, gold)
