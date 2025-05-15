from collections.abc import Collection, Sequence
from functools import lru_cache
from itertools import takewhile

__all__ = [
    "decimals",
    "format_float",
    "multi_split",
    "nth_occurrence",
    "seconds_to_time",
    "find_single_matching_string",
]


def decimals(text: str) -> int:
    if "." in text:
        return len(text) - text.rfind(".") - 1
    return 0


@lru_cache(maxsize=128, typed=True)
def format_float(value: float, precision: int = 10, *, prefix: str = "", suffix: str = "") -> str:
    return prefix + f"{value:.{precision}f}".rstrip("0").rstrip(".") + suffix


def seconds_to_time(seconds: int) -> str:
    if not seconds:
        return "00:00:00"
    hours: int = abs(seconds) // 3600
    minutes: int = (abs(seconds) - hours * 3600) // 60
    seconds: int = abs(seconds) % 60
    return f"{'-' if seconds < 0 else ''}{hours:02d}:{minutes:02d}:{seconds:02d}"


def multi_split(text: str, separators: Sequence[str]) -> list[str]:
    words: list[str] = []
    separator: str
    index: int = -1
    while index < len(text):
        index += 1
        for separator in separators:
            if text.startswith(separator, index):
                words.append(text[:index])
                text = text[(index + len(separator)) :]
                index = 0
    words.append(text)
    return words


def nth_occurrence(text: str, substrings: str | Sequence[str], n: int) -> int:
    if isinstance(substrings, str):
        substrings = (substrings,)
    index: int
    count: int = 0
    for index in range(len(text)):
        if any(text.startswith(substring, index) for substring in substrings):
            count += 1
            if count == n:
                return index
    return -1


def find_single_matching_string(s: str, variants: Collection[str]) -> str:
    if s in variants:
        return s

    numeric_suffix: str = "".join(takewhile(str.isnumeric, s[::-1]))[::-1]
    s = s.casefold()
    matching_variants: list[str]
    if numeric_suffix:
        s = s[: -len(numeric_suffix)]
        matching_variants = [v for v in variants if v.casefold().startswith(s) and v.endswith(numeric_suffix)]
    else:
        matching_variants = [v for v in variants if v.casefold().startswith(s)]
    if len(matching_variants) == 1:
        return matching_variants[0]
    if matching_variants:
        raise ValueError(f"Ambiguous value: {s!r}. Possible variants: {matching_variants}")
    raise ValueError(f"Ambiguous value: {s!r}. Provided variants: {variants}")
