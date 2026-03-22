"""
number_normalizer.py
--------------------
Hindi number word → digit converter for ASR output cleanup.

Features:
  - Converts Hindi number words to Arabic numerals
  - Handles compound numbers (e.g., तीन सौ चौवन → 354)
  - Skips idioms and non-numeric contexts via heuristic rules
  - Fully rule-based, no external dependencies beyond standard library
"""

import re
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Number word tables (Hindi Devanagari)
# ---------------------------------------------------------------------------

# Single units: 1–19
UNITS: dict = {
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5,
    "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8,
    "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12,
    "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16,
    "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
}

# Tens
TENS: dict = {
    "बीस": 20, "इक्कीस": 21, "बाईस": 22, "तेईस": 23,
    "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27,
    "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30, "इकत्तीस": 31,
    "बत्तीस": 32, "तैंतीस": 33, "चौंतीस": 34, "पैंतीस": 35,
    "छत्तीस": 36, "सैंतीस": 37, "अड़तीस": 38, "उनतालीस": 39,
    "चालीस": 40, "इकतालीस": 41, "बयालीस": 42, "तैंतालीस": 43,
    "चौवालीस": 44, "पैंतालीस": 45, "छियालीस": 46, "सैंतालीस": 47,
    "अड़तालीस": 48, "उनचास": 49, "पचास": 50, "इक्यावन": 51,
    "बावन": 52, "तिरपन": 53, "चौवन": 54, "पचपन": 55,
    "छप्पन": 56, "सत्तावन": 57, "अट्ठावन": 58, "उनसठ": 59,
    "साठ": 60, "इकसठ": 61, "बासठ": 62, "तिरसठ": 63,
    "चौंसठ": 64, "पैंसठ": 65, "छियासठ": 66, "सड़सठ": 67,
    "अड़सठ": 68, "उनहत्तर": 69, "सत्तर": 70, "इकहत्तर": 71,
    "बहत्तर": 72, "तिहत्तर": 73, "चौहत्तर": 74, "पचहत्तर": 75,
    "छिहत्तर": 76, "सतहत्तर": 77, "अठहत्तर": 78, "उनासी": 79,
    "अस्सी": 80, "इक्यासी": 81, "बयासी": 82, "तिरासी": 83,
    "चौरासी": 84, "पचासी": 85, "छियासी": 86, "सत्तासी": 87,
    "अट्ठासी": 88, "नवासी": 89, "नब्बे": 90, "इक्यानवे": 91,
    "बानवे": 92, "तिरानवे": 93, "चौरानवे": 94, "पचानवे": 95,
    "छियानवे": 96, "सत्तानवे": 97, "अट्ठानवे": 98, "निन्यानवे": 99,
}

# Scale multipliers
SCALES: dict = {
    "सौ": 100,
    "हजार": 1_000,
    "लाख": 100_000,
    "करोड़": 10_000_000,
    "अरब": 1_000_000_000,
}

# Combine all single-word numerals into one lookup
ALL_NUMBER_WORDS: dict = {**UNITS, **TENS}

# Public export: flat set of all Hindi number word tokens (units + tens + scales)
# Used by other modules (Q2 notebook, error_analysis.py) to detect number words in text
HINDI_NUMBER_WORDS: set = set(UNITS.keys()) | set(TENS.keys()) | set(SCALES.keys())

# Idioms – these contain number words but must NOT be converted
IDIOM_PATTERNS: List[re.Pattern] = [
    re.compile(r"दो-चार"),           # do-char = some/a few
    re.compile(r"दो-तीन"),
    re.compile(r"चार-पाँच"),
    re.compile(r"एक-दो"),
    re.compile(r"पाँच-दस"),
    re.compile(r"तीन-चार"),
    re.compile(r"पाँच-सात"),
]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class HindiNumberParser:
    """
    Converts a sequence of Hindi number tokens to an integer value.

    Follows standard Hindi compounding rules:
      units/tens → optionally multiplied by सौ
      result → optionally multiplied by हजार/लाख/करोड़
    """

    def parse_tokens(self, tokens: List[str]) -> Optional[int]:
        """
        Attempt to parse a list of space-separated Hindi number words into int.

        Args:
            tokens: List of individual word tokens.

        Returns:
            Integer value or None if tokens do not represent a valid number.
        """
        result = 0
        current = 0

        for token in tokens:
            if token in ALL_NUMBER_WORDS:
                current += ALL_NUMBER_WORDS[token]
            elif token == "सौ":
                if current == 0:
                    current = 1
                current *= 100
            elif token in SCALES and token != "सौ":
                scale = SCALES[token]
                if current == 0:
                    current = 1
                result += current * scale
                current = 0
            else:
                return None  # encountered a non-number word

        result += current
        return result if result > 0 else None


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class HindiNumberNormalizer:
    """
    Replaces Hindi number words in a sentence with their Arabic numeral forms.

    Design choices:
      - Greedy multi-word matching (longest match first)
      - Idiom protection: known idiomatic phrases are skipped
      - No conversion for zero-value sequences (ambiguity guard)

    Args:
        preserve_idioms: If True, identified idioms are left unchanged.
    """

    def __init__(self, preserve_idioms: bool = True):
        self.parser = HindiNumberParser()
        self.preserve_idioms = preserve_idioms

        # Build sorted vocab list (longest first for greedy matching)
        self._all_number_words: List[str] = sorted(
            list(ALL_NUMBER_WORDS.keys()) + list(SCALES.keys()),
            key=len,
            reverse=True,
        )

    def _protect_idioms(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Replace idiomatic patterns with placeholder tokens.

        Returns:
            Modified text and list of (placeholder, original) pairs.
        """
        placeholders: List[Tuple[str, str]] = []
        if not self.preserve_idioms:
            return text, placeholders
        for i, pattern in enumerate(IDIOM_PATTERNS):
            match = pattern.search(text)
            if match:
                placeholder = f"__IDIOM_{i}__"
                placeholders.append((placeholder, match.group()))
                text = pattern.sub(placeholder, text)
        return text, placeholders

    def _restore_idioms(
        self, text: str, placeholders: List[Tuple[str, str]]
    ) -> str:
        for placeholder, original in placeholders:
            text = text.replace(placeholder, original)
        return text

    def _find_number_spans(self, tokens: List[str]) -> List[Tuple[int, int]]:
        """
        Find contiguous spans of tokens that form Hindi numbers.

        Returns:
            List of (start_idx, end_idx_exclusive) spans.
        """
        spans: List[Tuple[int, int]] = []
        i = 0
        number_token_set = set(ALL_NUMBER_WORDS.keys()) | set(SCALES.keys())

        while i < len(tokens):
            if tokens[i] in number_token_set:
                j = i
                while j < len(tokens) and tokens[j] in number_token_set:
                    j += 1
                spans.append((i, j))
                i = j
            else:
                i += 1
        return spans

    def normalize(self, text: str) -> str:
        """
        Normalise a single sentence.

        Args:
            text: Raw Hindi sentence.

        Returns:
            Sentence with number words replaced by digits.
        """
        # Step 1: protect idioms
        text, placeholders = self._protect_idioms(text)

        # Step 2: tokenize
        tokens = text.split()

        # Step 3: find number spans
        spans = self._find_number_spans(tokens)

        # Step 4: replace spans with digits (process in reverse to preserve indices)
        for start, end in reversed(spans):
            span_tokens = tokens[start:end]
            value = self.parser.parse_tokens(span_tokens)
            if value is not None:
                tokens[start:end] = [str(value)]

        # Step 5: reassemble
        result = " ".join(tokens)

        # Step 6: restore idioms
        result = self._restore_idioms(result, placeholders)

        return result

    def normalize_batch(self, sentences: List[str]) -> List[str]:
        """Apply normalize() to a list of sentences."""
        return [self.normalize(s) for s in sentences]
