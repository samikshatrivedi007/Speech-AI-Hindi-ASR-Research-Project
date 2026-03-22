"""
spelling_checker.py
-------------------
Hindi spelling correctness classifier for ~1.75 lakh unique word inputs.

Approach:
  1. Dictionary lookup against a curated Hindi word list
  2. Levenshtein / edit-distance based similarity to nearest correct word
  3. Phonetic similarity option (Soundex-style for Devanagari)
  4. Confidence scoring: high / medium / low

Output per word:
  - word
  - label: "correct" / "incorrect"
  - confidence: "high" / "medium" / "low"
  - reason: human-readable explanation
  - nearest_match: closest correct word (if available)
"""

import csv
import difflib
import logging
import pathlib
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Set, Tuple

import editdistance
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLDS = {
    "high": (0.85, 1.0),
    "medium": (0.55, 0.85),
    "low": (0.0, 0.55),
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SpellingResult:
    """Per-word spelling classification result."""
    word: str
    label: str          # "correct" / "incorrect"
    confidence: str     # "high" / "medium" / "low"
    confidence_score: float
    reason: str
    nearest_match: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Phonetic encoder (Devanagari-aware)
# ---------------------------------------------------------------------------

class DevanagariPhoneticEncoder:
    """
    Simplified phonetic encoding for Devanagari text to detect similar-
    sounding words.  Maps characters that sound similar in Hindi SST to a
    canonical representation (similar to Soundex but Devanagari-aware).
    """

    # Pairs of visually/phonetically similar characters
    PHONETIC_MAP: Dict[str, str] = {
        "ण": "न", "श": "ष", "जा": "ज़",
        "ड": "ड़", "ढ": "ढ़",
        "ि": "ी", "ु": "ू",          # short/long vowels
        "े": "ै", "ो": "ौ",
        "ँ": "ं",                      # chandrabindu vs anusvara
        "ऑ": "ओ",
    }

    def encode(self, word: str) -> str:
        """Return phonetic canonical form of a Devanagari word."""
        result = word
        for original, replacement in self.PHONETIC_MAP.items():
            result = result.replace(original, replacement)
        return result


# ---------------------------------------------------------------------------
# Core spelling checker
# ---------------------------------------------------------------------------

class HindiSpellingChecker:
    """
    Classifies Hindi words as correct or incorrect using dictionary + edit-
    distance + optional phonetic matching.

    Args:
        dictionary_path: Path to a newline-separated list of correct Hindi
            words.  If None, a small built-in seed dictionary is used.
        max_edit_distance: Words within this Levenshtein distance of a
            dictionary word are candidates for correction.
        use_phonetics: If True, also compare phonetic encodings.
        batch_size: Number of words to process per tqdm batch.
    """

    # Fallback mini-dictionary when no external file is provided
    SEED_DICTIONARY: List[str] = [
        "नमस्ते", "भारत", "हिंदी", "पानी", "खाना", "घर", "परिवार",
        "स्कूल", "पढ़ना", "लिखना", "बोलना", "सुनना", "देखना",
        "आना", "जाना", "करना", "होना", "रहना", "मिलना",
        "अच्छा", "बुरा", "बड़ा", "छोटा", "नया", "पुराना",
        "दिन", "रात", "सुबह", "शाम", "समय", "साल", "महीना",
        "आज", "कल", "परसों", "अब", "तब", "यहाँ", "वहाँ",
        "मैं", "तुम", "वह", "हम", "आप", "वे", "यह", "ये",
        "किताब", "कलम", "कागज", "मेज", "कुर्सी", "दरवाजा",
        "काम", "नौकरी", "व्यापार", "सरकार", "देश", "राज्य",
    ]

    def __init__(
        self,
        dictionary_path: Optional[str] = None,
        max_edit_distance: int = 2,
        use_phonetics: bool = True,
        batch_size: int = 1000,
    ):
        self.max_edit_distance = max_edit_distance
        self.use_phonetics = use_phonetics
        self.batch_size = batch_size
        self.phonetic_encoder = DevanagariPhoneticEncoder()

        # Load dictionary
        self.dictionary: Set[str] = set()
        if dictionary_path and pathlib.Path(dictionary_path).exists():
            self.load_dictionary(dictionary_path)
        else:
            logger.warning(
                "No dictionary file provided or file not found. "
                "Using built-in seed dictionary (%d words). "
                "For production, provide a full dictionary file.",
                len(self.SEED_DICTIONARY),
            )
            self.dictionary = set(self.SEED_DICTIONARY)

        # Pre-compute phonetic encodings of dictionary words
        if self.use_phonetics:
            self._phonetic_dict: Dict[str, str] = {
                w: self.phonetic_encoder.encode(w) for w in self.dictionary
            }
        else:
            self._phonetic_dict = {}

        # Sorted list for difflib nearest-match searches
        self._dict_list: List[str] = sorted(self.dictionary)

        logger.info("Spelling checker ready. Dictionary size: %d words.", len(self.dictionary))

    def load_dictionary(self, path: str) -> None:
        """
        Load words from a newline-separated text file.

        Args:
            path: Path to the dictionary file.
        """
        with open(path, encoding="utf-8") as f:
            words = {line.strip() for line in f if line.strip()}
        self.dictionary.update(words)
        logger.info("Loaded %d words from '%s'.", len(words), path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @lru_cache(maxsize=None)
    def _nearest_match(self, word: str) -> Tuple[str, int]:
        """
        Find the nearest dictionary word to `word` by edit distance.

        Returns:
            (nearest_word, edit_distance)
        """
        best_word = ""
        best_dist = float("inf")
        for dict_word in self._dict_list:
            d = editdistance.eval(word, dict_word)
            if d < best_dist:
                best_dist = d
                best_word = dict_word
            if d == 0:
                break  # exact match
        return best_word, int(best_dist)

    def _is_phonetic_match(self, word: str) -> bool:
        """Check if the phonetic encoding of `word` matches any dict entry."""
        encoded = self.phonetic_encoder.encode(word)
        return any(enc == encoded for enc in self._phonetic_dict.values())

    def _compute_confidence_score(
        self,
        word: str,
        in_dict: bool,
        edit_dist: int,
        phonetic_match: bool,
    ) -> float:
        """
        Compute a float confidence score for the classification decision.

        Returns:
            Float in [0, 1].
        """
        if in_dict:
            return 1.0

        word_len = max(len(word), 1)
        # Normalised edit distance: 0 = identical, 1 = completely different
        norm_edit = 1.0 - min(edit_dist, word_len) / word_len

        if edit_dist == 0:
            return 1.0
        elif edit_dist <= 1:
            base = 0.80
        elif edit_dist <= 2:
            base = 0.55
        elif edit_dist <= 3:
            base = 0.35
        else:
            base = 0.15

        phonetic_bonus = 0.10 if phonetic_match else 0.0
        return min(base + phonetic_bonus, 1.0)

    @staticmethod
    def _confidence_label(score: float) -> str:
        for label, (low, high) in CONFIDENCE_THRESHOLDS.items():
            if low <= score <= high:
                return label
        return "low"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_word(self, word: str) -> SpellingResult:
        """
        Classify a single word.

        Args:
            word: A single Hindi word (Devanagari).

        Returns:
            SpellingResult.
        """
        word = word.strip()
        if not word:
            return SpellingResult(
                word=word, label="incorrect", confidence="low",
                confidence_score=0.0, reason="Empty input."
            )

        # --- Dictionary lookup ---
        if word in self.dictionary:
            return SpellingResult(
                word=word,
                label="correct",
                confidence="high",
                confidence_score=1.0,
                reason="Exact match found in dictionary.",
                nearest_match=word,
            )

        # --- Edit distance ---
        nearest, dist = self._nearest_match(word)

        # --- Phonetic check ---
        phonetic_match = self._is_phonetic_match(word) if self.use_phonetics else False

        confidence_score = self._compute_confidence_score(
            word, in_dict=False, edit_dist=dist, phonetic_match=phonetic_match
        )
        confidence_label = self._confidence_label(confidence_score)

        if dist <= self.max_edit_distance:
            label = "incorrect"
            reason = (
                f"Not in dictionary. Nearest match '{nearest}' at edit "
                f"distance {dist}. Likely a misspelling."
            )
        else:
            label = "incorrect"
            reason = (
                f"Not in dictionary. Nearest match '{nearest}' at edit "
                f"distance {dist} (outside correction range)."
            )
            if phonetic_match:
                reason += " Phonetic encoding matches a dictionary word."

        return SpellingResult(
            word=word,
            label=label,
            confidence=confidence_label,
            confidence_score=round(confidence_score, 4),
            reason=reason,
            nearest_match=nearest,
        )

    def check_batch(self, words: List[str]) -> List[SpellingResult]:
        """
        Classify a batch of words.

        Args:
            words: List of Hindi words.

        Returns:
            List of SpellingResult.
        """
        results: List[SpellingResult] = []
        for word in tqdm(words, desc="Checking spelling", unit="word"):
            results.append(self.check_word(word))
        return results

    def check_stream(self, word_iterator: Iterator[str]) -> Iterator[SpellingResult]:
        """
        Memory-efficient generator for large word streams.

        Args:
            word_iterator: Any iterator yielding Hindi words.

        Yields:
            SpellingResult for each word.
        """
        for word in word_iterator:
            yield self.check_word(word)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_results(
        self,
        results: List[SpellingResult],
        output_path: str = "outputs/spelling_results.csv",
    ) -> None:
        """Save results to CSV."""
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        rows = [r.to_dict() for r in results]
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info("Saved %d spelling results to '%s'.", len(rows), output_path)

    # ------------------------------------------------------------------
    # Low-confidence analysis
    # ------------------------------------------------------------------

    def low_confidence_analysis(
        self,
        results: List[SpellingResult],
        n_samples: int = 50,
    ) -> Dict[str, object]:
        """
        Analyse low-confidence predictions to identify failure patterns.

        Args:
            results: Full list of SpellingResult objects.
            n_samples: Max number of samples to inspect.

        Returns:
            Dict with samples, accuracy (if ground truth is known), and
            failure pattern summary.
        """
        low_conf = [r for r in results if r.confidence == "low"]
        sample = low_conf[:n_samples]

        # Failure pattern heuristics
        patterns: Dict[str, int] = {
            "very_short_word": 0,
            "very_long_word": 0,
            "num_letters_in_word": 0,
            "compound_word_candidate": 0,
            "unknown_root": 0,
        }

        for r in sample:
            w = r.word
            if len(w) <= 2:
                patterns["very_short_word"] += 1
            elif len(w) >= 20:
                patterns["very_long_word"] += 1
            if any(c.isdigit() for c in w):
                patterns["num_letters_in_word"] += 1
            if "ा" in w and "ी" in w:
                patterns["compound_word_candidate"] += 1
            if r.confidence_score < 0.2:
                patterns["unknown_root"] += 1

        return {
            "total_low_confidence": len(low_conf),
            "sample_size": len(sample),
            "failure_patterns": patterns,
            "sample_words": [r.word for r in sample],
        }
