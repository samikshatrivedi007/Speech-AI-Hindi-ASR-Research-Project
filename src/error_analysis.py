"""
error_analysis.py
-----------------
Systematic error taxonomy builder for Hindi ASR output analysis.

Features:
  - Dynamic, extensible error categorisation (no hardcoded labels)
  - Systematic sampling (every-N or random-seed based)
  - Per-error reasoning generation
  - CSV export compatible with Q1 notebook
"""

import re
import csv
import random
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

try:
    from evaluation import SampleResult
    from preprocessing import HindiTextNormalizer
    from number_normalizer import HINDI_NUMBER_WORDS
except ImportError:
    # Fallback when module is imported from a different working directory
    import sys, os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.dirname(__file__)))
    from evaluation import SampleResult
    from preprocessing import HindiTextNormalizer
    from number_normalizer import HINDI_NUMBER_WORDS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns used by classifiers
# ---------------------------------------------------------------------------

# Devanagari codepoint range
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Common English loanwords written in Devanagari (selected heuristic list)
ENGLISH_LOAN_WORDS_DEVANAGARI = {
    "स्कूल", "बस", "ट्रेन", "मोबाइल", "इंटरनेट", "कंप्यूटर",
    "टीवी", "रेडियो", "डॉक्टर", "पुलिस", "होटल", "हॉस्पिटल",
}

# HINDI_NUMBER_WORDS is imported from number_normalizer (canonical source)

ASCII_DIGIT_RE = re.compile(r"\d")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ErrorSample:
    """Represents one annotated ASR error instance."""
    sample_id: int
    reference: str
    prediction: str
    error_types: List[str]
    reasoning: str
    sample_wer: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_types"] = "|".join(d["error_types"])
        return d


# ---------------------------------------------------------------------------
# Rule-based error classifiers
# ---------------------------------------------------------------------------

class ErrorClassifier:
    """
    Stateless classifier that maps a (reference, prediction) pair to a set
    of error category labels and a human-readable reasoning string.

    Categories are dynamically determined — new categories can be added by
    extending the `_rules` list without modifying existing logic.
    """

    def __init__(self):
        # Each rule is (label, predicate_fn, reasoning_fn)
        # predicate returns True if the error type applies
        self._rules: List[Tuple[str, callable, callable]] = [
            (
                "insertion",
                lambda ref, pred: len(pred.split()) > len(ref.split()),
                lambda ref, pred: (
                    f"Prediction has {len(pred.split()) - len(ref.split())} "
                    "extra word(s) compared to reference."
                ),
            ),
            (
                "deletion",
                lambda ref, pred: len(pred.split()) < len(ref.split()),
                lambda ref, pred: (
                    f"Prediction is missing {len(ref.split()) - len(pred.split())} "
                    "word(s) present in reference."
                ),
            ),
            (
                "english_mix",
                lambda ref, pred: bool(LATIN_RE.search(pred)),
                lambda ref, pred: (
                    "Prediction contains Latin characters, suggesting English "
                    "words were not properly transcribed in Devanagari."
                ),
            ),
            (
                "number_error",
                self._is_number_error,
                lambda ref, pred: (
                    "Reference contains a number word, but prediction renders it "
                    "differently (digit vs. word or wrong number)."
                ),
            ),
            (
                "phonetic",
                self._is_phonetic_error,
                lambda ref, pred: (
                    "Words differ by a small edit distance (<=2 chars), suggesting "
                    "a phonetic confusion (similar-sounding characters)."
                ),
            ),
            (
                "hallucination",
                self._is_hallucination,
                lambda ref, pred: (
                    "Prediction contains tokens completely absent from reference "
                    "with no reference overlap, suggesting model hallucination."
                ),
            ),
        ]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _is_number_error(ref: str, pred: str) -> bool:
        ref_words = set(ref.split())
        pred_words = set(pred.split())
        ref_has_number_word = bool(ref_words & HINDI_NUMBER_WORDS)
        pred_has_digit = bool(ASCII_DIGIT_RE.search(pred))
        ref_has_digit = bool(ASCII_DIGIT_RE.search(ref))
        pred_has_number_word = bool(pred_words & HINDI_NUMBER_WORDS)
        return (ref_has_number_word and pred_has_digit) or (
            ref_has_digit and pred_has_number_word
        )

    @staticmethod
    def _is_phonetic_error(ref: str, pred: str) -> bool:
        """Check if differences look phonetic (small edits on individual words)."""
        try:
            import editdistance
        except ImportError:
            return False
        ref_words = ref.split()
        pred_words = pred.split()
        n = min(len(ref_words), len(pred_words))
        if n == 0:
            return False
        phonetic_pairs = [
            (r, p)
            for r, p in zip(ref_words, pred_words)
            if r != p and editdistance.eval(r, p) in (1, 2) and len(r) > 2
        ]
        return len(phonetic_pairs) > 0

    @staticmethod
    def _is_hallucination(ref: str, pred: str) -> bool:
        ref_words = set(ref.split())
        pred_words = set(pred.split())
        if not pred_words:
            return False
        overlap_ratio = len(ref_words & pred_words) / len(pred_words)
        return overlap_ratio < 0.2 and len(pred_words) > 3

    # -- public API --------------------------------------------------------

    def classify(self, reference: str, prediction: str) -> Tuple[List[str], str]:
        """
        Classify a (reference, prediction) pair.

        Returns:
            (list_of_error_labels, reasoning_string)
        """
        labels: List[str] = []
        reasons: List[str] = []

        for label, predicate, reasoning_fn in self._rules:
            if predicate(reference, prediction):
                labels.append(label)
                reasons.append(reasoning_fn(reference, prediction))

        if not labels:
            labels = ["substitution"]
            reasons = ["Generic word substitution with no specific pattern detected."]

        return labels, " | ".join(reasons)


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class ErrorSampler:
    """
    Draws a sample of erroneous ASR predictions for manual analysis.

    Args:
        n_samples: Number of errors to sample.
        strategy: 'systematic' (every-N) or 'random'.
        random_seed: Seed for reproducibility when strategy='random'.
    """

    def __init__(
        self,
        n_samples: int = 25,
        strategy: str = "systematic",
        random_seed: int = 42,
    ):
        self.n_samples = n_samples
        self.strategy = strategy
        self.random_seed = random_seed

    def sample(self, errors: List[SampleResult]) -> List[SampleResult]:
        """
        Sample from a list of erroneous SampleResult objects.

        Args:
            errors: All samples that have wer > 0.

        Returns:
            Sampled subset of size min(n_samples, len(errors)).
        """
        n = min(self.n_samples, len(errors))
        if n == 0:
            return []

        if self.strategy == "systematic":
            step = max(1, len(errors) // n)
            sampled = errors[::step][:n]
        elif self.strategy == "random":
            rng = random.Random(self.random_seed)
            sampled = rng.sample(errors, n)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}. Use 'systematic' or 'random'.")

        return sampled


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class ErrorAnalyzer:
    """
    Orchestrates systematic error sampling, classification, and export.

    Args:
        n_samples: Number of error samples to analyse.
        sampling_strategy: 'systematic' or 'random'.
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        n_samples: int = 25,
        sampling_strategy: str = "systematic",
        random_seed: int = 42,
    ):
        self.classifier = ErrorClassifier()
        self.sampler = ErrorSampler(
            n_samples=n_samples,
            strategy=sampling_strategy,
            random_seed=random_seed,
        )
        self.text_normalizer = HindiTextNormalizer()

    def analyse(self, sample_results: List[SampleResult]) -> List[ErrorSample]:
        """
        Run the full analysis pipeline.

        Args:
            sample_results: Output of evaluation.evaluate_samples().

        Returns:
            List of annotated ErrorSample objects.
        """
        # Filter only erroneous samples (WER > 0)
        errors = [s for s in sample_results if s.sample_wer > 0]
        logger.info(
            "Total errors: %d / %d samples (%.1f%%)",
            len(errors), len(sample_results),
            100 * len(errors) / max(len(sample_results), 1),
        )

        sampled = self.sampler.sample(errors)
        logger.info("Sampled %d errors for classification.", len(sampled))

        annotated: List[ErrorSample] = []
        for s in sampled:
            ref_norm = self.text_normalizer.normalize(s.reference)
            pred_norm = self.text_normalizer.normalize(s.prediction)
            labels, reasoning = self.classifier.classify(ref_norm, pred_norm)

            annotated.append(
                ErrorSample(
                    sample_id=s.sample_id,
                    reference=s.reference,
                    prediction=s.prediction,
                    error_types=labels,
                    reasoning=reasoning,
                    sample_wer=round(s.sample_wer, 4),
                )
            )

        return annotated

    def taxonomy_summary(self, annotated: List[ErrorSample]) -> Dict[str, int]:
        """
        Count occurrences of each error type across annotated samples.

        Returns:
            Dict mapping error label to count, sorted descending.
        """
        counts: Dict[str, int] = {}
        for sample in annotated:
            for label in sample.error_types:
                counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def save(
        self,
        annotated: List[ErrorSample],
        output_path: str = "outputs/error_samples.csv",
    ) -> None:
        """
        Save annotated error samples to CSV.

        Args:
            annotated: List of ErrorSample objects.
            output_path: Destination CSV path.
        """
        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if not annotated:
            logger.warning("No annotated errors to save.")
            return

        rows = [s.to_dict() for s in annotated]
        df_cols = ["sample_id", "reference", "prediction", "error_types",
                   "reasoning", "sample_wer"]
        import pandas as pd
        pd.DataFrame(rows, columns=df_cols).to_csv(output_path, index=False)
        logger.info("Saved %d error samples to '%s'.", len(rows), output_path)
