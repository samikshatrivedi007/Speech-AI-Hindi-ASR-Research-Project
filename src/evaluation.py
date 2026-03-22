"""
evaluation.py
-------------
WER / CER evaluation utilities for Hindi ASR.

Provides:
  - compute_wer(): word-level edit-distance WER using jiwer
  - compute_cer(): character-level edit distance CER
  - evaluate_dataset(): run inference over full dataset split
  - save_wer_results(): persist results to CSV
  - WERResult dataclass for structured results
"""

import csv
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from jiwer import wer, cer, compute_measures
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WERResult:
    """Holds evaluation result for a single dataset split or experiment."""
    split: str
    num_samples: int
    wer: float
    cer: float
    insertions: int = 0
    deletions: int = 0
    substitutions: int = 0
    hits: int = 0
    description: str = ""

    @property
    def wer_pct(self) -> str:
        return f"{self.wer * 100:.2f}%"

    @property
    def cer_pct(self) -> str:
        return f"{self.cer * 100:.2f}%"


@dataclass
class SampleResult:
    """Holds per-sample ASR evaluation result."""
    sample_id: int
    reference: str
    prediction: str
    sample_wer: float
    sample_cer: float


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_wer(references: List[str], predictions: List[str]) -> float:
    """
    Compute aggregate word error rate.

    Args:
        references: List of ground-truth transcriptions.
        predictions: List of model-generated transcriptions.

    Returns:
        WER as a float in [0, 1].
    """
    if len(references) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(references)} references vs "
            f"{len(predictions)} predictions."
        )
    return float(wer(references, predictions))


def compute_cer(references: List[str], predictions: List[str]) -> float:
    """
    Compute aggregate character error rate.

    Args:
        references: List of ground-truth transcriptions.
        predictions: List of model-generated transcriptions.

    Returns:
        CER as a float in [0, 1].
    """
    return float(cer(references, predictions))


def compute_detailed_measures(
    references: List[str], predictions: List[str]
) -> Dict[str, Any]:
    """
    Compute full jiwer measures (hits, substitutions, insertions, deletions).

    Returns:
        Dict with keys: wer, hits, substitutions, insertions, deletions.
    """
    measures = compute_measures(references, predictions)
    return {
        "wer": round(float(measures["wer"]), 6),
        "hits": int(measures["hits"]),
        "substitutions": int(measures["substitutions"]),
        "insertions": int(measures["insertions"]),
        "deletions": int(measures["deletions"]),
    }


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_samples(
    references: List[str],
    predictions: List[str],
) -> List[SampleResult]:
    """
    Compute per-sample WER and CER.

    Args:
        references: List of reference strings.
        predictions: List of hypothesis strings.

    Returns:
        List of SampleResult objects.
    """
    results = []
    for idx, (ref, pred) in enumerate(zip(references, predictions)):
        sample_wer = float(wer([ref], [pred])) if ref.strip() else 0.0
        sample_cer = float(cer([ref], [pred])) if ref.strip() else 0.0
        results.append(
            SampleResult(
                sample_id=idx,
                reference=ref,
                prediction=pred,
                sample_wer=sample_wer,
                sample_cer=sample_cer,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Dataset-level evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    transcribe_fn: Callable[[Any], str],
    dataset,
    text_key: str = "transcription",
    audio_key: str = "audio",
    split_name: str = "test",
    description: str = "",
) -> Tuple[WERResult, List[SampleResult]]:
    """
    Run inference over a HuggingFace dataset split and compute WER/CER.

    Args:
        transcribe_fn: Function that takes an audio dict and returns a string.
        dataset: HuggingFace Dataset object (a single split, not DatasetDict).
        text_key: Column name containing ground-truth transcription.
        audio_key: Column name containing audio dict (with 'array' and 'sampling_rate').
        split_name: Label for this split (used in the result).
        description: Optional human-readable label (e.g., "Baseline Whisper-small").

    Returns:
        Tuple of (WERResult, List[SampleResult]).
    """
    references: List[str] = []
    predictions: List[str] = []

    for sample in tqdm(dataset, desc=f"Evaluating [{split_name}]"):
        audio = sample[audio_key]
        ref = sample[text_key]
        try:
            pred = transcribe_fn(audio)
        except Exception as exc:
            logger.warning("Transcription failed for sample: %s", exc)
            pred = ""
        references.append(ref)
        predictions.append(pred)

    measures = compute_detailed_measures(references, predictions)
    cer_val = compute_cer(references, predictions)

    result = WERResult(
        split=split_name,
        num_samples=len(references),
        wer=measures["wer"],
        cer=cer_val,
        insertions=measures["insertions"],
        deletions=measures["deletions"],
        substitutions=measures["substitutions"],
        hits=measures["hits"],
        description=description,
    )

    sample_results = evaluate_samples(references, predictions)

    logger.info(
        "Evaluation complete | WER=%.4f | CER=%.4f | N=%d",
        result.wer, result.cer, result.num_samples,
    )
    return result, sample_results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_wer_results(
    results: List[WERResult],
    output_path: str = "outputs/wer_results.csv",
) -> None:
    """
    Append WER results to a CSV file (creates if it doesn't exist).

    Args:
        results: List of WERResult objects.
        output_path: Path to output CSV.
    """
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)

    import pathlib
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    file_exists = pathlib.Path(output_path).exists()
    df.to_csv(output_path, index=False, mode="a", header=not file_exists)
    logger.info("WER results saved to '%s'.", output_path)


def load_wer_results(path: str = "outputs/wer_results.csv") -> pd.DataFrame:
    """Load WER results CSV into a DataFrame."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_wer_results(
    baseline: WERResult,
    improved: WERResult,
) -> Dict[str, Any]:
    """
    Compare two WERResult objects to compute relative improvement.

    Returns:
        Dict with absolute and relative differences.
    """
    abs_diff = baseline.wer - improved.wer
    rel_improvement = abs_diff / baseline.wer * 100 if baseline.wer > 0 else 0.0

    return {
        "baseline_wer": baseline.wer,
        "improved_wer": improved.wer,
        "absolute_reduction": round(abs_diff, 6),
        "relative_improvement_pct": round(rel_improvement, 2),
        "baseline_description": baseline.description,
        "improved_description": improved.description,
    }
