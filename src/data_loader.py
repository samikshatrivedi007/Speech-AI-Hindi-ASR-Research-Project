"""
data_loader.py  (REFINED)
--------------------------
Data loading utilities for the JoshTalks Hindi ASR dataset.

Real dataset schema (from spreadsheet):
  user_id, recording_id, language, duration,
  rec_url_gcp, transcription_url_gcp, metadata_url_gcp

Transcription JSON format (per-segment):
  [{"start": 0.11, "end": 14.42, "speaker_id": 245746, "text": "..."},  ...]

Supports:
  - Loading the training manifest from Google Sheets CSV export
  - Downloading and parsing transcription JSONs from GCS
  - Downloading audio files from GCS
  - Slicing audio into per-segment chunks matching transcription segments
  - HuggingFace Dataset creation with 'audio' + 'transcription' columns
  - FLEURS Hindi evaluation split
"""

import os
import io
import json
import logging
import pathlib
import tempfile
import time
from typing import Dict, Iterator, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI/"
    "export?format=csv&gid=1786138861"
)

TARGET_SR = 16_000   # Whisper requires 16kHz
REQUEST_TIMEOUT = 60  # seconds
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2       # seconds between retries


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_with_retry(url: str, stream: bool = False, timeout: int = REQUEST_TIMEOUT):
    """GET a URL with automatic retries on transient failures."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            logger.warning("Attempt %d/%d failed for %s: %s — retrying in %ds",
                           attempt, RETRY_ATTEMPTS, url, exc, RETRY_DELAY)
            time.sleep(RETRY_DELAY)


# ---------------------------------------------------------------------------
# Transcription JSON parser
# ---------------------------------------------------------------------------

def parse_transcription_json(json_data: list) -> List[Dict]:
    """
    Parse a GCS transcription JSON array into a clean list of segments.

    Args:
        json_data: Parsed list of segment dicts from the JSON file.

    Returns:
        List of dicts with keys: start, end, speaker_id, text.
    """
    segments = []
    for seg in json_data:
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "speaker_id": seg.get("speaker_id", -1),
            "text": text,
        })
    return segments


def fetch_transcription_segments(transcription_url: str) -> List[Dict]:
    """
    Download and parse a transcription JSON from GCS.

    Args:
        transcription_url: URL to the *_transcription.json file.

    Returns:
        List of segment dicts (start, end, speaker_id, text).
    """
    try:
        resp = _get_with_retry(transcription_url)
        return parse_transcription_json(resp.json())
    except Exception as exc:
        logger.warning("Failed to fetch transcription from %s: %s", transcription_url, exc)
        return []


# ---------------------------------------------------------------------------
# Audio segment extractor
# ---------------------------------------------------------------------------

def extract_segment(
    audio_array: np.ndarray,
    sr: int,
    start_sec: float,
    end_sec: float,
    target_sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Extract and resample a time slice from a waveform.

    Args:
        audio_array: Full recording waveform (float32).
        sr: Sampling rate of audio_array.
        start_sec: Segment start time in seconds.
        end_sec: Segment end time in seconds.
        target_sr: Desired output sampling rate.

    Returns:
        Segment waveform at target_sr.
    """
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(audio_array), int(end_sec * sr))
    segment = audio_array[start_sample:end_sample].astype(np.float32)

    if sr != target_sr:
        segment = librosa.resample(segment, orig_sr=sr, target_sr=target_sr)

    return segment


# ---------------------------------------------------------------------------
# Main dataset loader
# ---------------------------------------------------------------------------

class JoshTalksDatasetLoader:
    """
    Loads the JoshTalks ~10-hour Hindi ASR dataset.

    Workflow:
      1. Download the manifest CSV from Google Sheets (or load from cache)
      2. For each recording, download the transcription JSON — parse segments
      3. Download audio from GCS (or load from cache)
      4. Slice audio into per-segment chunks aligned with transcription
      5. Return a HuggingFace Dataset of (audio_array, transcription) pairs

    Args:
        manifest_url: URL to the Google Sheets CSV export.
        cache_dir: Local directory for caching audio + JSON files.
        target_sr: Audio sampling rate for output (Whisper needs 16kHz).
        max_segment_duration: Skip segments longer than this value (seconds).
        min_segment_duration: Skip segments shorter than this value (seconds).
        max_recordings: If set, limit to first N recordings (for dev/testing).
    """

    def __init__(
        self,
        manifest_url: str = DATASET_CSV_URL,
        cache_dir: str = "data/raw",
        target_sr: int = TARGET_SR,
        max_segment_duration: float = 30.0,
        min_segment_duration: float = 0.5,
        max_recordings: Optional[int] = None,
    ):
        self.manifest_url = manifest_url
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        self.max_segment_duration = max_segment_duration
        self.min_segment_duration = min_segment_duration
        self.max_recordings = max_recordings

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def load_manifest(self, local_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load the dataset manifest CSV.

        Args:
            local_path: If set, read from this local CSV instead of the URL.

        Returns:
            DataFrame with one row per recording.
        """
        if local_path and pathlib.Path(local_path).exists():
            df = pd.read_csv(local_path)
            logger.info("Manifest loaded from local file: %d recordings.", len(df))
            return df

        cached_path = self.cache_dir / "manifest.csv"
        if cached_path.exists():
            df = pd.read_csv(cached_path)
            logger.info("Manifest loaded from cache: %d recordings.", len(df))
            return df

        logger.info("Downloading manifest from %s …", self.manifest_url)
        df = pd.read_csv(self.manifest_url)
        df.to_csv(cached_path, index=False)
        logger.info("Manifest downloaded: %d recordings.", len(df))
        return df

    # ------------------------------------------------------------------
    # Audio download
    # ------------------------------------------------------------------

    def _download_audio(self, rec_url: str, recording_id: str) -> Optional[pathlib.Path]:
        """Download audio from GCS and cache it locally."""
        local_path = self.cache_dir / f"{recording_id}_audio.wav"
        if local_path.exists():
            return local_path

        try:
            resp = _get_with_retry(rec_url, stream=True)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return local_path
        except Exception as exc:
            logger.warning("Audio download failed [%s]: %s", recording_id, exc)
            return None

    def _load_audio(self, audio_path: pathlib.Path) -> Tuple[np.ndarray, int]:
        """Load audio file at native sample rate (resampling happens per-segment)."""
        audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
        return audio.astype(np.float32), int(sr)

    # ------------------------------------------------------------------
    # Segment generator
    # ------------------------------------------------------------------

    def iter_segments(
        self, manifest: pd.DataFrame
    ) -> Iterator[Dict]:
        """
        Yield one dict per audio segment across all recordings.

        Yields:
            {
                "audio": np.ndarray (16kHz float32),
                "transcription": str,
                "recording_id": str,
                "speaker_id": int,
                "start": float,
                "end": float,
            }
        """
        recordings = manifest.itertuples(index=False)
        if self.max_recordings is not None:
            import itertools
            recordings = itertools.islice(recordings, self.max_recordings)

        for row in recordings:
            recording_id = str(row.recording_id)

            # 1. Download transcription JSON
            segments = fetch_transcription_segments(row.transcription_url_gcp)
            if not segments:
                logger.warning("No segments for recording %s — skipping.", recording_id)
                continue

            # 2. Download audio
            audio_path = self._download_audio(row.rec_url_gcp, recording_id)
            if audio_path is None:
                continue

            # 3. Load full audio
            try:
                full_audio, native_sr = self._load_audio(audio_path)
            except Exception as exc:
                logger.warning("Could not load audio [%s]: %s", recording_id, exc)
                continue

            # 4. Slice and yield per-segment
            for seg in segments:
                duration = seg["end"] - seg["start"]
                if duration < self.min_segment_duration or duration > self.max_segment_duration:
                    continue

                seg_audio = extract_segment(
                    full_audio, native_sr, seg["start"], seg["end"], self.target_sr
                )
                if len(seg_audio) == 0:
                    continue

                yield {
                    "audio": seg_audio,
                    "transcription": seg["text"],
                    "recording_id": recording_id,
                    "speaker_id": seg["speaker_id"],
                    "start": seg["start"],
                    "end": seg["end"],
                }

    # ------------------------------------------------------------------
    # Build HuggingFace Dataset
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        manifest: Optional[pd.DataFrame] = None,
        val_fraction: float = 0.1,
        random_seed: int = 42,
    ) -> DatasetDict:
        """
        Build a HuggingFace DatasetDict from the JoshTalks data.

        Args:
            manifest: Pre-loaded manifest DataFrame (loads from URL if None).
            val_fraction: Fraction of recordings to hold out for validation.
            random_seed: Reproducibility seed.

        Returns:
            DatasetDict with 'train' and 'validation' splits.
        """
        if manifest is None:
            manifest = self.load_manifest()

        # Shuffle and split recordings (NOT segments) to avoid data leakage
        manifest = manifest.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        n_val = max(1, int(len(manifest) * val_fraction))
        val_df = manifest.iloc[:n_val]
        train_df = manifest.iloc[n_val:]

        logger.info(
            "Building dataset: %d train recordings, %d validation recordings.",
            len(train_df), len(val_df),
        )

        def _collect(df: pd.DataFrame) -> Dict[str, list]:
            data = {"audio": [], "transcription": [], "recording_id": [],
                    "speaker_id": [], "start": [], "end": []}
            for item in tqdm(self.iter_segments(df), desc="Loading segments"):
                for key in data:
                    data[key].append(item[key])
            return data

        train_data = _collect(train_df)
        val_data = _collect(val_df)

        logger.info(
            "Segments collected: %d train, %d validation.",
            len(train_data["audio"]), len(val_data["audio"]),
        )

        def _to_hf_dataset(data: Dict) -> Dataset:
            # Convert numpy arrays to a format the Audio column accepts
            return Dataset.from_dict({
                "audio": [{"array": a, "sampling_rate": self.target_sr}
                           for a in data["audio"]],
                "transcription": data["transcription"],
                "recording_id": data["recording_id"],
                "speaker_id": data["speaker_id"],
                "start": data["start"],
                "end": data["end"],
            }).cast_column("audio", Audio(sampling_rate=self.target_sr))

        return DatasetDict({
            "train": _to_hf_dataset(train_data),
            "validation": _to_hf_dataset(val_data),
        })


# ---------------------------------------------------------------------------
# FLEURS Hindi evaluation loader
# ---------------------------------------------------------------------------

def load_fleurs_hindi_test(cache_dir: str = "data/raw") -> Dataset:
    """
    Load ONLY the test split of FLEURS Hindi for evaluation.

    Returns:
        HuggingFace Dataset with 'audio' and 'transcription' columns.
    """
    logger.info("Loading FLEURS Hindi test split …")
    ds = load_dataset(
        "google/fleurs",
        "hi_in",
        split="test",
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    # FLEURS uses 'transcription' column
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    logger.info("FLEURS test split loaded: %d samples.", len(ds))
    return ds


# ---------------------------------------------------------------------------
# Q4 Lattice data loader
# ---------------------------------------------------------------------------

class Q4LatticeDataLoader:
    """
    Loads the Q4 five-model evaluation spreadsheet.

    Spreadsheet columns:
      segment_url_link, Human, Model H, Model i, Model k, Model l, Model m, Model n

    Returns structured dicts suitable for LatticeBuilder.
    """

    SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU/"
        "export?format=csv&gid=1432279672"
    )

    MODEL_COLUMNS = ["Model H", "Model i", "Model k", "Model l", "Model m", "Model n"]

    def __init__(self, sheet_url: Optional[str] = None):
        self.sheet_url = sheet_url or self.SHEET_URL

    def load(self, local_csv: Optional[str] = None) -> pd.DataFrame:
        """Load the spreadsheet and return a clean DataFrame."""
        if local_csv and pathlib.Path(local_csv).exists():
            df = pd.read_csv(local_csv)
        else:
            df = pd.read_csv(self.sheet_url)

        # Strip trailing whitespace/newlines from all string columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        # Remove header-only rows
        df = df[df["Human"].notna() & (df["Human"] != "nan")]

        logger.info("Q4 lattice data loaded: %d rows.", len(df))
        return df

    def to_evaluation_records(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert the DataFrame to evaluation records.

        Returns:
            List of dicts with keys:
              - segment_url: audio URL
              - reference: human transcription
              - hypotheses: {model_name: transcription, ...}
        """
        records = []
        for _, row in df.iterrows():
            hyps = {}
            for col in self.MODEL_COLUMNS:
                if col in df.columns:
                    val = str(row.get(col, "")).strip()
                    if val and val not in ("nan", "None", ""):
                        hyps[col] = val
            records.append({
                "segment_url": row.get("segment_url_link", ""),
                "reference": row.get("Human", ""),
                "hypotheses": hyps,
            })
        return records


# ---------------------------------------------------------------------------
# Q3 Word list loader
# ---------------------------------------------------------------------------

class Q3WordListLoader:
    """
    Loads the ~1.77 lakh unique Hindi words from the Q3 Google Sheet.

    Sheet URL:
      https://docs.google.com/spreadsheets/d/17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw/
    """

    SHEET_URL = (
        "https://docs.google.com/spreadsheets/d/"
        "17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw/"
        "export?format=csv&gid=0"
    )

    def __init__(self, sheet_url: Optional[str] = None, cache_path: str = "data/processed/q3_words.csv"):
        self.sheet_url = sheet_url or self.SHEET_URL
        self.cache_path = pathlib.Path(cache_path)

    def load(self) -> List[str]:
        """
        Download and return the unique word list.

        Returns:
            List of unique Hindi words (strings).
        """
        if self.cache_path.exists():
            df = pd.read_csv(self.cache_path, header=None)
            words = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
            logger.info("Q3 word list loaded from cache: %d words.", len(words))
            return words

        logger.info("Downloading Q3 word list from Google Sheets …")
        try:
            df = pd.read_csv(self.sheet_url, header=None)
        except Exception as exc:
            logger.error("Failed to download Q3 word list: %s", exc)
            raise

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.cache_path, index=False, header=False)

        words = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        logger.info("Q3 word list downloaded: %d unique words.", len(words))
        return words
