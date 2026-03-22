"""
preprocessing.py
----------------
Audio and text preprocessing utilities for Hindi ASR pipeline.

Covers:
  - Audio resampling, loudness normalisation, silence trimming
  - Text normalisation (Hindi-specific: Devanagari cleanup, diacritics, etc.)
  - Feature extraction using Whisper's feature extractor
  - HuggingFace map-compatible dataset preprocessing
"""

import re
import logging
from typing import Any, Dict, Optional

import numpy as np
import librosa
from transformers import WhisperFeatureExtractor, WhisperTokenizer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation constants
# ---------------------------------------------------------------------------

# Unicode ranges
DEVANAGARI_RANGE = r"[\u0900-\u097F]"
DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Characters to strip from text
STRIP_CHARS = re.compile(r"[।॥\-–—:;,!?\"'()\[\]{}]")

# Punctuation / noise characters commonly produced by Whisper
WHISPER_HALLUCINATIONS = re.compile(
    r"\[.*?\]|\(.*?\)|<.*?>|♪|♫|【.*?】|（.*?）"
)


# ---------------------------------------------------------------------------
# Text Normaliser
# ---------------------------------------------------------------------------

class HindiTextNormalizer:
    """
    Normalises Hindi transcription text for WER computation.

    Steps:
      1. Strip Whisper artefacts / hallucinations
      2. Replace Devanagari digits with ASCII digits
      3. Lowercase Latin characters (for mixed-script text)
      4. Remove unwanted punctuation
      5. Collapse multiple spaces
    """

    def normalize(self, text: str) -> str:
        """
        Args:
            text: Raw transcription string.

        Returns:
            Normalised string.
        """
        text = WHISPER_HALLUCINATIONS.sub("", text)
        text = text.translate(DEVANAGARI_DIGITS)  # ० → 0, etc.
        text = STRIP_CHARS.sub(" ", text)
        text = re.sub(r"[A-Za-z]+", lambda m: m.group().lower(), text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def batch_normalize(self, texts):
        return [self.normalize(t) for t in texts]


# ---------------------------------------------------------------------------
# Audio Preprocessor
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """
    Prepares raw audio waveforms for Whisper model input.

    Args:
        target_sr: Target sampling rate (Whisper expects 16 kHz).
        normalize_loudness: If True, normalise peak amplitude to –1 dBFS.
        trim_silence: If True, trim leading/trailing silence using librosa.
        silence_threshold_db: dB threshold below which signal is considered silence.
    """

    def __init__(
        self,
        target_sr: int = 16_000,
        normalize_loudness: bool = True,
        trim_silence: bool = True,
        silence_threshold_db: float = 20.0,
    ):
        self.target_sr = target_sr
        self.normalize_loudness = normalize_loudness
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db

    def process_array(self, audio_array: np.ndarray, source_sr: int) -> np.ndarray:
        """
        Process a raw waveform array.

        Args:
            audio_array: Waveform as 1-D float32 numpy array.
            source_sr: Sampling rate of the input array.

        Returns:
            Processed waveform at `self.target_sr`.
        """
        # 1. Convert to float32
        audio = audio_array.astype(np.float32)

        # 2. Resample if needed
        if source_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=source_sr, target_sr=self.target_sr)

        # 3. Convert stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # 4. Trim silence
        if self.trim_silence:
            audio, _ = librosa.effects.trim(
                audio, top_db=self.silence_threshold_db
            )

        # 5. Normalise loudness
        if self.normalize_loudness:
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak * 0.98  # leave a little headroom

        return audio

    def process_file(self, audio_path: str) -> np.ndarray:
        """
        Load an audio file and run the full preprocessing pipeline.

        Args:
            audio_path: Path to audio file (any format supported by librosa).

        Returns:
            Processed waveform at self.target_sr.
        """
        audio, source_sr = librosa.load(audio_path, sr=None, mono=False)
        return self.process_array(audio, source_sr)


# ---------------------------------------------------------------------------
# Whisper Feature Extractor Wrapper
# ---------------------------------------------------------------------------

class WhisperInputProcessor:
    """
    Combines audio preprocessing + Whisper feature extraction + tokenisation
    into a single map-compatible object for HuggingFace datasets.

    Args:
        model_name: HuggingFace model name or local path.
        language: Language tag for Whisper tokeniser (e.g. "hindi").
        task: "transcribe" or "translate".
        max_audio_length: Maximum audio duration in seconds before truncation.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        language: str = "hindi",
        task: str = "transcribe",
        max_audio_length: float = 30.0,
    ):
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name, language=language, task=task
        )
        self.text_normalizer = HindiTextNormalizer()
        self.audio_preprocessor = AudioPreprocessor(
            target_sr=self.feature_extractor.sampling_rate
        )
        self.max_audio_length = max_audio_length
        self.sr = self.feature_extractor.sampling_rate

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        HuggingFace dataset.map-compatible preprocessing function.

        Args:
            batch: A single dataset example containing 'audio' and 'transcription'.

        Returns:
            Dict with 'input_features', 'labels' ready for Whisper training.
        """
        audio = batch["audio"]
        waveform = self.audio_preprocessor.process_array(
            audio["array"], audio["sampling_rate"]
        )

        # Truncate / pad to max_audio_length
        max_samples = int(self.max_audio_length * self.sr)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Extract log-mel spectrogram features
        input_features = self.feature_extractor(
            waveform,
            sampling_rate=self.sr,
            return_tensors="np",
        ).input_features[0]

        # Normalise and tokenise the transcription text
        text_key = "transcription" if "transcription" in batch else "sentence"
        raw_text = batch.get(text_key, batch.get("raw_transcription", ""))
        normalised_text = self.text_normalizer.normalize(raw_text)
        labels = self.tokenizer(normalised_text).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }

    def decode(self, token_ids) -> str:
        """Decode token ids to a string, skipping special tokens."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        return self.feature_extractor

    def get_tokenizer(self) -> WhisperTokenizer:
        return self.tokenizer
