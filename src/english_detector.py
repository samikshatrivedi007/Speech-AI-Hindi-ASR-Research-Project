"""
english_detector.py
-------------------
Detects English words written in Devanagari script (code-switching) in
Hindi ASR output and tags them with [EN]…[/EN] markers.

Approach (layered heuristics):
  1. Dictionary lookup — known English loanwords/borrowings in Devanagari
  2. Phonotactic heuristics — Devanagari syllable patterns that don't occur
     in native Hindi (consonant clusters borrowed from English)
  3. Optional: per-word confidence score

Tagging format:  [EN]word[/EN]
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in English loanword dictionary (Devanagari forms)
# ---------------------------------------------------------------------------
# This list is intentionally extensive for heuristic coverage.
# In production, supplement with a larger dictionary or embeddings.

ENGLISH_DEVANAGARI_DICT: Set[str] = {
    # Tech / digital
    "कंप्यूटर", "इंटरनेट", "मोबाइल", "स्मार्टफोन", "टैबलेट",
    "लैपटॉप", "वेबसाइट", "ईमेल", "सॉफ्टवेयर", "हार्डवेयर",
    "एप", "ऐप", "वीडियो", "ऑडियो", "पॉडकास्ट", "ब्लॉग",
    "यूट्यूब", "गूगल", "फेसबुक", "ट्विटर", "इंस्टाग्राम",
    # Transport
    "बस", "ट्रेन", "मेट्रो", "टैक्सी", "ट्रक", "बाइक",
    "स्कूटर", "कार", "जीप", "ट्राम", "फ्लाइट", "एयरपोर्ट",
    # Medical / services
    "डॉक्टर", "हॉस्पिटल", "क्लिनिक", "नर्स", "एम्बुलेंस",
    "पुलिस", "होटल", "रेस्टोरेंट", "बैंक", "एटीएम",
    # Education
    "स्कूल", "कॉलेज", "यूनिवर्सिटी", "क्लास", "एग्जाम",
    "प्रोफेसर", "टीचर", "स्टूडेंट", "लाइब्रेरी",
    # Entertainment
    "सिनेमा", "थिएटर", "टीवी", "रेडियो", "म्यूजिक",
    "सॉन्ग", "एलबम", "बैंड", "कॉन्सर्ट", "स्पोर्ट्स",
    "क्रिकेट", "फुटबॉल", "टेनिस", "गोल्फ", "मैच",
    # Business / finance
    "ऑफिस", "मैनेजर", "डायरेक्टर", "सीईओ", "मार्केट",
    "इन्वेस्टमेंट", "लोन", "इंश्योरेंस", "पेमेंट", "रिफंड",
    # Common adjectives / nouns
    "डिजिटल", "ऑनलाइन", "ऑफलाइन", "स्मार्ट", "फ्री",
    "प्रीमियम", "स्पेशल", "नॉर्मल", "रेगुलर", "ओरिजिनल",
    "फ्रेश", "क्लीन", "सेफ", "सिंपल", "क्विक", "फास्ट",
}

# ---------------------------------------------------------------------------
# Phonotactic patterns indicative of English origin in Devanagari
# ---------------------------------------------------------------------------
# Patterns: consonant clusters that rarely occur in native tatsama words

ENGLISH_PHONOTACTIC_PATTERNS: List[re.Pattern] = [
    re.compile(r"[क-ह][़]"),          # Nukta-modified consonants (ड़, ढ़ etc.) — not a reliable marker alone
    re.compile(r"^ऑ"),                 # ऑ (rounded 'o') onset — near-exclusively for English loanwords
    re.compile(r"[ैौ]ट$"),             # -ट ending with diphthong (e.g., "बाइट", "बूट")
    re.compile(r"^[अआइईउऊ]ं[कगट]"),   # -ing like endings transliterated
]

# Heuristic token-length guard: very short tokens (≤ 2 chars devanagari) skip phonotactics
MIN_CHARS_FOR_PHONOTACTIC = 3


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class EnglishDevanagariDetector:
    """
    Detects English words transliterated into Devanagari and tags them.

    Args:
        custom_dictionary: Optional additional set of known English loanwords
            in Devanagari to supplement the built-in list.
        use_phonotactics: If True, also use phonotactic heuristics.
        confidence_threshold: Minimum confidence (0–1) to tag a word.
    """

    def __init__(
        self,
        custom_dictionary: Optional[Set[str]] = None,
        use_phonotactics: bool = True,
        confidence_threshold: float = 0.5,
    ):
        self.dictionary: Set[str] = ENGLISH_DEVANAGARI_DICT.copy()
        if custom_dictionary:
            self.dictionary.update(custom_dictionary)
        self.use_phonotactics = use_phonotactics
        self.confidence_threshold = confidence_threshold

    def _score_word(self, word: str) -> float:
        """
        Compute an English-origin confidence score for a single Devanagari word.

        Scoring logic:
          - Dictionary hit → 0.95
          - Phonotactic match → +0.45 per pattern (capped at 0.9 if no dict hit)
          - No signal → 0.0

        Args:
            word: A single whitespace-free token.

        Returns:
            Float confidence in [0, 1].
        """
        score = 0.0

        # 1. Dictionary check
        if word in self.dictionary:
            return 0.95

        # 2. Phonotactic heuristics
        if self.use_phonotactics and len(word) >= MIN_CHARS_FOR_PHONOTACTIC:
            for pattern in ENGLISH_PHONOTACTIC_PATTERNS:
                if pattern.search(word):
                    score += 0.45

        return min(score, 0.9)

    def detect(self, text: str) -> List[Tuple[str, float]]:
        """
        Detect English words in Devanagari text.

        Args:
            text: Input sentence.

        Returns:
            List of (word, confidence) for words identified as English.
        """
        tokens = text.split()
        results: List[Tuple[str, float]] = []
        for token in tokens:
            # Strip punctuation before scoring
            clean = re.sub(r"[^\u0900-\u097F]", "", token)
            if not clean:
                continue
            confidence = self._score_word(clean)
            if confidence >= self.confidence_threshold:
                results.append((clean, confidence))
        return results

    def tag(self, text: str) -> str:
        """
        Return text with English words wrapped in [EN]…[/EN] tags.

        Args:
            text: Input sentence.

        Returns:
            Sentence with English loanwords tagged.
        """
        tokens = text.split()
        tagged_tokens: List[str] = []

        for token in tokens:
            clean = re.sub(r"[^\u0900-\u097F]", "", token)
            if clean:
                confidence = self._score_word(clean)
                if confidence >= self.confidence_threshold:
                    token = f"[EN]{token}[/EN]"
            tagged_tokens.append(token)

        return " ".join(tagged_tokens)

    def tag_batch(self, sentences: List[str]) -> List[str]:
        """Apply tag() to a list of sentences."""
        return [self.tag(s) for s in sentences]

    def annotate(self, text: str) -> Dict[str, object]:
        """
        Full annotation for a sentence.

        Returns:
            Dict with:
              - original: input text
              - tagged: tagged version
              - detected_words: list of (word, confidence) pairs
              - count: number of detected English words
        """
        tagged = self.tag(text)
        detected = self.detect(text)
        return {
            "original": text,
            "tagged": tagged,
            "detected_words": detected,
            "count": len(detected),
        }
