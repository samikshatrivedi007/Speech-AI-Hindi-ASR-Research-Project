"""
smoke_test.py
-------------
Quick sanity tests for all 6 core modules.
Run from the project root:   python smoke_test.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

PASS = "\u2713"
FAIL = "\u2717"
errors = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}  {detail}")
        errors.append(name)


# ─── 1. number_normalizer ────────────────────────────────────────────────────
print("\n[1] number_normalizer")
from number_normalizer import HindiNumberNormalizer, HINDI_NUMBER_WORDS

n = HindiNumberNormalizer(preserve_idioms=True)
check("simple unit",        n.normalize("दस साल")          == "10 साल")
check("compound (3×100+54)",n.normalize("तीन सौ चौवन")      == "354")
check("scale (हजार)",       n.normalize("पाँच हजार")        == "5000")
check("fused ten+unit",     n.normalize("पच्चीस मिनट")      == "25 मिनट")
check("idiom preserved",    n.normalize("दो-चार बातें")      == "दो-चार बातें")
check("HINDI_NUMBER_WORDS set has दस", "दस" in HINDI_NUMBER_WORDS)

# ─── 2. english_detector ─────────────────────────────────────────────────────
print("\n[2] english_detector")
from english_detector import EnglishDevanagariDetector, ENGLISH_DEVANAGARI_DICT

det = EnglishDevanagariDetector(confidence_threshold=0.5)

# Use words confirmed to be in the dictionary
check("कंप्यूटर in ENGLISH_DEVANAGARI_DICT",  "कंप्यूटर" in ENGLISH_DEVANAGARI_DICT)
check("इंटरव्यू in ENGLISH_DEVANAGARI_DICT",  "इंटरव्यू" in ENGLISH_DEVANAGARI_DICT)
check("जॉब in ENGLISH_DEVANAGARI_DICT",       "जॉब"       in ENGLISH_DEVANAGARI_DICT)
check("मोबाइल in ENGLISH_DEVANAGARI_DICT",    "मोबाइल"    in ENGLISH_DEVANAGARI_DICT)

tagged = det.tag("मेरा इंटरव्यू अच्छा गया और मुझे जॉब मिल गई")
check("tag() adds [EN] markers",              "[EN]" in tagged, f"got: {tagged!r}")
check("इंटरव्यू detected as English",         any(w == "इंटरव्यू" for w, _ in det.detect("इंटरव्यू अच्छा था")))
check("मोबाइल detected as English",           any(w == "मोबाइल"    for w, _ in det.detect("मोबाइल चार्ज करो")))
check("tag_batch() returns list",             isinstance(det.tag_batch(["टेस्ट"]), list))

# ─── 3. spelling_checker ─────────────────────────────────────────────────────
print("\n[3] spelling_checker")
from spelling_checker import HindiSpellingChecker

chk = HindiSpellingChecker()
# English loanword bypass — must be in ENGLISH_DEVANAGARI_DICT
r_loan = chk.check_word("मोबाइल")
check("loanword मोबाइल → correct",   r_loan.label == "correct",
      f"got label={r_loan.label}")
check("loanword confidence=high",    r_loan.confidence == "high")

r_loan2 = chk.check_word("इंटरव्यू")
check("loanword इंटरव्यू → correct", r_loan2.label == "correct",
      f"got label={r_loan2.label}")

# Known Hindi word (in seed dictionary)
r_dict = chk.check_word("नमस्ते")
check("नमस्ते → correct (dict hit)",  r_dict.label == "correct")

# Likely misspelling
r_bad = chk.check_word("नमसते")    # missing ् (halant)
check("नमसते → incorrect",           r_bad.label == "incorrect")

# Empty string
r_empty = chk.check_word("")
check("empty string → incorrect",    r_empty.label == "incorrect")

# ─── 4. lattice_builder ──────────────────────────────────────────────────────
print("\n[4] lattice_builder")
from lattice_builder import LatticeBuilder, LatticeWERComputer, build_and_evaluate

builder  = LatticeBuilder()
computer = LatticeWERComputer(agreement_discount=0.5, min_hypotheses_for_discount=2)

hyps = [
    "उसने चौदह किताबें खरीदीं",
    "उसने 14 किताबें खरीदी",
    "उसने चौदह पुस्तकें खरीदीं",
]
lat = builder.build(hyps)
check("lattice has ≥4 nodes",          len(lat) >= 4,  f"got {len(lat)}")
check("position 0 has उसने",           lat[0].matches("उसने"))
check("position 1 has both चौदह & 14", "चौदह" in lat[1].alternatives
                                       and "14" in lat[1].alternatives)

result = build_and_evaluate(hyps, "उसने चौदह किताबें खरीदीं")
check("lattice_wer ≤ standard_wer",   result["lattice_wer"] <= result["standard_wer"],
      f"lat={result['lattice_wer']} std={result['standard_wer']}")
check("lattice_bins is list",          isinstance(result["lattice_bins"], list))

# alignment: longer backbone nodes not dropped
hyps2 = ["एक दो तीन चार पाँच", "एक दो तीन"]
lat2 = builder.build(hyps2)
check("backbone nodes not dropped",    len(lat2) >= 5, f"got {len(lat2)}")

# ─── 5. error_analysis ───────────────────────────────────────────────────────
print("\n[5] error_analysis")
from error_analysis import ErrorAnalyzer, ErrorClassifier

ec = ErrorClassifier()
labels, reason = ec.classify("पाँच सौ रुपये", "500 रुपये")
check("number_error detected",         "number_error" in labels, f"got {labels}")

labels2, _ = ec.classify("मैं कल दिल्ली जाऊँगा", "मैंने कल दिल्ली जाया")
check("substitution detected",         len(labels2) > 0)

# ─── 6. evaluation ───────────────────────────────────────────────────────────
print("\n[6] evaluation")
from evaluation import compute_wer, compute_cer, evaluate_samples, WERResult

check("WER identical strings = 0",    compute_wer(["हेलो वर्ल्ड"], ["हेलो वर्ल्ड"]) == 0.0)
check("WER completely wrong > 0",     compute_wer(["अ"], ["ब"]) > 0.0)
check("CER identical = 0",            compute_cer(["test"], ["test"]) == 0.0)
samples = evaluate_samples(["हेलो"], ["हेलो वर्ल्ड"])
check("evaluate_samples returns list", len(samples) == 1)
r = WERResult(split="test", num_samples=1, wer=0.25, cer=0.10)
check("WERResult.wer_pct",            r.wer_pct == "25.00%")

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"FAILED: {len(errors)} test(s) failed: {errors}")
    sys.exit(1)
else:
    total = 6 + 4 + 4 + 6 + 2 + 5   # rough count
    print(f"All smoke tests PASSED {PASS}")
