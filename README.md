# AI Researcher Intern — Speech & Audio Assignment

A complete, production-quality implementation of four Hindi ASR research tasks:

1. **Q1** – Whisper Fine-Tuning + WER Evaluation + Error Analysis
2. **Q2** – ASR Output Cleanup Pipeline (Number Normalisation + English Detection)
3. **Q3** – Hindi Spelling Correctness Detection with Confidence Scoring
4. **Q4** – Lattice-Based Fair WER Evaluation System

---

## 📁 Project Structure

```
joshTalk/
├── data/
│   ├── raw/                   # Downloaded audio + cached HuggingFace datasets
│   └── processed/             # Manifest CSVs, processed features
│
├── notebooks/
│   ├── Q1_finetuning.ipynb          # ASR pipeline end-to-end
│   ├── Q2_cleanup_pipeline.ipynb    # Number norm + English tagging
│   ├── Q3_spelling_detection.ipynb  # Spelling classifier
│   └── Q4_lattice.ipynb             # Lattice WER system
│
├── src/
│   ├── data_loader.py         # HuggingFace + CSV + URL dataset loaders
│   ├── preprocessing.py       # Audio + text preprocessing for Whisper
│   ├── whisper_finetune.py    # Fine-tuning pipeline (Seq2SeqTrainer)
│   ├── evaluation.py          # WER/CER computation + CSV export
│   ├── error_analysis.py      # Dynamic error taxonomy + systematic sampling
│   ├── number_normalizer.py   # Hindi number word → digit converter
│   ├── english_detector.py    # English loanword detector with [EN] tagging
│   ├── spelling_checker.py    # Dictionary + edit distance spelling checker
│   └── lattice_builder.py     # DP alignment + lattice WER computer
│
├── outputs/                   # Generated CSVs, plots, model checkpoints
├── reports/
│   └── final_report.md        # Full methodology + results report
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Setup

### 1. Clone / navigate to the project

```bash
cd path/to/joshTalk
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run on Google Colab

Upload the folder to Google Drive, then open any notebook and run the first cell which has the `!pip install` commands.

---

## 🚀 How to Run Each Question

### Q1 — Fine-Tuning + WER Evaluation

```bash
jupyter notebook notebooks/Q1_finetuning.ipynb
```

**Steps inside the notebook:**
1. Loads real JoshTalks ~10h dataset using the GCS URL manifests and segment JSON schemas
2. Computes baseline WER with unmodified Whisper-small against FLEURS test set
3. Fine-tunes on JoshTalks training data (GPU recommended)
4. Evaluates fine-tuned model
5. Runs systematic error analysis on 25 samples
6. Applies number normalisation as a post-processing improvement

**Outputs:**
- `outputs/wer_results.csv` — baseline vs fine-tuned WER
- `outputs/error_samples.csv` — annotated error samples
- `outputs/q1_results.png` — WER comparison + error taxonomy chart

### Q2 — Cleanup Pipeline

```bash
jupyter notebook notebooks/Q2_cleanup_pipeline.ipynb
```

**Covers:**
- 15+ Hindi number normalisation test cases including compound numbers and idiom protection
- English word detection at tunable confidence thresholds
- Combined before/after pipeline on representative ASR outputs

### Q3 — Spelling Detection

```bash
jupyter notebook notebooks/Q3_spelling_detection.ipynb
```

**Covers:**
- Single-word examples with full reasoning
- Batch classification of real 1.77 Lakh Hindi words directly from the assignment spreadsheet
- Confidence distribution visualisation
- Low-confidence failure pattern analysis
- Accuracy evaluation on labelled ground-truth samples

### Q4 — Lattice WER System

```bash
jupyter notebook notebooks/Q4_lattice.ipynb
```

**Covers:**
- Build word-level lattice from 6 ASR hypothesis columns (Human + Models H, i, k, l, m, n) from spreadsheet
- Inspect per-position alternatives and alignment logic
- Compute standard WER vs lattice WER per model
- Agreement-discount ablation (0.0 → 1.0)

---

## 📊 Sample Outputs

### WER Comparison (Q1)

| Model | WER |
|---|---|
| Baseline Whisper-small | ~58% |
| Fine-tuned (500 steps) | ~37% |
| + Number Normalisation | ~35% |

### Error Taxonomy (Q1)

| Type | Count |
|---|---|
| substitution | 10 |
| phonetic | 7 |
| deletion | 4 |
| number_error | 2 |
| english_mix | 2 |

### Lattice WER vs Standard WER (Q4)

| Utterance | Standard WER | Lattice WER |
|---|---|---|
| Sentence 1 | 0.250 | 0.125 |
| Sentence 2 | 0.333 | 0.167 |
| Corpus avg | 0.300 | 0.170 |

---

## 🔑 Key Insights

- **Fine-tuning on domain data is essential** — Whisper-small's Hindi generalisation is limited without task-specific supervision
- **Number normalisation is a high-ROI fix** — cheap post-processing that reduces a significant fraction of lexical WER
- **Code-switching (Hindi–English) is pervasive** — any Hindi ASR system must handle Devanagari transliterations of English words
- **Lattice WER is a fairer metric** in multi-model settings where the reference may lag behind consistent model consensus
- **Edit distance + phonetics > dictionary alone** — the phonetic encoder recovers cases where the dictionary is missing inflected/variant forms

---

## 🧩 Extending the Project

| Task | How to Extend |
|---|---|
| Larger model | Change `MODEL_NAME = 'openai/whisper-medium'` in Q1 |
| Full Hindi dictionary | Pass `dictionary_path=` to `HindiSpellingChecker` |
| More ASR models in lattice | Add more strings to `hypotheses` list in Q4 |
| Custom error categories | Add new `(label, predicate, reasoning)` tuples to `ErrorClassifier._rules` |
| External Morphological Analyzer | Integrate IndicNLP to correctly categorise inflected valid Hindi words in Q3 |

---

## 📦 Requirements Summary

- Python 3.9+
- PyTorch 2.0+ (CUDA optional but recommended for Q1)
- HuggingFace Transformers, Datasets, Evaluate
- jiwer (WER computation)
- editdistance (spelling checker)
- librosa, soundfile (audio processing)
- pandas, matplotlib, seaborn (analysis + visualisation)

Full list: see `requirements.txt`
