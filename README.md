# AI Researcher Intern â€” Speech & Audio Assignment

A complete, production-quality implementation of four Hindi ASR research tasks:

1. **Task 1** â€“ Whisper Fine-Tuning + WER Evaluation + Error Analysis
2. **Task 2** â€“ ASR Output Cleanup Pipeline (Number Normalisation + English Detection)
3. **Task 3** â€“ Hindi Spelling Correctness Detection with Confidence Scoring
4. **Task 4** â€“ Lattice-Based Fair WER Evaluation System

---

## ðŸ“ Project Structure

```
joshTalk/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/                   # Downloaded audio + cached HuggingFace datasets
â”‚   â””â”€â”€ processed/             # Manifest CSVs, processed features
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ task_01_asr_finetuning.ipynb          # ASR pipeline end-to-end
â”‚   â”œâ”€â”€ task_02_cleanup_pipeline.ipynb    # Number norm + English tagging
â”‚   â”œâ”€â”€ task_03_spelling_detection.ipynb  # Spelling classifier
â”‚   â””â”€â”€ task_04_lattice_evaluation.ipynb             # Lattice WER system
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_loader.py         # HuggingFace + CSV + URL dataset loaders
â”‚   â”œâ”€â”€ preprocessing.py       # Audio + text preprocessing for Whisper
â”‚   â”œâ”€â”€ whisper_finetune.py    # Fine-tuning pipeline (Seq2SeqTrainer)
â”‚   â”œâ”€â”€ evaluation.py          # WER/CER computation + CSV export
â”‚   â”œâ”€â”€ error_analysis.py      # Dynamic error taxonomy + systematic sampling
â”‚   â”œâ”€â”€ number_normalizer.py   # Hindi number word â†’ digit converter
â”‚   â”œâ”€â”€ english_detector.py    # English loanword detector with [EN] tagging
â”‚   â”œâ”€â”€ spelling_checker.py    # Dictionary + edit distance spelling checker
â”‚   â””â”€â”€ lattice_builder.py     # DP alignment + lattice WER computer
â”‚
â”œâ”€â”€ outputs/                   # Generated CSVs, plots, model checkpoints
â”œâ”€â”€ reports/
â”‚   â””â”€â”€ final_report.md        # Full methodology + results report
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
```

---

## âš¡ Quick Setup

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

## ðŸš€ How to Run Each Question

### Task 1 â€” Fine-Tuning + WER Evaluation

```bash
jupyter notebook notebooks/task_01_asr_finetuning.ipynb
```

**Steps inside the notebook:**
1. Loads real JoshTalks ~10h dataset using the GCS URL manifests and segment JSON schemas
2. Computes baseline WER with unmodified Whisper-small against FLEURS test set
3. Fine-tunes on JoshTalks training data (GPU recommended)
4. Evaluates fine-tuned model
5. Runs systematic error analysis on 25 samples
6. Applies number normalisation as a post-processing improvement

**Outputs:**
- `outputs/wer_results.csv` â€” baseline vs fine-tuned WER
- `outputs/error_samples.csv` â€” annotated error samples
- `outputs/task_01_results.png` â€” WER comparison + error taxonomy chart

### Task 2 â€” Cleanup Pipeline

```bash
jupyter notebook notebooks/task_02_cleanup_pipeline.ipynb
```

**Covers:**
- 15+ Hindi number normalisation test cases including compound numbers and idiom protection
- English word detection at tunable confidence thresholds
- Combined before/after pipeline on representative ASR outputs

### Task 3 â€” Spelling Detection

```bash
jupyter notebook notebooks/task_03_spelling_detection.ipynb
```

**Covers:**
- Single-word examples with full reasoning
- Batch classification of real 1.77 Lakh Hindi words directly from the assignment spreadsheet
- Confidence distribution visualisation
- Low-confidence failure pattern analysis
- Accuracy evaluation on labelled ground-truth samples

### Task 4 â€” Lattice WER System

```bash
jupyter notebook notebooks/task_04_lattice_evaluation.ipynb
```

**Covers:**
- Build word-level lattice from 6 ASR hypothesis columns (Human + Models H, i, k, l, m, n) from spreadsheet
- Inspect per-position alternatives and alignment logic
- Compute standard WER vs lattice WER per model
- Agreement-discount ablation (0.0 â†’ 1.0)

---

## ðŸ“Š Sample Outputs

### WER Comparison (Task 1)

| Model | WER |
|---|---|
| Baseline Whisper-small | ~58% |
| Fine-tuned (500 steps) | ~37% |
| + Number Normalisation | ~35% |

### Error Taxonomy (Task 1)

| Type | Count |
|---|---|
| substitution | 10 |
| phonetic | 7 |
| deletion | 4 |
| number_error | 2 |
| english_mix | 2 |

### Lattice WER vs Standard WER (Task 4)

| Utterance | Standard WER | Lattice WER |
|---|---|---|
| Sentence 1 | 0.250 | 0.125 |
| Sentence 2 | 0.333 | 0.167 |
| Corpus avg | 0.300 | 0.170 |

---

## ðŸ”‘ Key Insights

- **Fine-tuning on domain data is essential** â€” Whisper-small's Hindi generalisation is limited without task-specific supervision
- **Number normalisation is a high-ROI fix** â€” cheap post-processing that reduces a significant fraction of lexical WER
- **Code-switching (Hindiâ€“English) is pervasive** â€” any Hindi ASR system must handle Devanagari transliterations of English words
- **Lattice WER is a fairer metric** in multi-model settings where the reference may lag behind consistent model consensus
- **Edit distance + phonetics > dictionary alone** â€” the phonetic encoder recovers cases where the dictionary is missing inflected/variant forms

---

## ðŸ§© Extending the Project

| Task | How to Extend |
|---|---|
| Larger model | Change `MODEL_NAME = 'openai/whisper-medium'` in Task 1 |
| Full Hindi dictionary | Pass `dictionary_path=` to `HindiSpellingChecker` |
| More ASR models in lattice | Add more strings to `hypotheses` list in Task 4 |
| Custom error categories | Add new `(label, predicate, reasoning)` tuples to `ErrorClassifier._rules` |
| External Morphological Analyzer | Integrate IndicNLP to correctly categorise inflected valid Hindi words in Task 3 |

---

## ðŸ“¦ Requirements Summary

- Python 3.9+
- PyTorch 2.0+ (CUDA optional but recommended for Task 1)
- HuggingFace Transformers, Datasets, Evaluate
- jiwer (WER computation)
- editdistance (spelling checker)
- librosa, soundfile (audio processing)
- pandas, matplotlib, seaborn (analysis + visualisation)

Full list: see `requirements.txt`
