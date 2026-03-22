# 🎙️ AI Researcher Intern - Speech & Audio Assignment

A complete, production-quality implementation of Hindi ASR research tasks:

## 📌 Tasks Covered

1. **Task 1** - Whisper Fine-Tuning + WER Evaluation + Error Analysis
2. **Task 2** - ASR Cleanup Pipeline (Number Normalisation + English Detection)
3. **Task 3** - Hindi Spelling Detection with Confidence Scoring
4. **Task 4** - Lattice-Based Fair WER Evaluation

---

## 📁 Project Structure

```
joshTalk/
│
├── data/
│   ├── raw/                   
│   └── processed/             
│
├── notebooks/
│   ├── task_01_asr_finetuning.ipynb
│   ├── task_02_cleanup_pipeline.ipynb
│   ├── task_03_spelling_detection.ipynb
│   └── task_04_lattice_evaluation.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── whisper_finetune.py
│   ├── evaluation.py
│   ├── error_analysis.py
│   ├── number_normalizer.py
│   ├── english_detector.py
│   ├── spelling_checker.py
│   └── lattice_builder.py
│
├── outputs/
├── reports/
│   └── final_report.md
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Setup

### 1. Navigate to project

```bash
cd path/to/joshTalk
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### 🔹 Task 1 - Fine-Tuning + Evaluation

```bash
jupyter notebook notebooks/task_01_asr_finetuning.ipynb
```

Includes:

* Dataset loading (~10h Hindi ASR data)
* Baseline Whisper-small evaluation
* Fine-tuning
* WER comparison
* Error analysis (25 samples)
* Post-processing improvement

Outputs:

* `outputs/wer_results.csv`
* `outputs/error_samples.csv`
* `outputs/task_01_results.png`

---

### 🔹 Task 2 - Cleanup Pipeline

```bash
jupyter notebook notebooks/task_02_cleanup_pipeline.ipynb
```

Includes:

* Hindi number normalization (compound + edge cases)
* English word detection in Devanagari
* Before/after pipeline results

---

### 🔹 Task 3 - Spelling Detection

```bash
jupyter notebook notebooks/task_03_spelling_detection.ipynb
```

Includes:

* Dictionary + edit distance approach
* Confidence scoring
* Low-confidence analysis
* Evaluation on dataset

---

### 🔹 Task 4 - Lattice Evaluation

```bash
jupyter notebook notebooks/task_04_lattice_evaluation.ipynb
```

Includes:

* Multi-model alignment
* Lattice construction
* Flexible WER computation
* Agreement-based scoring

---

## 📊 Results

### WER Comparison

| Model                  | WER  |
| ---------------------- | ---- |
| Baseline Whisper-small | ~58% |
| Fine-tuned model       | ~37% |
| + Cleanup pipeline     | ~35% |

---

### Error Taxonomy

| Type         | Count |
| ------------ | ----- |
| Substitution | 10    |
| Phonetic     | 7     |
| Deletion     | 4     |
| Number Error | 2     |
| English Mix  | 2     |

---

### Lattice WER

| Metric  | Standard WER | Lattice WER |
| ------- | ------------ | ----------- |
| Average | 0.300        | 0.170       |

---

## 🔑 Key Insights

* Fine-tuning improves Hindi ASR performance significantly
* Number normalization gives high impact with low effort
* Code-mixed Hindi-English speech requires special handling
* Lattice-based evaluation provides fairer performance measurement
* Combining phonetics with edit distance improves spelling detection

---

## 🧩 Extensions

* Upgrade to Whisper-medium or larger models
* Add full Hindi dictionary
* Extend lattice system for more models
* Improve number parsing

---

## 📦 Requirements

* Python 3.9+
* PyTorch 2.0+
* HuggingFace Transformers & Datasets
* jiwer
* librosa
* pandas, matplotlib, seaborn

See `requirements.txt` for full details.
