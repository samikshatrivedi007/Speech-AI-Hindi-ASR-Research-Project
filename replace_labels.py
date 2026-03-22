import os
import re
from glob import glob

replacements = [
    (re.compile(r'\bQuestion 1\b', re.IGNORECASE), "Task 1"),
    (re.compile(r'\bQuestion 2\b', re.IGNORECASE), "Task 2"),
    (re.compile(r'\bQuestion 3\b', re.IGNORECASE), "Task 3"),
    (re.compile(r'\bQuestion 4\b', re.IGNORECASE), "Task 4"),
    (re.compile(r'\bQ1\b'), "Task 1"),
    (re.compile(r'\bQ2\b'), "Task 2"),
    (re.compile(r'\bQ3\b'), "Task 3"),
    (re.compile(r'\bQ4\b'), "Task 4"),
    # Update file references
    (re.compile(r'\bQ1_finetuning\.ipynb\b'), "task_01_asr_finetuning.ipynb"),
    (re.compile(r'\bQ2_cleanup_pipeline\.ipynb\b'), "task_02_cleanup_pipeline.ipynb"),
    (re.compile(r'\bQ3_spelling_detection\.ipynb\b'), "task_03_spelling_detection.ipynb"),
    (re.compile(r'\bQ4_lattice\.ipynb\b'), "task_04_lattice_evaluation.ipynb"),
    # Common lower case cases
    (re.compile(r'\bq1_results\b'), "task_01_results"),
    (re.compile(r'\bq2_cleanup_wer\b'), "task_02_cleanup_wer"),
    (re.compile(r'\bq3_spelling_distribution\b'), "task_03_spelling_distribution"),
    (re.compile(r'\bq3_words\b'), "task_03_words"),
    (re.compile(r'\bq4_lattice_wer_comparison\b'), "task_04_lattice_wer_comparison"),
    (re.compile(r'\bq4_discount_ablation\b'), "task_04_discount_ablation"),
    (re.compile(r'\bq2_asr_pairs\b'), "task_02_asr_pairs"),
]

# Files to update
target_files = []
target_files.extend(glob("README.md"))
target_files.extend(glob("reports/final_report.md"))
target_files.extend(glob("notebooks/*.ipynb"))
target_files.extend(glob(r'C:\Users\samik\.gemini\antigravity\brain\*\*.md'))

artifact_path = r"C:\Users\samik\.gemini\antigravity\brain\d252dc1b-b892-49e9-abf2-15a081a99996"
target_files.extend(glob(os.path.join(artifact_path, "*.md")))


for file_path in target_files:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content
    for pattern, replacement in replacements:
        content = pattern.sub(replacement, content)

    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {file_path}")
print("Done!")
