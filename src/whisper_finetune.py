"""
whisper_finetune.py
-------------------
Fine-tuning pipeline for OpenAI Whisper on Hindi ASR datasets.

Implements:
  - WhisperFinetuner class wrapping HuggingFace Trainer
  - Data collator with dynamic padding
  - Training arguments configuration
  - Checkpoint saving and loading helpers
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import DatasetDict
import evaluate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Dynamic padding collator for Whisper fine-tuning.

    Pads input features to the fixed Whisper input size (3000 frames),
    and pads label sequences to the longest label in the batch.

    Args:
        processor: WhisperProcessor (feature extractor + tokenizer).
        decoder_start_token_id: Token used to start decoder sequence.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate input features from labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features (Whisper expects fixed 3000-frame log-mel)
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels and replace padding token with -100 (ignored in loss)
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Slice off the decoder start token if prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Fine-tuner
# ---------------------------------------------------------------------------

class WhisperFinetuner:
    """
    End-to-end Whisper fine-tuning wrapper.

    Usage:
        finetuner = WhisperFinetuner(model_name="openai/whisper-small",
                                     output_dir="outputs/whisper-hindi")
        finetuner.setup(processed_dataset)
        finetuner.train()
        wer = finetuner.evaluate()

    Args:
        model_name: HuggingFace model name or local checkpoint.
        output_dir: Directory to save model checkpoints and logs.
        language: Language for Whisper (e.g. "hindi").
        task: "transcribe" or "translate".
        num_train_epochs: Number of full training passes.
        per_device_train_batch_size: Batch size per GPU/CPU for training.
        per_device_eval_batch_size: Batch size per GPU/CPU for evaluation.
        learning_rate: Peak learning rate.
        warmup_steps: Number of warmup steps for LR scheduler.
        gradient_accumulation_steps: Steps before optimizer update.
        fp16: Use 16-bit mixed precision training (requires GPU).
        max_steps: If set, overrides num_train_epochs.
        eval_steps: Run evaluation every N steps.
        save_steps: Save checkpoint every N steps.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        output_dir: str = "outputs/whisper-hindi",
        language: str = "hindi",
        task: str = "transcribe",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 1e-5,
        warmup_steps: int = 200,
        gradient_accumulation_steps: int = 2,
        fp16: bool = False,
        max_steps: int = -1,
        eval_steps: int = 200,
        save_steps: int = 200,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.language = language
        self.task = task
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        self.model: Optional[WhisperForConditionalGeneration] = None
        self.processor: Optional[WhisperProcessor] = None
        self.trainer: Optional[Seq2SeqTrainer] = None
        self.dataset: Optional[DatasetDict] = None
        self.wer_metric = evaluate.load("wer")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, processed_dataset: DatasetDict) -> None:
        """
        Initialise model, processor, data collator, and trainer.

        Args:
            processed_dataset: DatasetDict with 'train' and 'validation'
                splits, already preprocessed (input_features + labels).
        """
        self.dataset = processed_dataset

        logger.info("Loading model: %s", self.model_name)
        self.processor = WhisperProcessor.from_pretrained(
            self.model_name, language=self.language, task=self.task
        )
        processor = self.processor
        assert processor is not None, "Processor failed to load"
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name)

        # Force language and task tokens to avoid model defaulting to English
        self.model.generation_config.language = self.language
        self.model.generation_config.task = self.task
        self.model.generation_config.forced_decoder_ids = None

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.num_train_epochs,
            max_steps=self.max_steps,
            gradient_checkpointing=True,
            fp16=self.fp16,
            eval_strategy="steps",
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            tokenizer=processor.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Trainer configured. Ready to train.")

    def _compute_metrics(self, pred) -> Dict[str, float]:
        """Internal metric computation called by Trainer after each eval step."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 (padding) with pad token so decoding works
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the fine-tuning loop."""
        if self.trainer is None:
            raise RuntimeError("Call setup() before train().")
        logger.info("Starting training …")
        self.trainer.train()
        logger.info("Training complete. Saving model to '%s'.", self.output_dir)
        self.trainer.save_model(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, split: str = "test") -> Dict[str, float]:
        """
        Run generation on a dataset split and compute WER.

        Args:
            split: Dataset split key to evaluate on.

        Returns:
            Dict with keys 'wer', 'predictions', 'references'.
        """
        if self.trainer is None or self.dataset is None:
            raise RuntimeError("Call setup() before evaluate().")

        if split not in self.dataset:
            raise KeyError(f"Split '{split}' not in dataset. Available: {list(self.dataset.keys())}")

        logger.info("Evaluating on '%s' split …", split)
        predictions_output = self.trainer.predict(self.dataset[split])

        pred_ids = predictions_output.predictions
        label_ids = predictions_output.label_ids
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.batch_decode(label_ids, skip_special_tokens=True)

        wer = self.wer_metric.compute(predictions=pred_str, references=label_str)
        logger.info("WER on '%s': %.4f", split, wer)

        return {
            "wer": round(wer, 4),
            "predictions": pred_str,
            "references": label_str,
        }

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a saved checkpoint for inference or continued training.

        Args:
            checkpoint_path: Path to the saved Whisper model directory.
        """
        logger.info("Loading checkpoint from '%s'.", checkpoint_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        self.processor = WhisperProcessor.from_pretrained(checkpoint_path)

    # ------------------------------------------------------------------
    # Transcribe helper
    # ------------------------------------------------------------------

    def transcribe(self, waveform: np.ndarray, sampling_rate: int = 16_000) -> str:
        """
        Transcribe a single waveform using the current model.

        Args:
            waveform: Audio waveform as float32 numpy array.
            sampling_rate: Sampling rate of the waveform.

        Returns:
            Transcription string.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call setup() or load_checkpoint().")

        inputs = self.processor(
            waveform, sampling_rate=sampling_rate, return_tensors="pt"
        )
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                language=self.language,
                task=self.task,
            )
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
