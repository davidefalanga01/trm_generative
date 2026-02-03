from dataclasses import dataclass, field
from typing import Any, Optional, Iterable, Dict
import itertools
import os

import numpy as np
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

from tokenizer_configs import get_tokenizer_for_dataset

@dataclass
class GSM8KDatasetConfig:
    seed: int = 42
    seq_len: int = 512  
    max_prompt: int = 256
    max_answer: int = 256
    batch_size: int = 32
    streaming: bool = True
    tokenizer_name: Optional[str] = None
    cache_dir: Optional[str] = "data/hf/gsm8k"

@dataclass
class TextDatasetMetadata:
    pad_id: int
    vocab_size: int
    seq_len: int
    num_dataset_identifiers: int = 1
    total_groups: int = 1
    mean_dataset_examples: int = 0
    sets: list = field(default_factory=lambda: ["train"])
    ignore_label_id: Any = None
    blank_identifier_id: int = 0

class GSM8KDataset(IterableDataset):
    """Question -> CoT + Answer"""

    def __init__(self, config: GSM8KDatasetConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.tokenizer = get_tokenizer_for_dataset("gsm8k", config.tokenizer_name)
        
        self.dataset = load_dataset(
            "openai/gsm8k", 
            "main", 
            split=self.split, 
            streaming=config.streaming,
            cache_dir=config.cache_dir
        )
        
        self.metadata = TextDatasetMetadata(
            pad_id=self.tokenizer.pad_token_id,
            vocab_size=self.tokenizer.vocab_size,
            seq_len=config.seq_len,
            sets=[split]
        )
        self._epoch = 0

    def _tokenize_example(self, question: str, answer: str):
        """
        Tokenize prompt and answer separately to mask prompt tokens in the loss.
        """
        # Format Prompt/Response
        full_prompt = f"Question: {question.strip()}\nAnswer: "
        full_answer = f"{answer.strip()}{self.tokenizer.eos_token}"
        
        prompt_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(full_answer, add_special_tokens=False)
        
        if len(prompt_ids) + len(answer_ids) > self.config.seq_len:
            if len(prompt_ids) >= self.config.max_prompt:
                # Truncate question
                prompt_ids = prompt_ids[-self.config.max_prompt:]
            
            remaining = self.config.seq_len - len(prompt_ids)
            if len(answer_ids) > remaining:
                # Truncate answer
                answer_ids = answer_ids[:self.config.max_answer]

        print(f"Tokenized Prompt IDs: {len(prompt_ids)} tokens")
        print(f"Tokenized Answer IDs: {len(answer_ids)} tokens")
        
        # Labels: -100 on prompt tokens, value on answer tokens
        combined_ids = prompt_ids + answer_ids
        labels = ([-100] * len(prompt_ids)) + answer_ids
        
        # Padding
        pad_len = self.config.seq_len - len(combined_ids)
        if pad_len > 0:
            combined_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            
        return np.array(combined_ids, dtype=np.int32), np.array(labels, dtype=np.int32)

    def __iter__(self):
        self._epoch += 1
        # Shuffling for streaming dataset
        if self.config.streaming:
            dataset_iter = self.dataset.shuffle(seed=self.config.seed + self._epoch, buffer_size=2500)
        else:
            dataset_iter = self.dataset

        batch_inputs = []
        batch_labels = []

        for item in dataset_iter:
            q, a = item["question"], item["answer"]
            
            input_ids, labels = self._tokenize_example(q, a)
            
            batch_inputs.append(input_ids)
            batch_labels.append(labels)

            if len(batch_inputs) == self.config.batch_size:
                yield {
                    "inputs": torch.from_numpy(np.stack(batch_inputs)).long(),
                    "labels": torch.from_numpy(np.stack(batch_labels)).long(),
                    "dataset_ids": torch.zeros(self.config.batch_size, dtype=torch.long),
                    "puzzle_identifiers": torch.zeros(self.config.batch_size, dtype=torch.long),
                }
                batch_inputs, batch_labels = [], []

        if batch_inputs:
            yield {
                "inputs": torch.from_numpy(np.stack(batch_inputs)).long(),
                "labels": torch.from_numpy(np.stack(batch_labels)).long(),
                "dataset_ids": torch.zeros(len(batch_inputs), dtype=torch.long),
                "puzzle_identifiers": torch.zeros(len(batch_inputs), dtype=torch.long),
            }
