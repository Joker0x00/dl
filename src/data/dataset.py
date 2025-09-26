# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from transformers import AutoTokenizer
import json

class BertDataset(Dataset):
    def __init__(self, file: str, tokenizer_name: str, max_length: int = 128):
        self.samples = [json.loads(line) for line in open(file, encoding="utf-8")]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text, label = sample["text"], int(sample["label"])
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]).squeeze(0)),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def build_dataloader(train_file, val_file, test_file, batch_size, num_workers, pin_memory, shuffle, transform=None):

    train_ds = BertDataset(train_file, transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    val_ds = BertDataset(val_file, transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_ds, test_loader = None
    if test_file != None:
        test_ds = BertDataset(test_file, transform)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader