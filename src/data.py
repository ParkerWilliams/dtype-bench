"""
Data Loading for Dtype Benchmark

Contains data loaders for:
- OpenWebText: Streaming dataset for language modeling (GPT-2 CLM, RoBERTa MLM)
- EUR-Lex: Multi-label document classification with dynamic collator
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm


# =============================================================================
# OpenWebText Dataset (Language Modeling)
# =============================================================================

class OpenWebTextDataset:
    """
    Streaming dataset for OpenWebText from HuggingFace.

    Tokenizes and stores in memory for fast random access during training.

    Usage:
        dataset = OpenWebTextDataset(
            subset_size=100000,
            block_size=1024,
            tokenizer_type="gpt2"
        )
        x, y = dataset.get_batch(batch_size=12)
    """

    def __init__(
        self,
        subset_size: int = 100000,
        block_size: int = 1024,
        tokenizer_type: str = "gpt2",  # "gpt2" or "roberta"
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            subset_size: Number of documents to load (controls memory/time)
            block_size: Sequence length for batches
            tokenizer_type: "gpt2" for tiktoken, "roberta" for HuggingFace
            split: Dataset split (only "train" available for OpenWebText)
            cache_dir: Optional cache directory for tokenized data
        """
        self.block_size = block_size
        self.tokenizer_type = tokenizer_type

        # Setup tokenizer
        if tokenizer_type == "gpt2":
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
            self.encode = lambda x: self.enc.encode(x, allowed_special={'<|endoftext|>'})
            self.vocab_size = self.enc.n_vocab
        else:
            from transformers import AutoTokenizer
            self.enc = AutoTokenizer.from_pretrained("roberta-base")
            self.encode = lambda x: self.enc.encode(x, add_special_tokens=False)
            self.vocab_size = self.enc.vocab_size

        # Check for cached data
        cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"owt_{tokenizer_type}_{subset_size}.npy")
            if os.path.exists(cache_path):
                print(f"Loading cached tokenized data from {cache_path}")
                self.data = np.load(cache_path)
                print(f"Loaded {len(self.data):,} tokens")
                return

        # Load and tokenize from HuggingFace
        print(f"Loading OpenWebText subset ({subset_size:,} documents)...")
        from datasets import load_dataset

        # Load subset
        ds = load_dataset(
            "openwebtext",
            split=f"train[:{subset_size}]",
            trust_remote_code=True
        )

        # Tokenize all documents
        print("Tokenizing...")
        all_tokens = []
        for example in tqdm(ds, desc="Tokenizing"):
            tokens = self.encode(example["text"])
            all_tokens.extend(tokens)

        self.data = np.array(all_tokens, dtype=np.uint16)
        print(f"Total tokens: {len(self.data):,}")

        # Cache if path provided
        if cache_path:
            np.save(cache_path, self.data)
            print(f"Cached tokenized data to {cache_path}")

    def get_batch(self, batch_size: int, device: str = "cuda"):
        """
        Get a random batch of sequences.

        Args:
            batch_size: Number of sequences
            device: Target device

        Returns:
            x: Input tokens [batch, block_size]
            y: Target tokens [batch, block_size] (shifted by 1)
        """
        # Random starting positions
        ix = torch.randint(len(self.data) - self.block_size - 1, (batch_size,))

        # Extract sequences
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])

        return x.to(device), y.to(device)

    def __len__(self):
        """Number of tokens in dataset."""
        return len(self.data)


class OpenWebTextMLMDataset(Dataset):
    """
    OpenWebText dataset prepared for Masked Language Modeling (RoBERTa).

    Pre-tokenizes text and applies dynamic masking during __getitem__.
    """

    def __init__(
        self,
        subset_size: int = 100000,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            subset_size: Number of documents to load
            max_length: Maximum sequence length
            mlm_probability: Probability of masking a token
            cache_dir: Optional cache directory
        """
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Load documents
        print(f"Loading OpenWebText subset ({subset_size:,} documents) for MLM...")
        from datasets import load_dataset

        ds = load_dataset(
            "openwebtext",
            split=f"train[:{subset_size}]",
            trust_remote_code=True
        )

        # Store raw texts (tokenize on-the-fly for memory efficiency)
        self.texts = [ex["text"] for ex in tqdm(ds, desc="Loading texts")]
        print(f"Loaded {len(self.texts):,} documents")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Returns tokenized and masked example.

        Returns:
            dict with 'input_ids', 'attention_mask', 'labels'
        """
        text = self.texts[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create labels (copy of input_ids)
        labels = input_ids.clone()

        # Apply random masking
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        # Don't mask padding
        probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of time, replace with random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% of time, keep original (already done by default)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# =============================================================================
# EUR-Lex Dataset (Document Classification)
# =============================================================================

class EURLexDataset(Dataset):
    """
    EUR-Lex multi-label document classification dataset.

    Loads from HuggingFace datasets and returns raw text + label indices.
    Use with EURLexCollator for batching.

    Dataset: NLP-AUEB/eurlex (EURLEX57K)
    Labels: EUROVOC concept IDs (strings converted to integers)
    """

    def __init__(self, split: str = "train", max_examples: Optional[int] = None):
        """
        Args:
            split: Dataset split ("train", "development", "test")
                   Note: Uses "development" not "validation"
            max_examples: Optional limit on number of examples
        """
        from datasets import load_dataset

        # Map common split names
        split_map = {"validation": "development", "dev": "development"}
        actual_split = split_map.get(split, split)

        print(f"Loading EUR-Lex {actual_split} split...")

        # Load dataset from NLP-AUEB/eurlex (EURLEX57K)
        ds = load_dataset("NLP-AUEB/eurlex", split=actual_split, trust_remote_code=True)

        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))

        # Build label vocabulary (eurovoc_concepts are strings like "192", "2356")
        # We need to map them to contiguous integers
        print("Building label vocabulary...")
        all_label_strs = set()
        for ex in ds:
            all_label_strs.update(ex["eurovoc_concepts"])

        # Sort for reproducibility and create mapping
        # Use tuple key to avoid comparing int vs str (Python 3 TypeError)
        def label_sort_key(x):
            try:
                return (0, int(x))  # Numeric labels first, sorted by value
            except ValueError:
                return (1, x)  # Non-numeric labels after, sorted alphabetically

        sorted_labels = sorted(all_label_strs, key=label_sort_key)
        self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        self.num_labels = len(self.label_to_id)

        # Process examples
        self.examples = []
        for ex in tqdm(ds, desc=f"Processing {actual_split}"):
            label_ids = [self.label_to_id[lbl] for lbl in ex["eurovoc_concepts"] if lbl in self.label_to_id]
            self.examples.append({
                "text": ex["text"],
                "label_ids": label_ids
            })

        print(f"Loaded {len(self.examples):,} examples with {self.num_labels} labels")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class EURLexCollator:
    """
    Dynamic collator for EUR-Lex that can switch between multi-label and single-label.

    For single-label: Uses the first (most frequent) label per document.
    """

    tokenizer: Any
    max_length: int = 512
    num_labels: int = 4271
    mode: str = "multi"  # "multi" for BCEWithLogits, "single" for CrossEntropy

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        label_ids_list = [f["label_ids"] for f in features]

        # Tokenize
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        batch_size = len(features)

        if self.mode == "multi":
            # Multi-hot encoding for BCEWithLogitsLoss
            labels = torch.zeros(batch_size, self.num_labels)
            for i, label_ids in enumerate(label_ids_list):
                for lid in label_ids:
                    if lid < self.num_labels:
                        labels[i, lid] = 1.0
        else:
            # Single-label: use first (most frequent) label
            labels = torch.tensor(
                [label_ids[0] if label_ids else 0 for label_ids in label_ids_list],
                dtype=torch.long
            )

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels
        }


# =============================================================================
# Data Factory Functions
# =============================================================================

def create_lm_dataset(
    model_arch: str,
    task: str,
    subset_size: int = 100000,
    block_size: int = 1024,
    max_length: int = 512,
    cache_dir: Optional[str] = None,
):
    """
    Create language modeling dataset.

    Args:
        model_arch: "gpt2" or "roberta"
        task: "clm" or "mlm"
        subset_size: Number of documents to load
        block_size: Sequence length for CLM
        max_length: Max length for MLM
        cache_dir: Optional cache directory

    Returns:
        Dataset appropriate for the task
    """
    if task == "clm":
        return OpenWebTextDataset(
            subset_size=subset_size,
            block_size=block_size,
            tokenizer_type=model_arch,
            cache_dir=cache_dir
        )
    elif task == "mlm":
        return OpenWebTextMLMDataset(
            subset_size=subset_size,
            max_length=max_length,
            cache_dir=cache_dir
        )
    else:
        raise ValueError(f"Unknown LM task: {task}")


def create_classification_dataloader(
    split: str,
    task: str,  # "multi_cls" or "single_cls"
    batch_size: int = 8,
    max_length: int = 512,
    num_labels: Optional[int] = None,  # If None, uses dataset's num_labels
    max_examples: Optional[int] = None,
    num_workers: int = 0,
    tokenizer_name: str = "roberta-base",
) -> DataLoader:
    """
    Create classification dataloader with appropriate collator.

    Args:
        split: Dataset split ("train", "development", "test")
        task: "multi_cls" or "single_cls"
        batch_size: Batch size
        max_length: Max sequence length
        num_labels: Number of classification labels (None = use dataset's count)
        max_examples: Optional limit on examples
        num_workers: DataLoader workers
        tokenizer_name: Tokenizer to use

    Returns:
        DataLoader with appropriate collator
    """
    from transformers import AutoTokenizer

    # Load dataset
    dataset = EURLexDataset(split=split, max_examples=max_examples)

    # Use dataset's num_labels if not specified
    actual_num_labels = num_labels if num_labels is not None else dataset.num_labels

    # Setup collator
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    mode = "multi" if task == "multi_cls" else "single"
    collator = EURLexCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        num_labels=actual_num_labels,
        mode=mode
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True
    )


def create_mlm_dataloader(
    subset_size: int = 100000,
    max_length: int = 512,
    batch_size: int = 8,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create MLM dataloader for RoBERTa.

    Args:
        subset_size: Number of documents
        max_length: Max sequence length
        batch_size: Batch size
        num_workers: DataLoader workers

    Returns:
        DataLoader for MLM training
    """
    dataset = OpenWebTextMLMDataset(
        subset_size=subset_size,
        max_length=max_length
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
