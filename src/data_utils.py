"""
src/data_utils.py
─────────────────────────────────────────────────────────────────
Data cleaning, tokenisation, and DataLoader construction.

Public API
----------
    clean_chinese_text(text)                               → str
    load_jsonl(path)                                       → Tuple[List[str], List[int]]
    preprocess(texts, labels, tokenizer, max_length)       → datasets.Dataset
    build_dataloaders_from_data(reviews, ratings, cfg, tok)
                                                           → Tuple[DataLoader, DataLoader]
    build_dataloaders(cfg, tok)                            → Tuple[DataLoader, DataLoader]
"""

import json
import re
import collections
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast

from .logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_chinese_text(text: str) -> str:
    """
    Normalise a raw Chinese restaurant review.

    Steps
    -----
    1. Strip structural whitespace (\\n \\r \\t).
    2. Remove URLs beginning with http / https / www.
    3. Remove e-mail addresses.
    4. Delete every character that is NOT:
         - CJK Unified Ideographs (\\u4e00-\\u9fff)
         - ASCII alphanumerics (A-Za-z0-9)
         - Common ASCII punctuation  . , ! ? ; : ( )
         - Full-width / Chinese punctuation  ，。！？、；：（）【】「」『』《》〈〉
         - Hyphen, double-quote, single-quote, space
    5. Collapse runs of the same punctuation character into a single instance.
    """
    # ── 1. Structural whitespace ──────────────────────────────────────────────
    text = text.replace("\n", "").replace("\r", "").replace("\t", "")

    # ── 2. URLs ───────────────────────────────────────────────────────────────
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # ── 3. E-mail addresses ───────────────────────────────────────────────────
    text = re.sub(r"\S+@\S+", "", text)

    # ── 4. Character whitelist  ─────────────────
    text = re.sub(
        r"""[^\u4e00-\u9fffA-Za-z0-9.,!?;:()，。！？、；：（）【】「」『』《》〈〉\-"' ]""",
        "",
        text,
    )

    # ── 5. Collapse repeated punctuation  ─────────────────────
    # r'\1+' : backslash + '1' + '+' in a raw string.
    # The regex engine reads \1 as "backreference to group 1", so the full
    # pattern matches any run of 2+ identical punctuation chars and
    # replaces it with a single instance.
    text = re.sub(r"([！？。，、～\-.])\1+", r"\1", text)

    return text


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Union[str, Path]) -> Tuple[List[str], List[int]]:
    """
    Load the two-line JSONL training file used throughout the project.

    Expected file format (exactly two JSON lines)
    ----------------------------------------------
        {"Reviews": ["review text 1", "review text 2", ...]}
        {"Ratings": [5, 3, 4, ...]}


    Parameters
    ----------
    path : str or Path
        Path to the .jsonl file.

    Returns
    -------
    Tuple[List[str], List[int]]
        (reviews, ratings).  Ratings are integers in [1, 5].

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If fewer than 2 JSON lines are found, or reviews and ratings have
        different lengths.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")

    records: List[Any] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if len(records) < 2:
        raise ValueError(
            f"Expected at least 2 JSON lines in {path}, got {len(records)}."
        )

    reviews: List[str] = records[0]["Reviews"]
    ratings: List[int] = records[1]["Ratings"]

    if len(reviews) != len(ratings):
        raise ValueError(
            f"Length mismatch: {len(reviews)} reviews vs {len(ratings)} ratings."
        )

    logger.info("Loaded %d samples from %s", len(reviews), path)
    logger.info("Rating distribution: %s", dict(collections.Counter(ratings)))
    return reviews, ratings


# ─────────────────────────────────────────────────────────────────────────────
# Dataset construction
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(
    texts: List[str],
    labels: List[int],
    tokenizer: BertTokenizerFast,
    max_length: int = 128,
) -> Dataset:
    """
    Tokenise a list of texts and wrap them in a HuggingFace Dataset.

    Label shift
    -----------
    Star ratings 1–5 are shifted to 0–4 (``[i - 1 for i in labels]``) so they
    can be used directly as zero-based class indices in PyTorch.

    Parameters
    ----------
    texts : List[str]
        Pre-cleaned review strings.
    labels : List[int]
        Raw 1-based star ratings.
    tokenizer : BertTokenizerFast
        Tokenizer that matches the BERT backbone.
    max_length : int
        Padding / truncation length.  128 matches the original notebook.

    Returns
    -------
    datasets.Dataset
        Columns: ``input_ids``, ``attention_mask``, ``labels`` — all as
        torch.Tensor (via ``set_format``).
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    zero_based_labels = [r - 1 for r in labels]

    dataset = Dataset.from_dict(
        {
            "input_ids":      encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels":         zero_based_labels,
        }
    )

    # Auto-convert to torch.Tensor
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders_from_data(
    reviews: List[str],
    ratings: List[int],
    cfg: Dict[str, Any],
    tokenizer: BertTokenizerFast,
) -> Tuple[DataLoader, DataLoader]:
    """
    Clean, tokenise, and split pre-loaded data into train/val DataLoaders.

    Parameters
    ----------
    reviews : List[str]
        Raw (uncleaned) review strings loaded from the JSONL file.
    ratings : List[int]
        Parallel list of 1-based star ratings.
    cfg : dict
        Full parsed train_config.json.
    tokenizer : BertTokenizerFast

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    tok_cfg   = cfg["tokenizer"]
    train_cfg = cfg["training"]

    # ── Text cleaning  ─────────────────────────────────────────
    logger.info("Cleaning %d review texts …", len(reviews))
    reviews = [clean_chinese_text(r) for r in reviews]

    # ── Tokenise → Dataset  ───────────────────────────────────
    dataset = preprocess(
        texts=reviews,
        labels=ratings,
        tokenizer=tokenizer,
        max_length=tok_cfg["max_length"],
    )

    # ── Train / validation split  ──
    split         = dataset.train_test_split(
        test_size=train_cfg["val_split"],   # 0.2 from config
        seed=train_cfg["seed"],             # reproducibility
    )
    train_dataset = split["train"]
    val_dataset   = split["test"]

    logger.info(
        "Split → train: %d  |  val: %d",
        len(train_dataset),
        len(val_dataset),
    )

    # ── DataLoaders (batch_size=16 / 1024, shuffle=True / False) ────
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["train_batch_size"],  # 16
        shuffle=True,                               # same as original
        pin_memory=True,                            # same as original
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["val_batch_size"],    # 1024
        shuffle=False,                              # same as original
        num_workers=2,
    )

    return train_loader, val_loader


def build_dataloaders(
    cfg: Dict[str, Any],
    tokenizer: BertTokenizerFast,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience wrapper: load from disk, then call ``build_dataloaders_from_data``.

    Use only when you do NOT need the raw reviews/ratings elsewhere.
    When you also need ratings for class-weight computation, call
    ``load_jsonl`` yourself and pass the results to
    ``build_dataloaders_from_data`` to avoid reading the file twice.

    Parameters
    ----------
    cfg : dict
        Full parsed train_config.json.
    tokenizer : BertTokenizerFast

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, val_loader)
    """
    reviews, ratings = load_jsonl(cfg["paths"]["train_data"])
    return build_dataloaders_from_data(reviews, ratings, cfg, tokenizer)
