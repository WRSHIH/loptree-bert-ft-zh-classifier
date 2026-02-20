"""
train.py
─────────────────────────────────────────────────────────────────
Training entry point for the BERT Ordinal Regression rating classifier.

Usage
-----
    python train.py
    python train.py --config configs/train_config.json
    python train.py --config configs/train_config.json --log-file logs/train.log
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import BertTokenizerFast, get_cosine_schedule_with_warmup
from torch.optim import AdamW

from src.data_utils import build_dataloaders_from_data, load_jsonl
from src.model import (
    BertForOrdinalRegression,
    compute_class_weights,
    logits_to_star_rating,
)
from src.logger import get_logger


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BertForOrdinalRegression on Chinese restaurant reviews."
    )
    parser.add_argument(
        "--config",
        default="configs/train_config.json",
        help="Path to train_config.json.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for log file output (e.g. logs/train.log).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop (one epoch)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: BertForOrdinalRegression,
    loader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    epoch_idx: int,
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Run one full pass over the training DataLoader.

    Parameters
    ----------
    model : BertForOrdinalRegression
    loader : DataLoader
        Training DataLoader.
    optimizer : AdamW
    scheduler
        Cosine LR scheduler.
    device : torch.device
    epoch_idx : int
        0-based epoch index (used only for the tqdm description).

    Returns
    -------
    Tuple[float, float, float, List[int], List[int]]
        (total_loss, accuracy, weighted_f1, all_preds_1based, all_labels_1based)

    """
    model.train()
    total_loss: float    = 0.0
    all_preds:  List[int] = []
    all_labels: List[int] = []

    for batch in tqdm(loader, desc=f"Training  Epoch {epoch_idx + 1}", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # ── Forward ──────────────────────────────────────────────────────────
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss   = outputs["loss"]    # weighted EMD loss
        logits = outputs["logits"]

        # ── Backward (same order as original: zero_grad → backward → step) ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # ── Collect predictions (1-based) ─────────────────────────────────────
        # logits_to_star_rating computes softmax+cumsum internally.  Wrapping
        # in no_grad() prevents building an autograd graph that would never be
        # used (logits.requires_grad=True after forward, but we only need the
        # predicted class indices for metrics — not gradients).
        with torch.no_grad():
            preds = logits_to_star_rating(logits)
        all_preds.extend(preds.cpu().tolist())

        # labels are 0-based (stored as 0-4); shift back to 1-based for metrics
        # (identical to original: labels.detach().cpu().numpy()+1)
        all_labels.extend((labels.detach().cpu() + 1).tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return total_loss, accuracy, f1, all_preds, all_labels


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (final pass on validation set)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    model: BertForOrdinalRegression,
    loader,
    device: torch.device,
) -> Tuple[float, float, str]:
    """
    Run inference on val_loader without gradient computation.

    Returns
    -------
    Tuple[float, float, str]
        (accuracy, weighted_f1, classification_report_string)

    """
    model.eval()
    all_preds:  List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = logits_to_star_rating(outputs["logits"])

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend((labels.detach().cpu() + 1).tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report   = classification_report(
        all_labels,
        all_preds,
        digits=4,
        target_names=["1星", "2星", "3星", "4星", "5星"],   # same labels as original
        zero_division=0,
    )
    return accuracy, f1, report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    logger = get_logger(__name__, log_file=args.log_file)

    # ── 1. Config ──────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    logger.info("Config loaded from %s", args.config)

    model_cfg  = cfg["model"]
    train_cfg  = cfg["training"]
    es_cfg     = cfg["early_stopping"]
    paths_cfg  = cfg["paths"]
    # Note: cfg["tokenizer"] is passed as part of the full cfg dict to
    # build_dataloaders_from_data, which reads it internally.

    checkpoint_dir = Path(paths_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 3. Tokenizer ───────────────────────────────────────────────────────────
    model_name = model_cfg["pretrained_name"]
    tokenizer  = BertTokenizerFast.from_pretrained(model_name)

    # ── 4. Load data ONCE  ────────────────────────────────────────
    # We need raw ratings to compute class weights AND we need the
    # full dataset for DataLoaders.  By loading once and passing
    # the data to build_dataloaders_from_data, we avoid reading the file twice.
    reviews, ratings = load_jsonl(paths_cfg["train_data"])

    # ── 5. Class weights (compute_class_weights(Ratings)) ────────────
    # Computed from ALL ratings BEFORE the train/val split, matching the
    # original notebook's execution order.
    class_weights = compute_class_weights(ratings)

    # ── 6. DataLoaders ────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders_from_data(
        reviews=reviews,
        ratings=ratings,
        cfg=cfg,
        tokenizer=tokenizer,
    )

    # ── 7. Model ─────────────────────────────────────────────────────
    model = BertForOrdinalRegression(
        model_name=model_name,
        class_weights=class_weights,
        num_labels=model_cfg["num_labels"],                     # 5
        hidden_size=model_cfg["classifier_hidden_size"],        # 256
        input_dropout=model_cfg["hidden_dropout_prob"],         # 0.1
        classifier_dropout=model_cfg["classifier_dropout"],     # 0.3
    ).to(device)

    # ── 8. Optimiser (AdamW, lr=2e-5, weight_decay=0.01) ────────────
    epochs    = train_cfg["epochs"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],       # 2e-5
        weight_decay=train_cfg["weight_decay"],  # 0.01
    )

    # ── 9. Scheduler (cosine with 5% warmup) ────────────────────────
    num_training_steps = len(train_loader) * epochs
    num_warmup_steps   = int(num_training_steps * train_cfg["warmup_ratio"])  # 5%
    scheduler          = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    logger.info(
        "Training steps: %d  |  Warmup: %d", num_training_steps, num_warmup_steps
    )

    # ── 10. Early stopping state ─────────────────────────────────
    # TRAINING accuracy (all_preds / all_labels come
    # from training batches, not from val_loader).  The threshold and counter
    # are now named accordingly.
    #
    #   Original:  best_val_acc = 0.90   ← misleading name in original notebook
    #   Fixed:     best_train_acc = ...  ← correctly reflects what is monitored
    best_train_acc   = es_cfg["min_train_accuracy"]   # 0.90 (from config)
    patience         = es_cfg["patience"]             # 3
    early_stop_count = 0
    best_ckpt_path   = checkpoint_dir / paths_cfg["best_model_name"]

    # ── 11. Training loop ────────────────────────────────────────────
    for epoch in range(epochs):

        total_loss, train_acc, train_f1, all_preds, all_labels = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # ── Per-epoch console output ─────────────────────────────
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {total_loss:.4f}, "
            f"Accuracy: {train_acc:.4f}"
        )
        print(f"Weighted F1 Score: {train_f1:.4f}")
        print(
            classification_report(
                all_labels,
                all_preds,
                digits=4,
                target_names=["1星", "2星", "3星", "4星", "5星"],
                zero_division=0,
            )
        )

        # ── Early stopping + checkpoint ─────────────────
        # Condition: training accuracy improved beyond the stored best
        if train_acc > best_train_acc:
            best_train_acc   = train_acc
            early_stop_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"New best model saved! Accuracy: {train_acc:.4f}")
            logger.info("Best checkpoint → %s  (acc=%.4f)", best_ckpt_path, train_acc)
        else:
            early_stop_count += 1
            print(
                f"No improvement. EarlyStopCounter = {early_stop_count}/{patience}"
            )
            if early_stop_count >= patience:
                print("Early stopping triggered!")
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    # ── 12. Final evaluation on validation set ──────────────────────
    logger.info("Loading best checkpoint for final evaluation …")
    if best_ckpt_path.exists():
        # weights_only=True: safe loading, suppresses FutureWarning in PyTorch ≥2.0
        model.load_state_dict(
            torch.load(best_ckpt_path, map_location=device, weights_only=True)
        )
    # model is already on `device` from step 7; no need to call .to(device) again.

    val_acc, val_f1, val_report = evaluate(model, val_loader, device)

    print(f"整體正確率 (Accuracy): {val_acc:.4f}")
    print(f"Weighted F1 Score: {val_f1:.4f}")
    print(val_report)

    # ── 13. Save model + tokenizer artefacts ────────────────────────
    saved_model_dir = Path(paths_cfg["saved_model_dir"])
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(saved_model_dir)
    torch.save(
        model.state_dict(),
        saved_model_dir / "bert_zh_FT_classifier.bin",
    )
    logger.info("Artefacts saved to %s", saved_model_dir)


if __name__ == "__main__":
    main()
