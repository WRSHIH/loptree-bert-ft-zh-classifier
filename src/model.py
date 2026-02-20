"""
src/model.py
─────────────────────────────────────────────────────────────────
Model architecture, loss function, and inference utilities.

Public API
----------
    weighted_emd_loss(preds, targets, class_weights, r)  → torch.Tensor
    logits_to_star_rating(logits)                         → torch.Tensor (1-based)
    compute_class_weights(ratings)                        → torch.Tensor
    BertForOrdinalRegression                              (nn.Module)

"""

import collections
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss function
# ─────────────────────────────────────────────────────────────────────────────

def weighted_emd_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    r: int = 2,
) -> torch.Tensor:
    """
    Weighted Earth Mover's Distance (EMD) loss for ordinal classification.

    Motivation
    ----------
    Cross-entropy treats every misclassification identically.  For star ratings,
    predicting 1★ when the truth is 5★ should cost far more than predicting 4★.
    EMD measures the area between the predicted and true *cumulative* distributions,
    automatically penalising predictions proportionally to their ordinal distance
    from the ground truth.  The per-class weights further correct for class
    imbalance (rare classes receive larger weights).

    Parameters
    ----------
    preds : torch.Tensor, shape (B, C)
        Softmax probabilities, i.e. the output of ``F.softmax(logits, dim=-1)``.
    targets : torch.Tensor, shape (B,)
        Zero-based ground-truth class indices in [0, C-1].
    class_weights : torch.Tensor, shape (C,)
        Per-class scalar weight (e.g. normalised inverse frequency).
        Must already be on the same device as ``preds`` (handled by
        ``register_buffer`` in the model).
    r : int
        Power applied to |CDF_pred - CDF_true|.  ``r=2`` (squared EMD) is the
        default, matching the original notebook.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # Ensure class_weights are on the same device as the predictions
    class_weights = class_weights.to(preds.device)

    # Cumulative distribution functions: shape (B, C)
    cdf_preds = torch.cumsum(preds, dim=1)
    cdf_true  = torch.cumsum(
        F.one_hot(targets, num_classes=preds.size(1)).float(),
        dim=1,
    )

    # Absolute difference between CDFs: shape (B, C)
    abs_diff = torch.abs(cdf_preds - cdf_true)

    # Apply per-class weights, broadcast over the batch dimension
    # class_weights.unsqueeze(0) → shape (1, C), broadcasts with (B, C)
    weighted_diff = class_weights.unsqueeze(0) * torch.pow(abs_diff, r)  # (B, C)

    # Mean over batch of summed ordinal differences — scalar
    loss = torch.mean(torch.sum(weighted_diff, dim=1))
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Inference utility
# ─────────────────────────────────────────────────────────────────────────────

def logits_to_star_rating(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw classifier logits to 1-based star ratings.

    We interpret the softmax output as an ordinal probability distribution
    and threshold its CDF at 0.5 (the median), which is more robust than a
    plain argmax for ordinal tasks where the distribution may be bi-modal.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
        Raw (un-normalised) model outputs.

    Returns
    -------
    torch.Tensor, shape (B,)
        Predicted star ratings in [1, C].
    """
    probs       = F.softmax(logits, dim=-1)           # (B, C)
    cdf         = torch.cumsum(probs, dim=-1)          # (B, C)
    # Count how many class CDFs are ≤ 0.5, then add 1 to make the index 1-based
    predictions = torch.sum(cdf <= 0.5, dim=-1) + 1   # (B,)
    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Class-weight helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(ratings: List[int]) -> torch.Tensor:
    """
    Compute normalised inverse-frequency class weights.

    Parameters
    ----------
    ratings : List[int]
        Raw 1-based star ratings from the full (unsplit) training corpus,
        as loaded by ``load_jsonl``.

    Returns
    -------
    torch.Tensor, shape (5,)
        Weights that sum to 1.0; rare classes receive larger values.
    """
    raw_counter  = collections.Counter(ratings)
    # Default to 1 for any class absent in the data (avoids division-by-zero)
    label_counts = [raw_counter.get(i + 1, 1) for i in range(5)]
    counts       = torch.tensor(label_counts, dtype=torch.float)  # (5,)
    weights      = 1.0 / counts                                    # inverse frequency
    normalised   = weights / weights.sum()                         # sum to 1.0
    logger.info(
        "Class weights (1★–5★): %s",
        [f"{w:.4f}" for w in normalised.tolist()],
    )
    return normalised


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class BertForOrdinalRegression(nn.Module):
    """
    BERT encoder with a two-layer MLP head for 5-class ordinal regression.

    Architecture (same as original)
    --------------------------------
    1. Pretrained BERT backbone (any HuggingFace AutoModel).
    2. Max-pool over the last hidden-state sequence dimension to produce a
       fixed-size sentence vector — more robust than [CLS] pooling for
       short, noisy reviews.
    3. Dropout(0.1) → Linear(H, 256) → ReLU → Dropout(0.3) → Linear(256, 5).
    4. Loss: Weighted EMD (``weighted_emd_loss``).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``"bert-base-chinese"``).
    class_weights : Optional[torch.Tensor]
        Per-class normalised inverse-frequency weights for the EMD loss.
        Registered as a ``buffer`` so it moves with ``.to(device)`` and is
        saved/loaded with ``state_dict``.  Defaults to uniform weights.
    config : optional
        Pretrained config override.  Defaults to loading from HuggingFace.
    num_labels : int
        Number of ordinal classes (default: 5).
    hidden_size : int
        Width of the intermediate MLP layer (default: 256, same as original).
    input_dropout : float
        Dropout probability applied after pooling (default: 0.1, same as original).
    classifier_dropout : float
        Dropout probability inside the MLP (default: 0.3, same as original).
    """

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        class_weights: Optional[torch.Tensor] = None,
        config=None,
        num_labels: int = 5,
        hidden_size: int = 256,
        input_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
    ) -> None:
        super().__init__()

        # ── BERT backbone ────────────────
        bert_config = config or AutoConfig.from_pretrained(model_name)
        self.bert   = AutoModel.from_pretrained(model_name, config=bert_config)

        # ── Dropout after pooling  ──────────────────
        self.dropout    = nn.Dropout(input_dropout)

        # ── Classification head ─────────────────
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),  # (H → 256)
            nn.ReLU(),
            nn.Dropout(classifier_dropout),                         # 0.3
            nn.Linear(hidden_size, num_labels),                     # (256 → 5)
        )

        # ── Class weights as a buffer ─────────────
        # Using register_buffer ensures:
        #   (a) weights move automatically when model.to(device) is called
        #   (b) weights are included in state_dict for checkpoint save/load
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.register_buffer("class_weights", torch.ones(num_labels))

        logger.info(
            "BertForOrdinalRegression — backbone: %s | labels: %d",
            model_name, num_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (B, L)
        attention_mask : torch.Tensor, shape (B, L)
        labels : Optional[torch.Tensor], shape (B,)
            Zero-based class indices [0, num_labels-1].
            When provided, loss is computed and included in the returned dict.

        Returns
        -------
        dict
            ``"logits"`` : torch.Tensor (B, C) — always present.
            ``"loss"``   : torch.Tensor scalar — present only when labels given.

        """
        # ── Encode ────────────────────────────────────────────────────────────
        outputs     = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state                    # (B, L, H)

        # Max-pool over sequence length → (B, H)
        pooled_output = torch.max(last_hidden, dim=1).values

        # ── Classify ──────────────────────────────────────────────────────────
        pooled_output = self.dropout(pooled_output)                # Dropout(0.1)
        logits        = self.classifier(pooled_output)             # (B, C)
        probs         = F.softmax(logits, dim=-1)                  # (B, C)

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if labels is not None:
            # Cast to long
            if labels.dtype != torch.long:
                labels = labels.long()
            result["loss"] = weighted_emd_loss(
                preds=probs,
                targets=labels,
                class_weights=self.class_weights,  # buffer already on correct device
            )

        return result
