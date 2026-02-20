# BERT Rating Classifier

A production-grade Chinese restaurant review → star-rating classifier built on top of `bert-base-chinese`, using a custom **Weighted Earth Mover's Distance (EMD)** loss for ordinal regression.

---

## Project Structure

```
bert_rating_classifier/
├── configs/
│   └── train_config.json      # All hyperparameters in one place
├── src/
│   ├── __init__.py             # Public package API
│   ├── data_utils.py           # Text cleaning, tokenisation, DataLoader factory
│   ├── model.py                # BertForOrdinalRegression + loss + inference utils
│   └── logger.py               # Unified logging (stdout + optional file)
├── scripts/
│   └── scrape_gmaps.py         # Google Maps Places scraper (standalone)
├── train.py                    # Training entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Prepare data

Either place an existing `Train_data1.jsonl` in `data/`, or run the scraper to build it from scratch:

```bash
export GMAPS_API_KEY="your_key_here"
python scripts/scrape_gmaps.py \
    --ids-out  data/place_ids.txt \
    --jsonl-out data/Train_data1.jsonl
```

The JSONL format expected by the training pipeline:

```json
{"Reviews": ["很好吃！", "普通", ...]}
{"Ratings": [5, 3, ...]}
```

### 3 — Configure

Edit `configs/train_config.json` to adjust hyperparameters, paths, or the model backbone.

### 4 — Train

```bash
python train.py
# or with an explicit config and log file:
python train.py --config configs/train_config.json --log-file logs/train.log
```

Checkpoints are written to `checkpoints/best_model.pth`.  
Final model + tokenizer artefacts are saved to `saved_model/`.

---

## Model Design

### Architecture

```
BERT backbone (bert-base-chinese)
    │
    ├─ last_hidden_state  (B, L, 768)
    │         ↓  max-pool over L
    │  pooled_output  (B, 768)
    │         ↓  Dropout(0.1)
    │  Linear(768 → 256) → ReLU → Dropout(0.3)
    │         ↓
    │  Linear(256 → 5)   [logits]
    │         ↓  softmax
    └─ probs  (B, 5)
```

### Loss — Weighted EMD

Ordinary cross-entropy treats every misclassification equally.  For ordinal star ratings, predicting 1★ when the truth is 5★ should cost much more than predicting 4★.

The **Earth Mover's Distance** (Wasserstein-1 distance) penalises predictions proportionally to how far they are from the truth on the ordinal scale:

```
EMD = Σ_c  w_c · |CDF_pred(c) − CDF_true(c)|^r
```

where `w_c` is the inverse-frequency weight for class `c` and `r=2` (squared EMD).

### Inference

A CDF-based thresholding scheme is used instead of `argmax`, which is more robust for ordinal tasks:

```python
cdf   = cumsum(softmax(logits), dim=-1)
pred  = sum(cdf <= 0.5) + 1   # 1-based star rating
```

---

## Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `model.pretrained_name` | `bert-base-chinese` | HuggingFace model ID |
| `model.num_labels` | `5` | Number of ordinal classes |
| `model.classifier_hidden_size` | `256` | Intermediate MLP width |
| `training.epochs` | `10` | Max training epochs |
| `training.train_batch_size` | `16` | Mini-batch size |
| `training.learning_rate` | `2e-5` | AdamW LR |
| `training.weight_decay` | `0.01` | L2 regularisation |
| `training.warmup_ratio` | `0.05` | Fraction of steps for LR warm-up |
| `training.val_split` | `0.2` | Validation set fraction |
| `early_stopping.min_val_accuracy` | `0.90` | Minimum accuracy to save checkpoint |
| `early_stopping.patience` | `3` | Epochs without improvement before stopping |

---
