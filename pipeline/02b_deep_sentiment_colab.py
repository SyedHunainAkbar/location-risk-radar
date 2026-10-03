"""Offline (Colab GPU) deep-learning sentiment benchmark (LA3).

Trains GRU and LSTM models with GloVe 100d embeddings, both frozen and trainable,
on the SAME leakage-free grouped split used by stage 02, with early stopping. It
reports the shared metric schema with bootstrap 95% CIs and appends its rows to
``artifacts/sentiment_benchmark.parquet`` so the final notebook renders one table
across the classical (LA2) and deep (LA3) models.

This script is OFFLINE ONLY. TensorFlow/Keras and GloVe live here and never in the
app or the online pipeline modules. Run it on a Colab GPU, download GloVe once, and
commit the resulting benchmark parquet.

Usage (Colab):
    !python pipeline/02b_deep_sentiment_colab.py \
        --reviews data/cohort_reviews.parquet \
        --locations artifacts/cohort_locations.parquet \
        --glove glove.6B.100d.txt \
        --artifacts-dir artifacts
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config, io, sentiment  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deep sentiment benchmark (offline).")
    p.add_argument("--reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument(
        "--glove",
        type=Path,
        required=True,
        help="Path to glove.6B.100d.txt (download once on Colab).",
    )
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-words", type=int, default=40_000)
    return p.parse_args(argv)


def set_global_seed() -> None:
    import random

    import tensorflow as tf

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)


def load_labeled_split(reviews_path: Path, locations_path: Path):
    """Reproduce the exact stage-02 grouped split (same seed, same labels)."""
    reviews = io.read_parquet(reviews_path)
    labeled = sentiment.map_polarity_labels(reviews)
    train, test = sentiment.grouped_split(labeled)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"]))
    return train, test


def build_tokenizer(train_texts, max_words: int):
    from tensorflow.keras.preprocessing.text import Tokenizer

    tok = Tokenizer(num_words=max_words, oov_token="<oov>")
    tok.fit_on_texts(train_texts)
    return tok


def to_padded(tok, texts, maxlen: int):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")


def load_glove_matrix(glove_path: Path, tok, max_words: int, dim: int) -> np.ndarray:
    """Build an embedding matrix from a GloVe text file for the tokenizer vocab."""
    index: dict[str, np.ndarray] = {}
    with open(glove_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            index[parts[0]] = np.asarray(parts[1:], dtype="float32")
    vocab = min(max_words, len(tok.word_index) + 1)
    matrix = np.zeros((vocab, dim), dtype="float32")
    hits = 0
    for word, i in tok.word_index.items():
        if i >= vocab:
            continue
        vec = index.get(word)
        if vec is not None:
            matrix[i] = vec
            hits += 1
    print(f"GloVe coverage: {hits}/{vocab} tokens matched.")
    return matrix


def build_model(kind: str, embedding_matrix: np.ndarray, trainable: bool, maxlen: int):
    """Build a GRU or LSTM classifier with a GloVe embedding layer."""
    from tensorflow.keras import layers, models

    vocab, dim = embedding_matrix.shape
    rnn = layers.GRU if kind == "GRU" else layers.LSTM
    model = models.Sequential(
        [
            layers.Input(shape=(maxlen,)),
            layers.Embedding(
                vocab,
                dim,
                weights=[embedding_matrix],
                trainable=trainable,
                mask_zero=True,
            ),
            layers.Bidirectional(rnn(64)),
            layers.Dropout(0.3),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def benchmark_row_from_probs(model_name, features, y_true, y_prob) -> dict:
    """Build a shared-schema row from predicted probabilities (reuses sentiment)."""
    y_pred = (y_prob >= 0.5).astype(int)
    base = sentiment.evaluate(y_true, y_pred, y_prob)
    acc_lo, acc_hi = sentiment.bootstrap_ci(y_true, y_pred, y_prob, "accuracy")
    f1_lo, f1_hi = sentiment.bootstrap_ci(y_true, y_pred, y_prob, "macro_f1")
    auc_lo, auc_hi = sentiment.bootstrap_ci(y_true, y_pred, y_prob, "roc_auc")
    return {
        "model": model_name,
        "features": features,
        "accuracy": base["accuracy"],
        "accuracy_lo": acc_lo,
        "accuracy_hi": acc_hi,
        "macro_f1": base["macro_f1"],
        "macro_f1_lo": f1_lo,
        "macro_f1_hi": f1_hi,
        "recall_neg": base["recall_neg"],
        "recall_pos": base["recall_pos"],
        "roc_auc": base["roc_auc"],
        "roc_auc_lo": auc_lo,
        "roc_auc_hi": auc_hi,
        "n_test": int(len(y_true)),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    import time

    from tensorflow.keras.callbacks import EarlyStopping

    set_global_seed()
    train, test = load_labeled_split(args.reviews, args.locations)
    maxlen = config.MAX_SEQUENCE_LEN

    tok = build_tokenizer(train["text"].astype(str).tolist(), args.max_words)
    X_train = to_padded(tok, train["text"].astype(str).tolist(), maxlen)
    X_test = to_padded(tok, test["text"].astype(str).tolist(), maxlen)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    emb = load_glove_matrix(args.glove, tok, args.max_words, config.GLOVE_DIM)
    early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    rows: list[dict] = []
    for kind in ("GRU", "LSTM"):
        for trainable in (False, True):
            tag = "trainable" if trainable else "frozen"
            name = f"{kind} (GloVe {tag})"
            print(f"\nTraining {name} ...")
            model = build_model(kind, emb, trainable, maxlen)
            t0 = time.time()
            model.fit(
                X_train,
                y_train,
                validation_split=0.1,
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=[early],
                verbose=2,
            )
            train_s = time.time() - t0
            t1 = time.time()
            y_prob = model.predict(X_test, batch_size=args.batch_size).ravel()
            infer_ms = 1000 * (time.time() - t1) / max(len(X_test), 1)
            row = benchmark_row_from_probs(name, f"glove100-{tag}", y_test, y_prob)
            row["train_seconds"] = round(train_s, 1)
            row["infer_ms_per_review"] = round(infer_ms, 3)
            rows.append(row)

    deep = pd.DataFrame(rows)

    # Join with LA2 rows so the final table shows all models together.
    bench_path = Path(args.artifacts_dir) / config.SENTIMENT_BENCHMARK_FILE
    if bench_path.exists():
        la2 = io.read_parquet(bench_path)
        combined = pd.concat([la2, deep], ignore_index=True)
    else:
        print("[warn] LA2 benchmark not found; writing deep rows only.")
        combined = deep
    io.write_parquet(combined, bench_path)

    best = combined.sort_values("macro_f1", ascending=False).iloc[0]
    print("\nFULL BENCHMARK (LA2 + LA3)")
    print("=" * 72)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(combined.round(4).to_string(index=False))
    print(f"\nBest by macro F1: {best['features']} + {best['model']} ({best['macro_f1']:.4f}).")
    print(
        "\nLatency/accuracy note: classical TF-IDF + linear models score in "
        "microseconds per review on CPU and run inside the 1 GB app, while the "
        "RNNs need a GPU and tens of milliseconds per review. We choose the "
        "production scorer on macro F1 unless a deep model's gain is both outside "
        "the bootstrap CI and worth the serving cost; otherwise the calibrated "
        "classical scorer wins on deployability."
    )
    return combined


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
