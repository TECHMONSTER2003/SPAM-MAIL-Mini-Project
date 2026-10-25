#!/usr/bin/env python3
# =============================================================================
#  SPAM FILTER DEMO - Supervised learning (Research Methodology seminar)
#  Dataset: UCI SMS Spam Collection (Almeida & Gomez Hidalgo, 2011), CC BY 4.0
#  NOTE: this is an SMS dataset. Results do NOT show the model can detect
#        phishing e-mail.
# =============================================================================
#
#  >>> DEMO SETTINGS - change any value below, save, and run again. <<<
#  Every run is added to results_log.csv and the last runs are printed
#  side by side at the end, so you can compare.
#  Default values give: Accuracy 0.9806, Precision 0.9030, Recall 0.9453, F1 0.9237
# =============================================================================

# --- 1. Data split -----------------------------------------------------------
TEST_SIZE      = 0.2     # share of data kept aside for testing (0.05 - 0.5)
RANDOM_STATE   = 42      # seed for the shuffle; change it to see a different split
TRAIN_FRACTION = 1.0     # use only this share of the training set (0.02 - 1.0)
                         # e.g. 0.05 -> learn from ~200 messages instead of ~4,100

# --- 2. Data cleaning ---------------------------------------------------------
DROP_DUPLICATES = True   # False -> duplicate messages can land in train AND test (leakage)

# --- 3. Features (TF-IDF: turning text into numbers) --------------------------
NGRAM_RANGE  = (1, 2)    # (1, 1) = single words only, (1, 2) = words + word pairs
MIN_DF       = 2         # ignore words seen in fewer than this many training messages
MAX_FEATURES = None      # None = keep all; e.g. 20, 100, 500 -> model sees fewer words
STOP_WORDS   = None      # None, or "english" to remove common words (the, to, my ...)
LOWERCASE    = True      # False -> "FREE" and "free" become different features

# --- 4. Model ----------------------------------------------------------------
MODEL        = "lr"      # "lr" = Logistic Regression, "nb" = Multinomial Naive Bayes
CLASS_WEIGHT = "balanced"  # LR only: "balanced" (give rare spam class more weight) or None
C_VALUE      = 1.0       # LR only: lower (0.01) = simpler model, higher (100) = fits training data harder
NB_ALPHA     = 1.0       # NB only: smoothing; larger = smoother, less sensitive to rare words

# --- 5. Decision -------------------------------------------------------------
THRESHOLD    = 0.5       # message is SPAM if P(spam) >= THRESHOLD
                         # lower (0.3) -> catches more spam, blocks more normal messages
                         # higher (0.8) -> blocks fewer normal messages, misses more spam

# --- 6. Output ---------------------------------------------------------------
TOP_FEATURES = 10        # how many top spam / ham words to print (0 = hide)
SAVE_PLOT    = True      # save confusion_matrix.png next to this script
SHOW_LAST_RUNS = 8       # how many previous runs to show in the comparison table (0 = hide)
# =============================================================================
#  End of settings. Normally nothing below needs to change.
# =============================================================================

"""
Pipeline: load -> validate/clean -> stratified train/test split -> Pipeline(TF-IDF -> classifier)
fitted on training data only -> evaluate on held-out test set -> interactive prediction.

Command-line options (settings above stay the main control):
  --download         download the dataset from UCI first
  --data PATH        use a different dataset path
  --fallback         tiny SYNTHETIC dataset, only to check the code runs
  --message "text"   classify one message and exit
  --no-interactive   skip the typing loop at the end
"""


import argparse
import csv
import io
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = SCRIPT_DIR / "data" / "SMSSpamCollection"
RESULTS_LOG = SCRIPT_DIR / "results_log.csv"
PLOT_PATH = SCRIPT_DIR / "confusion_matrix.png"
VALID_LABELS = ("ham", "spam")   # order fixes confusion-matrix rows/columns
POSITIVE_LABEL = "spam"           # the class we want to detect
LABEL_MEANING = {
    "spam": "SPAM - unsolicited / unwanted message (positive class)",
    "ham": "HAM  - legitimate message (negative class)",
}

# Synthetic fallback data. Written for this demo; NOT drawn from any dataset.
# Only for checking that the code runs. Metrics computed on it mean nothing.
FALLBACK_MESSAGES = [
    ("spam", "WINNER! You have been selected for a 1000 cash prize. Call 09061234567 now"),
    ("spam", "URGENT: your account will be suspended. Reply with your PIN to keep it active"),
    ("spam", "Free entry to win a new phone. Text WIN to 80888 now. T&C apply"),
    ("spam", "Congratulations, you won a holiday voucher. Claim at the link before midnight"),
    ("spam", "Your bank card is blocked. Verify details at the link to unblock immediately"),
    ("spam", "Limited offer: claim your free ringtone, text YES to 85023"),
    ("spam", "You have won a guaranteed cash reward. Call now to claim your prize"),
    ("spam", "Final notice: unpaid parcel fee. Pay now via link or parcel will be returned"),
    ("spam", "Exclusive deal for you. Reply STOP to opt out. Win big today"),
    ("spam", "Claim your bonus caller prize now, call our hotline, offer ends today"),
    ("spam", "Your mobile number won a lottery draw. Send your bank details to claim"),
    ("spam", "Get a loan approved instantly with no checks. Text LOAN now"),
    ("ham", "Are we still meeting for lunch at 1 tomorrow?"),
    ("ham", "I will reach the office by 10, traffic is bad today"),
    ("ham", "Can you send me the lab report before the review meeting"),
    ("ham", "Mom says dinner is ready, come home soon"),
    ("ham", "The seminar has moved to room 204, see you there"),
    ("ham", "Thanks for the notes, they really helped with the assignment"),
    ("ham", "Did you finish the python script for the project"),
    ("ham", "Running late, start the meeting without me"),
    ("ham", "Happy birthday! Have a great day"),
    ("ham", "Please call me when you are free this evening"),
    ("ham", "The train is delayed by twenty minutes"),
    ("ham", "Let us revise the research methodology slides on Sunday"),
    ("ham", "I left the charger in your bag, bring it tomorrow"),
    ("ham", "Ok see you at the library after class"),
    ("ham", "Can you check whether the router is rebooting again"),
    ("ham", "Got the tickets for the weekend, will share details"),
]

EXAMPLE_MESSAGES = [
    "Congratulations! You have won a free holiday. Call 09061234567 now to claim your prize",
    "Hi, are we still meeting at the library at 5 today?",
    "Your account has been suspended. Verify your details at the link immediately",
]


# --------------------------------------------------------------------------
# Data acquisition and validation
# --------------------------------------------------------------------------
def download_dataset(target: Path, timeout: int = 30) -> None:
    """Download the official UCI zip and extract the SMSSpamCollection file."""
    print(f"[download] {UCI_ZIP_URL}")
    try:
        with urllib.request.urlopen(UCI_ZIP_URL, timeout=timeout) as resp:
            payload = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            f"Download failed ({exc}). Download the zip manually from the UCI page "
            f"and place 'SMSSpamCollection' at: {target}"
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            member = next((m for m in zf.namelist() if m.endswith("SMSSpamCollection")), None)
            if member is None:
                raise RuntimeError(f"'SMSSpamCollection' not found in zip: {zf.namelist()}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))
    except zipfile.BadZipFile as exc:
        raise RuntimeError("Downloaded file is not a valid zip archive.") from exc
    print(f"[download] saved -> {target}")


def load_sms_collection(path: Path) -> tuple[pd.DataFrame, int]:
    """
    Parse the UCI file: one message per line, '<label>\\t<message>'.

    Parsed manually (split on the FIRST tab) rather than with pd.read_csv,
    because messages contain quote characters that a CSV parser can
    misinterpret, silently merging or dropping rows.
    Returns (dataframe, number_of_malformed_lines).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"  Fix: run with --download, or download the zip from\n"
            f"  https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
            f"  and extract 'SMSSpamCollection' to {path.parent}/"
        )

    rows, malformed = [], 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                malformed += 1
                continue
            rows.append(parts)
    return pd.DataFrame(rows, columns=["label", "message"]), malformed


def validate_and_clean(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Check columns, normalise labels, drop invalid/empty/duplicate rows. Prints a report."""
    expected = {"label", "message"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Expected columns {sorted(expected)}, found {list(df.columns)}")

    n_raw = len(df)
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    invalid_label = ~df["label"].isin(VALID_LABELS)
    empty_message = df["message"].eq("") | df["message"].str.lower().isin({"nan", "none"})
    print(f"  rows read                 : {n_raw}")
    print(f"  invalid label (dropped)   : {int(invalid_label.sum())}")
    print(f"  empty message (dropped)   : {int((empty_message & ~invalid_label).sum())}")
    df = df[~invalid_label & ~empty_message]

    # Same text with BOTH labels -> label noise; drop all copies.
    conflicting = df.groupby("message")["label"].nunique()
    conflicting = set(conflicting[conflicting > 1].index)
    if conflicting:
        df = df[~df["message"].isin(conflicting)]
    print(f"  conflicting-label texts   : {len(conflicting)}")

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset="message", keep="first")
        print(f"  exact duplicates (dropped): {before - len(df)}")
    else:
        print("  exact duplicates          : KEPT (may leak across train/test)")

    if df["label"].nunique() < 2:
        raise ValueError("Need both 'ham' and 'spam' rows after cleaning.")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Settings check
# --------------------------------------------------------------------------
def check_settings() -> None:
    """Fail early with a clear message if a setting at the top is out of range."""
    problems = []
    if not 0.05 <= TEST_SIZE <= 0.5:
        problems.append("TEST_SIZE must be between 0.05 and 0.5")
    if not 0.02 <= TRAIN_FRACTION <= 1.0:
        problems.append("TRAIN_FRACTION must be between 0.02 and 1.0")
    if MODEL not in ("lr", "nb"):
        problems.append('MODEL must be "lr" or "nb"')
    if CLASS_WEIGHT not in ("balanced", None):
        problems.append('CLASS_WEIGHT must be "balanced" or None')
    if not 0.0 < THRESHOLD < 1.0:
        problems.append("THRESHOLD must be between 0 and 1 (e.g. 0.5)")
    if not (isinstance(NGRAM_RANGE, tuple) and len(NGRAM_RANGE) == 2
            and 1 <= NGRAM_RANGE[0] <= NGRAM_RANGE[1] <= 3):
        problems.append("NGRAM_RANGE must look like (1, 1), (1, 2) or (1, 3)")
    if not (isinstance(MIN_DF, int) and MIN_DF >= 1):
        problems.append("MIN_DF must be a whole number >= 1")
    if MAX_FEATURES is not None and not (isinstance(MAX_FEATURES, int) and MAX_FEATURES >= 1):
        problems.append("MAX_FEATURES must be None or a whole number >= 1")
    if STOP_WORDS not in (None, "english"):
        problems.append('STOP_WORDS must be None or "english"')
    if C_VALUE <= 0 or NB_ALPHA <= 0:
        problems.append("C_VALUE and NB_ALPHA must be > 0")
    if problems:
        raise ValueError("Fix these settings at the top of the file:\n  - " + "\n  - ".join(problems))


def print_settings() -> None:
    print("[0] Settings used for this run")
    print(f"  TEST_SIZE={TEST_SIZE}  RANDOM_STATE={RANDOM_STATE}  TRAIN_FRACTION={TRAIN_FRACTION}  "
          f"DROP_DUPLICATES={DROP_DUPLICATES}")
    print(f"  NGRAM_RANGE={NGRAM_RANGE}  MIN_DF={MIN_DF}  MAX_FEATURES={MAX_FEATURES}  "
          f"STOP_WORDS={STOP_WORDS}  LOWERCASE={LOWERCASE}")
    if MODEL == "lr":
        print(f"  MODEL=lr (Logistic Regression)  CLASS_WEIGHT={CLASS_WEIGHT}  C_VALUE={C_VALUE}  "
              f"THRESHOLD={THRESHOLD}")
    else:
        print(f"  MODEL=nb (Naive Bayes)  NB_ALPHA={NB_ALPHA}  THRESHOLD={THRESHOLD}  "
              f"(CLASS_WEIGHT/C_VALUE not used by NB)")
    print()


def print_class_distribution(y: pd.Series, title: str) -> None:
    counts = y.value_counts().reindex(VALID_LABELS, fill_value=0)
    total = int(counts.sum())
    print(f"  {title} (n={total})")
    for label, count in counts.items():
        print(f"    {label:<5} {count:>6}  ({count / total:6.2%})")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """TF-IDF + classifier in one Pipeline, so fit() only ever sees training data."""
    vectoriser = TfidfVectorizer(
        lowercase=LOWERCASE,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_features=MAX_FEATURES,
        stop_words=STOP_WORDS,
        sublinear_tf=True,
    )
    if MODEL == "lr":
        classifier = LogisticRegression(C=C_VALUE, class_weight=CLASS_WEIGHT, solver="liblinear",
                                        max_iter=1000, random_state=RANDOM_STATE)
    else:
        classifier = MultinomialNB(alpha=NB_ALPHA)
    return Pipeline([("tfidf", vectoriser), ("clf", classifier)])


def spam_probability(pipeline: Pipeline, texts) -> np.ndarray:
    spam_col = list(pipeline.classes_).index(POSITIVE_LABEL)
    return pipeline.predict_proba(texts)[:, spam_col]


def predict_labels(pipeline: Pipeline, texts) -> np.ndarray:
    """Apply THRESHOLD to P(spam) instead of the library's fixed 0.5."""
    return np.where(spam_probability(pipeline, texts) >= THRESHOLD, POSITIVE_LABEL, "ham")


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=list(VALID_LABELS))
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "cm": cm,
    }


def evaluate(pipeline: Pipeline, X_train, y_train, X_test, y_test) -> dict:
    m = compute_metrics(y_test, predict_labels(pipeline, X_test))
    train_m = compute_metrics(y_train, predict_labels(pipeline, X_train))
    m["train_f1"] = train_m["f1"]

    print(f"\n[4] Evaluation on held-out test set (positive class = 'spam', THRESHOLD={THRESHOLD})")
    print(f"  Accuracy  = (TP+TN)/all    = {m['accuracy']:.4f}")
    print(f"  Precision = TP/(TP+FP)     = {m['precision']:.4f}")
    print(f"  Recall    = TP/(TP+FN)     = {m['recall']:.4f}")
    print(f"  F1-score  = 2PR/(P+R)      = {m['f1']:.4f}")
    print("\n  Confusion matrix (rows = actual, columns = predicted)")
    print(f"  {'':>14}{'pred ham':>12}{'pred spam':>12}")
    print(f"  {'actual ham':>14}{m['tn']:>12}{m['fp']:>12}    TN={m['tn']}  FP={m['fp']}")
    print(f"  {'actual spam':>14}{m['fn']:>12}{m['tp']:>12}    FN={m['fn']}  TP={m['tp']}")
    print(f"\n  FP: {m['fp']} legitimate message(s) wrongly flagged as spam")
    print(f"  FN: {m['fn']} spam message(s) missed")
    print(f"\n  Training-set F1 = {train_m['f1']:.4f}  vs  test-set F1 = {m['f1']:.4f}"
          f"  (gap = {train_m['f1'] - m['f1']:+.4f})")
    print("  (Training score is NOT a valid result - a big gap means overfitting.)")
    print("\n  Per-class report:")
    y_pred = predict_labels(pipeline, X_test)
    print(classification_report(y_test, y_pred, labels=list(VALID_LABELS), digits=4, zero_division=0))
    return m


def majority_baseline(y_train: pd.Series, y_test: pd.Series) -> None:
    """Always predicts the most frequent training class -> shows why accuracy alone misleads."""
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    print(f"  Baseline 'always {dummy.classes_[np.argmax(dummy.class_prior_)]}': "
          f"accuracy={acc:.4f}, spam recall={rec:.4f}")


def show_top_features(pipeline: Pipeline, n: int) -> None:
    clf = pipeline.named_steps["clf"]
    if n <= 0:
        return
    terms = pipeline.named_steps["tfidf"].get_feature_names_out()
    if isinstance(clf, LogisticRegression):
        score = clf.coef_[0] if clf.classes_[1] == POSITIVE_LABEL else -clf.coef_[0]
    else:  # Naive Bayes: log P(word|spam) - log P(word|ham)
        s = list(clf.classes_).index(POSITIVE_LABEL)
        score = clf.feature_log_prob_[s] - clf.feature_log_prob_[1 - s]
    n = min(n, len(terms) // 2) or 1
    order = np.argsort(score)
    print(f"\n  Top {n} terms pushing towards SPAM : {', '.join(terms[order[-n:][::-1]])}")
    print(f"  Top {n} terms pushing towards HAM  : {', '.join(terms[order[:n]])}")


def save_confusion_plot(cm: np.ndarray, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        print("  [plot] matplotlib not installed; skipped (pip install matplotlib)")
        return
    disp = ConfusionMatrixDisplay(cm, display_labels=list(VALID_LABELS))
    disp.plot(cmap="Blues", colorbar=False, values_format="d")
    plt.title("Confusion matrix - held-out test set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  [plot] saved -> {out_path}")


# --------------------------------------------------------------------------
# Run log (compare runs with different settings)
# --------------------------------------------------------------------------
LOG_FIELDS = ["time", "model", "test_size", "seed", "train_frac", "dedup", "ngram", "min_df",
              "max_feat", "stop", "class_wt", "C", "nb_alpha", "threshold", "n_train", "n_test",
              "features", "accuracy", "precision", "recall", "f1", "train_f1", "TN", "FP", "FN", "TP"]


def log_run(m: dict, n_train: int, n_test: int, n_features: int) -> None:
    row = {
        "time": datetime.now().strftime("%H:%M:%S"), "model": MODEL, "test_size": TEST_SIZE,
        "seed": RANDOM_STATE, "train_frac": TRAIN_FRACTION, "dedup": DROP_DUPLICATES,
        "ngram": f"{NGRAM_RANGE[0]}-{NGRAM_RANGE[1]}", "min_df": MIN_DF, "max_feat": MAX_FEATURES,
        "stop": STOP_WORDS, "class_wt": CLASS_WEIGHT if MODEL == "lr" else "-",
        "C": C_VALUE if MODEL == "lr" else "-", "nb_alpha": NB_ALPHA if MODEL == "nb" else "-",
        "threshold": THRESHOLD, "n_train": n_train, "n_test": n_test, "features": n_features,
        "accuracy": round(m["accuracy"], 4), "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4), "f1": round(m["f1"], 4), "train_f1": round(m["train_f1"], 4),
        "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"],
    }
    try:
        new_file = not RESULTS_LOG.exists()
        with RESULTS_LOG.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
    except OSError as exc:  # e.g. file open in Excel on Windows
        print(f"  [log] could not write {RESULTS_LOG.name}: {exc} (close it in Excel and re-run)")


def show_last_runs(n: int) -> None:
    if n <= 0 or not RESULTS_LOG.exists():
        return
    try:
        log = pd.read_csv(RESULTS_LOG).tail(n)
    except (pd.errors.ParserError, OSError) as exc:
        print(f"  [log] could not read {RESULTS_LOG.name}: {exc} (delete the file to reset)")
        return
    cols = ["time", "model", "test_size", "seed", "train_frac", "dedup", "max_feat", "class_wt",
            "threshold", "n_train", "accuracy", "precision", "recall", "f1", "FP", "FN"]
    cols = [c for c in cols if c in log.columns]
    print(f"\n[6] Last {len(log)} run(s) from {RESULTS_LOG.name} (delete the file to start fresh)")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(log[cols].to_string(index=False))


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def explain_prediction(pipeline: Pipeline, text: str) -> None:
    p_spam = float(spam_probability(pipeline, [text])[0])
    label = POSITIVE_LABEL if p_spam >= THRESHOLD else "ham"
    print(f"  Message   : {text}")
    print(f"  Predicted : {LABEL_MEANING[label]}")
    print(f"  P(spam)   : {p_spam:.2%}  (threshold {THRESHOLD:.0%}; a model estimate, not a guarantee)")


def interactive_loop(pipeline: Pipeline) -> None:
    print("\n[5] Interactive prediction - type a message, or press Enter / type 'quit' to exit")
    print("    Try, for example:")
    for example in EXAMPLE_MESSAGES:
        print(f"      - {example}")
    while True:
        try:
            text = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"", "quit", "exit", "q"}:
            break
        explain_prediction(pipeline, text)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised SMS spam classification demo "
                                            "(edit DEMO SETTINGS at the top of the file)")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to SMSSpamCollection")
    p.add_argument("--download", action="store_true", help="download the dataset from UCI first")
    p.add_argument("--fallback", action="store_true",
                   help="use the tiny built-in SYNTHETIC dataset (illustrative only)")
    p.add_argument("--message", type=str, help="classify one message and skip the interactive loop")
    p.add_argument("--no-interactive", action="store_true", help="skip the interactive loop")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        check_settings()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_settings()

    # [1] Load
    print("[1] Loading data")
    if args.fallback:
        print("  !! SYNTHETIC FALLBACK DATA - illustrative only. Metrics below are NOT "
              "research results and must not be reported. !!")
        raw = pd.DataFrame(FALLBACK_MESSAGES, columns=["label", "message"])
        malformed = 0
    else:
        try:
            if args.download or not args.data.is_file():
                if not args.download:
                    print(f"  {args.data} not found; attempting download (use --fallback to skip)")
                download_dataset(args.data)
            raw, malformed = load_sms_collection(args.data)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("hint : re-run with --fallback to test the code on synthetic data", file=sys.stderr)
            return 1
        print(f"  source: {args.data}")
    print(f"  malformed lines skipped   : {malformed}")

    # [2] Validate
    print("\n[2] Validating and cleaning")
    try:
        df = validate_and_clean(raw, drop_duplicates=DROP_DUPLICATES)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_class_distribution(df["label"], "Usable dataset")

    # [3] Split BEFORE any fitting, stratified so both sets keep the class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE,
    )
    print(f"\n[3] Train/test split (TEST_SIZE={TEST_SIZE}, RANDOM_STATE={RANDOM_STATE}, stratified)")
    if TRAIN_FRACTION < 1.0:
        try:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=TRAIN_FRACTION, stratify=y_train,
                random_state=RANDOM_STATE)
        except ValueError as exc:
            print(f"error: TRAIN_FRACTION={TRAIN_FRACTION} is too small for this data ({exc})",
                  file=sys.stderr)
            return 2
        print(f"  TRAIN_FRACTION={TRAIN_FRACTION}: model learns from only part of the training set")
    print_class_distribution(y_train, "Training set")
    print_class_distribution(y_test, "Test set")

    pipeline = build_pipeline()
    try:
        pipeline.fit(X_train, y_train)     # TF-IDF + classifier fitted on TRAINING data only
    except ValueError as exc:              # e.g. MIN_DF too high -> empty vocabulary
        print(f"error: training failed ({exc}). Try a lower MIN_DF or a higher TRAIN_FRACTION.",
              file=sys.stderr)
        return 2
    n_features = len(pipeline.named_steps["tfidf"].vocabulary_)
    model_name = (f"Logistic Regression (CLASS_WEIGHT={CLASS_WEIGHT}, C={C_VALUE})"
                  if MODEL == "lr" else f"Multinomial Naive Bayes (alpha={NB_ALPHA})")
    print(f"  Model: TF-IDF ({n_features} features learned from training set) -> {model_name}")

    # [4] Evaluate
    metrics = evaluate(pipeline, X_train, y_train, X_test, y_test)
    majority_baseline(y_train, y_test)
    show_top_features(pipeline, TOP_FEATURES)
    if SAVE_PLOT:
        save_confusion_plot(metrics["cm"], PLOT_PATH)
    if args.fallback:
        print("\n  !! Reminder: synthetic data - the numbers above are meaningless. !!")
    else:
        log_run(metrics, len(X_train), len(X_test), n_features)
        show_last_runs(SHOW_LAST_RUNS)

    # [5] Predict new messages
    if args.message:
        print("\n[5] Prediction for supplied message")
        explain_prediction(pipeline, args.message)
    elif not args.no_interactive:
        interactive_loop(pipeline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
