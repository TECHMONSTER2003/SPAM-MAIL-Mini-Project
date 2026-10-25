# Supervised Message Filtering — Demo Code - Vijayendiran A G 

 Hi Learners, Give a read on this readme file, and it will help you understand the code! Happy learning ;)

SMS spam classifier built with TF-IDF + Logistic Regression (scikit-learn) 

> **Scope:** the dataset is the **UCI SMS Spam Collection** — short SMS messages, mostly UK spam and
> Singapore student ham, collected around 2006–2011. Results on it show how supervised text
> classification works. They do **not** show that this model can detect phishing e-mail reliably.

## 1. Files

```
spam_demo/
├── spam_filter_demo.py      # complete script: load → validate → split → train → evaluate → predict
├── requirements.txt
├── README.md
└── data/
    └── SMSSpamCollection    # you place (or --download) this file here
```

## 2. Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

No GPU is needed. Training takes a few seconds on a laptop CPU.

## 3. Dataset setup

| Item | Detail |
|---|---|
| Name | SMS Spam Collection |
| Source | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| Creators | Tiago A. Almeida, José María Gómez Hidalgo |
| DOI | 10.24432/C5CC84 |
| Licence | CC BY 4.0 (credit the creators) |
| Format | One file, `SMSSpamCollection`; each line is `label<TAB>message`; labels `ham` / `spam` |
| Size listed by UCI | 5,574 instances |

**Option A (automatic):**

```bash
python spam_filter_demo.py --download
```

**Option B (manual):** download the zip from the UCI page, extract it, copy the file named
`SMSSpamCollection` (no extension) into `spam_demo/data/`.

If the file is missing, the script tries the download automatically and, if that fails, tells you
exactly where to put the file.

## 4. Running

```bash
python spam_filter_demo.py                         # full run + interactive prediction
python spam_filter_demo.py --plot                  # also saves confusion_matrix.png
python spam_filter_demo.py --message "Free entry to win a prize, text WIN to 80888"
python spam_filter_demo.py --model nb              # Multinomial Naive Bayes
python spam_filter_demo.py --class-weight none     # unweighted LR, to compare recall
python spam_filter_demo.py --keep-duplicates       # observe the effect of duplicate texts
```

Live demo (60–90 s): run the default command, point at the class distribution, the four metrics and
the confusion matrix, then type two of the example messages below.

## 5. Example messages for the interactive prompt

| # | Message | Purpose |
|---|---|---|
| 1 | `Congratulations! You have won a free holiday. Call 09061234567 now to claim your prize` | Resembles the prize/premium-number spam in the training data |
| 2 | `Hi, are we still meeting at the library at 5 today?` | Ordinary personal message |
| 3 | `Your account has been suspended. Verify your details at the link immediately` | Phishing-style wording that is rare in this SMS dataset — a probe of generalisation |

In the sandbox run (Section 8), message 3 was classified as spam with a score of only about 53 %,
i.e. close to the decision boundary. Use it to show that the model is uncertain on text that does
not resemble its training data. Your score may differ slightly with a different library version.

## 6. What each section of the code does

| Section | Function(s) | What happens |
|---|---|---|
| [1] Load | `download_dataset`, `load_sms_collection` | Reads the file line by line and splits on the **first** tab. A CSV parser is avoided because quote characters in messages can merge or drop rows silently. Malformed lines are counted, not hidden. |
| [2] Validate | `validate_and_clean` | Checks the two expected columns, normalises labels, drops rows whose label is not `ham`/`spam`, empty messages, texts that appear with **both** labels, and exact duplicate texts. Prints how many rows each rule removed. |
| [3] Split | `train_test_split(..., stratify=..., random_state=42)` | Holds out 20 % as a test set **before any fitting**. Stratification keeps the spam ratio the same in both sets. |
| Train | `build_pipeline` → `pipeline.fit(X_train, y_train)` | TF-IDF vectoriser and classifier are chained in one `Pipeline`, so both are fitted on training data only. |
| [4] Evaluate | `evaluate`, `majority_baseline`, `show_top_features` | Predicts the test set; prints accuracy, precision, recall, F1, confusion matrix, a per-class report, a "always ham" baseline and the terms with the largest weights. |
| [5] Predict | `predict_message`, `interactive_loop` | Vectorises a new message with the **training** vocabulary, outputs the class and the model's probability estimate. |

### Why duplicates are removed

The raw file contains repeated texts (the sandbox run removed 414). If one copy lands in the
training set and another in the test set, the model is being tested on a message it has already
seen. That is a form of data leakage and makes the test score optimistic. `--keep-duplicates`
lets you compare. In the sandbox run, keeping duplicates gave higher scores, but the test set was
also different, so the difference cannot be attributed to leakage alone.

## 7. Concepts used in the code

**Why text must become numbers.** Logistic Regression computes a weighted sum of numeric inputs.
A message has to be turned into a fixed-length numeric vector first.

**TF-IDF (term frequency–inverse document frequency).** Each column is a word or word pair from the
training vocabulary. A term's value in a message grows with how often it appears in that message
(TF, here `1 + log(tf)`) and shrinks if the term appears in many training messages (IDF). Common
words such as "the" get little weight; distinctive words such as "claim" or "txt" get more. Each
row is then L2-normalised so long and short messages are comparable. Settings used:
`ngram_range=(1, 2)` (words and word pairs), `min_df=2` (ignore terms seen once).

**Model fitting.** Logistic Regression learns one weight per TF-IDF feature plus a bias. Fitting
adjusts these weights to minimise log-loss on the training labels. `class_weight="balanced"`
increases the penalty for mistakes on the rarer spam class so the model does not simply favour
"ham". Despite its name, Logistic Regression is a classifier: it outputs a probability through
the sigmoid function and assigns a class by thresholding it (default 0.5).

**Prediction.** For a new message the fitted vectoriser maps it onto the training vocabulary (words
never seen in training are ignored), the classifier computes the probability of spam, and the
higher-probability class is reported.

**Why training and test data are separate.** The test set stands in for future, unseen messages.
Scoring on training data measures memorisation, not generalisation.

**Interpreting the metrics (positive class = spam).**

| Metric | Question it answers | Cost of a low value in this task |
|---|---|---|
| Accuracy | What fraction of all messages was labelled correctly? | Can look high even when spam is missed, because ham is ~87 % of the data |
| Precision | Of messages flagged as spam, how many really were spam? | Low → legitimate messages blocked (false positives) |
| Recall | Of all real spam, how much was caught? | Low → unwanted messages delivered (false negatives) |
| F1 | Harmonic mean of precision and recall | Summarises the trade-off in one number |

The script prints a majority-class baseline: always predicting "ham" already gives about 88 %
accuracy with zero spam recall, which is why accuracy alone is not enough.

**Why predictions are not security guarantees.** The model has only learned statistical patterns
from one historical SMS corpus. New spam styles, other languages, URLs, attackers deliberately
changing wording (adversarial adaptation) and changes in message traffic over time (distribution
shift) can all reduce performance. A probability of 0.95 is the model's estimate, not a certainty.

## 8. Execution record (sandbox verification)

The code was executed in a sandbox where the UCI host was unreachable, so it was run on a public
GitHub mirror of the same `SMSSpamCollection` file. That copy has 5,574 lines with 4,827 ham and
747 spam, matching the instance count listed by UCI. **Re-run on your machine with the official
UCI file and use your own output in the presentation.**

Configuration: Python 3.12.3, scikit-learn 1.8.0, `random_state=42`, `test_size=0.2`,
Logistic Regression `class_weight="balanced"`.

```
rows read 5574 | duplicates dropped 414 | usable 5160 (ham 4518, spam 642)
train 4128 | test 1032 (ham 904, spam 128)
Accuracy 0.9806 | Precision 0.9030 | Recall 0.9453 | F1 0.9237
TN=891  FP=13  FN=7  TP=121
Baseline 'always ham': accuracy 0.8760, spam recall 0.0000
```

Comparison runs on the same split (same sandbox):

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| LR, `class_weight="balanced"` (default) | 0.9806 | 0.9030 | 0.9453 | 0.9237 |
| LR, `--class-weight none` | 0.9651 | 1.0000 | 0.7188 | 0.8364 |
| Multinomial NB (`--model nb`) | 0.9632 | 1.0000 | 0.7031 | 0.8257 |

These are single runs on one split. They are not a statistically tested comparison.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Package installed in another interpreter | Activate the venv, then `python -m pip install -r requirements.txt` |
| `Dataset not found` / `Download failed` | No file in `data/`, or UCI blocked by network/proxy | Download the zip in a browser, extract `SMSSpamCollection` into `data/`, or pass `--data <path>` |
| File extracted as `SMSSpamCollection.txt` or inside a sub-folder | Archive tool behaviour | Rename/move it, or pass the actual path with `--data` |
| `Need both 'ham' and 'spam' rows` | Wrong file (e.g. the `readme`) | Point `--data` to `SMSSpamCollection` |
| Large "invalid label" count | File saved with a different format (e.g. CSV with header) | Use the original UCI file, unmodified |
| `UnicodeDecodeError` | Should not occur — reader uses `errors="replace"` | Re-download the file if text looks corrupted |
| Interactive prompt does nothing in an IDE | Some IDE consoles don't support `input()` | Run in a terminal, or use `--message "..."` |
| Metrics slightly different from Section 8 | Different scikit-learn version or file copy | Expected; report your own run and its versions |
| `--plot` prints "matplotlib not installed" | Optional dependency missing | `pip install matplotlib` |

## 10. Fallback (no dataset available)

```bash
python spam_filter_demo.py --fallback
```

Uses 28 **synthetic** messages written for this demo, not taken from any dataset. It only proves
the code runs end to end. With 6 test messages the metrics are meaningless; the script prints a
warning before and after them. **Do not report fallback numbers as results.**

## 11. References

1. T. A. Almeida and J. M. Gómez Hidalgo, "SMS Spam Collection," UCI Machine Learning Repository, 2011, doi: 10.24432/C5CC84.
2. T. A. Almeida, J. M. Gómez Hidalgo, and A. Yamakami, "Contributions to the study of SMS spam filtering: New collection and results," in *Proc. ACM Symp. Document Engineering (DocEng)*, 2011, doi: 10.1145/2034691.2034742.
3. F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.
4. scikit-learn developers, "Feature extraction." https://scikit-learn.org/stable/modules/feature_extraction.html
5. scikit-learn developers, "Metrics and scoring." https://scikit-learn.org/stable/modules/model_evaluation.html
