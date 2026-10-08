# 📝 Fake Review Classifier

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

An NLP-based system that distinguishes **Computer-Generated (CG)** product reviews from **Original (OR)** ones using TF-IDF feature extraction and a classical Machine Learning classifier, served through an interactive Streamlit web app.

> **Scope note:** this model distinguishes Computer-Generated (CG) reviews from Original (OR) reviews **according to the labeling of the training dataset used** (GPT-2-generated review text vs. real verified-purchase Amazon reviews). It does **not** claim to detect every type of fake review — see [Known Limitations](#-known-limitations-read-this) below.

---

## 📌 Overview

Online marketplaces are increasingly targeted by AI-generated fake reviews designed to manipulate product ratings. This project builds a text classification pipeline that flags a review as **Fake (CG)** or **Genuine (OR)** based on its written content, using the [Fake Reviews Dataset](https://osf.io/tyue9/) (Salminen et al.) of real vs. GPT-2-generated Amazon reviews.

## 🎯 Problem Statement

Given a product review's text, predict whether it was:
- **Genuine / OR (0)** — a real, verified-purchase review, or
- **Fake / CG (1)** — a GPT-2-generated review designed to mimic real feedback.

## ✨ Features

- 📂 Automated data profiling (missing values, duplicates, class balance)
- 🧹 Full NLP text-cleaning pipeline, **identical in training and inference** (URLs, HTML, emojis, punctuation, numbers, stopwords, lemmatization)
- 🔢 TF-IDF feature extraction, tuned via `GridSearchCV`
- 🤖 5-model comparison with 5-fold cross-validation (Logistic Regression, Linear SVM, Naive Bayes, Random Forest, XGBoost)
- 📊 Full evaluation suite: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, probability distributions, decision-threshold sweep
- 🔍 Formal error analysis (100 random test samples + 20 manually written reviews) with every miss explained
- 📈 Interactive Streamlit app: confidence scores, model metadata, timestamp, top influential terms per prediction, example reviews
- ✅ Robust error handling and input validation

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| NLP | NLTK (stopwords, WordNet lemmatizer) |
| ML | scikit-learn (TF-IDF, Logistic Regression, Linear SVM, Naive Bayes, Random Forest), XGBoost |
| Visualization | Matplotlib, Seaborn |
| App | Streamlit |
| Dev environment | Jupyter Notebook |

## 📂 Dataset

`data/fake_reviews_dataset.csv` — 40,432 Amazon product reviews.

| Column | Description |
|---|---|
| `category` | Product category |
| `rating` | Star rating (1–5) |
| `label` | `CG` = computer-generated (fake), `OR` = original (genuine) |
| `text_` | The review text |

| Metric | Value |
|---|---|
| Rows | 40,432 |
| Missing values | 0 |
| Duplicate reviews | 20 (removed during cleaning) |
| Class balance | ~50% CG / ~50% OR — perfectly balanced |

## 📁 Project Structure

```
fake-review-classifier/
│
├── app.py                          # Streamlit application
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── fake_reviews_dataset.csv    # Raw dataset
│   └── cleaned_fake_reviews.csv    # Cleaned dataset (generated)
│
├── models/
│   ├── classifier.pkl              # Trained classifier (Linear SVM, calibrated)
│   └── vectorizer.pkl              # Fitted TF-IDF vectorizer
│
├── notebooks/
│   ├── data_processing.ipynb       # Cleaning & preprocessing
│   └── model_training.ipynb        # GridSearch, model comparison, bias/error analysis, feature importance
│
├── reports/
│   ├── model_comparison.csv        # All 5 models, CV + test metrics, timing
│   ├── tfidf_gridsearch_results.csv
│   ├── best_tfidf_params.json
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── probability_distribution.png
│   ├── error_analysis.json         # 100 random + 20 manual review results
│   └── top_features.json           # Top TF-IDF features per class
│
└── screenshots/                    # App UI screenshots (see screenshots/README.md)
```

## 🔬 Step 1 — Root Cause Diagnosis (why "obviously fake" reviews were predicted Genuine)

A prior version of this model appeared to classify almost everything as "Genuine." A full pipeline audit checked every likely cause:

| Checked | Result |
|---|---|
| Inverted / incorrect label mapping | ❌ Not the cause — confirmed `label_num`: `OR→0`, `CG→1`, and `model.classes_ == [0, 1]` throughout |
| Dataset imbalance | ❌ Not the cause — classes are ~50/50 |
| Different preprocessing in training vs. inference | ❌ Not the cause — `app.py`'s `clean_text()` is line-for-line identical to `notebooks/data_processing.ipynb`'s |
| Poor TF-IDF configuration | ⚠️ **Contributing factor** — an earlier version used unconstrained trigrams with too many features; grid search (Step 3 below) fixed this |
| Overfitting | ⚠️ **Contributing factor** — earlier version had a 6.6-point train/test accuracy gap (97.8% vs 91.2%) from unregularized trigrams; reduced to 4.9 points (95.4% vs 90.5%) with `class_weight='balanced'`, `C=0.5`, and GridSearch-tuned n-grams |
| Poor probability threshold | ❌ Checked via a 0.4–0.6 threshold sweep — 0.5 is already near-optimal for F1, no shift needed |
| Bugs in `app.py` / model loading / vectorizer mismatch | ❌ Not found — verified the app loads the correct artifacts and applies matching preprocessing |
| **Dataset-scope mismatch** | ✅ **Root cause of the specific complaint** — see below |

**The actual root cause:** the training dataset's "Fake (CG)" class is GPT-2-generated text written to *sound like* a real Amazon review — not spam, not promotional ad copy, not ALL-CAPS hype. Reviewing the model's own top TF-IDF features (Step 7) confirms this: no spam/promotional vocabulary appears anywhere in the "Fake" indicator list. The model was never shown "BUY NOW!!! CLICK HERE!!!"-style text labeled as fake, so it has no basis to associate that style with the Fake class. **On data from the same distribution as training, the pipeline is accurate and unbiased** (see Step 6). The mismatch is specifically between the dataset's definition of "fake" and a human's intuitive definition of "obviously fake" (which often means spammy/promotional). This is documented plainly rather than papered over — see [Known Limitations](#-known-limitations-read-this).

## 🧹 Step 2 — Data Preprocessing (identical in training and inference)

Verified byte-for-byte identical between `notebooks/data_processing.ipynb` and `app.py`:

1. **URL removal**
2. **HTML tag removal**
3. **Emoji removal**
4. **Lowercasing**
5. **Number removal**
6. **Punctuation removal**
7. **Whitespace normalization**
8. **Stopword removal**
9. **Lemmatization** (WordNet)
10. **Duplicate removal** (training data only)

## 🔢 Step 3 — TF-IDF GridSearch

`GridSearchCV` (3-fold, scored on F1) over:

```
ngram_range:   (1,1), (1,2)
min_df:        2, 5
max_df:        0.90, 0.95
max_features:  5000, 10000, 15000
```

**Best config:** `ngram_range=(1,2)`, `min_df=2`, `max_df=0.90`, `max_features=10000`. Full grid results: `reports/tfidf_gridsearch_results.csv`.

**Why it wins:** bigrams capture short discriminative phrases (`"would recommend"`, `"also love"`) that unigrams miss; `min_df=2` filters one-off noise; `max_df=0.9` drops near-universal words; capping at 10k features balances vocabulary coverage against overfitting risk.

## 🤖 Step 4 — Model Comparison (5-fold Cross-Validation)

| Model | CV F1 | Test Accuracy | Test F1 | Test ROC-AUC | Train Time (s) | Predict Time (s) |
|---|---|---|---|---|---|---|
| **Linear SVM** ✅ | **0.885** | **90.5%** | **0.905** | **0.969** | 0.27 | 0.0015 |
| Logistic Regression | 0.871 | 89.6% | 0.896 | 0.964 | 0.15 | 0.0012 |
| Multinomial Naive Bayes | 0.869 | 87.5% | 0.874 | 0.954 | 0.01 | 0.0019 |
| XGBoost | 0.795 | 82.6% | 0.819 | 0.913 | 19.07 | 0.0296 |
| Random Forest | 0.794 | 82.3% | 0.812 | 0.911 | 5.24 | 0.0946 |

Full table with precision/recall: `reports/model_comparison.csv`.

**Why Linear SVM wins:** TF-IDF features are high-dimensional and sparse; linear models handle that better than tree-based models, which split less efficiently across thousands of sparse dimensions and are far slower here. Naive Bayes is fast but its independence assumption caps its ceiling. Linear SVM is wrapped in `CalibratedClassifierCV` for well-calibrated probability output (needed for the app's confidence score).

## 📊 Step 5 — Bias Investigation

- **Class distribution (test set):** 4,043 Genuine / 4,040 Fake — balanced.
- **Prediction distribution:** 4,032 Genuine / 4,051 Fake — balanced, no skew toward either class.
- **Confusion matrix:** 3,653/4,043 Genuine correct (90.4%), 3,661/4,040 Fake correct (90.6%) — nearly identical error rates on both classes.
- **Probability distribution:** clearly bimodal, separated by true class (`reports/probability_distribution.png`) — the model is confidently discriminating, not defaulting to one side.
- **Threshold sweep (0.4–0.6):** F1 peaks essentially at the default 0.5 threshold — no threshold correction needed.

**Conclusion: on data from the training distribution, this model shows no bias toward either class.**

## 🔎 Step 6 — Error Analysis

**(a) 100 random test-set samples:** 91/100 correct. The 9 errors split roughly evenly — 4 actual-Fake reviews called Genuine, 5 actual-Genuine reviews called Fake. No directional bias.

**(b) 20 manually written reviews** (10 written to sound like classic promotional spam, 10 written to sound like typical honest reviews — evaluated against human-intuition labels, *not* the dataset's own definition):

- All **10/10** "genuine-style" reviews were classified correctly.
- Only **3/10** "spam-style" reviews were classified as Fake; the other 7 were called Genuine.
- **Why:** as explained in Step 1, the training data's Fake examples don't include spam/hype-style text, so the model has no learned association between that style and the Fake label. This isn't a random error — it's a systematic, explainable gap tied to what the dataset actually contains. Full results: `reports/error_analysis.json`.

## 🧠 Step 7 — Feature Analysis

Top features by trained model coefficient (not inferred or guessed — read directly from `LinearSVC.coef_`, averaged across `CalibratedClassifierCV`'s folds):

**Top toward Fake:** `reason gave`, `admit`, `also love`, `problem really`, `downside`, `would recommend`, `bought friend`, `story start`, `material good`

**Top toward Genuine:** `even`, `though`, `without`, `much`, `however`, `really`, `instead`, `could`, `actually`, `otherwise`

These are GPT-2 stylistic tics (Fake) vs. natural human hedging/qualifying language (Genuine) — not spam indicators. Full list: `reports/top_features.json`.

## 🖥️ Step 8 — Streamlit App

- Sidebar with pipeline description, model metadata (name, test accuracy, test F1), and an explicit limitations warning
- Example review dropdown
- Input validation (empty, too short, symbols-only)
- Spinner while classifying
- Color-coded result (🟥 Fake, 🟩 Genuine) with confidence progress bar and **prediction timestamp**
- **Most influential TF-IDF terms found in the review**, shown as "influential features," explicitly *not* claimed as a verified causal explanation
- Clear button, friendly error messages for missing model files
- Limitations footer restating the CG-vs-OR scope

## 🚀 Installation

```bash
git clone <your-repo-url>
cd fake-review-classifier
pip install -r requirements.txt
```

## ▶️ How to Run

Model artifacts (`models/classifier.pkl`, `models/vectorizer.pkl`) are already included.

```bash
streamlit run app.py
```

To regenerate everything from scratch, run the notebooks in order:
1. `notebooks/data_processing.ipynb`
2. `notebooks/model_training.ipynb`

**Compute note:** this was developed on a single-CPU-core container. The GridSearch and 5-fold CV model comparison in `model_training.ipynb` run on stratified subsamples for tractability; the winning model/config is refit on the full training set for the final saved artifacts and all reported test-set metrics (this is called out inline in the notebook).

## ⚠️ Known Limitations (read this)

- **This is a CG-vs-OR classifier, not a general fake-review detector.** It reliably distinguishes GPT-2-generated review text from real verified-purchase reviews, *as defined by this specific dataset*. It was not trained to detect spam, incentivized reviews, coordinated review farms, or promotional copy — those patterns don't appear in the "Fake" training examples, and the error analysis above (Step 6) demonstrates this concretely rather than asserting it.
- **Confidence on out-of-distribution text is less reliable.** Very short or stylistically unusual text can get lower-confidence or incorrect predictions.
- **The "influential terms" shown in the app are feature weights, not a verified explanation** of the model's reasoning — they indicate which TF-IDF features had the largest weighted contribution, nothing more.
- **Mild residual overfitting remains** (train 95.4% vs. test 90.5%). Further regularization, more training data, or ensembling could narrow this further.

## 🚀 Future Improvements

- Augment training data with labeled spam/promotional examples if the goal expands beyond CG-vs-OR detection.
- Fine-tune a transformer (BERT / RoBERTa / DistilBERT / Sentence-Transformers) for stronger generalization.
- Full-grid (not subsampled) hyperparameter search once more compute is available.
- Batch review upload (CSV) for bulk predictions.
- Deploy to Streamlit Community Cloud.

---

## 👩‍💻 Author

**Khushi R**

AI & Data Science Graduate
- GitHub: [Khushi-datascientist](https://github.com/Khushi-datascientist)
- LinkedIn: [itz-khushi](https://www.linkedin.com/in/itz-khushi)

## 📄 License

This project is licensed under the [MIT License](LICENSE).
