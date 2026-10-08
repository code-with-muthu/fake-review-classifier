"""
Fake Review Classifier — Streamlit Application
================================================

Loads a trained TF-IDF vectorizer + Linear SVM classifier and predicts
whether a submitted product review is Fake (computer-generated, per the
training dataset's CG/OR labeling) or Genuine.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import re
import string
from datetime import datetime
from typing import List, Optional, Tuple

import joblib
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_PATH = "models/classifier.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
BEST_MODEL_FILE = "reports/best_model.txt"
COMPARISON_FILE = "reports/model_comparison.csv"
MAX_REVIEW_CHARS = 5000
MIN_REVIEW_CHARS = 5
TOP_TERMS_TO_SHOW = 8

EXAMPLE_REVIEWS = {
    "Example: likely genuine": (
        "The zipper broke after two weeks and customer service never responded "
        "to my emails. Decent design otherwise, but the build quality could be "
        "better for the price."
    ),
    "Example: likely fake (dataset-style)": (
        "I purchased this unit for my wife and she loves it. She also loves "
        "how easy it is to use and how great it looks in our kitchen. Highly "
        "recommend this product to everyone looking for a great gift!"
    ),
}


# ---------------------------------------------------------------------------
# Text cleaning (must mirror notebooks/data_processing.ipynb exactly)
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


@st.cache_resource(show_spinner=False)
def get_stopwords_and_lemmatizer():
    """Lazily download/load NLTK resources once per session."""
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    return set(stopwords.words("english")), WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Apply the exact same cleaning pipeline used in data_processing.ipynb.

    Steps: URL removal, HTML tag removal, emoji removal, lowercasing,
    number removal, punctuation removal, whitespace normalization,
    stopword removal, and lemmatization. Keeping this identical to the
    training-time preprocessing avoids train/inference skew.
    """
    stop_words, lemmatizer = get_stopwords_and_lemmatizer()

    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [w for w in words if w.isalpha() and w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts() -> Tuple[Optional[object], Optional[object], Optional[str]]:
    """Load the trained model and vectorizer.

    Returns:
        (model, vectorizer, error_message). error_message is None on success.
    """
    if not os.path.exists(MODEL_PATH):
        return None, None, (
            f"Model file not found at `{MODEL_PATH}`. Run `notebooks/data_processing.ipynb` "
            "and `notebooks/model_training.ipynb` first to generate it."
        )
    if not os.path.exists(VECTORIZER_PATH):
        return None, None, (
            f"Vectorizer file not found at `{VECTORIZER_PATH}`. Run "
            "`notebooks/data_processing.ipynb` and `notebooks/model_training.ipynb` first."
        )
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
    except Exception as exc:  # noqa: BLE001 - surface any load error to the UI
        return None, None, f"Failed to load model artifacts: {exc}"

    return model, vectorizer, None


@st.cache_data(show_spinner=False)
def load_metadata() -> dict:
    """Load model metadata (best model name, comparison table) if available."""
    meta = {"model_name": "Unknown", "test_f1": None, "test_accuracy": None}
    if os.path.exists(BEST_MODEL_FILE):
        with open(BEST_MODEL_FILE) as f:
            meta["model_name"] = f.read().strip()
    if os.path.exists(COMPARISON_FILE):
        import pandas as pd
        comp = pd.read_csv(COMPARISON_FILE)
        row = comp[comp["Model"] == meta["model_name"]]
        if not row.empty:
            meta["test_f1"] = float(row.iloc[0].get("Test_F1", np.nan))
            meta["test_accuracy"] = float(row.iloc[0].get("Test_Accuracy", np.nan))
    return meta


def get_linear_coef(model) -> Optional[np.ndarray]:
    """Extract a linear coefficient vector from the model if available.

    Handles CalibratedClassifierCV-wrapped linear models by averaging the
    coefficients across its internal calibrated estimators.
    """
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)[0]
    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for cc in model.calibrated_classifiers_:
            est = getattr(cc, "estimator", None)
            if est is not None and hasattr(est, "coef_"):
                coefs.append(np.asarray(est.coef_)[0])
        if coefs:
            return np.mean(coefs, axis=0)
    return None


def top_influential_terms(cleaned_text: str, vectorizer, model, top_n: int = TOP_TERMS_TO_SHOW) -> List[Tuple[str, float]]:
    """Return the terms present in this review with the largest |coefficient|.

    These are influential TF-IDF features for this specific model, not a
    verified causal explanation of the prediction.
    """
    coef = get_linear_coef(model)
    if coef is None:
        return []

    vec = vectorizer.transform([cleaned_text])
    if vec.nnz == 0:
        return []

    feat_names = vectorizer.get_feature_names_out()
    indices = vec.indices
    values = vec.data
    contributions = [(feat_names[i], float(coef[i]) * float(v)) for i, v in zip(indices, values)]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n]


def classify_review(review: str, model, vectorizer) -> Tuple[str, float, str]:
    """Classify a single review.

    Returns:
        (label, confidence, cleaned_text)
    """
    cleaned = clean_text(review)
    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    label = "Fake" if pred == 1 else "Genuine"
    return label, float(proba[pred]), cleaned


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_sidebar(meta: dict) -> None:
    with st.sidebar:
        st.header("About")
        st.markdown(
            "This app distinguishes **Computer-Generated (CG)** reviews from "
            "**Original (OR)** reviews, using the labeling scheme of the "
            "training dataset."
        )
        st.warning(
            "⚠️ This does **not** claim to detect every type of fake review "
            "(e.g. spam/promotional text). See the Limitations note below the "
            "result.",
            icon="⚠️",
        )
        st.markdown("**Pipeline**")
        st.markdown(
            "- Text cleaning & lemmatization\n"
            "- TF-IDF (tuned via grid search)\n"
            "- Linear SVM (best of 5 models compared)"
        )
        st.markdown("**Model metadata**")
        st.markdown(f"- Model: `{meta['model_name']}`")
        if meta["test_accuracy"] is not None:
            st.markdown(f"- Test accuracy: `{meta['test_accuracy']:.1%}`")
        if meta["test_f1"] is not None:
            st.markdown(f"- Test F1: `{meta['test_f1']:.3f}`")
        st.markdown("**Dataset**")
        st.markdown(
            "40k+ Amazon product reviews, balanced between genuine "
            "verified-purchase reviews (OR) and GPT-2-generated reviews (CG)."
        )
        st.divider()
        st.caption("Built with Streamlit · scikit-learn · NLTK")


def render_result(label: str, confidence: float, influential: List[Tuple[str, float]]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if label == "Fake":
        st.error(f"🚨 **Prediction: Fake (Computer-Generated)**  \nConfidence: {confidence:.1%}")
    else:
        st.success(f"✅ **Prediction: Genuine**  \nConfidence: {confidence:.1%}")
    st.progress(confidence)
    st.caption(f"Predicted at {timestamp}")

    if influential:
        st.markdown("**Most influential TF-IDF terms found in this review** (not a verified explanation, just the highest-weighted features present):")
        cols = st.columns(2)
        for i, (term, weight) in enumerate(influential):
            direction = "→ Fake" if weight > 0 else "→ Genuine"
            cols[i % 2].markdown(f"- `{term}` ({direction}, weight={weight:+.3f})")
    else:
        st.caption("No influential TF-IDF terms found (review may be too short or contain only out-of-vocabulary words).")


def main() -> None:
    st.set_page_config(page_title="Fake Review Classifier", page_icon="📝", layout="centered")

    st.title("📝 Fake Review Classifier")
    st.markdown(
        "Enter a product review below. The model predicts whether it's "
        "**Fake (Computer-Generated)** or **Genuine**, based on patterns "
        "learned from a labeled dataset of real vs. GPT-2-generated Amazon reviews."
    )

    model, vectorizer, error = load_artifacts()
    if error:
        st.error(error)
        st.stop()

    meta = load_metadata()
    render_sidebar(meta)

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    example_choice = st.selectbox(
        "Try an example (optional):",
        ["-- none --", *EXAMPLE_REVIEWS.keys()],
    )
    if example_choice != "-- none --":
        st.session_state.review_text = EXAMPLE_REVIEWS[example_choice]

    review = st.text_area(
        "Enter a review:",
        value=st.session_state.review_text,
        height=150,
        max_chars=MAX_REVIEW_CHARS,
        key="review_input",
    )

    col1, col2 = st.columns([1, 1])
    classify_clicked = col1.button("Classify", type="primary", use_container_width=True)
    clear_clicked = col2.button("Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.review_text = ""
        st.rerun()

    if classify_clicked:
        stripped = review.strip()
        if not stripped:
            st.warning("Please enter some review text first.")
        elif len(stripped) < MIN_REVIEW_CHARS:
            st.warning(f"Review is too short to classify reliably (minimum {MIN_REVIEW_CHARS} characters).")
        elif not any(ch.isalpha() for ch in stripped):
            st.warning("Please enter text containing actual words — symbols/numbers alone can't be classified.")
        else:
            with st.spinner("Classifying review..."):
                label, confidence, cleaned = classify_review(stripped, model, vectorizer)
                influential = top_influential_terms(cleaned, vectorizer, model)
            render_result(label, confidence, influential)

    st.divider()
    st.markdown("**Limitations (read before trusting a result):**")
    st.caption(
        "This model was trained to distinguish GPT-2-generated review text (CG) "
        "from real verified-purchase reviews (OR) in one specific dataset. It is "
        "**not** a general spam/promotional-content detector — hand-written "
        "promotional or hype-style text was not part of the 'Fake' training "
        "examples, so such text may be misclassified as Genuine. Predictions on "
        "text very different in style from the training data should be treated "
        "with caution."
    )
    st.caption("Fake Review Classifier · For educational/demo purposes.")


if __name__ == "__main__":
    main()
