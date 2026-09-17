import json
import string
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------- Page setup ----------
st.set_page_config(page_title="Emotion Reader", page_icon="🌊", layout="centered")

MODELS_DIR = Path(__file__).parent / "models"

# ---------- Load model (cached so it only loads once per session) ----------
@st.cache_resource
def load_model():
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")
    model = joblib.load(MODELS_DIR / "logreg_model.pkl")
    with open(MODELS_DIR / "meta.json") as f:
        meta = json.load(f)
    return vectorizer, model, meta

vectorizer, model, meta = load_model()
STOPWORDS = set(meta["stopwords"])

EMOTION_COLORS = {
    "sadness": "#4C6FA5",
    "anger": "#B23A32",
    "love": "#C25B7C",
    "surprise": "#D89B2E",
    "fear": "#5B4B8A",
    "joy": "#3F8F45",
}

# ---------- Preprocessing (must match training exactly) ----------
def remove_punctuation(txt: str) -> str:
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_digits(txt: str) -> str:
    return "".join(c for c in txt if not c.isdigit())


def remove_non_ascii(txt: str) -> str:
    return "".join(c for c in txt if c.isascii())


def remove_stopwords(txt: str) -> str:
    # Plain whitespace split (punctuation/digits already stripped by this point),
    # so this avoids an nltk data download at deploy time.
    return " ".join(w for w in txt.split() if w not in STOPWORDS)


def preprocess(text: str) -> str:
    text = text.lower()
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_non_ascii(text)
    text = remove_stopwords(text)
    return text


def predict(text: str):
    cleaned = preprocess(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    classes = [meta["classes"][list(model.classes_).index(c)] for c in model.classes_]
    order = np.argsort(-probs)
    ranked = [(classes[i], probs[i]) for i in order]

    matched = vec.nnz  # number of non-zero tf-idf features that fired
    return ranked, matched


# ---------- UI ----------
st.markdown(
    "<h1 style='text-align:center; font-weight:600;'>What feeling is<br>hiding in this sentence?</h1>",
    unsafe_allow_html=True,
)
st.caption(
    f"TF-IDF (with word pairs) + class-balanced logistic regression · "
    f"trained on 16,000 labeled sentences · {model.classes_.shape[0]} emotions"
)

text = st.text_area(
    "Your sentence",
    placeholder="Write a sentence, a diary line, a text message...",
    label_visibility="collapsed",
    height=110,
)

analyze = st.button("Read it", type="primary")

if analyze:
    if not text.strip():
        st.warning("Type something first.")
    else:
        ranked, matched = predict(text)
        top_label, top_prob = ranked[0]
        runner_up_prob = ranked[1][1] if len(ranked) > 1 else 0
        margin = top_prob - runner_up_prob
        low_signal = matched == 0 or margin < 0.08

        color = EMOTION_COLORS.get(top_label, "#333333")

        st.markdown("<br>", unsafe_allow_html=True)
        if low_signal:
            st.markdown(
                f"<p style='text-align:center; color:#888;'>no strong signal — closest guess is</p>"
                f"<h2 style='text-align:center; color:{color}; font-style:italic;'>{top_label}</h2>"
                f"<p style='text-align:center; color:#888;'>the six emotions scored close together</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<p style='text-align:center; color:#888;'>reads mostly as</p>"
                f"<h2 style='text-align:center; color:{color}; font-style:italic;'>{top_label}</h2>"
                f"<p style='text-align:center; color:#888;'>{top_prob*100:.0f}% confidence</p>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars, one per emotion, ranked highest first
        for label, prob in ranked:
            c = EMOTION_COLORS.get(label, "#333333")
            pct = prob * 100
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:6px;">
                  <div style="width:78px; font-size:14px; color:#666;">{label}</div>
                  <div style="flex:1; background:#eee; border-radius:999px; height:10px; overflow:hidden;">
                    <div style="width:{pct:.1f}%; background:{c}; height:100%; border-radius:999px;"></div>
                  </div>
                  <div style="width:44px; text-align:right; font-size:13px; color:#666;">{pct:.0f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("Six emotions: sadness, anger, love, surprise, fear, joy.")
