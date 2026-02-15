import streamlit as st
import pandas as pd
import tensorflow as tf
import zipfile
import re
import json
from tensorflow import keras
from transformers import RobertaTokenizerFast, TFRobertaModel

# load slang dict
@st.cache_resource
def load_slang_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

slang_dict = load_slang_dict("slang_dict.json")

# preprocess pipeline
def clean_text(text):
    text = text.lower()
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\t|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return text.strip()

def normalize_slang(text, slang_dict):
    tokens = text.split()
    normalized = [slang_dict.get(t, t) for t in tokens]
    return " ".join(normalized)

def preprocess_text(text):
    text = clean_text(text)
    text = normalize_slang(text, slang_dict)
    return text

st.set_page_config(
    page_title="ABSA Gojek Review",
    layout="wide"
)

st.markdown("""
<style>
html, body { font-family: 'Inter', sans-serif; }
.hero {
    background: linear-gradient(135deg, #00AA13, #00C853);
    padding: 40px;
    border-radius: 20px;
    color: white;
    margin-bottom: 30px;
}
.card {
    background: var(--background-color);
    padding: 28px;
    border-radius: 18px;
    border: 1px solid var(--secondary-background-color);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    margin-top: 25px;
}
.badge-positive {
    background: rgba(0, 200, 83, 0.15);
    color: #00C853;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
}
.badge-negative {
    background: rgba(244, 67, 54, 0.15);
    color: #F44336;
    padding: 6px 16px;
    border-radius: 999px;
    font-weight: 700;
}
.stButton > button {
    background: linear-gradient(135deg, #00AA13, #00C853);
    color: white;
    border-radius: 14px;
    font-weight: 700;
    height: 48px;
    border: none;
}
.footer {
    text-align: center;
    color: #999;
    margin-top: 60px;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)

sentiment_map = {0: "Negatif", 1: "Positif"}

aspect_map = {
    0: "Kenaikan Biaya & Penurunan Manfaat Promo GoFood",
    1: "Masalah Driver & Layanan (Ketersediaan, Profesionalitas, Keselamatan)",
    2: "Masalah Akun, GoPaylater & Transaksi",
    3: "Pujian & Kepuasan Pengguna",
    4: "Masalah Performa & Stabilitas Aplikasi"
}

# sidebar
with st.sidebar:
    st.markdown("## ABSA Gojek Dashboard")
    debug_mode = st.checkbox("Debug: Tampilkan Preprocessing")
    st.write("""
    *Fitur Utama*
    - Aspect-Based Sentiment Analysis
    - Single Prediction
    - Batch Prediction
    """)
    st.divider()
    st.caption("""
    Skripsi  
    **Wayan Farel Nickholas Sadewa | 2208561051**
    """)

# load model
@st.cache_resource
def load_tokenizer(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    return RobertaTokenizerFast.from_pretrained(extract_path)

@st.cache_resource
def load_models():
    aspect_model = keras.models.load_model(
        "roberta_aspect/roberta_aspect_model.h5",
        custom_objects={"TFRobertaModel": TFRobertaModel},
        compile=False
    )
    sentiment_model = keras.models.load_model(
        "roberta_sentiment/roberta_sentiment_model.h5",
        custom_objects={"TFRobertaModel": TFRobertaModel},
        compile=False
    )
    return aspect_model, sentiment_model


tokenizer_aspect = load_tokenizer(
    "roberta_aspect/roberta_tokenizer_aspect.zip",
    "roberta_aspect/roberta_tokenizer_aspect"
)

tokenizer_sentiment = load_tokenizer(
    "roberta_sentiment/roberta_tokenizer_sentiment.zip",
    "roberta_sentiment/roberta_tokenizer_sentiment"
)

aspect_model, sentiment_model = load_models()

# predict function
def predict(text):
    preprocessed_text = preprocess_text(text)

    encoded_aspect = tokenizer_aspect(
        preprocessed_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

    encoded_sentiment = tokenizer_sentiment(
        preprocessed_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="tf"
    )

    aspect_pred = aspect_model(
        {
            "input_1": encoded_aspect["input_ids"],
            "input_2": encoded_aspect["attention_mask"]
        },
        training=False
    )

    sentiment_pred = sentiment_model(
        {
            "input_1": encoded_sentiment["input_ids"],
            "input_2": encoded_sentiment["attention_mask"]
        },
        training=False
    ).numpy()

    aspect_idx = int(tf.argmax(aspect_pred, axis=1)[0])
    sentiment_idx = int(tf.argmax(sentiment_pred, axis=1)[0])

    return (
        preprocessed_text,
        aspect_map[aspect_idx],
        sentiment_map[sentiment_idx],
        float(sentiment_pred[0][sentiment_idx])
    )

st.markdown("""
<div class="hero">
    <h1>Aspect-Based Sentiment Analysis</h1>
    <p>Analisis ulasan aplikasi <b>Gojek</b> menggunakan <b>RoBERTa</b></p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Review", "Batch CSV"])

# single predict
with tab1:
    user_input = st.text_area("Masukkan review pengguna:")

    if st.button("Prediksi", use_container_width=True):
        if not user_input or user_input.strip() == "":
            st.warning("⚠️ Input tidak valid. Silakan masukkan review terlebih dahulu.")
            st.stop()

        pre, aspect, sentiment, conf = predict(user_input)

        if debug_mode:
            st.markdown("### Debug: Hasil Preprocessing")
            st.code(pre)

        badge = "badge-positive" if sentiment == "Positif" else "badge-negative"

        st.markdown(f"""
        <div class="card">
            <p><b>Aspect</b><br>{aspect}</p>
            <p><b>Sentiment</b><br><span class="{badge}">{sentiment}</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(conf)
        st.caption(f"Confidence: {conf:.2%}")

# batch predict
with tab2:
    uploaded_file = st.file_uploader("Upload CSV (kolom: review)", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception:
            st.error("File CSV tidak dapat dibaca. Pastikan format file benar.")
            st.stop()

        if "review" not in df.columns:
            st.error("Format data tidak sesuai. Kolom wajib: review")
            st.stop()

        if st.button("Proses Batch", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results_list = []
            total = len(df)

            for i, review in enumerate(df["review"].astype(str)):
                result = predict(review)
                results_list.append(result)

                progress = int((i + 1) / total * 100)
                progress_bar.progress(progress)
                status_text.text(f"Memproses data {i + 1} dari {total}")

            results = pd.DataFrame(
                results_list,
                columns=["preprocessed_review", "aspect", "sentiment", "confidence"]
            )

            results["confidence"] = (results["confidence"] * 100).round(2)
            df = pd.concat([df, results], axis=1)

            progress_bar.empty()
            status_text.empty()

            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "hasil_absa_gojek.csv",
                "text/csv",
                use_container_width=True
            )

#footer
st.markdown("""
<div class="footer">
© 2026 – ABSA Gojek | Wayan Farel Nickholas Sadewea
</div>
""", unsafe_allow_html=True)