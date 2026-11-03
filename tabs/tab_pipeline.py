"""
tabs/tab_pipeline.py
Tab: Pipeline
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_text, tokenize_remove_stopwords, lemmatize_tokens, load_nltk, get_stop_words


def render(df=None, **kwargs):
    """Render konten tab ini."""
    """Render konten tab ini. Dipanggil dari app.py."""
    st.subheader("🔧 Pipeline Preprocessing NLP")

    steps = [
        ("1️⃣ Case Folding",      "Mengubah seluruh teks menjadi huruf kecil (lowercase)"),
        ("2️⃣ Normalisasi",       "Ekspansi singkatan & slang (tbh → to be honest, dll.)"),
        ("3️⃣ Cleaning",          "Hapus URL, angka, emoji, tanda baca, karakter khusus"),
        ("4️⃣ Tokenisasi",        "Memecah kalimat menjadi token kata individual"),
        ("5️⃣ Stopword Removal",  "Hapus kata umum tidak bermakna (the, is, at, …)"),
        ("6️⃣ Lemmatisasi",       "Kembalikan kata ke bentuk dasar (playing→play)"),
    ]
    cols = st.columns(3)
    for i,(title,desc) in enumerate(steps):
        with cols[i%3]:
            st.info(f"**{title}**\n\n{desc}")

    st.divider()
    st.markdown("**🔬 Demo Preprocessing Interaktif**")
    sample = st.text_area(
        "Masukkan teks untuk diproses:",
        value="Tbh I love the open world! Nice story but rewards could be better. Don't waste ur time grinding!",
        height=80,
    )
    if st.button("▶️ Proses Teks"):
        load_nltk(); sw = get_stop_words()
        st.markdown("---")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Teks asli:**");         st.code(sample)
            st.markdown("**Setelah case folding:**"); st.code(sample.lower())
            cleaned = preprocess_text(sample)
            st.markdown("**Setelah normalisasi + cleaning:**"); st.code(cleaned)
        with cb:
            tok = tokenize_remove_stopwords(cleaned, sw)
            st.markdown("**Setelah tokenisasi + stopword:**"); st.code(str(tok))
            lem = lemmatize_tokens(tok)
            st.markdown("**Setelah lemmatisasi:**"); st.code(str(lem))
            st.markdown("**Teks akhir:**"); st.success(' '.join(lem))

    st.divider()
    st.markdown("**Parameter Vectorization & Seleksi Fitur**")
    st.table(pd.DataFrame({
        'Tahap':  ['TF-IDF','Chi-Square','SMOTE','StandardScaler','PCA'],
        'Detail': ['max_features=5000','SelectKBest k=1000','sampling_strategy=auto',
                   'zero mean, unit variance','n_components=0.95'],
        'Output': ['5.000 fitur','1.000 fitur terpilih','Data seimbang',
                   'Fitur ternormalisasi','~300 komponen'],
    }))


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 3 — TF-IDF, CHI-SQUARE & PCA
    # ══════════════════════════════════════════════════════════════════════════════
