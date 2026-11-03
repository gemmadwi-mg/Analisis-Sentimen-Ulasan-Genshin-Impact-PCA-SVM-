"""
utils/data_loader.py
Fungsi untuk memuat dan memvalidasi dataset CSV.
"""

import pandas as pd
import streamlit as st


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame | None:
    """
    Muat CSV dari file upload Streamlit.
    Wajib memiliki kolom 'content' dan 'sentiment'.
    """
    try:
        df = pd.read_csv(uploaded_file)
        missing = [c for c in ['content', 'sentiment'] if c not in df.columns]
        if missing:
            st.error(f"Kolom tidak ditemukan: {missing}. "
                     f"Pastikan CSV memiliki kolom 'content' dan 'sentiment'.")
            return None
        df = df.dropna(subset=['content', 'sentiment'])
        df['content']   = df['content'].astype(str)
        df['sentiment'] = df['sentiment'].str.upper().str.strip()
        return df
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None


def load_default(path: str = "genshin.csv") -> pd.DataFrame | None:
    """
    Muat CSV dari path lokal (file default).
    Digunakan jika pengguna belum mengupload file.
    """
    try:
        df = pd.read_csv(path)
        df = df.dropna(subset=['content', 'sentiment'])
        df['content']   = df['content'].astype(str)
        df['sentiment'] = df['sentiment'].str.upper().str.strip()
        return df
    except Exception:
        return None
