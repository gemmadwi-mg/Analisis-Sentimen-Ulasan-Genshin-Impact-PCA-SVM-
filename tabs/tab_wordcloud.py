"""
tabs/tab_wordcloud.py
Tab: Wordcloud
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_text, tokenize_remove_stopwords, lemmatize_tokens, load_nltk, get_stop_words


def render(df=None, **kwargs):
    """Render konten tab ini.
    Render konten tab ini. Dipanggil dari app.py."""
    import matplotlib.pyplot as plt

    st.subheader("☁️ Word Cloud")

    load_nltk(); sw = get_stop_words()
    df['clean_wc']  = df['content'].apply(preprocess_text)
    df['tokens_wc'] = df['clean_wc'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens_wc'] = df['tokens_wc'].apply(lemmatize_tokens)
    df['final_wc']  = df['tokens_wc'].apply(lambda t: ' '.join(t))

    wc_t1, wc_t2, wc_t3 = st.tabs(["😊 Positif","😞 Negatif","📋 Semua"])

    def show_wc(texts, title, cmap):
        try:
            from wordcloud import WordCloud
            text = ' '.join(texts.dropna().tolist())
            if not text.strip():
                st.warning("Tidak ada teks."); return
            wc = WordCloud(width=700, height=300, background_color='white',
                           colormap=cmap, max_words=120,
                           collocations=False).generate(text)
            fig, ax = plt.subplots(figsize=(10,4))
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            st.pyplot(fig, use_container_width=True); plt.close()
        except ImportError:
            st.warning("Install wordcloud: `pip install wordcloud`")

    def show_top_words(texts, title, color, n=15):
        from collections import Counter
        words = ' '.join(texts.dropna().tolist()).split()
        freq  = Counter(words).most_common(n)
        wdf   = pd.DataFrame(freq, columns=['Kata','Frekuensi'])
        fig, ax = plt.subplots(figsize=(8,5))
        bars = ax.barh(wdf['Kata'][::-1], wdf['Frekuensi'][::-1],
                       color=color, alpha=0.8, edgecolor='white')
        for bar, v in zip(bars, wdf['Frekuensi'][::-1]):
            ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                    str(v), va='center', ha='left', fontsize=9)
        ax.set_title(f'Top {n} Kata — {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frekuensi'); ax.grid(axis='x', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with wc_t1:
        pos = df[df['sentiment']=='POSITIVE']['final_wc']
        show_wc(pos, "Word Cloud — Positif", 'Greens')
        show_top_words(pos, "Positif", '#28a745')

    with wc_t2:
        neg = df[df['sentiment']=='NEGATIVE']['final_wc']
        show_wc(neg, "Word Cloud — Negatif", 'Reds')
        show_top_words(neg, "Negatif", '#dc3545')

    with wc_t3:
        show_wc(df['final_wc'], "Word Cloud — Semua", 'viridis')
        show_top_words(df['final_wc'], "Semua", '#6f42c1')

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 8 — PCA VS TANPA PCA  ← MENGGUNAKAN HASIL TRAINING NYATA
    # ══════════════════════════════════════════════════════════════════════════════
