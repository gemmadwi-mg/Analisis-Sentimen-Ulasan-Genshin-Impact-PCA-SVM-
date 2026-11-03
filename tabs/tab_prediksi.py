"""
tabs/tab_prediksi.py
Tab: Prediksi
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
    st.subheader("🔍 Prediksi Sentimen")

    if 'r_pca' not in st.session_state:
        st.info("Jalankan pipeline di sidebar terlebih dahulu.", icon="👈")
    else:
        r_pca   = st.session_state['r_pca']
        r_nopca = st.session_state['r_nopca']
        feat    = st.session_state['feat']

        mode_pred = st.radio(
            "Gunakan model:", ["Dengan PCA","Tanpa PCA"], horizontal=True
        )
        r_use = r_pca if mode_pred == "Dengan PCA" else r_nopca

        st.divider()
        c1,c2 = st.columns(2)
        ex_pos = "This game is absolutely amazing! The open world and story are breathtaking. Love every moment!"
        ex_neg = "Totally disappointed. Gacha is unfair and pay-to-win. Game crashes all the time. Waste of money!"

        if c1.button("😊 Contoh Positif"): st.session_state['pred_input'] = ex_pos
        if c2.button("😞 Contoh Negatif"): st.session_state['pred_input'] = ex_neg

        input_text = st.text_area(
            "Teks ulasan:",
            value=st.session_state.get('pred_input',''),
            height=100,
            placeholder="Ketik atau paste teks ulasan game di sini..."
        )

        if st.button("🔮 Analisis Sentimen", type="primary"):
            if not input_text.strip():
                st.warning("Masukkan teks terlebih dahulu.")
            else:
                with st.spinner("Menganalisis..."):
                    load_nltk(); sw = get_stop_words()
                    clean  = preprocess_text(input_text)
                    tokens = tokenize_remove_stopwords(clean, sw)
                    tokens = lemmatize_tokens(tokens)
                    final  = ' '.join(tokens)

                    try:
                        X_new = feat['tfidf'].transform([final])
                        X_new = feat['chi2_sel'].transform(X_new)

                        if mode_pred == "Dengan PCA":
                            X_new = r_pca['scaler'].transform(X_new.toarray())
                            X_new = r_pca['pca'].transform(X_new)

                        pred_label = r_use['svm'].predict(X_new)[0]
                        pred_proba = r_use['svm'].predict_proba(X_new)[0]
                        pred_class = feat['le'].inverse_transform([pred_label])[0]

                        st.divider()
                        res_col, conf_col = st.columns([1,2])
                        with res_col:
                            if pred_class == 'POSITIVE':
                                st.success(f"😊 **POSITIF**\nConfidence: {max(pred_proba)*100:.1f}%")
                            else:
                                st.error(f"😞 **NEGATIF**\nConfidence: {max(pred_proba)*100:.1f}%")
                            st.caption(f"Model: {st.session_state['kernel'].upper()} | {mode_pred}")

                        with conf_col:
                            st.markdown("**Probabilitas per Kelas:**")
                            for cls, prob in zip(feat['classes'], pred_proba):
                                st.markdown(f"**{cls}**")
                                st.progress(float(prob), text=f"{prob*100:.2f}%")

                        st.divider()
                        ca, cb = st.columns(2)
                        with ca:
                            st.markdown("**Teks bersih:**"); st.code(final)
                        with cb:
                            st.markdown("**Token:**"); st.code(str(tokens))

                        if 'history' not in st.session_state:
                            st.session_state['history'] = []
                        st.session_state['history'].insert(0, {
                            'Teks':       input_text[:80]+('...' if len(input_text)>80 else ''),
                            'Sentimen':   pred_class,
                            'Confidence': f"{max(pred_proba)*100:.1f}%",
                            'Model':      f"{st.session_state['kernel'].upper()} | {mode_pred}",
                        })
                    except Exception as e:
                        st.error(f"Error prediksi: {e}")

        if st.session_state.get('history'):
            st.divider()
            st.markdown("**📋 Riwayat Prediksi**")
            st.dataframe(
                pd.DataFrame(st.session_state['history'][:10]),
                use_container_width=True, hide_index=True
            )
            if st.button("🗑️ Hapus Riwayat"):
                st.session_state['history'] = []
                st.rerun()

    # ── Footer ─────────────────────────────────────────────────────────────────────
    st.divider()
    st.caption("Dashboard Analisis Sentimen Genshin Impact | SVM + TF-IDF + Chi-Square + SMOTE + PCA")
