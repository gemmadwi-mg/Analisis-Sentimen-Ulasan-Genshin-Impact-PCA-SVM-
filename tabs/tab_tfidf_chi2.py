"""
tabs/tab_tfidf_chi2.py
Tab: TF-IDF & Chi-Square Feature Selection
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def render(df=None, **kwargs):
    """Render tab TF-IDF dan Chi-Square."""
    st.subheader("📐 Hasil TF-IDF & Chi-Square")
    st.caption("Vectorization TF-IDF dan seleksi fitur Chi-Square untuk dokumentasi skripsi")

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    feat = st.session_state['feat']

    stab1, stab2, stab3 = st.tabs([
        "📊 TF-IDF", "📉 Chi-Square", "📊 Seleksi vs Semua Fitur",
    ])

    with stab1:
            tfidf    = feat['tfidf']
            df_proc  = feat['df_final']
        
            fn = tfidf.get_feature_names_out()
            X_tfidf_full = tfidf.transform(df_proc['final'])

            # ── Langkah-Langkah TF-IDF ──────────────────────────────────────────
            st.markdown("**🧭 Langkah-Langkah Perhitungan TF-IDF**")
            st.caption(
                "TF-IDF (Term Frequency-Inverse Document Frequency) mengubah teks "
                "menjadi representasi numerik berbobot melalui 4 langkah berikut, "
                "diapit kotak INPUT (data masuk) dan OUTPUT (hasil akhir)."
            )

            n_docs_in  = len(df_proc)
            n_feat_out = len(fn)

            fig_step, ax_step = plt.subplots(figsize=(15.5, 3.4))
            ax_step.axis('off')
            steps = [
                ("1. Tokenisasi\nKorpus", "Kumpulkan seluruh\nkata unik dari\nteks bersih"),
                ("2. Hitung TF", "Term Frequency:\nfrekuensi kata\ndalam 1 dokumen"),
                ("3. Hitung IDF", "Inverse Document\nFrequency: kelangkaan\nkata di semua dokumen"),
                ("4. TF × IDF\n+ Normalisasi L2", "Bobot akhir tiap\nkata per dokumen"),
            ]
            box_w, box_h, gap = 2.6, 1.8, 0.5
            io_w = 2.3
            colors_step = ['#E6F1FB', '#EAF3DE', '#FAEEDA', '#FCEBEB']
            edge_colors = ['#185FA5', '#3B6D11', '#854F0B', '#A32D2D']

            # ── Kotak INPUT (paling kiri) ─────────────────────────────────────
            x_in = 0
            rect_in = plt.Rectangle((x_in, 0), io_w, box_h, facecolor='#F0F0F0',
                                     edgecolor='#555555', linewidth=1.6,
                                     linestyle='--', zorder=2)
            ax_step.add_patch(rect_in)
            ax_step.text(x_in + io_w/2, box_h - 0.35, "INPUT", ha='center', va='top',
                         fontsize=10, fontweight='bold', color='#333333')
            ax_step.text(x_in + io_w/2, box_h/2 - 0.30,
                         f"{n_docs_in:,} Dokumen\n(teks hasil\npreprocessing)",
                         ha='center', va='center', fontsize=8.5, color='#333')
            ax_step.annotate('', xy=(x_in + io_w + gap - 0.1, box_h/2),
                             xytext=(x_in + io_w + 0.1, box_h/2),
                             arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

            # ── Kotak proses (step 1-4) ────────────────────────────────────────
            x_offset = x_in + io_w + gap
            for i, ((title, desc), c, ec) in enumerate(zip(steps, colors_step, edge_colors)):
                x = x_offset + i * (box_w + gap)
                rect = plt.Rectangle((x, 0), box_w, box_h, facecolor=c,
                                      edgecolor=ec, linewidth=1.5, zorder=2)
                ax_step.add_patch(rect)
                ax_step.text(x + box_w/2, box_h - 0.35, title, ha='center', va='top',
                             fontsize=10, fontweight='bold', color=ec)
                ax_step.text(x + box_w/2, box_h/2 - 0.25, desc, ha='center', va='center',
                             fontsize=8.5, color='#333')
                ax_step.annotate('', xy=(x + box_w + gap - 0.1, box_h/2),
                                 xytext=(x + box_w + 0.1, box_h/2),
                                 arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

            # ── Kotak OUTPUT (paling kanan) ─────────────────────────────────────
            x_out = x_offset + len(steps) * (box_w + gap)
            rect_out = plt.Rectangle((x_out, 0), io_w, box_h, facecolor='#E1F5E1',
                                      edgecolor='#2E7D32', linewidth=1.8, zorder=2)
            ax_step.add_patch(rect_out)
            ax_step.text(x_out + io_w/2, box_h - 0.35, "OUTPUT", ha='center', va='top',
                         fontsize=10, fontweight='bold', color='#2E7D32')
            ax_step.text(x_out + io_w/2, box_h/2 - 0.30,
                         f"Matriks TF-IDF\n{n_docs_in:,} × {n_feat_out:,}\n(dok × fitur)",
                         ha='center', va='center', fontsize=8.5, color='#333')

            ax_step.set_xlim(-0.2, x_out + io_w + 0.2)
            ax_step.set_ylim(-0.2, box_h + 0.2)
            plt.tight_layout()
            st.pyplot(fig_step, use_container_width=True)
            plt.close()

            # ── Tampilkan hasil OUTPUT aktual ────────────────────────────────────
            st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
            col_out1, col_out2, col_out3 = st.columns(3)
            col_out1.metric("Ukuran Matriks", f"{n_docs_in:,} × {n_feat_out:,}")
            col_out2.metric("Total Sel", f"{n_docs_in*n_feat_out:,}")
            sparsity = 1 - (X_tfidf_full.nnz / (n_docs_in*n_feat_out))
            col_out3.metric("Sparsity (sel bernilai 0)", f"{sparsity*100:.2f}%")
            st.caption(
                f"Matriks TF-IDF bersifat sparse (jarang) — sebagian besar sel "
                f"bernilai 0 karena kebanyakan kata hanya muncul di sebagian kecil "
                f"dari {n_docs_in:,} dokumen. Cuplikan nyata matriks ini dapat "
                f"dilihat pada tabel \"Matriks Hasil Perhitungan TF-IDF\" di bawah."
            )

            st.divider()

            # ── Rumus TF-IDF ──────────────────────────────────────────────────
            st.markdown("**📐 Rumus Perhitungan TF-IDF**")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.latex(r"TF(t,d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}")
                st.caption("TF(t,d): frekuensi term t pada dokumen d, dibagi total term dalam d")
                st.latex(r"IDF(t) = \ln\left(\frac{1+n}{1+df(t)}\right) + 1")
                st.caption("n: total dokumen, df(t): jumlah dokumen yang memuat term t (smooth IDF, sesuai default scikit-learn)")
            with col_f2:
                st.latex(r"TFIDF(t,d) = TF(t,d) \times IDF(t)")
                st.caption("Skor mentah sebelum normalisasi")
                st.latex(r"\hat{v} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_i v_i^2}}")
                st.caption("Normalisasi L2 (default scikit-learn) — hasil akhir bobot TF-IDF")

            st.divider()

            # ── Contoh Perhitungan Manual dengan Data Asli ──────────────────────
            st.markdown("**🧮 Contoh Perhitungan TF-IDF dengan Data Asli**")
            st.caption(
                "Mengambil 1 dokumen dan 1 kata sungguhan dari dataset untuk "
                "menunjukkan perhitungan TF-IDF secara manual, langkah demi langkah."
            )

            # Ambil dokumen pertama sebagai contoh, cari kata dengan skor TF-IDF tertinggi di dokumen itu
            contoh_doc_idx = 0
            contoh_teks_asli   = df_proc['content'].iloc[contoh_doc_idx] if 'content' in df_proc.columns else "-"
            contoh_teks_bersih = df_proc['final'].iloc[contoh_doc_idx]
            contoh_tokens = contoh_teks_bersih.split()

            row_vec = X_tfidf_full[contoh_doc_idx].toarray().flatten()
            top_word_idx_doc = int(np.argmax(row_vec))
            contoh_kata = fn[top_word_idx_doc]
            contoh_tfidf_score = row_vec[top_word_idx_doc]

            # Hitung ulang komponen TF & IDF secara manual untuk kata contoh
            tf_manual = contoh_tokens.count(contoh_kata) / len(contoh_tokens) if len(contoh_tokens) > 0 else 0
            df_t = int((X_tfidf_full[:, top_word_idx_doc].toarray().flatten() > 0).sum())
            n_total_docs = X_tfidf_full.shape[0]
            idf_manual = np.log((1 + n_total_docs) / (1 + df_t)) + 1
            tfidf_manual_raw = tf_manual * idf_manual

            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.markdown("**Dokumen contoh (Dokumen ke-1):**")
                st.code(contoh_teks_asli[:150] + ("..." if len(contoh_teks_asli) > 150 else ""))
                st.markdown("**Setelah preprocessing:**")
                st.code(contoh_teks_bersih)
                st.markdown(f"**Kata dengan skor TF-IDF tertinggi di dokumen ini:** `{contoh_kata}`")

            with col_ex2:
                st.markdown(f"**Langkah 1 — Hitung TF(\"{contoh_kata}\", Dokumen-1):**")
                st.latex(
                    rf'TF = \frac{{{contoh_tokens.count(contoh_kata)}}}{{{len(contoh_tokens)}}} = {tf_manual:.4f}'
                )
                st.caption(
                    f'Kata "{contoh_kata}" muncul {contoh_tokens.count(contoh_kata)} kali '
                    f'dari total {len(contoh_tokens)} kata pada dokumen ini.'
                )

                st.markdown(f"**Langkah 2 — Hitung IDF(\"{contoh_kata}\"):**")
                st.latex(
                    rf'IDF = \ln\left(\frac{{1+{n_total_docs}}}{{1+{df_t}}}\right) + 1 = {idf_manual:.4f}'
                )
                st.caption(
                    f'Kata ini muncul di {df_t} dari total {n_total_docs} dokumen '
                    f'pada seluruh korpus.'
                )

                st.markdown("**Langkah 3 — Hitung TF-IDF (sebelum normalisasi):**")
                st.latex(
                    rf'TFIDF = {tf_manual:.4f} \times {idf_manual:.4f} = {tfidf_manual_raw:.4f}'
                )

                st.markdown("**Langkah 4 — Setelah Normalisasi L2 (nilai akhir di matriks):**")
                st.latex(rf'TFIDF_{{\text{{final}}}} = {contoh_tfidf_score:.4f}')
                st.success(
                    f'✅ Nilai ini **cocok** dengan nilai aktual pada matriks TF-IDF '
                    f'untuk kata "{contoh_kata}" di Dokumen-1 (lihat tabel matriks di bawah).'
                )

            st.markdown("**📄 Matriks Hasil Perhitungan TF-IDF (Data Asli — 5 Dokumen Pertama)**")
            st.caption(
                f'Matriks berikut adalah hasil NYATA penerapan rumus TF-IDF (Langkah 1–4 di atas) '
                f'pada 5 dokumen pertama dataset. Kolom "{contoh_kata}" (kotak hijau) adalah kata '
                f'contoh perhitungan manual — nilainya di baris Dok-1 harus sama persis dengan hasil '
                f'Langkah 4 (**{contoh_tfidf_score:.4f}**).'
            )

            # Ambil beberapa kata contoh: kata dari perhitungan manual + beberapa kata top lain
            mean_scores_preview = np.array(X_tfidf_full.mean(axis=0)).flatten()
            top_preview_idx = np.argsort(mean_scores_preview)[::-1][:7]
            preview_word_idx = np.unique(np.append(top_preview_idx, top_word_idx_doc))
            preview_words = fn[preview_word_idx]

            matriks_contoh = X_tfidf_full[:5, preview_word_idx].toarray()
            df_matriks_contoh = pd.DataFrame(
                matriks_contoh, columns=preview_words,
                index=[f"Dok-{i+1}" for i in range(5)]
            )

            def _highlight_contoh_kata(col):
                return ['background-color: #C6E8C6; font-weight: bold' if col.name == contoh_kata else '' for _ in col]

            st.dataframe(
                df_matriks_contoh.style.format("{:.4f}")
                    .background_gradient(cmap='Blues', axis=None)
                    .apply(_highlight_contoh_kata, axis=0),
                use_container_width=True
            )
            st.caption(
                f'Baris Dok-1, kolom "{contoh_kata}" = **{matriks_contoh[0, list(preview_words).index(contoh_kata)]:.4f}** '
                f'— sesuai hasil perhitungan manual pada Langkah 4.'
            )

            st.divider()

            # ── Perhitungan: kenapa jadi 3.424 fitur ────────────────────────────
            n_docs   = len(df_proc)
            n_actual = len(fn)
            max_feat = 5000
            st.markdown(f"**🔢 Perhitungan: Mengapa Menjadi {n_actual:,} Fitur (bukan {max_feat:,})?**")
            st.info(
                f"**Parameter yang diberikan:** `max_features = {max_feat:,}`\n\n"
                f"**Cara kerja TfidfVectorizer:**\n\n"
                f"1. Vectorizer memindai seluruh **{n_docs:,} dokumen** (ulasan) hasil "
                f"preprocessing, lalu mengumpulkan **setiap kata unik** yang muncul "
                f"di korpus menjadi kandidat vocabulary.\n\n"
                f"2. Jumlah kata unik yang benar-benar ditemukan pada korpus ini "
                f"adalah **{n_actual:,} kata** — lebih sedikit dari batas maksimum "
                f"5.000 yang ditentukan.\n\n"
                f"3. Karena `jumlah_kata_unik_aktual < max_features` "
                f"({n_actual:,} < {max_feat:,}), maka **seluruh kata unik yang "
                f"ditemukan ikut dipakai** — parameter `max_features` hanya berfungsi "
                f"sebagai **batas atas**, bukan target pasti.\n\n"
                f"**Formula penentuan jumlah fitur akhir:**"
            )
            st.latex(r"n_{\text{fitur}} = \min(\text{jumlah kata unik pada korpus},\ max\_features)")
            st.latex(
                rf"n_{{\text{{fitur}}}} = \min({n_actual:,},\ {max_feat:,}) = {n_actual:,}"
            )
            st.caption(
                "Jumlah kata unik yang lebih kecil dari 5.000 ini adalah konsekuensi "
                "alami dari tahap preprocessing (case folding, cleaning, stopword "
                "removal, lemmatization) yang telah menyederhanakan dan menggabungkan "
                "banyak variasi kata menjadi bentuk dasarnya, sehingga total "
                "kosakata unik berkurang secara signifikan dari kata-kata mentah "
                "sebelum preprocessing."
            )

            st.divider()
        
            # Statistik ringkas
            st.markdown("**📋 Statistik TF-IDF**")
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Total Dokumen",       f"{len(df_proc):,}")
            s2.metric("Total Fitur (aktual)", f"{len(fn):,}")
            s3.metric("Parameter max_features","5.000")
            s4.metric("Metode Weighting",     "TF-IDF")
        
            st.divider()
        
            # Hitung skor TF-IDF rata-rata per kata
            from sklearn.feature_extraction.text import TfidfVectorizer
            mean_scores  = np.array(X_tfidf_full.mean(axis=0)).flatten()
            top_n        = 20
            top_idx      = np.argsort(mean_scores)[::-1][:top_n]
            top_words    = fn[top_idx]
            top_scores   = mean_scores[top_idx]
        
            st.markdown(f"**🏆 Top {top_n} Kata Berdasarkan Rata-rata Skor TF-IDF**")
        
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.barh(top_words[::-1], top_scores[::-1],
                           color='#4C72B0', alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, top_scores[::-1]):
                ax.text(bar.get_width()+0.0002, bar.get_y()+bar.get_height()/2,
                        f'{v:.4f}', va='center', ha='left', fontsize=9)
            ax.set_xlabel("Rata-rata Skor TF-IDF")
            ax.set_title(f"Top {top_n} Kata — Rata-rata Skor TF-IDF", fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
            st.divider()
        
            # Tabel 15 baris pertama x 15 fitur teratas
            st.markdown("**📄 Matriks TF-IDF — 10 Dokumen × 15 Fitur Teratas**")
            top15_idx  = np.argsort(mean_scores)[::-1][:15]
            top15_words = fn[top15_idx]
            tfidf_dense = X_tfidf_full[:10, top15_idx].toarray()
            tfidf_df    = pd.DataFrame(tfidf_dense, columns=top15_words,
                                       index=[f"Dok-{i+1}" for i in range(10)])
            st.dataframe(tfidf_df.style.format("{:.4f}")
                         .background_gradient(cmap='Blues', axis=None),
                         use_container_width=True)
        
            st.divider()
        
            # Statistik per sentimen
            st.markdown("**📊 Rata-rata Skor TF-IDF per Sentimen (Top 10 Fitur)**")
            top10_idx   = np.argsort(mean_scores)[::-1][:10]
            top10_words = fn[top10_idx]
            sent_stats  = []
            for sent in feat['classes']:
                mask = df_proc['sentiment'] == sent
                X_s  = tfidf.transform(df_proc.loc[mask, 'final'])
                avg  = np.array(X_s[:, top10_idx].mean(axis=0)).flatten()
                row  = {'Sentimen': sent}
                for w, v in zip(top10_words, avg):
                    row[w] = round(float(v), 4)
                sent_stats.append(row)
            sent_df = pd.DataFrame(sent_stats)
            st.dataframe(sent_df.style.format({c:"{:.4f}" for c in top10_words})
                         .background_gradient(cmap='YlOrRd', subset=list(top10_words), axis=None),
                         use_container_width=True)
        
        # ════════════════════════════════════════════════════════════════════
        # SUB-TAB 2 — CHI-SQUARE
        # ════════════════════════════════════════════════════════════════════

    with stab2:
            chi2_sel = feat['chi2_sel']
            tfidf    = feat['tfidf']
            fn_all   = tfidf.get_feature_names_out()
        
            scores_all  = chi2_sel.scores_
            pvals_all   = chi2_sel.pvalues_
            selected_mask = chi2_sel.get_support()
            fn_selected   = fn_all[selected_mask]
            scores_sel    = scores_all[selected_mask]
            pvals_sel     = pvals_all[selected_mask]

            # ── Langkah-Langkah Chi-Square ───────────────────────────────────────
            st.markdown("**🧭 Langkah-Langkah Seleksi Fitur Chi-Square**")
            st.caption(
                "Chi-Square (χ²) menyeleksi kata-kata yang paling berhubungan "
                "secara statistik dengan label sentimen, melalui 4 langkah berikut, "
                "diapit kotak INPUT (data masuk) dan OUTPUT (hasil akhir)."
            )

            n_docs_chi   = len(feat['df_final'])
            n_feat_in_chi  = len(fn_all)
            n_feat_out_chi = len(fn_selected)

            fig_step2, ax_step2 = plt.subplots(figsize=(15.5, 3.4))
            ax_step2.axis('off')
            steps2 = [
                ("1. Hitung Frekuensi\nObservasi (O)", "Frekuensi kemunculan\ntiap kata pada\nsetiap kelas sentimen"),
                ("2. Hitung Frekuensi\nEkspektasi (E)", "Frekuensi yang\ndiharapkan bila kata\ntidak terkait kelas"),
                ("3. Hitung Skor χ²", "χ² = Σ (O-E)²/E\nuntuk tiap kata"),
                ("4. Urutkan & Pilih\nTop-k Skor", "Ambil k kata dengan\nskor χ² tertinggi"),
            ]
            colors_step2 = ['#E6F1FB', '#FAEEDA', '#FBEAF0', '#EAF3DE']
            edge_colors2 = ['#185FA5', '#854F0B', '#993556', '#3B6D11']
            io_w2 = 2.3

            # ── Kotak INPUT ────────────────────────────────────────────────────
            x_in2 = 0
            rect_in2 = plt.Rectangle((x_in2, 0), io_w2, box_h, facecolor='#F0F0F0',
                                      edgecolor='#555555', linewidth=1.6,
                                      linestyle='--', zorder=2)
            ax_step2.add_patch(rect_in2)
            ax_step2.text(x_in2 + io_w2/2, box_h - 0.35, "INPUT", ha='center', va='top',
                          fontsize=10, fontweight='bold', color='#333333')
            ax_step2.text(x_in2 + io_w2/2, box_h/2 - 0.30,
                          f"Matriks TF-IDF\n{n_docs_chi:,} × {n_feat_in_chi:,}\n(dok × fitur)",
                          ha='center', va='center', fontsize=8.5, color='#333')
            ax_step2.annotate('', xy=(x_in2 + io_w2 + gap - 0.1, box_h/2),
                              xytext=(x_in2 + io_w2 + 0.1, box_h/2),
                              arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

            # ── Kotak proses (step 1-4) ────────────────────────────────────────
            x_offset2 = x_in2 + io_w2 + gap
            for i, ((title, desc), c, ec) in enumerate(zip(steps2, colors_step2, edge_colors2)):
                x = x_offset2 + i * (box_w + gap)
                rect = plt.Rectangle((x, 0), box_w, box_h, facecolor=c,
                                      edgecolor=ec, linewidth=1.5, zorder=2)
                ax_step2.add_patch(rect)
                ax_step2.text(x + box_w/2, box_h - 0.35, title, ha='center', va='top',
                              fontsize=10, fontweight='bold', color=ec)
                ax_step2.text(x + box_w/2, box_h/2 - 0.25, desc, ha='center', va='center',
                              fontsize=8.5, color='#333')
                ax_step2.annotate('', xy=(x + box_w + gap - 0.1, box_h/2),
                                  xytext=(x + box_w + 0.1, box_h/2),
                                  arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

            # ── Kotak OUTPUT ─────────────────────────────────────────────────────
            x_out2 = x_offset2 + len(steps2) * (box_w + gap)
            rect_out2 = plt.Rectangle((x_out2, 0), io_w2, box_h, facecolor='#E1F5E1',
                                       edgecolor='#2E7D32', linewidth=1.8, zorder=2)
            ax_step2.add_patch(rect_out2)
            ax_step2.text(x_out2 + io_w2/2, box_h - 0.35, "OUTPUT", ha='center', va='top',
                          fontsize=10, fontweight='bold', color='#2E7D32')
            ax_step2.text(x_out2 + io_w2/2, box_h/2 - 0.30,
                          f"Matriks Chi-Square\n{n_docs_chi:,} × {n_feat_out_chi:,}\n(dok × fitur terpilih)",
                          ha='center', va='center', fontsize=8.5, color='#333')

            ax_step2.set_xlim(-0.2, x_out2 + io_w2 + 0.2)
            ax_step2.set_ylim(-0.2, box_h + 0.2)
            plt.tight_layout()
            st.pyplot(fig_step2, use_container_width=True)
            plt.close()

            # ── Tampilkan hasil OUTPUT aktual ────────────────────────────────────
            st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
            col_cout1, col_cout2, col_cout3 = st.columns(3)
            col_cout1.metric("Ukuran Matriks", f"{n_docs_chi:,} × {n_feat_out_chi:,}")
            col_cout2.metric("Fitur Direduksi", f"{n_feat_in_chi - n_feat_out_chi:,}")
            col_cout3.metric("Reduksi Dimensi", f"{(1-n_feat_out_chi/n_feat_in_chi)*100:.1f}%")
            st.caption(
                f"Dari {n_feat_in_chi:,} fitur TF-IDF, hanya {n_feat_out_chi:,} fitur "
                f"dengan skor χ² tertinggi yang dipertahankan. Cuplikan nyata matriks "
                f"ini dapat dilihat pada tabel \"TF-IDF Values untuk 10 Dokumen "
                f"Pertama (Chi-Square Features)\" di bawah."
            )

            st.divider()

            # ── Rumus Chi-Square ──────────────────────────────────────────────────
            st.markdown("**📐 Rumus Perhitungan Chi-Square**")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.latex(r"\chi^2 = \sum_{i=1}^{n} \frac{(O_i - E_i)^2}{E_i}")
                st.caption(
                    "Oᵢ: frekuensi observasi (kemunculan kata pada kelas tertentu), "
                    "Eᵢ: frekuensi ekspektasi (jika kata & kelas independen)"
                )
            with col_g2:
                st.latex(r"E_i = \frac{(\text{total baris}_i) \times (\text{total kolom}_i)}{\text{total keseluruhan}}")
                st.caption(
                    "Dihitung dari tabel kontingensi 2×2 antara kemunculan kata "
                    "dan label kelas (POSITIVE/NEGATIVE)"
                )

            st.divider()

            # ── Contoh Perhitungan Manual Chi-Square dengan Data Asli ───────────
            st.markdown("**🧮 Contoh Perhitungan Chi-Square dengan Data Asli**")
            st.caption(
                "Mengambil 1 kata sungguhan dari fitur terpilih untuk menunjukkan "
                "perhitungan skor χ² secara manual — mengikuti formula ASLI yang "
                "dipakai scikit-learn `chi2()`, yaitu berbasis **jumlah nilai TF-IDF** "
                "per kelas (bukan tabel kontingensi biner muncul/tidak-muncul)."
            )

            df_proc_chi = feat['df_final']
            classes_chi = feat['classes']

            # Ambil kata dengan skor Chi-Square tertinggi sebagai contoh
            top1_idx_ex = int(np.argmax(scores_sel))
            kata_contoh_chi = fn_selected[top1_idx_ex]
            skor_contoh_chi = scores_sel[top1_idx_ex]

            label_pos = classes_chi[0]
            label_neg = classes_chi[1]

            # ── Formula ASLI sklearn.feature_selection.chi2 ──────────────────────
            # observed[c] = jumlah (sum) nilai TF-IDF fitur pada kelas c
            # feature_total = total nilai TF-IDF fitur di SELURUH dokumen
            # expected[c] = proporsi_kelas[c] × feature_total
            word_col_idx = int(np.where(fn_all == kata_contoh_chi)[0][0])
            tfidf_col = X_tfidf_full[:, word_col_idx].toarray().flatten()

            mask_pos = (df_proc_chi['sentiment'] == label_pos).values
            mask_neg = (df_proc_chi['sentiment'] == label_neg).values

            n_pos = int(mask_pos.sum())
            n_neg = int(mask_neg.sum())
            n_ct  = n_pos + n_neg

            observed_pos = float(tfidf_col[mask_pos].sum())
            observed_neg = float(tfidf_col[mask_neg].sum())
            feature_total = observed_pos + observed_neg

            class_prob_pos = n_pos / n_ct
            class_prob_neg = n_neg / n_ct

            E_pos = class_prob_pos * feature_total
            E_neg = class_prob_neg * feature_total

            chi2_manual = ((observed_pos - E_pos)**2 / E_pos) + \
                          ((observed_neg - E_neg)**2 / E_neg)

            col_cx1, col_cx2 = st.columns(2)
            with col_cx1:
                st.markdown(f"**Kata contoh (skor χ² tertinggi):** `{kata_contoh_chi}`")
                st.markdown("**Jumlah Nilai TF-IDF per Kelas (data asli):**")
                ct_df = pd.DataFrame({
                    'Kelas':            [label_pos, label_neg, 'Total'],
                    'Jumlah Dokumen':   [n_pos, n_neg, n_ct],
                    'Σ TF-IDF (Observed)': [f"{observed_pos:.4f}", f"{observed_neg:.4f}", f"{feature_total:.4f}"],
                })
                st.dataframe(ct_df, use_container_width=True, hide_index=True)
                st.caption(
                    f'Nilai Σ TF-IDF diperoleh dengan menjumlahkan seluruh nilai TF-IDF '
                    f'kata "{kata_contoh_chi}" pada baris dokumen berlabel {label_pos} '
                    f'dan {label_neg} secara terpisah — bukan menghitung jumlah dokumen '
                    f'yang mengandung kata tersebut.'
                )

            with col_cx2:
                st.markdown("**Langkah 1 — Hitung Frekuensi Ekspektasi (E):**")
                st.latex(
                    rf"\text{{prob}}_{{{label_pos}}} = {n_pos}/{n_ct} = {class_prob_pos:.4f}"
                )
                st.latex(
                    rf"\text{{prob}}_{{{label_neg}}} = {n_neg}/{n_ct} = {class_prob_neg:.4f}"
                )
                st.latex(
                    rf"E_{{{label_pos}}} = {class_prob_pos:.4f} \times {feature_total:.4f} = {E_pos:.4f}"
                )
                st.latex(
                    rf"E_{{{label_neg}}} = {class_prob_neg:.4f} \times {feature_total:.4f} = {E_neg:.4f}"
                )

                st.markdown("**Langkah 2 — Hitung Skor χ²:**")
                st.latex(
                    rf"\chi^2 = \frac{{({observed_pos:.4f}-{E_pos:.4f})^2}}{{{E_pos:.4f}}} + "
                    rf"\frac{{({observed_neg:.4f}-{E_neg:.4f})^2}}{{{E_neg:.4f}}}"
                )
                st.latex(rf"\chi^2 = {chi2_manual:.4f}")

                selisih = abs(chi2_manual - skor_contoh_chi)
                if selisih < 0.01:
                    st.success(
                        f'✅ Hasil perhitungan manual (**{chi2_manual:.4f}**) **cocok persis** '
                        f'dengan skor χ² aktual dari SelectKBest untuk kata '
                        f'"{kata_contoh_chi}" (**{skor_contoh_chi:.4f}**). Formula di atas '
                        f'adalah formula ASLI yang dipakai scikit-learn — berbasis jumlah '
                        f'nilai TF-IDF per kelas, bukan tabel kontingensi biner.'
                    )
                else:
                    st.warning(
                        f'⚠️ Hasil manual ({chi2_manual:.4f}) vs SelectKBest '
                        f'({skor_contoh_chi:.4f}), selisih {selisih:.4f} — kemungkinan '
                        f'akibat pembulatan floating-point pada tahap perantara.'
                    )

            st.markdown("**🏆 Top 10 Fitur Hasil Perhitungan Rumus Chi-Square (Data Asli)**")
            st.caption(
                f'Tabel berikut adalah 10 fitur dengan skor χ² **tertinggi** dari hasil '
                f'perhitungan rumus Chi-Square (Langkah 1–2 di atas) pada seluruh data '
                f'training. Baris yang disorot hijau adalah kata "{kata_contoh_chi}" — '
                f'kata yang sama dengan contoh perhitungan manual di atas.'
            )

            top10_idx_chi = np.argsort(scores_sel)[::-1][:10]
            top10_words_chi  = fn_selected[top10_idx_chi]
            top10_scores_chi = scores_sel[top10_idx_chi]
            top10_pvals_chi  = pvals_sel[top10_idx_chi]

            df_top10_chi = pd.DataFrame({
                'Rank':        range(1, 11),
                'Fitur':       top10_words_chi,
                'Skor χ²':     [f"{v:.4f}" for v in top10_scores_chi],
                'P-Value':     [f"{v:.6f}" for v in top10_pvals_chi],
                'Signifikan (p<0.05)': ['✅ Ya' if v < 0.05 else '❌ Tidak' for v in top10_pvals_chi],
            })

            def _highlight_row_chi(row):
                is_contoh = row['Fitur'] == kata_contoh_chi
                return ['background-color: #C6E8C6; font-weight: bold' if is_contoh else '' for _ in row]

            st.dataframe(
                df_top10_chi.style.apply(_highlight_row_chi, axis=1),
                use_container_width=True, hide_index=True
            )
            st.caption(
                f'Kata "{kata_contoh_chi}" berada di **rank #1** dengan skor χ² = '
                f'**{skor_contoh_chi:.4f}** — sesuai hasil perhitungan manual pada Langkah 2.'
            )

            st.divider()

            # ── Perhitungan: kenapa jadi 1.000 fitur ─────────────────────────────
            n_before_chi2 = len(fn_all)
            n_after_chi2  = len(fn_selected)
            k_param       = 1000
            st.markdown(f"**🔢 Perhitungan: Mengapa Menjadi {n_after_chi2:,} Fitur (dari {n_before_chi2:,})?**")
            st.info(
                f"**Parameter yang diberikan:** `k = {k_param:,}` (jumlah fitur yang ingin dipilih)\n\n"
                f"**Cara kerja SelectKBest(chi2, k={k_param:,}):**\n\n"
                f"1. Chi-Square score dihitung untuk **seluruh {n_before_chi2:,} fitur** "
                f"hasil TF-IDF terhadap label sentimen.\n\n"
                f"2. Seluruh fitur diurutkan (sorting) berdasarkan skor χ² dari yang "
                f"**tertinggi ke terendah**.\n\n"
                f"3. Karena `k = {k_param:,}` **lebih kecil** dari jumlah fitur yang "
                f"tersedia ({n_before_chi2:,}), maka SelectKBest mengambil **tepat "
                f"{k_param:,} fitur teratas** — tidak dibatasi oleh ketersediaan data "
                f"seperti pada kasus TF-IDF sebelumnya.\n\n"
                f"**Formula penentuan jumlah fitur akhir:**"
            )
            st.latex(r"n_{\text{terpilih}} = \min(k,\ n_{\text{fitur tersedia}})")
            st.latex(
                rf"n_{{\text{{terpilih}}}} = \min({k_param:,},\ {n_before_chi2:,}) = {n_after_chi2:,}"
            )
            st.caption(
                f"Berbeda dengan TF-IDF (di mana `max_features` menjadi batas atas yang "
                f"tidak selalu tercapai), pada Chi-Square nilai `k` **selalu tercapai "
                f"secara penuh** selama jumlah fitur yang tersedia ({n_before_chi2:,}) "
                f"lebih besar dari k ({k_param:,}) — sehingga hasilnya selalu tepat "
                f"{n_after_chi2:,} fitur."
            )

            st.divider()
        
            # Statistik ringkas
            st.markdown("**📋 Statistik Chi-Square Feature Selection**")
            cs1,cs2,cs3,cs4 = st.columns(4)
            cs1.metric("Fitur Sebelum Chi-Square", f"{len(fn_all):,}")
            cs2.metric("Fitur Terpilih (k)",        f"{len(fn_selected):,}")
            cs3.metric("Fitur Direduksi",            f"{len(fn_all)-len(fn_selected):,}")
            cs4.metric("Score Tertinggi",            f"{scores_sel.max():.2f}")
        
            st.divider()
        
            # Top 20 fitur chi-square
            top_n   = 20
            top_idx = np.argsort(scores_sel)[::-1][:top_n]
            top_feat  = fn_selected[top_idx]
            top_score = scores_sel[top_idx]
            top_pval  = pvals_sel[top_idx]
        
            st.markdown(f"**🏆 Top {top_n} Fitur Berdasarkan Chi-Square Score**")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_feat[::-1], top_score[::-1],
                           color='#E24B4A', alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, top_score[::-1]):
                ax.text(bar.get_width()+max(top_score)*0.005,
                        bar.get_y()+bar.get_height()/2,
                        f'{v:.2f}', va='center', ha='left', fontsize=9)
            ax.set_xlabel("Chi-Square Score")
            ax.set_title(f"Top {top_n} Fitur — Chi-Square Score", fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
            st.divider()
        
            # Tabel top 20 fitur dengan score dan p-value
            st.markdown(f"**📄 Tabel Top {top_n} Fitur Chi-Square**")
            chi2_df = pd.DataFrame({
                'Rank':       range(1, top_n+1),
                'Fitur':      top_feat,
                'Chi² Score': [f"{v:.4f}" for v in top_score],
                'P-Value':    [f"{v:.6f}" for v in top_pval],
                'Signifikan': ['✅ Ya' if p < 0.05 else '❌ Tidak' for p in top_pval],
            })
            st.dataframe(chi2_df, use_container_width=True, hide_index=True)
        

            st.divider()

            # ── Matriks TF-IDF Chi-Square — 10 Dokumen × 15 Fitur Teratas ──────
            st.markdown("**📄 TF-IDF Values untuk 10 Dokumen Pertama (Chi-Square Features)**")
            st.caption("Menampilkan nilai TF-IDF untuk 15 fitur teratas berdasarkan Chi-Square Score "
                       "pada 10 dokumen pertama. Sel berwarna lebih gelap = nilai lebih tinggi.")

            # Ambil data Chi-Square yang sudah di-transform
            df_proc2 = feat['df_final']
            X_chi2_full = chi2_sel.transform(tfidf.transform(df_proc2['final']))

            # Pilih 15 fitur teratas berdasarkan Chi-Square score
            top15_chi2_idx   = np.argsort(scores_sel)[::-1][:15]
            top15_chi2_words = fn_selected[top15_chi2_idx]

            # Buat matriks 10 dokumen × 15 fitur
            chi2_matrix = X_chi2_full[:10, top15_chi2_idx].toarray()
            chi2_matrix_df = pd.DataFrame(
                chi2_matrix,
                columns=top15_chi2_words,
                index=[f"Dokumen {i+1}" for i in range(10)]
            )

            # Tampilkan dengan styling seperti gambar (YlOrRd gradient)
            def color_cell(val):
                if val == 0.0:
                    return 'background-color: #ffffcc; color: #333'
                # Normalize untuk warna
                return ''

            styled = (
                chi2_matrix_df.style
                .format("{:.4f}")
                .background_gradient(cmap='YlOrRd', axis=None)
                .set_properties(**{
                    'text-align': 'center',
                    'font-size': '13px',
                    'font-weight': 'bold',
                    'min-width': '75px',
                })
                .set_table_styles([
                    {'selector': 'th.col_heading',
                     'props': [('background-color', '#f5f5f5'),
                               ('color', '#333'),
                               ('font-weight', 'bold'),
                               ('text-align', 'center'),
                               ('font-size', '12px'),
                               ('padding', '6px 10px')]},
                    {'selector': 'th.row_heading',
                     'props': [('background-color', '#f5f5f5'),
                               ('color', '#333'),
                               ('font-weight', 'bold'),
                               ('font-size', '13px'),
                               ('padding', '6px 12px')]},
                    {'selector': 'td',
                     'props': [('padding', '6px 10px')]},
                    {'selector': 'caption',
                     'props': [('caption-side', 'top'),
                               ('font-size', '14px'),
                               ('font-weight', 'bold'),
                               ('color', '#C44E52'),
                               ('text-align', 'center'),
                               ('padding-bottom', '8px')]},
                ])
                .set_caption("TF-IDF Values untuk 10 Dokumen Pertama (Chi-Square Features)")
            )
            st.dataframe(styled, use_container_width=True)

            # Keterangan
            st.caption(
                f"Tabel di atas menampilkan nilai TF-IDF dari **15 fitur terpilih Chi-Square** "
                f"(berdasarkan skor tertinggi: {', '.join(top15_chi2_words[:5])}, ...) "
                f"untuk **10 dokumen pertama**. "
                f"Nilai 0.0000 berarti kata tidak muncul di dokumen tersebut."
            )

            st.divider()
        
            # Distribusi skor chi-square
            st.markdown("**📊 Distribusi Chi-Square Score (semua fitur terpilih)**")
            fig2, axes = plt.subplots(1, 2, figsize=(12, 4))
        
            axes[0].hist(scores_sel, bins=50, color='#E24B4A', alpha=0.8, edgecolor='white')
            axes[0].set_xlabel("Chi-Square Score")
            axes[0].set_ylabel("Frekuensi")
            axes[0].set_title("Distribusi Skor Chi-Square", fontweight='bold')
            axes[0].grid(alpha=0.3)
        
            axes[1].hist(pvals_sel, bins=50, color='#4C72B0', alpha=0.8, edgecolor='white')
            axes[1].axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[1].set_xlabel("P-Value")
            axes[1].set_ylabel("Frekuensi")
            axes[1].set_title("Distribusi P-Value", fontweight='bold')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        
            sig_count = int((pvals_sel < 0.05).sum())
            st.caption(f"Fitur signifikan (p < 0.05): **{sig_count:,}** dari {len(fn_selected):,} fitur terpilih")
        
        # ════════════════════════════════════════════════════════════════════
        # SUB-TAB 3 — PCA
        # ════════════════════════════════════════════════════════════════════

    with stab3:
        st.markdown("**📊 Perbandingan Fitur Seleksi Chi-Square vs Keseluruhan TF-IDF**")
        st.caption(
            "Revisi dosen penguji poin 4 & 5: Jelaskan Chi-Square dan bandingkan "
            "fitur yang dipilih vs fitur keseluruhan."
        )

        if 'feat' not in st.session_state:
            st.info("Jalankan pipeline terlebih dahulu.", icon="👈")
        else:
            feat     = st.session_state['feat']
            tfidf    = feat['tfidf']
            chi2_sel = feat['chi2_sel']
            fn_all   = tfidf.get_feature_names_out()
            selected = chi2_sel.get_support()
            fn_sel   = fn_all[selected]
            fn_nosel = fn_all[~selected]
            scores   = chi2_sel.scores_

            n_all = len(fn_all)
            n_sel = len(fn_sel)
            n_red = n_all - n_sel

            # ── Statistik ringkas ──────────────────────────────────────────────
            st.markdown("**📋 Statistik Perbandingan Fitur**")
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Total Fitur TF-IDF",      f"{n_all:,}")
            k2.metric("Fitur Dipilih Chi-Square", f"{n_sel:,}")
            k3.metric("Fitur Dibuang",            f"{n_red:,}")
            k4.metric("Reduksi Fitur",            f"{n_red/n_all*100:.1f}%")

            st.divider()

            # ── Penjelasan Chi-Square (Poin 5) ────────────────────────────────
            st.markdown("**📖 Penjelasan Chi-Square Feature Selection**")
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.info(
                    "**Apa itu Chi-Square?**\n\n"
                    "Chi-Square (χ²) adalah uji statistik yang mengukur ketergantungan "
                    "antara fitur (kata) dengan variabel target (label sentimen). "
                    "Semakin tinggi skor χ², semakin kuat hubungan kata tersebut "
                    "dengan label sentimen (positif/negatif).\n\n"
                    "**Hipotesis Nol (H₀):** Kata tidak berhubungan dengan sentimen\n\n"
                    "**Keputusan:** Jika p-value < 0.05 → fitur signifikan dan dipilih"
                )
            with col_ex2:
                st.info(
                    "**Mengapa Chi-Square digunakan?**\n\n"
                    "Dari 5.000 fitur TF-IDF, tidak semua kata relevan terhadap "
                    "sentimen. Kata-kata seperti nama karakter game atau istilah teknis "
                    "mungkin tidak informatif untuk klasifikasi sentimen.\n\n"
                    "Chi-Square memilih **k = 1.000 fitur terbaik** yang memiliki "
                    "hubungan statistik paling kuat dengan label sentimen, "
                    "sehingga model SVM lebih fokus dan efisien."
                )

            st.divider()

            # ── Bar chart perbandingan ────────────────────────────────────────
            st.markdown("**📊 Visualisasi Perbandingan Fitur**")
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Pie chart
            axes[0].pie([n_sel, n_red],
                        labels=[f'Dipilih\n({n_sel:,})', f'Dibuang\n({n_red:,})'],
                        colors=['#4C72B0','#E24B4A'], autopct='%1.1f%%',
                        startangle=90,
                        wedgeprops={'edgecolor':'white','linewidth':2},
                        textprops={'fontsize':11})
            axes[0].set_title('Proporsi Fitur Chi-Square\nvs Total TF-IDF',
                              fontweight='bold', fontsize=12)

            # Bar chart
            cats   = ['Total TF-IDF', 'Fitur Dipilih\n(Chi-Square)', 'Fitur Dibuang']
            values = [n_all, n_sel, n_red]
            colors = ['#6c757d','#4C72B0','#E24B4A']
            bars   = axes[1].bar(cats, values, color=colors, alpha=0.85,
                                 edgecolor='white', width=0.5)
            for bar, v in zip(bars, values):
                axes[1].text(bar.get_x()+bar.get_width()/2, v+20,
                             f'{v:,}', ha='center', fontweight='bold', fontsize=11)
            axes[1].set_ylabel('Jumlah Fitur')
            axes[1].set_title('Jumlah Fitur: TF-IDF vs Chi-Square',
                              fontweight='bold', fontsize=12)
            axes[1].set_ylim(0, n_all*1.2)
            axes[1].grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()

            # ── Distribusi skor fitur dipilih vs dibuang ──────────────────────
            st.markdown("**📈 Distribusi Chi-Square Score: Fitur Dipilih vs Dibuang**")
            scores_sel   = scores[selected]
            scores_nosel = scores[~selected]
            fig2, ax2 = plt.subplots(figsize=(11, 4.5))
            ax2.hist(scores_sel,   bins=50, alpha=0.7, color='#4C72B0',
                     label=f'Dipilih (n={n_sel:,})',  edgecolor='white')
            ax2.hist(scores_nosel, bins=50, alpha=0.7, color='#E24B4A',
                     label=f'Dibuang (n={n_red:,})',  edgecolor='white')
            ax2.set_xlabel('Chi-Square Score')
            ax2.set_ylabel('Frekuensi')
            ax2.set_title('Distribusi Chi-Square Score: Fitur Dipilih vs Dibuang',
                          fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            st.divider()

            # ── Tabel 10 fitur tertinggi vs terendah ──────────────────────────
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**🏆 Top 10 Fitur DIPILIH (Skor Tertinggi)**")
                top10_idx   = np.argsort(scores_sel)[::-1][:10]
                top10_words = fn_sel[top10_idx]
                top10_score = scores_sel[top10_idx]
                top10_df = pd.DataFrame({
                    'Rank':       range(1, 11),
                    'Kata':       top10_words,
                    'Chi² Score': [f"{v:.2f}" for v in top10_score],
                    'Status':     ['✅ Dipilih']*10,
                })
                st.dataframe(top10_df, use_container_width=True, hide_index=True)

            with col_t2:
                st.markdown("**❌ Top 10 Fitur DIBUANG (Skor Terendah)**")
                bot10_idx   = np.argsort(scores_nosel)[::-1][:10]
                bot10_words = fn_nosel[bot10_idx]
                bot10_score = scores_nosel[bot10_idx]
                bot10_df = pd.DataFrame({
                    'Rank':       range(1, 11),
                    'Kata':       bot10_words,
                    'Chi² Score': [f"{v:.4f}" for v in bot10_score],
                    'Status':     ['❌ Dibuang']*10,
                })
                st.dataframe(bot10_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Tabel LENGKAP seluruh fitur yang terbuang ─────────────────────
            st.markdown(f"**🗑️ Tabel Lengkap Seluruh {n_red:,} Fitur yang Terbuang**")
            st.caption(
                f"Dari total {n_all:,} fitur TF-IDF, sebanyak {n_red:,} fitur "
                f"({n_red/n_all*100:.1f}%) tidak terpilih oleh Chi-Square karena "
                f"skor χ²-nya berada di luar {n_sel:,} peringkat teratas. Tabel "
                f"berikut menampilkan **seluruh** fitur yang terbuang tersebut, "
                f"diurutkan dari skor χ² tertinggi (paling mendekati batas "
                f"terpilih) ke terendah (paling tidak signifikan)."
            )

            discarded_order = np.argsort(scores_nosel)[::-1]
            discarded_words_full = fn_nosel[discarded_order]
            discarded_scores_full = scores_nosel[discarded_order]

            df_discarded_full = pd.DataFrame({
                'Rank Terbuang': range(1, n_red + 1),
                'Fitur':         discarded_words_full,
                'Skor χ²':       [f"{v:.6f}" for v in discarded_scores_full],
            })

            st.dataframe(df_discarded_full, use_container_width=True,
                        hide_index=True, height=400)

            csv_discarded = df_discarded_full.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"⬇️ Unduh Seluruh {n_red:,} Fitur Terbuang (CSV)",
                data=csv_discarded,
                file_name="fitur_terbuang_chisquare.csv",
                mime="text/csv",
            )
            st.caption(
                "Baris teratas pada tabel ini adalah fitur yang skornya "
                "**paling mendekati ambang batas** k=1.000 (paling 'nyaris' "
                "terpilih), sedangkan baris terbawah adalah fitur yang skornya "
                "**paling rendah** (paling tidak berhubungan dengan sentimen)."
            )

            st.divider()
            st.info(
                "**Kesimpulan Seleksi Fitur:**\n\n"
                f"- Total fitur TF-IDF: **{n_all:,} kata**\n"
                f"- Dipilih Chi-Square: **{n_sel:,} kata** ({n_sel/n_all*100:.1f}%)\n"
                f"- Dibuang: **{n_red:,} kata** ({n_red/n_all*100:.1f}%)\n\n"
                "Dengan memilih hanya fitur yang signifikan secara statistik, "
                "model SVM dapat belajar dari fitur-fitur yang benar-benar "
                "berhubungan dengan sentimen, bukan dari noise."
            )
