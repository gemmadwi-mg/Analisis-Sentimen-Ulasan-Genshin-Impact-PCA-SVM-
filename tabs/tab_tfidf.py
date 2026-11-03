"""
tabs/tab_tfidf.py
Tab: Tfidf
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def render(df=None, **kwargs):
    """Render konten tab ini."""
    """Render konten tab ini. Dipanggil dari app.py."""
    import matplotlib.pyplot as plt

    st.subheader("📐 Hasil TF-IDF, Chi-Square & PCA")
    st.caption("Hasil vectorization dan seleksi fitur untuk keperluan dokumentasi skripsi")

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
    else:
        feat  = st.session_state['feat']
        r_pca = st.session_state['r_pca']

        # ── Sub-tabs ─────────────────────────────────────────────────────────
        stab1, stab2, stab3 = st.tabs(["📊 TF-IDF","📉 Chi-Square","🔷 PCA"])

        # ════════════════════════════════════════════════════════════════════
        # SUB-TAB 1 — TF-IDF
        # ════════════════════════════════════════════════════════════════════
        with stab1:
            tfidf    = feat['tfidf']
            df_proc  = feat['df_final']

            fn = tfidf.get_feature_names_out()

            # Statistik ringkas
            st.markdown("**📋 Statistik TF-IDF**")
            s1,s2,s3,s4 = st.columns(4)
            s1.metric("Total Dokumen",       f"{len(df_proc):,}")
            s2.metric("Total Fitur (max)",    f"{len(fn):,}")
            s3.metric("Parameter max_features","5.000")
            s4.metric("Metode Weighting",     "TF-IDF")

            st.divider()

            # Hitung skor TF-IDF rata-rata per kata
            from sklearn.feature_extraction.text import TfidfVectorizer
            X_tfidf_full = tfidf.transform(df_proc['final'])
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
            pca    = r_pca['pca']
            scaler = r_pca['scaler']

            orig_dim = r_pca['orig_dim']
            n_comp   = r_pca['pca_n']
            var_exp  = r_pca['pca_var']
            red_pct  = (1 - n_comp / orig_dim) * 100

            # Statistik ringkas
            st.markdown("**📋 Statistik PCA**")
            p1,p2,p3,p4 = st.columns(4)
            p1.metric("Dimensi Awal (Chi-Square)", f"{orig_dim:,}")
            p2.metric("Komponen PCA",               f"{n_comp:,}")
            p3.metric("Variance Explained",         f"{var_exp*100:.2f}%")
            p4.metric("Reduksi Dimensi",            f"{red_pct:.1f}%")

            st.divider()

            evr = pca.explained_variance_ratio_
            cumvar = np.cumsum(evr)

            # Plot 1: Cumulative explained variance
            st.markdown("**📈 Cumulative Explained Variance**")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(range(1, len(cumvar)+1), cumvar*100,
                    color='#4C72B0', linewidth=2, marker='.', markersize=3)
            ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, label='Threshold 95%')
            ax.axvline(x=n_comp, color='green', linestyle='--', linewidth=1.5,
                       label=f'Komponen terpilih: {n_comp}')
            ax.fill_between(range(1, len(cumvar)+1), cumvar*100,
                            alpha=0.12, color='#4C72B0')
            ax.set_xlabel("Jumlah Komponen PCA", fontsize=11)
            ax.set_ylabel("Cumulative Explained Variance (%)", fontsize=11)
            ax.set_title("Cumulative Explained Variance per Komponen PCA",
                         fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()

            # Plot 2: Individual explained variance (top 30)
            st.markdown("**📊 Explained Variance per Komponen (Top 30)**")
            show_n = min(30, len(evr))
            fig2, ax2 = plt.subplots(figsize=(12, 4.5))
            ax2.bar(range(1, show_n+1), evr[:show_n]*100,
                    color='#55A868', alpha=0.85, edgecolor='white')
            ax2.set_xlabel("Komponen PCA", fontsize=11)
            ax2.set_ylabel("Explained Variance (%)", fontsize=11)
            ax2.set_title(f"Explained Variance per Komponen (Top {show_n})",
                          fontsize=13, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            st.divider()

            # Tabel explained variance per komponen (top 20)
            st.markdown("**📄 Tabel Explained Variance per Komponen (Top 20)**")
            show_tbl = min(20, len(evr))
            pca_tbl  = pd.DataFrame({
                'Komponen':            [f"PC-{i+1}" for i in range(show_tbl)],
                'Explained Variance (%)': [f"{v*100:.4f}" for v in evr[:show_tbl]],
                'Cumulative (%)':         [f"{c*100:.4f}" for c in cumvar[:show_tbl]],
                'Eigenvalue':             [f"{e:.4f}" for e in pca.explained_variance_[:show_tbl]],
            })
            st.dataframe(pca_tbl, use_container_width=True, hide_index=True)

            st.divider()

            # Reduksi dimensi visual
            st.markdown("**🗜️ Visualisasi Reduksi Dimensi**")
            fig3, ax3 = plt.subplots(figsize=(7, 4))
            cats   = ['Dimensi Awal\n(Chi-Square)', 'Komponen PCA\n(variance 95%)']
            values = [orig_dim, n_comp]
            colors = ['#6c757d','#4C72B0']
            bars   = ax3.bar(cats, values, color=colors, alpha=0.85,
                             width=0.4, edgecolor='white')
            for bar, v in zip(bars, values):
                ax3.text(bar.get_x()+bar.get_width()/2, v+orig_dim*0.01,
                         f'{v:,}', ha='center', fontweight='bold', fontsize=13)
            ax3.set_ylabel("Jumlah Dimensi")
            ax3.set_title(f"Reduksi Dimensi PCA\n{orig_dim} → {n_comp} komponen ({red_pct:.1f}% berkurang)",
                          fontweight='bold')
            ax3.grid(axis='y', alpha=0.3)
            # Anotasi panah
            ax3.annotate('', xy=(1, n_comp+orig_dim*0.05),
                         xytext=(0, orig_dim*0.6),
                         arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=2))
            ax3.text(0.5, (orig_dim+n_comp)/2,
                     f'−{red_pct:.1f}%', ha='center', color='#4C72B0',
                     fontsize=12, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()


    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 4 — SPLIT DATA & SMOTE
    # ══════════════════════════════════════════════════════════════════════════════
