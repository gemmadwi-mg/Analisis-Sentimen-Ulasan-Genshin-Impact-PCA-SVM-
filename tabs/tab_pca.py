"""
tabs/tab_pca.py
Tab: PCA (Principal Component Analysis)
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.pipeline import pca_all_splits, pca_reduction_by_threshold


def render(df=None, **kwargs):
    """Render tab PCA."""
    st.subheader("🔷 Hasil PCA")
    st.caption("Reduksi dimensi dengan Principal Component Analysis untuk dokumentasi skripsi")

    if 'r_pca' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    r_pca        = st.session_state['r_pca']
    feat         = st.session_state.get('feat', {})
    split_label  = st.session_state.get('split_label', '80:20')
    pca          = r_pca['pca']
    orig_dim     = r_pca['orig_dim']
    n_comp       = r_pca['pca_n']
    var_exp      = r_pca['pca_var']
    red_pct      = (1 - n_comp / orig_dim) * 100
    evr          = pca.explained_variance_ratio_
    cumvar       = np.cumsum(evr)

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    stab1, stab2, stab3, stab4, stab5 = st.tabs([
        "📋 Statistik & Visualisasi",
        "📄 Tabel Komponen",
        "📊 Perbandingan Semua Rasio Split",
        "🎚️ Perbandingan Threshold PCA",
        "🧭 Arah & Interpretasi Komponen",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Statistik & Visualisasi
    # ══════════════════════════════════════════════════════════════════════
    with stab1:
        st.markdown(f"**Rasio split aktif: `{split_label}`**")

        # ── Langkah-Langkah PCA ────────────────────────────────────────────
        st.markdown("**🧭 Langkah-Langkah Perhitungan Principal Component Analysis (PCA)**")
        st.caption(
            "PCA mereduksi dimensi data dengan mencari sumbu/arah baru (komponen "
            "utama) yang menangkap variansi data sebesar mungkin, melalui 5 langkah "
            "berikut, diapit kotak INPUT (data masuk) dan OUTPUT (hasil akhir)."
        )

        n_train_pca = feat['X_train_sm'].shape[0] if feat else 0

        fig_pcastep, ax_pcastep = plt.subplots(figsize=(16.5, 3.4))
        ax_pcastep.axis('off')
        pca_steps = [
            ("1. Standardisasi\nData", "StandardScaler:\nmean=0, std=1\npada tiap fitur"),
            ("2. Hitung Matriks\nKovariansi", "Mengukur hubungan\nantar pasangan\nfitur"),
            ("3. Hitung Eigenvalue\n& Eigenvector", "Cari arah (vektor)\ndengan variansi\nterbesar"),
            ("4. Urutkan &\nHitung Kumulatif", "Urutkan eigenvalue\nterbesar → terkecil,\njumlahkan variansinya"),
            ("5. Pilih Komponen\nSesuai Threshold", "Ambil komponen\nhingga variansi kumulatif\n≥ threshold"),
        ]
        box_w2, box_h2, gap2 = 2.3, 1.8, 0.35
        io_w3 = 2.3
        colors_pca = ['#E6F1FB', '#EAF3DE', '#FAEEDA', '#FBEAF0', '#EEEDFE']
        edge_pca   = ['#185FA5', '#3B6D11', '#854F0B', '#993556', '#534AB7']

        # ── Kotak INPUT ────────────────────────────────────────────────────────
        x_in3 = 0
        rect_in3 = plt.Rectangle((x_in3, 0), io_w3, box_h2, facecolor='#F0F0F0',
                                  edgecolor='#555555', linewidth=1.6,
                                  linestyle='--', zorder=2)
        ax_pcastep.add_patch(rect_in3)
        ax_pcastep.text(x_in3 + io_w3/2, box_h2 - 0.35, "INPUT", ha='center', va='top',
                        fontsize=10, fontweight='bold', color='#333333')
        ax_pcastep.text(x_in3 + io_w3/2, box_h2/2 - 0.30,
                        f"Matriks Chi-Square\n{n_train_pca:,} × {orig_dim:,}\n(data training)",
                        ha='center', va='center', fontsize=8.5, color='#333')
        ax_pcastep.annotate('', xy=(x_in3 + io_w3 + gap2 - 0.08, box_h2/2),
                            xytext=(x_in3 + io_w3 + 0.08, box_h2/2),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

        # ── Kotak proses (step 1-5) ────────────────────────────────────────────
        x_offset3 = x_in3 + io_w3 + gap2
        for i, ((title, desc), c, ec) in enumerate(zip(pca_steps, colors_pca, edge_pca)):
            x = x_offset3 + i * (box_w2 + gap2)
            rect = plt.Rectangle((x, 0), box_w2, box_h2, facecolor=c,
                                  edgecolor=ec, linewidth=1.5, zorder=2)
            ax_pcastep.add_patch(rect)
            ax_pcastep.text(x + box_w2/2, box_h2 - 0.35, title, ha='center', va='top',
                            fontsize=9.5, fontweight='bold', color=ec)
            ax_pcastep.text(x + box_w2/2, box_h2/2 - 0.25, desc, ha='center', va='center',
                            fontsize=8, color='#333')
            ax_pcastep.annotate('', xy=(x + box_w2 + gap2 - 0.08, box_h2/2),
                                xytext=(x + box_w2 + 0.08, box_h2/2),
                                arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

        # ── Kotak OUTPUT ───────────────────────────────────────────────────────
        x_out3 = x_offset3 + len(pca_steps) * (box_w2 + gap2)
        rect_out3 = plt.Rectangle((x_out3, 0), io_w3, box_h2, facecolor='#E1F5E1',
                                   edgecolor='#2E7D32', linewidth=1.8, zorder=2)
        ax_pcastep.add_patch(rect_out3)
        ax_pcastep.text(x_out3 + io_w3/2, box_h2 - 0.35, "OUTPUT", ha='center', va='top',
                        fontsize=10, fontweight='bold', color='#2E7D32')
        ax_pcastep.text(x_out3 + io_w3/2, box_h2/2 - 0.30,
                        f"Matriks PCA\n{n_train_pca:,} × {n_comp:,}\n(komponen utama)",
                        ha='center', va='center', fontsize=8.5, color='#333')

        ax_pcastep.set_xlim(-0.2, x_out3 + io_w3 + 0.2)
        ax_pcastep.set_ylim(-0.2, box_h2 + 0.2)
        plt.tight_layout()
        st.pyplot(fig_pcastep, use_container_width=True)
        plt.close()

        # ── Tampilkan hasil OUTPUT aktual ──────────────────────────────────────
        st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
        col_pout1, col_pout2, col_pout3, col_pout4 = st.columns(4)
        col_pout1.metric("Ukuran Matriks", f"{n_train_pca:,} × {n_comp:,}")
        col_pout2.metric("Fitur Direduksi", f"{orig_dim - n_comp:,}")
        col_pout3.metric("Reduksi Dimensi", f"{red_pct:.1f}%")
        col_pout4.metric("Variansi Dipertahankan", f"{var_exp*100:.2f}%")
        st.caption(
            f"Dari {orig_dim:,} fitur Chi-Square, PCA meringkasnya menjadi "
            f"{n_comp:,} komponen utama (threshold 95%, rasio split {split_label}) "
            f"sambil tetap mempertahankan {var_exp*100:.2f}% variansi data asli. "
            f"Cuplikan nyata matriks ini dapat dilihat pada sub-tab \"📄 Tabel "
            f"Komponen\"."
        )

        st.divider()

        # ── Rumus PCA ───────────────────────────────────────────────────────
        st.markdown("**📐 Rumus Perhitungan PCA**")
        col_pf1, col_pf2 = st.columns(2)
        with col_pf1:
            st.latex(r"z_i = \frac{x_i - \mu}{\sigma}")
            st.caption("Standardisasi (StandardScaler): μ=rata-rata, σ=standar deviasi tiap fitur")
            st.latex(r"C = \frac{1}{n-1} Z^T Z")
            st.caption("Matriks kovariansi C dari data ternormalisasi Z (ukuran p×p, p=jumlah fitur)")
        with col_pf2:
            st.latex(r"C v_i = \lambda_i v_i")
            st.caption("Eigenvector vᵢ dan eigenvalue λᵢ dari matriks kovariansi C")
            st.latex(r"\text{Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{p} \lambda_j}")
            st.caption("Proporsi variansi yang dijelaskan komponen ke-i terhadap total variansi")

        st.divider()

        # ── Contoh Perhitungan Manual PCA dengan Data Asli ──────────────────────
        st.markdown("**🧮 Contoh Perhitungan PCA dengan Data Asli**")
        st.caption(
            "PCA pada aplikasi ini bekerja pada ratusan dimensi (fitur), sehingga "
            "tidak mungkin ditulis manual seluruhnya. Berikut contoh perhitungan "
            "PCA yang disederhanakan menjadi **2 fitur asli** dan **10 dokumen asli** "
            "dari data training, agar setiap langkah dapat ditelusuri secara manual. "
            "Prinsip matematikanya **identik** dengan PCA skala penuh (1.000 fitur) "
            "yang dijalankan aplikasi."
        )

        if feat:
            chi2_sel_ex = feat['chi2_sel']
            tfidf_ex    = feat['tfidf']
            fn_all_ex   = tfidf_ex.get_feature_names_out()
            fn_sel_ex   = fn_all_ex[chi2_sel_ex.get_support()]

            X_train_sm_ex = feat['X_train_sm']
            n_sample_docs = min(10, X_train_sm_ex.shape[0])
            X_sub_full    = X_train_sm_ex[:n_sample_docs].toarray()

            # Pilih 2 fitur dengan variansi tertinggi di subset ini agar contohnya informatif
            var_per_feat = X_sub_full.var(axis=0)
            top2_idx     = np.argsort(var_per_feat)[::-1][:2]
            fitur_a, fitur_b = fn_sel_ex[top2_idx[0]], fn_sel_ex[top2_idx[1]]
            X_2d = X_sub_full[:, top2_idx]  # shape (10, 2) — data ASLI

            df_contoh_pca = pd.DataFrame(
                X_2d, columns=[f'"{fitur_a}"', f'"{fitur_b}"'],
                index=[f'Dok-{i+1}' for i in range(n_sample_docs)]
            )
            st.markdown(f"**Langkah 0 — Data Asli (nilai TF-IDF Chi-Square, 2 fitur × {n_sample_docs} dokumen):**")
            st.dataframe(df_contoh_pca.style.format("{:.4f}"), use_container_width=True)

            # Langkah 1: Standardisasi
            mean_a, mean_b = X_2d[:, 0].mean(), X_2d[:, 1].mean()
            std_a, std_b   = X_2d[:, 0].std(ddof=0), X_2d[:, 1].std(ddof=0)
            Z = (X_2d - [mean_a, mean_b]) / [std_a if std_a > 0 else 1, std_b if std_b > 0 else 1]

            col_pex1, col_pex2 = st.columns(2)
            with col_pex1:
                st.markdown("**Langkah 1 — Standardisasi (Z-score):**")
                st.latex(rf'\mu_{{"{fitur_a}"}} = {mean_a:.4f}, \quad \sigma_{{"{fitur_a}"}} = {std_a:.4f}')
                st.latex(rf'\mu_{{"{fitur_b}"}} = {mean_b:.4f}, \quad \sigma_{{"{fitur_b}"}} = {std_b:.4f}')
                df_z = pd.DataFrame(Z, columns=[f'Z("{fitur_a}")', f'Z("{fitur_b}")'],
                                    index=[f'Dok-{i+1}' for i in range(n_sample_docs)])
                st.dataframe(df_z.style.format("{:.4f}"), use_container_width=True)

            # Langkah 2: Matriks kovariansi
            cov_matrix = np.cov(Z, rowvar=False, ddof=1)

            # Langkah 3: Eigenvalue & eigenvector (dihitung nyata dengan numpy)
            eigvals_ex, eigvecs_ex = np.linalg.eigh(cov_matrix)
            # urutkan dari terbesar ke terkecil
            order = np.argsort(eigvals_ex)[::-1]
            eigvals_ex = eigvals_ex[order]
            eigvecs_ex = eigvecs_ex[:, order]
            var_ratio_ex = eigvals_ex / eigvals_ex.sum()

            with col_pex2:
                st.markdown("**Langkah 2 — Matriks Kovariansi (2×2):**")
                st.latex(
                    rf"C = \begin{{bmatrix}} {cov_matrix[0,0]:.4f} & {cov_matrix[0,1]:.4f} \\ "
                    rf"{cov_matrix[1,0]:.4f} & {cov_matrix[1,1]:.4f} \end{{bmatrix}}"
                )
                st.markdown("**Langkah 3 — Eigenvalue (dihitung nyata dari data di atas):**")
                st.latex(rf"\lambda_1 = {eigvals_ex[0]:.4f}, \quad \lambda_2 = {eigvals_ex[1]:.4f}")
                st.markdown("**Langkah 4 — Variance Ratio tiap komponen:**")
                st.latex(
                    rf"\text{{Variance Ratio}}_1 = \frac{{{eigvals_ex[0]:.4f}}}{{{eigvals_ex[0]:.4f}+{eigvals_ex[1]:.4f}}} = {var_ratio_ex[0]*100:.2f}\%"
                )
                st.latex(rf"\text{{Variance Ratio}}_2 = {var_ratio_ex[1]*100:.2f}\%")

            st.success(
                f"✅ Pada contoh sederhana 2-fitur ini, **PC-1 sudah menjelaskan "
                f"{var_ratio_ex[0]*100:.2f}% variansi** dari kedua fitur "
                f'"{fitur_a}" dan "{fitur_b}". Pada skala penuh (menggunakan '
                f"seluruh {orig_dim:,} fitur Chi-Square dan {feat['X_train_sm'].shape[0]:,} "
                f"dokumen training), proses matematika yang **persis sama** "
                f"(standardisasi → kovariansi → eigenvalue/eigenvector → variance "
                f"ratio) menghasilkan **{n_comp:,} komponen** untuk mencapai "
                f"variansi kumulatif {var_exp*100:.2f}% (threshold 95%) — "
                f"lihat hasil aktualnya pada bagian berikut."
            )
        else:
            st.warning("Data training tidak tersedia untuk membuat contoh perhitungan.")

        st.divider()

        # ── Penjelasan Threshold ──────────────────────────────────────────────
        st.markdown("**❓ Apa itu Threshold pada PCA?**")
        st.info(
            "**Threshold** adalah nilai ambang batas variansi kumulatif (dalam %) "
            "yang ingin dipertahankan dari data asli. PCA akan terus menambahkan "
            "komponen (diurutkan dari eigenvalue terbesar) **sampai** variansi "
            "kumulatifnya mencapai atau melewati nilai threshold tersebut.\n\n"
            "**Formula penentuan jumlah komponen k:**"
        )
        st.latex(
            r"k = \min \left\{ k \in \mathbb{N} \;\Big|\; "
            r"\frac{\sum_{i=1}^{k} \lambda_i}{\sum_{j=1}^{p} \lambda_j} \geq \text{threshold} \right\}"
        )
        st.caption(
            "Artinya: cari jumlah komponen k **paling sedikit** sehingga variansi "
            "kumulatif (jumlah eigenvalue k komponen pertama dibagi total seluruh "
            "eigenvalue) sudah mencapai persentase threshold yang ditentukan."
        )

        col_th1, col_th2, col_th3 = st.columns(3)
        with col_th1:
            st.markdown("**Threshold 75%**")
            st.caption(
                "Mempertahankan 75% informasi data. Komponen yang dibutuhkan "
                "**paling sedikit** di antara ketiga threshold — reduksi dimensi "
                "paling agresif, cocok untuk efisiensi komputasi maksimal, namun "
                "berisiko kehilangan informasi lebih banyak."
            )
        with col_th2:
            st.markdown("**Threshold 95%**")
            st.caption(
                "Mempertahankan 95% informasi data. Nilai yang paling umum "
                "digunakan dalam penelitian karena memberikan **keseimbangan** "
                "antara reduksi dimensi dan retensi informasi."
            )
        with col_th3:
            st.markdown("**Threshold 99%**")
            st.caption(
                "Mempertahankan 99% informasi data. Membutuhkan komponen "
                "**paling banyak** di antara ketiganya — informasi paling utuh, "
                "namun reduksi dimensi paling sedikit (kompresi paling lemah)."
            )

        st.divider()

        # ── Perhitungan nyata: kenapa jadi sekian komponen ────────────────────
        st.markdown(f"**🔢 Perhitungan: Mengapa Menjadi {n_comp:,} Komponen (Threshold 95%)?**")
        st.info(
            f"**Dimensi awal (fitur Chi-Square):** {orig_dim:,}\n\n"
            f"**Langkah perhitungan:**\n\n"
            f"1. Setelah data distandardisasi dan dihitung matriks kovariansinya, "
            f"diperoleh **{orig_dim:,} eigenvalue** (satu untuk tiap fitur).\n\n"
            f"2. Eigenvalue diurutkan dari yang **terbesar ke terkecil**, lalu "
            f"dihitung variansi kumulatifnya satu per satu.\n\n"
            f"3. PCA berhenti menambahkan komponen tepat ketika variansi kumulatif "
            f"**pertama kali menyentuh atau melewati 95%** — pada data ini, hal itu "
            f"terjadi tepat di komponen ke-**{n_comp:,}**, dengan variansi kumulatif "
            f"aktual **{var_exp*100:.2f}%**.\n\n"
            f"4. Sisa **{orig_dim - n_comp:,} komponen** lainnya ({red_pct:.1f}% dari "
            f"total) dibuang karena kontribusinya terhadap variansi sudah sangat "
            f"kecil setelah threshold tercapai."
        )
        st.latex(
            rf"k = {n_comp:,} \quad \text{{karena}} \quad "
            rf"\sum_{{i=1}}^{{{n_comp}}} \text{{Variance Ratio}}_i = {var_exp*100:.2f}\% \geq 95\%"
        )
        st.caption(
            "Jumlah komponen ini **spesifik untuk rasio split dan data training "
            "yang sedang aktif** — bila rasio split diganti (ukuran data training "
            "berbeda), jumlah komponen dapat berbeda pula karena struktur "
            "variansi datanya juga berubah. Lihat sub-tab \"📊 Perbandingan Semua "
            "Rasio Split\" untuk detailnya."
        )

        st.divider()

        # ── Mengapa 1.000 fitur Chi-Square bisa jadi lebih sedikit komponen ────
        st.markdown(
            f"**❓ Mengapa {orig_dim:,} Fitur Chi-Square Bisa Direduksi Menjadi "
            f"Hanya {n_comp:,} Komponen PCA?**"
        )
        st.info(
            f"Timbul pertanyaan wajar: kalau ada {orig_dim:,} fitur hasil seleksi "
            f"Chi-Square, kenapa PCA bisa \"meringkasnya\" jadi {n_comp:,} komponen "
            f"saja tanpa kehilangan informasi berarti (tetap {var_exp*100:.2f}% "
            f"variansi)? Jawabannya: **banyak dari {orig_dim:,} fitur tersebut "
            f"saling berkorelasi (redundan)**, sehingga informasinya sebagian "
            f"tumpang tindih. PCA memanfaatkan redundansi ini — bukan membuang "
            f"informasi, melainkan **menggabungkan fitur-fitur yang bergerak "
            f"bersamaan** menjadi satu komponen baru."
        )

        # Hitung matriks korelasi nyata dari sebagian fitur, untuk ilustrasi konkret
        X_train_sm_corr = feat['X_train_sm'].toarray()
        # Ambil 3 fitur dengan variansi tertinggi sebagai contoh ilustrasi nyata
        var_per_f = X_train_sm_corr.var(axis=0)
        top3_var_idx = np.argsort(var_per_f)[::-1][:3]
        fn_chi = feat['chi2_sel'].get_support()
        fn_all_names = feat['tfidf'].get_feature_names_out()[fn_chi]
        contoh_3fitur = fn_all_names[top3_var_idx]
        corr_mat = np.corrcoef(X_train_sm_corr[:, top3_var_idx], rowvar=False)

        col_mek1, col_mek2 = st.columns(2)
        with col_mek1:
            st.markdown(f"**Contoh nyata — matriks korelasi 3 fitur dengan variansi tertinggi:**")
            corr_df = pd.DataFrame(
                corr_mat, columns=contoh_3fitur, index=contoh_3fitur
            )
            st.dataframe(corr_df.style.format("{:.4f}").background_gradient(
                cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)
            st.caption(
                "Nilai mendekati +1 atau −1 = dua fitur sangat berkorelasi (redundan). "
                "Nilai mendekati 0 = fitur relatif independen."
            )
        with col_mek2:
            st.markdown("**Ilustrasi sederhana (angka bulat, bukan data penelitian):**")
            st.latex(r"""
            Kov = \begin{bmatrix} 1{,}00 & 0{,}90 & 0{,}10 \\
            0{,}90 & 1{,}00 & 0{,}05 \\
            0{,}10 & 0{,}05 & 1{,}00 \end{bmatrix}
            """)
            st.caption(
                "Jika x1 & x2 berkorelasi tinggi (0,90) sementara x3 independen, "
                "dekomposisi eigen menghasilkan eigenvalue ≈ 1,90 / 1,00 / 0,10. "
                "Threshold 95% cukup diambil PC-1 + PC-2 (96,6% variansi) — "
                "3 fitur asli diringkas jadi 2 komponen, karena x1 & x2 yang "
                "redundan otomatis \"digabung\" secara linear menjadi satu arah "
                "(PC-1)."
            )

        st.markdown(
            f"Prinsip yang sama, dalam skala jauh lebih besar ({orig_dim:,} fitur, "
            f"dengan ratusan hingga ribuan pasangan kata yang cenderung muncul "
            f"bersamaan pada ulasan bersentimen sama, misalnya \"good\" dan "
            f"\"amazing\"), menjelaskan mengapa Chi-Square dan PCA sama-sama "
            f"berperan namun pada **sumbu yang berbeda**: Chi-Square memilih "
            f"*fitur mana* yang relevan terhadap sentimen (5.000 → 1.000), "
            f"sedangkan PCA memampatkan *redundansi antar-fitur terpilih* itu "
            f"menjadi komponen yang lebih sedikit namun tetap representatif "
            f"({orig_dim:,} → {n_comp:,})."
        )

        st.divider()

        # Metrik ringkas
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Dimensi Awal (Chi-Square)", f"{orig_dim:,}")
        p2.metric("Komponen PCA",               f"{n_comp:,}")
        p3.metric("Variance Explained",         f"{var_exp*100:.2f}%")
        p4.metric("Reduksi Dimensi",            f"{red_pct:.1f}%")

        st.divider()

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

        # Reduksi dimensi visual
        st.markdown("**🗜️ Visualisasi Reduksi Dimensi**")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        cats   = ['Dimensi Awal\n(Chi-Square)', 'Komponen PCA\n(variance 95%)']
        values = [orig_dim, n_comp]
        colors = ['#6c757d', '#4C72B0']
        bars   = ax3.bar(cats, values, color=colors, alpha=0.85,
                         width=0.4, edgecolor='white')
        for bar, v in zip(bars, values):
            ax3.text(bar.get_x()+bar.get_width()/2, v+orig_dim*0.01,
                     f'{v:,}', ha='center', fontweight='bold', fontsize=13)
        ax3.set_ylabel("Jumlah Dimensi")
        ax3.set_title(
            f"Reduksi Dimensi PCA\n"
            f"{orig_dim} → {n_comp} komponen ({red_pct:.1f}% berkurang)",
            fontweight='bold'
        )
        ax3.grid(axis='y', alpha=0.3)
        ax3.annotate('', xy=(1, n_comp+orig_dim*0.05),
                     xytext=(0, orig_dim*0.6),
                     arrowprops=dict(arrowstyle='->', color='#4C72B0', lw=2))
        ax3.text(0.5, (orig_dim+n_comp)/2,
                 f'−{red_pct:.1f}%', ha='center', color='#4C72B0',
                 fontsize=12, fontweight='bold')
        plt.tight_layout()
        col_v, _ = st.columns([1, 1])
        with col_v:
            st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Tabel Komponen
    # ══════════════════════════════════════════════════════════════════════
    with stab2:
        st.markdown(f"**Rasio split aktif: `{split_label}`**")

        thresh_choice = st.radio(
            "Pilih threshold variance PCA:",
            ["75%", "95%", "99%"],
            index=1, horizontal=True, key="tbl_komponen_thresh"
        )
        thresh_val = {"75%": 0.75, "95%": 0.95, "99%": 0.99}[thresh_choice]

        if thresh_choice == "95%":
            # Sudah tersedia dari model utama (r_pca) — tidak perlu hitung ulang
            evr_show    = evr
            cumvar_show = cumvar
            eigval_show = pca.explained_variance_
        else:
            # Hitung PCA ulang untuk threshold 75% atau 99% menggunakan data training yang sama
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA as _PCA

            with st.spinner(f"⏳ Menghitung PCA threshold {thresh_choice}..."):
                X_train_sm = feat['X_train_sm']
                X_tr_d  = X_train_sm.toarray()
                scaler_t = StandardScaler()
                X_tr_sc  = scaler_t.fit_transform(X_tr_d)
                pca_t = _PCA(n_components=thresh_val, random_state=42)
                pca_t.fit(X_tr_sc)

            evr_show    = pca_t.explained_variance_ratio_
            cumvar_show = np.cumsum(evr_show)
            eigval_show = pca_t.explained_variance_

        n_total_comp = len(evr_show)
        st.markdown(f"**📄 Tabel Explained Variance — Seluruh {n_total_comp} Komponen (Threshold {thresh_choice})**")
        st.caption(
            f"Menampilkan seluruh {n_total_comp} komponen PCA pada rasio split "
            f"{split_label} dengan threshold variance {thresh_choice}."
        )

        pca_tbl = pd.DataFrame({
            'Komponen':               [f"PC-{i+1}" for i in range(n_total_comp)],
            'Explained Variance (%)': [f"{v*100:.4f}" for v in evr_show],
            'Cumulative (%)':         [f"{c*100:.4f}" for c in cumvar_show],
            'Eigenvalue':             [f"{e:.4f}" for e in eigval_show],
        })

        st.dataframe(pca_tbl, use_container_width=True, hide_index=True, height=500)

        csv_bytes = pca_tbl.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Unduh Seluruh {n_total_comp} Komponen — Threshold {thresh_choice} (CSV)",
            data=csv_bytes,
            file_name=f"pca_komponen_lengkap_rasio_{split_label.replace(':','-')}_threshold_{thresh_choice.replace('%','')}.csv",
            mime="text/csv",
            key=f"dl_{thresh_choice}",
        )

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Perbandingan Semua Rasio Split
    # ══════════════════════════════════════════════════════════════════════
    with stab3:
        st.markdown("**Perbandingan jumlah fitur sebelum dan sesudah reduksi PCA "
                    "untuk semua rasio split**")
        st.caption("PCA dijalankan ulang untuk setiap rasio menggunakan data yang sama "
                   "(TF-IDF + Chi-Square). Parameter: n_components=0.95, random_state=42.")

        # Jalankan PCA semua split
        df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "default"
        with st.spinner("⏳ Menghitung PCA untuk semua rasio split..."):
            all_pca = pca_all_splits(df_hash, df)

        st.divider()

        # ── Tabel perbandingan ─────────────────────────────────────────────
        st.markdown("**📄 Tabel Perbandingan Reduksi Dimensi per Rasio Split**")

        tbl_rows = []
        for r in all_pca:
            is_active = "⭐" if r['label'] == split_label else ""
            tbl_rows.append({
                'Rasio Split':           r['label'] + (' ' + is_active if is_active else ''),
                'Data Training (SMOTE)': f"{r['n_train']:,}",
                'Data Testing':          f"{r['n_test']:,}",
                'Dimensi Awal':          f"{r['orig_dim']:,}",
                'Komponen PCA':          f"{r['n_comp']:,}",
                'Fitur Direduksi':       f"{r['orig_dim'] - r['n_comp']:,}",
                'Reduksi (%)':           f"{r['red_pct']:.2f}%",
                'Variance Explained':    f"{r['var_exp']*100:.2f}%",
            })
        tbl_df = pd.DataFrame(tbl_rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Tabel PCA SEBELUM vs SESUDAH SMOTE ──────────────────────────────
        st.markdown("**📄 Tabel Perbandingan Komponen PCA: Sebelum vs Sesudah SMOTE**")
        st.caption(
            "PCA (threshold 0.95) dihitung dua kali per rasio split: sekali pada "
            "data training SEBELUM SMOTE, dan sekali pada data training SESUDAH "
            "SMOTE. Ini menunjukkan pengaruh penambahan data sintetis SMOTE "
            "terhadap struktur variansi data dan jumlah komponen yang dibutuhkan."
        )

        tbl_smote_rows = []
        for r in all_pca:
            is_active = "⭐" if r['label'] == split_label else ""
            delta_comp = r['n_comp'] - r['n_comp_before_smote']
            tbl_smote_rows.append({
                'Rasio Split':              r['label'] + (' ' + is_active if is_active else ''),
                'n Training (Sebelum SMOTE)': f"{r['n_train_before_smote']:,}",
                'n Training (Sesudah SMOTE)': f"{r['n_train']:,}",
                'Komponen PCA (Sebelum SMOTE)': f"{r['n_comp_before_smote']:,}",
                'Komponen PCA (Sesudah SMOTE)': f"{r['n_comp']:,}",
                'Δ Komponen':                f"{delta_comp:+,}",
                'Var. Explained (Sebelum)':  f"{r['var_exp_before_smote']*100:.2f}%",
                'Var. Explained (Sesudah)':  f"{r['var_exp']*100:.2f}%",
                'Maks. Komponen Mungkin (Sebelum)': f"{r['max_possible_comp_before']:,}",
                'Maks. Komponen Mungkin (Sesudah)': f"{r['max_possible_comp_after']:,}",
            })
        st.dataframe(pd.DataFrame(tbl_smote_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Penjelasan: kenapa komponen berbeda meski dimensi awal & threshold sama ──
        st.markdown("**❓ Mengapa Jumlah Komponen PCA Berbeda di Setiap Rasio Split?**")
        st.info(
            "**Padahal dimensi awal sama-sama 1.000 fitur dan threshold sama-sama 0.95, "
            "kenapa jumlah komponen PCA yang dihasilkan bisa berbeda?**\n\n"
            "Ada dua alasan utama:\n\n"
            "**1. Jumlah komponen PCA dibatasi oleh jumlah SAMPEL, bukan hanya fitur.**\n"
            "Secara matematis, PCA maksimal hanya bisa menghasilkan komponen sebanyak "
            "`min(jumlah sampel, jumlah fitur)`. Jika jumlah sampel training pada suatu "
            "rasio (misal 70:30) lebih kecil dari 1.000 fitur, maka jumlah komponen PCA "
            "otomatis dibatasi oleh jumlah sampel tersebut — bukan oleh 1.000 fitur.\n\n"
            "**2. Struktur variansi data berbeda-beda tergantung ukuran & komposisi sampel.**\n"
            "Threshold 0.95 berarti *'ambil komponen sesedikit mungkin sampai variansi "
            "kumulatif mencapai 95%'*. Berapa banyak komponen yang dibutuhkan untuk "
            "mencapai itu bergantung pada seberapa 'tersebar' atau 'terkonsentrasi' "
            "variansi data pada sampel tersebut (dilihat dari eigenvalue matriks "
            "kovariansinya) — bukan semata jumlah fitur asalnya. Sampel yang lebih kecil "
            "atau komposisi SMOTE yang berbeda menghasilkan struktur korelasi antar fitur "
            "yang sedikit berbeda, sehingga jumlah komponen untuk mencapai 95% variansi "
            "pun ikut berbeda.\n\n"
            "**Kesimpulan:** Dimensi awal (1.000) hanyalah *batas atas* jumlah fitur yang "
            "tersedia untuk PCA — bukan jaminan jumlah komponen akhir. Jumlah komponen "
            "aktual selalu ≤ `min(n_sampel_training, 1.000)`, dan nilai pastinya "
            "ditentukan oleh struktur variansi data pada sampel training di rasio "
            "tersebut."
        )

        st.warning(
            "**❗ Apakah berarti Chi-Square tidak berpengaruh?** Tidak — Chi-Square "
            "tetap berpengaruh nyata, hanya saja pada **sumbu yang berbeda**.\n\n"
            "Bayangkan data training sebagai tabel: **baris** = jumlah sampel, "
            "**kolom** = jumlah fitur.\n\n"
            "- **Chi-Square mengontrol jumlah KOLOM** — memangkas dari 5.000 fitur "
            "TF-IDF menjadi 1.000 fitur, dan ini **konsisten sama di semua rasio "
            "split**. Tanpa Chi-Square (masih 5.000 fitur), jumlah komponen PCA yang "
            "mungkin dihasilkan justru akan jauh lebih besar lagi.\n\n"
            "- **PCA bergantung pada KOLOM *dan* BARIS** (rumus `min(sampel, fitur)`). "
            "Chi-Square sudah membatasi kolom menjadi 1.000, tapi jumlah **baris** "
            "(sampel training) ditentukan oleh rasio split & SMOTE — bukan oleh "
            "Chi-Square. Karena itulah jumlah komponen bisa berbeda antar rasio, "
            "meskipun jumlah kolom (fitur) sudah tetap 1.000.\n\n"
            "Singkatnya: **Chi-Square berhasil menekan batas atas dari 5.000 → 1.000**, "
            "namun tidak mengontrol berapa banyak sampel yang tersedia di setiap "
            "rasio split — sehingga PCA masih bisa dibatasi oleh jumlah sampel, "
            "bukan oleh hasil kerja Chi-Square."
        )

        st.divider()

        # ── Bar chart: komponen sebelum vs sesudah SMOTE ────────────────────
        st.markdown("**📊 Visualisasi: Komponen PCA Sebelum vs Sesudah SMOTE**")
        fig_sm, ax_sm = plt.subplots(figsize=(10, 5))
        labels_sm    = [r['label'] for r in all_pca]
        before_vals  = [r['n_comp_before_smote'] for r in all_pca]
        after_vals   = [r['n_comp'] for r in all_pca]
        x_sm = np.arange(len(labels_sm)); w_sm = 0.35
        b1 = ax_sm.bar(x_sm-w_sm/2, before_vals, w_sm, label='Sebelum SMOTE',
                       color='#E24B4A', alpha=0.85, edgecolor='white')
        b2 = ax_sm.bar(x_sm+w_sm/2, after_vals, w_sm, label='Sesudah SMOTE',
                       color='#4C72B0', alpha=0.85, edgecolor='white')
        for bar, v in zip(b1, before_vals):
            ax_sm.text(bar.get_x()+bar.get_width()/2, v+3,
                       f'{v}', ha='center', fontweight='bold', fontsize=10)
        for bar, v in zip(b2, after_vals):
            ax_sm.text(bar.get_x()+bar.get_width()/2, v+3,
                       f'{v}', ha='center', fontweight='bold', fontsize=10)
        ax_sm.set_xticks(x_sm); ax_sm.set_xticklabels(labels_sm, fontsize=11)
        ax_sm.set_ylabel('Jumlah Komponen PCA', fontsize=11)
        ax_sm.set_title('Jumlah Komponen PCA: Sebelum vs Sesudah SMOTE\n'
                        '(dimensi awal sama: 1.000 fitur, threshold sama: 0.95)',
                        fontweight='bold', fontsize=12)
        ax_sm.legend(fontsize=10); ax_sm.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_sm, use_container_width=True)
        plt.close()

        st.divider()

        # ── Bar chart: Dimensi Awal vs Komponen PCA ────────────────────────
        st.markdown("**📊 Jumlah Fitur: Sebelum vs Sesudah PCA per Rasio Split**")

        labels_r   = [r['label'] for r in all_pca]
        orig_vals  = [r['orig_dim'] for r in all_pca]
        comp_vals  = [r['n_comp']   for r in all_pca]
        red_vals   = [r['orig_dim'] - r['n_comp'] for r in all_pca]

        x = np.arange(len(labels_r))
        w = 0.35

        fig4, ax4 = plt.subplots(figsize=(11, 5))
        b1 = ax4.bar(x - w/2, orig_vals, w,
                     label='Sebelum PCA (Chi-Square)',
                     color='#6c757d', alpha=0.85, edgecolor='white')
        b2 = ax4.bar(x + w/2, comp_vals, w,
                     label='Sesudah PCA (Komponen)',
                     color='#4C72B0', alpha=0.85, edgecolor='white')

        for bar, v in zip(b1, orig_vals):
            ax4.text(bar.get_x()+bar.get_width()/2, v+8,
                     f'{v:,}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', color='#333')
        for bar, v in zip(b2, comp_vals):
            ax4.text(bar.get_x()+bar.get_width()/2, v+8,
                     f'{v:,}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', color='#4C72B0')

        ax4.set_xticks(x)
        ax4.set_xticklabels(labels_r, fontsize=12)
        ax4.set_ylabel("Jumlah Fitur / Komponen", fontsize=11)
        ax4.set_title("Jumlah Fitur Sebelum vs Sesudah Reduksi PCA\n"
                      "(semua rasio split, variance=0.95)",
                      fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.divider()

        # ── Stacked bar: komposisi dimensi ────────────────────────────────
        st.markdown("**📊 Komposisi Dimensi: Dipertahankan vs Direduksi**")

        fig5, ax5 = plt.subplots(figsize=(11, 5))
        b3 = ax5.bar(labels_r, comp_vals,
                     label='Dipertahankan (Komponen PCA)',
                     color='#4C72B0', alpha=0.85, edgecolor='white')
        b4 = ax5.bar(labels_r, red_vals, bottom=comp_vals,
                     label='Direduksi (Fitur dihapus)',
                     color='#E24B4A', alpha=0.70, edgecolor='white')

        for i, (cv, rv) in enumerate(zip(comp_vals, red_vals)):
            total = cv + rv
            ax5.text(i, cv/2, f'{cv}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white')
            ax5.text(i, cv + rv/2, f'{rv}', ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white')

        ax5.set_ylabel("Jumlah Fitur", fontsize=11)
        ax5.set_title("Komposisi Dimensi per Rasio Split\n"
                      "(biru = dipertahankan PCA, merah = direduksi)",
                      fontsize=13, fontweight='bold')
        ax5.legend(fontsize=11)
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        st.divider()

        # ── Line chart: % reduksi & variance explained ─────────────────────
        st.markdown("**📈 Persentase Reduksi & Variance Explained per Rasio Split**")

        var_vals = [r['var_exp']*100 for r in all_pca]
        red_pcts = [r['red_pct']     for r in all_pca]

        fig6, ax6 = plt.subplots(figsize=(10, 4.5))
        ax6b = ax6.twinx()

        l1, = ax6.plot(labels_r, red_pcts, 'o-',
                       color='#E24B4A', linewidth=2.5, markersize=8,
                       label='Reduksi Dimensi (%)')
        l2, = ax6b.plot(labels_r, var_vals, 's--',
                        color='#4C72B0', linewidth=2.5, markersize=8,
                        label='Variance Explained (%)')

        for i, (x_p, y_p) in enumerate(zip(labels_r, red_pcts)):
            ax6.annotate(f'{y_p:.1f}%', (x_p, y_p),
                         textcoords='offset points', xytext=(0, 10),
                         ha='center', fontsize=9, color='#E24B4A', fontweight='bold')
        for i, (x_p, y_p) in enumerate(zip(labels_r, var_vals)):
            ax6b.annotate(f'{y_p:.2f}%', (x_p, y_p),
                          textcoords='offset points', xytext=(0, -16),
                          ha='center', fontsize=9, color='#4C72B0', fontweight='bold')

        ax6.set_ylabel("Reduksi Dimensi (%)", fontsize=11, color='#E24B4A')
        ax6b.set_ylabel("Variance Explained (%)", fontsize=11, color='#4C72B0')
        ax6.set_xlabel("Rasio Split", fontsize=11)
        ax6.set_title("Persentase Reduksi Dimensi & Variance Explained\n"
                      "per Rasio Split", fontsize=13, fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='#E24B4A')
        ax6b.tick_params(axis='y', labelcolor='#4C72B0')
        lines  = [l1, l2]
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, fontsize=10, loc='lower right')
        ax6.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

        st.divider()

        # ── Kesimpulan otomatis ────────────────────────────────────────────
        st.markdown("**💡 Kesimpulan**")
        min_comp = min(all_pca, key=lambda r: r['n_comp'])
        max_comp = max(all_pca, key=lambda r: r['n_comp'])
        st.info(
            f"- Dimensi awal (Chi-Square) sama untuk semua rasio: **{all_pca[0]['orig_dim']:,} fitur**\n"
            f"- Komponen PCA terkecil: **{min_comp['n_comp']}** "
            f"(rasio {min_comp['label']}, reduksi {min_comp['red_pct']:.1f}%)\n"
            f"- Komponen PCA terbesar: **{max_comp['n_comp']}** "
            f"(rasio {max_comp['label']}, reduksi {max_comp['red_pct']:.1f}%)\n"
            f"- Semua rasio mempertahankan variance ≥ **{min(r['var_exp'] for r in all_pca)*100:.2f}%**\n"
            f"- Rasio aktif saat ini: **{split_label}** "
            f"→ {r_pca['pca_n']} komponen ({red_pct:.1f}% reduksi)"
        )


    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — Perbandingan Threshold PCA (Revisi Poin 6)
    # ══════════════════════════════════════════════════════════════════════
    with stab4:
        st.markdown("**🎚️ Penjelasan Threshold PCA dan Perbandingan Antar Nilai**")
        st.caption(
            "Revisi dosen penguji poin 6: jelaskan berapa komponen PCA yang digunakan, "
            "arti threshold, dan perbandingan jika menggunakan angka threshold yang berbeda."
        )

        if 'feat' not in st.session_state:
            st.info("Jalankan pipeline terlebih dahulu.", icon="👈")
        else:
            feat = st.session_state['feat']
            df_hash_red = str(hash(tuple(df['content'].tolist()))) if df is not None else "default"

            # ── Tabel Ringkas: Hasil Reduksi Dimensi 3 Threshold × 3 Rasio ───────
            st.markdown("**📊 Hasil Reduksi Dimensi dengan PCA pada Tiga Threshold Variansi**")
            st.caption(
                "Tabel ringkas jumlah komponen, variansi kumulatif, dan persentase "
                "reduksi dimensi untuk threshold 75%, 95%, dan 99%, pada ketiga rasio "
                "split data (70:30, 80:20, 90:10). Perhitungan HANYA mencakup PCA "
                "(tanpa pelatihan SVM), sehingga cepat dijalankan."
            )

            with st.spinner("⏳ Menghitung reduksi dimensi PCA untuk 3 threshold × 3 rasio..."):
                red_result = pca_reduction_by_threshold(df_hash_red, df)

            df_red = pd.DataFrame(red_result['rows'])
            # Urutkan rasio sesuai urutan baku
            urutan_rasio = {'70:30': 0, '80:20': 1, '90:10': 2}
            df_red['_sort'] = df_red['Rasio Split'].map(urutan_rasio)
            df_red = df_red.sort_values(['_sort', 'Threshold']).drop(columns='_sort').reset_index(drop=True)

            st.dataframe(df_red, use_container_width=True, hide_index=True)

            csv_red = df_red.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh Hasil Reduksi Dimensi PCA (CSV)",
                data=csv_red,
                file_name="perbandingan_threshold_pca.csv",
                mime="text/csv",
                key="dl_pca_reduction_threshold",
            )
            st.caption(
                f"Dimensi awal (fitur hasil Chi-Square): **{red_result['orig_dim']:,}**. "
                f"Tabel di atas siap diunduh sebagai CSV — pastikan tombol unduh "
                f"benar-benar diklik dan file tersimpan sebelum digunakan pada "
                f"dokumen skripsi, agar tidak menghasilkan berkas kosong."
            )

            # ── Grafik: jumlah komponen per threshold × rasio ────────────────────
            fig_red, ax_red = plt.subplots(figsize=(9, 4.5))
            x_pos = np.arange(len(red_result['ratios']))
            width_red = 0.25
            colors_thresh = {'75%':'#4C72B0','95%':'#E24B4A','99%':'#55A868'}
            for i, thresh_label in enumerate(['75%','95%','99%']):
                vals = [
                    df_red[(df_red['Rasio Split']==r) & (df_red['Threshold']==thresh_label)]['Komponen PCA (k)'].values[0]
                    for r in red_result['ratios']
                ]
                bars = ax_red.bar(x_pos + (i-1)*width_red, vals, width_red,
                                  label=thresh_label, color=colors_thresh[thresh_label],
                                  alpha=0.87, edgecolor='white')
                for bar, v in zip(bars, vals):
                    ax_red.text(bar.get_x()+bar.get_width()/2, v+5, str(v),
                               ha='center', fontsize=8, fontweight='bold')
            ax_red.set_xticks(x_pos)
            ax_red.set_xticklabels(red_result['ratios'])
            ax_red.set_xlabel('Rasio Split', fontsize=11)
            ax_red.set_ylabel('Jumlah Komponen PCA (k)', fontsize=11)
            ax_red.set_title('Jumlah Komponen PCA per Threshold × Rasio Split\n(Data Asli, Tanpa SVM)',
                             fontsize=12, fontweight='bold')
            ax_red.legend(title='Threshold', fontsize=9)
            ax_red.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_red, use_container_width=True)
            plt.close()

            st.divider()
            st.markdown("**📖 Apa Arti n_components = 0.95?**")
            col_ex1, col_ex2 = st.columns(2)
            with col_ex1:
                st.info(
                    "**n_components = 0.95 artinya:**\n\n"
                    "PCA secara otomatis memilih jumlah komponen "
                    "minimum yang diperlukan untuk mempertahankan "
                    "**minimal 95% variansi (informasi) total** dari data asli.\n\n"
                    "Jadi bukan jumlah komponen yang ditentukan manual, "
                    "melainkan PCA yang menentukan sendiri berapa komponen "
                    "yang dibutuhkan agar tidak kehilangan lebih dari 5% informasi."
                )
            with col_ex2:
                st.info(
                    "**Implikasi threshold 0.95:**\n\n"
                    "- Threshold **lebih tinggi** (misal 0.99) → lebih banyak komponen → "
                    "lebih banyak informasi dipertahankan → dimensi lebih besar\n\n"
                    "- Threshold **lebih rendah** (misal 0.80) → lebih sedikit komponen → "
                    "kompresi lebih agresif → dimensi lebih kecil\n\n"
                    "- Threshold **0.95** dipilih sebagai keseimbangan antara "
                    "retensi informasi dan efisiensi komputasi"
                )

            st.divider()

            # ── Hitung PCA untuk berbagai threshold ───────────────────────────
            st.markdown("**📊 Perbandingan Hasil PCA pada Berbagai Nilai Threshold**")

            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            from imblearn.over_sampling import SMOTE
            from sklearn.svm import SVC
            from sklearn.metrics import accuracy_score, f1_score

            X_train_sm = feat['X_train_sm']
            X_test     = feat['X_test']
            y_train    = feat['y_train_sm']
            y_test     = feat['y_test']

            X_tr_d = X_train_sm.toarray()
            X_te_d = X_test.toarray()
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_d)
            X_te_sc = scaler.transform(X_te_d)

            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
            threshold_results = []

            with st.spinner("⏳ Menghitung PCA untuk semua nilai threshold..."):
                for thresh in thresholds:
                    pca_t = PCA(n_components=thresh, random_state=42)
                    X_tr_pca = pca_t.fit_transform(X_tr_sc)
                    X_te_pca = pca_t.transform(X_te_sc)
                    n_comp   = int(pca_t.n_components_)
                    var_exp  = float(pca_t.explained_variance_ratio_.sum())
                    red_pct  = (1 - n_comp / X_tr_d.shape[1]) * 100

                    # Train SVM RBF cepat untuk melihat pengaruh threshold
                    svm_t = SVC(kernel='rbf', C=1, gamma=0.001,
                                random_state=42, probability=False)
                    svm_t.fit(X_tr_pca, y_train)
                    y_pred = svm_t.predict(X_te_pca)

                    threshold_results.append({
                        'Threshold': thresh,
                        'Komponen PCA': n_comp,
                        'Variance Explained (%)': f"{var_exp*100:.2f}%",
                        'Reduksi Dimensi (%)': f"{red_pct:.1f}%",
                        'Accuracy (SVM RBF)': f"{accuracy_score(y_test, y_pred)*100:.2f}%",
                        'F1-Score (SVM RBF)': f"{f1_score(y_test, y_pred, average='binary', zero_division=0):.4f}",
                        'Aktif': "⭐ (digunakan)" if thresh == 0.95 else "",
                    })

            thresh_df = pd.DataFrame(threshold_results)
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Visualisasi perbandingan threshold ────────────────────────────
            st.markdown("**📈 Visualisasi Pengaruh Threshold terhadap Komponen & Performa**")
            n_comps  = [r['Komponen PCA'] for r in threshold_results]
            accs     = [float(r['Accuracy (SVM RBF)'].replace('%',''))/100
                        for r in threshold_results]
            f1s      = [float(r['F1-Score (SVM RBF)'])
                        for r in threshold_results]
            thresh_lbls = [str(t) for t in thresholds]

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle('Pengaruh Nilai Threshold PCA terhadap Komponen dan Performa SVM RBF',
                         fontsize=13, fontweight='bold')

            # Plot 1: Jumlah komponen
            bars = axes[0].bar(thresh_lbls, n_comps, color=[
                '#E24B4A' if t == 0.95 else '#4C72B0' for t in thresholds
            ], alpha=0.85, edgecolor='white')
            for bar, v in zip(bars, n_comps):
                axes[0].text(bar.get_x()+bar.get_width()/2, v+2,
                             str(v), ha='center', fontsize=9, fontweight='bold')
            axes[0].set_xlabel('Threshold'); axes[0].set_ylabel('Jumlah Komponen PCA')
            axes[0].set_title('Jumlah Komponen per Threshold', fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)

            # Plot 2: Accuracy
            axes[1].plot(thresh_lbls, [a*100 for a in accs], 'o-',
                         color='#55A868', linewidth=2.5, markersize=8)
            axes[1].axvline(x=thresh_lbls.index(str(0.95)),
                            color='red', linestyle='--', linewidth=1.5,
                            label='Threshold aktif (0.95)')
            for i, (x, v) in enumerate(zip(thresh_lbls, accs)):
                axes[1].annotate(f'{v*100:.2f}%', (x, v*100),
                                 textcoords='offset points', xytext=(0,8),
                                 ha='center', fontsize=8.5, fontweight='bold')
            axes[1].set_xlabel('Threshold'); axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Accuracy SVM RBF per Threshold', fontweight='bold')
            axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

            # Plot 3: F1-Score
            axes[2].plot(thresh_lbls, f1s, 's-',
                         color='#E24B4A', linewidth=2.5, markersize=8)
            axes[2].axvline(x=thresh_lbls.index(str(0.95)),
                            color='red', linestyle='--', linewidth=1.5,
                            label='Threshold aktif (0.95)')
            for i, (x, v) in enumerate(zip(thresh_lbls, f1s)):
                axes[2].annotate(f'{v:.4f}', (x, v),
                                 textcoords='offset points', xytext=(0,8),
                                 ha='center', fontsize=8.5, fontweight='bold')
            axes[2].set_xlabel('Threshold'); axes[2].set_ylabel('F1-Score')
            axes[2].set_title('F1-Score SVM RBF per Threshold', fontweight='bold')
            axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()

            # ── Kesimpulan ────────────────────────────────────────────────────
            best_acc_idx = max(range(len(accs)), key=lambda i: accs[i])
            best_f1_idx  = max(range(len(f1s)),  key=lambda i: f1s[i])
            st.info(
                f"**Kesimpulan Perbandingan Threshold:**\n\n"
                f"- Threshold **0.95** (digunakan) → **{n_comps[thresholds.index(0.95)]} komponen**, "
                f"Accuracy {accs[thresholds.index(0.95)]*100:.2f}%, "
                f"F1 {f1s[thresholds.index(0.95)]:.4f}\n\n"
                f"- Accuracy terbaik pada threshold **{thresholds[best_acc_idx]}** "
                f"({accs[best_acc_idx]*100:.2f}%, {n_comps[best_acc_idx]} komponen)\n\n"
                f"- F1 terbaik pada threshold **{thresholds[best_f1_idx]}** "
                f"({f1s[best_f1_idx]:.4f}, {n_comps[best_f1_idx]} komponen)\n\n"
                f"- Threshold **0.95 dipilih** karena memberikan keseimbangan optimal antara "
                f"reduksi dimensi yang signifikan dan performa model yang kompetitif."
            )


    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 5 — Arah & Interpretasi Komponen PCA
    # ══════════════════════════════════════════════════════════════════════
    with stab5:
        st.markdown("**🧭 Interpretasi Arah Komponen PCA — PC mengarah ke mana?**")
        st.caption(
            "Setiap Principal Component (PC) adalah kombinasi linear dari fitur asli. "
            "Loading terbesar (positif/negatif) menunjukkan fitur mana yang paling "
            "berkontribusi pada arah komponen tersebut."
        )

        if 'feat' not in st.session_state:
            st.info("Jalankan pipeline terlebih dahulu.", icon="👈")
        else:
            feat     = st.session_state['feat']
            pca_obj  = r_pca['pca']
            chi2_sel = feat['chi2_sel']
            tfidf    = feat['tfidf']

            # Dapatkan nama fitur yang dipilih Chi-Square
            fn_all       = tfidf.get_feature_names_out()
            selected_mask= chi2_sel.get_support()
            fn_selected  = fn_all[selected_mask]

            # Komponen PCA (loadings)
            components   = pca_obj.components_   # shape: (n_comp, n_features_chi2)
            evr_vals     = pca_obj.explained_variance_ratio_
            n_pc         = pca_obj.n_components_

            # ── Pilihan jumlah PC yang ditampilkan ────────────────────────────
            n_show = st.slider(
                "Tampilkan berapa PC pertama?",
                min_value=1, max_value=min(20, n_pc), value=min(10, n_pc),
                key="pc_n_show"
            )
            n_top_features = st.slider(
                "Tampilkan berapa fitur teratas per PC?",
                min_value=3, max_value=20, value=10,
                key="pc_n_feat"
            )

            st.divider()

            # ── Tabel loading teratas per PC ──────────────────────────────────
            st.markdown(f"**📄 Tabel Loading Teratas {n_top_features} Fitur per PC (PC1–PC{n_show})**")
            st.caption(
                "Loading positif (+): fitur mendorong ke arah positif PC tersebut. "
                "Loading negatif (−): fitur mendorong ke arah negatif PC tersebut. "
                "Nilai absolut terbesar = kontribusi terbesar."
            )

            tbl_rows = []
            for pc_idx in range(n_show):
                loadings   = components[pc_idx]
                top_pos_idx= np.argsort(loadings)[::-1][:n_top_features]
                top_neg_idx= np.argsort(loadings)[:n_top_features]

                top_pos_words  = fn_selected[top_pos_idx]
                top_pos_vals   = loadings[top_pos_idx]
                top_neg_words  = fn_selected[top_neg_idx]
                top_neg_vals   = loadings[top_neg_idx]

                # Interpretasi sentimen
                pos_sentiment = "Positif" if any(w in ['good','great','love','amazing',
                    'best','beautiful','fun','enjoy','perfect','awesome','excellent']
                    for w in top_pos_words) else "Campuran"
                neg_sentiment = "Negatif" if any(w in ['bad','boring','hate','waste',
                    'terrible','awful','broken','fail','disappoint','crash','unfair']
                    for w in top_pos_words) else "Campuran"

                tbl_rows.append({
                    f'PC-{pc_idx+1}':        f"({evr_vals[pc_idx]*100:.2f}% var)",
                    'Top Fitur (+)':          ', '.join([f"{w}({v:+.3f})"
                                               for w,v in zip(top_pos_words[:5], top_pos_vals[:5])]),
                    'Top Fitur (−)':          ', '.join([f"{w}({v:+.3f})"
                                               for w,v in zip(top_neg_words[:5], top_neg_vals[:5])]),
                    'Var Explained':          f"{evr_vals[pc_idx]*100:.4f}%",
                    'Kumulatif':              f"{evr_vals[:pc_idx+1].sum()*100:.2f}%",
                })

            tbl_df = pd.DataFrame(tbl_rows)
            st.dataframe(tbl_df, use_container_width=True, hide_index=True)

            st.divider()

            # ── Detail per PC dengan bar chart ────────────────────────────────
            st.markdown(f"**📊 Detail Loading per PC — Bar Chart**")
            pc_sel = st.selectbox(
                "Pilih komponen PC:",
                [f"PC-{i+1} ({evr_vals[i]*100:.2f}%)" for i in range(n_show)],
                key="pc_detail_sel"
            )
            pc_idx_sel = int(pc_sel.split('-')[1].split(' ')[0]) - 1
            loadings_sel = components[pc_idx_sel]

            # Top positive + top negative
            top_pos_i = np.argsort(loadings_sel)[::-1][:n_top_features]
            top_neg_i = np.argsort(loadings_sel)[:n_top_features]

            # Gabungkan dan urutkan
            combined_idx  = np.concatenate([top_pos_i, top_neg_i])
            combined_idx  = np.unique(combined_idx)
            combined_load = loadings_sel[combined_idx]
            combined_words= fn_selected[combined_idx]
            sort_order    = np.argsort(combined_load)
            combined_load = combined_load[sort_order]
            combined_words= combined_words[sort_order]

            fig_pc, ax_pc = plt.subplots(figsize=(12, max(5, len(combined_words)*0.4)))
            colors_bar = ['#E24B4A' if v < 0 else '#4C72B0' for v in combined_load]
            bars = ax_pc.barh(combined_words, combined_load,
                              color=colors_bar, alpha=0.85, edgecolor='white')
            ax_pc.axvline(x=0, color='black', linewidth=1)
            for bar, v in zip(bars, combined_load):
                xpos = v + (0.001 if v >= 0 else -0.001)
                ha   = 'left' if v >= 0 else 'right'
                ax_pc.text(xpos, bar.get_y()+bar.get_height()/2,
                           f'{v:+.4f}', va='center', ha=ha,
                           fontsize=9, fontweight='bold')
            ax_pc.set_xlabel('Loading (kontribusi terhadap PC)', fontsize=11)
            ax_pc.set_title(
                f'{pc_sel} — Top Fitur Loading\n'
                'Biru (+) = mendorong ke arah positif PC | '
                'Merah (\u2212) = mendorong ke arah negatif PC',
                fontsize=12, fontweight='bold'
            )
            ax_pc.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_pc, use_container_width=True)
            plt.close()

            st.divider()

            # ── Heatmap loading PC1–PC10 vs top fitur ────────────────────────
            st.markdown(f"**🔥 Heatmap Loading — PC1 s.d. PC{min(n_show,10)} vs Top Fitur**")
            n_hm_pc   = min(n_show, 10)
            n_hm_feat = min(n_top_features, 20)

            # Pilih fitur yang sering muncul di top loading
            loading_matrix = components[:n_hm_pc]
            # Ambil top fitur berdasarkan absolute loading tertinggi
            abs_sum   = np.abs(loading_matrix).sum(axis=0)
            top_f_idx = np.argsort(abs_sum)[::-1][:n_hm_feat]
            top_f_words = fn_selected[top_f_idx]

            hm_data = pd.DataFrame(
                loading_matrix[:, top_f_idx],
                index=[f"PC-{i+1} ({evr_vals[i]*100:.1f}%)" for i in range(n_hm_pc)],
                columns=top_f_words,
            )
            fig_hm, ax_hm = plt.subplots(figsize=(max(12, n_hm_feat*0.7), n_hm_pc*0.7+2))
            import seaborn as sns
            sns.heatmap(
                hm_data, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, ax=ax_hm,
                linewidths=0.3, linecolor='white',
                annot_kws={'size': 8},
                cbar_kws={'label': 'Loading', 'shrink': 0.6},
            )
            ax_hm.set_title(
                f'Heatmap Loading PCA — PC1 s.d. PC{n_hm_pc} vs {n_hm_feat} Fitur Teratas\n'
                '(Merah = kontribusi positif kuat | Biru = kontribusi negatif kuat)',
                fontsize=12, fontweight='bold'
            )
            ax_hm.set_xlabel('Fitur (kata)', fontsize=10)
            ax_hm.set_ylabel('Komponen PCA', fontsize=10)
            ax_hm.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig_hm, use_container_width=True)
            plt.close()

            st.divider()

            # ── Interpretasi naratif ──────────────────────────────────────────
            st.markdown("**📖 Interpretasi Naratif PC1 & PC2**")
            for pc_i in range(min(3, n_pc)):
                load_i    = components[pc_i]
                top_p_idx = np.argsort(load_i)[::-1][:5]
                top_n_idx = np.argsort(load_i)[:5]
                pos_words = list(fn_selected[top_p_idx])
                neg_words = list(fn_selected[top_n_idx])
                pos_vals  = load_i[top_p_idx]
                neg_vals  = load_i[top_n_idx]

                with st.expander(
                    f"📌 PC-{pc_i+1} — Variance Explained: "
                    f"{evr_vals[pc_i]*100:.4f}% | Kumulatif: "
                    f"{evr_vals[:pc_i+1].sum()*100:.2f}%"
                ):
                    col_i1, col_i2 = st.columns(2)
                    with col_i1:
                        st.markdown("**Arah Positif (+) — Fitur dominan:**")
                        for w, v in zip(pos_words, pos_vals):
                            st.markdown(f"- `{w}` → loading **{v:+.4f}**")
                        # Tebak tema
                        pos_sentiment_words = [w for w in pos_words
                                               if w in ['good','great','love','amazing',
                                                        'best','beautiful','fun','enjoy',
                                                        'perfect','awesome','excellent',
                                                        'recommend','nice','wonderful']]
                        if pos_sentiment_words:
                            st.success(
                                f"**Interpretasi:** PC-{pc_i+1} arah positif cenderung "
                                f"merepresentasikan **sentimen positif** "
                                f"(kata: {', '.join(pos_sentiment_words)})"
                            )
                        else:
                            st.info(
                                f"**Interpretasi:** PC-{pc_i+1} arah positif didominasi "
                                f"oleh kata: {', '.join(pos_words[:3])}"
                            )

                    with col_i2:
                        st.markdown("**Arah Negatif (−) — Fitur dominan:**")
                        for w, v in zip(neg_words, neg_vals):
                            st.markdown(f"- `{w}` → loading **{v:+.4f}**")
                        neg_sentiment_words = [w for w in neg_words
                                               if w in ['bad','boring','hate','waste',
                                                        'terrible','awful','broken',
                                                        'fail','disappoint','crash',
                                                        'unfair','worst','bug','lag']]
                        if neg_sentiment_words:
                            st.error(
                                f"**Interpretasi:** PC-{pc_i+1} arah negatif cenderung "
                                f"merepresentasikan **sentimen negatif** "
                                f"(kata: {', '.join(neg_sentiment_words)})"
                            )
                        else:
                            st.info(
                                f"**Interpretasi:** PC-{pc_i+1} arah negatif didominasi "
                                f"oleh kata: {', '.join(neg_words[:3])}"
                            )

            st.divider()
            st.info(
                '**Catatan Penting:**\n\n'
                'PCA tidak langsung menghasilkan komponen yang bermakna sentimen positif '
                'atau negatif — PCA hanya memaksimalkan variansi. Namun, karena data '
                'mengandung ulasan dengan sentimen berbeda, komponen-komponen awal (PC1, PC2, ...) '
                'cenderung menangkap pola perbedaan kosa kata antar sentimen. '
                'Loading positif besar pada kata-kata sentimen tertentu mengindikasikan bahwa '
                'PC tersebut membedakan dokumen berdasarkan kehadiran kata-kata tersebut.'
            )
