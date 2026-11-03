"""
tabs/tab_matriks_lengkap.py
Tab: Matriks Evaluasi Lengkap (Blueprint Tabel Skripsi — Data Asli)
Menjalankan skenario lengkap: 3 rasio split x 4 arsitektur x 4 kernel x
grid parameter (C, gamma, threshold PCA) menggunakan DATA ASLI dari model,
mengikuti struktur tabel kosong (blueprint) yang dirancang di Bab III.
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from models.pipeline import full_evaluation_matrix, run_10fold_cv_validation, svm_prediction_detail_8020, predict_best_model

RATIOS  = ['90:10', '80:20', '70:30']
KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
KERNEL_LABEL  = {'linear':'Linear','rbf':'RBF','poly':'Polynomial','sigmoid':'Sigmoid'}
KERNEL_LETTER = {'linear':'A','rbf':'B','poly':'C','sigmoid':'D'}
ARCHS = ['Tanpa PCA + Sebelum SMOTE', 'Tanpa PCA + Setelah SMOTE',
         'Dengan PCA + Sebelum SMOTE', 'Dengan PCA + Setelah SMOTE']


def _rows_to_df(rows, is_pca, orig_dim=1000, feature_label="Fitur Chi-Square"):
    """Ubah list of dict hasil evaluasi menjadi DataFrame siap tampil."""
    data = []
    for i, r in enumerate(rows, 1):
        row = {'No': i}
        if is_pca:
            row['% PCA'] = f"{r['pct']}%"
            row['Komponen Hasil Reduksi'] = f"{r['n_comp']} PC (dari {orig_dim} {feature_label})"
        else:
            row['Komponen Fitur'] = f"{orig_dim} {feature_label}"
        row['Parameter C'] = r['C']
        if r.get('gamma') is not None:
            row['Parameter γ'] = r['gamma']
        row['Accuracy (%)']  = f"{r['acc']*100:.2f}"
        row['Precision (%)'] = f"{r['prec']*100:.2f}"
        row['Recall (%)']    = f"{r['rec']*100:.2f}"
        row['F1-Score (%)']  = f"{r['f1']*100:.2f}"
        data.append(row)
    return pd.DataFrame(data)


def _highlight_best(df, metric_col='F1-Score (%)'):
    """Styler yang menyorot baris dengan F1-Score tertinggi."""
    def _style(row):
        is_best = row.name == df[metric_col].astype(float).idxmax()
        return ['background-color: #EAF3DE; font-weight: bold' if is_best else '' for _ in row]
    return df.style.apply(_style, axis=1)


def _compute_best_overall(result):
    """Menghitung kombinasi dengan F1-Score tertinggi dari suatu hasil matriks evaluasi."""
    results = result['results']
    best_overall = None
    for ratio in RATIOS:
        for arch in ARCHS:
            arch_data = results[ratio][arch]
            for kernel in KERNELS:
                for r in arch_data['rows'][kernel]:
                    cand = {
                        'ratio': ratio, 'arch': arch, 'kernel': kernel,
                        'C': r['C'], 'gamma': r.get('gamma'),
                        'pct': r.get('pct'), 'n_comp': r.get('n_comp'),
                        'acc': r['acc'], 'prec': r['prec'],
                        'rec': r['rec'], 'f1': r['f1'],
                    }
                    if best_overall is None or cand['f1'] > best_overall['f1']:
                        best_overall = cand
    return best_overall


def _render_matrix_results(result, key_prefix, feature_label, df=None):
    """Menampilkan hasil matriks evaluasi (720 kombinasi) — dipakai untuk
    skenario Dengan Chi-Square maupun Tanpa Chi-Square (fitur TF-IDF penuh)."""
    results  = result['results']
    orig_dim = result['orig_dim']

    # ── Ringkasan global: kombinasi terbaik ───────────────────────────────
    st.divider()
    st.markdown("**🏆 Kombinasi Terbaik Secara Keseluruhan (berdasarkan F1-Score)**")

    b = _compute_best_overall(result)
    m1,m2,m3,m4,m5,m6 = st.columns(6)
    m1.metric("Rasio",     b['ratio'])
    m2.metric("Arsitektur",b['arch'].replace(' + ', '\n+ '))
    m3.metric("Kernel",    KERNEL_LABEL[b['kernel']])
    m4.metric("C",         b['C'])
    m5.metric("Accuracy",  f"{b['acc']*100:.2f}%")
    m6.metric("F1-Score",  f"{b['f1']*100:.2f}%")
    if b.get('gamma') is not None:
        st.caption(f"gamma = {b['gamma']}" + (f" | % PCA = {b['pct']}% ({b['n_comp']} komponen)" if b.get('pct') else ""))

    st.divider()

    # ── Navigasi: Arsitektur → Rasio → Kernel ────────────────────────────
    ARCH_NUMBERED = [
        "1. SVM Tanpa PCA Sebelum SMOTE",
        "2. SVM Tanpa PCA Setelah SMOTE",
        "3. SVM Dengan PCA + Sebelum SMOTE",
        "4. SVM Dengan PCA + Setelah SMOTE",
    ]
    ARCH_DESC = {
        "1. SVM Tanpa PCA Sebelum SMOTE":    f"Pelatihan pada {orig_dim:,} {feature_label} asli",
        "2. SVM Tanpa PCA Setelah SMOTE":    f"Pelatihan pada {orig_dim:,} {feature_label} pasca-SMOTE",
        "3. SVM Dengan PCA + Sebelum SMOTE": "Pelatihan pada komponen PC sebelum SMOTE",
        "4. SVM Dengan PCA + Setelah SMOTE": "Pelatihan pada komponen PC pasca-SMOTE",
    }
    arch_tabs = st.tabs(ARCH_NUMBERED)

    for arch_num, arch, arch_tab in zip(ARCH_NUMBERED, ARCHS, arch_tabs):
        with arch_tab:
            st.markdown(f"### {arch_num}")
            st.caption(ARCH_DESC[arch_num])
            is_pca = 'Dengan PCA' in arch

            ratio_tabs = st.tabs([f"📊 Rasio {r}" for r in RATIOS])
            for ratio, ratio_tab in zip(RATIOS, ratio_tabs):
                with ratio_tab:
                    arch_data = results[ratio][arch]

                    st.caption(
                        f"Data training: {arch_data['n_train']:,} sampel | "
                        f"Data testing: {arch_data['n_test']:,} sampel | "
                        f"Dimensi awal: {orig_dim:,} {feature_label}"
                    )

                    for kernel in KERNELS:
                        letter = KERNEL_LETTER[kernel]
                        klabel = KERNEL_LABEL[kernel]
                        with st.expander(f"**{letter}. Kernel {klabel}**", expanded=(kernel=='rbf')):
                            rows = arch_data['rows'][kernel]
                            cell_label = "Fitur Chi-Square" if "Chi-Square" in feature_label else "Fitur TF-IDF"
                            table_df = _rows_to_df(rows, is_pca, orig_dim, cell_label)
                            st.dataframe(
                                _highlight_best(table_df),
                                use_container_width=True, hide_index=True
                            )
                            best_idx = table_df['F1-Score (%)'].astype(float).idxmax()
                            best_row = table_df.loc[best_idx]
                            st.caption(
                                f"🏆 Baris terbaik: No {int(best_row['No'])} — "
                                f"F1-Score = {best_row['F1-Score (%)']}%, "
                                f"Accuracy = {best_row['Accuracy (%)']}%"
                            )

    st.divider()

    # ── Grafik ringkasan: F1 terbaik per kernel per rasio ─────────────────
    st.markdown("**📊 F1-Score Terbaik per Kernel — Semua Rasio (gabungan semua arsitektur)**")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(RATIOS)); w = 0.18
    colors = {'linear':'#4C72B0','rbf':'#55A868','poly':'#E24B4A','sigmoid':'#F5A623'}
    for ki, kernel in enumerate(KERNELS):
        vals = []
        for ratio in RATIOS:
            best_f1 = 0
            for arch in ARCHS:
                for r in results[ratio][arch]['rows'][kernel]:
                    best_f1 = max(best_f1, r['f1'])
            vals.append(best_f1)
        bars = ax.bar(x + (ki-1.5)*w, vals, w, label=KERNEL_LABEL[kernel],
                     color=colors[kernel], alpha=0.87, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                   f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(RATIOS, fontsize=11)
    ax.set_ylabel('F1-Score Terbaik', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'F1-Score Terbaik per Kernel per Rasio ({feature_label})\n(dari semua arsitektur & parameter)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    # ── Grafik ringkasan: Accuracy terbaik per kernel per rasio ────────────
    st.markdown("**📊 Accuracy Terbaik per Kernel — Semua Rasio (gabungan semua arsitektur)**")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(RATIOS)); w = 0.18
    for ki, kernel in enumerate(KERNELS):
        vals = []
        for ratio in RATIOS:
            best_acc = 0
            for arch in ARCHS:
                for r in results[ratio][arch]['rows'][kernel]:
                    best_acc = max(best_acc, r['acc'])
            vals.append(best_acc)
        bars = ax.bar(x + (ki-1.5)*w, vals, w, label=KERNEL_LABEL[kernel],
                     color=colors[kernel], alpha=0.87, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                   f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(RATIOS, fontsize=11)
    ax.set_ylabel('Accuracy Terbaik', fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f'Accuracy Terbaik per Kernel per Rasio ({feature_label})\n(dari semua arsitektur & parameter)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    st.divider()

    # ── Word Cloud sesuai model terbaik (berdasarkan label HASIL PREDIKSI) ─
    st.markdown("**☁️ Word Cloud Sesuai Model Terbaik**")
    pca_suffix = f", PCA {b['pct']}%" if b.get('pct') else ""
    st.caption(
        f"Word Cloud dibangun dari label **hasil prediksi** model dengan konfigurasi "
        f"terbaik (Kernel {KERNEL_LABEL[b['kernel']]}, Rasio {b['ratio']}, "
        f"{b['arch']}{pca_suffix}), "
        f"bukan sekadar label sentimen asli, sehingga mencerminkan kata-kata yang "
        f"benar-benar memengaruhi keputusan klasifikasi model."
    )
    if df is not None:
        wc_key = f"{key_prefix}_best_model_pred"
        if st.button("☁️ Jalankan Prediksi & Buat Word Cloud Model Terbaik", key=f"btn_{wc_key}"):
            with st.spinner("⏳ Melatih ulang model terbaik & memprediksi data uji..."):
                df_hash = str(hash(tuple(df['content'].tolist())))
                pred_result = predict_best_model(df_hash, df, b, use_chi2=result.get('use_chi2', True))
                st.session_state[wc_key] = pred_result
            st.success("✅ Prediksi selesai!", icon="🎉")

        if wc_key in st.session_state:
            pred_result = st.session_state[wc_key]
            test_idx  = pred_result['test_idx']
            y_pred    = pred_result['y_pred']
            classes   = pred_result['classes']
            df_final  = pred_result['df_final']

            texts_pred = df_final.iloc[test_idx]['final'].reset_index(drop=True)
            pred_labels = pd.Series([classes[p] for p in y_pred])

            wc_c1, wc_c2 = st.columns(2)
            for col, cls_name, cmap in [(wc_c1, 'POSITIVE', 'Greens'), (wc_c2, 'NEGATIVE', 'Reds')]:
                with col:
                    subset = texts_pred[pred_labels == cls_name]
                    try:
                        from wordcloud import WordCloud
                        text = ' '.join(subset.dropna().tolist())
                        if text.strip():
                            wc = WordCloud(width=500, height=300, background_color='white',
                                           colormap=cmap, max_words=100,
                                           collocations=False).generate(text)
                            fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                            ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off')
                            ax_wc.set_title(f'Prediksi {cls_name} (n={len(subset)})',
                                            fontsize=12, fontweight='bold')
                            st.pyplot(fig_wc, use_container_width=True); plt.close()
                        else:
                            st.warning(f"Tidak ada data terprediksi {cls_name}.")
                    except ImportError:
                        st.warning("Install wordcloud: `pip install wordcloud`")

    st.divider()
    st.markdown("**📥 Ekspor Hasil ke Dokumen Word**")
    st.caption(
        "Unduh seluruh matriks evaluasi sebagai dokumen Word terisi, dengan "
        "urutan penyajian: Arsitektur (1-4) → Rasio Split Data → Kernel — "
        "siap disalin ke Bab IV skripsi."
    )

    if st.button("📄 Buat Dokumen Word", key=f"{key_prefix}_btn_docx"):
        with st.spinner("⏳ Membuat dokumen Word..."):
            docx_bytes = _build_docx(result)
        fname = "Tabel_Evaluasi_SVM_Terisi.docx" if result.get('use_chi2', True) \
                else "Tabel_Evaluasi_SVM_TanpaChiSquare_Terisi.docx"
        st.download_button(
            label=f"⬇️ Unduh {fname}",
            data=docx_bytes,
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"{key_prefix}_dl_docx",
        )


def render(df=None, **kwargs):
    st.subheader("📋 Matriks Evaluasi Lengkap (Blueprint Tabel Skripsi)")
    st.caption(
        "Menjalankan skenario evaluasi lengkap — 3 rasio split (90:10, 80:20, 70:30) "
        "× 4 arsitektur (PCA/NoPCA × Sebelum/Setelah SMOTE) × 4 kernel × grid parameter "
        "— menggunakan **data asli** dari model (bukan tabel kosong). Struktur mengikuti "
        "blueprint tabel evaluasi Bab IV skripsi."
    )

    with st.expander("📖 Penjelasan Lengkap Metodologi Pengujian (720 Kombinasi)", expanded=False):
        st.markdown(
            "Proses klasifikasi menggunakan algoritma SVM diujikan pada empat jenis "
            "kernel, yaitu **Linear**, **Radial Basis Function (RBF)**, **Polynomial**, "
            "dan **Sigmoid**, dengan kombinasi parameter **C ∈ {0,1; 1; 10}**, "
            "**gamma (γ) ∈ {0,001; 0,1; 1}** (untuk kernel RBF, Polynomial, dan "
            "Sigmoid), menghasilkan **30 kombinasi kernel-parameter** (3 untuk "
            "Linear, dan 9 masing-masing untuk RBF, Polynomial, serta Sigmoid).\n\n"
            "Setiap kombinasi kernel-parameter diujikan pada **empat skenario "
            "perlakuan data**, yaitu tanpa PCA sebelum SMOTE, tanpa PCA setelah "
            "SMOTE, dengan PCA sebelum SMOTE, dan dengan PCA setelah SMOTE. Khusus "
            "pada kedua skenario \"dengan PCA\", pengujian turut dilakukan pada "
            "**ketiga nilai threshold variansi (75%, 95%, dan 99%)** sebagaimana "
            "dijelaskan pada tahap Transformation, sehingga skenario dengan PCA "
            "menghasilkan **3 kali lipat** jumlah pengujian dibandingkan skenario "
            "tanpa PCA.\n\n"
            "Dengan demikian, total pengujian pada setiap rasio split data adalah:\n\n"
            "**(30 kombinasi × 2 skenario tanpa PCA) + (30 kombinasi × 3 threshold "
            "× 2 skenario dengan PCA) = 60 + 180 = 240 kombinasi.**\n\n"
            "Dengan tiga rasio split data yang digunakan (90:10, 80:20, 70:30), "
            "**total keseluruhan pengujian pada tahap Data Mining adalah "
            "720 kombinasi eksperimen.**"
        )
        st.divider()
        st.markdown(
            "**Rincian perhitungan 30 kombinasi kernel-parameter:**"
        )
        rincian_df = pd.DataFrame({
            'Kernel':          ['Linear', 'RBF', 'Polynomial', 'Sigmoid', 'Total'],
            'Parameter':       ['C saja (gamma tidak berlaku)', 'C × gamma', 'C × gamma', 'C × gamma', ''],
            'Jumlah Kombinasi':['3', '9', '9', '9', '30'],
        })
        st.dataframe(rincian_df, use_container_width=True, hide_index=True)

        st.markdown("**Rincian perhitungan 240 kombinasi per rasio split:**")
        rincian2_df = pd.DataFrame({
            'Skenario':        ['Tanpa PCA + Sebelum SMOTE', 'Tanpa PCA + Setelah SMOTE',
                                'Dengan PCA + Sebelum SMOTE (× 3 threshold)',
                                'Dengan PCA + Setelah SMOTE (× 3 threshold)', 'Total per Rasio'],
            'Jumlah Kombinasi':['30', '30', '90', '90', '240'],
        })
        st.dataframe(rincian2_df, use_container_width=True, hide_index=True)
        st.caption(
            "240 kombinasi per rasio × 3 rasio split data (90:10, 80:20, 70:30) "
            "= **720 kombinasi eksperimen total** pada tahap Data Mining."
        )
        st.info(
            "Perlu ditegaskan bahwa pendekatan pengujian ini bersifat **ekshaustif "
            "(exhaustive one-by-one testing)** terhadap seluruh kombinasi parameter "
            "yang ditetapkan, dan **bukan** merupakan proses pencarian parameter "
            "optimal menggunakan grid search maupun teknik tuning otomatis lainnya. "
            "Dengan demikian, hasil pada tab ini tidak menyimpulkan kernel atau "
            "kombinasi parameter mana yang secara mutlak paling unggul, melainkan "
            "menyajikan gambaran performa SVM secara menyeluruh pada setiap "
            "kombinasi yang diujikan."
        )

    if df is None:
        st.warning("Dataset belum tersedia.", icon="⚠️")
        return

    df_hash = str(hash(tuple(df['content'].tolist())))

    st.warning(
        "⏳ **Perhatian:** Tab ini melatih **720 model SVM** sekaligus "
        "(3 rasio × 4 arsitektur × grid parameter). Proses pertama kali bisa "
        "memakan waktu **beberapa menit**. Hasil akan di-cache otomatis setelahnya.",
        icon="⚠️"
    )

    run_matrix = st.button("🚀 Jalankan Matriks Evaluasi Lengkap", type="primary")

    if run_matrix or 'full_matrix_result' in st.session_state:
        if run_matrix:
            with st.spinner("⏳ Melatih 720 model SVM (3 rasio × 4 arsitektur × grid parameter)... mohon tunggu"):
                result = full_evaluation_matrix(df_hash, df)
                st.session_state['full_matrix_result'] = result
            st.success("✅ Matriks evaluasi lengkap selesai dijalankan!", icon="🎉")

        result  = st.session_state['full_matrix_result']
        _render_matrix_results(result, key_prefix='chi2', feature_label='fitur Chi-Square', df=df)

        st.divider()
        st.markdown("---")
        st.markdown("## 🆚 Perbandingan: Tanpa Feature Selection Chi-Square")
        st.caption(
            "Menjalankan ULANG seluruh 720 kombinasi eksperimen yang sama (3 rasio "
            "× 4 arsitektur × grid parameter), namun kali ini SVM dilatih langsung "
            "pada seluruh fitur hasil TF-IDF (tanpa diseleksi Chi-Square terlebih "
            "dahulu), sebagai pembanding untuk mengukur kontribusi nyata tahap "
            "seleksi fitur Chi-Square terhadap performa klasifikasi."
        )
        st.warning(
            "⏳ **Perhatian:** Karena dimensi fitur jauh lebih besar (~3.000-5.000 "
            "fitur TF-IDF dibandingkan 1.000 fitur Chi-Square), skenario tanpa PCA "
            "pada bagian ini akan berjalan **lebih lambat** dibandingkan matriks "
            "utama di atas, terutama pada kernel non-linear (RBF, Polynomial, "
            "Sigmoid). Skenario dengan PCA tetap mereduksi dimensi seperti biasa "
            "sehingga relatif tidak terpengaruh.",
            icon="⚠️"
        )

        run_matrix_nochi2 = st.button(
            "🚀 Jalankan Matriks Evaluasi Lengkap — Tanpa Chi-Square",
            type="primary", key="btn_run_nochi2"
        )

        if run_matrix_nochi2 or 'full_matrix_result_nochi2' in st.session_state:
            if run_matrix_nochi2:
                with st.spinner("⏳ Melatih 720 model SVM TANPA seleksi Chi-Square (fitur TF-IDF penuh)... proses ini bisa memakan waktu lebih lama, mohon tunggu"):
                    result_nochi2 = full_evaluation_matrix(df_hash, df, use_chi2=False)
                    st.session_state['full_matrix_result_nochi2'] = result_nochi2
                st.success("✅ Matriks evaluasi Tanpa Chi-Square selesai dijalankan!", icon="🎉")

            result_nochi2 = st.session_state['full_matrix_result_nochi2']
            _render_matrix_results(result_nochi2, key_prefix='nochi2', feature_label='fitur TF-IDF (tanpa seleksi)', df=df)

            # ── Perbandingan ringkas Dengan vs Tanpa Chi-Square ──────────────────
            st.divider()
            st.markdown("## 📊 Ringkasan Perbandingan: Dengan Chi-Square vs Tanpa Chi-Square")

            def _best_f1(res_dict):
                best = None
                for ratio in RATIOS:
                    for arch in ARCHS:
                        for kernel in KERNELS:
                            for r in res_dict['results'][ratio][arch]['rows'][kernel]:
                                if best is None or r['f1'] > best['f1']:
                                    best = r
                return best

            def _avg_f1(res_dict):
                vals = []
                for ratio in RATIOS:
                    for arch in ARCHS:
                        for kernel in KERNELS:
                            for r in res_dict['results'][ratio][arch]['rows'][kernel]:
                                vals.append(r['f1'])
                return sum(vals) / len(vals)

            chi2_result = st.session_state['full_matrix_result']
            best_chi2   = _best_f1(chi2_result)
            best_nochi2 = _best_f1(result_nochi2)
            avg_chi2    = _avg_f1(chi2_result)
            avg_nochi2  = _avg_f1(result_nochi2)

            df_compare_fs = pd.DataFrame({
                'Skenario': ['Dengan Chi-Square (1.000 fitur)',
                            f"Tanpa Chi-Square ({result_nochi2['orig_dim']:,} fitur TF-IDF)"],
                'Dimensi Fitur Awal': [f"{chi2_result['orig_dim']:,}", f"{result_nochi2['orig_dim']:,}"],
                'Rata-Rata F1-Score (720 kombinasi)': [f"{avg_chi2*100:.2f}%", f"{avg_nochi2*100:.2f}%"],
                'F1-Score Maksimum': [f"{best_chi2['f1']*100:.2f}%", f"{best_nochi2['f1']*100:.2f}%"],
            })
            st.dataframe(df_compare_fs, use_container_width=True, hide_index=True)

            selisih = (avg_chi2 - avg_nochi2) * 100
            arah = "lebih tinggi" if selisih > 0 else "lebih rendah"
            st.info(
                f"Rata-rata F1-Score dengan seleksi fitur Chi-Square "
                f"({avg_chi2*100:.2f}%) {arah} {abs(selisih):.2f} poin persentase "
                f"dibandingkan tanpa seleksi Chi-Square ({avg_nochi2*100:.2f}%), "
                f"dari total 1.440 kombinasi eksperimen (720 + 720) yang telah diujikan."
            )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # HASIL PREDIKSI 10 DOKUMEN — RASIO 80:20, SEMUA KERNEL & PARAMETER
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("## 🔍 Hasil Prediksi 10 Dokumen — Rasio 80:20, Dengan PCA + Setelah SMOTE, Semua Kernel & Parameter")
        st.caption(
            "Melatih seluruh 30 kombinasi kernel-parameter (Linear × C, dan "
            "RBF/Polynomial/Sigmoid × C × gamma) pada rasio split 80:20, dengan "
            "skenario **Dengan PCA + Setelah SMOTE** (StandardScaler → PCA "
            "threshold 95% diterapkan pada data training hasil SMOTE), kemudian "
            "mencatat hasil prediksi terhadap 10 dokumen uji pertama menggunakan "
            "clean text (teks hasil preprocessing), sesuai struktur per-kernel "
            "seperti pada Matriks Evaluasi Lengkap di atas."
        )

        run_pred10 = st.button("🚀 Jalankan Prediksi 10 Dokumen (30 Kombinasi)", key="btn_pred10")

        if run_pred10 or 'pred10_result' in st.session_state:
            if run_pred10:
                with st.spinner("⏳ Melatih 30 kombinasi kernel-parameter (Dengan PCA + Setelah SMOTE) pada rasio 80:20..."):
                    pred10_result = svm_prediction_detail_8020(df_hash, df)
                    st.session_state['pred10_result'] = pred10_result
                st.success("✅ Prediksi 10 dokumen selesai untuk semua kombinasi!", icon="🎉")

            pred10_result = st.session_state['pred10_result']

            # ── Tabel referensi 10 dokumen (clean text + label aktual) ──────────
            st.markdown("**📄 Referensi 10 Dokumen Uji (Clean Text Hasil Preprocessing)**")
            df_ref10 = pd.DataFrame({
                'Dok':          [f'Dok-{i+1}' for i in range(10)],
                'Clean Text (Hasil Preprocessing)': pred10_result['clean_texts_10'],
                'Label Aktual': pred10_result['actual_10'],
            })
            st.dataframe(df_ref10, use_container_width=True, hide_index=True)
            st.caption(
                f"Data training: {pred10_result['n_train']:,} sampel | "
                f"Data testing: {pred10_result['n_test']:,} sampel | "
                f"Dimensi awal: {pred10_result['orig_dim']:,} fitur Chi-Square → "
                f"direduksi PCA menjadi {pred10_result['n_comp']:,} komponen "
                f"(variansi dipertahankan {pred10_result['var_exp']*100:.2f}%)"
            )

            st.divider()

            # ── Tabel prediksi per kernel (30 kombinasi, kolom = 10 dokumen) ────
            st.markdown("**📊 Tabel Prediksi per Kernel (Semua Kombinasi Parameter)**")
            for kernel in KERNELS:
                letter = KERNEL_LETTER[kernel]
                klabel = KERNEL_LABEL[kernel]
                rows = pred10_result['kernel_results'][kernel]

                with st.expander(f"**{letter}. Kernel {klabel}** ({len(rows)} kombinasi parameter)",
                                 expanded=(kernel=='rbf')):
                    table_rows = []
                    for idx, r in enumerate(rows, 1):
                        row_dict = {
                            'No': idx,
                            'C': r['C'],
                            'Gamma': r['gamma'] if r['gamma'] is not None else '-',
                        }
                        benar = 0
                        for i in range(10):
                            pred_lbl = r['pred_10'][i]
                            actual_lbl = pred10_result['actual_10'][i]
                            cocok = pred_lbl == actual_lbl
                            benar += cocok
                            row_dict[f'D{i+1}'] = f"{pred_lbl[:3]} {'✓' if cocok else '✗'}"
                        row_dict['Benar/10'] = f"{benar}/10"
                        row_dict['Accuracy (%)'] = f"{r['acc']*100:.2f}"
                        row_dict['F1-Score (%)'] = f"{r['f1']*100:.2f}"
                        table_rows.append(row_dict)

                    df_kernel_pred = pd.DataFrame(table_rows)
                    best_idx = max(range(len(rows)), key=lambda i: rows[i]['f1'])

                    def _highlight_best_pred(row):
                        if row.name == best_idx:
                            return ['background-color: #EAF3DE'] * len(row)
                        return [''] * len(row)

                    st.dataframe(
                        df_kernel_pred.style.apply(_highlight_best_pred, axis=1),
                        use_container_width=True, hide_index=True
                    )
                    st.caption(
                        f"D1-D10 = prediksi untuk Dok-1 s.d. Dok-10 (3 huruf pertama "
                        f"label + ✓/✗ dibanding label aktual). Baris hijau = "
                        f"kombinasi dengan F1-Score tertinggi pada kernel ini."
                    )

            st.divider()

            # ── Ekspor ke Word ────────────────────────────────────────────────
            st.markdown("**📥 Ekspor Hasil Prediksi ke Dokumen Word**")
            st.caption(
                "Unduh tabel referensi 10 dokumen beserta hasil prediksi seluruh "
                "30 kombinasi kernel-parameter (rasio 80:20) sebagai dokumen Word."
            )
            if st.button("📄 Buat Dokumen Word — Prediksi 10 Dokumen", key="btn_docx_pred10"):
                with st.spinner("⏳ Membuat dokumen Word..."):
                    docx_pred10_bytes = _build_docx_pred10(pred10_result)
                st.download_button(
                    label="⬇️ Unduh Prediksi_10_Dokumen_SVM_80-20.docx",
                    data=docx_pred10_bytes,
                    file_name="Prediksi_10_Dokumen_SVM_80-20.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dl_docx_pred10",
                )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # UJI VALIDASI TAMBAHAN: 10-FOLD CROSS VALIDATION
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("## 🔟 Uji Validasi Tambahan: 10-Fold Cross Validation")
        st.caption(
            "Sesuai Bab 4.5.6: validasi tambahan pada model dengan performa optimal, "
            "menggunakan StratifiedKFold 10-fold. Setiap fold menerapkan SMOTE dan PCA "
            "hanya pada data training-nya sendiri (tanpa leakage antar fold)."
        )
        st.caption(
            "Kombinasi terbaik otomatis diambil dari hasil skenario **Dengan "
            "Chi-Square** (720 kombinasi utama) di atas."
        )

        # Hitung ulang kombinasi terbaik di scope ini (dari hasil Chi-Square)
        b = _compute_best_overall(st.session_state['full_matrix_result'])

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            use_best = st.checkbox(
                "Gunakan kombinasi terbaik otomatis", value=True, key="cv10_use_best"
            )

        if use_best:
            cv_kernel  = b['kernel']
            cv_C       = b['C']
            cv_gamma   = b['gamma'] if b.get('gamma') is not None else 'scale'
            cv_pca     = 'Dengan PCA' in b['arch']
            cv_thresh  = (b['pct'] / 100) if b.get('pct') else 0.95
            cv_smote   = 'Setelah SMOTE' in b['arch']
            st.info(
                f"**Model yang divalidasi:** Kernel `{cv_kernel.upper()}` | "
                f"C=`{cv_C}` | gamma=`{cv_gamma}` | "
                f"PCA: `{'Ya (' + str(int(cv_thresh*100)) + '%)' if cv_pca else 'Tidak'}` | "
                f"SMOTE: `{'Ya' if cv_smote else 'Tidak'}`"
            )
        else:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                cv_kernel = st.selectbox("Kernel:", KERNELS, key="cv10_kernel")
                cv_C      = st.select_slider("C:", [0.1, 1, 10, 100], value=1, key="cv10_c")
            with col_m2:
                cv_gamma_opt = st.select_slider(
                    "gamma:", ["0.001", "0.01", "0.1", "1", "scale"], value="0.001", key="cv10_gamma"
                )
                cv_gamma = cv_gamma_opt if cv_gamma_opt == 'scale' else float(cv_gamma_opt)
            with col_m3:
                cv_pca    = st.checkbox("Gunakan PCA", value=True, key="cv10_pca")
                cv_thresh = st.select_slider(
                    "Threshold PCA:", [0.75, 0.95, 0.99], value=0.95, key="cv10_thresh"
                ) if cv_pca else 0.95
            cv_smote = st.checkbox("Gunakan SMOTE", value=True, key="cv10_smote")

        if st.button("🔟 Jalankan 10-Fold Cross Validation", type="primary", key="cv10_run"):
            df_hash_cv = str(hash(tuple(df['content'].tolist())))
            with st.spinner("⏳ Menjalankan 10-fold cross validation..."):
                cv_result = run_10fold_cv_validation(
                    df_hash_cv, cv_kernel, cv_C, cv_gamma,
                    cv_pca, cv_thresh, cv_smote, df
                )
            st.session_state['cv10_result'] = cv_result
            st.success("✅ 10-Fold Cross Validation selesai!", icon="🎉")

        if 'cv10_result' in st.session_state:
            cv = st.session_state['cv10_result']

            st.divider()
            st.markdown("**📋 Tabel Hasil Validasi (sesuai format Tabel 4.5.6)**")

            model_desc = (
                f"SVM {cv['kernel'].upper()} (C={cv['C']}, γ={cv['gamma']}"
                + (f", PCA {int(cv['pca_threshold']*100)}%" if cv['use_pca'] else ", Tanpa PCA")
                + (", SMOTE" if cv['use_smote'] else ", Tanpa SMOTE") + ")"
            )
            summary_tbl = pd.DataFrame([{
                'Model Klasifikasi Optimal':   model_desc,
                'Skenario Validasi':           f"{cv['n_folds']}-Fold Cross Validation",
                'Rata-Rata Accuracy (%)':      f"{cv['mean_acc']*100:.2f} ± {cv['std_acc']*100:.2f}",
                'Rata-Rata Precision (%)':     f"{cv['mean_prec']*100:.2f} ± {cv['std_prec']*100:.2f}",
                'Rata-Rata Recall (%)':        f"{cv['mean_rec']*100:.2f} ± {cv['std_rec']*100:.2f}",
                'Rata-Rata F1-Score (%)':      f"{cv['mean_f1']*100:.2f} ± {cv['std_f1']*100:.2f}",
                'Status Stabilitas':           cv['stability'],
            }])
            st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

            st.divider()

            # ── Tabel per fold ────────────────────────────────────────────────
            st.markdown("**📄 Detail Skor per Fold**")
            fold_tbl = pd.DataFrame([{
                'Fold':        r['fold'],
                'Accuracy (%)':  f"{r['acc']*100:.2f}",
                'Precision (%)': f"{r['prec']*100:.2f}",
                'Recall (%)':    f"{r['rec']*100:.2f}",
                'F1-Score (%)':  f"{r['f1']*100:.2f}",
                'n Train':     f"{r['n_train']:,}",
                'n Test':      f"{r['n_test']:,}",
            } for r in cv['fold_results']])
            st.dataframe(fold_tbl, use_container_width=True, hide_index=True)

            st.divider()

            # ── Grafik: skor per fold (line chart 4 metrik) ────────────────────
            st.markdown("**📈 Grafik Skor per Fold — 10-Fold Cross Validation**")
            folds = [r['fold'] for r in cv['fold_results']]
            fig_cv, ax_cv = plt.subplots(figsize=(11, 5))
            metric_colors = {'acc':'#4C72B0','prec':'#55A868','rec':'#E24B4A','f1':'#F5A623'}
            metric_labels = {'acc':'Accuracy','prec':'Precision','rec':'Recall','f1':'F1-Score'}
            for mk in ['acc','prec','rec','f1']:
                vals = [r[mk] for r in cv['fold_results']]
                ax_cv.plot(folds, vals, 'o-', label=metric_labels[mk],
                          color=metric_colors[mk], linewidth=2, markersize=7)
            ax_cv.set_xlabel('Fold ke-', fontsize=11)
            ax_cv.set_ylabel('Skor', fontsize=11)
            ax_cv.set_xticks(folds)
            ax_cv.set_ylim(0, 1.05)
            ax_cv.set_title(f'Skor per Fold — {model_desc}', fontsize=12, fontweight='bold')
            ax_cv.legend(fontsize=10); ax_cv.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_cv, use_container_width=True); plt.close()

            st.divider()

            # ── Grafik: boxplot distribusi skor ─────────────────────────────────
            st.markdown("**📦 Boxplot Distribusi Skor 10-Fold CV**")
            fig_box, ax_box = plt.subplots(figsize=(8, 5))
            data_box = [
                [r['acc']  for r in cv['fold_results']],
                [r['prec'] for r in cv['fold_results']],
                [r['rec']  for r in cv['fold_results']],
                [r['f1']   for r in cv['fold_results']],
            ]
            bp = ax_box.boxplot(data_box, labels=['Accuracy','Precision','Recall','F1-Score'],
                                patch_artist=True, medianprops={'color':'white','linewidth':2})
            for patch, color in zip(bp['boxes'], ['#4C72B0','#55A868','#E24B4A','#F5A623']):
                patch.set_facecolor(color); patch.set_alpha(0.8)
            ax_box.set_ylabel('Skor', fontsize=11)
            ax_box.set_title('Distribusi Skor 10-Fold Cross Validation', fontsize=12, fontweight='bold')
            ax_box.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            col_bp, _ = st.columns([1,1])
            with col_bp:
                st.pyplot(fig_box, use_container_width=True)
            plt.close()

            st.divider()

            # ── Grafik: bar chart rata-rata ± std ───────────────────────────────
            st.markdown("**📊 Rata-Rata ± Standar Deviasi per Metrik**")
            fig_bar, ax_bar = plt.subplots(figsize=(9, 5))
            metric_names = ['Accuracy','Precision','Recall','F1-Score']
            means = [cv['mean_acc'], cv['mean_prec'], cv['mean_rec'], cv['mean_f1']]
            stds  = [cv['std_acc'],  cv['std_prec'],  cv['std_rec'],  cv['std_f1']]
            bars = ax_bar.bar(metric_names, means, yerr=stds, capsize=6,
                              color=['#4C72B0','#55A868','#E24B4A','#F5A623'],
                              alpha=0.85, edgecolor='white')
            for bar, m, s in zip(bars, means, stds):
                ax_bar.text(bar.get_x()+bar.get_width()/2, m+s+0.02,
                           f'{m*100:.2f}±{s*100:.2f}%', ha='center',
                           fontsize=9, fontweight='bold')
            ax_bar.set_ylabel('Skor', fontsize=11)
            ax_bar.set_ylim(0, 1.15)
            ax_bar.set_title(f'Rata-Rata ± Std — {model_desc}', fontsize=12, fontweight='bold')
            ax_bar.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True); plt.close()

            st.divider()
            st.success(
                f"**Kesimpulan Validasi 10-Fold CV:**\n\n"
                f"- Model **{model_desc}** menghasilkan rata-rata F1-Score "
                f"**{cv['mean_f1']*100:.2f}% ± {cv['std_f1']*100:.2f}%** "
                f"pada 10 fold.\n\n"
                f"- Status stabilitas: **{cv['stability']}** "
                f"(std F1 = {cv['std_f1']:.4f}) — "
                + ("model konsisten di berbagai subset data."
                   if '✅' in cv['stability'] else
                   "performa model cukup bervariasi antar subset data, "
                   "perlu diperhatikan pada tahap generalisasi.")
            )
    else:
        st.info("Klik tombol di atas untuk menjalankan matriks evaluasi lengkap.", icon="👈")


def _build_docx(result) -> bytes:
    """Bangun dokumen Word terisi menggunakan python-docx."""
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    results  = result['results']
    orig_dim = result['orig_dim']
    use_chi2 = result.get('use_chi2', True)
    feat_label_short = f'{orig_dim:,} Fitur Chi-Square' if use_chi2 else f'{orig_dim:,} Fitur TF-IDF (Tanpa Seleksi)'

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width

    def set_cell_shading(cell, color_hex):
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), color_hex)
        cell._tc.get_or_add_tcPr().append(shd)

    title_text = 'TABEL HASIL EVALUASI SVM (Terisi — Data Asli)' if use_chi2 \
        else 'TABEL HASIL EVALUASI SVM — TANPA SELEKSI CHI-SQUARE (Terisi — Data Asli)'
    title = doc.add_heading(title_text, level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        f'Rasio 90:10, 80:20, 70:30 — Hasil eksperimen aktual dari model SVM '
        f'(dimensi awal: {feat_label_short})'
    )
    run.italic = True

    table_no = 1
    ARCH_NUMBERED = {
        'Tanpa PCA + Sebelum SMOTE':    '1. SVM Tanpa PCA Sebelum SMOTE',
        'Tanpa PCA + Setelah SMOTE':    '2. SVM Tanpa PCA Setelah SMOTE',
        'Dengan PCA + Sebelum SMOTE':   '3. SVM Dengan PCA + Sebelum SMOTE',
        'Dengan PCA + Setelah SMOTE':   '4. SVM Dengan PCA + Setelah SMOTE',
    }
    ARCH_DESC_DOCX = {
        'Tanpa PCA + Sebelum SMOTE':    f'Pelatihan pada {feat_label_short} asli',
        'Tanpa PCA + Setelah SMOTE':    f'Pelatihan pada {feat_label_short} pasca-SMOTE',
        'Dengan PCA + Sebelum SMOTE':   'Pelatihan pada komponen PC sebelum SMOTE',
        'Dengan PCA + Setelah SMOTE':   'Pelatihan pada komponen PC pasca-SMOTE',
    }

    for arch in ARCHS:
        doc.add_heading(ARCH_NUMBERED.get(arch, f'SVM {arch}'), level=2)
        desc_p = doc.add_paragraph()
        desc_run = desc_p.add_run(ARCH_DESC_DOCX.get(arch, ''))
        desc_run.italic = True
        is_pca = 'Dengan PCA' in arch

        for ratio in RATIOS:
            doc.add_heading(f'Rasio Split Data {ratio}', level=3)
            arch_data = results[ratio][arch]

            for kernel in KERNELS:
                letter = KERNEL_LETTER[kernel]
                klabel = KERNEL_LABEL[kernel]
                doc.add_heading(f'{letter}. Kernel {klabel}', level=4)
                cap = doc.add_paragraph()
                cap_run = cap.add_run(f'Tabel {table_no} Performa Kernel {klabel} {arch} (Rasio {ratio})')
                cap_run.bold = True
                table_no += 1

                rows = arch_data['rows'][kernel]
                has_gamma = rows[0].get('gamma') is not None

                if is_pca and has_gamma:
                    headers = ['No','% PCA','Komponen Hasil Reduksi','Param. C','Param. γ',
                              'Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)']
                elif is_pca and not has_gamma:
                    headers = ['No','% PCA','Komponen Hasil Reduksi','Parameter C',
                              'Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)']
                elif not is_pca and has_gamma:
                    headers = ['No','Komponen Fitur','Param. C','Param. γ',
                              'Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)']
                else:
                    headers = ['No','Komponen Fitur','Parameter C',
                              'Accuracy (%)','Precision (%)','Recall (%)','F1-Score (%)']

                table = doc.add_table(rows=1, cols=len(headers))
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.style = 'Table Grid'

                hdr_cells = table.rows[0].cells
                for i, h in enumerate(headers):
                    hdr_cells[i].text = h
                    set_cell_shading(hdr_cells[i], '2E75B6')
                    for p_ in hdr_cells[i].paragraphs:
                        p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for r_ in p_.runs:
                            r_.font.bold = True
                            r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                            r_.font.size = Pt(9)

                best_f1_idx = max(range(len(rows)), key=lambda i: rows[i]['f1'])

                for idx, r in enumerate(rows, 1):
                    row_cells = table.add_row().cells
                    vals = [str(idx)]
                    if is_pca:
                        vals.append(f"{r['pct']}%")
                        vals.append(f"{r['n_comp']} PC (dari {orig_dim} {'Fitur Chi-Square' if use_chi2 else 'Fitur TF-IDF'})")
                    else:
                        vals.append(f"{orig_dim} {'Fitur Chi-Square' if use_chi2 else 'Fitur TF-IDF'}")
                    vals.append(str(r['C']))
                    if has_gamma:
                        vals.append(str(r['gamma']))
                    vals.append(f"{r['acc']*100:.2f}")
                    vals.append(f"{r['prec']*100:.2f}")
                    vals.append(f"{r['rec']*100:.2f}")
                    vals.append(f"{r['f1']*100:.2f}")

                    for i, v in enumerate(vals):
                        row_cells[i].text = v
                        for p_ in row_cells[i].paragraphs:
                            p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            for r_ in p_.runs:
                                r_.font.size = Pt(9)
                        if idx - 1 == best_f1_idx:
                            set_cell_shading(row_cells[i], 'EAF3DE')

                doc.add_paragraph()

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _build_docx_pred10(pred10_result) -> bytes:
    """Membangun dokumen Word berisi hasil prediksi 10 dokumen uji untuk
    seluruh 30 kombinasi kernel-parameter pada rasio split 80:20."""
    from docx import Document
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.section import WD_ORIENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from io import BytesIO

    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width, section.page_height = section.page_height, section.page_width

    def set_cell_shading(cell, color_hex):
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), color_hex)
        cell._tc.get_or_add_tcPr().append(shd)

    title = doc.add_heading('HASIL PREDIKSI 10 DOKUMEN UJI — RASIO 80:20, DENGAN PCA + SETELAH SMOTE (SEMUA KERNEL & PARAMETER)', level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run(
        f"Data training: {pred10_result['n_train']:,} sampel | "
        f"Data testing: {pred10_result['n_test']:,} sampel | "
        f"Dimensi awal: {pred10_result['orig_dim']:,} fitur Chi-Square -> "
        f"direduksi PCA menjadi {pred10_result['n_comp']:,} komponen "
        f"(variansi dipertahankan {pred10_result['var_exp']*100:.2f}%)"
    )
    run.italic = True

    # ── Tabel referensi 10 dokumen ────────────────────────────────────────────
    doc.add_heading('Referensi 10 Dokumen Uji (Clean Text Hasil Preprocessing)', level=2)
    ref_table = doc.add_table(rows=1, cols=3)
    ref_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    ref_table.style = 'Table Grid'
    hdr = ref_table.rows[0].cells
    for i, h in enumerate(['Dokumen', 'Clean Text (Hasil Preprocessing)', 'Label Aktual']):
        hdr[i].text = h
        set_cell_shading(hdr[i], '2E75B6')
        for p_ in hdr[i].paragraphs:
            p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r_ in p_.runs:
                r_.font.bold = True
                r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r_.font.size = Pt(9)
    for i in range(10):
        row_cells = ref_table.add_row().cells
        vals = [f'Dok-{i+1}', pred10_result['clean_texts_10'][i], pred10_result['actual_10'][i]]
        for j, v in enumerate(vals):
            row_cells[j].text = str(v)
            for p_ in row_cells[j].paragraphs:
                for r_ in p_.runs:
                    r_.font.size = Pt(9)

    doc.add_paragraph()

    # ── Tabel prediksi per kernel ──────────────────────────────────────────────
    doc.add_heading('Tabel Prediksi per Kernel (Semua Kombinasi Parameter)', level=2)
    table_no = 1
    for kernel in KERNELS:
        letter = KERNEL_LETTER[kernel]
        klabel = KERNEL_LABEL[kernel]
        rows = pred10_result['kernel_results'][kernel]

        doc.add_heading(f'{letter}. Kernel {klabel}', level=3)
        cap = doc.add_paragraph()
        cap_run = cap.add_run(f'Tabel {table_no} Hasil Prediksi 10 Dokumen — Kernel {klabel} (Rasio 80:20)')
        cap_run.bold = True
        table_no += 1

        headers = ['No', 'C', 'Gamma'] + [f'D{i+1}' for i in range(10)] + \
                  ['Benar/10', 'Accuracy (%)', 'F1-Score (%)']
        table = doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = 'Table Grid'

        hdr_cells = table.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            set_cell_shading(hdr_cells[i], '2E75B6')
            for p_ in hdr_cells[i].paragraphs:
                p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r_ in p_.runs:
                    r_.font.bold = True
                    r_.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                    r_.font.size = Pt(8)

        best_idx = max(range(len(rows)), key=lambda i: rows[i]['f1'])

        for idx, r in enumerate(rows, 1):
            row_cells = table.add_row().cells
            benar = sum(
                r['pred_10'][i] == pred10_result['actual_10'][i] for i in range(10)
            )
            vals = [str(idx), str(r['C']), str(r['gamma']) if r['gamma'] is not None else '-']
            for i in range(10):
                cocok = r['pred_10'][i] == pred10_result['actual_10'][i]
                vals.append(f"{r['pred_10'][i][:3]} {'OK' if cocok else 'X'}")
            vals.append(f"{benar}/10")
            vals.append(f"{r['acc']*100:.2f}")
            vals.append(f"{r['f1']*100:.2f}")

            for i, v in enumerate(vals):
                row_cells[i].text = v
                for p_ in row_cells[i].paragraphs:
                    p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for r_ in p_.runs:
                        r_.font.size = Pt(8)
                if idx - 1 == best_idx:
                    set_cell_shading(row_cells[i], 'EAF3DE')

        doc.add_paragraph()

    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
