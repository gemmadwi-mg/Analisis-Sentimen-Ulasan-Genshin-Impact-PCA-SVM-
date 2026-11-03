"""
tabs/tab_evaluasi.py
Tab: Evaluasi
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

    st.subheader("📈 Evaluasi Model SVM")

    if 'r_pca' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar untuk melatih model.", icon="👈")
    else:
        r_pca   = st.session_state['r_pca']
        r_nopca = st.session_state['r_nopca']
        feat    = st.session_state['feat']
        k_used  = st.session_state['kernel'].upper()
        C_used     = st.session_state.get('C_param', 1)
        gamma_used = st.session_state.get('gamma_param', '0.001')
        degree_used= st.session_state.get('degree_param', 2)
        param_str  = f"C={C_used}, γ={gamma_used}"
        if st.session_state['kernel'] == 'poly':
            param_str += f", degree={degree_used}"
        s_used  = st.session_state['split_label']

        mode = st.radio("Tampilkan hasil:", ["Dengan PCA","Tanpa PCA"], horizontal=True)
        r = r_pca if mode == "Dengan PCA" else r_nopca
        classes = feat['classes']

        # ── Diagram Langkah SVM ─────────────────────────────────────────────
        st.markdown("**🧭 Diagram Proses Klasifikasi SVM**")
        st.caption(
            "SVM mencari hyperplane optimal yang memisahkan kelas dengan "
            "margin maksimum, melalui 4 langkah berikut, diapit kotak INPUT "
            "(data masuk) dan OUTPUT (hasil akhir)."
        )

        n_dim_svm = r_pca['pca_n'] if mode == "Dengan PCA" else r_pca['orig_dim']
        n_train_svm = r['n_train']
        n_test_svm  = r.get('n_test', len(feat['y_test']))

        fig_svm, ax_svm = plt.subplots(figsize=(15.5, 3.4))
        ax_svm.axis('off')
        box_w_v, box_h_v, gap_v, io_w_v = 2.7, 1.8, 0.5, 2.5

        steps_svm = [
            ("1. Terima Data\nLatih (X, y)", "Matriks fitur X\n+ label sentimen y\nsebagai target"),
            ("2. Terapkan\nKernel Trick", f"Kernel: {k_used}\nmemetakan data ke\nruang berdimensi lebih tinggi"),
            ("3. Cari Hyperplane\nOptimal", "Maksimalkan margin\nantar kelas dengan\nparameter C"),
            ("4. Tentukan Support\nVectors", "Titik data terdekat\ndengan hyperplane\nyang menentukan batas"),
        ]
        colors_v = ['#E6F1FB', '#FAEEDA', '#FBEAF0', '#EEEDFE']
        edges_v  = ['#185FA5', '#854F0B', '#993556', '#534AB7']

        # Kotak INPUT
        x_in_v = 0
        rect_in_v = plt.Rectangle((x_in_v, 0), io_w_v, box_h_v, facecolor='#F0F0F0',
                                   edgecolor='#555555', linewidth=1.6, linestyle='--', zorder=2)
        ax_svm.add_patch(rect_in_v)
        ax_svm.text(x_in_v + io_w_v/2, box_h_v - 0.35, "INPUT", ha='center', va='top',
                    fontsize=10, fontweight='bold', color='#333333')
        ax_svm.text(x_in_v + io_w_v/2, box_h_v/2 - 0.30,
                    f"Data Training\n{n_train_svm:,} × {n_dim_svm:,}\n({mode})",
                    ha='center', va='center', fontsize=8.5, color='#333')
        ax_svm.annotate('', xy=(x_in_v + io_w_v + gap_v - 0.1, box_h_v/2),
                        xytext=(x_in_v + io_w_v + 0.1, box_h_v/2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

        # Kotak proses
        x_offset_v = x_in_v + io_w_v + gap_v
        for i, ((title, desc), c, ec) in enumerate(zip(steps_svm, colors_v, edges_v)):
            x = x_offset_v + i * (box_w_v + gap_v)
            rect = plt.Rectangle((x, 0), box_w_v, box_h_v, facecolor=c,
                                  edgecolor=ec, linewidth=1.5, zorder=2)
            ax_svm.add_patch(rect)
            ax_svm.text(x + box_w_v/2, box_h_v - 0.35, title, ha='center', va='top',
                        fontsize=9.5, fontweight='bold', color=ec)
            ax_svm.text(x + box_w_v/2, box_h_v/2 - 0.25, desc, ha='center', va='center',
                        fontsize=8, color='#333')
            ax_svm.annotate('', xy=(x + box_w_v + gap_v - 0.1, box_h_v/2),
                            xytext=(x + box_w_v + 0.1, box_h_v/2),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

        # Kotak OUTPUT
        x_out_v = x_offset_v + len(steps_svm) * (box_w_v + gap_v)
        rect_out_v = plt.Rectangle((x_out_v, 0), io_w_v, box_h_v, facecolor='#E1F5E1',
                                    edgecolor='#2E7D32', linewidth=1.8, zorder=2)
        ax_svm.add_patch(rect_out_v)
        ax_svm.text(x_out_v + io_w_v/2, box_h_v - 0.35, "OUTPUT", ha='center', va='top',
                    fontsize=10, fontweight='bold', color='#2E7D32')
        ax_svm.text(x_out_v + io_w_v/2, box_h_v/2 - 0.30,
                    f"Model Terlatih +\nPrediksi {n_test_svm:,} data uji\nAcc={r['accuracy']*100:.1f}%",
                    ha='center', va='center', fontsize=8.5, color='#333')

        ax_svm.set_xlim(-0.2, x_out_v + io_w_v + 0.2)
        ax_svm.set_ylim(-0.2, box_h_v + 0.2)
        plt.tight_layout()
        st.pyplot(fig_svm, use_container_width=True)
        plt.close()

        st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
        col_svo1, col_svo2, col_svo3, col_svo4 = st.columns(4)
        col_svo1.metric("Accuracy", f"{r['accuracy']*100:.2f}%")
        col_svo2.metric("F1-Score", f"{r['f1']:.4f}")
        col_svo3.metric("Waktu Training", f"{r['train_time']:.3f}s")
        col_svo4.metric("Data Diprediksi", f"{n_test_svm:,}")
        st.caption(
            f"Model SVM kernel {k_used} dilatih pada {n_train_svm:,} data "
            f"training ({n_dim_svm:,} dimensi), lalu digunakan memprediksi "
            f"{n_test_svm:,} data testing — lihat metrik lengkap & confusion "
            f"matrix di bawah/tab terkait."
        )

        st.divider()

        # ── Contoh Hasil Prediksi 10 Data — Rasio 80:20, Semua Kernel ────────────
        st.markdown("**🔍 Contoh Hasil Prediksi 10 Data Uji — Rasio 80:20, Semua Kernel & Parameter**")
        st.caption(
            "Menampilkan hasil prediksi 10 dokumen data uji pertama pada rasio "
            "split 80:20 (tetap, tidak mengikuti pilihan rasio di sidebar), "
            "dibandingkan di keempat kernel dengan parameter C=1 dan gamma=0,001 "
            "(parameter default yang digunakan pada Data Mining), tanpa PCA."
        )

        from models.pipeline import prepare_features, train_without_pca

        with st.spinner("⏳ Melatih 4 kernel pada rasio 80:20 untuk contoh prediksi..."):
            df_hash_pred = str(hash(tuple(df['content'].tolist())))
            feat_8020 = prepare_features(df_hash_pred, 0.8, df, 1000)

            KERNEL_LIST = ['linear', 'rbf', 'poly', 'sigmoid']
            KERNEL_NAMES = {'linear':'Linear', 'rbf':'RBF', 'poly':'Polynomial', 'sigmoid':'Sigmoid'}
            pred_per_kernel = {}
            for kn in KERNEL_LIST:
                r_kn = train_without_pca(df_hash_pred + "_" + kn, kn, feat_8020,
                                          C=1, gamma="0.001", degree=2)
                pred_per_kernel[kn] = r_kn['svm'].predict(feat_8020['X_test'][:10])

        classes_8020 = feat_8020['classes']
        y_test_10    = feat_8020['y_test'][:10]
        idx_test_10  = feat_8020.get('test_idx', None)

        # Ambil cuplikan teks asli 10 dokumen uji pertama (jika tersedia index-nya)
        if idx_test_10 is not None:
            teks_10 = df.iloc[idx_test_10[:10]]['content'].values
        else:
            teks_10 = df['content'].values[:10]

        pred_table_rows = []
        for i in range(10):
            row = {
                'No': i + 1,
                'Cuplikan Ulasan': (teks_10[i][:60] + '...') if len(teks_10[i]) > 60 else teks_10[i],
                'Label Aktual': classes_8020[y_test_10[i]],
            }
            for kn in KERNEL_LIST:
                pred_label = classes_8020[pred_per_kernel[kn][i]]
                benar = pred_label == row['Label Aktual']
                row[f'{KERNEL_NAMES[kn]}'] = f"{pred_label} {'✓' if benar else '✗'}"
            pred_table_rows.append(row)

        df_pred_10 = pd.DataFrame(pred_table_rows)
        st.dataframe(df_pred_10, use_container_width=True, hide_index=True)

        # Ringkasan akurasi 10 sampel per kernel
        st.markdown("**Ringkasan Akurasi pada 10 Sampel Ini (per kernel)**")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        for col, kn in zip([col_p1,col_p2,col_p3,col_p4], KERNEL_LIST):
            benar_count = sum(
                classes_8020[pred_per_kernel[kn][i]] == classes_8020[y_test_10[i]]
                for i in range(10)
            )
            col.metric(f"{KERNEL_NAMES[kn]}", f"{benar_count}/10 benar")

        st.caption(
            "Tabel ini hanya untuk ilustrasi 10 sampel — akurasi sesungguhnya "
            "(dihitung dari seluruh data uji) tetap mengacu pada metrik Accuracy/"
            "F1-Score pada tab Matriks Evaluasi Lengkap, bukan dari 10 sampel ini."
        )

        st.divider()

        st.markdown(f"**Konfigurasi:** Kernel `{k_used}` | Parameter `{param_str}` | Rasio `{s_used}` | {mode}")
        st.divider()

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Accuracy",  f"{r['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{r['precision']:.4f}")
        m3.metric("Recall",    f"{r['recall']:.4f}")
        m4.metric("F1-Score",  f"{r['f1']:.4f}")

        t1,t2,t3 = st.columns(3)
        t1.metric("Waktu Training",  f"{r['train_time']:.3f}s")
        t2.metric("Waktu Prediksi",  f"{r['pred_time']:.3f}s")
        t3.metric("Data Training",   f"{r['n_train']:,} sampel")

        if mode == "Dengan PCA":
            st.divider()
            st.markdown("**Informasi PCA**")
            p1,p2,p3 = st.columns(3)
            p1.metric("Dimensi Awal",      r_pca['orig_dim'])
            p2.metric("Komponen PCA",       r_pca['pca_n'])
            p3.metric("Variance Explained", f"{r_pca['pca_var']*100:.2f}%")

        st.divider()
        st.markdown("**Classification Report**")
        cr_rows = []
        for cls in classes:
            row = r['cr'].get(cls, {})
            cr_rows.append({
                'Kelas':     cls,
                'Precision': f"{row.get('precision',0):.4f}",
                'Recall':    f"{row.get('recall',0):.4f}",
                'F1-Score':  f"{row.get('f1-score',0):.4f}",
                'Support':   int(row.get('support',0)),
            })
        st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Visualisasi Metrik per Kelas**")
        fig, axes = plt.subplots(1, 3, figsize=(12,4))
        for ax, metric, label, color in zip(
            axes,
            ['precision','recall','f1-score'],
            ['Precision','Recall','F1-Score'],
            ['#4C72B0','#55A868','#C44E52']
        ):
            vals = [r['cr'].get(cls,{}).get(metric,0) for cls in classes]
            bars = ax.bar(classes, vals, color=color, alpha=0.8, edgecolor='white')
            ax.set_title(label, fontweight='bold')
            ax.set_ylim(0, 1.1)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.02,
                        f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 6 — CONFUSION MATRIX
    # ══════════════════════════════════════════════════════════════════════════════
