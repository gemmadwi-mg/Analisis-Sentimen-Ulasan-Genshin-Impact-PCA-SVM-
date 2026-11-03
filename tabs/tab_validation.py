"""
tabs/tab_validation.py
Tab: Test Validation — Train / Validation / Test
Memvalidasi model menggunakan 3-way split untuk mendeteksi
overfitting/underfitting dan memastikan generalisasi model.
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.pipeline import run_test_validation

KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
CLR_K   = {'linear':'#4C72B0','rbf':'#55A868','poly':'#E24B4A','sigmoid':'#F5A623'}
CLR_SET = {'Train':'#4C72B0','Validation':'#F5A623','Test':'#55A868','CV':'#E24B4A'}


def _bar3(ax, kernel_res, metric, title, ylabel):
    """Bar chart Train vs Val vs Test untuk satu metrik."""
    x     = np.arange(len(KERNELS))
    w     = 0.25
    sets  = [('Train','train'), ('Validation','val'), ('Test','test')]
    for i, (lbl, key) in enumerate(sets):
        vals = [kernel_res[k][key][metric] for k in KERNELS]
        bars = ax.bar(x + i*w - w, vals, w,
                      label=lbl, color=CLR_SET[lbl],
                      alpha=0.85, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([k.upper() for k in KERNELS], fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    lo = min(kernel_res[k][s][metric] for k in KERNELS for s in ['train','val','test'])
    ax.set_ylim(max(0, lo-0.08), 1.08)
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)


def render(df=None, **kwargs):
    st.subheader("✅ Test Validation")
    st.caption(
        "Validasi model menggunakan 3-way split (Train / Validation / Test) "
        "dan 5-Fold Cross-Validation untuk mendeteksi overfitting/underfitting "
        "dan membuktikan kemampuan generalisasi model."
    )

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    feat    = st.session_state['feat']
    s_used  = st.session_state.get('split_label', '80:20')
    df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "def"

    # ── Kontrol ────────────────────────────────────────────────────────────────
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        use_pca = st.radio("Mode:", ["Dengan PCA","Tanpa PCA"],
                           horizontal=True, key="val_pca")
    with col_c2:
        val_pct = st.select_slider(
            "Ukuran Validation Set (dari training):",
            options=[10, 15, 20, 25, 30],
            value=20,
            format_func=lambda x: f"{x}%",
            key="val_ratio"
        )
    with col_c3:
        st.info(
            f"**Rasio split:** {s_used}\n\n"
            f"**Val set:** {val_pct}% dari training\n\n"
            f"**CV:** StratifiedKFold 5-fold"
        )

    pca_flag  = (use_pca == "Dengan PCA")
    val_ratio = val_pct / 100
    cache_key = f"{df_hash}_{s_used}_{pca_flag}_{val_ratio}"

    with st.spinner("⏳ Menjalankan validasi untuk semua kernel..."):
        res = run_test_validation(cache_key, pca_flag, val_ratio, feat)

    kr      = res['kernel_results']
    classes = res['classes']

    # ── Diagram Langkah Test Validation ─────────────────────────────────────
    st.markdown("**🧭 Diagram Proses Test Validation**")
    st.caption(
        "Data training dipecah lagi menjadi Train/Validation, lalu divalidasi "
        "dengan K-Fold Cross-Validation, melalui 3 langkah berikut, diapit "
        "kotak INPUT (data masuk) dan OUTPUT (hasil akhir)."
    )

    fig_val, ax_val = plt.subplots(figsize=(13.5, 3.4))
    ax_val.axis('off')
    box_w_vl, box_h_vl, gap_vl, io_w_vl = 2.9, 1.8, 0.5, 2.5

    steps_val = [
        ("1. Bagi Training\nmenjadi Train/Val", f"Sisihkan {val_pct}%\ndari training sebagai\nvalidation set"),
        ("2. Latih & Uji di\nTrain / Val / Test", "Bandingkan performa\npada 3 subset untuk\ndeteksi overfitting"),
        ("3. StratifiedKFold\n5-Fold CV", "Ulangi pelatihan 5×\npada training, hitung\nrata-rata & std skor"),
    ]
    colors_val = ['#E6F1FB', '#FAEEDA', '#EAF3DE']
    edges_val  = ['#185FA5', '#854F0B', '#3B6D11']

    # Kotak INPUT
    x_in_vl = 0
    rect_in_vl = plt.Rectangle((x_in_vl, 0), io_w_vl, box_h_vl, facecolor='#F0F0F0',
                                edgecolor='#555555', linewidth=1.6, linestyle='--', zorder=2)
    ax_val.add_patch(rect_in_vl)
    ax_val.text(x_in_vl + io_w_vl/2, box_h_vl - 0.35, "INPUT", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#333333')
    ax_val.text(x_in_vl + io_w_vl/2, box_h_vl/2 - 0.30,
               f"Data Training\n{res['n_train_orig']:,} sampel\n(rasio {s_used})",
               ha='center', va='center', fontsize=8.5, color='#333')
    ax_val.annotate('', xy=(x_in_vl + io_w_vl + gap_vl - 0.1, box_h_vl/2),
                    xytext=(x_in_vl + io_w_vl + 0.1, box_h_vl/2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak proses
    x_offset_vl = x_in_vl + io_w_vl + gap_vl
    for i, ((title, desc), c, ec) in enumerate(zip(steps_val, colors_val, edges_val)):
        x = x_offset_vl + i * (box_w_vl + gap_vl)
        rect = plt.Rectangle((x, 0), box_w_vl, box_h_vl, facecolor=c,
                              edgecolor=ec, linewidth=1.5, zorder=2)
        ax_val.add_patch(rect)
        ax_val.text(x + box_w_vl/2, box_h_vl - 0.35, title, ha='center', va='top',
                   fontsize=9.5, fontweight='bold', color=ec)
        ax_val.text(x + box_w_vl/2, box_h_vl/2 - 0.25, desc, ha='center', va='center',
                   fontsize=8, color='#333')
        ax_val.annotate('', xy=(x + box_w_vl + gap_vl - 0.1, box_h_vl/2),
                        xytext=(x + box_w_vl + 0.1, box_h_vl/2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak OUTPUT
    x_out_vl = x_offset_vl + len(steps_val) * (box_w_vl + gap_vl)
    rect_out_vl = plt.Rectangle((x_out_vl, 0), io_w_vl, box_h_vl, facecolor='#E1F5E1',
                                 edgecolor='#2E7D32', linewidth=1.8, zorder=2)
    ax_val.add_patch(rect_out_vl)
    ax_val.text(x_out_vl + io_w_vl/2, box_h_vl - 0.35, "OUTPUT", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2E7D32')
    ax_val.text(x_out_vl + io_w_vl/2, box_h_vl/2 - 0.30,
               "Metrik Train/Val/\nTest + Rata-rata &\nStd 5-Fold CV",
               ha='center', va='center', fontsize=8.5, color='#333')

    ax_val.set_xlim(-0.2, x_out_vl + io_w_vl + 0.2)
    ax_val.set_ylim(-0.2, box_h_vl + 0.2)
    plt.tight_layout()
    st.pyplot(fig_val, use_container_width=True)
    plt.close()

    st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
    col_vlo1, col_vlo2, col_vlo3 = st.columns(3)
    col_vlo1.metric("Training Set", f"{res['n_train_orig']:,}")
    col_vlo2.metric("Validation Set", f"{res['n_val']:,}", f"{val_pct}%")
    col_vlo3.metric("Test Set", f"{res['n_test']:,}")
    st.caption(
        "Hasil metrik lengkap per kernel (Train vs Validation vs Test, dan "
        "5-Fold Cross-Validation) dapat dilihat pada sub-tab di bawah."
    )

    st.divider()

    # ── Info ukuran data ───────────────────────────────────────────────────────
    st.markdown("**📋 Pembagian Data untuk Validasi**")
    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Training Set",    f"{res['n_train_orig']:,}")
    d2.metric("Validation Set",  f"{res['n_val']:,}", f"{val_pct}% dari training")
    d3.metric("Test Set",        f"{res['n_test']:,}", "Hold-out")
    if res['pca_info']:
        d4.metric("Komponen PCA", res['pca_info']['n_comp'])
    else:
        d4.metric("Mode", "Tanpa PCA")

    st.divider()

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    stab1, stab2, stab3, stab4, stab5 = st.tabs([
        "📊 Ringkasan Validasi",
        "📈 Train vs Val vs Test",
        "🔍 CV per Fold",
        "🟦 Confusion Matrix Validasi",
        "⚠️ Deteksi Overfitting",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Ringkasan Validasi
    # ══════════════════════════════════════════════════════════════════════════
    with stab1:
        st.markdown("**📊 Tabel Ringkasan Semua Kernel — Train / Validation / Test**")

        rows = []
        for kernel in KERNELS:
            r = kr[kernel]
            rows.append({
                'Kernel':           kernel.upper(),
                'CV F1 (mean)':     f"{r['cv_f1_mean']:.4f}",
                'CV F1 (±std)':     f"±{r['cv_f1_std']:.4f}",
                'Train Acc':        f"{r['train']['accuracy']*100:.2f}%",
                'Val Acc':          f"{r['val']['accuracy']*100:.2f}%",
                'Test Acc':         f"{r['test']['accuracy']*100:.2f}%",
                'Train F1':         f"{r['train']['f1']:.4f}",
                'Val F1':           f"{r['val']['f1']:.4f}",
                'Test F1':          f"{r['test']['f1']:.4f}",
                'Gap Train-Val':    f"{r['gap_train_val']*100:+.2f}%",
                'Status':           r['overfit_status'],
            })
        summary_df = pd.DataFrame(rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        st.divider()

        # Visual ringkasan status
        st.markdown("**📌 Status Fit per Kernel**")
        col_s = st.columns(len(KERNELS))
        for col, kernel in zip(col_s, KERNELS):
            r = kr[kernel]
            status = r['overfit_status']
            with col:
                if "✅" in status:
                    st.success(f"**{kernel.upper()}**\n\n{status}")
                else:
                    st.warning(f"**{kernel.upper()}**\n\n{status}")
                st.markdown(
                    f"Train: `{r['train']['accuracy']*100:.2f}%`\n\n"
                    f"Val: `{r['val']['accuracy']*100:.2f}%`\n\n"
                    f"Test: `{r['test']['accuracy']*100:.2f}%`"
                )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Train vs Val vs Test
    # ══════════════════════════════════════════════════════════════════════════
    with stab2:
        st.markdown("**📈 Perbandingan Metrik: Train vs Validation vs Test**")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Perbandingan Performa Train / Validation / Test\n{use_pca} | Rasio {s_used}',
            fontsize=14, fontweight='bold'
        )
        for ax, metric, title, ylabel in zip(
            axes.flatten(),
            ['accuracy','precision','recall','f1'],
            ['Accuracy','Precision','Recall','F1-Score'],
            ['Accuracy','Precision','Recall','F1-Score'],
        ):
            _bar3(ax, kr, metric, title, ylabel)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

        # Line chart tren per kernel
        st.markdown("**📈 Tren Accuracy: Train → Validation → Test per Kernel**")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        stages = ['Train', 'Validation', 'Test']
        for kernel in KERNELS:
            r = kr[kernel]
            vals = [r['train']['accuracy'],
                    r['val']['accuracy'],
                    r['test']['accuracy']]
            ax2.plot(stages, vals, 'o-', label=kernel.upper(),
                     color=CLR_K[kernel], linewidth=2.5, markersize=9)
            for i, v in enumerate(vals):
                ax2.annotate(f'{v*100:.2f}%', (stages[i], v),
                             textcoords='offset points', xytext=(0, 10),
                             ha='center', fontsize=9, fontweight='bold',
                             color=CLR_K[kernel])
        lo = min(kr[k][s]['accuracy']
                 for k in KERNELS for s in ['train','val','test'])
        ax2.set_ylim(max(0, lo-0.08), 1.08)
        ax2.set_ylabel('Accuracy', fontsize=11)
        ax2.set_title('Tren Accuracy: Train → Validation → Test',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.divider()

        # Line chart F1
        st.markdown("**📈 Tren F1-Score: Train → Validation → Test per Kernel**")
        fig3, ax3 = plt.subplots(figsize=(11, 5))
        for kernel in KERNELS:
            r = kr[kernel]
            vals = [r['train']['f1'], r['val']['f1'], r['test']['f1']]
            ax3.plot(stages, vals, 's--', label=kernel.upper(),
                     color=CLR_K[kernel], linewidth=2.5, markersize=9)
            for i, v in enumerate(vals):
                ax3.annotate(f'{v:.4f}', (stages[i], v),
                             textcoords='offset points', xytext=(0, -16),
                             ha='center', fontsize=9, fontweight='bold',
                             color=CLR_K[kernel])
        lo2 = min(kr[k][s]['f1'] for k in KERNELS for s in ['train','val','test'])
        ax3.set_ylim(max(0, lo2-0.08), 1.08)
        ax3.set_ylabel('F1-Score', fontsize=11)
        ax3.set_title('Tren F1-Score: Train → Validation → Test',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10); ax3.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — CV per Fold
    # ══════════════════════════════════════════════════════════════════════════
    with stab3:
        st.markdown("**🔍 Hasil 5-Fold Cross-Validation per Kernel**")
        st.caption(
            "Setiap kernel diuji menggunakan StratifiedKFold 5-fold "
            "pada data training. Nilai CV yang konsisten (std kecil) "
            "menunjukkan model yang stabil."
        )

        # Tabel CV per fold per kernel
        cv_rows = []
        for kernel in KERNELS:
            r = kr[kernel]
            fold_scores = r['cv_scores_f1']
            row = {'Kernel': kernel.upper()}
            for i, s in enumerate(fold_scores, 1):
                row[f'Fold {i}'] = f"{s:.4f}"
            row['Mean'] = f"{r['cv_f1_mean']:.4f}"
            row['Std']  = f"±{r['cv_f1_std']:.4f}"
            cv_rows.append(row)
        st.dataframe(pd.DataFrame(cv_rows),
                     use_container_width=True, hide_index=True)

        st.divider()

        # Bar chart CV per fold
        st.markdown("**📊 Visualisasi CV F1 per Fold — Semua Kernel**")
        fig4, axes4 = plt.subplots(2, 2, figsize=(14, 8))
        fig4.suptitle('CV F1-Score per Fold (StratifiedKFold 5-fold)',
                      fontsize=13, fontweight='bold')
        for ax, kernel in zip(axes4.flatten(), KERNELS):
            r     = kr[kernel]
            folds = [f'Fold {i}' for i in range(1, 6)]
            scores= r['cv_scores_f1']
            bars  = ax.bar(folds, scores, color=CLR_K[kernel],
                           alpha=0.85, edgecolor='white')
            ax.axhline(y=r['cv_f1_mean'], color='red', linestyle='--',
                       linewidth=1.5, label=f"Mean={r['cv_f1_mean']:.4f}")
            ax.fill_between(
                range(len(folds)),
                [r['cv_f1_mean']-r['cv_f1_std']]*5,
                [r['cv_f1_mean']+r['cv_f1_std']]*5,
                alpha=0.15, color='red', label=f"±Std={r['cv_f1_std']:.4f}"
            )
            for bar, v in zip(bars, scores):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.003,
                        f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
            ax.set_title(f'Kernel: {kernel.upper()}', fontweight='bold', fontsize=11)
            ax.set_ylabel('F1-Score', fontsize=10)
            lo = min(scores) - 0.05
            ax.set_ylim(max(0, lo), min(1.08, max(scores)+0.08))
            ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.divider()

        # Ringkasan stabilitas
        st.markdown("**📋 Ringkasan Stabilitas CV per Kernel**")
        stab_rows = []
        for kernel in KERNELS:
            r = kr[kernel]
            cv_scores = r['cv_scores_f1']
            stab_rows.append({
                'Kernel':       kernel.upper(),
                'CV F1 Mean':   f"{r['cv_f1_mean']:.4f}",
                'CV F1 Std':    f"{r['cv_f1_std']:.4f}",
                'CV F1 Min':    f"{min(cv_scores):.4f}",
                'CV F1 Max':    f"{max(cv_scores):.4f}",
                'Range':        f"{max(cv_scores)-min(cv_scores):.4f}",
                'Stabilitas':   '✅ Stabil' if r['cv_f1_std'] < 0.03
                                else ('⚠️ Cukup Stabil' if r['cv_f1_std'] < 0.06
                                      else '❌ Tidak Stabil'),
            })
        st.dataframe(pd.DataFrame(stab_rows),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — Confusion Matrix Validasi
    # ══════════════════════════════════════════════════════════════════════════
    with stab4:
        st.markdown("**🟦 Confusion Matrix pada Validation Set & Test Set**")

        kernel_cm = st.selectbox(
            "Pilih kernel:", [k.upper() for k in KERNELS], key="val_cm_kernel"
        )
        kernel_cm_lower = kernel_cm.lower()
        r = kr[kernel_cm_lower]

        col_val, col_test = st.columns(2)
        for col, data, title, cmap_n in [
            (col_val,  r['val'],  f"Validation Set\n(n={res['n_val']:,})",  'YlOrRd'),
            (col_test, r['test'], f"Test Set\n(n={res['n_test']:,})",        'Blues'),
        ]:
            with col:
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
                sns.heatmap(
                    data['cm'], annot=True, fmt='d', cmap=cmap_n,
                    xticklabels=classes, yticklabels=classes,
                    ax=ax_cm, linewidths=0.5, linecolor='gray',
                    annot_kws={'size': 14, 'weight': 'bold'}, cbar=False
                )
                ax_cm.set_xlabel('Predicted', fontsize=11)
                ax_cm.set_ylabel('True', fontsize=11)
                ax_cm.set_title(
                    f'{kernel_cm} — {title}\nAcc: {data["accuracy"]*100:.2f}%',
                    fontsize=11, fontweight='bold'
                )
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close()

        st.divider()

        # Tabel metrik Validation vs Test
        st.markdown(f"**📋 Perbandingan Metrik: Validation vs Test — Kernel {kernel_cm}**")
        cmp_rows = []
        for metric_name, key in [
            ('Accuracy (%)', 'accuracy'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('F1-Score', 'f1'),
        ]:
            vv = r['val'][key]
            vt = r['test'][key]
            delta = vv - vt
            cmp_rows.append({
                'Metrik':     metric_name,
                'Validation': f"{vv*100:.2f}%" if key=='accuracy' else f"{vv:.4f}",
                'Test':       f"{vt*100:.2f}%" if key=='accuracy' else f"{vt:.4f}",
                'Δ (Val−Test)': f"{delta*100:+.2f}%" if key=='accuracy' else f"{delta:+.4f}",
                'Konsisten?': '✅ Ya' if abs(delta) < 0.05 else '⚠️ Perlu Perhatian',
            })
        st.dataframe(pd.DataFrame(cmp_rows),
                     use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 5 — Deteksi Overfitting
    # ══════════════════════════════════════════════════════════════════════════
    with stab5:
        st.markdown("**⚠️ Analisis Overfitting / Underfitting**")
        st.caption(
            "Gap antara accuracy Training dan Validation/Test menunjukkan "
            "apakah model overfit (terlalu hafal data training) atau "
            "underfit (model terlalu sederhana)."
        )

        # Penjelasan konsep
        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            st.info(
                "**✅ Fit Baik:**\n\n"
                "Train Acc ≈ Val Acc ≈ Test Acc\n\n"
                "Gap Train-Val < 10%\n\n"
                "Model generalisasi dengan baik"
            )
        with col_o2:
            st.warning(
                "**⚠️ Overfitting:**\n\n"
                "Train Acc >> Val Acc\n\n"
                "Gap Train-Val > 10%\n\n"
                "Model terlalu hafal training"
            )
        with col_o3:
            st.error(
                "**❌ Underfitting:**\n\n"
                "Train Acc < 70%\n\n"
                "Model terlalu sederhana\n\n"
                "Perlu fitur/parameter lebih baik"
            )

        st.divider()

        # Tabel gap dan status
        st.markdown("**📊 Tabel Analisis Gap Train-Val-Test per Kernel**")
        gap_rows = []
        for kernel in KERNELS:
            r = kr[kernel]
            gap_tv = r['gap_train_val']
            gap_vt = r['gap_val_test']
            gap_rows.append({
                'Kernel':          kernel.upper(),
                'Train Acc':       f"{r['train']['accuracy']*100:.2f}%",
                'Val Acc':         f"{r['val']['accuracy']*100:.2f}%",
                'Test Acc':        f"{r['test']['accuracy']*100:.2f}%",
                'Δ Train-Val':     f"{gap_tv*100:+.2f}%",
                'Δ Val-Test':      f"{gap_vt*100:+.2f}%",
                'CV F1 Std':       f"{r['cv_f1_std']:.4f}",
                'Status':          r['overfit_status'],
            })
        st.dataframe(pd.DataFrame(gap_rows),
                     use_container_width=True, hide_index=True)

        st.divider()

        # Visualisasi gap
        st.markdown("**📊 Visualisasi Gap Accuracy: Train vs Val vs Test**")
        fig5, ax5 = plt.subplots(figsize=(11, 5))
        x  = np.arange(len(KERNELS)); w = 0.28
        tr_vals  = [kr[k]['train']['accuracy'] for k in KERNELS]
        val_vals = [kr[k]['val']['accuracy']   for k in KERNELS]
        te_vals  = [kr[k]['test']['accuracy']  for k in KERNELS]

        b1 = ax5.bar(x-w, tr_vals,  w, label='Train',      color=CLR_SET['Train'],      alpha=0.85, edgecolor='white')
        b2 = ax5.bar(x,   val_vals, w, label='Validation',  color=CLR_SET['Validation'], alpha=0.85, edgecolor='white')
        b3 = ax5.bar(x+w, te_vals,  w, label='Test',        color=CLR_SET['Test'],       alpha=0.85, edgecolor='white')

        for bars, vals in [(b1,tr_vals),(b2,val_vals),(b3,te_vals)]:
            for bar, v in zip(bars, vals):
                ax5.text(bar.get_x()+bar.get_width()/2, v+0.004,
                         f'{v*100:.1f}%', ha='center', va='bottom',
                         fontsize=8.5, fontweight='bold')

        ax5.set_xticks(x); ax5.set_xticklabels([k.upper() for k in KERNELS], fontsize=11)
        ax5.set_ylabel('Accuracy', fontsize=11)
        ax5.set_title('Perbandingan Accuracy: Train vs Validation vs Test\n'
                      '(Ideal: ketiga nilai berdekatan → tidak overfit)',
                      fontsize=12, fontweight='bold')
        lo = min(tr_vals+val_vals+te_vals) - 0.08
        ax5.set_ylim(max(0, lo), 1.1)
        ax5.legend(fontsize=10); ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        st.divider()

        # Kesimpulan
        st.markdown("**💡 Kesimpulan Validasi**")
        good_kernels = [k for k in KERNELS if '✅' in kr[k]['overfit_status']]
        warn_kernels = [k for k in KERNELS if '⚠️' in kr[k]['overfit_status']]

        best_test_k = max(KERNELS, key=lambda k: kr[k]['test']['f1'])
        best_cv_k   = max(KERNELS, key=lambda k: kr[k]['cv_f1_mean'])

        st.info(
            f"**Rangkuman Hasil Validasi ({use_pca} | Rasio {s_used}):**\n\n"
            f"- Data Training: **{res['n_train_orig']:,}** | "
            f"Validation: **{res['n_val']:,}** | Test: **{res['n_test']:,}**\n\n"
            f"- Kernel dengan status **Fit Baik**: "
            f"{', '.join([k.upper() for k in good_kernels]) if good_kernels else 'Tidak ada'}\n\n"
            f"- Kernel yang perlu perhatian: "
            f"{', '.join([k.upper() for k in warn_kernels]) if warn_kernels else 'Tidak ada'}\n\n"
            f"- Kernel dengan **Test F1 terbaik**: `{best_test_k.upper()}` "
            f"(F1={kr[best_test_k]['test']['f1']:.4f}, "
            f"Acc={kr[best_test_k]['test']['accuracy']*100:.2f}%)\n\n"
            f"- Kernel dengan **CV F1 paling stabil**: `{best_cv_k.upper()}` "
            f"(mean={kr[best_cv_k]['cv_f1_mean']:.4f} ±{kr[best_cv_k]['cv_f1_std']:.4f})\n\n"
            f"- Model terbukti **dapat digeneralisasi** apabila selisih "
            f"Train-Val-Test < 10% dan CV std < 0.03."
        )
