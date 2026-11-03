"""
tabs/tab_confusion.py
Tab: Confusion Matrix + Perhitungan Manual Metrik (Revisi Poin 3)
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _manual_metrics(cm, classes):
    """Hitung metrik performa secara manual dari confusion matrix."""
    rows = []
    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        precision  = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall     = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1         = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
        specificity= TN / (TN + FP) if (TN + FP) > 0 else 0
        rows.append({
            'Kelas': cls, 'TP': int(TP), 'FP': int(FP),
            'FN': int(FN), 'TN': int(TN),
            'Precision': precision, 'Recall': recall,
            'F1-Score': f1, 'Specificity': specificity,
        })
    total = cm.sum()
    acc = np.trace(cm) / total
    return rows, float(acc)


def render(df=None, **kwargs):
    st.subheader("🟦 Confusion Matrix & Perhitungan Manual Metrik")
    st.caption(
        "Confusion matrix disimulasikan dengan perhitungan manual untuk menghasilkan "
        "metrik-metrik performansi. (Revisi dosen penguji poin 3)"
    )

    if 'r_pca' not in st.session_state:
        st.info("Jalankan pipeline terlebih dahulu.", icon="👈")
        return

    r_pca   = st.session_state['r_pca']
    r_nopca = st.session_state['r_nopca']
    classes = st.session_state['feat']['classes']
    k_used  = st.session_state['kernel'].upper()
    C_used     = st.session_state.get('C_param', 1)
    gamma_used = st.session_state.get('gamma_param', '0.001')
    degree_used= st.session_state.get('degree_param', 2)
    param_str  = f"C={C_used}, γ={gamma_used}"
    if st.session_state['kernel'] == 'poly':
        param_str += f", degree={degree_used}"
    s_used  = st.session_state['split_label']

    st.markdown(f"**Kernel:** `{k_used}` | **Parameter:** `{param_str}` | **Rasio:** `{s_used}`")

    # ── Diagram Langkah Confusion Matrix ────────────────────────────────────
    st.markdown("**🧭 Diagram Proses Confusion Matrix & Perhitungan Metrik**")
    st.caption(
        "Confusion matrix membandingkan label aktual dengan hasil prediksi "
        "model, melalui 3 langkah berikut, diapit kotak INPUT (data masuk) "
        "dan OUTPUT (hasil akhir)."
    )

    r_active = r_pca if 'r_pca' in st.session_state else r_nopca
    n_test_cm = r_nopca.get('n_test', 0)

    fig_cm_diag, ax_cm_diag = plt.subplots(figsize=(13.5, 3.4))
    ax_cm_diag.axis('off')
    box_w_c, box_h_c, gap_c, io_w_c = 2.7, 1.8, 0.5, 2.5

    steps_cm = [
        ("1. Bandingkan\ny_true vs y_pred", "Cocokkan label aktual\ndengan label hasil\nprediksi model SVM"),
        ("2. Susun Matriks\nTP/FP/FN/TN", "Hitung jumlah prediksi\nbenar & salah per\nkelas (2×2 matriks)"),
        ("3. Hitung Metrik\nTurunan", "Accuracy, Precision,\nRecall, F1-Score dari\nTP/FP/FN/TN"),
    ]
    colors_cm = ['#E6F1FB', '#FAEEDA', '#EAF3DE']
    edges_cm  = ['#185FA5', '#854F0B', '#3B6D11']

    # Kotak INPUT
    x_in_c = 0
    rect_in_c = plt.Rectangle((x_in_c, 0), io_w_c, box_h_c, facecolor='#F0F0F0',
                               edgecolor='#555555', linewidth=1.6, linestyle='--', zorder=2)
    ax_cm_diag.add_patch(rect_in_c)
    ax_cm_diag.text(x_in_c + io_w_c/2, box_h_c - 0.35, "INPUT", ha='center', va='top',
                    fontsize=10, fontweight='bold', color='#333333')
    ax_cm_diag.text(x_in_c + io_w_c/2, box_h_c/2 - 0.30,
                    f"y_true & y_pred\n{n_test_cm:,} data uji\n(hasil prediksi SVM)",
                    ha='center', va='center', fontsize=8.5, color='#333')
    ax_cm_diag.annotate('', xy=(x_in_c + io_w_c + gap_c - 0.1, box_h_c/2),
                        xytext=(x_in_c + io_w_c + 0.1, box_h_c/2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak proses
    x_offset_c = x_in_c + io_w_c + gap_c
    for i, ((title, desc), c, ec) in enumerate(zip(steps_cm, colors_cm, edges_cm)):
        x = x_offset_c + i * (box_w_c + gap_c)
        rect = plt.Rectangle((x, 0), box_w_c, box_h_c, facecolor=c,
                              edgecolor=ec, linewidth=1.5, zorder=2)
        ax_cm_diag.add_patch(rect)
        ax_cm_diag.text(x + box_w_c/2, box_h_c - 0.35, title, ha='center', va='top',
                        fontsize=9.5, fontweight='bold', color=ec)
        ax_cm_diag.text(x + box_w_c/2, box_h_c/2 - 0.25, desc, ha='center', va='center',
                        fontsize=8, color='#333')
        ax_cm_diag.annotate('', xy=(x + box_w_c + gap_c - 0.1, box_h_c/2),
                            xytext=(x + box_w_c + 0.1, box_h_c/2),
                            arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak OUTPUT
    x_out_c = x_offset_c + len(steps_cm) * (box_w_c + gap_c)
    rect_out_c = plt.Rectangle((x_out_c, 0), io_w_c, box_h_c, facecolor='#E1F5E1',
                                edgecolor='#2E7D32', linewidth=1.8, zorder=2)
    ax_cm_diag.add_patch(rect_out_c)
    ax_cm_diag.text(x_out_c + io_w_c/2, box_h_c - 0.35, "OUTPUT", ha='center', va='top',
                    fontsize=10, fontweight='bold', color='#2E7D32')
    ax_cm_diag.text(x_out_c + io_w_c/2, box_h_c/2 - 0.30,
                    f"Matriks 2×2 +\nAcc={r_active['accuracy']*100:.1f}%\nF1={r_active['f1']:.4f}",
                    ha='center', va='center', fontsize=8.5, color='#333')

    ax_cm_diag.set_xlim(-0.2, x_out_c + io_w_c + 0.2)
    ax_cm_diag.set_ylim(-0.2, box_h_c + 0.2)
    plt.tight_layout()
    st.pyplot(fig_cm_diag, use_container_width=True)
    plt.close()

    st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
    col_cmo1, col_cmo2, col_cmo3, col_cmo4 = st.columns(4)
    col_cmo1.metric("Accuracy", f"{r_active['accuracy']*100:.2f}%")
    col_cmo2.metric("Precision", f"{r_active['precision']:.4f}")
    col_cmo3.metric("Recall", f"{r_active['recall']:.4f}")
    col_cmo4.metric("F1-Score", f"{r_active['f1']:.4f}")
    st.caption(
        f"Confusion matrix dibangun dari {n_test_cm:,} data uji — lihat "
        f"matriks lengkap dan perhitungan manual TP/FP/FN/TN pada sub-tab "
        f"di bawah."
    )

    st.divider()

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    stab1, stab2, stab3 = st.tabs([
        "🟦 Confusion Matrix",
        "🧮 Perhitungan Manual",
        "📊 Normalisasi",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Confusion Matrix
    # ══════════════════════════════════════════════════════════════════════════
    with stab1:
        col_pca, col_nopca = st.columns(2)
        for col, r, title, cmap_name in [
            (col_pca,   r_pca,   f"Dengan PCA\nAcc: {r_pca['accuracy']*100:.2f}%",   'Blues'),
            (col_nopca, r_nopca, f"Tanpa PCA\nAcc: {r_nopca['accuracy']*100:.2f}%",  'Oranges'),
        ]:
            with col:
                st.markdown(f"**{'✅ Dengan PCA' if 'PCA' in title else '🔴 Tanpa PCA'}**")
                fig, ax = plt.subplots(figsize=(5, 4.5))
                sns.heatmap(r['cm'], annot=True, fmt='d', cmap=cmap_name,
                            xticklabels=classes, yticklabels=classes,
                            ax=ax, linewidths=0.5, linecolor='gray',
                            annot_kws={'size':14,'weight':'bold'}, cbar=False)
                ax.set_xlabel('Predicted Label', fontsize=11)
                ax.set_ylabel('True Label', fontsize=11)
                ax.set_title(title, fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Perhitungan Manual
    # ══════════════════════════════════════════════════════════════════════════
    with stab2:
        st.markdown("**🧮 Simulasi Perhitungan Manual Metrik dari Confusion Matrix**")
        st.caption(
            "Setiap metrik dihitung langsung dari nilai TP, FP, FN, TN pada confusion matrix "
            "tanpa menggunakan fungsi sklearn, sehingga dapat diverifikasi secara manual."
        )

        mode_sel = st.radio("Pilih mode:", ["Dengan PCA", "Tanpa PCA"], horizontal=True)
        r_sel    = r_pca if mode_sel == "Dengan PCA" else r_nopca
        cm       = r_sel['cm']

        # ── Tampilkan confusion matrix nilai ──────────────────────────────────
        st.markdown("**Step 1: Confusion Matrix**")
        cm_df = pd.DataFrame(cm, index=[f"Aktual {c}" for c in classes],
                             columns=[f"Prediksi {c}" for c in classes])
        st.dataframe(cm_df.style.background_gradient(cmap='Blues', axis=None)
                     .format("{:,}"), use_container_width=False)

        st.divider()

        # ── Hitung TP, FP, FN, TN per kelas ──────────────────────────────────
        st.markdown("**Step 2: Ekstraksi TP, FP, FN, TN per Kelas**")

        manual_rows, acc_manual = _manual_metrics(cm, classes)

        formula_rows = []
        for row in manual_rows:
            cls = row['Kelas']
            formula_rows.append({
                'Kelas': cls,
                'TP': f"{row['TP']}",
                'FP': f"{row['FP']}",
                'FN': f"{row['FN']}",
                'TN': f"{row['TN']}",
                'Formula TP':  f"Prediksi {cls} BENAR dan Aktual {cls}",
                'Formula FP':  f"Prediksi {cls} tapi Aktual BUKAN {cls}",
                'Formula FN':  f"Aktual {cls} tapi Prediksi BUKAN {cls}",
                'Formula TN':  f"Prediksi BUKAN {cls} dan Aktual BUKAN {cls}",
            })
        st.dataframe(pd.DataFrame(formula_rows)[['Kelas','TP','FP','FN','TN']],
                     use_container_width=True, hide_index=True)

        st.divider()

        # ── Perhitungan metrik dengan formula eksplisit ────────────────────────
        st.markdown("**Step 3: Perhitungan Manual Metrik Performansi**")

        for row in manual_rows:
            cls = row['Kelas']
            TP, FP, FN, TN = row['TP'], row['FP'], row['FN'], row['TN']
            prec = row['Precision']
            rec  = row['Recall']
            f1   = row['F1-Score']
            spec = row['Specificity']

            with st.expander(f"📌 Kelas: **{cls}** — TP={TP}, FP={FP}, FN={FN}, TN={TN}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Precision:**")
                    st.latex(r"\text{Precision} = \frac{TP}{TP + FP} = \frac{" +
                             f"{TP}" + r"}{" + f"{TP}+{FP}" + r"} = \frac{" +
                             f"{TP}" + r"}{" + f"{TP+FP}" + r"} = " +
                             f"{prec:.4f}")
                    st.markdown("**Recall (Sensitivity):**")
                    st.latex(r"\text{Recall} = \frac{TP}{TP + FN} = \frac{" +
                             f"{TP}" + r"}{" + f"{TP}+{FN}" + r"} = \frac{" +
                             f"{TP}" + r"}{" + f"{TP+FN}" + r"} = " +
                             f"{rec:.4f}")
                with col2:
                    st.markdown("**F1-Score:**")
                    st.latex(r"\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}"
                             r"{\text{Precision} + \text{Recall}} = \frac{2 \times " +
                             f"{prec:.4f}" + r" \times " + f"{rec:.4f}" + r"}{" +
                             f"{prec:.4f}+{rec:.4f}" + r"} = " + f"{f1:.4f}")
                    st.markdown("**Specificity:**")
                    st.latex(r"\text{Specificity} = \frac{TN}{TN + FP} = \frac{" +
                             f"{TN}" + r"}{" + f"{TN}+{FP}" + r"} = " +
                             f"{spec:.4f}")

        st.divider()

        # ── Accuracy keseluruhan ───────────────────────────────────────────────
        st.markdown("**Step 4: Accuracy Keseluruhan**")
        total     = int(cm.sum())
        correct   = int(np.trace(cm))
        incorrect = total - correct
        st.latex(r"\text{Accuracy} = \frac{\sum TP}{\text{Total Data}} = \frac{" +
                 f"{correct}" + r"}{" + f"{total}" + r"} = " + f"{acc_manual:.4f}" +
                 r" \approx " + f"{acc_manual*100:.2f}\\%")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data Test",      f"{total:,}")
        c2.metric("Prediksi Benar (TP)",  f"{correct:,}")
        c3.metric("Prediksi Salah",       f"{incorrect:,}")

        st.divider()

        # ── Tabel ringkasan semua metrik ───────────────────────────────────────
        st.markdown("**Step 5: Rangkuman Semua Metrik (Manual)**")
        summary = pd.DataFrame([{
            'Kelas':       row['Kelas'],
            'TP':          row['TP'],
            'FP':          row['FP'],
            'FN':          row['FN'],
            'TN':          row['TN'],
            'Precision':   f"{row['Precision']:.4f}",
            'Recall':      f"{row['Recall']:.4f}",
            'F1-Score':    f"{row['F1-Score']:.4f}",
            'Specificity': f"{row['Specificity']:.4f}",
        } for row in manual_rows])
        summary.loc[len(summary)] = {
            'Kelas': 'Accuracy', 'TP': '-', 'FP': '-', 'FN': '-', 'TN': '-',
            'Precision': '-', 'Recall': '-',
            'F1-Score': '-', 'Specificity': f"{acc_manual:.4f}",
        }
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.info(
            "✅ Semua metrik di atas dihitung secara manual dari confusion matrix "
            "menggunakan formula standar, bukan menggunakan fungsi sklearn, "
            "sehingga dapat diverifikasi langsung."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Normalisasi
    # ══════════════════════════════════════════════════════════════════════════
    with stab3:
        st.markdown("**Confusion Matrix Normalisasi (%)**")
        col_n1, col_n2 = st.columns(2)
        for col, r, title, cmap_n in [
            (col_n1, r_pca,   "Normalisasi — Dengan PCA",   'Blues'),
            (col_n2, r_nopca, "Normalisasi — Tanpa PCA",    'Oranges'),
        ]:
            cm_norm = r['cm'].astype('float') / r['cm'].sum(axis=1)[:,np.newaxis] * 100
            with col:
                fig, ax = plt.subplots(figsize=(5, 4.5))
                sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap=cmap_n,
                            xticklabels=classes, yticklabels=classes,
                            ax=ax, linewidths=0.5, linecolor='gray',
                            annot_kws={'size':13}, cbar=False)
                ax.set_xlabel('Predicted Label', fontsize=11)
                ax.set_ylabel('True Label', fontsize=11)
                ax.set_title(title, fontsize=12, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
