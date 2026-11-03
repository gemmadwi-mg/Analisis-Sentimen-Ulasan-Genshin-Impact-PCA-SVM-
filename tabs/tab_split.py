"""
tabs/tab_split.py
Tab: Split Data (terpisah dari SMOTE)
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def render(df=None, **kwargs):
    """Render tab Split Data."""
    st.subheader("✂️ Hasil Split Data")
    st.caption("Pembagian data training dan testing untuk dokumentasi skripsi")

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    feat        = st.session_state['feat']
    classes     = feat['classes']
    split_label = st.session_state['split_label']

    y_all   = np.concatenate([feat['y_train_ori'], feat['y_test']])
    y_train = feat['y_train_ori']
    y_test  = feat['y_test']

    n_total = feat['n_total']
    n_train = len(y_train)
    n_test  = len(y_test)
    n_feat_dim = feat.get('n_features', feat.get('k_features', '?'))

    # ── Diagram Langkah Split Data ─────────────────────────────────────────
    st.markdown("**🧭 Diagram Proses Split Data**")
    st.caption(
        "Data dibagi menjadi data training dan testing menggunakan stratified "
        "split, diapit kotak INPUT (data masuk) dan OUTPUT (hasil pembagian)."
    )

    fig_sp, ax_sp = plt.subplots(figsize=(11, 3.2))
    ax_sp.axis('off')
    box_w_sp, box_h_sp, gap_sp, io_w_sp = 3.4, 1.8, 0.6, 2.4

    # Kotak INPUT
    x_in = 0
    rect_in = plt.Rectangle((x_in, 0), io_w_sp, box_h_sp, facecolor='#F0F0F0',
                             edgecolor='#555555', linewidth=1.6, linestyle='--', zorder=2)
    ax_sp.add_patch(rect_in)
    ax_sp.text(x_in + io_w_sp/2, box_h_sp - 0.35, "INPUT", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#333333')
    ax_sp.text(x_in + io_w_sp/2, box_h_sp/2 - 0.30,
               f"Matriks Fitur\n{n_total:,} × {n_feat_dim}\n(seluruh dataset)",
               ha='center', va='center', fontsize=8.5, color='#333')
    ax_sp.annotate('', xy=(x_in + io_w_sp + gap_sp - 0.1, box_h_sp/2),
                   xytext=(x_in + io_w_sp + 0.1, box_h_sp/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak proses
    x_proc = x_in + io_w_sp + gap_sp
    rect_proc = plt.Rectangle((x_proc, 0), box_w_sp, box_h_sp, facecolor='#E6F1FB',
                               edgecolor='#185FA5', linewidth=1.5, zorder=2)
    ax_sp.add_patch(rect_proc)
    ax_sp.text(x_proc + box_w_sp/2, box_h_sp - 0.35, "Stratified Split", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#185FA5')
    ax_sp.text(x_proc + box_w_sp/2, box_h_sp/2 - 0.30,
               f"train_test_split()\nrasio {split_label}\nstratify=label sentimen",
               ha='center', va='center', fontsize=8.5, color='#333')
    ax_sp.annotate('', xy=(x_proc + box_w_sp + gap_sp - 0.1, box_h_sp/2),
                   xytext=(x_proc + box_w_sp + 0.1, box_h_sp/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak OUTPUT (dipecah 2: train & test, ditumpuk vertikal)
    x_out = x_proc + box_w_sp + gap_sp
    out_h = (box_h_sp - 0.2) / 2
    rect_out1 = plt.Rectangle((x_out, out_h + 0.2), io_w_sp, out_h, facecolor='#E1F5E1',
                               edgecolor='#2E7D32', linewidth=1.8, zorder=2)
    ax_sp.add_patch(rect_out1)
    ax_sp.text(x_out + io_w_sp/2, out_h + 0.2 + out_h/2,
               f"OUTPUT: Training\n{n_train:,} × {n_feat_dim}",
               ha='center', va='center', fontsize=8, fontweight='bold', color='#2E7D32')

    rect_out2 = plt.Rectangle((x_out, 0), io_w_sp, out_h, facecolor='#FCEBEB',
                               edgecolor='#A32D2D', linewidth=1.8, zorder=2)
    ax_sp.add_patch(rect_out2)
    ax_sp.text(x_out + io_w_sp/2, out_h/2,
               f"OUTPUT: Testing\n{n_test:,} × {n_feat_dim}",
               ha='center', va='center', fontsize=8, fontweight='bold', color='#A32D2D')

    ax_sp.set_xlim(-0.2, x_out + io_w_sp + 0.2)
    ax_sp.set_ylim(-0.2, box_h_sp + 0.2)
    plt.tight_layout()
    st.pyplot(fig_sp, use_container_width=True)
    plt.close()

    st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
    col_spo1, col_spo2, col_spo3 = st.columns(3)
    col_spo1.metric("Data Training", f"{n_train:,}", f"{n_train/n_total*100:.1f}%")
    col_spo2.metric("Data Testing", f"{n_test:,}", f"{n_test/n_total*100:.1f}%")
    col_spo3.metric("Rasio Aktif", split_label)
    st.caption(
        f"Stratified split memastikan proporsi kelas POSITIVE/NEGATIVE pada "
        f"data training dan testing tetap sebanding dengan proporsi pada "
        f"keseluruhan {n_total:,} data — lihat tabel distribusi kelas di bawah."
    )

    st.divider()
    st.markdown("**📋 Statistik Ringkas Split Data**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Data",     f"{n_total:,}")
    k2.metric("Data Training",  f"{n_train:,}", f"{n_train/n_total*100:.1f}%")
    k3.metric("Data Testing",   f"{n_test:,}",  f"{n_test/n_total*100:.1f}%")

    st.divider()
    st.markdown(f"**Rasio Split Aktif: `{split_label}`**")

    # ── Tabel distribusi per kelas ────────────────────────────────────────
    st.markdown("**📄 Distribusi Kelas pada Setiap Subset Data**")
    tbl_rows = []
    for cls_idx, cls_name in enumerate(classes):
        cnt_all   = int((y_all == cls_idx).sum())
        cnt_train = int((y_train == cls_idx).sum())
        cnt_test  = int((y_test == cls_idx).sum())
        tbl_rows.append({
            'Kelas':         cls_name,
            'Total Dataset': f"{cnt_all:,}",
            '% Total':       f"{cnt_all/n_total*100:.2f}%",
            'Data Training': f"{cnt_train:,}",
            '% Training':    f"{cnt_train/n_train*100:.2f}%",
            'Data Testing':  f"{cnt_test:,}",
            '% Testing':     f"{cnt_test/n_test*100:.2f}%",
        })
    tbl_rows.append({
        'Kelas':         'TOTAL',
        'Total Dataset': f"{n_total:,}",
        '% Total':       '100.00%',
        'Data Training': f"{n_train:,}",
        '% Training':    '100.00%',
        'Data Testing':  f"{n_test:,}",
        '% Testing':     '100.00%',
    })
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Bar chart split ────────────────────────────────────────────────────
    st.markdown("**📊 Visualisasi Distribusi Split Data per Kelas**")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    subsets = [y_all, y_train, y_test]
    titles  = [f"Dataset Total\n(n={n_total:,})",
               f"Data Training\n(n={n_train:,})",
               f"Data Testing\n(n={n_test:,})"]
    bar_colors = ['#4C72B0', '#55A868', '#C44E52']
    for ax, y_s, title, color in zip(axes, subsets, titles, bar_colors):
        counts = [int((y_s == i).sum()) for i in range(len(classes))]
        bars = ax.bar(classes, counts, color=color, alpha=0.85, edgecolor='white', width=0.5)
        for bar, v in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, v + max(counts)*0.02,
                    f'{v:,}', ha='center', fontweight='bold', fontsize=11)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_ylabel("Jumlah Sampel")
        ax.set_ylim(0, max(counts)*1.2)
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.divider()

    # ── Pie chart proporsi ────────────────────────────────────────────────
    st.markdown("**🥧 Proporsi Train vs Test**")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sizes  = [n_train, n_test]
    labels = [f"Training\n{n_train:,} ({n_train/n_total*100:.1f}%)",
              f"Testing\n{n_test:,} ({n_test/n_total*100:.1f}%)"]
    ax2.pie(sizes, labels=labels, colors=['#4C72B0', '#E24B4A'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    ax2.set_title(f"Proporsi Split Data\n(Rasio {split_label})", fontweight='bold')
    col_pie, _ = st.columns([1, 2])
    with col_pie:
        st.pyplot(fig2, use_container_width=True)
    plt.close()

    st.divider()

    # ── Tabel semua rasio split (ukuran data saja, tanpa SMOTE) ─────────────
    st.markdown("**📄 Ukuran Data Training & Testing — Semua Rasio Split**")
    st.caption("Perbandingan jumlah data pada seluruh rasio split yang tersedia di sidebar.")
    all_ratios = {"70:30": 0.7, "80:20": 0.8, "90:10": 0.9}
    sum_rows = []
    for lbl, ratio in all_ratios.items():
        n_tr = round(n_total * ratio)
        n_te = n_total - n_tr
        is_active = " ⭐ (aktif)" if lbl == split_label else ""
        sum_rows.append({
            'Rasio Split':   lbl + is_active,
            'Data Training': f"{n_tr:,}",
            'Data Testing':  f"{n_te:,}",
            '% Training':    f"{n_tr/n_total*100:.0f}%",
            '% Testing':     f"{n_te/n_total*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)
