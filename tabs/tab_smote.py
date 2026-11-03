"""
tabs/tab_smote.py
Tab: SMOTE (terpisah dari Split Data)
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.pipeline import smote_distribution_multi_ratio


def render(df=None, **kwargs):
    """Render tab SMOTE."""
    st.subheader("⚖️ Hasil SMOTE (Synthetic Minority Over-sampling Technique)")
    st.caption("Penyeimbangan kelas pada data training untuk dokumentasi skripsi")

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    feat        = st.session_state['feat']
    classes     = feat['classes']
    split_label = st.session_state['split_label']

    y_train    = feat['y_train_ori']
    y_train_sm = feat['y_train_sm']

    n_train    = len(y_train)
    n_train_sm = len(y_train_sm)
    n_added    = n_train_sm - n_train
    inc_pct    = (n_added / n_train) * 100 if n_train > 0 else 0

    n_feat_dim_sm = feat.get('n_features', feat.get('k_features', '?'))

    # ── Diagram Langkah SMOTE ───────────────────────────────────────────────
    st.markdown("**🧭 Diagram Proses SMOTE**")
    st.caption(
        "SMOTE membangkitkan data sintetis pada kelas minoritas melalui "
        "interpolasi k-nearest-neighbors, melalui 3 langkah berikut, diapit "
        "kotak INPUT (data masuk) dan OUTPUT (hasil akhir)."
    )

    fig_sm, ax_sm = plt.subplots(figsize=(14.5, 3.4))
    ax_sm.axis('off')
    box_w_s, box_h_s, gap_s, io_w_s = 2.7, 1.8, 0.5, 2.4

    steps_smote = [
        ("1. Identifikasi\nKelas Minoritas", "Bandingkan jumlah\nsampel tiap kelas\npada data training"),
        ("2. Cari k-Nearest\nNeighbors", "Untuk tiap sampel\nminoritas, cari k\ntetangga terdekat"),
        ("3. Interpolasi\nData Sintetis", "x_baru = x + gap ×\n(x_tetangga − x),\ngap acak [0,1]"),
    ]
    colors_sm = ['#E6F1FB', '#FAEEDA', '#EAF3DE']
    edges_sm  = ['#185FA5', '#854F0B', '#3B6D11']

    # Kotak INPUT
    x_in_s = 0
    rect_in_s = plt.Rectangle((x_in_s, 0), io_w_s, box_h_s, facecolor='#F0F0F0',
                               edgecolor='#555555', linewidth=1.6, linestyle='--', zorder=2)
    ax_sm.add_patch(rect_in_s)
    ax_sm.text(x_in_s + io_w_s/2, box_h_s - 0.35, "INPUT", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#333333')
    ax_sm.text(x_in_s + io_w_s/2, box_h_s/2 - 0.30,
               f"Data Training\n{n_train:,} × {n_feat_dim_sm}\n(belum seimbang)",
               ha='center', va='center', fontsize=8.5, color='#333')
    ax_sm.annotate('', xy=(x_in_s + io_w_s + gap_s - 0.1, box_h_s/2),
                   xytext=(x_in_s + io_w_s + 0.1, box_h_s/2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak proses
    x_offset_s = x_in_s + io_w_s + gap_s
    for i, ((title, desc), c, ec) in enumerate(zip(steps_smote, colors_sm, edges_sm)):
        x = x_offset_s + i * (box_w_s + gap_s)
        rect = plt.Rectangle((x, 0), box_w_s, box_h_s, facecolor=c,
                              edgecolor=ec, linewidth=1.5, zorder=2)
        ax_sm.add_patch(rect)
        ax_sm.text(x + box_w_s/2, box_h_s - 0.35, title, ha='center', va='top',
                   fontsize=9.5, fontweight='bold', color=ec)
        ax_sm.text(x + box_w_s/2, box_h_s/2 - 0.25, desc, ha='center', va='center',
                   fontsize=8, color='#333')
        ax_sm.annotate('', xy=(x + box_w_s + gap_s - 0.1, box_h_s/2),
                       xytext=(x + box_w_s + 0.1, box_h_s/2),
                       arrowprops=dict(arrowstyle='->', lw=2, color='#666'))

    # Kotak OUTPUT
    x_out_s = x_offset_s + len(steps_smote) * (box_w_s + gap_s)
    rect_out_s = plt.Rectangle((x_out_s, 0), io_w_s, box_h_s, facecolor='#E1F5E1',
                                edgecolor='#2E7D32', linewidth=1.8, zorder=2)
    ax_sm.add_patch(rect_out_s)
    ax_sm.text(x_out_s + io_w_s/2, box_h_s - 0.35, "OUTPUT", ha='center', va='top',
               fontsize=10, fontweight='bold', color='#2E7D32')
    ax_sm.text(x_out_s + io_w_s/2, box_h_s/2 - 0.30,
               f"Data Training\n{n_train_sm:,} × {n_feat_dim_sm}\n(seimbang, IR=1,0)",
               ha='center', va='center', fontsize=8.5, color='#333')

    ax_sm.set_xlim(-0.2, x_out_s + io_w_s + 0.2)
    ax_sm.set_ylim(-0.2, box_h_s + 0.2)
    plt.tight_layout()
    st.pyplot(fig_sm, use_container_width=True)
    plt.close()

    st.markdown("**📤 Hasil OUTPUT Aktual dari Diagram di Atas**")
    col_smo1, col_smo2, col_smo3 = st.columns(3)
    col_smo1.metric("Training Sebelum", f"{n_train:,}")
    col_smo2.metric("Training Sesudah", f"{n_train_sm:,}")
    col_smo3.metric("Data Sintetis", f"+{n_added:,}", f"+{inc_pct:.1f}%")
    st.caption(
        f"SMOTE menambahkan {n_added:,} data sintetis pada kelas minoritas "
        f"sehingga jumlah sampel tiap kelas pada data training menjadi setara "
        f"— lihat tabel distribusi lengkap di bawah."
    )

    st.divider()

    # ── Statistik utama (rasio aktif) ──────────────────────────────────────
    st.markdown(f"**📋 Statistik Ringkas SMOTE — Rasio Aktif: `{split_label}`**")
    k1, k2, k3 = st.columns(3)
    k1.metric("Training Sebelum SMOTE", f"{n_train:,}")
    k2.metric("Training Sesudah SMOTE", f"{n_train_sm:,}")
    k3.metric("Data Sintetis Ditambahkan", f"+{n_added:,}", f"+{inc_pct:.1f}%")

    st.divider()

    # ── Sub-tabs ────────────────────────────────────────────────────────────
    ss1, ss2, ss3, ss4 = st.tabs([
        "⚖️ Sebelum SMOTE",
        "✅ Setelah SMOTE",
        "📊 Perbandingan (Rasio Aktif)",
        "🔟 Perbandingan 90:10 / 80:20 / 70:30",
    ])

    counts_before = {cls: int((y_train == i).sum()) for i, cls in enumerate(classes)}
    total_before  = sum(counts_before.values())
    minor_b = min(counts_before.values())
    major_b = max(counts_before.values())
    imb_ratio_b = minor_b / major_b

    counts_after = {cls: int((y_train_sm == i).sum()) for i, cls in enumerate(classes)}
    total_after  = sum(counts_after.values())
    minor_a = min(counts_after.values())
    major_a = max(counts_after.values())
    imb_ratio_a = minor_a / major_a

    # ════════════════════════════════════════
    # Sub-tab 1: Sebelum SMOTE
    # ════════════════════════════════════════
    with ss1:
        st.markdown("**Distribusi kelas pada data training SEBELUM SMOTE**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Training", f"{total_before:,}")
        m2.metric("Kelas Mayor",    f"{major_b:,}")
        m3.metric("Kelas Minor",    f"{minor_b:,}")
        m4.metric("Imbalance Ratio", f"{imb_ratio_b:.4f}",
                  help="Rasio kelas minor / kelas mayor. Nilai 1.0 = sempurna seimbang")

        st.divider()

        st.markdown("**📄 Tabel Distribusi Kelas (Sebelum SMOTE)**")
        rows_b = []
        for i, cls in enumerate(classes):
            cnt = counts_before[cls]
            rows_b.append({
                'Kelas':      cls,
                'Jumlah':     f"{cnt:,}",
                'Persentase': f"{cnt/total_before*100:.2f}%",
                'Status':     '🟢 Mayor' if cnt == major_b else '🔴 Minor',
            })
        rows_b.append({'Kelas': 'TOTAL', 'Jumlah': f"{total_before:,}",
                       'Persentase': '100.00%', 'Status': '-'})
        st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)

        st.divider()

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown("**Bar Chart**")
            fig, ax = plt.subplots(figsize=(5, 4))
            cnt_list = [counts_before[c] for c in classes]
            bars = ax.bar(classes, cnt_list, color=['#28a745', '#dc3545'],
                          alpha=0.85, edgecolor='white', width=0.5)
            for bar, v in zip(bars, cnt_list):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(cnt_list)*0.02,
                        f'{v:,}', ha='center', fontweight='bold', fontsize=12)
            ax.set_title("Distribusi Kelas\nSebelum SMOTE", fontweight='bold')
            ax.set_ylabel("Jumlah Sampel")
            ax.set_ylim(0, max(cnt_list)*1.25)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b2:
            st.markdown("**Pie Chart**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(cnt_list, labels=classes, colors=['#28a745', '#dc3545'],
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 11},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            ax2.set_title("Proporsi Kelas\nSebelum SMOTE", fontweight='bold')
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        if imb_ratio_b < 0.9:
            st.warning(f"⚠️ Dataset tidak seimbang — Imbalance Ratio: **{imb_ratio_b:.4f}** "
                       f"(kelas minor {minor_b:,} vs mayor {major_b:,}). SMOTE diperlukan.")

    # ════════════════════════════════════════
    # Sub-tab 2: Setelah SMOTE
    # ════════════════════════════════════════
    with ss2:
        st.markdown("**Distribusi kelas pada data training SETELAH SMOTE**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Training", f"{total_after:,}", f"+{n_added:,} (sintetis)")
        m2.metric("Kelas Mayor",    f"{major_a:,}")
        m3.metric("Kelas Minor",    f"{minor_a:,}")
        m4.metric("Imbalance Ratio", f"{imb_ratio_a:.4f}",
                  help="Nilai mendekati 1.0 = sempurna seimbang")

        st.divider()

        st.markdown("**📄 Tabel Distribusi Kelas (Setelah SMOTE)**")
        rows_a = []
        for i, cls in enumerate(classes):
            cnt_b = counts_before[cls]
            cnt_a = counts_after[cls]
            added = cnt_a - cnt_b
            rows_a.append({
                'Kelas':         cls,
                'Sebelum SMOTE': f"{cnt_b:,}",
                'Setelah SMOTE': f"{cnt_a:,}",
                'Ditambahkan':   f"+{added:,}" if added > 0 else "—",
                'Persentase':    f"{cnt_a/total_after*100:.2f}%",
            })
        rows_a.append({
            'Kelas': 'TOTAL', 'Sebelum SMOTE': f"{total_before:,}",
            'Setelah SMOTE': f"{total_after:,}", 'Ditambahkan': f"+{n_added:,}",
            'Persentase': '100.00%',
        })
        st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)

        st.divider()

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("**Bar Chart**")
            fig, ax = plt.subplots(figsize=(5, 4))
            cnt_list = [counts_after[c] for c in classes]
            bars = ax.bar(classes, cnt_list, color=['#28a745', '#dc3545'],
                          alpha=0.85, edgecolor='white', width=0.5)
            for bar, v in zip(bars, cnt_list):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(cnt_list)*0.02,
                        f'{v:,}', ha='center', fontweight='bold', fontsize=12)
            ax.set_title("Distribusi Kelas\nSetelah SMOTE", fontweight='bold')
            ax.set_ylabel("Jumlah Sampel")
            ax.set_ylim(0, max(cnt_list)*1.25)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_a2:
            st.markdown("**Pie Chart**")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.pie(cnt_list, labels=classes, colors=['#28a745', '#dc3545'],
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 11},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
            ax2.set_title("Proporsi Kelas\nSetelah SMOTE", fontweight='bold')
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.success(f"✅ Setelah SMOTE — Imbalance Ratio: **{imb_ratio_a:.4f}** "
                   f"(mendekati 1.0, dataset seimbang)")

    # ════════════════════════════════════════
    # Sub-tab 3: Perbandingan (Rasio Aktif)
    # ════════════════════════════════════════
    with ss3:
        st.markdown(f"**Perbandingan Sebelum vs Sesudah SMOTE — Rasio Aktif `{split_label}`**")

        st.markdown("**📊 Perbandingan Jumlah per Kelas: Sebelum vs Sesudah SMOTE**")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(classes)); w = 0.35
        before_vals = [counts_before[c] for c in classes]
        after_vals  = [counts_after[c] for c in classes]
        b1 = ax.bar(x-w/2, before_vals, w, label='Sebelum SMOTE',
                    color='#E24B4A', alpha=0.85, edgecolor='white')
        b2 = ax.bar(x+w/2, after_vals, w, label='Setelah SMOTE',
                    color='#55A868', alpha=0.85, edgecolor='white')
        for bar, v in zip(b1, before_vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+max(after_vals)*0.01,
                    f'{v:,}', ha='center', fontsize=10, fontweight='bold')
        for bar, v in zip(b2, after_vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+max(after_vals)*0.01,
                    f'{v:,}', ha='center', fontsize=10, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
        ax.set_ylabel("Jumlah Sampel", fontsize=11)
        ax.set_title(f"Distribusi Kelas: Sebelum vs Setelah SMOTE\n(Rasio {split_label})",
                    fontweight='bold', fontsize=12)
        ax.legend(fontsize=11)
        ax.set_ylim(0, max(after_vals)*1.2)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

        st.markdown("**⚖️ Perbandingan Imbalance Ratio**")
        ib1, ib2, ib3 = st.columns(3)
        ib1.metric("Imbalance Ratio Sebelum SMOTE", f"{imb_ratio_b:.4f}")
        ib2.metric("Imbalance Ratio Setelah SMOTE",  f"{imb_ratio_a:.4f}")
        ib3.metric("Peningkatan Keseimbangan", f"{(imb_ratio_a - imb_ratio_b):.4f}")

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar(['Sebelum SMOTE', 'Setelah SMOTE'], [imb_ratio_b, imb_ratio_a],
                color=['#E24B4A', '#55A868'], alpha=0.85, edgecolor='white', width=0.4)
        ax3.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ratio Ideal (1.0)')
        for i, v in enumerate([imb_ratio_b, imb_ratio_a]):
            ax3.text(i, v+0.01, f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)
        ax3.set_ylabel("Imbalance Ratio (Minor/Mayor)")
        ax3.set_title("Perbandingan Imbalance Ratio\nSebelum vs Setelah SMOTE", fontweight='bold')
        ax3.set_ylim(0, 1.2)
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        col_ib, _ = st.columns([1, 1])
        with col_ib:
            st.pyplot(fig3, use_container_width=True)
        plt.close()

    # ════════════════════════════════════════
    # Sub-tab 4: Perbandingan 90:10 / 80:20 / 70:30 (DATA NYATA)
    # ════════════════════════════════════════
    with ss4:
        st.markdown("**🔟 Perbandingan SMOTE Nyata — Skenario 90:10, 80:20, 70:30**")
        st.caption(
            "Data pada sub-tab ini dihitung ULANG secara nyata (bukan estimasi/skala) "
            "untuk masing-masing skenario rasio split."
        )

        df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "def"
        with st.spinner("⏳ Menghitung distribusi SMOTE untuk 90:10, 80:20, 70:30..."):
            multi = smote_distribution_multi_ratio(df_hash, df)

        RATIOS  = multi['ratios']
        CLASSES = multi['classes']
        res     = multi['results']

        # ── Tabel ringkasan lengkap ────────────────────────────────────────
        st.markdown("**📄 Tabel Ringkasan Distribusi SMOTE — Semua Skenario**")
        tbl_rows = []
        for ratio in RATIOS:
            r = res[ratio]
            is_active = " ⭐ (aktif)" if ratio == split_label else ""
            row = {'Rasio Split': ratio + is_active}
            for cls in CLASSES:
                row[f'{cls} (Sebelum)'] = f"{r['counts_before'][cls]:,}"
            row['Total Sebelum'] = f"{r['total_before']:,}"
            row['IR Sebelum']    = f"{r['ir_before']:.4f}"
            for cls in CLASSES:
                row[f'{cls} (Sesudah)'] = f"{r['counts_after'][cls]:,}"
            row['Total Sesudah'] = f"{r['total_after']:,}"
            row['IR Sesudah']    = f"{r['ir_after']:.4f}"
            row['Data Sintetis'] = f"+{r['data_sintetis']:,}"
            tbl_rows.append(row)
        st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

        st.divider()

        # ── Bar chart grouped: sebelum vs sesudah, per rasio per kelas ──────
        st.markdown("**📊 Distribusi Kelas Sebelum vs Sesudah SMOTE — per Skenario**")
        fig4, axes4 = plt.subplots(1, 3, figsize=(15, 5))
        for ax, ratio in zip(axes4, RATIOS):
            r = res[ratio]
            x = np.arange(len(CLASSES)); w = 0.35
            before_v = [r['counts_before'][c] for c in CLASSES]
            after_v  = [r['counts_after'][c] for c in CLASSES]
            b1 = ax.bar(x-w/2, before_v, w, label='Sebelum SMOTE',
                        color='#E24B4A', alpha=0.85, edgecolor='white')
            b2 = ax.bar(x+w/2, after_v, w, label='Setelah SMOTE',
                        color='#55A868', alpha=0.85, edgecolor='white')
            for bar, v in zip(b1, before_v):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(after_v)*0.02,
                        f'{v:,}', ha='center', fontsize=9, fontweight='bold')
            for bar, v in zip(b2, after_v):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(after_v)*0.02,
                        f'{v:,}', ha='center', fontsize=9, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(CLASSES, fontsize=10)
            ax.set_title(f'Rasio {ratio}' + (' ⭐' if ratio == split_label else ''),
                        fontweight='bold', fontsize=12)
            ax.set_ylabel("Jumlah Sampel", fontsize=10)
            ax.set_ylim(0, max(after_v)*1.25)
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.divider()

        # ── Bar chart Imbalance Ratio per skenario ──────────────────────────
        st.markdown("**⚖️ Perbandingan Imbalance Ratio — Semua Skenario**")
        fig5, ax5 = plt.subplots(figsize=(9, 5))
        x5 = np.arange(len(RATIOS)); w5 = 0.35
        ir_before_vals = [res[r]['ir_before'] for r in RATIOS]
        ir_after_vals  = [res[r]['ir_after'] for r in RATIOS]
        b1 = ax5.bar(x5-w5/2, ir_before_vals, w5, label='Sebelum SMOTE',
                    color='#E24B4A', alpha=0.85, edgecolor='white')
        b2 = ax5.bar(x5+w5/2, ir_after_vals, w5, label='Setelah SMOTE',
                    color='#55A868', alpha=0.85, edgecolor='white')
        for bar, v in zip(b1, ir_before_vals):
            ax5.text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
        for bar, v in zip(b2, ir_after_vals):
            ax5.text(bar.get_x()+bar.get_width()/2, v+0.01,
                     f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')
        ax5.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='Ideal (1.0)')
        ax5.set_xticks(x5); ax5.set_xticklabels(RATIOS, fontsize=11)
        ax5.set_ylabel("Imbalance Ratio", fontsize=11)
        ax5.set_title("Imbalance Ratio Sebelum vs Sesudah SMOTE\nuntuk Skenario 90:10, 80:20, 70:30",
                     fontweight='bold', fontsize=12)
        ax5.set_ylim(0, 1.2)
        ax5.legend(fontsize=10)
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        st.divider()

        # ── Bar chart data sintetis ditambahkan ─────────────────────────────
        st.markdown("**📊 Jumlah Data Sintetis yang Ditambahkan SMOTE per Skenario**")
        fig6, ax6 = plt.subplots(figsize=(8, 4.5))
        sintetis_vals = [res[r]['data_sintetis'] for r in RATIOS]
        bars6 = ax6.bar(RATIOS, sintetis_vals, color='#F5A623', alpha=0.85,
                        edgecolor='white', width=0.5)
        for bar, v in zip(bars6, sintetis_vals):
            ax6.text(bar.get_x()+bar.get_width()/2, v+1,
                     f'+{v:,}', ha='center', fontweight='bold', fontsize=11)
        ax6.set_ylabel("Jumlah Data Sintetis", fontsize=11)
        ax6.set_title("Data Sintetis SMOTE per Skenario Rasio Split", fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        col_s, _ = st.columns([1, 1])
        with col_s:
            st.pyplot(fig6, use_container_width=True)
        plt.close()

        st.divider()
        st.info(
            "**Kesimpulan:**\n\n"
            + "\n".join([
                f"- Rasio **{ratio}**: {res[ratio]['total_before']:,} → "
                f"{res[ratio]['total_after']:,} sampel training "
                f"(+{res[ratio]['data_sintetis']:,} sintetis), "
                f"IR {res[ratio]['ir_before']:.4f} → {res[ratio]['ir_after']:.4f}"
                for ratio in RATIOS
            ])
        )
