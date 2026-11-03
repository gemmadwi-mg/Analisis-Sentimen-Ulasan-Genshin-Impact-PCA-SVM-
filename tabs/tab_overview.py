"""
tabs/tab_overview.py
Tab: Overview — Data Description (Revisi Dosen Penguji Poin 1)
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def render(df=None, **kwargs):
    """Render tab Overview dengan Data Description lengkap."""
    st.subheader("📊 Data Description")
    st.caption(
        "Deskripsi lengkap dataset sebelum proses seleksi data dan pemodelan. "
        "Sesuai masukan dosen penguji: jelaskan data keseluruhan terlebih dahulu."
    )

    val_counts = df['sentiment'].value_counts()
    n_pos = int(val_counts.get('POSITIVE', 0))
    n_neg = int(val_counts.get('NEGATIVE', 0))
    n_tot = len(df)

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.tabs([
        "📋 Deskripsi Umum",
        "📊 Distribusi Data",
        "🔍 Statistik Teks",
        "👁️ Preview Data",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Deskripsi Umum
    # ══════════════════════════════════════════════════════════════════════════
    with s1:
        st.markdown("**📋 Informasi Umum Dataset**")
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Ulasan",    f"{n_tot:,}")
        k2.metric("Positif 😊",      f"{n_pos:,}", f"{n_pos/n_tot*100:.1f}%")
        k3.metric("Negatif 😞",      f"{n_neg:,}", f"{n_neg/n_tot*100:.1f}%")
        k4.metric("Total Kolom",     f"{len(df.columns)}")

        st.divider()
        st.markdown("**📄 Deskripsi Kolom Dataset**")

        col_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_null = int(df[col].isnull().sum())
            n_unique = int(df[col].nunique())
            sample = str(df[col].dropna().iloc[0])[:60] + ('...' if len(str(df[col].dropna().iloc[0])) > 60 else '')
            col_info.append({
                'Nama Kolom':      col,
                'Tipe Data':       dtype,
                'Nilai Null':      f"{n_null:,}",
                'Nilai Unik':      f"{n_unique:,}",
                'Contoh Nilai':    sample,
            })
        st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**📌 Sumber dan Konteks Dataset**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(
                "**Nama Dataset:** Ulasan Genshin Impact (Google Play Store)\n\n"
                "**Jumlah Data:** 944 ulasan\n\n"
                "**Label Sentimen:** POSITIVE, NEGATIVE\n\n"
                "**Kolom Utama:** content (teks ulasan), sentiment (label)\n\n"
                "**Kolom Pendukung:** score, at, userName, appVersion, dll."
            )
        with col_b:
            st.info(
                "**Metode Labeling:** Pre-labeled menggunakan model transformer\n\n"
                "**Rentang Waktu:** Maret 2024 (berdasarkan kolom 'at')\n\n"
                "**Bahasa:** Inggris\n\n"
                "**Versi Aplikasi:** 4.5.0_21307647_21321795\n\n"
                "**Kolom Score:** Confidence score hasil labeling (0.54–1.00)"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Distribusi Data
    # ══════════════════════════════════════════════════════════════════════════
    with s2:
        st.markdown("**📊 Distribusi Keseluruhan Dataset**")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Distribusi Sentimen**")
            fig, ax = plt.subplots(figsize=(5, 5))
            wedges, texts, autotexts = ax.pie(
                val_counts.values, labels=val_counts.index,
                colors=['#28a745', '#dc3545'],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 12},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            for at in autotexts:
                at.set_fontweight('bold')
            ax.set_title("Distribusi Label Sentimen\n(Keseluruhan Dataset)", fontsize=13, fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_r:
            st.markdown("**Jumlah Data per Kelas**")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            bars = ax2.bar(val_counts.index, val_counts.values,
                           color=['#28a745', '#dc3545'], alpha=0.85,
                           edgecolor='white', width=0.5)
            for bar, v in zip(bars, val_counts.values):
                ax2.text(bar.get_x()+bar.get_width()/2, v+5,
                         f'{v:,}\n({v/n_tot*100:.1f}%)',
                         ha='center', fontweight='bold', fontsize=12)
            ax2.set_ylabel("Jumlah Ulasan")
            ax2.set_title("Jumlah Data per Kelas Sentimen", fontsize=13, fontweight='bold')
            ax2.set_ylim(0, max(val_counts.values)*1.25)
            ax2.grid(axis='y', alpha=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.divider()
        st.markdown("**⚖️ Analisis Keseimbangan Kelas (Class Imbalance)**")
        minor = min(val_counts.values)
        major = max(val_counts.values)
        ir = minor / major
        c1, c2, c3 = st.columns(3)
        c1.metric("Kelas Mayor",      f"{val_counts.idxmax()} ({major:,})")
        c2.metric("Kelas Minor",      f"{val_counts.idxmin()} ({minor:,})")
        c3.metric("Imbalance Ratio",  f"{ir:.4f}",
                  help="Rasio kelas minor/mayor. Nilai < 1.0 menunjukkan ketidakseimbangan")
        if ir < 0.9:
            st.warning(
                f"⚠️ Dataset tidak seimbang (IR = {ir:.4f}). "
                f"SMOTE akan diterapkan pada data training untuk menyeimbangkan distribusi kelas."
            )
        else:
            st.success(f"✅ Dataset relatif seimbang (IR = {ir:.4f}). SMOTE tetap diterapkan untuk memastikan keseimbangan optimal.")

        st.divider()
        if 'score' in df.columns:
            st.markdown("**📈 Distribusi Confidence Score Labeling**")
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            for sent, color in [('POSITIVE','#28a745'),('NEGATIVE','#dc3545')]:
                mask = df['sentiment'] == sent
                ax3.hist(df.loc[mask,'score'], bins=30, alpha=0.6,
                         label=sent, color=color, edgecolor='white')
            ax3.set_xlabel("Confidence Score")
            ax3.set_ylabel("Frekuensi")
            ax3.set_title("Distribusi Confidence Score Labeling per Kelas Sentimen",
                          fontweight='bold')
            ax3.legend()
            ax3.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)
            plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Statistik Teks
    # ══════════════════════════════════════════════════════════════════════════
    with s3:
        df['text_len_words'] = df['content'].str.split().str.len()
        df['text_len_chars'] = df['content'].str.len()

        st.markdown("**📝 Statistik Panjang Teks Ulasan**")
        stat_rows = []
        for sent in ['POSITIVE','NEGATIVE','ALL']:
            mask = df['sentiment'] == sent if sent != 'ALL' else [True]*len(df)
            sub = df.loc[mask, 'text_len_words']
            stat_rows.append({
                'Kelas':        sent,
                'Min (kata)':   f"{sub.min():.0f}",
                'Max (kata)':   f"{sub.max():.0f}",
                'Mean (kata)':  f"{sub.mean():.1f}",
                'Median (kata)':f"{sub.median():.1f}",
                'Std (kata)':   f"{sub.std():.1f}",
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**📊 Distribusi Panjang Teks per Kelas**")
        fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
        for sent, color in [('POSITIVE','#28a745'),('NEGATIVE','#dc3545')]:
            mask = df['sentiment'] == sent
            axes4[0].hist(df.loc[mask,'text_len_words'], bins=30, alpha=0.6,
                          label=sent, color=color, edgecolor='white')
            axes4[1].hist(df.loc[mask,'text_len_chars'], bins=30, alpha=0.6,
                          label=sent, color=color, edgecolor='white')
        for ax, xlabel, title in zip(axes4,
            ['Jumlah Kata','Jumlah Karakter'],
            ['Distribusi Panjang Ulasan (Kata)','Distribusi Panjang Ulasan (Karakter)']):
            ax.set_xlabel(xlabel); ax.set_ylabel("Frekuensi")
            ax.set_title(title, fontweight='bold')
            ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

        st.divider()
        st.markdown("**📦 Boxplot Panjang Teks per Kelas**")
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        data_box = [df.loc[df['sentiment']==s,'text_len_words'].values
                    for s in ['POSITIVE','NEGATIVE']]
        bp = ax5.boxplot(data_box, labels=['POSITIVE','NEGATIVE'],
                         patch_artist=True,
                         medianprops={'color':'white','linewidth':2})
        for patch, color in zip(bp['boxes'],['#28a745','#dc3545']):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax5.set_ylabel("Jumlah Kata per Ulasan")
        ax5.set_title("Boxplot Panjang Ulasan per Kelas", fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        col_box, _ = st.columns([1,1])
        with col_box:
            st.pyplot(fig5, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — Preview Data
    # ══════════════════════════════════════════════════════════════════════════
    with s4:
        st.markdown("**👁️ Preview Dataset**")
        n_rows = st.slider("Tampilkan berapa baris?", 5, 50, 10)
        show_all_cols = st.checkbox("Tampilkan semua kolom", value=False)

        cols_show = ['content','sentiment','score'] if not show_all_cols else df.columns.tolist()
        cols_show = [c for c in cols_show if c in df.columns]
        st.dataframe(df[cols_show].head(n_rows), use_container_width=True)

        st.divider()
        st.markdown("**📊 Contoh Ulasan per Kelas**")
        for sent, color_box in [('POSITIVE','success'),('NEGATIVE','error')]:
            mask = df['sentiment'] == sent
            sample = df.loc[mask,'content'].sample(3, random_state=42).tolist()
            st.markdown(f"**{sent}:**")
            for s in sample:
                if color_box == 'success':
                    st.success(f"😊 {s[:200]}{'...' if len(s)>200 else ''}")
                else:
                    st.error(f"😞 {s[:200]}{'...' if len(s)>200 else ''}")
