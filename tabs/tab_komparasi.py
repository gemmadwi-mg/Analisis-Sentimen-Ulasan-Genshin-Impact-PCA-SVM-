"""
tabs/tab_komparasi.py
Tab: Komparasi Metodologi (Revisi Bu Yuni Poin 2, 4, 6b)
Menjalankan SEMUA kondisi sekaligus:
  - Dengan PCA dan Tanpa PCA
  - Semua kernel × semua rasio split
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.pipeline import (smote_vs_nosmote_full,
                              chisq_vs_full_features,
                              pca_threshold_compare)

KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
RATIOS  = ['70:30', '80:20', '90:10']
CLR_K   = {'linear':'#4C72B0','rbf':'#55A868','poly':'#E24B4A','sigmoid':'#F5A623'}
CLR_SM  = '#55A868';  CLR_NS  = '#E24B4A'
CLR_CHI = '#4C72B0';  CLR_FL  = '#F5A623'
CLR_PCA = '#4C72B0';  CLR_NP  = '#E24B4A'
THRESH_C = {0.70:'#1F3864',0.75:'#2E75B6',0.80:'#4C72B0',
            0.85:'#5BA4CF',0.90:'#85C1E9',0.95:'#E24B4A',0.99:'#1a5c30'}


def _ann(ax, bars, fmt='.3f', color='black', off=0.003):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, v+off,
                f'{v:{fmt}}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=color)


def _mk_key(mk): 
    return {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[mk]


# ── Helper: tabel delta PCA vs NoPCA ──────────────────────────────────────────
def _delta_table(res_pca, res_np, key_fn, ratios, kernels, metric_name):
    rows = []
    for ratio in ratios:
        for kernel in kernels:
            vp = key_fn(res_pca,  ratio, kernel)
            vn = key_fn(res_np,   ratio, kernel)
            d  = vp - vn
            rows.append({
                'Rasio':       ratio,
                'Kernel':      kernel.upper(),
                f'{metric_name} (PCA)':    f"{vp:.4f}",
                f'{metric_name} (NoPCA)':  f"{vn:.4f}",
                'Δ (PCA−NoPCA)':           f"{d:+.4f}",
                'PCA Lebih Baik?': '✅' if d>0 else ('➖' if abs(d)<0.001 else '❌'),
            })
    return pd.DataFrame(rows)


def render(df=None, **kwargs):
    st.subheader("🔎 Komparasi Metodologi")
    st.caption(
        "Menjawab revisi Bu Yuni: **Poin 2** SMOTE vs Tanpa SMOTE · "
        "**Poin 4** Chi-Square vs Semua Fitur TF-IDF · "
        "**Poin 6b** Threshold PCA · Semua dijalankan **dengan PCA dan tanpa PCA**."
    )

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
        return

    s_used  = st.session_state.get('split_label', '80:20')
    df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "def"

    st.info(
        f"⏳ **Catatan:** Tab ini melatih banyak model sekaligus (kedua kondisi PCA). "
        f"Hasil di-cache otomatis setelah pertama kali dijalankan.",
        icon="ℹ️"
    )

    poin2, poin4, poin6b = st.tabs([
        "📌 Poin 2: SMOTE vs Tanpa SMOTE",
        "📌 Poin 4: Chi-Square vs Semua Fitur",
        "📌 Poin 6b: Threshold PCA",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # POIN 2 — SMOTE vs Tanpa SMOTE (dengan PCA & tanpa PCA)
    # ══════════════════════════════════════════════════════════════════════════
    with poin2:
        st.markdown("### Poin 2: Perbandingan SMOTE vs Tanpa SMOTE")
        st.caption("Semua kernel × semua rasio × dengan PCA & tanpa PCA")

        with st.spinner("⏳ Melatih: SMOTE vs NoSMOTE (dengan PCA)..."):
            r2_pca = smote_vs_nosmote_full(f"{df_hash}_pca",   True,  df)
        with st.spinner("⏳ Melatih: SMOTE vs NoSMOTE (tanpa PCA)..."):
            r2_np  = smote_vs_nosmote_full(f"{df_hash}_nopca", False, df)

        classes = r2_pca['classes']

        metric2 = st.selectbox(
            "Metrik:", ["Accuracy","F1-Score","Precision","Recall"], key="km2_m"
        )
        mk2 = _mk_key(metric2)

        s2a, s2b, s2c, s2d = st.tabs([
            "📄 Tabel Lengkap",
            "📊 Visualisasi per Kernel",
            "🔥 Heatmap",
            "🔄 PCA vs Tanpa PCA (Pengaruh)",
        ])

        def _p2_val(res, ratio, kernel, cond):
            return res['results'][ratio][kernel][cond][mk2]

        # ── Tabel lengkap ─────────────────────────────────────────────────────
        with s2a:
            tbl2 = []
            for ratio in RATIOS:
                for kernel in KERNELS:
                    for pca_lbl, res in [("Dengan PCA", r2_pca), ("Tanpa PCA", r2_np)]:
                        r   = res['results'][ratio][kernel]
                        v_s = r['smote'][mk2]; v_n = r['nosmote'][mk2]
                        d   = v_s - v_n
                        tbl2.append({
                            'Rasio':    ratio+(' ⭐' if ratio==s_used else ''),
                            'Kernel':   kernel.upper(),
                            'Mode PCA': pca_lbl,
                            'n Train (SMOTE)':    f"{r['n_train_smote']:,}",
                            'n Train (NoSMOTE)':  f"{r['n_train_nosmote']:,}",
                            'IR (SMOTE)':         f"{r['ir_smote']:.4f}",
                            'IR (NoSMOTE)':       f"{r['ir_nosmote']:.4f}",
                            f'{metric2} (SMOTE)':   f"{v_s:.4f}",
                            f'{metric2} (NoSMOTE)': f"{v_n:.4f}",
                            'Δ (S−NS)':  f"{d:+.4f}",
                            'SMOTE ✅?': '✅' if d>0 else ('➖' if abs(d)<0.001 else '❌'),
                        })
            st.dataframe(pd.DataFrame(tbl2), use_container_width=True,
                         hide_index=True, height=420)

            # Ringkasan win count
            for pca_lbl, tbl_sub in [
                ("Dengan PCA",  [r for r in tbl2 if r['Mode PCA']=="Dengan PCA"]),
                ("Tanpa PCA",   [r for r in tbl2 if r['Mode PCA']=="Tanpa PCA"]),
            ]:
                wins = sum(1 for r in tbl_sub if r['SMOTE ✅?']=='✅')
                total = len(tbl_sub)
                st.markdown(
                    f"**{pca_lbl}:** SMOTE unggul pada "
                    f"**{wins}/{total}** kombinasi ({wins/total*100:.0f}%)"
                )

        # ── Visualisasi per kernel ─────────────────────────────────────────────
        with s2b:
            st.markdown(f"**{metric2}: SMOTE vs Tanpa SMOTE — Per Kernel**")
            for pca_lbl, res in [("Dengan PCA", r2_pca), ("Tanpa PCA", r2_np)]:
                st.markdown(f"**{pca_lbl}**")
                fig, axes = plt.subplots(2, 2, figsize=(14, 9))
                fig.suptitle(
                    f'{metric2}: SMOTE vs Tanpa SMOTE — {pca_lbl}',
                    fontsize=13, fontweight='bold'
                )
                for ax, kernel in zip(axes.flatten(), KERNELS):
                    sm_v  = [res['results'][r][kernel]['smote'][mk2]   for r in RATIOS]
                    nos_v = [res['results'][r][kernel]['nosmote'][mk2] for r in RATIOS]
                    x = np.arange(len(RATIOS)); w = 0.35
                    b1 = ax.bar(x-w/2, sm_v,  w, label='SMOTE',
                                color=CLR_SM, alpha=0.87, edgecolor='white')
                    b2 = ax.bar(x+w/2, nos_v, w, label='NoSMOTE',
                                color=CLR_NS, alpha=0.87, edgecolor='white')
                    _ann(ax, b1, color='#1a5c30'); _ann(ax, b2, color='#8B0000')
                    ax.set_xticks(x)
                    ax.set_xticklabels(RATIOS, fontsize=8, rotation=15)
                    lo = min(min(sm_v),min(nos_v))-0.05
                    ax.set_ylim(max(0,lo), 1.08)
                    ax.set_title(kernel.upper(), fontweight='bold')
                    ax.set_ylabel(metric2, fontsize=9)
                    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Heatmap ───────────────────────────────────────────────────────────
        with s2c:
            st.markdown(f"**🔥 Heatmap Δ {metric2} (SMOTE − NoSMOTE)**")
            for pca_lbl, res in [("Dengan PCA", r2_pca), ("Tanpa PCA", r2_np)]:
                st.markdown(f"**{pca_lbl}**")
                dm = pd.DataFrame(
                    [[res['results'][r][k]['smote'][mk2] -
                      res['results'][r][k]['nosmote'][mk2]
                      for r in RATIOS] for k in KERNELS],
                    index=[k.upper() for k in KERNELS], columns=RATIOS
                )
                fig, ax = plt.subplots(figsize=(9, 3))
                sns.heatmap(dm.astype(float), annot=True, fmt='+.4f',
                            cmap='RdYlGn', center=0, ax=ax,
                            linewidths=0.5, linecolor='white',
                            annot_kws={'size':10,'weight':'bold'},
                            cbar_kws={'shrink':0.8})
                ax.set_title(f'Δ {metric2} SMOTE−NoSMOTE | {pca_lbl}',
                             fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Pengaruh PCA terhadap SMOTE ────────────────────────────────────────
        with s2d:
            st.markdown(f"**🔄 Pengaruh PCA: {metric2} SMOTE (PCA vs Tanpa PCA)**")
            for cond, cond_lbl in [('smote','SMOTE'),('nosmote','NoSMOTE')]:
                st.markdown(f"**Kondisi: {cond_lbl}**")
                dm = pd.DataFrame(
                    [[r2_pca['results'][r][k][cond][mk2] -
                      r2_np['results'][r][k][cond][mk2]
                      for r in RATIOS] for k in KERNELS],
                    index=[k.upper() for k in KERNELS], columns=RATIOS
                )
                fig, ax = plt.subplots(figsize=(9, 3))
                sns.heatmap(dm.astype(float), annot=True, fmt='+.4f',
                            cmap='RdYlGn', center=0, ax=ax,
                            linewidths=0.5, linecolor='white',
                            annot_kws={'size':10,'weight':'bold'},
                            cbar_kws={'shrink':0.8})
                ax.set_title(f'Δ {metric2} (PCA − Tanpa PCA) | Kondisi: {cond_lbl}',
                             fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

            # Kesimpulan
            wins_pca = sum(
                1 for r in RATIOS for k in KERNELS
                if r2_pca['results'][r][k]['smote'][mk2] >
                   r2_np['results'][r][k]['smote'][mk2]
            )
            total = len(RATIOS)*len(KERNELS)
            smote_wins_pca  = sum(1 for r in RATIOS for k in KERNELS
                                  if r2_pca['results'][r][k]['smote'][mk2] >
                                     r2_pca['results'][r][k]['nosmote'][mk2])
            smote_wins_np   = sum(1 for r in RATIOS for k in KERNELS
                                  if r2_np['results'][r][k]['smote'][mk2] >
                                     r2_np['results'][r][k]['nosmote'][mk2])
            st.info(
                f"**Kesimpulan Poin 2:**\n\n"
                f"- SMOTE unggul **dengan PCA**: {smote_wins_pca}/{total} kombinasi\n\n"
                f"- SMOTE unggul **tanpa PCA**: {smote_wins_np}/{total} kombinasi\n\n"
                f"- PCA meningkatkan SMOTE pada: {wins_pca}/{total} kombinasi"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # POIN 4 — Chi-Square vs Semua Fitur TF-IDF
    # ══════════════════════════════════════════════════════════════════════════
    with poin4:
        with st.spinner("⏳ Melatih: Chi-Square vs Full TF-IDF (dengan PCA)..."):
            r4_pca = chisq_vs_full_features(f"{df_hash}_pca",   True,  df)
        with st.spinner("⏳ Melatih: Chi-Square vs Full TF-IDF (tanpa PCA)..."):
            r4_np  = chisq_vs_full_features(f"{df_hash}_nopca", False, df)

        n_chi  = r4_pca['n_chi_feat']
        n_full = r4_pca['n_full_feat']

        st.markdown(
            f"### Perbandingan Performa SVM: {n_chi:,} Fitur Chi-Square "
            f"vs {n_full:,} Fitur TF-IDF Keseluruhan"
        )
        st.caption(
            f"Diuji pada empat kernel (Linear, RBF, Polynomial, Sigmoid) dan "
            f"tiga rasio split data yang sama (90:10, 80:20, 70:30), dengan "
            f"parameter tetap C = 1 dan gamma = 0,001. Data latih dan data uji "
            f"identik di kedua skenario (hasil split yang sama), sehingga "
            f"perbandingan bersifat adil."
        )

        st.markdown(
            f"**Chi-Square:** `{n_chi:,}` fitur | "
            f"**TF-IDF Keseluruhan:** `{n_full:,}` fitur | "
            f"**Reduksi:** `{(1-n_chi/n_full)*100:.0f}%`"
        )

        metric4 = st.selectbox(
            "Metrik:", ["Accuracy","F1-Score","Precision","Recall"], key="km4_m"
        )
        mk4 = _mk_key(metric4)

        s4a, s4b, s4c, s4d = st.tabs([
            "📄 Tabel Lengkap",
            "📊 Visualisasi per Kernel",
            "🔥 Heatmap",
            "⏱️ Perbandingan Waktu Training",
        ])

        # ── Tabel lengkap ─────────────────────────────────────────────────────
        with s4a:
            tbl4 = []
            for ratio in RATIOS:
                for kernel in KERNELS:
                    for pca_lbl, res in [("Dengan PCA", r4_pca), ("Tanpa PCA", r4_np)]:
                        r    = res['results'][ratio][kernel]
                        v_c  = r['chi2'][mk4]; v_f = r['full'][mk4]
                        d    = v_c - v_f
                        tbl4.append({
                            'Rasio':    ratio+(' ⭐' if ratio==s_used else ''),
                            'Kernel':   kernel.upper(),
                            'Mode PCA': pca_lbl,
                            'Dim Chi':  f"{r['n_chi']:,}",
                            'Dim Full': f"{r['n_full']:,}",
                            f'{metric4} (Chi 1K)':  f"{v_c:.4f}",
                            f'{metric4} (Full 5K)': f"{v_f:.4f}",
                            'Δ (Chi−Full)': f"{d:+.4f}",
                            'Chi ✅?':  '✅' if d>0 else ('➖' if abs(d)<0.001 else '❌'),
                        })
            st.dataframe(pd.DataFrame(tbl4), use_container_width=True,
                         hide_index=True, height=420)

            for pca_lbl, cond in [("Dengan PCA","pca"),("Tanpa PCA","nopca")]:
                sub = [r for r in tbl4 if r['Mode PCA']==pca_lbl]
                wins = sum(1 for r in sub if r['Chi ✅?']=='✅')
                st.markdown(
                    f"**{pca_lbl}:** Chi-Square unggul pada "
                    f"**{wins}/{len(sub)}** kombinasi ({wins/len(sub)*100:.0f}%)"
                )

        # ── Visualisasi per kernel ─────────────────────────────────────────────
        with s4b:
            for pca_lbl, res in [("Dengan PCA", r4_pca), ("Tanpa PCA", r4_np)]:
                st.markdown(f"**{metric4}: Chi-Square vs Full TF-IDF — {pca_lbl}**")
                fig, axes = plt.subplots(2, 2, figsize=(14, 9))
                fig.suptitle(
                    f'{metric4}: {n_chi:,} Fitur Chi-Sq vs {n_full:,} Fitur TF-IDF — {pca_lbl}',
                    fontsize=13, fontweight='bold'
                )
                for ax, kernel in zip(axes.flatten(), KERNELS):
                    c_v = [res['results'][r][kernel]['chi2'][mk4] for r in RATIOS]
                    f_v = [res['results'][r][kernel]['full'][mk4] for r in RATIOS]
                    x = np.arange(len(RATIOS)); w = 0.35
                    b1 = ax.bar(x-w/2, c_v, w, label=f'Chi-Sq {n_chi:,}',
                                color=CLR_CHI, alpha=0.87, edgecolor='white')
                    b2 = ax.bar(x+w/2, f_v, w, label=f'TF-IDF {n_full:,}',
                                color=CLR_FL,  alpha=0.87, edgecolor='white')
                    _ann(ax, b1, color='#1a3a6c'); _ann(ax, b2, color='#7d4a00')
                    ax.set_xticks(x)
                    ax.set_xticklabels(RATIOS, fontsize=8, rotation=15)
                    lo = min(min(c_v),min(f_v))-0.05
                    ax.set_ylim(max(0,lo), 1.08)
                    ax.set_title(kernel.upper(), fontweight='bold')
                    ax.set_ylabel(metric4, fontsize=9)
                    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Heatmap ───────────────────────────────────────────────────────────
        with s4c:
            st.markdown(f"**🔥 Heatmap Δ {metric4} (Chi-Square − Full TF-IDF)**")
            col_h1, col_h2 = st.columns(2)
            for col, pca_lbl, res in [
                (col_h1, "Dengan PCA", r4_pca),
                (col_h2, "Tanpa PCA",  r4_np),
            ]:
                dm = pd.DataFrame(
                    [[res['results'][r][k]['chi2'][mk4] -
                      res['results'][r][k]['full'][mk4]
                      for r in RATIOS] for k in KERNELS],
                    index=[k.upper() for k in KERNELS], columns=RATIOS
                )
                with col:
                    st.markdown(f"**{pca_lbl}**")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.heatmap(dm.astype(float), annot=True, fmt='+.4f',
                                cmap='RdYlGn', center=0, ax=ax,
                                linewidths=0.5, linecolor='white',
                                annot_kws={'size':9,'weight':'bold'},
                                cbar_kws={'shrink':0.7})
                    ax.set_title(f'Δ {metric4} Chi−Full\n{pca_lbl}',
                                 fontweight='bold', fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True); plt.close()

            # Heatmap PCA effect on Chi-Square
            st.markdown(f"**🔥 Pengaruh PCA pada Chi-Square: Δ {metric4} (PCA − Tanpa PCA)**")
            for feat_type, feat_lbl in [
                ('chi2', 'Chi-Square 1K'), ('full', 'TF-IDF Full 5K')
            ]:
                dm_pca = pd.DataFrame(
                    [[r4_pca['results'][r][k][feat_type][mk4] -
                      r4_np['results'][r][k][feat_type][mk4]
                      for r in RATIOS] for k in KERNELS],
                    index=[k.upper() for k in KERNELS], columns=RATIOS
                )
                st.markdown(f"**{feat_lbl}**")
                fig, ax = plt.subplots(figsize=(9, 3))
                sns.heatmap(dm_pca.astype(float), annot=True, fmt='+.4f',
                            cmap='RdYlGn', center=0, ax=ax,
                            linewidths=0.5, linecolor='white',
                            annot_kws={'size':10,'weight':'bold'},
                            cbar_kws={'shrink':0.8})
                ax.set_title(f'Δ {metric4} PCA−NoPCA | {feat_lbl}',
                             fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Waktu training ────────────────────────────────────────────────────
        with s4d:
            st.markdown("**⏱️ Perbandingan Waktu Training: Chi-Square vs Full TF-IDF**")
            for pca_lbl, res in [("Dengan PCA", r4_pca), ("Tanpa PCA", r4_np)]:
                st.markdown(f"**{pca_lbl}**")
                time_rows = []
                for ratio in RATIOS:
                    row = {'Rasio': ratio+(' ⭐' if ratio==s_used else '')}
                    t_chi_total = 0; t_ful_total = 0
                    for kernel in KERNELS:
                        r = res['results'][ratio][kernel]
                        tc = r['chi2']['train_time']
                        tf = r['full']['train_time']
                        row[f'{kernel.upper()} Chi'] = f"{tc:.3f}s"
                        row[f'{kernel.upper()} Full']= f"{tf:.3f}s"
                        t_chi_total += tc; t_ful_total += tf
                    row['Total Chi']  = f"{t_chi_total:.3f}s"
                    row['Total Full'] = f"{t_ful_total:.3f}s"
                    row['Hemat (%)']  = f"{(1-t_chi_total/t_ful_total)*100:+.1f}%"
                    time_rows.append(row)
                st.dataframe(pd.DataFrame(time_rows),
                             use_container_width=True, hide_index=True)

            # Kesimpulan
            chi_wins_pca = sum(1 for r in RATIOS for k in KERNELS
                               if r4_pca['results'][r][k]['chi2'][mk4] >
                                  r4_pca['results'][r][k]['full'][mk4])
            chi_wins_np  = sum(1 for r in RATIOS for k in KERNELS
                               if r4_np['results'][r][k]['chi2'][mk4] >
                                  r4_np['results'][r][k]['full'][mk4])
            total = len(RATIOS)*len(KERNELS)
            st.info(
                f"**Kesimpulan Poin 4:**\n\n"
                f"- Chi-Square unggul **dengan PCA**: {chi_wins_pca}/{total} kombinasi\n\n"
                f"- Chi-Square unggul **tanpa PCA**: {chi_wins_np}/{total} kombinasi\n\n"
                f"- Seleksi Chi-Square mereduksi dimensi "
                f"**{(1-n_chi/n_full)*100:.0f}%** ({n_full:,}→{n_chi:,} fitur) "
                f"dengan waktu training lebih cepat."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # POIN 6b — Threshold PCA semua kernel × semua rasio
    # ══════════════════════════════════════════════════════════════════════════
    with poin6b:
        st.markdown("### Poin 6b: Perbandingan Threshold PCA")
        st.caption(
            "Threshold [0.70–0.99] × semua kernel × semua rasio split. "
            "Untuk Poin 6b, PCA selalu aktif (perbandingan antar threshold)."
        )

        with st.spinner("⏳ Menguji semua threshold × kernel × rasio..."):
            res6 = pca_threshold_compare(df_hash, df)

        THRESHOLDS = res6['thresholds']
        orig_dim   = res6['orig_dim']

        metric6 = st.selectbox(
            "Metrik:", ["Accuracy","F1-Score"], key="km6_m"
        )
        mk6 = _mk_key(metric6)

        s6a, s6b, s6c, s6d = st.tabs([
            "📄 Jumlah Komponen",
            "📊 Metrik per Threshold × Kernel",
            "🔥 Heatmap Threshold × Rasio",
            "💡 Kesimpulan",
        ])

        # ── Tabel komponen ────────────────────────────────────────────────────
        with s6a:
            st.markdown("**📄 Jumlah Komponen & Variance per Threshold × Rasio**")
            comp_rows = []
            for thresh in THRESHOLDS:
                row = {'Threshold': str(thresh)+(' ⭐' if thresh==0.95 else '')}
                for ratio in RATIOS:
                    r = res6['results'][thresh][ratio]['rbf']
                    row[ratio] = f"{r['n_comp']} ({r['var_exp']*100:.1f}%)"
                comp_rows.append(row)
            st.dataframe(pd.DataFrame(comp_rows),
                         use_container_width=True, hide_index=True)
            st.caption("Format: komponen (variance%). Representatif: kernel RBF.")

            # Bar chart komponen per threshold
            fig6c, ax6c = plt.subplots(figsize=(11, 4.5))
            x6  = np.arange(len(RATIOS)); w6 = 0.1
            for ti, thresh in enumerate(THRESHOLDS):
                n_comps = [res6['results'][thresh][r]['rbf']['n_comp'] for r in RATIOS]
                offset  = (ti - len(THRESHOLDS)/2) * w6
                bars = ax6c.bar(x6+offset, n_comps, w6,
                                label=str(thresh)+(' ⭐' if thresh==0.95 else ''),
                                color=THRESH_C[thresh], alpha=0.85, edgecolor='white')
            ax6c.set_xticks(x6); ax6c.set_xticklabels(RATIOS, fontsize=11)
            ax6c.set_ylabel('Jumlah Komponen PCA', fontsize=11)
            ax6c.set_title('Jumlah Komponen PCA per Threshold × Rasio Split\n'
                           '(kernel RBF, representatif)',
                           fontsize=13, fontweight='bold')
            ax6c.legend(fontsize=9, title='Threshold', ncol=4)
            ax6c.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6c, use_container_width=True); plt.close()

        # ── Metrik per threshold × kernel ────────────────────────────────────
        with s6b:
            st.markdown(f"**📊 {metric6} per Threshold — Semua Kernel × Semua Rasio**")
            fig6d, axes6d = plt.subplots(2, 2, figsize=(14, 10))
            fig6d.suptitle(
                f'{metric6} per Threshold PCA — Semua Kernel\n'
                f'(⭐ = threshold 0.95)',
                fontsize=14, fontweight='bold'
            )
            for ax, kernel in zip(axes6d.flatten(), KERNELS):
                for ratio in RATIOS:
                    vals = [res6['results'][t][ratio][kernel][mk6] for t in THRESHOLDS]
                    ax.plot([str(t) for t in THRESHOLDS], vals,
                            'o-', alpha=0.7, linewidth=1.5, markersize=5,
                            label=ratio)
                # Highlight 0.95
                ax.axvline(x=THRESHOLDS.index(0.95), color='red',
                           linestyle='--', linewidth=1.5, alpha=0.7)
                lo = min(res6['results'][t][r][kernel][mk6]
                         for t in THRESHOLDS for r in RATIOS) - 0.03
                ax.set_ylim(max(0,lo), 1.08)
                ax.set_title(kernel.upper(), fontweight='bold', fontsize=11)
                ax.set_xlabel('Threshold', fontsize=9)
                ax.set_ylabel(metric6, fontsize=9)
                ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6d, use_container_width=True); plt.close()

            # Tabel rata-rata metrik per threshold (semua kernel × rasio)
            st.markdown(f"**📄 Rata-rata {metric6} per Threshold (semua kernel × rasio)**")
            avg_rows = []
            for thresh in THRESHOLDS:
                avg_val = np.mean([res6['results'][thresh][r][k][mk6]
                                   for r in RATIOS for k in KERNELS])
                n_comp  = int(np.mean([res6['results'][thresh][r]['rbf']['n_comp']
                                       for r in RATIOS]))
                red_pct = np.mean([res6['results'][thresh][r]['rbf']['red_pct']
                                   for r in RATIOS])
                var_exp = np.mean([res6['results'][thresh][r]['rbf']['var_exp']
                                   for r in RATIOS]) * 100
                avg_rows.append({
                    'Threshold':        str(thresh)+(' ⭐' if thresh==0.95 else ''),
                    f'Avg {metric6}':   f"{avg_val:.4f}",
                    'Avg n Komponen':   n_comp,
                    'Avg Var Exp (%)':  f"{var_exp:.2f}%",
                    'Avg Reduksi (%)':  f"{red_pct:.1f}%",
                })
            st.dataframe(pd.DataFrame(avg_rows),
                         use_container_width=True, hide_index=True)

        # ── Heatmap ───────────────────────────────────────────────────────────
        with s6c:
            st.markdown(f"**🔥 Heatmap {metric6} — Threshold × Rasio (semua kernel)**")
            for kernel in KERNELS:
                st.markdown(f"**Kernel: {kernel.upper()}**")
                heat = pd.DataFrame(
                    [[res6['results'][t][r][kernel][mk6] for r in RATIOS]
                     for t in THRESHOLDS],
                    index=[str(t)+(' ⭐' if t==0.95 else '') for t in THRESHOLDS],
                    columns=RATIOS
                )
                fig, ax = plt.subplots(figsize=(9, 4.5))
                sns.heatmap(heat.astype(float), annot=True,
                            fmt='.4f' if mk6=='f1' else '.3f',
                            cmap='YlOrRd', ax=ax,
                            linewidths=0.5, linecolor='white',
                            annot_kws={'size':10,'weight':'bold'},
                            cbar_kws={'shrink':0.8})
                ax.set_title(f'{metric6}: Threshold × Rasio — {kernel.upper()}',
                             fontweight='bold')
                ax.set_xlabel('Rasio Split'); ax.set_ylabel('Threshold PCA')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True); plt.close()

        # ── Kesimpulan ────────────────────────────────────────────────────────
        with s6d:
            st.markdown("**💡 Kesimpulan Poin 6b: Pemilihan Threshold PCA 0.95**")

            # Threshold terbaik per kernel
            best_thresh_rows = []
            for kernel in KERNELS:
                best_t = max(THRESHOLDS,
                             key=lambda t: np.mean([
                                 res6['results'][t][r][kernel][mk6]
                                 for r in RATIOS
                             ]))
                avg_095 = np.mean([res6['results'][0.95][r][kernel][mk6]
                                   for r in RATIOS])
                avg_best= np.mean([res6['results'][best_t][r][kernel][mk6]
                                   for r in RATIOS])
                n_095   = int(np.mean([res6['results'][0.95][r][kernel]['n_comp']
                                       for r in RATIOS]))
                n_best  = int(np.mean([res6['results'][best_t][r][kernel]['n_comp']
                                       for r in RATIOS]))
                best_thresh_rows.append({
                    'Kernel':              kernel.upper(),
                    'Threshold Terbaik':   str(best_t)+(' ⭐' if best_t==0.95 else ''),
                    f'Avg {metric6} (best)': f"{avg_best:.4f}",
                    'n Komp (best)':       n_best,
                    f'Avg {metric6} (0.95)':f"{avg_095:.4f}",
                    'n Komp (0.95)':       n_095,
                    'Selisih':             f"{avg_095-avg_best:+.4f}",
                    '0.95 Optimal?':       '✅ Ya' if best_t==0.95 else
                                           f"⚠️ {best_t} lebih baik",
                })
            st.dataframe(pd.DataFrame(best_thresh_rows),
                         use_container_width=True, hide_index=True)

            # Bar chart perbandingan semua threshold
            fig6e, ax6e = plt.subplots(figsize=(11, 5))
            x6e = np.arange(len(THRESHOLDS)); w6e = 0.15
            for ki, kernel in enumerate(KERNELS):
                avgs = [np.mean([res6['results'][t][r][kernel][mk6]
                                 for r in RATIOS]) for t in THRESHOLDS]
                bars = ax6e.bar(x6e+(ki-1.5)*w6e, avgs, w6e,
                                label=kernel.upper(), color=CLR_K[kernel],
                                alpha=0.87, edgecolor='white')
            ax6e.axvline(x=THRESHOLDS.index(0.95), color='red',
                         linestyle='--', linewidth=2, label='Threshold 0.95')
            ax6e.set_xticks(x6e)
            ax6e.set_xticklabels(
                [str(t)+(' ⭐' if t==0.95 else '') for t in THRESHOLDS],
                fontsize=10
            )
            ax6e.set_ylabel(f'Rata-rata {metric6}', fontsize=11)
            ax6e.set_title(f'Rata-rata {metric6} per Threshold & Kernel\n'
                           f'(avg {len(RATIOS)} rasio split)',
                           fontsize=13, fontweight='bold')
            lo6e = min(np.mean([res6['results'][t][r][k][mk6]
                                 for r in RATIOS for k in KERNELS])
                        for t in THRESHOLDS) - 0.03
            ax6e.set_ylim(max(0,lo6e), 1.08)
            ax6e.legend(fontsize=9); ax6e.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig6e, use_container_width=True); plt.close()

            # Info box kesimpulan
            best_global = max(THRESHOLDS,
                              key=lambda t: np.mean([
                                  res6['results'][t][r][k][mk6]
                                  for r in RATIOS for k in KERNELS
                              ]))
            avg_095_global = np.mean([res6['results'][0.95][r][k][mk6]
                                      for r in RATIOS for k in KERNELS])
            avg_best_global= np.mean([res6['results'][best_global][r][k][mk6]
                                      for r in RATIOS for k in KERNELS])
            n_095_avg = int(np.mean([res6['results'][0.95][r]['rbf']['n_comp']
                                     for r in RATIOS]))

            st.success(
                f"**Kesimpulan Poin 6b:**\n\n"
                f"- Threshold terbaik secara global: **{best_global}** "
                f"(avg {metric6} = {avg_best_global:.4f})\n\n"
                f"- Threshold **0.95** (penelitian): avg {metric6} = "
                f"**{avg_095_global:.4f}**, rata-rata {n_095_avg} komponen\n\n"
                f"- Selisih 0.95 vs terbaik: **{avg_095_global-avg_best_global:+.4f}**\n\n"
                f"- Threshold 0.95 dipilih karena: (1) kompetitif di semua kernel × rasio; "
                f"(2) mempertahankan ≥95% informasi; (3) standar literatur PCA teks; "
                f"(4) keseimbangan optimal antara reduksi dimensi dan performa."
            )
