"""
tabs/tab_smote_compare.py
Tab: Perbandingan Dengan SMOTE vs Tanpa SMOTE
     — semua kernel × semua rasio split × 2 mode PCA
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.pipeline import svm_all_kernels_splits, svm_all_kernels_splits_nosmote

KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
RATIOS  = ['70:30', '80:20', '90:10']
CLR_SM  = '#55A868'   # hijau  = dengan SMOTE
CLR_NS  = '#E24B4A'   # merah  = tanpa SMOTE
CLR_KN  = {'linear': '#4C72B0', 'rbf': '#55A868',
           'poly': '#E24B4A', 'sigmoid': '#F5A623'}


def _annotate(ax, bars, fmt='.3f', color='black', offset=0.003):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:{fmt}}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=color)


def render(df=None, **kwargs):
    st.subheader("⚖️ Perbandingan: Dengan SMOTE vs Tanpa SMOTE")
    st.caption("Semua kernel × semua rasio split × 2 mode (PCA & tanpa PCA) — hasil training nyata")

    if 'feat' not in st.session_state:
        st.warning("Jalankan pipeline terlebih dahulu.", icon="👈")
        return

    s_used  = st.session_state.get('split_label', '80:20')
    k_used  = st.session_state.get('kernel', 'rbf').upper()
    df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "default"

    col_sp, col_md = st.columns(2)
    with col_sp:
        mode_sel = st.selectbox(
            "Mode PCA:", ["Dengan PCA", "Tanpa PCA"], key="smote_mode"
        )
    with col_md:
        metric_sel = st.selectbox(
            "Metrik:", ["Accuracy", "F1-Score", "Precision", "Recall"],
            key="smote_metric"
        )

    mk   = {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[metric_sel]
    suf  = 'pca' if mode_sel == "Dengan PCA" else 'np'
    mk_sm = f'{mk}_{suf}'

    with st.spinner("⏳ Melatih semua model dengan SMOTE..."):
        res_sm = svm_all_kernels_splits(df_hash, df)
    with st.spinner("⏳ Melatih semua model tanpa SMOTE..."):
        res_ns = svm_all_kernels_splits_nosmote(df_hash, df)

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    stab1, stab2, stab3, stab4, stab5 = st.tabs([
        "📄 Tabel Lengkap",
        "📊 Per Kernel",
        "📈 Per Rasio Split",
        "🔥 Heatmap Δ",
        "💡 Kesimpulan",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Tabel Lengkap
    # ══════════════════════════════════════════════════════════════════════
    with stab1:
        st.markdown(f"**📄 Tabel Lengkap — {metric_sel} | {mode_sel}**")
        st.caption("⭐ = kombinasi aktif dari sidebar")

        rows = []
        for kernel in KERNELS:
            for ratio in RATIOS:
                sm = res_sm[kernel][ratio]
                ns = res_ns[kernel][ratio]
                is_cur = " ⭐" if ratio == s_used and kernel.upper() == k_used else ""
                v_sm = sm[mk_sm]
                v_ns = ns[mk_sm]
                delta = v_sm - v_ns
                rows.append({
                    'Kernel':            kernel.upper() + is_cur,
                    'Rasio Split':       ratio,
                    'Train (SMOTE)':     f"{sm['n_train']:,}",
                    'Train (NoSMOTE)':   f"{ns['n_train']:,}",
                    'Test':              f"{sm['n_test']:,}",
                    f'{metric_sel} (SMOTE)':   f"{v_sm*100:.2f}%" if mk=='acc' else f"{v_sm:.4f}",
                    f'{metric_sel} (NoSMOTE)': f"{v_ns*100:.2f}%" if mk=='acc' else f"{v_ns:.4f}",
                    'Δ (SMOTE−NoSMOTE)':       f"{delta*100:+.2f}%" if mk=='acc' else f"{delta:+.4f}",
                    'SMOTE Lebih Baik?': "✅ Ya" if delta > 0 else ("➖ Sama" if delta == 0 else "❌ Tidak"),
                })

        tbl_df = pd.DataFrame(rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)

        # Ringkasan cepat
        smote_wins = sum(1 for r in rows if r['SMOTE Lebih Baik?'] == "✅ Ya")
        no_wins    = sum(1 for r in rows if r['SMOTE Lebih Baik?'] == "❌ Tidak")
        draw       = sum(1 for r in rows if r['SMOTE Lebih Baik?'] == "➖ Sama")
        total      = len(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("SMOTE Lebih Baik",  f"{smote_wins}/{total}", f"{smote_wins/total*100:.0f}%")
        c2.metric("NoSMOTE Lebih Baik",f"{no_wins}/{total}",    f"{no_wins/total*100:.0f}%")
        c3.metric("Sama",              f"{draw}/{total}")

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Per Kernel
    # ══════════════════════════════════════════════════════════════════════
    with stab2:
        st.markdown(f"**📊 {metric_sel} — SMOTE vs NoSMOTE | {mode_sel}**")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'{metric_sel}: Dengan SMOTE vs Tanpa SMOTE\n{mode_sel} — Semua Kernel',
            fontsize=14, fontweight='bold'
        )
        for ax, kernel in zip(axes.flatten(), KERNELS):
            sm_vals = [res_sm[kernel][r][mk_sm] for r in RATIOS]
            ns_vals = [res_ns[kernel][r][mk_sm] for r in RATIOS]
            x = np.arange(len(RATIOS)); w = 0.35
            b1 = ax.bar(x-w/2, sm_vals, w, label='Dengan SMOTE',
                        color=CLR_SM, alpha=0.87, edgecolor='white')
            b2 = ax.bar(x+w/2, ns_vals, w, label='Tanpa SMOTE',
                        color=CLR_NS, alpha=0.87, edgecolor='white')
            _annotate(ax, b1, color='#1a5c30')
            _annotate(ax, b2, color='#8B0000')
            ax.set_xticks(x); ax.set_xticklabels(RATIOS, fontsize=9)
            ax.set_title(f'Kernel: {kernel.upper()}', fontweight='bold', fontsize=11)
            ax.set_ylabel(metric_sel, fontsize=10)
            lo = min(min(sm_vals), min(ns_vals))
            ax.set_ylim(max(0, lo-0.05), 1.08)
            ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

        # Line chart tren semua kernel
        st.markdown(f"**📈 Tren {metric_sel} — SMOTE vs NoSMOTE per Kernel**")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(f'Tren {metric_sel}: SMOTE vs NoSMOTE — Semua Kernel | {mode_sel}',
                      fontsize=12, fontweight='bold')
        for ax, res, title in zip(axes2,
                                   [res_sm, res_ns],
                                   ['Dengan SMOTE', 'Tanpa SMOTE']):
            for kernel in KERNELS:
                vals = [res[kernel][r][mk_sm] for r in RATIOS]
                ax.plot(RATIOS, vals, 'o-', label=kernel.upper(),
                        color=CLR_KN[kernel], linewidth=2.2, markersize=7)
                for i, v in enumerate(vals):
                    ax.annotate(f'{v:.3f}', (RATIOS[i], v),
                                textcoords='offset points', xytext=(0, 7),
                                ha='center', fontsize=7.5,
                                color=CLR_KN[kernel], fontweight='bold')
            lo = min(res[k][r][mk_sm] for k in KERNELS for r in RATIOS)
            ax.set_ylim(max(0, lo-0.05), 1.08)
            ax.set_xlabel('Rasio Split', fontsize=10)
            ax.set_ylabel(metric_sel, fontsize=10)
            ax.set_title(title, fontweight='bold', fontsize=11)
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Per Rasio Split
    # ══════════════════════════════════════════════════════════════════════
    with stab3:
        st.markdown(f"**📈 {metric_sel} per Rasio Split — Semua Kernel | {mode_sel}**")

        fig3, axes3 = plt.subplots(1, 5, figsize=(18, 5))
        fig3.suptitle(
            f'{metric_sel}: SMOTE vs NoSMOTE per Rasio Split (Hijau=SMOTE | Merah=NoSMOTE)',
            fontsize=12, fontweight='bold'
        )
        for ax, ratio in zip(axes3, RATIOS):
            sm_vals = [res_sm[k][ratio][mk_sm] for k in KERNELS]
            ns_vals = [res_ns[k][ratio][mk_sm] for k in KERNELS]
            x = np.arange(len(KERNELS)); w = 0.35
            b1 = ax.bar(x-w/2, sm_vals, w, color=CLR_SM, alpha=0.87,
                        edgecolor='white', label='SMOTE')
            b2 = ax.bar(x+w/2, ns_vals, w, color=CLR_NS, alpha=0.87,
                        edgecolor='white', label='NoSMOTE')
            _annotate(ax, b1, color='#1a5c30', offset=0.002)
            _annotate(ax, b2, color='#8B0000', offset=0.002)
            ax.set_xticks(x)
            ax.set_xticklabels([k.upper() for k in KERNELS], fontsize=8, rotation=15)
            lo = min(min(sm_vals), min(ns_vals))
            ax.set_ylim(max(0, lo-0.05), 1.08)
            ax.set_title(f'Rasio {ratio}', fontweight='bold', fontsize=10)
            ax.set_ylabel(metric_sel, fontsize=9)
            ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.divider()

        # Selisih Δ per rasio
        st.markdown(f"**📊 Selisih Δ {metric_sel} (SMOTE − NoSMOTE) per Kernel per Rasio**")
        fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4.5))
        fig4.suptitle(
            f'Δ {metric_sel} (SMOTE − NoSMOTE) — Hijau=SMOTE lebih baik | Merah=NoSMOTE lebih baik',
            fontsize=11, fontweight='bold'
        )
        for ax, ratio in zip(axes4, RATIOS):
            deltas = [res_sm[k][ratio][mk_sm] - res_ns[k][ratio][mk_sm] for k in KERNELS]
            colors = ['#55A868' if d >= 0 else '#E24B4A' for d in deltas]
            bars = ax.bar([k.upper() for k in KERNELS], deltas,
                          color=colors, alpha=0.85, edgecolor='white', width=0.5)
            ax.axhline(0, color='black', linewidth=1)
            for bar, v in zip(bars, deltas):
                va  = 'bottom' if v >= 0 else 'top'
                yp  = v + (0.001 if v >= 0 else -0.003)
                ax.text(bar.get_x()+bar.get_width()/2, yp,
                        f'{v:+.3f}', ha='center', va=va, fontsize=8, fontweight='bold')
            ax.set_title(f'Rasio {ratio}', fontweight='bold', fontsize=10)
            ax.set_ylabel(f'Δ {metric_sel}', fontsize=9)
            ax.tick_params(axis='x', labelsize=8, rotation=15)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — Heatmap Δ
    # ══════════════════════════════════════════════════════════════════════
    with stab4:
        st.markdown(f"**🔥 Heatmap {metric_sel} — SMOTE vs NoSMOTE | {mode_sel}**")

        col_h1, col_h2 = st.columns(2)
        for col, res, title, cmap in [
            (col_h1, res_sm, 'Dengan SMOTE',  'Greens'),
            (col_h2, res_ns, 'Tanpa SMOTE',   'Reds'),
        ]:
            matrix = pd.DataFrame(
                [[res[k][r][mk_sm] for r in RATIOS] for k in KERNELS],
                index=[k.upper() for k in KERNELS],
                columns=RATIOS,
            )
            with col:
                st.markdown(f"**{title}**")
                fig5, ax5 = plt.subplots(figsize=(6, 3.5))
                sns.heatmap(matrix, annot=True, fmt='.4f', cmap=cmap,
                            ax=ax5, linewidths=0.5, linecolor='white',
                            vmin=matrix.values.min()-0.01,
                            vmax=matrix.values.max()+0.01,
                            annot_kws={'size': 10, 'weight': 'bold'},
                            cbar_kws={'shrink': 0.8})
                ax5.set_title(f'{metric_sel} — {title}', fontweight='bold', fontsize=11)
                ax5.set_xlabel('Rasio Split', fontsize=10)
                ax5.set_ylabel('Kernel', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig5, use_container_width=True)
                plt.close()

        st.divider()

        # Heatmap Δ
        st.markdown(f"**🔥 Heatmap Selisih Δ {metric_sel} (SMOTE − NoSMOTE)**")
        delta_matrix = pd.DataFrame(
            [[res_sm[k][r][mk_sm] - res_ns[k][r][mk_sm] for r in RATIOS]
             for k in KERNELS],
            index=[k.upper() for k in KERNELS],
            columns=RATIOS,
        )
        fig6, ax6 = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(delta_matrix, annot=True, fmt='+.4f', cmap='RdYlGn',
                    ax=ax6, linewidths=0.5, linecolor='white', center=0,
                    annot_kws={'size': 10, 'weight': 'bold'},
                    cbar_kws={'label': 'Δ (SMOTE − NoSMOTE)', 'shrink': 0.8})
        ax6.set_title(
            f'Selisih Δ {metric_sel}: SMOTE − NoSMOTE\n'
            f'(Hijau = SMOTE lebih baik | Merah = NoSMOTE lebih baik)',
            fontweight='bold', fontsize=12)
        ax6.set_xlabel('Rasio Split', fontsize=10)
        ax6.set_ylabel('Kernel', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 5 — Kesimpulan
    # ══════════════════════════════════════════════════════════════════════
    with stab5:
        st.markdown("**💡 Kesimpulan Pengaruh SMOTE terhadap Performa Model**")

        # Ringkasan per kernel
        summary_rows = []
        for kernel in KERNELS:
            sm_f1s  = [res_sm[kernel][r][f'f1_{suf}']  for r in RATIOS]
            ns_f1s  = [res_ns[kernel][r][f'f1_{suf}']  for r in RATIOS]
            sm_accs = [res_sm[kernel][r][f'acc_{suf}'] for r in RATIOS]
            ns_accs = [res_ns[kernel][r][f'acc_{suf}'] for r in RATIOS]
            wins    = sum(1 for s, n in zip(sm_f1s, ns_f1s) if s >= n)
            summary_rows.append({
                'Kernel':              kernel.upper(),
                'SMOTE Unggul (dari 5)': f"{wins}/5",
                'Avg Acc (SMOTE)':     f"{np.mean(sm_accs)*100:.2f}%",
                'Avg Acc (NoSMOTE)':   f"{np.mean(ns_accs)*100:.2f}%",
                'Δ Avg Acc':           f"{(np.mean(sm_accs)-np.mean(ns_accs))*100:+.2f}%",
                'Avg F1 (SMOTE)':      f"{np.mean(sm_f1s):.4f}",
                'Avg F1 (NoSMOTE)':    f"{np.mean(ns_f1s):.4f}",
                'Δ Avg F1':            f"{np.mean(sm_f1s)-np.mean(ns_f1s):+.4f}",
            })
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        st.divider()

        # Bar chart rata-rata Accuracy & F1
        fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))
        fig7.suptitle(f'Rata-rata Performa: SMOTE vs NoSMOTE | {mode_sel}',
                      fontsize=13, fontweight='bold')
        for ax, mk5, lbl in zip(axes7, [f'acc_{suf}', f'f1_{suf}'],
                                 ['Accuracy', 'F1-Score']):
            sm_avg = [np.mean([res_sm[k][r][mk5] for r in RATIOS]) for k in KERNELS]
            ns_avg = [np.mean([res_ns[k][r][mk5] for r in RATIOS]) for k in KERNELS]
            x = np.arange(len(KERNELS)); w = 0.35
            b1 = ax.bar(x-w/2, sm_avg, w, label='Dengan SMOTE',
                        color=CLR_SM, alpha=0.87, edgecolor='white')
            b2 = ax.bar(x+w/2, ns_avg, w, label='Tanpa SMOTE',
                        color=CLR_NS, alpha=0.87, edgecolor='white')
            _annotate(ax, b1, color='#1a5c30')
            _annotate(ax, b2, color='#8B0000')
            ax.set_xticks(x)
            ax.set_xticklabels([k.upper() for k in KERNELS], fontsize=11)
            lo = min(min(sm_avg), min(ns_avg))
            ax.set_ylim(max(0, lo-0.05), 1.08)
            ax.set_title(f'Rata-rata {lbl}', fontweight='bold', fontsize=12)
            ax.set_ylabel(lbl, fontsize=11)
            ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close()

        st.divider()

        # Kesimpulan otomatis
        total_comb = len(KERNELS) * len(RATIOS)
        smote_wins_total = sum(
            1 for k in KERNELS for r in RATIOS
            if res_sm[k][r][f'f1_{suf}'] >= res_ns[k][r][f'f1_{suf}']
        )
        best_sm = max(
            [(k, r, res_sm[k][r][f'f1_{suf}']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )
        best_ns = max(
            [(k, r, res_ns[k][r][f'f1_{suf}']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )
        verdict = "**SMOTE meningkatkan performa**" if smote_wins_total > total_comb/2 \
                  else "**Tanpa SMOTE lebih baik**" if smote_wins_total < total_comb/2 \
                  else "**SMOTE tidak berpengaruh signifikan**"

        st.info(
            f"**Mode: {mode_sel} | Metrik: F1-Score**\n\n"
            f"- Total kombinasi diuji: **{total_comb}** "
            f"({len(KERNELS)} kernel × {len(RATIOS)} rasio)\n"
            f"- SMOTE unggul pada: **{smote_wins_total}/{total_comb}** kombinasi\n"
            f"- {verdict} pada dataset ini\n\n"
            f"**🏆 Terbaik Dengan SMOTE:** "
            f"Kernel `{best_sm[0].upper()}` | Rasio `{best_sm[1]}` | F1 = `{best_sm[2]:.4f}`\n\n"
            f"**🏆 Terbaik Tanpa SMOTE:** "
            f"Kernel `{best_ns[0].upper()}` | Rasio `{best_ns[1]}` | F1 = `{best_ns[2]:.4f}`"
        )
