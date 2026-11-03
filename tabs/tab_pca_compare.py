"""
tabs/tab_pca_compare.py
Tab: Perbandingan SVM dengan PCA vs Tanpa PCA
     — semua kernel × semua rasio split
Dipanggil dari app.py via render().
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.pipeline import svm_all_kernels_splits

KERNELS   = ['linear', 'rbf', 'poly', 'sigmoid']
RATIOS    = ['70:30', '80:20', '90:10']
CLR_PCA   = '#4C72B0'
CLR_NP    = '#E24B4A'
CLR_KN    = {'linear': '#4C72B0', 'rbf': '#55A868',
             'poly':   '#E24B4A', 'sigmoid': '#F5A623'}


def _bar_annotate(ax, bars, fmt='.3f', color='black', offset=0.003):
    for bar in bars:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + offset,
                f'{v:{fmt}}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=color)


def render(df=None, **kwargs):
    st.subheader("🔄 Perbandingan SVM: Dengan PCA vs Tanpa PCA")
    st.caption("Semua kernel (linear, rbf, poly, sigmoid) × semua rasio split — hasil training nyata")

    if 'r_pca' not in st.session_state:
        st.warning("Jalankan pipeline terlebih dahulu.", icon="👈")
        return

    s_used  = st.session_state.get('split_label', '80:20')
    k_used  = st.session_state.get('kernel', 'rbf').upper()
    df_hash = str(hash(tuple(df['content'].tolist()))) if df is not None else "default"

    with st.spinner("⏳ Melatih semua kernel × semua rasio (×2 mode PCA/NoPCA)..."):
        all_res = svm_all_kernels_splits(df_hash, df)

    # ── Sub-tabs ───────────────────────────────────────────────────────────────
    stab1, stab2, stab3, stab4, stab5 = st.tabs([
        "📄 Tabel Lengkap",
        "📊 Per Kernel",
        "📈 Per Rasio Split",
        "🔥 Heatmap",
        "💡 Kesimpulan",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 1 — Tabel Lengkap
    # ══════════════════════════════════════════════════════════════════════
    with stab1:
        st.markdown("**📄 Tabel Lengkap: Semua Kernel × Semua Rasio Split**")
        st.caption("Parameter SVM: C=1, gamma=0.001, degree=2 (poly)")

        # Filter metrik yang ditampilkan
        metric_opt = st.selectbox(
            "Tampilkan metrik:", ["Accuracy", "F1-Score", "Precision", "Recall"],
            key="tbl_metric"
        )
        mk = {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[metric_opt]

        rows = []
        for kernel in KERNELS:
            for ratio in RATIOS:
                r = all_res[kernel][ratio]
                is_cur = " ⭐" if ratio == s_used and kernel.upper() == k_used else ""
                rows.append({
                    'Kernel':        kernel.upper() + is_cur,
                    'Rasio Split':   ratio,
                    'Train (SMOTE)': f"{r['n_train']:,}",
                    'Test':          f"{r['n_test']:,}",
                    'Dim. Awal':     f"{r['orig_dim']:,}",
                    'Komp. PCA':     f"{r['n_comp']:,}",
                    f'{metric_opt} (PCA)':   f"{r[f'{mk}_pca']*100:.2f}%" if mk=='acc' else f"{r[f'{mk}_pca']:.4f}",
                    f'{metric_opt} (NoPCA)': f"{r[f'{mk}_np']*100:.2f}%"  if mk=='acc' else f"{r[f'{mk}_np']:.4f}",
                    'Δ (PCA−NoPCA)': f"{(r[f'{mk}_pca']-r[f'{mk}_np'])*100:+.2f}%" if mk=='acc'
                                     else f"{r[f'{mk}_pca']-r[f'{mk}_np']:+.4f}",
                    'Time PCA (s)':  f"{r['tt_pca']:.3f}",
                    'Time NoPCA (s)':f"{r['tt_np']:.3f}",
                })
        tbl_df = pd.DataFrame(rows)
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 2 — Per Kernel (semua rasio)
    # ══════════════════════════════════════════════════════════════════════
    with stab2:
        st.markdown("**📊 Perbandingan PCA vs Tanpa PCA — Dipilah per Kernel**")

        metric_k = st.selectbox(
            "Metrik:", ["F1-Score", "Accuracy", "Precision", "Recall"],
            key="per_kernel_metric"
        )
        mk2 = {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[metric_k]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Perbandingan {metric_k}: Dengan PCA vs Tanpa PCA\nSemua Kernel × Semua Rasio Split',
            fontsize=14, fontweight='bold'
        )
        for ax, kernel in zip(axes.flatten(), KERNELS):
            pca_vals  = [all_res[kernel][r][f'{mk2}_pca'] for r in RATIOS]
            nopca_vals= [all_res[kernel][r][f'{mk2}_np']  for r in RATIOS]
            x = np.arange(len(RATIOS)); w = 0.35
            b1 = ax.bar(x-w/2, pca_vals,   w, label='Dengan PCA',
                        color=CLR_PCA, alpha=0.87, edgecolor='white')
            b2 = ax.bar(x+w/2, nopca_vals, w, label='Tanpa PCA',
                        color=CLR_NP,  alpha=0.87, edgecolor='white')
            _bar_annotate(ax, b1, color=CLR_PCA)
            _bar_annotate(ax, b2, color='#8B0000')
            ax.set_xticks(x); ax.set_xticklabels(RATIOS, fontsize=9)
            ax.set_title(f'Kernel: {kernel.upper()}', fontweight='bold', fontsize=11)
            ax.set_ylabel(metric_k, fontsize=10)
            lo = min(min(pca_vals), min(nopca_vals))
            ax.set_ylim(max(0, lo - 0.05), 1.08)
            ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.divider()

        # Line chart tren per kernel
        st.markdown(f"**📈 Tren {metric_k} per Rasio Split — Semua Kernel**")
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(f'Tren {metric_k}: PCA vs Tanpa PCA — Semua Kernel',
                      fontsize=13, fontweight='bold')

        for ax, mode, mode_lbl, mk_suf in zip(
            axes2,
            ['pca', 'np'],
            ['Dengan PCA', 'Tanpa PCA'],
            [f'{mk2}_pca', f'{mk2}_np'],
        ):
            for kernel in KERNELS:
                vals = [all_res[kernel][r][mk_suf] for r in RATIOS]
                ax.plot(RATIOS, vals, 'o-', label=kernel.upper(),
                        color=CLR_KN[kernel], linewidth=2.2, markersize=7)
                for i, v in enumerate(vals):
                    ax.annotate(f'{v:.3f}', (RATIOS[i], v),
                                textcoords='offset points', xytext=(0, 7),
                                ha='center', fontsize=7.5,
                                color=CLR_KN[kernel], fontweight='bold')
            lo = min(all_res[k][r][mk_suf] for k in KERNELS for r in RATIOS)
            ax.set_ylim(max(0, lo - 0.05), 1.08)
            ax.set_xlabel('Rasio Split', fontsize=10)
            ax.set_ylabel(metric_k, fontsize=10)
            ax.set_title(mode_lbl, fontweight='bold', fontsize=11)
            ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Per Rasio Split (semua kernel)
    # ══════════════════════════════════════════════════════════════════════
    with stab3:
        st.markdown("**📈 Perbandingan Semua Kernel — Dipilah per Rasio Split**")

        metric_r = st.selectbox(
            "Metrik:", ["F1-Score", "Accuracy", "Precision", "Recall"],
            key="per_ratio_metric"
        )
        mk3 = {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[metric_r]

        fig3, axes3 = plt.subplots(1, 5, figsize=(18, 5))
        fig3.suptitle(
            f'{metric_r}: Semua Kernel per Rasio Split (Biru=PCA / Merah=NoPCA)',
            fontsize=13, fontweight='bold'
        )
        for ax, ratio in zip(axes3, RATIOS):
            pca_vals   = [all_res[k][ratio][f'{mk3}_pca'] for k in KERNELS]
            nopca_vals = [all_res[k][ratio][f'{mk3}_np']  for k in KERNELS]
            x = np.arange(len(KERNELS)); w = 0.35
            b1 = ax.bar(x-w/2, pca_vals,   w, color=CLR_PCA, alpha=0.87,
                        edgecolor='white', label='PCA')
            b2 = ax.bar(x+w/2, nopca_vals, w, color=CLR_NP,  alpha=0.87,
                        edgecolor='white', label='NoPCA')
            _bar_annotate(ax, b1, color=CLR_PCA, offset=0.002)
            _bar_annotate(ax, b2, color='#8B0000', offset=0.002)
            ax.set_xticks(x)
            ax.set_xticklabels([k.upper() for k in KERNELS], fontsize=8, rotation=15)
            lo = min(min(pca_vals), min(nopca_vals))
            ax.set_ylim(max(0, lo - 0.05), 1.08)
            ax.set_title(f'Rasio {ratio}', fontweight='bold', fontsize=10)
            ax.set_ylabel(metric_r, fontsize=9)
            ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        st.divider()

        # Selisih Δ per rasio
        st.markdown(f"**📊 Selisih Δ {metric_r} (PCA − NoPCA) per Kernel per Rasio**")
        fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4.5))
        fig4.suptitle(f'Selisih Δ {metric_r} (PCA − NoPCA) — Hijau=PCA lebih baik | Merah=NoPCA lebih baik',
                      fontsize=12, fontweight='bold')
        for ax, ratio in zip(axes4, RATIOS):
            deltas = [all_res[k][ratio][f'{mk3}_pca'] - all_res[k][ratio][f'{mk3}_np']
                      for k in KERNELS]
            colors = ['#55A868' if d >= 0 else '#E24B4A' for d in deltas]
            bars = ax.bar([k.upper() for k in KERNELS], deltas,
                          color=colors, alpha=0.85, edgecolor='white', width=0.5)
            ax.axhline(0, color='black', linewidth=1)
            for bar, v in zip(bars, deltas):
                ypos = v + (0.001 if v >= 0 else -0.003)
                va   = 'bottom' if v >= 0 else 'top'
                ax.text(bar.get_x()+bar.get_width()/2, ypos,
                        f'{v:+.3f}', ha='center', va=va,
                        fontsize=8, fontweight='bold')
            ax.set_title(f'Rasio {ratio}', fontweight='bold', fontsize=10)
            ax.set_ylabel(f'Δ {metric_r}', fontsize=9)
            ax.tick_params(axis='x', labelsize=8, rotation=15)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 4 — Heatmap
    # ══════════════════════════════════════════════════════════════════════
    with stab4:
        metric_h = st.selectbox(
            "Metrik:", ["F1-Score", "Accuracy", "Precision", "Recall"],
            key="heatmap_metric"
        )
        mk4 = {'Accuracy':'acc','F1-Score':'f1','Precision':'prec','Recall':'rec'}[metric_h]

        st.markdown(f"**🔥 Heatmap {metric_h} — Kernel × Rasio Split**")

        col_h1, col_h2 = st.columns(2)

        for col, mode, mode_lbl, mk_suf, cmap in [
            (col_h1, 'pca',  'Dengan PCA',  f'{mk4}_pca', 'Blues'),
            (col_h2, 'np',   'Tanpa PCA',   f'{mk4}_np',  'Oranges'),
        ]:
            matrix = pd.DataFrame(
                [[all_res[k][r][mk_suf] for r in RATIOS] for k in KERNELS],
                index=[k.upper() for k in KERNELS],
                columns=RATIOS,
            )
            with col:
                st.markdown(f"**{mode_lbl}**")
                fig5, ax5 = plt.subplots(figsize=(6, 3.5))
                sns.heatmap(
                    matrix, annot=True, fmt='.4f', cmap=cmap,
                    ax=ax5, linewidths=0.5, linecolor='white',
                    vmin=matrix.values.min()-0.01,
                    vmax=matrix.values.max()+0.01,
                    annot_kws={'size': 10, 'weight': 'bold'},
                    cbar_kws={'shrink': 0.8},
                )
                ax5.set_title(f'{metric_h} — {mode_lbl}', fontweight='bold', fontsize=11)
                ax5.set_xlabel('Rasio Split', fontsize=10)
                ax5.set_ylabel('Kernel', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig5, use_container_width=True)
                plt.close()

        st.divider()

        # Heatmap selisih Δ
        st.markdown(f"**🔥 Heatmap Selisih Δ {metric_h} (PCA − NoPCA)**")
        delta_matrix = pd.DataFrame(
            [[all_res[k][r][f'{mk4}_pca'] - all_res[k][r][f'{mk4}_np']
              for r in RATIOS] for k in KERNELS],
            index=[k.upper() for k in KERNELS],
            columns=RATIOS,
        )
        fig6, ax6 = plt.subplots(figsize=(8, 3.5))
        sns.heatmap(
            delta_matrix, annot=True, fmt='+.4f', cmap='RdYlGn',
            ax=ax6, linewidths=0.5, linecolor='white',
            center=0,
            annot_kws={'size': 10, 'weight': 'bold'},
            cbar_kws={'label': 'Δ (PCA − NoPCA)', 'shrink': 0.8},
        )
        ax6.set_title(f'Selisih Δ {metric_h}: PCA − NoPCA\n'
                      f'(Hijau = PCA lebih baik | Merah = NoPCA lebih baik)',
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
        st.markdown("**💡 Kesimpulan Otomatis — Semua Kernel × Semua Rasio Split**")

        # Cari kombinasi terbaik
        best_pca  = max(
            [(k, r, all_res[k][r]['f1_pca']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )
        best_np   = max(
            [(k, r, all_res[k][r]['f1_np']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )
        best_acc_pca = max(
            [(k, r, all_res[k][r]['acc_pca']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )
        best_acc_np  = max(
            [(k, r, all_res[k][r]['acc_np']) for k in KERNELS for r in RATIOS],
            key=lambda x: x[2]
        )

        # Hitung berapa kali PCA menang per kernel
        st.markdown("**📊 Ringkasan: Berapa Kali PCA Unggul F1 per Kernel**")
        summary_rows = []
        for kernel in KERNELS:
            pca_wins = sum(1 for r in RATIOS
                           if all_res[kernel][r]['f1_pca'] >= all_res[kernel][r]['f1_np'])
            avg_f1_pca  = np.mean([all_res[kernel][r]['f1_pca']  for r in RATIOS])
            avg_f1_np   = np.mean([all_res[kernel][r]['f1_np']   for r in RATIOS])
            avg_acc_pca = np.mean([all_res[kernel][r]['acc_pca'] for r in RATIOS])
            avg_acc_np  = np.mean([all_res[kernel][r]['acc_np']  for r in RATIOS])
            avg_tt_pca  = np.mean([all_res[kernel][r]['tt_pca']  for r in RATIOS])
            avg_tt_np   = np.mean([all_res[kernel][r]['tt_np']   for r in RATIOS])
            summary_rows.append({
                'Kernel':             kernel.upper(),
                'PCA Unggul (dari 5)':f"{pca_wins}/5",
                'Avg Acc (PCA)':      f"{avg_acc_pca*100:.2f}%",
                'Avg Acc (NoPCA)':    f"{avg_acc_np*100:.2f}%",
                'Avg F1 (PCA)':       f"{avg_f1_pca:.4f}",
                'Avg F1 (NoPCA)':     f"{avg_f1_np:.4f}",
                'Avg Δ F1':           f"{avg_f1_pca-avg_f1_np:+.4f}",
                'Avg Time PCA (s)':   f"{avg_tt_pca:.3f}",
                'Avg Time NoPCA (s)': f"{avg_tt_np:.3f}",
            })
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df, use_container_width=True, hide_index=True)

        st.divider()

        # Bar chart rata-rata F1 per kernel
        st.markdown("**📊 Rata-rata F1-Score per Kernel (PCA vs NoPCA)**")
        fig7, ax7 = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(KERNELS)); w = 0.35
        avg_f1_pca_list = [np.mean([all_res[k][r]['f1_pca'] for r in RATIOS]) for k in KERNELS]
        avg_f1_np_list  = [np.mean([all_res[k][r]['f1_np']  for r in RATIOS]) for k in KERNELS]
        b1 = ax7.bar(x-w/2, avg_f1_pca_list, w, label='Dengan PCA',
                     color=CLR_PCA, alpha=0.87, edgecolor='white')
        b2 = ax7.bar(x+w/2, avg_f1_np_list,  w, label='Tanpa PCA',
                     color=CLR_NP,  alpha=0.87, edgecolor='white')
        _bar_annotate(ax7, b1, color=CLR_PCA)
        _bar_annotate(ax7, b2, color='#8B0000')
        ax7.set_xticks(x)
        ax7.set_xticklabels([k.upper() for k in KERNELS], fontsize=12)
        ax7.set_ylabel('Rata-rata F1-Score', fontsize=11)
        ax7.set_title('Rata-rata F1-Score per Kernel (avg 5 rasio split)',
                      fontweight='bold', fontsize=12)
        lo = min(min(avg_f1_pca_list), min(avg_f1_np_list))
        ax7.set_ylim(max(0, lo - 0.05), 1.08)
        ax7.legend(fontsize=11); ax7.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close()

        st.divider()

        # Info box kesimpulan
        st.info(
            f"**🏆 Model terbaik Dengan PCA:**\n"
            f"Kernel `{best_pca[0].upper()}` | Rasio `{best_pca[1]}` | "
            f"F1 = `{best_pca[2]:.4f}` | Acc = `{best_acc_pca[2]*100:.2f}%`\n\n"
            f"**🏆 Model terbaik Tanpa PCA:**\n"
            f"Kernel `{best_np[0].upper()}` | Rasio `{best_np[1]}` | "
            f"F1 = `{best_np[2]:.4f}` | Acc = `{best_acc_np[2]*100:.2f}%`\n\n"
            f"**📌 Total kombinasi diuji:** "
            f"{len(KERNELS)} kernel × {len(RATIOS)} rasio × 2 mode = "
            f"**{len(KERNELS)*len(RATIOS)*2} model**"
        )

    # ══════════════════════════════════════════════════════════════════════
    # SUB-TAB 3 — Grafik Akurasi Semua Skenario (seperti paper)
    # ══════════════════════════════════════════════════════════════════════
    with stab3:
        import matplotlib.patches as mpatches

        st.markdown("**📊 Grafik Perbandingan Akurasi SVM dengan dan tanpa PCA**")
        st.caption("Semua kernel × semua skenario rasio split — nilai akurasi nyata dari model")

        df_hash2 = str(hash(tuple(df['content'].tolist()))) if df is not None else "default"
        with st.spinner("⏳ Mengambil data akurasi semua skenario..."):
            all_res2 = svm_all_kernels_splits(df_hash2, df)

        KERNELS2 = ['linear', 'rbf', 'poly', 'sigmoid']
        RATIOS2  = ['70:30', '80:20', '90:10']
        COLORS_NP2  = {
            'linear': '#A8D5E2', 'rbf': '#5BA4CF',
            'poly':   '#2E75B6', 'sigmoid': '#1A4E79',
        }
        COLORS_PCA2 = {
            'linear': '#7BC8A4', 'rbf': '#3A9E7A',
            'poly':   '#1E6B52', 'sigmoid': '#0F3D2E',
        }

        n_s   = len(RATIOS2)
        x2    = np.arange(n_s)
        bw    = 0.08
        total = len(KERNELS2) * 2

        fig_acc, ax_acc = plt.subplots(figsize=(16, 7))

        for ki, kernel in enumerate(KERNELS2):
            off_np  = (ki*2)   * bw - (total*bw/2)
            off_pca = (ki*2+1) * bw - (total*bw/2)
            np_vals  = [all_res2[kernel][r]['acc_np']  * 100 for r in RATIOS2]
            pca_vals = [all_res2[kernel][r]['acc_pca'] * 100 for r in RATIOS2]

            b1 = ax_acc.bar(x2+off_np,  np_vals,  bw,
                            color=COLORS_NP2[kernel], alpha=0.92,
                            edgecolor='white', linewidth=0.5,
                            label=f'SVM Tanpa PCA - {kernel.capitalize()}')
            b2 = ax_acc.bar(x2+off_pca, pca_vals, bw,
                            color=COLORS_PCA2[kernel], alpha=0.92,
                            edgecolor='white', linewidth=0.5,
                            label=f'SVM PCA - {kernel.capitalize()}')

            for bar in list(b1)+list(b2):
                h = bar.get_height()
                ax_acc.text(bar.get_x()+bar.get_width()/2, h+0.3,
                            f'{h:.0f}%', ha='center', va='bottom',
                            fontsize=7, fontweight='bold', color='#333')

        for i in range(1, n_s):
            ax_acc.axvline(x=i-0.5, color='#cccccc', linewidth=0.8, linestyle='--')

        ax_acc.set_xticks(x2)
        ax_acc.set_xticklabels(
            [f'Skenario {i+1}\n({r})' for i, r in enumerate(RATIOS2)],
            fontsize=11
        )
        ax_acc.set_ylabel('Akurasi (%)', fontsize=12)
        ax_acc.set_title(
            'Perbandingan Akurasi SVM dengan dan tanpa PCA\n'
            '(Semua Kernel × Semua Skenario Rasio Split)',
            fontsize=13, fontweight='bold', pad=15
        )
        lo_all = min(all_res2[k][r]['acc_np'] for k in KERNELS2 for r in RATIOS2)*100
        ax_acc.set_ylim(max(50, lo_all-10), 110)
        ax_acc.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
        ax_acc.grid(axis='y', alpha=0.3, linestyle='--')
        ax_acc.set_axisbelow(True)

        legend_patches = []
        for kernel in KERNELS2:
            legend_patches.append(mpatches.Patch(
                color=COLORS_NP2[kernel],
                label=f'SVM Tanpa PCA - {kernel.capitalize()}'
            ))
            legend_patches.append(mpatches.Patch(
                color=COLORS_PCA2[kernel],
                label=f'SVM PCA - {kernel.capitalize()}'
            ))
        ax_acc.legend(handles=legend_patches, loc='upper right',
                      fontsize=8.5, framealpha=0.9, ncol=2,
                      bbox_to_anchor=(1, 1))
        plt.tight_layout()
        st.pyplot(fig_acc, use_container_width=True)
        plt.close()
        st.caption(
            "Gambar: Grafik Akurasi SVM dengan dan tanpa PCA — "
            "Semua Kernel (Linear, RBF, Poly, Sigmoid) × "
            "Semua Skenario Rasio Split (70:30 s/d 90:10)"
        )
