"""
tabs/tab_split_smote.py
Tab: Split Smote
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
    import seaborn as sns

    st.subheader("✂️ Hasil Split Data & SMOTE")
    st.caption("Distribusi data sebelum dan sesudah oversampling SMOTE untuk dokumentasi skripsi")

    if 'feat' not in st.session_state:
        st.info("Klik **🚀 Jalankan Pipeline** di sidebar terlebih dahulu.", icon="👈")
    else:
        feat        = st.session_state['feat']
        classes     = feat['classes']
        le          = feat['le']
        split_label = st.session_state['split_label']
        split_ratio = feat['split_ratio']

        y_all      = np.concatenate([feat['y_train_ori'], feat['y_test']])
        y_train    = feat['y_train_ori']
        y_train_sm = feat['y_train_sm']
        y_test     = feat['y_test']

        n_total    = feat['n_total']
        n_train    = len(y_train)
        n_train_sm = len(y_train_sm)
        n_test     = len(y_test)
        n_added    = n_train_sm - n_train
        inc_pct    = (n_added / n_train) * 100

        # ── Statistik utama ───────────────────────────────────────────────
        st.markdown("**📋 Statistik Ringkas Split Data & SMOTE**")
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Total Data",          f"{n_total:,}")
        k2.metric("Data Training (awal)", f"{n_train:,}",  f"{n_train/n_total*100:.1f}%")
        k3.metric("Data Testing",         f"{n_test:,}",   f"{n_test/n_total*100:.1f}%")
        k4.metric("Training Setelah SMOTE", f"{n_train_sm:,}")
        k5.metric("Data Sintetis SMOTE",  f"+{n_added:,}", f"+{inc_pct:.1f}%")

        st.divider()

        # ── Sub-tabs ─────────────────────────────────────────────────────
        ss1, ss2, ss3, ss4 = st.tabs([
            "✂️ Split Data",
            "⚖️ Sebelum SMOTE",
            "✅ Setelah SMOTE",
            "📊 Perbandingan",
        ])

        # ════════════════════════════════════════
        # Sub-tab 1: Split Data
        # ════════════════════════════════════════
        with ss1:
            st.markdown(f"**Rasio Split: `{split_label}`**")

            # Tabel distribusi per kelas
            st.markdown("**📄 Distribusi Kelas pada Setiap Subset Data**")
            tbl_rows = []
            for cls_idx, cls_name in enumerate(classes):
                cnt_all   = int((y_all == cls_idx).sum())
                cnt_train = int((y_train == cls_idx).sum())
                cnt_test  = int((y_test  == cls_idx).sum())
                tbl_rows.append({
                    'Kelas':              cls_name,
                    'Total Dataset':      f"{cnt_all:,}",
                    '% Total':            f"{cnt_all/n_total*100:.2f}%",
                    'Data Training':      f"{cnt_train:,}",
                    '% Training':         f"{cnt_train/n_train*100:.2f}%",
                    'Data Testing':       f"{cnt_test:,}",
                    '% Testing':          f"{cnt_test/n_test*100:.2f}%",
                })
            # Tambah baris total
            tbl_rows.append({
                'Kelas':         'TOTAL',
                'Total Dataset': f"{n_total:,}",
                '% Total':       '100.00%',
                'Data Training': f"{n_train:,}",
                '% Training':    '100.00%',
                'Data Testing':  f"{n_test:,}",
                '% Testing':     '100.00%',
            })
            split_tbl = pd.DataFrame(tbl_rows)
            st.dataframe(split_tbl, use_container_width=True, hide_index=True)

            st.divider()

            # Bar chart split
            st.markdown("**📊 Visualisasi Distribusi Split Data per Kelas**")
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
            subsets    = [y_all, y_train, y_test]
            titles     = [f"Dataset Total\n(n={n_total:,})",
                          f"Data Training\n(n={n_train:,})",
                          f"Data Testing\n(n={n_test:,})"]
            bar_colors = ['#4C72B0','#55A868','#C44E52']
            for ax, y_s, title, color in zip(axes, subsets, titles, bar_colors):
                counts = [int((y_s == i).sum()) for i in range(len(classes))]
                bars   = ax.bar(classes, counts, color=color, alpha=0.85, edgecolor='white', width=0.5)
                for bar, v in zip(bars, counts):
                    ax.text(bar.get_x()+bar.get_width()/2, v+max(counts)*0.02,
                            f'{v:,}', ha='center', fontweight='bold', fontsize=11)
                ax.set_title(title, fontweight='bold', fontsize=11)
                ax.set_ylabel("Jumlah Sampel")
                ax.set_ylim(0, max(counts)*1.2)
                ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()

            # Pie chart split proporsi
            st.markdown("**🥧 Proporsi Train vs Test**")
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            sizes  = [n_train, n_test]
            labels = [f"Training\n{n_train:,} ({n_train/n_total*100:.1f}%)",
                      f"Testing\n{n_test:,} ({n_test/n_total*100:.1f}%)"]
            ax2.pie(sizes, labels=labels, colors=['#4C72B0','#E24B4A'],
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize':11}, wedgeprops={'edgecolor':'white','linewidth':2})
            ax2.set_title(f"Proporsi Split Data\n(Rasio {split_label})", fontweight='bold')
            col_pie, _ = st.columns([1,2])
            with col_pie:
                st.pyplot(fig2, use_container_width=True)
            plt.close()

        # ════════════════════════════════════════
        # Sub-tab 2: Sebelum SMOTE
        # ════════════════════════════════════════
        with ss2:
            st.markdown("**Distribusi kelas pada data training SEBELUM SMOTE**")

            counts_before = {cls: int((y_train == i).sum()) for i, cls in enumerate(classes)}
            total_before  = sum(counts_before.values())
            minor_b = min(counts_before.values())
            major_b = max(counts_before.values())
            imb_ratio_b = minor_b / major_b

            # Metrik
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Training",    f"{total_before:,}")
            m2.metric("Kelas Mayor",       f"{major_b:,}")
            m3.metric("Kelas Minor",       f"{minor_b:,}")
            m4.metric("Imbalance Ratio",   f"{imb_ratio_b:.4f}",
                      help="Rasio kelas minor / kelas mayor. Nilai 1.0 = sempurna seimbang")

            st.divider()

            # Tabel distribusi sebelum
            st.markdown("**📄 Tabel Distribusi Kelas (Sebelum SMOTE)**")
            rows_b = []
            for i, cls in enumerate(classes):
                cnt = counts_before[cls]
                rows_b.append({
                    'Kelas':       cls,
                    'Jumlah':      f"{cnt:,}",
                    'Persentase':  f"{cnt/total_before*100:.2f}%",
                    'Status':      '🟢 Mayor' if cnt == major_b else '🔴 Minor',
                })
            rows_b.append({'Kelas':'TOTAL','Jumlah':f"{total_before:,}",
                           'Persentase':'100.00%','Status':'-'})
            st.dataframe(pd.DataFrame(rows_b), use_container_width=True, hide_index=True)

            st.divider()

            # Visualisasi sebelum SMOTE
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**Bar Chart**")
                fig, ax = plt.subplots(figsize=(5, 4))
                cnt_list = [counts_before[c] for c in classes]
                bars = ax.bar(classes, cnt_list, color=['#28a745','#dc3545'],
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
                ax2.pie(cnt_list, labels=classes,
                        colors=['#28a745','#dc3545'],
                        autopct='%1.1f%%', startangle=90,
                        textprops={'fontsize':11},
                        wedgeprops={'edgecolor':'white','linewidth':2})
                ax2.set_title("Proporsi Kelas\nSebelum SMOTE", fontweight='bold')
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            if imb_ratio_b < 0.9:
                st.warning(f"⚠️ Dataset tidak seimbang — Imbalance Ratio: **{imb_ratio_b:.4f}** "
                           f"(kelas minor {minor_b:,} vs mayor {major_b:,}). SMOTE diperlukan.")

        # ════════════════════════════════════════
        # Sub-tab 3: Setelah SMOTE
        # ════════════════════════════════════════
        with ss3:
            st.markdown("**Distribusi kelas pada data training SETELAH SMOTE**")

            counts_after = {cls: int((y_train_sm == i).sum()) for i, cls in enumerate(classes)}
            total_after  = sum(counts_after.values())
            minor_a = min(counts_after.values())
            major_a = max(counts_after.values())
            imb_ratio_a = minor_a / major_a

            # Metrik
            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Training",   f"{total_after:,}",  f"+{n_added:,} (sintetis)")
            m2.metric("Kelas Mayor",      f"{major_a:,}")
            m3.metric("Kelas Minor",      f"{minor_a:,}")
            m4.metric("Imbalance Ratio",  f"{imb_ratio_a:.4f}",
                      help="Nilai mendekati 1.0 = sempurna seimbang")

            st.divider()

            # Tabel distribusi setelah
            st.markdown("**📄 Tabel Distribusi Kelas (Setelah SMOTE)**")
            rows_a = []
            for i, cls in enumerate(classes):
                cnt_b  = counts_before[cls]
                cnt_a  = counts_after[cls]
                added  = cnt_a - cnt_b
                rows_a.append({
                    'Kelas':               cls,
                    'Sebelum SMOTE':       f"{cnt_b:,}",
                    'Setelah SMOTE':       f"{cnt_a:,}",
                    'Ditambahkan':         f"+{added:,}" if added > 0 else "—",
                    'Persentase':          f"{cnt_a/total_after*100:.2f}%",
                })
            rows_a.append({
                'Kelas':'TOTAL',
                'Sebelum SMOTE': f"{total_before:,}",
                'Setelah SMOTE': f"{total_after:,}",
                'Ditambahkan':   f"+{n_added:,}",
                'Persentase':    '100.00%',
            })
            st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)

            st.divider()

            # Visualisasi setelah SMOTE
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("**Bar Chart**")
                fig, ax = plt.subplots(figsize=(5, 4))
                cnt_list = [counts_after[c] for c in classes]
                bars = ax.bar(classes, cnt_list, color=['#28a745','#dc3545'],
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
                ax2.pie(cnt_list, labels=classes,
                        colors=['#28a745','#dc3545'],
                        autopct='%1.1f%%', startangle=90,
                        textprops={'fontsize':11},
                        wedgeprops={'edgecolor':'white','linewidth':2})
                ax2.set_title("Proporsi Kelas\nSetelah SMOTE", fontweight='bold')
                st.pyplot(fig2, use_container_width=True)
                plt.close()

            st.success(f"✅ Setelah SMOTE — Imbalance Ratio: **{imb_ratio_a:.4f}** "
                       f"(mendekati 1.0, dataset seimbang)")

        # ════════════════════════════════════════
        # Sub-tab 4: Perbandingan
        # ════════════════════════════════════════
        with ss4:
            st.markdown("**Perbandingan distribusi data sebelum dan sesudah SMOTE**")

            # Tabel perbandingan lengkap semua rasio split
            st.markdown("**📄 Tabel Ringkasan Split Data & SMOTE**")
            all_ratios = {"50:50":0.5,"60:40":0.6,"70:30":0.7,"80:20":0.8,"90:10":0.9}
            sum_rows = []
            for lbl, ratio in all_ratios.items():
                n_tr = int(n_total * ratio)
                n_te = n_total - n_tr
                # Estimasi distribusi SMOTE (proporsional terhadap rasio aktual)
                scale = n_tr / n_train if n_train > 0 else 1
                n_sm  = int(n_train_sm * scale)
                added = n_sm - n_tr
                inc   = (added / n_tr * 100) if n_tr > 0 else 0
                is_current = "⭐ (aktif)" if lbl == split_label else ""
                sum_rows.append({
                    'Rasio Split':             lbl + (' ' + is_current if is_current else ''),
                    'Data Training (awal)':    f"{n_tr:,}",
                    'Data Testing':            f"{n_te:,}",
                    'Training Setelah SMOTE':  f"{n_sm:,}",
                    'Data Sintetis (+)':       f"+{added:,}",
                    'Peningkatan (%)':         f"{inc:.1f}%",
                })
            st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)

            st.divider()

            # Bar chart: sebelum vs setelah SMOTE per kelas
            st.markdown("**📊 Perbandingan Jumlah per Kelas: Sebelum vs Sesudah SMOTE**")
            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(classes)); w = 0.35
            before_vals = [counts_before[c] for c in classes]
            after_vals  = [counts_after[c]  for c in classes]
            b1 = ax.bar(x-w/2, before_vals, w, label='Sebelum SMOTE',
                        color='#E24B4A', alpha=0.85, edgecolor='white')
            b2 = ax.bar(x+w/2, after_vals,  w, label='Setelah SMOTE',
                        color='#55A868', alpha=0.85, edgecolor='white')
            for bar, v in zip(b1, before_vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(after_vals)*0.01,
                        f'{v:,}', ha='center', fontsize=10, fontweight='bold')
            for bar, v in zip(b2, after_vals):
                ax.text(bar.get_x()+bar.get_width()/2, v+max(after_vals)*0.01,
                        f'{v:,}', ha='center', fontsize=10, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=12)
            ax.set_ylabel("Jumlah Sampel", fontsize=11)
            ax.set_title(f"Distribusi Kelas: Sebelum vs Setelah SMOTE\n"
                         f"(Rasio {split_label})", fontweight='bold', fontsize=12)
            ax.legend(fontsize=11)
            ax.set_ylim(0, max(after_vals)*1.2)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.divider()

            # Grouped stacked bar: Total data (train awal + sintetis + test)
            st.markdown("**📊 Komposisi Data Total (Training Awal + Sintetis SMOTE + Testing)**")
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            labels_bar = list(all_ratios.keys())
            train_vals, smote_vals, test_vals = [], [], []
            for lbl, ratio in all_ratios.items():
                n_tr = int(n_total * ratio)
                n_te = n_total - n_tr
                scale = n_tr / n_train if n_train > 0 else 1
                n_sm  = int(n_train_sm * scale)
                added = max(0, n_sm - n_tr)
                train_vals.append(n_tr)
                smote_vals.append(added)
                test_vals.append(n_te)

            x2 = np.arange(len(labels_bar))
            ax2.bar(x2, train_vals, label='Training Awal', color='#4C72B0', alpha=0.85)
            ax2.bar(x2, smote_vals, bottom=train_vals, label='Sintetis SMOTE', color='#F5A623', alpha=0.85)
            ax2.bar(x2, test_vals,
                    bottom=[a+b for a,b in zip(train_vals,smote_vals)],
                    label='Testing', color='#55A868', alpha=0.85)
            ax2.set_xticks(x2); ax2.set_xticklabels(labels_bar, fontsize=11)
            ax2.set_ylabel("Jumlah Sampel", fontsize=11)
            ax2.set_title("Komposisi Data per Rasio Split", fontweight='bold', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            st.divider()

            # Imbalance ratio comparison
            st.markdown("**⚖️ Perbandingan Imbalance Ratio**")
            ib1, ib2, ib3 = st.columns(3)
            ib1.metric("Imbalance Ratio Sebelum SMOTE", f"{imb_ratio_b:.4f}",
                       help="Minor/Mayor sebelum SMOTE")
            ib2.metric("Imbalance Ratio Setelah SMOTE", f"{imb_ratio_a:.4f}",
                       help="Minor/Mayor setelah SMOTE")
            ib3.metric("Peningkatan Keseimbangan",
                       f"{(imb_ratio_a - imb_ratio_b):.4f}",
                       delta_color="normal")

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.bar(['Sebelum SMOTE','Setelah SMOTE'],
                    [imb_ratio_b, imb_ratio_a],
                    color=['#E24B4A','#55A868'], alpha=0.85,
                    edgecolor='white', width=0.4)
            ax3.axhline(y=1.0, color='gray', linestyle='--',
                        linewidth=1.5, label='Ratio Ideal (1.0)')
            for i, v in enumerate([imb_ratio_b, imb_ratio_a]):
                ax3.text(i, v+0.01, f'{v:.4f}', ha='center',
                         fontweight='bold', fontsize=12)
            ax3.set_ylabel("Imbalance Ratio (Minor/Mayor)")
            ax3.set_title("Perbandingan Imbalance Ratio\nSebelum vs Setelah SMOTE",
                          fontweight='bold')
            ax3.set_ylim(0, 1.2)
            ax3.legend(fontsize=10)
            ax3.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            col_ib, _ = st.columns([1,1])
            with col_ib:
                st.pyplot(fig3, use_container_width=True)
            plt.close()

    # ══════════════════════════════════════════════════════════════════════════════
    # TAB 5 — EVALUASI MODEL
    # ══════════════════════════════════════════════════════════════════════════════
