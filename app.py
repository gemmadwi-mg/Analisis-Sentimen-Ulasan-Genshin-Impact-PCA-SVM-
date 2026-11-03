"""
app.py  ─  Entry point utama Streamlit
============================================================
Alur KDD yang diikuti (sesuai diagram):
  Mulai
  → Data Selection    : Overview & deskripsi dataset
  → Preprocessing     : Pipeline NLP (case folding → lemmatization)
  → Transformation    : Label Encoding → TF-IDF → Split Data
                        → SMOTE → Feature Selection → PCA
  → Data Mining (SVM) : Training & Evaluasi
  → Evaluation        : Confusion Matrix → Cross Validation
  → Selesai           : Prediksi Sentimen & Komparasi
"""

import streamlit as st

from utils.data_loader import load_data, load_default
from models.pipeline   import prepare_features, train_with_pca, train_without_pca

# ── Tab imports (urutan mengikuti alur KDD) ────────────────────────────────────
import tabs.tab_overview      as t_overview       # Data Selection
import tabs.tab_pipeline      as t_pipeline       # Preprocessing
import tabs.tab_tfidf_chi2    as t_tfidf_chi2     # Transformation: TF-IDF & Chi-Square
import tabs.tab_split         as t_split          # Transformation: Split Data
import tabs.tab_smote         as t_smote          # Transformation: SMOTE
import tabs.tab_pca           as t_pca            # Transformation: PCA
import tabs.tab_evaluasi      as t_evaluasi       # Data Mining: Hasil Training
import tabs.tab_confusion     as t_confusion      # Evaluation: Confusion Matrix
import tabs.tab_validation    as t_validation     # Evaluation: Cross Validation
import tabs.tab_komparasi     as t_komparasi      # Evaluation: Komparasi Metodologi
import tabs.tab_matriks_lengkap as t_matriks       # Evaluation: Matriks Evaluasi Lengkap
import tabs.tab_pca_compare   as t_compare        # Evaluation: PCA vs Tanpa PCA
import tabs.tab_smote_compare as t_smote_compare  # Evaluation: SMOTE vs Tanpa SMOTE
import tabs.tab_wordcloud     as t_wordcloud      # Tambahan: Word Cloud
import tabs.tab_prediksi      as t_prediksi       # Selesai: Prediksi

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Genshin Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: #f8f9fa; border-radius: 10px; padding: 10px; }
    .block-container { padding-top: 1.5rem; }
    .kdd-step {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎮 Genshin Sentiment")
    st.caption("Analisis Sentimen Ulasan Genshin Impact — SVM")
    st.divider()

    # Alur KDD visual di sidebar
    st.markdown("**📋 Alur KDD**")
    st.markdown("""
    🟢 **Data Selection**
    ↓ 🔧 **Preprocessing**
    ↓ 🔄 **Transformation**
    ↓ ⛏️ **Data Mining (SVM)**
    ↓ 📊 **Evaluation**
    ↓ 🏁 **Selesai**
    """)
    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload dataset CSV",
        type=['csv'],
        help="CSV wajib memiliki kolom 'content' dan 'sentiment'",
    )

    st.subheader("⚙️ Parameter SVM")

    ratio_labels = ["70:30", "80:20", "90:10"]
    ratio_values = {"70:30": 0.7, "80:20": 0.8, "90:10": 0.9}

    selected_label = st.select_slider(
        "Rasio Train:Test",
        options=ratio_labels,
        value="80:20",
    )
    split_ratio = ratio_values[selected_label]

    k_features_options = [300, 500, 800, 1000]
    k_features = st.select_slider(
        "Jumlah Fitur Chi-Square",
        options=k_features_options,
        value=1000,
        help="Jumlah fitur terpilih dari 5.000 fitur TF-IDF berdasarkan skor Chi-Square",
    )

    kernel = st.selectbox(
        "Kernel SVM",
        ['rbf', 'linear', 'poly', 'sigmoid'],
        help="Pilih kernel SVM yang akan digunakan",
        key="sidebar_kernel",
    )

    C_options = [0.1, 1, 10, 100, 1000]
    C_param = st.select_slider(
        "Parameter C (regularisasi)",
        options=C_options,
        value=1,
        help="C kecil = margin lebar (toleran kesalahan), C besar = margin ketat (rawan overfitting)",
        key="sidebar_C",
    )

    gamma_options = ["0.001", "0.01", "0.1", "1", "scale", "auto"]
    gamma_param = st.select_slider(
        "Parameter gamma",
        options=gamma_options,
        value="0.001",
        help="Gamma kecil = pengaruh luas per sampel, gamma besar = pengaruh sempit (rawan overfitting)",
        key="sidebar_gamma",
    )

    if kernel == 'poly':
        degree_param = st.select_slider(
            "Parameter degree (khusus kernel Poly)",
            options=[2, 3, 4],
            value=2,
            help="Derajat polinomial — makin tinggi makin kompleks modelnya",
            key="sidebar_degree",
        )
    else:
        degree_param = 2  # tidak dipakai untuk kernel selain poly

    if st.session_state.get('applied_best_params'):
        st.success(
            f"✅ Parameter terbaik untuk kernel **{st.session_state['applied_best_params']}** "
            f"telah diterapkan. Klik **🚀 Jalankan Pipeline KDD** untuk melatih ulang.",
            icon="🎯"
        )
        del st.session_state['applied_best_params']

    run_btn = st.button(
        "🚀 Jalankan Pipeline KDD",
        type="primary",
        use_container_width=True,
    )
    st.divider()
    degree_caption = f" | degree={degree_param}" if kernel == 'poly' else ""
    st.caption(f"Chi-Sq k={k_features} | C={C_param} | gamma={gamma_param}{degree_caption} | SMOTE auto | PCA variance=0.95")

# ── Load data ──────────────────────────────────────────────────────────────────
st.title("🎮 Genshin Impact — Analisis Sentimen SVM")
st.caption("Mengikuti alur KDD: Data Selection → Preprocessing → Transformation → Data Mining → Evaluation")

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_default("genshin.csv")
    if df is not None:
        st.info(
            "💡 Menggunakan file `genshin.csv` default. "
            "Upload file lain via sidebar.",
            icon="ℹ️",
        )

if df is None:
    st.warning("⬅️ Upload file CSV dataset via sidebar untuk memulai.", icon="👈")
    st.stop()

# ── Jalankan pipeline saat tombol diklik ───────────────────────────────────────
if run_btn:
    feat_hash = f"{hash(tuple(df['content'].tolist()))}_{selected_label}_{k_features}"

    with st.spinner("⏳ [Preprocessing] Membersihkan & mempersiapkan teks..."):
        feat = prepare_features(feat_hash, split_ratio, df, k_features=k_features)

    with st.spinner(f"⏳ [Data Mining] Training SVM {kernel.upper()} dengan PCA..."):
        r_pca = train_with_pca(
            f"{feat_hash}_{kernel}_{C_param}_{gamma_param}_{degree_param}",
            kernel, feat, C=C_param, gamma=gamma_param, degree=degree_param,
        )

    with st.spinner(f"⏳ [Data Mining] Training SVM {kernel.upper()} tanpa PCA..."):
        r_nopca = train_without_pca(
            f"{feat_hash}_{kernel}_{C_param}_{gamma_param}_{degree_param}",
            kernel, feat, C=C_param, gamma=gamma_param, degree=degree_param,
        )

    st.session_state.update({
        'feat':        feat,
        'r_pca':       r_pca,
        'r_nopca':     r_nopca,
        'kernel':      kernel,
        'split_label': selected_label,
        'split_ratio': split_ratio,
        'C_param':     C_param,
        'gamma_param': gamma_param,
        'degree_param':degree_param,
        'k_features':  k_features,
    })
    st.success("✅ Pipeline KDD selesai dijalankan!", icon="🎉")

# ══════════════════════════════════════════════════════════════════════════════
# TABS — Diurutkan sesuai alur KDD
# ══════════════════════════════════════════════════════════════════════════════

# Kelompok tab berdasarkan tahapan KDD
st.markdown("---")

# Header tahapan KDD sebagai expander/section
kdd_stages = {
    "🟢 Data Selection":          ["📊 Overview Dataset"],
    "🔧 Preprocessing":           ["🔧 Pipeline NLP"],
    "🔄 Transformation":          ["📐 TF-IDF & Chi-Square", "✂️ Split Data", "⚖️ SMOTE", "🔷 PCA"],
    "⛏️ Data Mining (SVM)":       ["📈 Hasil Training"],
    "📊 Evaluation":              ["🟦 Confusion Matrix", "✅ Cross Validation",
                                   "🔎 Komparasi Metodologi", "🔄 PCA vs Tanpa PCA",
                                   "⚖️ SMOTE vs Tanpa SMOTE", "☁️ Word Cloud"],
    "🏁 Selesai":                 ["🔍 Prediksi Sentimen"],
}

# Tampilkan progress alur KDD
cols_kdd = st.columns(len(kdd_stages))
for col, (stage, _) in zip(cols_kdd, kdd_stages.items()):
    col.markdown(
        f"<div style='text-align:center;font-size:11px;"
        f"padding:4px;border:1px solid #ddd;border-radius:8px;"
        f"background:#f8f9fa'>{stage}</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Semua tab diurutkan sesuai alur KDD ───────────────────────────────────────
(tab1,  tab2,  tab3,  tab4,  tab5,
 tab6,  tab7,  tab8,  tab9,  tab10,
 tab11, tab12, tab13, tab14, tab15) = st.tabs([
    # ── DATA SELECTION ──────────────────────────
    "📊 [1] Data Selection",

    # ── PREPROCESSING ───────────────────────────
    "🔧 [2] Preprocessing",

    # ── TRANSFORMATION ──────────────────────────
    "📐 [3] TF-IDF & Chi-Square",
    "✂️ [4] Split Data",
    "⚖️ [5] SMOTE",
    "🔷 [6] PCA",

    # ── DATA MINING ─────────────────────────────
    "📈 [7] Hasil Training SVM",

    # ── EVALUATION ──────────────────────────────
    "🟦 [8] Confusion Matrix",
    "✅ [9] Cross Validation",
    "🔎 [10] Komparasi Metodologi",
    "📋 [11] Matriks Evaluasi Lengkap",
    "🔄 [12] PCA vs Tanpa PCA",
    "⚖️ [13] SMOTE vs Tanpa SMOTE",
    "☁️ [14] Word Cloud",

    # ── SELESAI ─────────────────────────────────
    "🔍 [15] Prediksi Sentimen",
])

# ── Data Selection ─────────────────────────────────────────────────────────────
with tab1:
    st.info("**Tahap KDD: Data Selection** — Deskripsi dan eksplorasi dataset sebelum proses seleksi dan pemodelan.", icon="🟢")
    t_overview.render(df=df)

# ── Preprocessing ──────────────────────────────────────────────────────────────
with tab2:
    st.info("**Tahap KDD: Preprocessing** — Case folding → Normalisasi → Cleaning → Tokenizing → Stopword removal → Lemmatization", icon="🔧")
    t_pipeline.render(df=df)

# ── Transformation: TF-IDF & Chi-Square ───────────────────────────────────────
with tab3:
    st.info("**Tahap KDD: Transformation** — Vectorization TF-IDF dan seleksi fitur Chi-Square.", icon="🔄")
    t_tfidf_chi2.render(df=df)

# ── Transformation: Split Data ─────────────────────────────────────────────────
with tab4:
    st.info("**Tahap KDD: Transformation** — Pembagian data training & testing.", icon="🔄")
    t_split.render(df=df)

# ── Transformation: SMOTE ──────────────────────────────────────────────────────
with tab5:
    st.info("**Tahap KDD: Transformation** — Penyeimbangan kelas pada data training dengan SMOTE.", icon="🔄")
    t_smote.render(df=df)

# ── Transformation: PCA ───────────────────────────────────────────────────────
with tab6:
    st.info("**Tahap KDD: Transformation** — Reduksi dimensi dengan Principal Component Analysis (variance=0.95).", icon="🔄")
    t_pca.render(df=df)

# ── Data Mining: Hasil Training ───────────────────────────────────────────────
with tab7:
    st.info("**Tahap KDD: Data Mining** — Hasil training Support Vector Machine (SVM) dengan parameter terpilih.", icon="⛏️")
    t_evaluasi.render(df=df)

# ── Evaluation: Confusion Matrix ──────────────────────────────────────────────
with tab8:
    st.info("**Tahap KDD: Evaluation** — Confusion matrix dan perhitungan manual metrik performa (akurasi, presisi, recall, F1).", icon="📊")
    t_confusion.render(df=df)

# ── Evaluation: Cross Validation ──────────────────────────────────────────────
with tab9:
    st.info("**Tahap KDD: Evaluation** — Validasi model menggunakan Train/Validation/Test split dan 5-Fold Cross-Validation.", icon="📊")
    t_validation.render(df=df)

# ── Evaluation: Komparasi Metodologi ──────────────────────────────────────────
with tab10:
    st.info("**Tahap KDD: Evaluation** — Komparasi metodologi: SMOTE vs NoSMOTE, Chi-Square vs Full TF-IDF, dan threshold PCA.", icon="📊")
    t_komparasi.render(df=df)

# ── Evaluation: Matriks Evaluasi Lengkap ──────────────────────────────────────
with tab11:
    st.info("**Tahap KDD: Evaluation** — Matriks evaluasi lengkap (blueprint tabel skripsi) dengan data asli dari model.", icon="📊")
    t_matriks.render(df=df)

# ── Evaluation: PCA vs Tanpa PCA ──────────────────────────────────────────────
with tab12:
    st.info("**Tahap KDD: Evaluation** — Perbandingan performa SVM dengan PCA vs tanpa PCA pada semua kernel dan rasio split.", icon="📊")
    t_compare.render(df=df)

# ── Evaluation: SMOTE vs Tanpa SMOTE ──────────────────────────────────────────
with tab13:
    st.info("**Tahap KDD: Evaluation** — Perbandingan performa SVM dengan SMOTE vs tanpa SMOTE pada semua kernel dan rasio split.", icon="📊")
    t_smote_compare.render(df=df)

# ── Tambahan: Word Cloud ───────────────────────────────────────────────────────
with tab14:
    st.info("**Visualisasi Tambahan** — Kata-kata dominan per kategori sentimen dari hasil preprocessing.", icon="☁️")
    t_wordcloud.render(df=df)

# ── Selesai: Prediksi Sentimen ─────────────────────────────────────────────────
with tab15:
    st.info("**Selesai** — Gunakan model yang telah dilatih untuk memprediksi sentimen ulasan baru.", icon="🏁")
    t_prediksi.render(df=df)
