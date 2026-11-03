"""
models/pipeline.py
Pipeline ML:
  - prepare_features : TF-IDF → Chi-Square → Split → SMOTE
  - train_with_pca   : StandardScaler → PCA → SVM
  - train_without_pca: SVM langsung dari fitur Chi-Square
"""

import time
import streamlit as st
from utils.preprocessing import load_nltk, get_stop_words, preprocess_text
from utils.preprocessing import tokenize_remove_stopwords, lemmatize_tokens


# ── Feature preparation ────────────────────────────────────────────────────────
@st.cache_resource
def prepare_features(df_hash: str, split_ratio: float, _df, k_features: int = 1000):
    """
    Preprocessing bersama — hasilnya di-cache agar model PCA
    dan tanpa-PCA menggunakan data yang identik.

    Alur:
        raw text → clean → TF-IDF (5000) → Chi-Square (k_features)
        → train_test_split (stratified) → SMOTE

    Parameters
    ----------
    k_features : int — jumlah fitur yang dipilih Chi-Square (300, 500, 800, 1000)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    import numpy as np

    load_nltk()
    sw = get_stop_words()

    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    # Label encoding
    le = LabelEncoder()
    y  = le.fit_transform(df['sentiment'])

    # TF-IDF
    tfidf   = TfidfVectorizer(max_features=5000)
    X_tfidf = tfidf.fit_transform(df['final'])

    # Chi-Square feature selection
    k_feat   = min(k_features, X_tfidf.shape[1])
    chi2_sel = SelectKBest(chi2, k=k_feat)
    X_chi2   = chi2_sel.fit_transform(X_tfidf, y)

    # Train-test split (stratified) — turut melacak indeks asli baris dataframe
    idx_all = np.arange(len(y))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_chi2, y, idx_all,
        train_size=split_ratio,
        test_size=round(1 - split_ratio, 10),
        random_state=42,
        stratify=y,
    )

    # SMOTE oversampling
    try:
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    except Exception:
        X_train_sm, y_train_sm = X_train, y_train

    return {
        # Data untuk training
        'X_train_sm':  X_train_sm,
        'y_train_sm':  y_train_sm,
        # Data asli sebelum SMOTE (untuk visualisasi)
        'X_train_ori': X_train,
        'y_train_ori': y_train,
        # Data testing
        'X_test':      X_test,
        'y_test':      y_test,
        'test_idx':    idx_test,
        'train_idx':   idx_train,
        # Objek transformer (dipakai ulang saat prediksi)
        'le':          le,
        'tfidf':       tfidf,
        'chi2_sel':    chi2_sel,
        # Meta-info
        'classes':     le.classes_,
        'n_features':  X_chi2.shape[1],
        'k_features':  k_feat,
        'n_total':     len(y),
        'split_ratio': split_ratio,
        'df_final':    df[['content', 'final', 'sentiment']].copy(),
    }


# ── SVM dengan PCA ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_with_pca(feat_hash: str, kernel: str, _feat: dict,
                    C: float = 1, gamma="0.001", degree: int = 2) -> dict:
    """
    StandardScaler → PCA (variance=0.95) → SVM.

    Parameter SVM: dapat dikustomisasi (C, gamma, degree untuk poly).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    X_tr = _feat['X_train_sm'].toarray()
    X_te = _feat['X_test'].toarray()

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    pca      = PCA(n_components=0.95, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_sc)
    X_te_pca = pca.transform(X_te_sc)

    svm = _build_svm(kernel, C=C, gamma=gamma, degree=degree)

    t0 = time.time()
    svm.fit(X_tr_pca, _feat['y_train_sm'])
    train_t = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(X_te_pca)
    pred_t = time.time() - t0

    y_test  = _feat['y_test']
    classes = _feat['classes']

    return {
        **_metrics(y_test, y_pred, classes),
        'train_time': train_t,
        'pred_time':  pred_t,
        'n_train':    len(_feat['y_train_sm']),
        'n_test':     len(y_test),
        'pca_n':      int(pca.n_components_),
        'pca_var':    float(pca.explained_variance_ratio_.sum()),
        'pca_evr':    pca.explained_variance_ratio_,
        'pca_ev':     pca.explained_variance_,
        'orig_dim':   X_tr.shape[1],
        'scaler':     scaler,
        'pca':        pca,
        'svm':        svm,
    }


# ── SVM tanpa PCA ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_without_pca(feat_hash: str, kernel: str, _feat: dict,
                       C: float = 1, gamma="0.001", degree: int = 2) -> dict:
    """
    SVM langsung dari fitur Chi-Square (tanpa PCA/Scaler).

    Parameter SVM: dapat dikustomisasi (C, gamma, degree untuk poly).
    """
    from sklearn.svm import SVC
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )

    svm = _build_svm(kernel, C=C, gamma=gamma, degree=degree)

    t0 = time.time()
    svm.fit(_feat['X_train_sm'], _feat['y_train_sm'])
    train_t = time.time() - t0

    t0 = time.time()
    y_pred = svm.predict(_feat['X_test'])
    pred_t = time.time() - t0

    y_test  = _feat['y_test']
    classes = _feat['classes']

    return {
        **_metrics(y_test, y_pred, classes),
        'train_time': train_t,
        'pred_time':  pred_t,
        'n_train':    len(_feat['y_train_sm']),
        'n_test':     len(y_test),
        'svm':        svm,
    }


# ── Prediksi model terbaik (utk Word Cloud sesuai model terbaik) ────────────────
@st.cache_resource
def predict_best_model(df_hash: str, _df, best: dict, use_chi2: bool = True) -> dict:
    """
    Menjalankan ulang SATU konfigurasi model (hasil _compute_best_overall) secara
    utuh — preprocessing, TF-IDF, Chi-Square, split, SMOTE, PCA (jika ada), SVM —
    lalu mengembalikan prediksi pada data uji beserta teks asli & test_idx-nya,
    agar dapat dipakai membangun Word Cloud berdasarkan label HASIL PREDIKSI
    model terbaik (bukan sekadar label asli/ground-truth).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le    = LabelEncoder()
    y_all = le.fit_transform(df['sentiment'])
    tfidf = TfidfVectorizer(max_features=5000)
    X_tf  = tfidf.fit_transform(df['final'])

    if use_chi2:
        chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
        X_all    = chi2_sel.fit_transform(X_tf, y_all)
    else:
        X_all = X_tf

    ratio_val = {'90:10': 0.9, '80:20': 0.8, '70:30': 0.7}[best['ratio']]
    idx_all = np.arange(len(y_all))
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X_all, y_all, idx_all, train_size=ratio_val,
        test_size=round(1 - ratio_val, 10), random_state=42, stratify=y_all)

    is_pca    = best['arch'].startswith('Dengan PCA')
    is_after  = 'Setelah SMOTE' in best['arch']

    if is_after:
        try:
            sm = SMOTE(random_state=42)
            X_tr_use, y_tr_use = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_use, y_tr_use = X_tr, y_tr
    else:
        X_tr_use, y_tr_use = X_tr, y_tr

    if is_pca:
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_use.toarray())
        X_te_sc = scaler.transform(X_te.toarray())
        pca = PCA(n_components=best['pct'] / 100, random_state=42)
        X_tr_final = pca.fit_transform(X_tr_sc)
        X_te_final = pca.transform(X_te_sc)
    else:
        X_tr_final, X_te_final = X_tr_use, X_te

    svm = _build_svm(best['kernel'], C=best['C'],
                      gamma=(best['gamma'] if best['gamma'] is not None else 'scale'))
    svm.fit(X_tr_final, y_tr_use)
    y_pred = svm.predict(X_te_final)

    return {
        'y_test':  y_te,
        'y_pred':  y_pred,
        'test_idx': idx_te,
        'df_final': df[['content', 'final']].copy(),
        'classes': le.classes_,
    }


# ── Helpers internal ───────────────────────────────────────────────────────────
def _build_svm(kernel: str, C: float = 1, gamma="0.001", degree: int = 2):
    """
    Bangun objek SVC dengan parameter yang bisa dikustomisasi.

    Parameters
    ----------
    kernel : str   — 'linear', 'rbf', 'poly', 'sigmoid'
    C      : float — parameter regularisasi
    gamma  : float atau str — koefisien kernel ('scale', 'auto', atau angka)
    degree : int   — derajat polinomial (hanya dipakai untuk kernel 'poly')
    """
    from sklearn.svm import SVC
    # Konversi gamma dari string ke float jika bukan 'scale'/'auto'
    if isinstance(gamma, str) and gamma not in ('scale', 'auto'):
        gamma = float(gamma)
    params = dict(C=float(C), gamma=gamma, random_state=42, probability=True)
    if kernel == 'poly':
        params['degree'] = int(degree)
    return SVC(kernel=kernel, **params)


def _metrics(y_test, y_pred, classes) -> dict:
    """Hitung semua metrik evaluasi dan kembalikan sebagai dict."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, classification_report,
    )
    return {
        'accuracy':  float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'recall':    float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'f1':        float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
        'cm':        confusion_matrix(y_test, y_pred),
        'cr':        classification_report(
                         y_test, y_pred,
                         target_names=classes,
                         output_dict=True,
                     ),
    }


# ── PCA semua rasio split (untuk perbandingan) ─────────────────────────────────
@st.cache_resource
def pca_all_splits(df_hash: str, _df) -> list:
    """
    Jalankan PCA (variance=0.95) untuk semua 5 rasio split
    menggunakan data yang sama (TF-IDF + Chi-Square).
    Kembalikan list of dict, satu per rasio.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from imblearn.over_sampling import SMOTE

    load_nltk()
    sw = get_stop_words()

    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le      = LabelEncoder()
    y       = le.fit_transform(df['sentiment'])
    tfidf   = TfidfVectorizer(max_features=5000)
    X_tf    = tfidf.fit_transform(df['final'])
    chi2_sel= SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2  = chi2_sel.fit_transform(X_tf, y)
    orig_dim = X_chi2.shape[1]

    ratios = [
        ("70:30", 0.7),
        ("80:20", 0.8), ("90:10", 0.9),
    ]

    results = []
    for label, ratio in ratios:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y,
            train_size=ratio,
            test_size=round(1 - ratio, 10),
            random_state=42, stratify=y,
        )
        try:
            smote = SMOTE(random_state=42)
            X_tr_sm, _ = smote.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm = X_tr

        # ── PCA SEBELUM SMOTE ────────────────────────────────────────────────
        X_tr_ori_d  = X_tr.toarray()
        scaler_ori  = StandardScaler()
        X_tr_ori_sc = scaler_ori.fit_transform(X_tr_ori_d)
        pca_ori     = PCA(n_components=0.95, random_state=42)
        pca_ori.fit(X_tr_ori_sc)
        n_comp_before  = int(pca_ori.n_components_)
        var_exp_before = float(pca_ori.explained_variance_ratio_.sum())

        # ── PCA SESUDAH SMOTE ────────────────────────────────────────────────
        X_tr_d  = X_tr_sm.toarray()
        X_te_d  = X_te.toarray()
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_d)

        pca = PCA(n_components=0.95, random_state=42)
        pca.fit(X_tr_sc)

        n_comp  = int(pca.n_components_)
        var_exp = float(pca.explained_variance_ratio_.sum())
        red_pct = (1 - n_comp / orig_dim) * 100

        results.append({
            'label':     label,
            'ratio':     ratio,
            'n_train':   X_tr_sm.shape[0],
            'n_train_before_smote': X_tr.shape[0],
            'n_test':    X_te.shape[0],
            'orig_dim':  orig_dim,
            'n_comp':    n_comp,
            'var_exp':   var_exp,
            'red_pct':   red_pct,
            'evr':       pca.explained_variance_ratio_,
            # Info tambahan: PCA sebelum SMOTE
            'n_comp_before_smote':  n_comp_before,
            'var_exp_before_smote': var_exp_before,
            'max_possible_comp_before': min(X_tr.shape[0], orig_dim),
            'max_possible_comp_after':  min(X_tr_sm.shape[0], orig_dim),
        })

    return results


# ── SVM semua kernel × semua rasio split (PCA dan tanpa PCA) ──────────────────
@st.cache_resource
def svm_all_kernels_splits(df_hash: str, _df) -> dict:
    """
    Latih SVM untuk:
      - Semua kernel  : linear, rbf, poly, sigmoid
      - Semua rasio   : 70:30, 80:20, 90:10
      - Dua mode      : dengan PCA dan tanpa PCA

    Return dict:
      {
        kernel: {
          rasio_label: {
            acc_pca, prec_pca, rec_pca, f1_pca, tt_pca,
            acc_np,  prec_np,  rec_np,  f1_np,  tt_np,
            n_train, n_test, orig_dim, n_comp, var_exp
          }
        }
      }
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    from imblearn.over_sampling import SMOTE

    load_nltk()
    sw = get_stop_words()

    # ── Preprocessing & vectorization (sekali) ─────────────────────────────
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le       = LabelEncoder()
    y        = le.fit_transform(df['sentiment'])
    tfidf    = TfidfVectorizer(max_features=5000)
    X_tf     = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y)

    KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
    RATIOS  = [("70:30", 0.7),
               ("80:20", 0.8), ("90:10", 0.9)]

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly':
            p['degree'] = 2
        return SVC(kernel=k, **p)

    # ── Siapkan data per rasio (split + SMOTE + PCA) sekali ────────────────
    split_data = {}
    for label, ratio in RATIOS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y,
            train_size=ratio,
            test_size=round(1 - ratio, 10),
            random_state=42, stratify=y,
        )
        try:
            smote = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        # PCA
        X_tr_d  = X_tr_sm.toarray()
        X_te_d  = X_te.toarray()
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_d)
        X_te_sc = scaler.transform(X_te_d)
        pca     = PCA(n_components=0.95, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)

        split_data[label] = {
            'X_tr_sm':  X_tr_sm,
            'X_te':     X_te,
            'X_tr_pca': X_tr_pca,
            'X_te_pca': X_te_pca,
            'y_tr_sm':  y_tr_sm,
            'y_te':     y_te,
            'n_train':  len(y_tr_sm),
            'n_test':   len(y_te),
            'orig_dim': X_chi2.shape[1],
            'n_comp':   int(pca.n_components_),
            'var_exp':  float(pca.explained_variance_ratio_.sum()),
        }

    # ── Training semua kernel × semua rasio ────────────────────────────────
    results = {}
    for kernel in KERNELS:
        results[kernel] = {}
        for label, _ in RATIOS:
            d = split_data[label]

            # Dengan PCA
            svm_pca = build_svm(kernel)
            t0 = time.time()
            svm_pca.fit(d['X_tr_pca'], d['y_tr_sm'])
            tt_pca = time.time() - t0
            yp_pca = svm_pca.predict(d['X_te_pca'])

            # Tanpa PCA
            svm_np = build_svm(kernel)
            t0 = time.time()
            svm_np.fit(d['X_tr_sm'], d['y_tr_sm'])
            tt_np = time.time() - t0
            yp_np = svm_np.predict(d['X_te'])

            y_te = d['y_te']
            results[kernel][label] = {
                'n_train':  d['n_train'],
                'n_test':   d['n_test'],
                'orig_dim': d['orig_dim'],
                'n_comp':   d['n_comp'],
                'var_exp':  d['var_exp'],
                # Dengan PCA
                'acc_pca':  float(accuracy_score(y_te, yp_pca)),
                'prec_pca': float(precision_score(y_te, yp_pca, average='binary', zero_division=0)),
                'rec_pca':  float(recall_score(y_te, yp_pca, average='binary', zero_division=0)),
                'f1_pca':   float(f1_score(y_te, yp_pca, average='binary', zero_division=0)),
                'tt_pca':   tt_pca,
                # Tanpa PCA
                'acc_np':   float(accuracy_score(y_te, yp_np)),
                'prec_np':  float(precision_score(y_te, yp_np, average='binary', zero_division=0)),
                'rec_np':   float(recall_score(y_te, yp_np, average='binary', zero_division=0)),
                'f1_np':    float(f1_score(y_te, yp_np, average='binary', zero_division=0)),
                'tt_np':    tt_np,
            }

    return results


# ── SVM tanpa SMOTE — semua kernel × semua rasio ───────────────────────────────
@st.cache_resource
def svm_all_kernels_splits_nosmote(df_hash: str, _df) -> dict:
    """
    Sama persis dengan svm_all_kernels_splits TAPI tanpa SMOTE.
    Digunakan untuk membandingkan pengaruh SMOTE terhadap performa model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)

    load_nltk()
    sw = get_stop_words()

    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le       = LabelEncoder()
    y        = le.fit_transform(df['sentiment'])
    tfidf    = TfidfVectorizer(max_features=5000)
    X_tf     = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y)

    KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
    RATIOS  = [("70:30", 0.7),
               ("80:20", 0.8), ("90:10", 0.9)]

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly':
            p['degree'] = 2
        return SVC(kernel=k, **p)

    # Siapkan data per rasio (tanpa SMOTE)
    split_data = {}
    for label, ratio in RATIOS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y,
            train_size=ratio,
            test_size=round(1 - ratio, 10),
            random_state=42, stratify=y,
        )
        # ── Dengan PCA (tanpa SMOTE) ───────────────────────────────────────
        X_tr_d  = X_tr.toarray()
        X_te_d  = X_te.toarray()
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_d)
        X_te_sc = scaler.transform(X_te_d)
        pca     = PCA(n_components=0.95, random_state=42)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)

        split_data[label] = {
            'X_tr':     X_tr,
            'X_te':     X_te,
            'X_tr_pca': X_tr_pca,
            'X_te_pca': X_te_pca,
            'y_tr':     y_tr,
            'y_te':     y_te,
            'n_train':  len(y_tr),
            'n_test':   len(y_te),
            'orig_dim': X_chi2.shape[1],
            'n_comp':   int(pca.n_components_),
        }

    # Training semua kernel × semua rasio (tanpa SMOTE)
    results = {}
    for kernel in KERNELS:
        results[kernel] = {}
        for label, _ in RATIOS:
            d = split_data[label]

            # Dengan PCA
            svm_pca = build_svm(kernel)
            t0 = time.time()
            svm_pca.fit(d['X_tr_pca'], d['y_tr'])
            tt_pca = time.time() - t0
            yp_pca = svm_pca.predict(d['X_te_pca'])

            # Tanpa PCA
            svm_np = build_svm(kernel)
            t0 = time.time()
            svm_np.fit(d['X_tr'], d['y_tr'])
            tt_np = time.time() - t0
            yp_np = svm_np.predict(d['X_te'])

            y_te = d['y_te']
            results[kernel][label] = {
                'n_train':  d['n_train'],
                'n_test':   d['n_test'],
                'orig_dim': d['orig_dim'],
                'n_comp':   d['n_comp'],
                # Dengan PCA
                'acc_pca':  float(accuracy_score(y_te, yp_pca)),
                'prec_pca': float(precision_score(y_te, yp_pca, average='binary', zero_division=0)),
                'rec_pca':  float(recall_score(y_te, yp_pca, average='binary', zero_division=0)),
                'f1_pca':   float(f1_score(y_te, yp_pca, average='binary', zero_division=0)),
                'tt_pca':   tt_pca,
                # Tanpa PCA
                'acc_np':   float(accuracy_score(y_te, yp_np)),
                'prec_np':  float(precision_score(y_te, yp_np, average='binary', zero_division=0)),
                'rec_np':   float(recall_score(y_te, yp_np, average='binary', zero_division=0)),
                'f1_np':    float(f1_score(y_te, yp_np, average='binary', zero_division=0)),
                'tt_np':    tt_np,
            }

    return results



# ── Test Validation (Train / Validation / Test Split) ─────────────────────────
@st.cache_resource
def run_test_validation(feat_hash: str, use_pca: bool, val_ratio: float, _feat: dict) -> dict:
    """
    Validasi model menggunakan 3-way split: Train / Validation / Test.
    Alur:
      1. Data training (dari split utama) dibagi lagi → Training & Validation
      2. Model dilatih pada Training, dievaluasi pada Validation
      3. Model terbaik dievaluasi final pada Test (hold-out)
      4. Performa Train vs Val vs Test dibandingkan untuk deteksi overfitting

    Mengembalikan hasil untuk semua kernel dan mode PCA.
    """
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)
    import numpy as np

    X_train_sm = _feat['X_train_sm']
    X_test     = _feat['X_test']
    y_train    = _feat['y_train_sm']
    y_test     = _feat['y_test']
    classes    = _feat['classes']

    # ── Split Training → Train + Validation ───────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_sm, y_train,
        test_size=val_ratio,
        random_state=42,
        stratify=y_train,
    )

    # ── PCA (opsional) ────────────────────────────────────────────────────────
    if use_pca:
        X_tr_d   = X_tr.toarray()
        X_val_d  = X_val.toarray()
        X_te_d   = X_test.toarray()
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_d)
        X_val_sc = scaler.transform(X_val_d)
        X_te_sc  = scaler.transform(X_te_d)
        pca      = PCA(n_components=0.95, random_state=42)
        X_tr_fin  = pca.fit_transform(X_tr_sc)
        X_val_fin = pca.transform(X_val_sc)
        X_te_fin  = pca.transform(X_te_sc)
        pca_info  = {'n_comp': int(pca.n_components_),
                     'var_exp': float(pca.explained_variance_ratio_.sum())}
    else:
        X_tr_fin  = X_tr
        X_val_fin = X_val
        X_te_fin  = X_test
        pca_info  = {}

    KERNELS = ['linear', 'rbf', 'poly', 'sigmoid']
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=True)
        if k == 'poly':
            p['degree'] = 2
        return SVC(kernel=k, **p)

    def get_metrics(y_true, y_pred):
        return {
            'accuracy':  float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred,
                               average='binary', zero_division=0)),
            'recall':    float(recall_score(y_true, y_pred,
                               average='binary', zero_division=0)),
            'f1':        float(f1_score(y_true, y_pred,
                               average='binary', zero_division=0)),
            'cm':        confusion_matrix(y_true, y_pred),
            'cr':        classification_report(y_true, y_pred,
                               target_names=classes, output_dict=True),
        }

    kernel_results = {}
    for kernel in KERNELS:
        svm = build_svm(kernel)

        # ── 5-Fold CV pada data Training ──────────────────────────────────────
        cv_scores_acc = cross_val_score(svm, X_tr_fin, y_tr,
                                        cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores_f1  = cross_val_score(svm, X_tr_fin, y_tr,
                                        cv=cv, scoring='f1', n_jobs=-1)

        # ── Latih pada Training, evaluasi Train / Val / Test ──────────────────
        svm.fit(X_tr_fin, y_tr)

        y_pred_train = svm.predict(X_tr_fin)
        y_pred_val   = svm.predict(X_val_fin)
        y_pred_test  = svm.predict(X_te_fin)

        m_train = get_metrics(y_tr,   y_pred_train)
        m_val   = get_metrics(y_val,  y_pred_val)
        m_test  = get_metrics(y_test, y_pred_test)

        # ── Deteksi overfitting / underfitting ────────────────────────────────
        gap_train_val  = m_train['accuracy'] - m_val['accuracy']
        gap_val_test   = m_val['accuracy']   - m_test['accuracy']

        if gap_train_val > 0.10:
            overfit_status = "⚠️ Overfitting"
        elif m_train['accuracy'] < 0.70:
            overfit_status = "⚠️ Underfitting"
        else:
            overfit_status = "✅ Fit Baik"

        kernel_results[kernel] = {
            # Ukuran data
            'n_train':     len(y_tr),
            'n_val':       len(y_val),
            'n_test':      len(y_test),
            # CV
            'cv_acc_mean': float(cv_scores_acc.mean()),
            'cv_acc_std':  float(cv_scores_acc.std()),
            'cv_f1_mean':  float(cv_scores_f1.mean()),
            'cv_f1_std':   float(cv_scores_f1.std()),
            'cv_scores_f1': cv_scores_f1.tolist(),
            # Train
            'train': m_train,
            # Validation
            'val':   m_val,
            # Test (hold-out)
            'test':  m_test,
            # Gap
            'gap_train_val': gap_train_val,
            'gap_val_test':  gap_val_test,
            'overfit_status': overfit_status,
        }

    return {
        'kernel_results': kernel_results,
        'use_pca':        use_pca,
        'pca_info':       pca_info,
        'val_ratio':      val_ratio,
        'n_train_orig':   len(y_tr),
        'n_val':          int(len(y_val)),
        'n_test':         len(y_test),
        'classes':        classes,
        'split_label':    _feat.get('split_label', ''),
    }


# ── Poin 2: SMOTE vs Tanpa SMOTE — semua kernel × semua rasio ─────────────────
@st.cache_resource
def smote_vs_nosmote_full(df_hash: str, use_pca: bool, _df) -> dict:
    """
    Bandingkan performa SVM dengan SMOTE vs tanpa SMOTE
    untuk semua kernel × semua rasio split × 2 mode PCA.
    Menggunakan data split yang identik agar perbandingan adil.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 confusion_matrix)
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le       = LabelEncoder()
    y        = le.fit_transform(df['sentiment'])
    tfidf    = TfidfVectorizer(max_features=5000)
    X_tf     = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y)

    RATIOS  = [('70:30',0.7),('80:20',0.8),('90:10',0.9)]
    KERNELS = ['linear','rbf','poly','sigmoid']

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly': p['degree'] = 2
        return SVC(kernel=k, **p)

    def get_m(y_true, y_pred):
        return {
            'acc':  float(accuracy_score(y_true, y_pred)),
            'prec': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'rec':  float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1':   float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
            'cm':   confusion_matrix(y_true, y_pred),
        }

    results = {}
    for label, ratio in RATIOS:
        results[label] = {}
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y, train_size=ratio,
            test_size=round(1-ratio,10), random_state=42, stratify=y)

        # SMOTE
        try:
            sm = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        # PCA (opsional) — fit pada SMOTE, transform kedua set
        if use_pca:
            sc      = StandardScaler()
            X_sm_d  = X_tr_sm.toarray()
            X_nos_d = X_tr.toarray()
            X_te_d  = X_te.toarray()
            X_sm_sc  = sc.fit_transform(X_sm_d)
            X_te_sc  = sc.transform(X_te_d)
            # Scaler terpisah untuk no-SMOTE agar adil
            sc2     = StandardScaler()
            X_nos_sc = sc2.fit_transform(X_nos_d)
            X_te_sc2 = sc2.transform(X_te_d)

            pca1 = PCA(n_components=0.95, random_state=42)
            Xtr_sm_fin  = pca1.fit_transform(X_sm_sc)
            Xte_sm_fin  = pca1.transform(X_te_sc)

            pca2 = PCA(n_components=0.95, random_state=42)
            Xtr_nos_fin = pca2.fit_transform(X_nos_sc)
            Xte_nos_fin = pca2.transform(X_te_sc2)
        else:
            Xtr_sm_fin  = X_tr_sm;  Xte_sm_fin  = X_te
            Xtr_nos_fin = X_tr;     Xte_nos_fin = X_te

        for kernel in KERNELS:
            svm_sm  = build_svm(kernel)
            svm_nos = build_svm(kernel)
            svm_sm.fit(Xtr_sm_fin,  y_tr_sm)
            svm_nos.fit(Xtr_nos_fin, y_tr)

            m_sm  = get_m(y_te, svm_sm.predict(Xte_sm_fin))
            m_nos = get_m(y_te, svm_nos.predict(Xte_nos_fin))

            # Distribusi kelas training
            import numpy as np
            classes  = le.classes_
            cnt_sm   = {c: int((y_tr_sm==i).sum()) for i,c in enumerate(classes)}
            cnt_nos  = {c: int((y_tr==i).sum())    for i,c in enumerate(classes)}
            ir_sm    = min(cnt_sm.values())/max(cnt_sm.values())
            ir_nos   = min(cnt_nos.values())/max(cnt_nos.values())

            results[label][kernel] = {
                'smote':   m_sm,
                'nosmote': m_nos,
                'n_train_smote':   int(len(y_tr_sm)),
                'n_train_nosmote': int(len(y_tr)),
                'n_test':          int(len(y_te)),
                'ir_smote':   float(ir_sm),
                'ir_nosmote': float(ir_nos),
                'dist_smote':   cnt_sm,
                'dist_nosmote': cnt_nos,
            }

    return {
        'results': results,
        'use_pca': use_pca,
        'ratios':  [lbl for lbl,_ in RATIOS],
        'kernels': KERNELS,
        'classes': le.classes_,
    }


# ── Poin 4: Chi-Square vs Semua Fitur TF-IDF ──────────────────────────────────
@st.cache_resource
def chisq_vs_full_features(df_hash: str, use_pca: bool, _df) -> dict:
    """
    Bandingkan performa SVM menggunakan:
      A) 1.000 fitur terpilih Chi-Square
      B) 5.000 fitur penuh TF-IDF (tanpa seleksi)
    untuk semua kernel × semua rasio split.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)
    from imblearn.over_sampling import SMOTE
    import time as _time

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le      = LabelEncoder()
    y       = le.fit_transform(df['sentiment'])
    tfidf   = TfidfVectorizer(max_features=5000)
    X_full  = tfidf.fit_transform(df['final'])   # 5000 fitur penuh

    chi2_sel= SelectKBest(chi2, k=min(1000, X_full.shape[1]))
    X_chi2  = chi2_sel.fit_transform(X_full, y)  # 1000 fitur Chi-Square

    RATIOS  = [('70:30',0.7),('80:20',0.8),('90:10',0.9)]
    KERNELS = ['linear','rbf','poly','sigmoid']

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly': p['degree'] = 2
        return SVC(kernel=k, **p)

    def get_m(y_true, y_pred):
        return {
            'acc':  float(accuracy_score(y_true, y_pred)),
            'prec': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'rec':  float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1':   float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        }

    def prep_data(X_source, ratio):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_source, y, train_size=ratio,
            test_size=round(1-ratio,10), random_state=42, stratify=y)
        try:
            sm = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        if use_pca:
            sc  = StandardScaler()
            Xd  = X_tr_sm.toarray(); Xte = X_te.toarray()
            Xsc = sc.fit_transform(Xd); Xtesc = sc.transform(Xte)
            pca = PCA(n_components=0.95, random_state=42)
            Xf  = pca.fit_transform(Xsc); Xtef = pca.transform(Xtesc)
            n_dim = int(pca.n_components_)
        else:
            Xf = X_tr_sm; Xtef = X_te
            n_dim = X_source.shape[1]
        return Xf, Xtef, y_tr_sm, y_te, n_dim

    results = {}
    for label, ratio in RATIOS:
        results[label] = {}
        X_chi_tr, X_chi_te, y_chi_tr, y_te_chi, n_chi = prep_data(X_chi2, ratio)
        X_ful_tr, X_ful_te, y_ful_tr, y_te_ful, n_ful = prep_data(X_full, ratio)

        for kernel in KERNELS:
            # Chi-Square model
            svm_chi = build_svm(kernel)
            t0 = _time.time()
            svm_chi.fit(X_chi_tr, y_chi_tr)
            tt_chi = _time.time()-t0
            m_chi  = get_m(y_te_chi, svm_chi.predict(X_chi_te))
            m_chi['train_time'] = float(tt_chi)

            # Full TF-IDF model
            svm_ful = build_svm(kernel)
            t0 = _time.time()
            svm_ful.fit(X_ful_tr, y_ful_tr)
            tt_ful = _time.time()-t0
            m_ful  = get_m(y_te_ful, svm_ful.predict(X_ful_te))
            m_ful['train_time'] = float(tt_ful)

            results[label][kernel] = {
                'chi2':   m_chi, 'n_chi': n_chi,
                'full':   m_ful, 'n_full': n_ful,
                'n_test': int(len(y_te_chi)),
            }

    return {
        'results':     results,
        'use_pca':     use_pca,
        'ratios':      [lbl for lbl,_ in RATIOS],
        'kernels':     KERNELS,
        'n_chi_feat':  X_chi2.shape[1],
        'n_full_feat': X_full.shape[1],
    }


# ── Poin 6b: Perbandingan threshold PCA (semua kernel × semua rasio) ──────────
@st.cache_resource
def pca_threshold_compare(df_hash: str, _df) -> dict:
    """
    Uji semua threshold PCA [0.70,0.75,0.80,0.85,0.90,0.95,0.99]
    untuk semua kernel × semua rasio split.
    Kembalikan accuracy & F1 per kombinasi threshold × kernel × rasio.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    from imblearn.over_sampling import SMOTE
    import numpy as np

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le       = LabelEncoder()
    y        = le.fit_transform(df['sentiment'])
    tfidf    = TfidfVectorizer(max_features=5000)
    X_tf     = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y)

    THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    RATIOS     = [('70:30',0.7),('80:20',0.8),('90:10',0.9)]
    KERNELS    = ['linear','rbf','poly','sigmoid']

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly': p['degree'] = 2
        return SVC(kernel=k, **p)

    results = {t: {lbl: {} for lbl,_ in RATIOS} for t in THRESHOLDS}
    orig_dim = X_chi2.shape[1]

    for label, ratio in RATIOS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y, train_size=ratio,
            test_size=round(1-ratio,10), random_state=42, stratify=y)
        try:
            sm = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        # Scale sekali, pakai untuk semua threshold
        sc    = StandardScaler()
        X_d   = X_tr_sm.toarray()
        X_tde = X_te.toarray()
        X_sc  = sc.fit_transform(X_d)
        X_tsc = sc.transform(X_tde)

        for thresh in THRESHOLDS:
            pca = PCA(n_components=thresh, random_state=42)
            Xtr = pca.fit_transform(X_sc)
            Xte = pca.transform(X_tsc)
            n_c = int(pca.n_components_)
            var = float(pca.explained_variance_ratio_.sum())
            red = (1 - n_c/orig_dim)*100

            for kernel in KERNELS:
                svm = build_svm(kernel)
                svm.fit(Xtr, y_tr_sm)
                y_pred = svm.predict(Xte)
                results[thresh][label][kernel] = {
                    'acc': float(accuracy_score(y_te, y_pred)),
                    'f1':  float(f1_score(y_te, y_pred,
                                 average='binary', zero_division=0)),
                    'n_comp':  n_c,
                    'var_exp': var,
                    'red_pct': red,
                }

    return {
        'results':    results,
        'thresholds': THRESHOLDS,
        'ratios':     [lbl for lbl,_ in RATIOS],
        'kernels':    KERNELS,
        'orig_dim':   orig_dim,
    }


# ── Perbandingan Jumlah Fitur Chi-Square: 300, 500, 800, 1000 ─────────────────
@st.cache_resource
def chisq_k_compare(df_hash: str, use_pca: bool, _df) -> dict:
    """
    Uji performa SVM menggunakan berbagai jumlah fitur Chi-Square
    (k = 300, 500, 800, 1000) untuk semua kernel × semua rasio split.
    Berguna untuk membuktikan secara empiris jumlah fitur mana yang
    memberikan performa terbaik.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score)
    from imblearn.over_sampling import SMOTE
    import time as _time
    import numpy as np

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    from sklearn.preprocessing import LabelEncoder
    le      = LabelEncoder()
    y       = le.fit_transform(df['sentiment'])
    tfidf   = TfidfVectorizer(max_features=5000)
    X_tf    = tfidf.fit_transform(df['final'])

    K_VALUES = [300, 500, 800, 1000]
    RATIOS   = [('70:30',0.7),('80:20',0.8),('90:10',0.9)]
    KERNELS  = ['linear','rbf','poly','sigmoid']

    def build_svm(k):
        p = dict(C=1, gamma=0.001, random_state=42, probability=False)
        if k == 'poly': p['degree'] = 2
        return SVC(kernel=k, **p)

    def get_m(y_true, y_pred):
        return {
            'acc':  float(accuracy_score(y_true, y_pred)),
            'prec': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'rec':  float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1':   float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        }

    # Pra-hitung Chi-Square scores sekali (skor tidak berubah walau k berbeda)
    from sklearn.feature_selection import chi2 as chi2_func
    scores_all, _ = chi2_func(X_tf, y)
    feat_order    = np.argsort(scores_all)[::-1]  # index fitur terurut skor tertinggi

    results = {k: {lbl: {} for lbl, _ in RATIOS} for k in K_VALUES}

    for label, ratio in RATIOS:
        for k_val in K_VALUES:
            # Ambil k fitur teratas berdasarkan skor Chi-Square
            top_idx = feat_order[:k_val]
            X_k     = X_tf[:, top_idx]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_k, y, train_size=ratio,
                test_size=round(1-ratio,10), random_state=42, stratify=y)
            try:
                sm = SMOTE(random_state=42)
                X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
            except Exception:
                X_tr_sm, y_tr_sm = X_tr, y_tr

            if use_pca:
                sc     = StandardScaler()
                Xd     = X_tr_sm.toarray(); Xte_d = X_te.toarray()
                Xsc    = sc.fit_transform(Xd); Xte_sc = sc.transform(Xte_d)
                pca    = PCA(n_components=0.95, random_state=42)
                Xf     = pca.fit_transform(Xsc); Xtef = pca.transform(Xte_sc)
                n_comp = int(pca.n_components_)
                var_exp= float(pca.explained_variance_ratio_.sum())
            else:
                Xf = X_tr_sm; Xtef = X_te
                n_comp = k_val; var_exp = 1.0

            for kernel in KERNELS:
                svm = build_svm(kernel)
                t0 = _time.time()
                svm.fit(Xf, y_tr_sm)
                tt = _time.time() - t0
                y_pred = svm.predict(Xtef)
                m = get_m(y_te, y_pred)
                m['train_time'] = float(tt)
                m['n_comp']     = n_comp
                m['var_exp']    = var_exp
                results[k_val][label][kernel] = m

    return {
        'results':   results,
        'k_values':  K_VALUES,
        'ratios':    [lbl for lbl,_ in RATIOS],
        'kernels':   KERNELS,
        'use_pca':   use_pca,
        'n_total_tfidf': X_tf.shape[1],
    }


# ── Matriks Evaluasi Lengkap: Blueprint Tabel Skripsi (Data Asli) ─────────────
@st.cache_resource
def full_evaluation_matrix(df_hash: str, _df, use_chi2: bool = True) -> dict:
    """
    Menjalankan matriks evaluasi lengkap mengikuti blueprint tabel skripsi:
      - 3 rasio split : 90:10, 80:20, 70:30
      - 2 kondisi SMOTE: Sebelum SMOTE (data asli), Setelah SMOTE (oversampled)
      - 2 kondisi PCA  : Tanpa PCA, Dengan PCA (threshold 75%, 95%, 99%)
      - 4 kernel       : linear, rbf, poly, sigmoid
      - Grid parameter : C = [0.1, 1, 10], gamma = [0.001, 0.1, 1] (non-linear)
      - use_chi2       : True  -> fitur diseleksi Chi-Square (k=1000) sebelum
                                   digunakan (kondisi default penelitian ini)
                         False -> TANPA seleksi Chi-Square, langsung memakai
                                   seluruh fitur hasil TF-IDF (max_features=5000)

    Menghasilkan Accuracy, Precision, Recall, F1-Score menggunakan DATA ASLI
    dari model (bukan tabel kosong/template) — total 4 arsitektur × 4 kernel
    per rasio, dengan jumlah baris menyesuaikan grid parameter tiap kernel.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le    = LabelEncoder()
    y_all = le.fit_transform(df['sentiment'])
    tfidf = TfidfVectorizer(max_features=5000)
    X_tf  = tfidf.fit_transform(df['final'])

    if use_chi2:
        chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
        X_chi2   = chi2_sel.fit_transform(X_tf, y_all)
    else:
        # Tanpa seleksi Chi-Square: langsung pakai seluruh fitur TF-IDF
        X_chi2 = X_tf
    orig_dim = X_chi2.shape[1]

    RATIOS     = [('90:10', 0.9), ('80:20', 0.8), ('70:30', 0.7)]
    KERNELS    = ['linear', 'rbf', 'poly', 'sigmoid']
    C_VALS     = [0.1, 1, 10]
    GAMMA_VALS = [0.001, 0.1, 1]
    PCA_THRESH = [0.75, 0.95, 0.99]

    def build_svm(kernel, C, gamma):
        p = dict(kernel=kernel, C=C, gamma=gamma, random_state=42, probability=False)
        if kernel == 'poly':
            p['degree'] = 2
        return SVC(**p)

    def get_metrics(y_true, y_pred):
        return {
            'acc':  float(accuracy_score(y_true, y_pred)),
            'prec': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
            'rec':  float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
            'f1':   float(f1_score(y_true, y_pred, average='binary', zero_division=0)),
        }

    results = {}

    for ratio_label, ratio_val in RATIOS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y_all, train_size=ratio_val,
            test_size=round(1 - ratio_val, 10), random_state=42, stratify=y_all)

        X_tr_before, y_tr_before = X_tr, y_tr
        try:
            sm = SMOTE(random_state=42)
            X_tr_after, y_tr_after = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_after, y_tr_after = X_tr, y_tr

        results[ratio_label] = {}

        # ── Tanpa PCA: Sebelum & Setelah SMOTE ────────────────────────────────
        for X_tr_base, y_tr_base, arch_key in [
            (X_tr_before, y_tr_before, 'Tanpa PCA + Sebelum SMOTE'),
            (X_tr_after,  y_tr_after,  'Tanpa PCA + Setelah SMOTE'),
        ]:
            arch_rows = {k: [] for k in KERNELS}
            for kernel in KERNELS:
                gammas = [None] if kernel == 'linear' else GAMMA_VALS
                for C in C_VALS:
                    for g in gammas:
                        gamma_param = g if g is not None else 'scale'
                        svm = build_svm(kernel, C, gamma_param)
                        svm.fit(X_tr_base, y_tr_base)
                        y_pred = svm.predict(X_te)
                        m = get_metrics(y_te, y_pred)
                        arch_rows[kernel].append({'C': C, 'gamma': g, **m})
            results[ratio_label][arch_key] = {
                'rows': arch_rows,
                'n_train': int(len(y_tr_base)), 'n_test': int(len(y_te)),
            }

        # ── Dengan PCA: Sebelum & Setelah SMOTE, 3 threshold ─────────────────
        for X_tr_base, y_tr_base, arch_key in [
            (X_tr_before, y_tr_before, 'Dengan PCA + Sebelum SMOTE'),
            (X_tr_after,  y_tr_after,  'Dengan PCA + Setelah SMOTE'),
        ]:
            arch_rows = {k: [] for k in KERNELS}
            X_tr_dense = X_tr_base.toarray()
            X_te_dense = X_te.toarray()
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_dense)
            X_te_scaled = scaler.transform(X_te_dense)

            for thresh in PCA_THRESH:
                pca = PCA(n_components=thresh, random_state=42)
                X_tr_pca = pca.fit_transform(X_tr_scaled)
                X_te_pca = pca.transform(X_te_scaled)
                n_comp   = int(pca.n_components_)

                for kernel in KERNELS:
                    gammas = [None] if kernel == 'linear' else GAMMA_VALS
                    for C in C_VALS:
                        for g in gammas:
                            gamma_param = g if g is not None else 'scale'
                            svm = build_svm(kernel, C, gamma_param)
                            svm.fit(X_tr_pca, y_tr_base)
                            y_pred = svm.predict(X_te_pca)
                            m = get_metrics(y_te, y_pred)
                            arch_rows[kernel].append({
                                'pct': int(thresh * 100), 'n_comp': n_comp,
                                'C': C, 'gamma': g, **m
                            })
            results[ratio_label][arch_key] = {
                'rows': arch_rows,
                'n_train': int(len(y_tr_base)), 'n_test': int(len(y_te)),
            }

    return {
        'results':  results,
        'ratios':   [r for r, _ in RATIOS],
        'kernels':  KERNELS,
        'orig_dim': orig_dim,
        'use_chi2': use_chi2,
        'archs':    ['Tanpa PCA + Sebelum SMOTE', 'Tanpa PCA + Setelah SMOTE',
                     'Dengan PCA + Sebelum SMOTE', 'Dengan PCA + Setelah SMOTE'],
    }


# ── Uji Validasi Tambahan: 10-Fold Cross Validation (Model Optimal) ──────────
@st.cache_resource
def run_10fold_cv_validation(df_hash: str, kernel: str, C: float, gamma,
                              use_pca: bool, pca_threshold: float,
                              use_smote: bool, _df) -> dict:
    """
    Validasi tambahan menggunakan StratifiedKFold 10-fold pada model dengan
    konfigurasi optimal (kernel, C, gamma, PCA, SMOTE tertentu).

    Setiap fold: fit SMOTE (jika aktif) HANYA pada fold training, fit PCA
    (jika aktif) HANYA pada fold training lalu transform fold testing —
    menjaga agar tidak ada data leakage antar fold.

    Mengembalikan skor per fold (Accuracy, Precision, Recall, F1) beserta
    rata-rata, standar deviasi, dan status stabilitas model.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_score,
                                 recall_score, f1_score)
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le    = LabelEncoder()
    y_all = le.fit_transform(df['sentiment'])
    tfidf = TfidfVectorizer(max_features=5000)
    X_tf  = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y_all)

    def build_svm():
        p = dict(kernel=kernel, C=C, gamma=gamma, random_state=42, probability=False)
        if kernel == 'poly':
            p['degree'] = 2
        return SVC(**p)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_chi2, y_all), 1):
        X_tr, X_te = X_chi2[train_idx], X_chi2[test_idx]
        y_tr, y_te = y_all[train_idx], y_all[test_idx]

        if use_smote:
            try:
                sm = SMOTE(random_state=42)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
            except Exception:
                pass

        if use_pca:
            scaler = StandardScaler()
            X_tr_d = X_tr.toarray() if hasattr(X_tr, 'toarray') else X_tr
            X_te_d = X_te.toarray()
            X_tr_sc = scaler.fit_transform(X_tr_d)
            X_te_sc = scaler.transform(X_te_d)
            pca = PCA(n_components=pca_threshold, random_state=42)
            X_tr_fin = pca.fit_transform(X_tr_sc)
            X_te_fin = pca.transform(X_te_sc)
            n_comp = int(pca.n_components_)
        else:
            X_tr_fin = X_tr
            X_te_fin = X_te
            n_comp = X_chi2.shape[1]

        svm = build_svm()
        svm.fit(X_tr_fin, y_tr)
        y_pred = svm.predict(X_te_fin)

        fold_results.append({
            'fold':  fold_idx,
            'acc':   float(accuracy_score(y_te, y_pred)),
            'prec':  float(precision_score(y_te, y_pred, average='binary', zero_division=0)),
            'rec':   float(recall_score(y_te, y_pred, average='binary', zero_division=0)),
            'f1':    float(f1_score(y_te, y_pred, average='binary', zero_division=0)),
            'n_train': int(len(y_tr)), 'n_test': int(len(y_te)),
            'n_comp':  n_comp,
        })

    import numpy as np
    accs  = [r['acc']  for r in fold_results]
    precs = [r['prec'] for r in fold_results]
    recs  = [r['rec']  for r in fold_results]
    f1s   = [r['f1']   for r in fold_results]

    std_f1 = float(np.std(f1s))
    if std_f1 < 0.03:
        stability = "✅ Stabil"
    elif std_f1 < 0.06:
        stability = "⚠️ Cukup Stabil"
    else:
        stability = "❌ Tidak Stabil"

    return {
        'fold_results': fold_results,
        'mean_acc':  float(np.mean(accs)),  'std_acc':  float(np.std(accs)),
        'mean_prec': float(np.mean(precs)), 'std_prec': float(np.std(precs)),
        'mean_rec':  float(np.mean(recs)),  'std_rec':  float(np.std(recs)),
        'mean_f1':   float(np.mean(f1s)),   'std_f1':   std_f1,
        'stability': stability,
        'kernel': kernel, 'C': C, 'gamma': gamma,
        'use_pca': use_pca, 'pca_threshold': pca_threshold,
        'use_smote': use_smote,
        'n_folds': 10,
    }


# ── Distribusi SMOTE Nyata: Skenario 90:10, 80:20, 70:30 ──────────────────────
@st.cache_resource
def smote_distribution_multi_ratio(df_hash: str, _df) -> dict:
    """
    Menghitung distribusi kelas SEBELUM dan SESUDAH SMOTE secara nyata
    (bukan estimasi/skala) untuk 3 skenario rasio split: 90:10, 80:20, 70:30.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le    = LabelEncoder()
    y_all = le.fit_transform(df['sentiment'])
    classes = le.classes_

    tfidf = TfidfVectorizer(max_features=5000)
    X_tf  = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y_all)

    RATIOS = [('90:10', 0.9), ('80:20', 0.8), ('70:30', 0.7)]
    results = {}

    for label, ratio in RATIOS:
        X_train, X_test, y_train, y_test = train_test_split(
            X_chi2, y_all, train_size=ratio,
            test_size=round(1 - ratio, 10), random_state=42, stratify=y_all
        )

        counts_before = {classes[i]: int((y_train == i).sum()) for i in range(len(classes))}
        total_before  = sum(counts_before.values())
        minor_b = min(counts_before.values())
        major_b = max(counts_before.values())
        ir_before = minor_b / major_b

        try:
            sm = SMOTE(random_state=42)
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        except Exception:
            X_train_sm, y_train_sm = X_train, y_train

        counts_after = {classes[i]: int((y_train_sm == i).sum()) for i in range(len(classes))}
        total_after  = sum(counts_after.values())
        minor_a = min(counts_after.values())
        major_a = max(counts_after.values())
        ir_after = minor_a / major_a

        results[label] = {
            'counts_before': counts_before,
            'total_before':  total_before,
            'ir_before':     float(ir_before),
            'counts_after':  counts_after,
            'total_after':   total_after,
            'ir_after':      float(ir_after),
            'n_test':        int(len(y_test)),
            'data_sintetis': total_after - total_before,
        }

    return {
        'results': results,
        'ratios':  [lbl for lbl, _ in RATIOS],
        'classes': classes,
        'n_total': int(len(y_all)),
    }


# ── Reduksi Dimensi PCA pada Tiga Threshold Variansi (ringan, tanpa SVM) ──────
@st.cache_resource
def pca_reduction_by_threshold(df_hash: str, _df) -> dict:
    """
    Menghitung hasil reduksi dimensi PCA (jumlah komponen, variansi kumulatif,
    persentase reduksi) untuk TIGA threshold variansi (75%, 95%, 99%) pada
    KETIGA rasio split data (70:30, 80:20, 90:10).

    Fungsi ini HANYA menghitung PCA (tanpa melatih SVM), sehingga jauh lebih
    ringan dan cepat dibandingkan pca_threshold_compare(), cocok untuk
    menghasilkan tabel ringkasan "Hasil Reduksi Dimensi dengan PCA pada Tiga
    Threshold Variansi" secara langsung.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from imblearn.over_sampling import SMOTE

    load_nltk(); sw = get_stop_words()
    df = _df.copy()
    df['clean']  = df['content'].apply(preprocess_text)
    df['tokens'] = df['clean'].apply(lambda t: tokenize_remove_stopwords(t, sw))
    df['tokens'] = df['tokens'].apply(lemmatize_tokens)
    df['final']  = df['tokens'].apply(lambda t: ' '.join(t))

    le       = LabelEncoder()
    y        = le.fit_transform(df['sentiment'])
    tfidf    = TfidfVectorizer(max_features=5000)
    X_tf     = tfidf.fit_transform(df['final'])
    chi2_sel = SelectKBest(chi2, k=min(1000, X_tf.shape[1]))
    X_chi2   = chi2_sel.fit_transform(X_tf, y)
    orig_dim = X_chi2.shape[1]

    THRESHOLDS = [0.75, 0.95, 0.99]
    RATIOS     = [('70:30', 0.7), ('80:20', 0.8), ('90:10', 0.9)]

    rows = []
    for label, ratio in RATIOS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_chi2, y, train_size=ratio,
            test_size=round(1 - ratio, 10), random_state=42, stratify=y)
        try:
            sm = SMOTE(random_state=42)
            X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        except Exception:
            X_tr_sm, y_tr_sm = X_tr, y_tr

        sc      = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr_sm.toarray())

        for thresh in THRESHOLDS:
            pca = PCA(n_components=thresh, random_state=42)
            pca.fit(X_tr_sc)
            n_comp  = int(pca.n_components_)
            var_exp = float(pca.explained_variance_ratio_.sum())
            red_pct = (1 - n_comp / orig_dim) * 100

            rows.append({
                'Rasio Split':          label,
                'Threshold':            f"{int(thresh*100)}%",
                'Data Training (SMOTE)': int(len(y_tr_sm)),
                'Data Testing':         int(len(y_te)),
                'Dimensi Awal':         orig_dim,
                'Komponen PCA (k)':     n_comp,
                'Fitur Direduksi':      orig_dim - n_comp,
                'Reduksi (%)':          round(red_pct, 2),
                'Variansi Kumulatif (%)': round(var_exp * 100, 2),
            })

    return {
        'rows':       rows,
        'thresholds': THRESHOLDS,
        'ratios':     [lbl for lbl, _ in RATIOS],
        'orig_dim':   orig_dim,
    }


# ── Hasil Prediksi 10 Dokumen — Rasio 80:20, Semua Kernel & Parameter ────────
@st.cache_resource
def svm_prediction_detail_8020(df_hash: str, _df) -> dict:
    """
    Melatih SEMUA 30 kombinasi kernel-parameter (Linear x3, RBF/Poly/Sigmoid x9
    masing-masing) pada rasio split 80:20, DENGAN PCA SETELAH SMOTE (StandardScaler
    → PCA threshold 95% diterapkan pada data training hasil SMOTE, lalu ditransformasikan
    ke data testing), lalu mencatat hasil prediksi terhadap 10 dokumen uji pertama
    (menggunakan clean text hasil preprocessing sebagai referensi tampilan), beserta
    metrik Accuracy & F1-Score keseluruhan tiap kombinasi. Strukturnya mengikuti pola
    per-kernel seperti pada Matriks Evaluasi Lengkap.
    """
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    feat = prepare_features(df_hash, 0.8, _df, 1000)

    X_train_sm = feat['X_train_sm']
    y_train_sm = feat['y_train_sm']
    X_test     = feat['X_test']
    y_test     = feat['y_test']
    test_idx   = feat['test_idx']
    classes    = feat['classes']
    orig_dim   = feat['n_features']

    # ── Dengan PCA + Setelah SMOTE: StandardScaler → PCA (threshold 95%) ────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm.toarray())
    X_test_scaled  = scaler.transform(X_test.toarray())

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    n_comp = int(pca.n_components_)
    var_exp = float(pca.explained_variance_ratio_.sum())

    n_sample = min(10, X_test_pca.shape[0])
    X_test_10 = X_test_pca[:n_sample]
    y_test_10 = y_test[:n_sample]

    # Teks clean (hasil preprocessing) untuk 10 dokumen uji pertama
    df_final = feat['df_final']
    clean_texts_10 = df_final.iloc[test_idx[:n_sample]]['final'].tolist()
    raw_texts_10   = df_final.iloc[test_idx[:n_sample]]['content'].tolist()

    C_LIST     = [0.1, 1, 10]
    GAMMA_LIST = [0.001, 0.1, 1]
    KERNELS    = ['linear', 'rbf', 'poly', 'sigmoid']

    kernel_results = {}
    for kernel in KERNELS:
        rows = []
        if kernel == 'linear':
            param_combos = [(C, None) for C in C_LIST]
        else:
            param_combos = [(C, g) for C in C_LIST for g in GAMMA_LIST]

        for C, gamma in param_combos:
            kwargs = dict(kernel=kernel, C=C, random_state=42)
            if gamma is not None:
                kwargs['gamma'] = gamma
            if kernel == 'poly':
                kwargs['degree'] = 2

            svm = SVC(**kwargs)
            svm.fit(X_train_pca, y_train_sm)

            y_pred_all = svm.predict(X_test_pca)
            y_pred_10  = svm.predict(X_test_10)

            acc  = accuracy_score(y_test, y_pred_all)
            prec = precision_score(y_test, y_pred_all, average='binary', zero_division=0)
            rec  = recall_score(y_test, y_pred_all, average='binary', zero_division=0)
            f1   = f1_score(y_test, y_pred_all, average='binary', zero_division=0)

            rows.append({
                'C': C, 'gamma': gamma,
                'pred_10': [classes[p] for p in y_pred_10],
                'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
            })
        kernel_results[kernel] = rows

    return {
        'kernel_results': kernel_results,
        'clean_texts_10': clean_texts_10,
        'raw_texts_10':   raw_texts_10,
        'actual_10':      [classes[y] for y in y_test_10],
        'classes':        classes,
        'n_train':        int(X_train_pca.shape[0]),
        'n_test':         int(X_test_pca.shape[0]),
        'orig_dim':       orig_dim,
        'n_comp':         n_comp,
        'var_exp':        var_exp,
        'scenario':       'Dengan PCA + Setelah SMOTE',
    }
