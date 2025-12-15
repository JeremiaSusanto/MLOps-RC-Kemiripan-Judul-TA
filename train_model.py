"""
Script training model untuk deteksi kemiripan judul tugas akhir.
Membaca dataset_TA.csv, membuat pasangan berlabel, ekstraksi fitur,
training SVM dan Random Forest, dan menyimpan model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from Levenshtein import ratio as levenshtein_ratio
from preprocessing import preprocess_text

print("=" * 80)
print("TRAINING MODEL DETEKSI KEMIRIPAN JUDUL TUGAS AKHIR")
print("=" * 80)

# === 1. Load Dataset ===
print("\n[1/8] Loading dataset...")
df = pd.read_csv("data/dataset_TA.csv")
print(f"Dataset loaded: {len(df)} judul")

# Preprocessing
print("\n[2/8] Preprocessing teks (Sastrawi stemming)...")
df['judul_proc'] = df['Judul'].apply(preprocess_text)
df = df[df['judul_proc'].str.len() > 0].reset_index(drop=True)  # hapus judul kosong setelah preprocessing
print(f"Setelah preprocessing: {len(df)} judul valid")

# === 2. TF-IDF Vectorization ===
print("\n[3/8] Membuat representasi TF-IDF (unigram + bigram)...")
# Gunakan max_features untuk konsistensi dengan lightweight model
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, max_features=1000)
X_tfidf = tfidf.fit_transform(df['judul_proc'])
print(f"TF-IDF shape: {X_tfidf.shape}")

# === 3. Membuat Pasangan Berlabel (Auto-labeling) ===
print("\n[4/8] Membuat pasangan berlabel berdasarkan cosine similarity...")


def create_pairs_auto(top_thresh=0.6, low_thresh=0.25, max_pairs_per_doc=3):
    """
    Auto-labeling: pasangan dengan cosine >= top_thresh = mirip (1),
    pasangan dengan cosine <= low_thresh = tidak mirip (0)
    """
    sims = cosine_similarity(X_tfidf)
    pairs = []
    n = len(df)
    
    for i in range(n):
        # Positive pairs (mirip)
        pos_idxs = np.where((sims[i] >= top_thresh) & (np.arange(n) != i))[0]
        for j in pos_idxs[:max_pairs_per_doc]:
            pairs.append((i, int(j), 1, float(sims[i, j])))
        
        # Negative pairs (tidak mirip)
        neg_idxs = np.where((sims[i] <= low_thresh) & (np.arange(n) != i))[0]
        if len(neg_idxs) > 0:
            neg_samples = min(max_pairs_per_doc, len(neg_idxs))
            for j in np.random.choice(neg_idxs, neg_samples, replace=False):
                pairs.append((i, int(j), 0, float(sims[i, j])))
    
    pairs_df = pd.DataFrame(pairs, columns=['i', 'j', 'label', 'cosine'])
    return pairs_df


pairs_df = create_pairs_auto(top_thresh=0.45, low_thresh=0.25, max_pairs_per_doc=5)
print(f"Total pasangan: {len(pairs_df)} (Mirip: {(pairs_df['label']==1).sum()}, Tidak mirip: {(pairs_df['label']==0).sum()})")

# === 4. Ekstraksi Fitur Pasangan ===
print("\n[5/8] Ekstraksi fitur untuk setiap pasangan...")


def jaccard_sim(a_text, b_text):
    """Jaccard similarity pada token set"""
    a = set(a_text.split())
    b = set(b_text.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def pair_features(i, j):
    """Ekstraksi 3 fitur terbaik untuk pasangan (i, j)"""
    vi = X_tfidf[i].toarray().flatten()
    vj = X_tfidf[j].toarray().flatten()
    
    # 1. Cosine similarity (most important for text similarity)
    cos = float(cosine_similarity([vi], [vj])[0, 0])
    
    # 2. Jaccard similarity (token overlap)
    jacc = jaccard_sim(df.loc[i, 'judul_proc'], df.loc[j, 'judul_proc'])
    
    # 3. Levenshtein similarity (edit distance)
    lev = levenshtein_ratio(df.loc[i, 'Judul'], df.loc[j, 'Judul'])
    
    return [cos, jacc, lev]


X_list = []
y_list = []
for _, row in pairs_df.iterrows():
    X_list.append(pair_features(int(row['i']), int(row['j'])))
    y_list.append(row['label'])

X = np.array(X_list)
y = np.array(y_list)
print(f"Feature matrix shape: {X.shape}")

# === 5. Split & Scaling ===
print("\n[6/8] Split data (80% train, 20% test) dan scaling...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# === 6. Training & Hyperparameter Tuning ===
print("\n[7/8] Training Random Forest model dengan GridSearchCV...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid_rf,
    scoring='f1',
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_rf.fit(X_train_s, y_train)
best_rf = grid_rf.best_estimator_
print(f"Best RF params: {grid_rf.best_params_}")

# Evaluasi RF
y_pred_rf = best_rf.predict(X_test_s)
y_proba_rf = best_rf.predict_proba(X_test_s)[:, 1]
print("\nRandom Forest Test Results:")
print(classification_report(y_test, y_pred_rf, target_names=['Tidak Mirip', 'Mirip']))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")

# === 7. Simpan Model & Artefak ===
print("\n[8/8] Menyimpan model dan artefak...")
os.makedirs("model_outputs", exist_ok=True)

joblib.dump(tfidf, "model_outputs/tfidf.joblib")
joblib.dump(scaler, "model_outputs/scaler.joblib")
joblib.dump(best_rf, "model_outputs/best_rf.joblib")
df.to_csv("model_outputs/titles_preprocessed.csv", index=False)

print("\n✅ Model dan artefak berhasil disimpan ke folder 'model_outputs':")
print("  - tfidf.joblib")
print("  - scaler.joblib")
print("  - best_rf.joblib")
print("  - titles_preprocessed.csv")

print("\n" + "=" * 80)
print("TRAINING SELESAI! Model siap digunakan di aplikasi Streamlit.")
print("=" * 80)
