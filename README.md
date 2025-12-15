# Aplikasi Deteksi Kemiripan Judul Tugas Akhir

[![GitHub Repository](https://img.shields.io/badge/GitHub-MLOps--RC--Kemiripan--Judul--TA-blue?logo=github)](https://github.com/JeremiaSusanto/MLOps-RC-Kemiripan-Judul-TA)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://github.com/JeremiaSusanto/MLOps-RC-Kemiripan-Judul-TA)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)

Aplikasi berbasis Streamlit untuk deteksi kemiripan judul tugas akhir menggunakan **Random Forest** dengan dual model system (Full Model + Lightweight Model) dan 3 fitur similarity terbaik.

## 🚀 Quick Start

### 1. Instalasi Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Training Model

**PENTING:** Sebelum menjalankan aplikasi, Anda harus melatih model terlebih dahulu!

#### Opsi A: Lightweight Model (Recommended untuk GitHub)

```powershell
python train_model_lightweight.py
```

Model akan berukuran sangat kecil (~44 KB) dan sudah ter-commit di repository. **Cocok untuk deployment GitHub/Cloud.**

#### Opsi B: Full Model (TF-IDF Features Lebih Banyak)

```powershell
python train_model.py
```

Model berukuran ~114 KB dengan TF-IDF max_features=1000. **Cocok untuk development/production lokal.**

Script akan:

- Memuat dataset dari `data/dataset_TA.csv`
- Melakukan preprocessing dengan Sastrawi (stemming Bahasa Indonesia)
- Membuat pasangan berlabel secara otomatis berdasarkan cosine similarity
- Ekstraksi **3 fitur similarity terbaik** untuk setiap pasangan (Cosine, Jaccard, Levenshtein)
- Training model **Random Forest** dengan GridSearchCV
- Menyimpan model dan artefak ke folder `model_outputs/`

Output training:

- `model_outputs/tfidf.joblib` - TF-IDF vectorizer (max_features=1000)
- `model_outputs/scaler.joblib` - StandardScaler untuk 3 features
- `model_outputs/best_rf.joblib` - Trained Random Forest model
- `model_outputs/titles_preprocessed.csv` - Preprocessed corpus

Output lightweight:

- `model_outputs_lightweight/tfidf.joblib` - TF-IDF compressed (max_features=500)
- `model_outputs_lightweight/scaler.joblib` - StandardScaler compressed
- `model_outputs_lightweight/best_rf.joblib` - Random Forest compressed
- `model_outputs_lightweight/titles_preprocessed.csv` - Preprocessed corpus

### 3. Menjalankan Aplikasi Streamlit

```powershell
streamlit run main.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Struktur Folder

```
mlops-kemiripan-judul/
├── data/
│   └── dataset_TA.csv                 # Dataset 123 judul TA
├── model_outputs/                     # Full Model (~114 KB)
│   ├── tfidf.joblib
│   ├── scaler.joblib
│   ├── best_rf.joblib
│   └── titles_preprocessed.csv
├── model_outputs_lightweight/         # Lightweight Model (~44 KB)
│   ├── tfidf.joblib
│   ├── scaler.joblib
│   ├── best_rf.joblib
│   └── titles_preprocessed.csv
├── preprocessing.py                   # Modul preprocessing Sastrawi
├── train_model.py                     # Training full model
├── train_model_lightweight.py         # Training lightweight model
├── main.py                            # Aplikasi Streamlit (dual model)
├── requirements.txt                   # Dependencies Python
├── README.md                          # Dokumentasi
├── GITHUB_DEPLOYMENT.md               # Panduan deployment
└── kesimpulan.md                      # MLOps lifecycle documentation
```

## 🧠 Fitur Aplikasi

- ✅ **Preprocessing Teks**: Sastrawi stemmer untuk Bahasa Indonesia
- ✅ **Ekstraksi 3 Fitur Terbaik**: Cosine Similarity, Jaccard Similarity, Levenshtein Ratio
- ✅ **Dual Model System**: Full Model (1000 features) vs Lightweight Model (500 features)
- ✅ **Model Comparison**: 3 tab interface untuk membandingkan hasil kedua model
- ✅ **Random Forest**: Hyperparameter tuning dengan GridSearchCV
- ✅ **Visualisasi**: Plotly bar charts interaktif dengan monochrome theme
- ✅ **Export**: Download hasil ke CSV untuk setiap model
- ✅ **Professional UI**: Color palette monokrom (#2c3e50) dengan gradients

## 📊 Dataset

Dataset `dataset_TA.csv` berisi 123 judul tugas akhir dalam Bahasa Indonesia dengan topik beragam:

- Machine Learning & Deep Learning
- Sentiment Analysis
- Forecasting (SARIMA, LSTM, ARFIMA)
- Object Detection (YOLO, CNN)
- Classification & Prediction

## 👥 Tim Penyusun

- **Jaclin Alcavella** (122450015)
- **Jeremia Susanto** (122450022) - [GitHub](https://github.com/JeremiaSusanto)
- **Muhammad Zaky Zaiddan** (122450119)
- **Vira Putri Maharani** (122450129)

**Tugas Besar Projek Sains Data**  
Institut Teknologi Sumatera - 2025

---

## 🌐 Deployment ke GitHub

### Push ke GitHub Repository

```powershell
# 1. Initialize git (jika belum)
git init

# 2. Add semua file
git add .

# 3. Commit dengan pesan
git commit -m "Initial commit: Deteksi Kemiripan Judul TA"

# 4. Add remote repository
git branch -M main
git remote add origin https://github.com/JeremiaSusanto/MLOps-RC-Kemiripan-Judul-TA.git

# 5. Push ke GitHub
git push -u origin main
```

### Clone dan Setup (User Baru)

**Untuk Lightweight Model (sudah ada di repo):**

```powershell
git clone https://github.com/JeremiaSusanto/MLOps-RC-Kemiripan-Judul-TA.git
cd MLOps-RC-Kemiripan-Judul-TA
pip install -r requirements.txt
streamlit run main.py  # Model sudah siap!
```

**Untuk Full Model (training ulang):**

```powershell
git clone https://github.com/JeremiaSusanto/MLOps-RC-Kemiripan-Judul-TA.git
cd MLOps-RC-Kemiripan-Judul-TA
pip install -r requirements.txt
python train_model.py  # Training dengan akurasi maksimal
streamlit run main.py
```

### ☁️ Deployment ke Streamlit Cloud

1. Push repository ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Login dengan GitHub
4. Pilih repository Anda
5. Klik "Deploy"!

**Catatan:** Lightweight model sudah ter-commit, jadi aplikasi langsung bisa jalan di Streamlit Cloud tanpa perlu training ulang.

---

## 🔧 Troubleshooting

A: Jalankan `python train_model_lightweight.py` atau `python train_model.py` untuk generate model.

**Q: Aplikasi error di Streamlit Cloud?**  
A: Pastikan `model_outputs_lightweight/` ada di repository. Model ini sudah ter-commit (~44 KB).

**Q: Perbedaan Full Model vs Lightweight?**  
A: Full Model: TF-IDF 1000 features, ~114 KB. Lightweight: 500 features, ~44 KB (2.5x lebih kecil).

**Q: Mengapa Full Model kadang 0 semua?**  
A: Dataset sangat imbalanced. Threshold auto-labeling perlu disesuaikan (0.45 untuk Full, 0.5 untuk Lightweight).

**Q: Cara melihat perbandingan kedua model?**  
A: Aplikasi otomatis load kedua model jika tersedia, dengan 3 tab: Full Model, Lightweight Model, dan Perbandingan
**Q: Ingin akurasi lebih tinggi?**
A: Gunakan `python train_model.py` untuk full model dengan hyperparameter lebih kompleks.
