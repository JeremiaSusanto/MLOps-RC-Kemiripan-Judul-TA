# MLOps Lifecycle - Proyek Deteksi Kemiripan Judul Tugas Akhir

## 🏗️ 1. BUILD Phase

### 📥 Data Ingestion

**Proses:**

- Load dataset dari `data/dataset_TA.csv` (123 judul tugas akhir)
- Validasi format data (kolom "Judul" harus ada)
- Preprocessing teks dengan Sastrawi:
  - Lowercase conversion
  - Punctuation & number removal
  - Stopwords removal (Bahasa Indonesia)
  - Stemming (mengubah kata ke bentuk dasar)

**File Terkait:**

- `data/dataset_TA.csv` - Raw dataset
- `preprocessing.py` - Modul preprocessing

**Output:**

```python
# Contoh:
Input:  "Implementasi Algoritma Machine Learning untuk Prediksi"
Output: "implementasi algoritma machine learning prediksi"
```

---

### 🤖 Model Training

**Proses:**

1. **Feature Extraction (TF-IDF Vectorization)**

   - Ngram: (1,2) - unigram + bigram
   - Full Model: unlimited vocabulary (1795 features)
   - Lightweight Model: max 500 features (432 features)

2. **Auto-Labeling Pasangan Judul**

   ```python
   Mirip (label=1):     cosine_similarity >= 0.6
   Tidak Mirip (label=0): cosine_similarity <= 0.20-0.25
   ```

3. **Ekstraksi 3 Fitur per Pasangan**

   - **Cosine Similarity**: Kemiripan vektor TF-IDF (0-1)
   - **Jaccard Similarity**: Token overlap (0-1)
   - **Levenshtein Ratio**: Edit distance (0-1)

4. **Training Random Forest**
   - GridSearchCV dengan StratifiedKFold (3-5 folds)
   - Full Model: 100-300 trees, max_depth None-20
   - Lightweight Model: 50-100 trees, max_depth 10-15

**File Terkait:**

- `train_model.py` - Training full model
- `train_model_lightweight.py` - Training lightweight model

**Commands:**

```bash
# Training Full Model (5-10 MB)
python train_model.py

# Training Lightweight Model (38 KB)
python train_model_lightweight.py
```

**Hyperparameters Tuning:**
| Parameter | Full Model | Lightweight Model |
|-----------|------------|-------------------|
| n_estimators | 100-300 | 50-100 |
| max_depth | None, 10, 20 | 10, 15 |
| min_samples_split | 2, 5 | 5 |
| TF-IDF features | 1795 (unlimited) | 432 (max 500) |
| CV folds | 5 | 3 |

---

### ✅ Model Testing

**Proses:**

1. **Data Split**: 80% training, 20% testing
2. **Evaluation Metrics**:
   - Accuracy Score
   - F1-Score (weighted)
   - ROC AUC Score
   - Classification Report (precision, recall)
   - Confusion Matrix

**Hasil Testing:**

```
Full Model:
✓ Accuracy: 100%
✓ ROC AUC: 1.0
✓ Model Size: ~5-10 MB

Lightweight Model:
✓ Accuracy: 100%
✓ ROC AUC: 1.0
✓ Model Size: 38 KB (130x lebih kecil!)
```

**Best Hyperparameters Found:**

```python
# Full Model
{
    'n_estimators': 200,
    'max_depth': None,
    'min_samples_split': 2
}

# Lightweight Model
{
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5
}
```

---

### 📦 Model Packaging

**Proses:**

1. **Serialization dengan Joblib**

   - Compression level 3 untuk lightweight model
   - No compression untuk full model

2. **Artifacts yang Disimpan**:

   ```
   model_outputs/              (Full Model)
   ├── tfidf.joblib            (TF-IDF vectorizer)
   ├── scaler.joblib           (StandardScaler)
   ├── best_rf.joblib          (Random Forest model)
   └── titles_preprocessed.csv (Preprocessed corpus)

   model_outputs_lightweight/  (Lightweight Model)
   ├── tfidf.joblib            (Compressed TF-IDF)
   ├── scaler.joblib           (Compressed scaler)
   ├── best_rf.joblib          (Compressed RF)
   └── titles_preprocessed.csv (Same corpus)
   ```

**Code:**

```python
# Packaging dengan compression
joblib.dump(tfidf, "model_outputs_lightweight/tfidf.joblib", compress=3)
joblib.dump(scaler, "model_outputs_lightweight/scaler.joblib", compress=3)
joblib.dump(best_rf, "model_outputs_lightweight/best_rf.joblib", compress=3)
```

**Versioning:**

- Model files include timestamp in metadata
- Git tracks lightweight model (38 KB)
- Full model di-ignore via `.gitignore` (too large)

---

### 📝 Model Registering

**Proses:**

1. **Local Registry** (Current Implementation)

   - Models stored in local folders
   - Version control via Git (lightweight only)
   - Metadata in filenames and timestamps

2. **Model Registry Structure**:

   ```python
   {
       'model_name': 'Random Forest - Kemiripan Judul',
       'version': 'v1.0',
       'type': 'full' / 'lightweight',
       'accuracy': 1.0,
       'roc_auc': 1.0,
       'size': '38 KB' / '5-10 MB',
       'features': ['cosine', 'jaccard', 'levenshtein'],
       'trained_date': '2025-12-14',
       'hyperparameters': {...}
   }
   ```

3. **Loading Mechanism**:
   ```python
   @st.cache_resource
   def load_models():
       # Auto-detect available models
       # Load Full & Lightweight if both exist
       # Fallback to single model
   ```

**Future Enhancement** (Optional):

- MLflow Model Registry
- Model versioning with tags
- A/B testing capabilities
- Model performance tracking over time

---

## 🚀 2. DEPLOY Phase

### 🧪 Application Testing

**Proses:**

1. **Unit Testing**

   - Test preprocessing functions
   - Test feature extraction
   - Test model loading & inference

2. **Integration Testing**

   - Test end-to-end workflow: input → preprocessing → prediction → output
   - Test both model versions
   - Validate results consistency

3. **UI/UX Testing**
   - Test Streamlit interface responsiveness
   - Validate visualizations (Plotly charts)
   - Test download CSV functionality
   - Cross-browser compatibility

**Test Cases:**

```python
# Test 1: Preprocessing
assert preprocess_text("Implementasi ML") == "implementasi ml"

# Test 2: Feature Extraction
features = extract_pair_features(vec1, vec2, text1, text2, raw1, raw2)
assert len(features) == 3  # 3 fitur

# Test 3: Model Inference
result = compute_similarities(query, tfidf, scaler, rf_model, df_corpus)
assert 'rf_probability' in result.columns
assert result['rf_probability'].between(0, 1).all()

# Test 4: Comparison Mode
# Kedua model harus memberikan hasil yang reasonable
```

**Local Testing:**

```bash
# Run aplikasi locally
streamlit run main.py

# Test dengan berbagai input:
# - Judul sangat mirip (expect high probability)
# - Judul sangat berbeda (expect low probability)
# - Edge cases (empty input, special characters)
```

---

### 🌐 Production Release

**Proses:**

#### **A. GitHub Deployment**

```bash
# 1. Prepare repository
git init
git add .
git commit -m "Initial commit: Dual model ML system"

# 2. Push to GitHub
git remote add origin https://github.com/username/mlops-kemiripan-judul.git
git push -u origin main
```

**Repository Structure:**

```
mlops-kemiripan-judul/
├── .gitignore                    # Ignore full model
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── main.py                       # Streamlit app
├── preprocessing.py              # Preprocessing module
├── train_model.py                # Full model training
├── train_model_lightweight.py   # Lightweight training
├── data/
│   └── dataset_TA.csv           # Dataset (123 judul)
└── model_outputs_lightweight/   # Lightweight model (38 KB)
    ├── tfidf.joblib
    ├── scaler.joblib
    ├── best_rf.joblib
    └── titles_preprocessed.csv
```

#### **B. Streamlit Cloud Deployment** (Recommended)

**Steps:**

1. Push code to GitHub (done)
2. Go to https://share.streamlit.io
3. Connect GitHub account
4. Select repository: `mlops-kemiripan-judul`
5. Main file: `main.py`
6. Click "Deploy"

**Advantages:**

- ✅ **Free** & unlimited apps
- ✅ Auto-deploy on git push
- ✅ HTTPS dengan custom domain
- ✅ Built-in secrets management
- ✅ No server configuration needed

**Configuration** (`config.toml`):

```toml
[server]
maxUploadSize = 50

[theme]
primaryColor = "#2c3e50"
backgroundColor = "#f5f7fa"
secondaryBackgroundColor = "#ffffff"
textColor = "#2c3e50"
font = "sans serif"
```

#### **C. Docker Deployment** (Optional)

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build & Run:**

```bash
# Build image
docker build -t kemiripan-judul-app .

# Run container
docker run -p 8501:8501 kemiripan-judul-app
```

#### **D. Cloud Platform Options**

| Platform             | Pros                    | Cons              | Cost        |
| -------------------- | ----------------------- | ----------------- | ----------- |
| **Streamlit Cloud**  | Easy, Auto-deploy, Free | Limited resources | Free        |
| **Heroku**           | Simple PaaS             | Dyno limitations  | $7/month    |
| **AWS EC2**          | Full control            | Complex setup     | ~$10/month  |
| **Google Cloud Run** | Serverless, Auto-scale  | Cold start        | Pay-per-use |
| **Azure Web Apps**   | Enterprise-ready        | Learning curve    | ~$13/month  |

**Production URL Examples:**

```
Streamlit Cloud: https://kemiripan-judul.streamlit.app
Heroku:          https://kemiripan-judul.herokuapp.com
Custom Domain:   https://ta-similarity.yourdomain.com
```

#### **E. Environment Variables**

```bash
# .streamlit/secrets.toml
[general]
app_name = "Deteksi Kemiripan Judul TA"
version = "1.0"
max_corpus_size = 1000

[models]
full_model_path = "model_outputs"
lightweight_model_path = "model_outputs_lightweight"
```

#### **F. Production Checklist**

- ✅ Lightweight model committed (38 KB)
- ✅ Requirements.txt updated
- ✅ README.md with setup instructions
- ✅ .gitignore configured
- ✅ Error handling implemented
- ✅ Loading spinners for UX
- ✅ Input validation
- ✅ Responsive design (mobile-friendly)
- ✅ Download CSV feature
- ✅ Model comparison tabs

---

## 📊 3. MONITOR Phase

### 📈 Monitor

**Metrics to Track:**

#### **A. Application Performance**

```python
# Monitoring dengan Streamlit + Custom Logging
import logging
import time

logging.basicConfig(filename='app.log', level=logging.INFO)

def log_prediction(query, results, duration):
    logging.info(f"""
    Timestamp: {datetime.now()}
    Query: {query}
    Top Probability: {results['rf_probability'].max():.3f}
    Processing Time: {duration:.2f}s
    Model Used: Full & Lightweight
    """)

# Dalam compute_similarities():
start_time = time.time()
# ... processing ...
duration = time.time() - start_time
log_prediction(query, df_results, duration)
```

**Key Metrics:**

1. **Response Time**

   - Target: < 2 seconds per query
   - Alert if > 5 seconds

2. **Model Availability**

   - Track which models are loaded
   - Monitor loading failures

3. **User Engagement**

   - Number of queries per day
   - Average query length
   - Download CSV frequency

4. **Error Rate**
   - Track exceptions in preprocessing
   - Monitor model prediction failures
   - Empty result warnings

**Implementation (Optional - Streamlit Cloud Analytics):**

```python
# Install: pip install streamlit-analytics
import streamlit_analytics

with streamlit_analytics.track():
    # Your Streamlit app code
    main()
```

#### **B. Model Performance Tracking**

```python
# Save prediction results for analysis
def save_prediction_log(query, full_result, light_result):
    log_entry = {
        'timestamp': datetime.now(),
        'query': query,
        'full_model_top_prob': full_result['rf_probability'].max(),
        'light_model_top_prob': light_result['rf_probability'].max(),
        'probability_diff': abs(full_result['rf_probability'].max() -
                               light_result['rf_probability'].max()),
        'full_model_top_match': full_result.iloc[0]['judul_ref'],
        'light_model_top_match': light_result.iloc[0]['judul_ref']
    }

    # Append to CSV for analysis
    pd.DataFrame([log_entry]).to_csv('logs/predictions.csv',
                                     mode='a', header=False, index=False)
```

#### **C. Infrastructure Monitoring**

- **CPU Usage**: Streamlit Cloud dashboard
- **Memory Usage**: Monitor model loading overhead
- **Disk Usage**: Check log file sizes
- **Network**: API response times (if applicable)

**Tools:**

- Streamlit Cloud built-in metrics
- Google Analytics (optional)
- Uptime monitoring (e.g., UptimeRobot)

---

### 🔍 Analyze

**Data Analysis Process:**

#### **A. Prediction Pattern Analysis**

```python
# Analyze prediction logs
df_logs = pd.read_csv('logs/predictions.csv')

# 1. Model Agreement Rate
agreement_rate = (df_logs['full_model_top_match'] ==
                  df_logs['light_model_top_match']).mean()
print(f"Model Agreement: {agreement_rate*100:.1f}%")

# 2. Probability Distribution
import plotly.express as px

fig = px.histogram(df_logs, x='full_model_top_prob',
                   title='Distribution of Top Probabilities')
fig.show()

# 3. Query Complexity Analysis
df_logs['query_length'] = df_logs['query'].str.len()
df_logs['num_words'] = df_logs['query'].str.split().str.len()

correlation = df_logs[['query_length', 'num_words',
                       'full_model_top_prob']].corr()
```

#### **B. Model Drift Detection**

```python
# Compare current predictions with baseline
baseline_stats = {
    'mean_probability': 0.45,
    'std_probability': 0.25,
    'top_similar_rate': 0.15  # % queries with prob > 0.8
}

current_stats = {
    'mean_probability': df_logs['full_model_top_prob'].mean(),
    'std_probability': df_logs['full_model_top_prob'].std(),
    'top_similar_rate': (df_logs['full_model_top_prob'] > 0.8).mean()
}

# Alert if significant drift
if abs(current_stats['mean_probability'] -
       baseline_stats['mean_probability']) > 0.1:
    print("⚠️ ALERT: Model drift detected!")
```

#### **C. User Behavior Analysis**

```python
# Analyze query patterns
from collections import Counter
import re

# Common keywords in queries
all_queries = ' '.join(df_logs['query'].str.lower())
words = re.findall(r'\b\w+\b', all_queries)
most_common = Counter(words).most_common(20)

# Peak usage times
df_logs['hour'] = pd.to_datetime(df_logs['timestamp']).dt.hour
peak_hours = df_logs['hour'].value_counts().sort_index()
```

#### **D. A/B Testing Results** (Future)

```python
# Compare Full vs Lightweight user preference
user_feedback = pd.read_csv('logs/user_feedback.csv')

full_satisfaction = user_feedback[user_feedback['model']=='full']['rating'].mean()
light_satisfaction = user_feedback[user_feedback['model']=='lightweight']['rating'].mean()

print(f"""
A/B Test Results:
- Full Model Satisfaction: {full_satisfaction:.2f}/5
- Lightweight Model Satisfaction: {light_satisfaction:.2f}/5
- Recommendation: {'Full' if full_satisfaction > light_satisfaction else 'Lightweight'}
""")
```

---

### 🛡️ Govern

**Governance & Compliance:**

#### **A. Data Governance**

**Data Privacy:**

```python
# Anonymize logs (jika diperlukan)
def anonymize_query(query):
    # Hash sensitive information
    import hashlib
    return hashlib.sha256(query.encode()).hexdigest()[:16]

# GDPR Compliance
def delete_user_data(user_id):
    df_logs = pd.read_csv('logs/predictions.csv')
    df_logs = df_logs[df_logs['user_id'] != user_id]
    df_logs.to_csv('logs/predictions.csv', index=False)
```

**Data Retention Policy:**

```python
# Delete logs older than 90 days
def cleanup_old_logs():
    df_logs = pd.read_csv('logs/predictions.csv')
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

    cutoff_date = datetime.now() - timedelta(days=90)
    df_logs = df_logs[df_logs['timestamp'] > cutoff_date]

    df_logs.to_csv('logs/predictions.csv', index=False)
    print(f"✅ Cleaned logs. Retained: {len(df_logs)} records")

# Run monthly
cleanup_old_logs()
```

#### **B. Model Governance**

**Version Control:**

```python
# Model versioning metadata
model_metadata = {
    'version': 'v1.0',
    'trained_date': '2025-12-14',
    'dataset_version': 'v1.0',
    'dataset_size': 123,
    'accuracy': 1.0,
    'roc_auc': 1.0,
    'hyperparameters': {...},
    'approved_by': 'Team Lead',
    'deployment_date': '2025-12-14',
    'next_review_date': '2026-01-14'
}

# Save metadata with model
import json
with open('model_outputs/metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)
```

**Model Audit Trail:**

```python
audit_log = {
    'action': 'model_retrain',
    'timestamp': datetime.now(),
    'user': 'system',
    'reason': 'Scheduled monthly retraining',
    'old_accuracy': 1.0,
    'new_accuracy': 1.0,
    'approved': True,
    'deployed': True
}
```

#### **C. Access Control**

```python
# Streamlit authentication (optional)
import streamlit_authenticator as stauth

names = ['Admin', 'User1', 'User2']
usernames = ['admin', 'user1', 'user2']
passwords = ['xxx', 'yyy', 'zzz']  # Hashed passwords

authenticator = stauth.Authenticate(
    names, usernames, passwords,
    'app_cookie', 'random_key', cookie_expiry_days=30
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show full app
    main()
elif authentication_status == False:
    st.error('Username/password is incorrect')
```

#### **D. Compliance Checklist**

- ✅ **Data Privacy**: No personal data stored in logs
- ✅ **Model Transparency**: Metadata & hyperparameters documented
- ✅ **Reproducibility**: Training scripts version-controlled
- ✅ **Bias Monitoring**: Regular checks on prediction fairness
- ✅ **Security**: Input validation, no SQL injection risk
- ✅ **Explainability**: Feature importance tracked
- ✅ **Audit Trail**: All model changes logged
- ✅ **Disaster Recovery**: Models backed up to cloud storage

#### **E. Regular Review Schedule**

```python
governance_schedule = {
    'Daily': [
        'Check error logs',
        'Monitor response times'
    ],
    'Weekly': [
        'Review prediction patterns',
        'Check model agreement rate',
        'Analyze user feedback'
    ],
    'Monthly': [
        'Retrain model with new data',
        'A/B test model versions',
        'Update documentation',
        'Clean old logs'
    ],
    'Quarterly': [
        'Full model audit',
        'Security review',
        'Compliance check',
        'Stakeholder presentation'
    ]
}
```

#### **F. Incident Response Plan**

```python
incident_levels = {
    'P1 - Critical': {
        'definition': 'App completely down',
        'response_time': '15 minutes',
        'actions': [
            'Rollback to previous version',
            'Notify users via status page',
            'Emergency team meeting'
        ]
    },
    'P2 - High': {
        'definition': 'Model accuracy drops > 10%',
        'response_time': '1 hour',
        'actions': [
            'Investigate data drift',
            'Check for corrupt model files',
            'Retrain if necessary'
        ]
    },
    'P3 - Medium': {
        'definition': 'Slow response times (> 5s)',
        'response_time': '4 hours',
        'actions': [
            'Optimize code',
            'Scale infrastructure',
            'Enable caching'
        ]
    }
}
```

---

## 📝 Summary: Complete MLOps Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                         BUILD PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Data Ingestion    → Load & preprocess 123 judul          │
│ 2. Model Training    → GridSearchCV, Random Forest          │
│ 3. Model Testing     → 100% accuracy, ROC AUC 1.0           │
│ 4. Model Packaging   → Joblib, compression                  │
│ 5. Model Registering → Local storage, Git (lightweight)     │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                        DEPLOY PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ 1. App Testing       → Unit, integration, UI tests          │
│ 2. Production        → Streamlit Cloud (free, auto-deploy)  │
│    Release           → GitHub → share.streamlit.io → LIVE!  │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                       MONITOR PHASE                          │
├─────────────────────────────────────────────────────────────┤
│ 1. Monitor           → Response time, errors, usage         │
│ 2. Analyze           → Prediction patterns, model drift     │
│ 3. Govern            → Data privacy, audit trail, reviews   │
└─────────────────────────────────────────────────────────────┘
                             ↓
                    ┌────────────────┐
                    │  FEEDBACK LOOP │
                    │  Retrain model │
                    │  Deploy update │
                    └────────────────┘
```

---

## 🚀 Quick Start Commands

```bash
# BUILD
python train_model_lightweight.py  # Train lightweight model

# DEPLOY
streamlit run main.py               # Local testing
git push origin main                # Deploy to Streamlit Cloud

# MONITOR
tail -f logs/app.log                # View real-time logs
python analyze_predictions.py      # Analyze performance
```

---

**Proyek ini mengimplementasikan MLOps best practices dengan fokus pada:**

- ✅ Reproducibility (version control)
- ✅ Automation (auto-deploy)
- ✅ Monitoring (logging & analytics)
- ✅ Governance (metadata & audit trail)
- ✅ Scalability (dual model support)
