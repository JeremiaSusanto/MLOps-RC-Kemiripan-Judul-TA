"""
Aplikasi Streamlit untuk Deteksi Kemiripan Judul Tugas Akhir
Menggunakan model ML (SVM dan Random Forest) yang sudah di-training.
"""

import os
import math
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from Levenshtein import ratio as levenshtein_ratio
from preprocessing import preprocess_text


# === Cache Loading Model & Data ===
@st.cache_resource
def load_models():
    """
    Load trained models dan artefak.
    Load KEDUA model (full dan lightweight) jika tersedia untuk perbandingan.
    """
    models = {}
    
    # Cek dan load Full Model
    full_model_path = "model_outputs/tfidf.joblib"
    if os.path.exists(full_model_path):
        try:
            models['full'] = {
                'tfidf': joblib.load("model_outputs/tfidf.joblib"),
                'scaler': joblib.load("model_outputs/scaler.joblib"),
                'rf_model': joblib.load("model_outputs/best_rf.joblib"),
                'df_corpus': pd.read_csv("model_outputs/titles_preprocessed.csv"),
                'name': 'Full Model',
                'available': True
            }
            st.sidebar.success(f"✅ Full Model loaded successfully")
        except Exception as e:
            models['full'] = {'available': False}
            st.sidebar.error(f"❌ Full Model load failed: {str(e)}")
    else:
        models['full'] = {'available': False}
        st.sidebar.warning(f"⚠️ Full Model not found at: {os.path.abspath('model_outputs/')}")
    
    # Cek dan load Lightweight Model
    light_model_path = "model_outputs_lightweight/tfidf.joblib"
    if os.path.exists(light_model_path):
        try:
            models['lightweight'] = {
                'tfidf': joblib.load("model_outputs_lightweight/tfidf.joblib"),
                'scaler': joblib.load("model_outputs_lightweight/scaler.joblib"),
                'rf_model': joblib.load("model_outputs_lightweight/best_rf.joblib"),
                'df_corpus': pd.read_csv("model_outputs_lightweight/titles_preprocessed.csv"),
                'name': 'Lightweight Model',
                'available': True
            }
            st.sidebar.success(f"✅ Lightweight Model loaded successfully")
        except Exception as e:
            models['lightweight'] = {'available': False}
            st.sidebar.error(f"❌ Lightweight Model load failed: {str(e)}")
    else:
        models['lightweight'] = {'available': False}
        st.sidebar.warning(f"⚠️ Lightweight Model not found at: {os.path.abspath('model_outputs_lightweight/')}")
    
    # Validasi: minimal satu model harus tersedia
    if not models['full']['available'] and not models['lightweight']['available']:
        st.error("""
        ⚠️ Model belum tersedia!
        
        **Pilihan:**
        1. Training full model (akurasi terbaik):
           ```
           python train_model.py
           ```
        
        2. Training lightweight model (untuk GitHub, lebih cepat):
           ```
           python train_model_lightweight.py
           ```
        """)
        st.stop()
    
    # Info model yang tersedia
    available_models = []
    if models['full']['available']:
        available_models.append("Full Model ✅")
    if models['lightweight']['available']:
        available_models.append("Lightweight Model ✅")
    
    st.sidebar.info(f"🤖 Model Tersedia:\n" + "\n".join([f"- {m}" for m in available_models]))
    
    return models


# === Ekstraksi Fitur Pasangan ===
def jaccard_sim(a_text, b_text):
    """Jaccard similarity pada token set"""
    a = set(a_text.split())
    b = set(b_text.split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_pair_features(query_vec, ref_vec, query_proc, ref_proc, query_raw, ref_raw):
    """
    Ekstraksi 3 fitur terbaik untuk pasangan (query, reference):
    1. Cosine similarity (most important for text similarity)
    2. Jaccard token overlap
    3. Levenshtein ratio (edit distance)
    """
    vi = query_vec.toarray().flatten()
    vj = ref_vec.toarray().flatten()
    
    # 1. Cosine similarity
    cos = float(cosine_similarity([vi], [vj])[0, 0])
    
    # 2. Jaccard similarity
    jacc = jaccard_sim(query_proc, ref_proc)
    
    # 3. Levenshtein ratio
    lev = levenshtein_ratio(query_raw, ref_raw)
    
    return [cos, jacc, lev]


# === Compute Similarities with ML Models ===
def compute_similarities(query: str, tfidf, scaler, rf_model, df_corpus) -> pd.DataFrame:
    """
    Menghitung kemiripan query dengan setiap judul di korpus menggunakan Random Forest.
    Returns: DataFrame dengan kolom judul_ref, rf_probability, rf_label
    """
    # Preprocessing query
    query_proc = preprocess_text(query)
    if not query_proc.strip():
        st.warning("⚠️ Query tidak valid setelah preprocessing. Coba gunakan judul yang lebih deskriptif.")
        return pd.DataFrame()
    
    # TF-IDF transform
    query_tfidf = tfidf.transform([query_proc])
    
    # Ekstraksi fitur untuk setiap pasangan
    results = []
    for idx, row in df_corpus.iterrows():
        ref_raw = row['Judul']
        ref_proc = row['judul_proc']
        ref_tfidf = tfidf.transform([ref_proc])
        
        # Ekstrak 3 fitur
        feats = extract_pair_features(
            query_tfidf, ref_tfidf, 
            query_proc, ref_proc,
            query, ref_raw
        )
        
        # Scale features
        feats_scaled = scaler.transform([feats])
        
        # Predict with Random Forest
        rf_proba = rf_model.predict_proba(feats_scaled)[0, 1]  # probability class 1 (mirip)
        rf_label = "MIRIP" if rf_proba >= 0.5 else "TIDAK MIRIP"
        
        results.append({
            "judul_ref": ref_raw,
            "rf_probability": round(rf_proba, 3),
            "rf_label": rf_label
        })
    
    df = pd.DataFrame(results)
    return df


def main():
    st.set_page_config(page_title="Deteksi Kemiripan Judul Tugas Akhir", layout="wide")

    # Custom CSS - Professional Monochrome Theme
    st.markdown(
        """
        <style>
        /* Main App Background */
        .stApp { 
            background: linear-gradient(135deg, #f5f7fa 0%, #e8edf2 100%); 
            padding-top: 18px; 
        }
        
        /* Header Banner - Elegant Monochrome */
        .header-banner { 
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            padding: 30px 24px;
            border-radius: 16px;
            color: #ffffff;
            position: relative;
            z-index: 9999;
            box-shadow: 0 8px 24px rgba(44, 62, 80, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header-banner h1 {
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-bottom: 8px;
        }
        
        .header-banner p {
            color: #ecf0f1;
            opacity: 0.95;
        }
        
        /* Card Styling - Clean White Cards */
        .card { 
            background: #ffffff;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.08);
            margin-bottom: 20px;
            border: 1px solid #e8edf2;
        }
        
        /* Subheader Styling */
        .card h2, .card h3 {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 16px;
        }
        
        /* Small Notes */
        .small-note { 
            color: #7f8c8d;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(44, 62, 80, 0.2);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            box-shadow: 0 6px 16px rgba(44, 62, 80, 0.3);
            transform: translateY(-2px);
        }
        
        /* Input Fields */
        .stTextArea textarea {
            border: 2px solid #e8edf2;
            border-radius: 8px;
            font-size: 0.95rem;
        }
        
        .stTextArea textarea:focus {
            border-color: #2c3e50;
            box-shadow: 0 0 0 3px rgba(44, 62, 80, 0.1);
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        }
        
        [data-testid="stSidebar"] .stRadio > label {
            color: #ffffff !important;
            font-weight: 500;
        }
        
        [data-testid="stSidebar"] [role="radiogroup"] label {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* DataFrame Styling */
        .dataframe {
            border: 1px solid #e8edf2 !important;
            border-radius: 8px;
        }
        
        /* Success/Warning Messages */
        .stSuccess {
            background-color: #d5f4e6;
            border-left: 4px solid #2c3e50;
            color: #2c3e50;
        }
        
        .stWarning {
            background-color: #fff4e6;
            border-left: 4px solid #7f8c8d;
            color: #2c3e50;
        }
        
        /* Download Button */
        .stDownloadButton > button {
            background: #ffffff;
            color: #2c3e50;
            border: 2px solid #2c3e50;
            border-radius: 8px;
            font-weight: 600;
        }
        
        .stDownloadButton > button:hover {
            background: #2c3e50;
            color: #ffffff;
        }
        
        /* Header Position */
        header[data-testid="stAppHeader"] { 
            position: relative; 
            z-index: 10000;
        }
        
        /* Metrics Styling */
        [data-testid="stMetricValue"] {
            color: #2c3e50;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar menu
    menu = st.sidebar.radio(
        "Menu",
        ["🏠 Home", "📊 Model Evaluation", "ℹ️ About", "👥 Team"],
        index=0,
    )

    if menu == "🏠 Home":
        # Load models
        models = load_models()
        
        # Header
        st.markdown(
            """
            <div class="header-banner">
            <h1 style="margin:0; font-size:2rem; font-weight:700; line-height:1.3">
            Deteksi Kemiripan Judul Tugas Akhir
            </h1>
            <p style="margin-top:12px; font-size:1rem; line-height:1.5">
            Sistem deteksi kemiripan berbasis Machine Learning menggunakan Random Forest dengan perbandingan Full Model vs Lightweight Model
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # Input section

        st.subheader("📝 Input Judul")
        query = st.text_area(
            "Masukkan judul tugas akhir yang ingin dicek kemiripannya:",
            placeholder="Contoh: Implementasi Algoritma Machine Learning untuk Prediksi Harga Saham Menggunakan LSTM",
            height=100,
            label_visibility="collapsed"
        )
        
        # Pilih corpus dari model mana
        corpus_source = models['full'] if models['full']['available'] else models['lightweight']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"<p class='small-note'>Dataset: {len(corpus_source['df_corpus'])} judul dari dataset_TA.csv</p>", unsafe_allow_html=True)
        with col2:
            run = st.button("🔍 Cek Kemiripan", width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        # Hasil
        if run:
            if not query.strip():
                st.warning("⚠️ Mohon masukkan judul terlebih dahulu!")
            else:
                # Cek berapa model yang tersedia
                available_count = sum([models['full']['available'], models['lightweight']['available']])
                
                if available_count == 2:
                    # KEDUA MODEL TERSEDIA - Tampilkan perbandingan
                    st.success("✅ Analisis dengan 2 model selesai!")
                    
                    # Create tabs untuk perbandingan
                    tab1, tab2, tab3 = st.tabs(["📊 Full Model", "🎯 Lightweight Model", "⚖️ Perbandingan"])
                    
                    # Process dengan kedua model
                    with st.spinner("🔄 Memproses dengan Full Model..."):
                        df_full = compute_similarities(
                            query, 
                            models['full']['tfidf'],
                            models['full']['scaler'],
                            models['full']['rf_model'],
                            models['full']['df_corpus']
                        )
                        df_full = df_full.sort_values("rf_probability", ascending=False).reset_index(drop=True)
                        df_full.index = df_full.index + 1
                    
                    with st.spinner("🔄 Memproses dengan Lightweight Model..."):
                        df_light = compute_similarities(
                            query,
                            models['lightweight']['tfidf'],
                            models['lightweight']['scaler'],
                            models['lightweight']['rf_model'],
                            models['lightweight']['df_corpus']
                        )
                        df_light = df_light.sort_values("rf_probability", ascending=False).reset_index(drop=True)
                        df_light.index = df_light.index + 1
                    
                    # TAB 1: Full Model Results
                    with tab1:
                        st.subheader("🤖 Hasil Full Model")
                        st.dataframe(
                            df_full.style.background_gradient(
                                subset=["rf_probability"],
                                cmap="Greys",
                                vmin=0,
                                vmax=1,
                            ).format({"rf_probability": "{:.3f}"}),
                            width='stretch',
                            height=400,
                        )
                        
                        st.markdown("### 🏆 Top 5 Judul Paling Mirip")
                        top_full = df_full.head(5)[["judul_ref", "rf_probability", "rf_label"]]
                        st.dataframe(top_full, width='stretch', hide_index=True)
                        
                        # Visualization
                        st.markdown("### 📈 Visualisasi Probabilitas")
                        top10_full = df_full.head(10).copy()
                        top10_full['Judul (singkat)'] = top10_full['judul_ref'].str[:50] + "..."
                        
                        fig_full = px.bar(
                            top10_full,
                            y='Judul (singkat)',
                            x='rf_probability',
                            orientation='h',
                            color='rf_probability',
                            color_continuous_scale=['#e8edf2', '#7f8c8d', '#34495e', '#2c3e50'],
                            labels={'rf_probability': 'Probabilitas', 'Judul (singkat)': 'Judul'},
                            title='Full Model - Top 10'
                        )
                        fig_full.update_layout(
                            height=500,
                            yaxis=dict(categoryorder='total ascending', gridcolor='#e8edf2', showgrid=False),
                            xaxis=dict(gridcolor='#e8edf2', showgrid=True),
                            font=dict(family="sans-serif", size=12, color="#2c3e50"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_full, width='stretch')
                    
                    # TAB 2: Lightweight Model Results
                    with tab2:
                        st.subheader("🎯 Hasil Lightweight Model")
                        st.dataframe(
                            df_light.style.background_gradient(
                                subset=["rf_probability"],
                                cmap="Greys",
                                vmin=0,
                                vmax=1,
                            ).format({"rf_probability": "{:.3f}"}),
                            width='stretch',
                            height=400,
                        )
                        
                        st.markdown("### 🏆 Top 5 Judul Paling Mirip")
                        top_light = df_light.head(5)[["judul_ref", "rf_probability", "rf_label"]]
                        st.dataframe(top_light, width='stretch', hide_index=True)
                        
                        # Visualization
                        st.markdown("### 📈 Visualisasi Probabilitas")
                        top10_light = df_light.head(10).copy()
                        top10_light['Judul (singkat)'] = top10_light['judul_ref'].str[:50] + "..."
                        
                        fig_light = px.bar(
                            top10_light,
                            y='Judul (singkat)',
                            x='rf_probability',
                            orientation='h',
                            color='rf_probability',
                            color_continuous_scale=['#e8edf2', '#7f8c8d', '#34495e', '#2c3e50'],
                            labels={'rf_probability': 'Probabilitas', 'Judul (singkat)': 'Judul'},
                            title='Lightweight Model - Top 10'
                        )
                        fig_light.update_layout(
                            height=500,
                            yaxis=dict(categoryorder='total ascending', gridcolor='#e8edf2', showgrid=False),
                            xaxis=dict(gridcolor='#e8edf2', showgrid=True),
                            font=dict(family="sans-serif", size=12, color="#2c3e50"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_light, width='stretch')
                    
                    # TAB 3: Perbandingan
                    with tab3:
                        st.subheader("⚖️ Perbandingan Full vs Lightweight Model")
                        
                        # Merge top 10 dari kedua model
                        top10_comparison = df_full.head(10)[['judul_ref', 'rf_probability']].rename(
                            columns={'rf_probability': 'Full Model'}
                        ).merge(
                            df_light.head(10)[['judul_ref', 'rf_probability']].rename(
                                columns={'rf_probability': 'Lightweight Model'}
                            ),
                            on='judul_ref',
                            how='outer'
                        ).fillna(0)
                        
                        # Calculate difference
                        top10_comparison['Selisih'] = abs(
                            top10_comparison['Full Model'] - top10_comparison['Lightweight Model']
                        )
                        top10_comparison = top10_comparison.sort_values('Full Model', ascending=False)
                        
                        st.dataframe(
                            top10_comparison.style.format({
                                'Full Model': '{:.3f}',
                                'Lightweight Model': '{:.3f}',
                                'Selisih': '{:.3f}'
                            }),
                            width='stretch',
                            hide_index=True
                        )
                        
                        # Metrics comparison
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            avg_full = df_full['rf_probability'].mean()
                            st.metric("Avg Probability (Full)", f"{avg_full:.3f}")
                        with col2:
                            avg_light = df_light['rf_probability'].mean()
                            st.metric("Avg Probability (Light)", f"{avg_light:.3f}")
                        with col3:
                            diff = abs(avg_full - avg_light)
                            st.metric("Selisih Rata-rata", f"{diff:.3f}")
                        
                        # Side by side bar chart
                        st.markdown("### 📊 Visualisasi Perbandingan")
                        comparison_melted = top10_comparison.head(10).melt(
                            id_vars=['judul_ref'],
                            value_vars=['Full Model', 'Lightweight Model'],
                            var_name='Model',
                            value_name='Probability'
                        )
                        comparison_melted['Judul (singkat)'] = comparison_melted['judul_ref'].str[:40] + "..."
                        
                        fig_compare = px.bar(
                            comparison_melted,
                            x='Probability',
                            y='Judul (singkat)',
                            color='Model',
                            orientation='h',
                            barmode='group',
                            color_discrete_map={
                                'Full Model': '#2c3e50',
                                'Lightweight Model': '#7f8c8d'
                            }
                        )
                        fig_compare.update_layout(
                            height=600,
                            yaxis=dict(categoryorder='total ascending'),
                            font=dict(family="sans-serif", size=12, color="#2c3e50"),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_compare, width='stretch')
                        
                        # Download buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_full = df_full.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Full Model (CSV)",
                                data=csv_full,
                                file_name="hasil_full_model.csv",
                                mime="text/csv",
                            )
                        with col2:
                            csv_light = df_light.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download Lightweight Model (CSV)",
                                data=csv_light,
                                file_name="hasil_lightweight_model.csv",
                                mime="text/csv",
                            )
                
                else:
                    # HANYA SATU MODEL TERSEDIA
                    model_type = 'full' if models['full']['available'] else 'lightweight'
                    model = models[model_type]
                    
                    # Info untuk user
                    st.info(f"""
                    ℹ️ **Mode Single Model**
                    
                    Saat ini hanya **{model['name']}** yang tersedia. 
                    Fitur perbandingan model memerlukan kedua model (Full + Lightweight).
                    
                    💡 Untuk deployment di Streamlit Cloud, hanya Lightweight Model yang tersedia karena Full Model terlalu besar untuk Git.
                    """)
                    
                    with st.spinner("🔄 Memproses..."):
                        df_results = compute_similarities(
                            query,
                            model['tfidf'],
                            model['scaler'],
                            model['rf_model'],
                            model['df_corpus']
                        )
                    
                    if df_results.empty:
                        st.stop()
                    
                    st.success(f"✅ Analisis dengan {model['name']} selesai!")
                    
                    df_results = df_results.sort_values("rf_probability", ascending=False).reset_index(drop=True)
                    df_results.index = df_results.index + 1
                    
                    # Results table
                    st.subheader("📊 Hasil Prediksi")
                    st.dataframe(
                        df_results.style.background_gradient(
                            subset=["rf_probability"],
                            cmap="Greys",
                            vmin=0,
                            vmax=1,
                        ).format({"rf_probability": "{:.3f}"}),
                        width='stretch',
                        height=400,
                    )
                    
                    # Top similar titles
                    st.subheader("🏆 Top 5 Judul Paling Mirip")
                    top_rf = df_results.nlargest(5, "rf_probability")[["judul_ref", "rf_probability", "rf_label"]]
                    st.dataframe(top_rf, width='stretch', hide_index=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualization
                    st.subheader("📈 Visualisasi Probabilitas Kemiripan")
                    top10 = df_results.head(10).copy()
                    top10['Judul (singkat)'] = top10['judul_ref'].str[:50] + "..."
                    
                    fig = px.bar(
                        top10,
                        y='Judul (singkat)',
                        x='rf_probability',
                        orientation='h',
                        color='rf_probability',
                        color_continuous_scale=['#e8edf2', '#7f8c8d', '#34495e', '#2c3e50'],
                        labels={'rf_probability': 'Probabilitas', 'Judul (singkat)': 'Judul'}
                    )
                    fig.update_layout(
                        height=500,
                        yaxis=dict(categoryorder='total ascending', gridcolor='#e8edf2', showgrid=False),
                        xaxis=dict(gridcolor='#e8edf2', showgrid=True),
                        font=dict(family="sans-serif", size=12, color="#2c3e50"),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, width='stretch')
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Download button
                    csv = df_results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Hasil (CSV)",
                        data=csv,
                        file_name="hasil_kemiripan.csv",
                        mime="text/csv",
                    )

    elif menu == "📊 Model Evaluation":
        st.markdown(
            """
            <div class="header-banner">
            <h1 style="margin:0; font-size:2rem; font-weight:700">
            Evaluasi Model Machine Learning
            </h1>
            <p style="margin-top:12px; font-size:1rem; line-height:1.5">
            Metrik performa dan analisis mendalam dari model Random Forest yang digunakan
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        # Load models info
        models = load_models()
        
        # Pilihan model untuk evaluasi
        available_models = []
        if models['full']['available']:
            available_models.append("Full Model")
        if models['lightweight']['available']:
            available_models.append("Lightweight Model")
        
        if len(available_models) == 0:
            st.error("⚠️ Tidak ada model yang tersedia untuk evaluasi!")
            st.stop()
        
        # Selector jika ada lebih dari 1 model
        if len(available_models) > 1:
            selected_model = st.selectbox("Pilih Model untuk Evaluasi:", available_models)
        else:
            selected_model = available_models[0]
            st.info(f"📌 Menampilkan evaluasi untuk **{selected_model}**")
        
        # Load metrics
        model_dir = "model_outputs" if selected_model == "Full Model" else "model_outputs_lightweight"
        metrics_path = f"{model_dir}/metrics.json"
        
        if not os.path.exists(metrics_path):
            st.error(f"""
            ⚠️ File metrics.json tidak ditemukan!
            
            Model perlu di-train ulang untuk menghasilkan metrics:
            ```
            python {'train_model.py' if selected_model == 'Full Model' else 'train_model_lightweight.py'}
            ```
            """)
            st.stop()
        
        # Read metrics
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        # === 1. Overview Metrics ===
        st.subheader("🎯 Metrik Utama")
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "🎯 Accuracy",
            f"{metrics['accuracy']:.4f}",
            help="Persentase prediksi yang benar dari total prediksi"
        )
        
        col2.metric(
            "⚖️ F1-Score",
            f"{metrics['f1_score']:.4f}",
            help="Keseimbangan antara precision dan recall (weighted untuk dataset tidak seimbang)"
        )
        
        col3.metric(
            "📈 ROC AUC",
            f"{metrics['roc_auc']:.4f}",
            help="Kemampuan model membedakan antar kelas (0-1, semakin tinggi semakin baik)"
        )
        
        st.markdown("---")
        
        # === 2. Classification Report ===
        st.subheader("📊 Classification Report")
        st.markdown("""
        **Penjelasan Kolom:**
        - **Precision**: Dari yang diprediksi positif, berapa persen yang benar-benar positif?
        - **Recall**: Dari yang sebenarnya positif, berapa persen yang berhasil terdeteksi?
        - **F1-Score**: Keseimbangan antara precision dan recall
        - **Support**: Jumlah sampel di setiap kelas
        """)
        
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        
        # Format dan styling
        st.dataframe(
            report_df.style.background_gradient(
                subset=['precision', 'recall', 'f1-score'],
                cmap='YlGn',
                vmin=0,
                vmax=1
            ).format(precision=4),
            use_container_width=True,
            height=250
        )
        
        st.markdown("---")
        
        # === 3. Confusion Matrix ===
        st.subheader("🔢 Confusion Matrix")
        st.markdown("""
        Confusion Matrix menampilkan detail prediksi benar dan salah dari model:
        - **Baris**: Label aktual (ground truth)
        - **Kolom**: Label prediksi model
        """)
        
        cm = np.array(metrics['confusion_matrix'])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Tidak Mirip (0)', 'Mirip (1)'],
            yticklabels=['Tidak Mirip (0)', 'Mirip (1)'],
            cbar_kws={'label': 'Jumlah Prediksi'},
            ax=ax
        )
        ax.set_xlabel('Prediksi', fontsize=12, fontweight='bold')
        ax.set_ylabel('Aktual', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Penjelasan nilai confusion matrix
        st.markdown("### 📝 Interpretasi Confusion Matrix")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ True Positive (TP)**")
            st.write(f"{cm[1,1]:,} - Prediksi mirip, kenyataannya memang mirip")
            
            st.markdown("**⚠️ False Positive (FP)**")
            st.write(f"{cm[0,1]:,} - Prediksi mirip, tapi kenyataannya tidak mirip (Type I Error)")
        
        with col2:
            st.markdown("**✅ True Negative (TN)**")
            st.write(f"{cm[0,0]:,} - Prediksi tidak mirip, kenyataannya memang tidak mirip")
            
            st.markdown("**❌ False Negative (FN)**")
            st.write(f"{cm[1,0]:,} - Prediksi tidak mirip, tapi kenyataannya mirip (Type II Error)")
        
        st.markdown("---")
        
        # === 4. Best Parameters ===
        st.subheader("⚙️ Hyperparameters Terbaik")
        st.markdown("Hyperparameter optimal yang ditemukan melalui GridSearchCV:")
        
        params_df = pd.DataFrame([
            {"Parameter": k, "Value": str(v)} 
            for k, v in metrics['best_params'].items()
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # === 5. Dataset Info ===
        st.subheader("📂 Informasi Dataset")
        dataset_info = metrics['dataset_info']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📄 Total Judul", f"{dataset_info['total_samples']:,}")
            st.metric("🔗 Total Pasangan", f"{dataset_info['total_pairs']:,}")
            st.metric("✅ Pasangan Mirip (Label 1)", f"{dataset_info['positive_pairs']:,}")
        
        with col2:
            st.metric("❌ Pasangan Tidak Mirip (Label 0)", f"{dataset_info['negative_pairs']:,}")
            st.metric("📊 Data Training", f"{dataset_info['train_samples']:,}")
            st.metric("🧪 Data Testing", f"{dataset_info['test_samples']:,}")
        
        # Class distribution
        st.markdown("### 📊 Distribusi Kelas")
        dist_data = pd.DataFrame({
            'Kelas': ['Tidak Mirip', 'Mirip'],
            'Jumlah': [dataset_info['negative_pairs'], dataset_info['positive_pairs']]
        })
        
        fig_dist = px.bar(
            dist_data,
            x='Kelas',
            y='Jumlah',
            color='Kelas',
            color_discrete_map={'Tidak Mirip': '#e74c3c', 'Mirip': '#27ae60'},
            text='Jumlah'
        )
        fig_dist.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_dist.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="",
            yaxis_title="Jumlah Pasangan",
            font=dict(family="sans-serif", size=12, color="#2c3e50")
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Imbalance warning
        ratio = dataset_info['positive_pairs'] / dataset_info['negative_pairs']
        if ratio < 0.5 or ratio > 2.0:
            st.warning(f"""
            ⚠️ **Dataset Imbalanced Detected**
            
            Rasio Mirip:Tidak Mirip = {ratio:.2f}:1
            
            Model mungkin bias terhadap kelas mayoritas. Pertimbangkan teknik balancing seperti:
            - Oversampling (SMOTE)
            - Undersampling
            - Class weights adjustment
            """)

    elif menu == "ℹ️ About":
        st.markdown(
            """
            <div class="header-banner">
            <h1 style="margin:0; font-size:2rem; font-weight:700">
            Tentang Aplikasi
            </h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        st.markdown(
            """
            ### 🎯 Deteksi Kemiripan Judul Tugas Akhir
            
            Aplikasi ini menggunakan **Machine Learning** dengan algoritma Random Forest untuk mendeteksi 
            kemiripan antara judul tugas akhir secara otomatis dan akurat.
            
            ---
            
            ### 🔬 Metodologi
            
            **1. Preprocessing Teks**
            - Stemming menggunakan **Sastrawi** (Bahasa Indonesia)
            - Normalisasi teks (lowercase, remove punctuation)
            - Stopwords removal
            
            **2. Feature Extraction**
            
            Sistem mengekstraksi 3 fitur similarity terbaik:
            
            - **Cosine Similarity** - Mengukur kemiripan vektor TF-IDF antara dua judul
            - **Jaccard Similarity** - Menghitung overlap token/kata yang sama
            - **Levenshtein Ratio** - Menghitung jarak edit antar karakter
            
            **3. Model Machine Learning**
            - Algoritma: **Random Forest Classifier**
            - Hyperparameter tuning dengan **GridSearchCV**
            - Cross-validation untuk validasi model
            
            ---
            
            ### 📊 Dataset
            - **123 judul tugas akhir** dari berbagai topik
            - Bidang: Machine Learning, Deep Learning, NLP, Computer Vision, Time Series
            
            ---
            
            ### 🛠️ Teknologi
            - **Python** - Programming language
            - **Streamlit** - Web framework
            - **scikit-learn** - Machine learning library
            - **Sastrawi** - Indonesian text processing
            - **Plotly** - Interactive visualization
            
            ---
            
            ### 🎓 Projek Tugas Besar Sains Data
            
            Aplikasi ini membantu mendeteksi kemiripan judul untuk mencegah duplikasi topik penelitian 
            dan memberikan analisis mendalam tentang tingkat kemiripan antar judul tugas akhir.
            """
        )

    elif menu == "👥 Team":
        st.markdown(
            """
            <div class="header-banner">
            <h1 style="margin:0; font-size:2rem; font-weight:700">
            Tim Penyusun
            </h1>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                """
                <div class='card' style='text-align:center; padding:32px 24px;'>
                <div style='width:120px; height:120px; background: linear-gradient(135deg, #2c3e50, #34495e); 
                     border-radius:50%; margin: 0 auto 20px; display:flex; align-items:center; justify-content:center;
                     box-shadow: 0 4px 12px rgba(44, 62, 80, 0.2);'>
                    <span style='font-size:48px; color:white;'>👤</span>
                </div>
                <h3 style='color:#2c3e50; font-weight:600; margin:16px 0 8px;'>Jaclin Alcavella</h3>
                <p style='color:#7f8c8d; font-size:0.95rem; margin:0;'><strong>NIM:</strong> 122450015</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                """
                <div class='card' style='text-align:center; padding:32px 24px;'>
                <div style='width:120px; height:120px; background: linear-gradient(135deg, #2c3e50, #34495e); 
                     border-radius:50%; margin: 0 auto 20px; display:flex; align-items:center; justify-content:center;
                     box-shadow: 0 4px 12px rgba(44, 62, 80, 0.2);'>
                    <span style='font-size:48px; color:white;'>👤</span>
                </div>
                <h3 style='color:#2c3e50; font-weight:600; margin:16px 0 8px;'>Jeremia Susanto</h3>
                <p style='color:#7f8c8d; font-size:0.95rem; margin:0;'><strong>NIM:</strong> 122450022</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col3:
            st.markdown(
                """
                <div class='card' style='text-align:center; padding:32px 24px;'>
                <div style='width:120px; height:120px; background: linear-gradient(135deg, #2c3e50, #34495e); 
                     border-radius:50%; margin: 0 auto 20px; display:flex; align-items:center; justify-content:center;
                     box-shadow: 0 4px 12px rgba(44, 62, 80, 0.2);'>
                    <span style='font-size:48px; color:white;'>👤</span>
                </div>
                <h3 style='color:#2c3e50; font-weight:600; margin:16px 0 8px;'>Muhammad Zaky Zaiddan</h3>
                <p style='color:#7f8c8d; font-size:0.95rem; margin:0;'><strong>NIM:</strong> 122450119</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col4:
            st.markdown(
                """
                <div class='card' style='text-align:center; padding:32px 24px;'>
                <div style='width:120px; height:120px; background: linear-gradient(135deg, #2c3e50, #34495e); 
                     border-radius:50%; margin: 0 auto 20px; display:flex; align-items:center; justify-content:center;
                     box-shadow: 0 4px 12px rgba(44, 62, 80, 0.2);'>
                    <span style='font-size:48px; color:white;'>👤</span>
                </div>
                <h3 style='color:#2c3e50; font-weight:600; margin:16px 0 8px;'>Vira Putri Maharani</h3>
                <p style='color:#7f8c8d; font-size:0.95rem; margin:0;'><strong>NIM:</strong> 122450129</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
