import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Dashboard Sengketa BPN Purbalingga",
    page_icon="⚖️",
    layout="wide"
)

# ==============================================================================
# 2. HEADER BANNER & CARDS (TEMA BPN)
# ==============================================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    padding: 30px;
    border-radius: 15px;
    color: #1e293b;
    margin-bottom: 20px;
    border-top: 5px solid #ffca28;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
">
    <h1 style="color: #102a45; margin: 0; font-size: 28px; font-weight: 800;">
        Dashboard Analisis Sengketa Pertanahan BPN Purbalingga
    </h1>
    <p style="color: #475569; margin-top: 8px; font-size: 15px;">
        Transformasi Digital Berkas Sengketa Menjadi Insight Analitis Berbasis Data Science & Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    background-color: #102a45;
    padding: 20px;
    border-radius: 12px;
    border-right: 6px solid #ffca28;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 25px;
    color: #f1f5f9;
">
    <h3 style="color: #ffffff; margin-top: 0; font-size: 20px; font-weight: 700;">
        Selamat Datang di Sistem Analitik Sengketa! ⚖️
    </h3>
    <p style="color: #cbd5e1; font-size: 14px; margin-bottom: 10px;">
        Gunakan menu navigasi di sebelah kiri (Sidebar) untuk memfilter data dan menjelajahi fitur yang tersedia:
    </p>
    <ul style="color: #e2e8f0; line-height: 1.6; font-size: 13px; margin-bottom: 0;">
        <li><b>📊 Ringkasan Metrik</b>: Pantau total kasus, jumlah penyelesaian, dan berkas aktif secara otomatis.</li>
        <li><b>📈 Visualisasi & Word Cloud</b>: Analisis distribusi tipologi sengketa dan kata kunci dokumen.</li>
        <li><b>🤖 Model Klasifikasi Real-Time</b>: Pengujian otomatis tipologi sengketa menggunakan Naive Bayes.</li>
        <li><b>🗃️ Eksplorasi Data</b>: Tinjau dan unduh detail dataset sengketa secara langsung.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. LOAD DATASET
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("data_sengketa_purbalingga_clean.csv") 
    return df

try:
    df = load_data()
except Exception as e:
    st.error("Gagal memuat dataset. Pastikan file 'data_sengketa_purbalingga_clean.csv' ada di repository GitHub Anda.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR FILTER & PROFILE
# ==============================================================================
st.sidebar.header("🔍 Filter Data")

# Filter Tipologi Kasus
list_tipologi = ["Semua"] + list(df['tipologi_kasus'].dropna().unique()) if 'tipologi_kasus' in df.columns else ["Semua"]
pilihan_tipologi = st.sidebar.selectbox("Pilih Tipologi Kasus:", list_tipologi)

# Filter Status Penyelesaian
list_status = ["Semua"] + list(df['status_penyelesaian'].dropna().unique()) if 'status_penyelesaian' in df.columns else ["Semua"]
pilihan_status = st.sidebar.selectbox("Pilih Status Penyelesaian:", list_status)

# Penerapan Filter ke Dataframe
df_filtered = df.copy()
if pilihan_tipologi != "Semua":
    df_filtered = df_filtered[df_filtered['tipologi_kasus'] == pilihan_tipologi]

if pilihan_status != "Semua":
    df_filtered = df_filtered[df_filtered['status_penyelesaian'] == pilihan_status]

# Profile Pembuat di Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Tentang Project")
st.sidebar.info("""
**Pengembang:** Anindya Sukma Dwiyanda  
**Instansi:** BPN Kabupaten Purbalingga  
**Teknologi:** Python, Streamlit, Scikit-Learn, Plotly
""")

# ==============================================================================
# 5. RINGKASAN METRIK (KPI CARDS)
# ==============================================================================
col1, col2, col3 = st.columns(3)

total_berkas = len(df_filtered)
selesai = len(df_filtered[df_filtered['status_penyelesaian'] == 'Selesai']) if 'status_penyelesaian' in df_filtered.columns else 0
proses = total_berkas - selesai

col1.metric("Total Berkas Sengketa", f"{total_berkas} Berkas")
col2.metric("Kasus Selesai", f"{selesai} Berkas")
col3.metric("Dalam Proses", f"{proses} Berkas")

st.markdown("---")

# ==============================================================================
# 6. VISUALISASI DATA & WORD CLOUD
# ==============================================================================
st.subheader("📊 Visualisasi & Analisis Kasus Sengketa")

if not df_filtered.empty:
    # Membagi grafik menjadi 2 kolom sejajar
    col_chart1, col_chart2 = st.columns(2)

    # 1. Bar Chart Tipologi Kasus
    with col_chart1:
        if 'tipologi_kasus' in df_filtered.columns:
            df_tipologi = df_filtered['tipologi_kasus'].value_counts().reset_index()
            df_tipologi.columns = ['Tipologi Kasus', 'Jumlah']
            
            fig_bar = px.bar(
                df_tipologi, 
                x='Tipologi Kasus', 
                y='Jumlah',
                title="Distribusi Tipologi Kasus",
                color='Tipologi Kasus',
                text_auto=True,
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Jumlah Kasus")
            st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Donut Chart Status Penyelesaian (GRAFIK BARU)
    with col_chart2:
        if 'status_penyelesaian' in df_filtered.columns:
            df_status = df_filtered['status_penyelesaian'].value_counts().reset_index()
            df_status.columns = ['Status', 'Jumlah']
            
            fig_pie = px.pie(
                df_status, 
                names='Status', 
                values='Jumlah',
                title="Persentase Status Penyelesaian",
                hole=0.4, # Membuat efek donut chart
                color_discrete_sequence=['#2bf29a', '#ff4b4b'] # Hijau untuk selesai, merah/oranye untuk proses
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- DATA STORYTELLING INSIGHT ---
    if 'tipologi_kasus' in df_filtered.columns:
        top_tipologi = df_filtered['tipologi_kasus'].value_counts().idxmax()
        top_jumlah = df_filtered['tipologi_kasus'].value_counts().max()
        st.info(f"💡 **Key Insight:** Tipologi sengketa yang paling sering terjadi pada filter ini adalah **{top_tipologi}** dengan total **{top_jumlah} berkas**.")

    # --- WORD CLOUD RINGKASAN BERKAS ---
    st.markdown("#### ☁️ Word Cloud Ringkasan Berkas Sengketa")
    if 'resume_kasus_clean' in df_filtered.columns:
        teks_sengketa = " ".join(df_filtered['resume_kasus_clean'].dropna().astype(str))
        
        if teks_sengketa.strip() != "":
            wc = WordCloud(
                width=800, 
                height=350, 
                background_color='white',
                colormap='viridis'
            ).generate(teks_sengketa)
            
            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        else:
            st.warning("Teks resume tidak ditemukan untuk kombinasi filter ini.")

else:
    st.warning("Data tidak ditemukan untuk kombinasi filter ini.")
st.markdown("---")

# ==============================================================================
# 7. MODEL MACHINE LEARNING & SIMULASI REAL-TIME
# ==============================================================================
st.subheader("🤖 Evaluasi Model Naive Bayes & Simulasi Real-Time")

X = df['resume_kasus_clean'].dropna() if 'resume_kasus_clean' in df.columns else pd.Series()
y = df.loc[X.index, 'tipologi_kasus'] if 'tipologi_kasus' in df.columns else pd.Series()

if len(X) > 0 and len(y.unique()) > 1:
    vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2))
    X_vec = vec.fit_transform(X)

    model = MultinomialNB(alpha=0.1)
    model.fit(X_vec, y)
    y_pred = model.predict(X_vec)

    # Matriks Evaluasi
    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    fig_cm = px.imshow(
        cm,
        x=model.classes_,
        y=model.classes_,
        text_auto=True,
        color_continuous_scale='YlGnBu',
        title="Confusion Matrix Classification"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- SIMULASI PREDIKSI REAL-TIME ---
    st.markdown("#### 🧪 Uji Prediksi Kasus Baru")
    input_teks = st.text_area("Masukkan teks ringkasan berkas sengketa baru di sini untuk dites oleh model:", "")

    if st.button("Prediksi Tipologi Kasus"):
        if input_teks.strip() != "":
            teks_vec = vec.transform([input_teks])
            prediksi = model.predict(teks_vec)[0]
            st.success(f"**Hasil Prediksi Tipologi:** {prediksi}")
        else:
            st.warning("Silakan masukkan teks ringkasan kasus terlebih dahulu.")

st.markdown("---")

# ==============================================================================
# 8. TABEL DETAIL DATASET & TOMBOL DOWNLOAD CSV
# ==============================================================================
with st.expander("📂 Lihat & Unduh Tabel Detail Berkas Sengketa"):
    st.dataframe(df_filtered, use_container_width=True)
    
    # --- TOMBOL DOWNLOAD CSV ---
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data Hasil Filter (CSV)",
        data=csv_data,
        file_name="data_sengketa_bpn_filtered.csv",
        mime="text/csv"
    )
