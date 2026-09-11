import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------------------
# Konfigurasi Halaman Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Sengketa BPN Purbalingga",
    page_icon="⚖️",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('data_sengketa_purbalingga_clean.csv')

df = load_data()

# ------------------------------------------------------------------------------
# Sidebar Filter
# ------------------------------------------------------------------------------
st.sidebar.title("Filter Data")
list_tipologi = ["Semua"] + list(df['tipologi_kasus'].dropna().unique())
selected_tipologi = st.sidebar.selectbox("Pilih Tipologi Kasus:", list_tipologi)

list_status = ["Semua"] + list(df['status_penyelesaian'].dropna().unique())
selected_status = st.sidebar.selectbox("Pilih Status Penyelesaian:", list_status)

# Filter Logic
df_filtered = df.copy()
if selected_tipologi != "Semua":
    df_filtered = df_filtered[df_filtered['tipologi_kasus'] == selected_tipologi]
if selected_status != "Semua":
    df_filtered = df_filtered[df_filtered['status_penyelesaian'] == selected_status]

# ------------------------------------------------------------------------------
# Header & Key Metrics (KPI)
# ------------------------------------------------------------------------------
st.title("Analytical Dashboard Sengketa Pertanahan")
st.caption("Kantor Pertanahan Kabupaten Purbalingga — Integrated Data Science View")
st.markdown("---")

total_kasus = len(df_filtered)
selesai_kasus = len(df_filtered[df_filtered['status_penyelesaian'].str.contains('Selesai', case=False, na=False)])
proses_kasus = total_kasus - selesai_kasus

col1, col2, col3 = st.columns(3)
col1.metric("Total Berkas Sengketa", f"{total_kasus} Berkas")
col2.metric("Status Selesai", f"{selesai_kasus} Berkas")
col3.metric("Dalam Proses", f"{proses_kasus} Berkas")

st.markdown("---")

# ------------------------------------------------------------------------------
# Visualisasi Interaktif (EDA)
# ------------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Distribusi Tipologi Kasus Sengketa")
    if len(df_filtered) > 0:
        tipologi_counts = df_filtered['tipologi_kasus'].value_counts().reset_index()
        tipologi_counts.columns = ['Tipologi', 'Jumlah']
        fig_tipologi = px.bar(
            tipologi_counts, 
            y='Tipologi', 
            x='Jumlah', 
            orientation='h', 
            color='Tipologi',
            color_discrete_sequence=px.colors.qualitative.Bold,
            text='Jumlah'
        )
        fig_tipologi.update_layout(showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_tipologi, use_container_width=True)
    else:
        st.info("Tidak ada data untuk tipologi ini.")

with c2:
    st.subheader("Kata Kunci Dominan (Word Cloud)")
    if len(df_filtered) > 0 and df_filtered['resume_kasus_clean'].dropna().str.cat():
        all_words = " ".join(df_filtered['resume_kasus_clean'].dropna())
        wc = WordCloud(width=600, height=380, background_color='#0F172A', colormap='plasma').generate(all_words)
        fig_wc, ax_wc = plt.subplots(figsize=(6, 3.8))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("Tidak ada teks ringkasan untuk ditampilkan.")

# ------------------------------------------------------------------------------
# Evaluasi Machine Learning (Confusion Matrix)
# ------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Evaluasi Model Naive Bayes (Confusion Matrix)")

X = df['resume_kasus_clean'].dropna()
y = df.loc[X.index, 'tipologi_kasus']

if len(X) > 0 and len(y.unique()) > 1:
    vec = TfidfVectorizer()
    X_vec = vec.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)
    y_pred = model.predict(X_vec)

    cm = confusion_matrix(y, y_pred, labels=model.classes_)

    fig_cm = px.imshow(
        cm, 
        x=model.classes_, 
        y=model.classes_, 
        text_auto=True, 
        color_continuous_scale='YlGnBu',
        labels=dict(x="Prediksi Model", y="Aktual Data", color="Jumlah")
    )
    st.plotly_chart(fig_cm, use_container_width=True)
else:
    st.warning("Data belum cukup untuk melatih dan menampilkan evaluasi model Machine Learning.")

# ------------------------------------------------------------------------------
# Data Table View
# ------------------------------------------------------------------------------
with st.expander("Lihat Tabel Detail Berkas Sengketa"):
    st.dataframe(df_filtered, use_container_width=True)
