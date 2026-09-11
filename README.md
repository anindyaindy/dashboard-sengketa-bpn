# ⚖️ Dashboard Analitik Sengketa Pertanahan — BPN Purbalingga

Aplikasi web interaktif untuk memetakan, menganalisis, dan memprediksi tipologi kasus sengketa pertanahan di Kantor Pertanahan Kabupaten Purbalingga secara visual dan real-time.

Project ini dibangun untuk mentransformasi data sengketa pertanahan yang sebelumnya berbentuk laporan statis/Excel menjadi sebuah dashboard analitik berbasis **Streamlit** dan **Machine Learning**.

---

## 🚀 Fitur Utama

- **Ringkasan KPI Real-Time**: Melihat total berkas sengketa, jumlah kasus yang selesai, serta berkas yang masih dalam proses penyelesaian secara cepat.
- **Filter Interaktif**: Memudahkan pencarian data berdasarkan jenis tipologi kasus atau status penyelesaiannya melalui sidebar.
- **Visualisasi Data Interaktif**: 
  - Grafik batang interaktif (Plotly) untuk melihat distribusi jenis sengketa.
  - Word Cloud untuk mengetahui kata kunci utama yang paling sering muncul pada ringkasan surat/berkas sengketa.
- **Model Prediksi Teks (Machine Learning)**: Implementasi pemodelan klasifikasi teks berbasis **TF-IDF Vectorizer** dan **Multinomial Naive Bayes** untuk mengelompokkan jenis sengketa otomatis, dilengkapi visualisasi *Confusion Matrix*.
- **Eksplorasi Data**: Tabel interaktif untuk melihat rincian isi berkas sengketa secara langsung.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python
- **Framework Web**: Streamlit
- **Visualisasi Data**: Plotly Express, Matplotlib, WordCloud
- **Pengolahan Data & NLP**: Pandas, Scikit-learn (TF-IDF & Naive Bayes)

---

## 📂 Struktur Repository

| Nama File | Deskripsi |
| :--- | :--- |
| `app.py` | Script utama aplikasi web Streamlit |
| `data_sengketa_purbalingga_clean.csv` | Dataset sengketa pertanahan yang sudah dibersihkan |
| `requirements.txt` | Daftar library/package Python yang dibutuhkan untuk deployment |

---

## 💻 Cara Menjalankan Project Secara Lokal

Jika ingin menjalankan aplikasi ini di komputer lokal:

1. Clone repository ini:
   ```bash
   git clone [https://github.com/anindyaindy/dashboard-sengketa-bpn.git](https://github.com/anindyaindy/dashboard-sengketa-bpn.git)
   cd dashboard-sengketa-bpn
