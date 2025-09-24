# Tugas-Akhir-Alergen-Makanan
# Sistem Deteksi Alergen Berbasis Machine Learning untuk Smart Canteen

Proyek ini bertujuan untuk mengembangkan sebuah modul *machine learning* yang mampu mendeteksi 14 alergen utama dari daftar bahan makanan berbahasa Indonesia. Modul ini dirancang sebagai komponen inti dalam ekosistem Smart Canteen yang lebih besar, berfungsi sebagai layanan pengayaan data untuk meningkatkan keamanan pangan.

## Struktur Repositori
.
├── assets/              # Gambar, diagram, dan aset visual lainnya
├── data/                # Dataset mentah, terproses, dan hasil kurasi
│   ├── 01-dataMentah/
│   ├── 02-intermediateOutput/
│   └── 03_processed/
├── src/                 # Kode sumber utama Python
│   ├── class_dist.py   
│   ├── config.py        
│   ├── dataPreprocessing.py
│   ├── modelTraining.py
│   ├── tempCodeRunner.py
│   └── helper.py
├── models/              # File .joblib model yang sudah dilatih
├── reports/ 
│   ├── figures/         # Hasil plot confusion matrix dan kurva ROC AUC
│   └── metrics/         # Hasil analisis kuantitatif tiap model
├── pipeline.py          # Skrip untuk menjalankan pipeline model
├── pipeline.py          # Skrip untuk menjalankan prototype model dengan streamlit
└── README.md   


## Cara Menjalankan Proyek
### Prasyarat
1.  **Python 3.10+**
2.  **Git** (untuk mengkloning repositori)
3.  Disarankan menggunakan *virtual environment*

### Langkah 1: Kloning Repositori
```bash
git clone 
cd TA-klasifikasi-alergen
```

### Langkah 2: Instalasi Dependensi
Semua pustaka yang dibutuhkan tercantum dalam `requirements.txt`.
```bash
pip install -r requirements.txt
```

### Langkah 3: Menjalankan Prototipe
1.  Pastikan Anda berada di direktori utama proyek (`TA-klasifikasi-alergen`).
2.  Jalankan perintah berikut di terminal:
    ```bash
    streamlit run proto.py
    ```
3.  Buka browser Anda dan akses alamat URL yang ditampilkan

**Model terlatih (file `.joblib`)** merupakan artefak utama dari penelitian tugas akhir ini. File-file ini, yang tersimpan di dalam direktori `/models`, berisi seluruh *pipeline* (TF-IDF Vectorizer dan classifier Random Forest) yang siap digunakan untuk inferensi, seperti yang didemonstrasikan dalam aplikasi prototipe.