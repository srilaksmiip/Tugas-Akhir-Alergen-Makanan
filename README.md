# Tugas-Akhir-Alergen-Makanan
# Sistem Deteksi Alergen Berbasis Machine Learning untuk Smart Canteen

Proyek ini bertujuan untuk mengembangkan sebuah modul *machine learning* yang mampu mendeteksi 14 alergen utama dari daftar bahan makanan berbahasa Indonesia. Modul ini dirancang sebagai komponen inti dalam ekosistem Smart Canteen yang lebih besar, berfungsi sebagai layanan pengayaan data untuk meningkatkan keamanan pangan.

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

### Langkah 3: Menjalankan Model
Untuk mendapatkan model terlatih dapat dilakukan dengan
1. Jalankan perintah berikut di terminal:
    ```bash
    python pipeline.py
    ```
2. Hasil penelitian menemukan bahwa model terlatih paling efektif adalah RF sehingga skrip di atas akan menghasilkan model file .joblib RF saja. pada models/. Namun, jika ingin menghasilkan model lain, dapat melakukan konfigurasi sendiri pada skrip config.py

### Langkah 3: Menjalankan Prototipe
1.  Pastikan Anda berada di direktori utama proyek (`TA-klasifikasi-alergen`).
2.  Jalankan perintah berikut di terminal:
    ```bash
    streamlit run proto.py
    ```
3.  Buka browser Anda dan akses alamat URL yang ditampilkan