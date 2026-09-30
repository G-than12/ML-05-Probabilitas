<div align="center">

# 🧠 ML-05: Teori Probabilitas untuk Machine Learning
### INF2542 • Pembelajaran Mesin | Praktikum Modul 05

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Eksplorasi Fondasi Matematis Pembelajaran Mesin: Dari Peluang Sederhana, Probabilitas Bersyarat, Teorema Bayes, Simulasi Monte Carlo, hingga Studi Kasus Kelulusan Mahasiswa.</b>
</p>

[📌 Identitas Mahasiswa](#-identitas-mahasiswa) •
[📖 Ringkasan Modul](#-ringkasan-modul) •
[🔬 Topik & Formula](#-topik-dan-formula-kunci) •
[📊 Visualisasi & Analisis](#-visualisasi-dan-analisis) •
[🚀 Cara Menjalankan](#-cara-menjalankan) •
[📂 Struktur File](#-struktur-direktori)

---

</div>

## 📌 Identitas Mahasiswa

<table align="center">
  <tr>
    <td><b>Nama Lengkap</b></td>
    <td>: <b>Gathan Hilabi</b></td>
  </tr>
  <tr>
    <td><b>Nomor Induk Mahasiswa (NIM)</b></td>
    <td>: <b>60324059</b></td>
  </tr>
  <tr>
    <td><b>Mata Kuliah</b></td>
    <td>: INF2542 • Pembelajaran Mesin</td>
  </tr>
  <tr>
    <td><b>Sub-CPMK 6.1</b></td>
    <td>: Menggunakan teknik dan perangkat yang relevan untuk membangun pipeline, melatih, membandingkan, dan mengevaluasi model</td>
  </tr>
  <tr>
    <td><b>Institusi</b></td>
    <td>: UIN K.H. Abdurrahman Wahid Pekalongan</td>
  </tr>
</table>

---

## 📖 Ringkasan Modul

Teori probabilitas adalah fondasi matematika paling krusial dalam machine learning dan kecerdasan buatan. Sebagian besar algoritma machine learning beroperasi di bawah ketidakpastian (*uncertainty*) dan bekerja dengan memodelkan distribusi peluang data.

Repositori ini menyajikan modul interaktif dan praktikum komputasional komprehensif menggunakan Python, mengintegrasikan teori matematis, penalaran analitis, serta simulasi berbasis kode:
- Memahami konsep ruang sampel dan frekuensi relatif.
- Menghitung **probabilitas bersyarat** dan aplikasinya pada deteksi anomali (*fraud detection*).
- Menurunkan dan menerapkan **Teorema Bayes** sebagai dasar algoritma klasifikasi probabilistik (seperti *Naive Bayes*).
- Membuktikan *Law of Large Numbers* melalui **Simulasi Monte Carlo**.
- Melakukan studi empiris dan **Sensitivity Analysis** terhadap korelasi tingkat kehadiran dan kelulusan mahasiswa.

---

## 🔬 Topik dan Formula Kunci

Modul ini diorganisasikan ke dalam beberapa sub-pembahasan terstruktur:

### 1. Probabilitas Dasar & Aturan Komplemen
Peluang suatu kejadian $A$ di dalam ruang sampel $S$:
$$P(A) = \frac{n(A)}{N}$$

Dengan aturan komplemen kejadian tidak terjadi ($A'$ atau $\neg A$):
$$P(A') = 1 - P(A)$$

### 2. Probabilitas Bersyarat (*Conditional Probability*)
Peluang terjadinya kejadian $A$ dengan syarat kejadian $B$ telah terjadi:
$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{dengan } P(B) > 0$$

> [!NOTE]
> Diimplementasikan pada studi kasus deteksi transaksi mencurigakan (*fraud detection*), di mana kehadiran indikator tertentu secara signifikan mengubah peluang apriori suatu transaksi tergolong penipuan.

### 3. Teorema Bayes (*Bayes' Theorem*)
Memperbarui keyakinan probabilitas (*posterior*) berdasarkan bukti baru (*likelihood* dan *prior*):
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

Komponen Teorema Bayes:
- **Prior $P(A)$**: Probabilitas awal hipotesis sebelum ada data/bukti baru.
- **Likelihood $P(B \mid A)$**: Peluang munculnya bukti $B$ jika hipotesis $A$ benar.
- **Evidence / Marginal Likelihood $P(B)$**: Total peluang munculnya bukti $B$.
- **Posterior $P(A \mid B)$**: Probabilitas hipotesis $A$ setelah memperhitungkan bukti $B$.

### 4. Simulasi Monte Carlo & Hukum Bilangan Besar (*Law of Large Numbers*)
Eksperimen acak berulang berskala besar ($N = 10$ hingga $N = 100.000$) untuk menunjukkan konvergensi frekuensi empiris terhadap nilai probabilitas teoritis.

---

## 🏆 Tugas Mandiri & Eksperimen Rancang Sendiri

Praktikum dilengkapi dengan analisis mendalam pada dataset sintetis akademik mahasiswa:

1. **Pemodelan Data Mahasiswa**: Konstruksi dataset dengan atribut kehadiran presensi, nilai tugas, UTS, UAS, dan status kelulusan.
2. **Kalkulasi Probabilitas**:
   - Probabilitas kelulusan dasar: $P(\text{Lulus})$
   - Probabilitas bersyarat kehadiran tinggi: $P(\text{Lulus} \mid \text{Kehadiran} \ge 80\%)$
3. **Sensitivity Analysis (Analisis Sensitivitas Ambang Batas)**:
   - Pengujian variasi ambang batas kehadiran ($70\%, 75\%, 80\%, 85\%$) terhadap peluang kelulusan.
4. **Eksperimen Tambahan Rancang Sendiri**:
   - 🔄 **Inversi Bayes**: Menghitung $P(\text{Kehadiran Tinggi} \mid \text{Lulus})$ menggunakan aturan Bayes.
   - 🎯 **Multikondisi**: Evaluasi $P(\text{Lulus} \mid \text{Kehadiran} \ge 80\% \cap \text{Tugas} \ge 80)$.
   - 🎲 **Bootstrap Monte Carlo (1.000 Iterasi)**: Estimasi interval kepercayaan dan stabilitas estimasi probabilitas bersyarat.

---

## 📊 Visualisasi dan Analisis

Notebook ini menyertakan visualisasi data menggunakan `Matplotlib` dan `Seaborn`:

| No | Tipe Visualisasi | Deskripsi & Tujuan |
|---|---|---|
| 1 | **Bar Chart Perbandingan** | Membandingkan $P(\text{Lulus})$ dasar vs $P(\text{Lulus} \mid \text{Kehadiran Tinggi})$. |
| 2 | **Line Chart Sensitivitas** | Kurva perubahan peluang kelulusan terhadap kenaikan *threshold* presensi. |
| 3 | **Scatter Plot Distribusi** | Pemetaan sebaran kehadiran vs nilai akhir dengan pewarnaan status kelulusan (*decision boundary intuition*). |

---

## 💻 Tech Stack & Kebutuhan Lingkungan

- **Bahasa:** Python 3.9+
- **Lingkungan Kerja:** Jupyter Notebook / JupyterLab / Google Colab / VS Code
- **Pustaka Utama:**
  - `numpy` : Operasi vektor, kalkulasi matriks, dan generator bilangan acak
  - `pandas` : Manipulasi data tabular dan analisis statistik
  - `matplotlib.pyplot` : Pembuatan grafik dan visualisasi dasar
  - `seaborn` : Visualisasi data statistik modern

---

## 🚀 Cara Menjalankan

### 1. Kloning Repositori
```bash
git clone https://github.com/G-than12/ML-05-Probabilitas.git
cd ML-05-Probabilitas
```

### 2. (Opsional) Buat dan Aktifkan Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Pustaka yang Dibutuhkan
```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 4. Jalankan Jupyter Notebook
```bash
jupyter notebook 059_GathanHilabi_Pertemuan05.ipynb
```
*Atau buka file `.ipynb` langsung menggunakan ekstensi Jupyter di Visual Studio Code.*

---

## 📂 Struktur Direktori

```plaintext
ML-05-Probabilitas/
│
├── .gitignore                          # Konfigurasi file yang diabaikan Git
├── 059_GathanHilabi_Pertemuan05.ipynb  # Notebook utama praktikum & tugas mandiri
└── README.md                           # Dokumentasi komprehensif repositori
```

---

## 💡 Temuan Kunci (*Key Insights*)

> [!TIP]
> - **Dampak Kehadiran Terhadap Kelulusan**: Mahasiswa dengan persentase kehadiran tinggi ($\ge 80\%$) memiliki probabilitas kelulusan signifikan lebih tinggi dibandingkan probabilitas kelulusan agregat dasar.
> - **Penalaran Bayes**: Teorema Bayes membuktikan bahwa informasi apriori (*prior*) yang dikombinasikan dengan bukti empiris (*evidence*) dapat memitigasi kesalahan estimasi risiko pada data dengan bias kelas.
> - **Hukum Bilangan Besar**: Simulasi Monte Carlo memvalidasi bahwa semakin besar jumlah observasi sampel, dispersi estimasi probabilitas akan menyempit dan mendekati probabilitas analitis eksak.

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah lisensi **MIT License** - silakan gunakan untuk kebutuhan edukasi dan pembelajaran akademis.

<div align="center">
  <sub>Dikembangkan dengan dedikasi untuk Praktikum Pembelajaran Mesin • 2026</sub><br>
  <sub><b>Gathan Hilabi</b> — NIM 60324059</sub>
</div>
