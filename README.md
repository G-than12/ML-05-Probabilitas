<div align="center">

# 🧠 ML-05: Teori Probabilitas untuk Machine Learning

### Mata Kuliah: INF2542 • Pembelajaran Mesin | Praktikum Modul 05

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4c72b0?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

<p align="center">
  <b>Dokumentasi komprehensif, eksplorasi teori, bedah kode baris demi baris, dan analisis komputasional probabilitas untuk Machine Learning: Dari Peluang Sederhana, Aturan Komplemen, Probabilitas Bersyarat, Teorema Bayes, Simulasi Monte Carlo, hingga Studi Kasus Akademik Mahasiswa.</b>
</p>

---

[📌 Identitas Mahasiswa](#-identitas-mahasiswa) •
[📖 Pendahuluan & Filosofi](#-pendahuluan--filosofi-probabilitas-dalam-machine-learning) •
[📚 Penjelasan Materi & Teori](#-penjelasan-materi--teori-lengkap) •
[💻 Bedah Kode & Cara Kerja File](#-bedah-kode--cara-kerja-di-notebook) •
[📊 Analisis & Visualisasi](#-analisis-hasil--visualisasi) •
[🚀 Cara Menjalankan](#-cara-menjalankan-proyek) •
[📂 Struktur Direktori](#-struktur-direktori)

---

</div>

## 📌 Identitas Mahasiswa

<table align="center">
  <tr>
    <td width="200"><b>Nama Lengkap</b></td>
    <td>: <b>Gathan Hilabi</b></td>
  </tr>
  <tr>
    <td><b>Nomor Induk Mahasiswa (NIM)</b></td>
    <td>: <b>059</b></td>
  </tr>
  <tr>
    <td><b>Mata Kuliah</b></td>
    <td>: <b>INF2542 • Pembelajaran Mesin</b></td>
  </tr>
  <tr>
    <td><b>Modul Praktikum</b></td>
    <td>: Pertemuan 05 — Probabilitas untuk Machine Learning</td>
  </tr>
  <tr>
    <td><b>Capaian Pembelajaran (Sub-CPMK 6.1)</b></td>
    <td>: Menggunakan teknik dan perangkat yang relevan untuk membangun pipeline, melatih, membandingkan, dan mengevaluasi model</td>
  </tr>
  <tr>
    <td><b>Program Studi / Institusi</b></td>
    <td>: Informatika, UIN K.H. Abdurrahman Wahid Pekalongan</td>
  </tr>
</table>

---

## 📖 Pendahuluan & Filosofi: Probabilitas dalam Machine Learning

Dalam dunia nyata, data tidak pernah bersifat deterministik sempurna; data selalu mengandung derau (_noise_), ketidakpastian (_uncertainty_), serta observasi yang tidak lengkap. Oleh karena itu, **Machine Learning pada hakikatnya adalah seni penalaran di bawah ketidakpastian** (_reasoning under uncertainty_).

Probabilitas bukan sekadar alat hitung frekuensi, melainkan bahasa formal yang digunakan algoritma untuk:

1. **Membuat Prediksi Terkalibrasi**: Model klasifikasi tidak hanya mengeluarkan label kelas diskrit, melainkan nilai kepercayaan seperti $P(y = \text{Fraud} \mid X) = 0.94$.
2. **Mengoptimalkan Fungsi Objektif (_Loss Function_)**: Fungsi _Binary Cross-Entropy_ (Log-Loss) diturunkan secara langsung dari prinsip _Maximum Likelihood Estimation_ (MLE) berbasis probabilitas.
3. **Memperbarui Pengetahuan Melalui Pengalaman**: Seperti pada _Bayesian Inference_, prior pengetahuan digabungkan dengan bukti baru dari data untuk menghasilkan posterior yang optimal.

Repositori ini menyajikan implementasi komputasional dari setiap pilar probabilitas tersebut menggunakan Python, mulai dari konsep dasar hingga simulasi tingkat lanjut.

---

## 📚 Penjelasan Materi & Teori Lengkap

Berikut adalah penjelasan konsep teoretis dan formulasi matematis yang dipelajari pada modul ini:

```
                          ┌──────────────────────────┐
                          │   PROBABILITAS DASAR     │
                          │     P(A) = n(A) / N      │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │   PROBABILITAS BERSYARAT │
                          │   P(A|B) = P(A∩B) / P(B) │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │      TEOREMA BAYES       │
                          │ P(A|B)=P(B|A)·P(A) / P(B)│
                          └────────────┬─────────────┘
                                       │
                   ┌───────────────────┴───────────────────┐
                   ▼                                       ▼
     ┌──────────────────────────┐            ┌──────────────────────────┐
     │   SIMULASI MONTE CARLO   │            │     MACHINE LEARNING     │
     │   Law of Large Numbers   │            │  Naive Bayes & Log-Loss  │
     └──────────────────────────┘            └──────────────────────────┘
```

---

### 1. Probabilitas Sederhana & Aturan Komplemen

Probabilitas mengukur seberapa mungkin suatu kejadian terjadi dalam skala interval kontinu $[0, 1]$:

- $P(A) = 0$: Kejadian mustahil terjadi.
- $P(A) = 1$: Kejadian pasti terjadi.

#### Formula Dasar (Frekuensi Relatif):

$$
P(A) = \frac{n(A)}{N}
$$

Di mana:

- $n(A)$ = Jumlah kejadian yang memenuhi kriteria $A$.
- $N$ = Ukuran ruang sampel semesta ($S$).

#### Aturan Komplemen (_Complement Rule_):

Jika ruang sampel hanya memiliki dua kemungkinan yang bersifat saling lepas (_mutually exclusive_) dan menyeluruh (_exhaustive_), maka probabilitas komplemen ($A^c$ atau $\neg A$) adalah:

$$
P(A^c) = 1 - P(A) \iff P(A) + P(A^c) = 1
$$

---

### 2. Probabilitas Bersyarat (_Conditional Probability_)

Probabilitas bersyarat mengukur peluang terjadinya kejadian $A$ **dengan syarat** kejadian $B$ sudah diketahui telah terjadi.

#### Formula Matematis:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{asalkan } P(B) > 0
$$

Di mana:

- $P(A \mid B)$ : Peluang bersyarat terjadinya $A$ jika $B$ terjadi.
- $P(A \cap B)$ : Peluang terjadinya kedua peristiwa $A$ dan $B$ secara bersamaan (_joint probability_).
- $P(B)$ : Probabilitas marginal peristiwa syarat $B$.

> [!NOTE]
> **Aplikasi Kasus Transaksi Online & Fraud:**
> Secara umum, peluang transaksi penipuan (_fraud_) mungkin kecil (15%). Namun, jika diketahui transaksi tersebut dilakukan melalui kanal **Online**, peluang penipuan bisa melesat menjadi 40%. Informasi tambahan mempersempit ruang sampel dari semesta menjadi hanya subset $B$.

---

### 3. Teorema Bayes (_Bayes' Theorem_)

Teorema Bayes adalah hukum fundamental yang memungkinkan kita **membalik kondisi bersyarat**: menghitung $P(A \mid B)$ dari pengetahuan tentang $P(B \mid A)$, $P(A)$, dan $P(B)$.

#### Formula Matematis:

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

#### Anatomi 4 Pilar Teorema Bayes:

| Komponen       | Notasi        | Penjelasan Konseptual                                        | Contoh Kasus Deteksi Fraud                                  |
| -------------- | ------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| **Prior**      | $P(A)$        | Keyakinan awal terhadap hipotesis sebelum melihat bukti baru | Peluang awal transaksi adalah fraud (15%)                   |
| **Likelihood** | $P(B \mid A)$ | Peluang munculnya bukti jika hipotesis benar                 | Peluang transaksi online jika itu memang fraud (80%)        |
| **Evidence**   | $P(B)$        | Total probabilitas kemunculan bukti pada seluruh kemungkinan | Peluang transaksi dilakukan secara online (30%)             |
| **Posterior**  | $P(A \mid B)$ | Keyakinan yang diperbarui setelah memperhitungkan bukti      | Peluang transaksi adalah fraud setelah tahu dia online (40%) |

#### 📌 Relevansi ke Algoritma Naive Bayes Classifier

Pada klasifikasi teks (seperti filter spam email), algoritma Naive Bayes menghitung peluang posterior kelas target menggunakan asumsi independensi fitur:

$$
P(\text{Spam} \mid \text{Kata}_1, \text{Kata}_2, \dots, \text{Kata}_n) \propto P(\text{Spam}) \prod_{i=1}^n P(\text{Kata}_i \mid \text{Spam})
$$

Model mengasumsikan independensi kondisional antar-fitur agar komputasi peluang kelas target dapat dilakukan dengan sangat cepat dan efisien.

---

### 4. Simulasi Monte Carlo & Hukum Bilangan Besar (_Law of Large Numbers_)

Dalam kenyataannya, banyak ruang masalah yang terlalu kompleks untuk dihitung secara analitis tertutup. **Metode Monte Carlo** menggunakan komputasi acak berulang (_pseudorandom sampling_) untuk mengestimasi parameter matematis.

Berupa **Hukum Bilangan Besar (_Law of Large Numbers_ - LLN)**:

$$
\lim_{N \to \infty} \hat{P}_N(A) = P(A)
$$

Seiring bertambahnya jumlah eksperimen ($N \to \infty$), rata-rata frekuensi empiris akan konvergen mendekati nilai probabilitas teoretis yang sebenarnya, sementara variansi kesalahan (_sampling error_) mengecil dengan laju proporsional $\mathcal{O}\left(\frac{1}{\sqrt{N}}\right)$.

---

## 💻 Bedah Kode & Cara Kerja di Notebook

Di bawah ini adalah penjelasan terperinci mengenai setiap blok kode yang ada pada notebook [`059_GathanHilabi_Pertemuan05.ipynb`](059_GathanHilabi_Pertemuan05.ipynb), alur logika algoritma, dan output yang dihasilkan:

---

### 🔹 Bagian 1: Peluang Sederhana Menggunakan NumPy

```python
import numpy as np

# Data awal status kelulusan 10 mahasiswa
data = np.array([
    "Lulus", "Lulus", "Tidak Lulus",
    "Lulus", "Tidak Lulus", "Lulus",
    "Lulus", "Lulus", "Tidak Lulus", "Lulus"
])

total_data = len(data)
jumlah_lulus = np.sum(data == "Lulus")
p_lulus = jumlah_lulus / total_data

print("=== CODING 1: PROBABILITAS SEDERHANA ===")
print("Jumlah data :", total_data)
print("Jumlah lulus:", jumlah_lulus)
print("P(Lulus)    :", p_lulus)
```

**🔍 Cara Kerja Kode:**

1. `data == "Lulus"` menghasilkan array boolean NumPy: `[True, True, False, True, False, True, True, True, False, True]`.
2. `np.sum(...)` menjumlahkan nilai boolean tersebut (`True` dihitung 1, `False` dihitung 0), menghasilkan nilai integer `7`.
3. Menghitung rasio $7 / 10 = 0.70$.
4. **Hasil Output:** $P(\text{Lulus}) = 0.70$ (atau 70%).

---

### 🔹 Bagian 2: Probabilitas Bersyarat Kasus Transaksi Online & Fraud

```python
# Coding 2: Implementasi Probabilitas Bersyarat
fraud_online = 12
total_online = 30

p_fraud_given_online = fraud_online / total_online

print("=== CODING 2: CONDITIONAL PROBABILITY ===")
print("Jumlah transaksi Fraud & Online :", fraud_online)
print("Total transaksi Online          :", total_online)
print("P(Fraud | Online)               =", p_fraud_given_online)
```

**🔍 Cara Kerja Kode:**

1. Mengisolasi ruang sampel hanya pada transaksi yang bersifat _Online_ (`total_online = 30`).
2. Menghitung proporsi kejadian _Fraud_ di dalam ruang sampel tersaring tersebut (`fraud_online = 12`).
3. **Hasil Output:** $P(\text{Fraud} \mid \text{Online}) = \frac{12}{30} = 0.40$ (40%). Informasi kondisi _Online_ menaikkan risiko fraud dari 15% menjadi 40%.

---

### 🔹 Bagian 3: Teorema Bayes di Python

```python
# Coding 3: Implementasi Teorema Bayes di Python
p_fraud = 0.15                 # P(A) - Prior
p_online_given_fraud = 0.80    # P(B|A) - Likelihood
p_online = 0.30                # P(B) - Evidence

# Menghitung Posterior P(A|B)
p_fraud_given_online = (p_online_given_fraud * p_fraud) / p_online

print("=== CODING 3: TEOREMA BAYES ===")
print("Posterior P(Fraud | Online) =", round(p_fraud_given_online, 3))
```

**🔍 Cara Kerja Kode:**

1. Kode mengalikan _Likelihood_ ($0.80$) dengan _Prior_ ($0.15$), yang menghasilkan peluang bersama $P(\text{Online} \cap \text{Fraud}) = 0.12$.
2. Peluang bersama kemudian dinormalisasi dengan membaginya terhadap _Evidence_ ($0.30$).
3. **Hasil Output:** $\frac{0.12}{0.30} = 0.40$. Terbukti identik dengan perhitungan bersyarat langsung pada Bagian 2.

---

### 🔹 Bagian 4: Simulasi Monte Carlo & Pembuktian Law of Large Numbers

```python
import numpy as np
import matplotlib.pyplot as plt

# Eksperimen variasi ukuran sampel N dan Seed
n_values = [100, 1000, 10000]
seeds = [42, 7, 123]

for n in n_values:
    for s in seeds:
        np.random.seed(s)
        h = np.random.randint(1, 7, size=n)
        est = np.mean(h % 2 == 0)
        selisih = abs(est - 0.5)
        print(f"N: {n:5d} | Seed: {s:3d} | Estimasi: {est:.4f} | Error: {selisih:.4f}")
```

**🔍 Cara Kerja Kode:**

1. `np.random.randint(1, 7, size=n)` melempar dadu virtual $6$-sisi sebanyak $n$ kali.
2. `h % 2 == 0` mengecek angka genap ($2, 4, 6$).
3. `np.mean(...)` menghitung frekuensi relatif kemunculan sisi genap.
4. **Temuan:**
   - Pada $N = 100$, variansi acak masih tinggi (error mencapai $0.0600$).
   - Pada $N = 10.000$, estimasi konvergen mendekati nilai analitis $0.5000$ (error turun menjadi $< 0.0050$).

---

### 🔹 Bagian 5: Tugas Mandiri — Analisis Dataset 20 Mahasiswa

Pada Tugas Mandiri (Slide 21), dibangun sebuah pipeline analitis lengkap berbasis `pandas`:

#### 1. Pembentukan Dataset & Perhitungan Nilai Akhir

```python
import numpy as np
import pandas as pd

np.random.seed(42)

data_mahasiswa = {
    "NIM": [f"603240{i:02d}" for i in range(1, 21)],
    "Nama": [
        "Muhammad Zaki Musyafa", "Shofy Fuadi", "M Irfan Ali", "Rachel Karima", "Ahmad Turmudi",
        "Said Fachry", "Gathan Hilabi", "Hamdi Yahya", "Karunia Raharjo", "Risqi Agung",
        "Imam Prawira", "Rangga dzikri", "Didi Purnomo", "Eka Visi", "Inggil Himawan",
        "Azka Failandri", "Mutiara Sofia", "Lailatul Sofia", "Fadhil Naja", "Surya Ardhianto"
    ],
    "Kehadiran (%)": [88, 92, 65, 78, 95, 60, 90, 85, 70, 92, 58, 82, 87, 72, 94, 62, 80, 85, 68, 96],
    "Nilai_Tugas":   [85, 90, 60, 75, 92, 55, 88, 84, 68, 91, 50, 80, 85, 70, 95, 58, 78, 82, 65, 98],
    "Nilai_Ujian":   [80, 85, 58, 70, 90, 52, 86, 82, 65, 88, 55, 76, 83, 68, 92, 54, 75, 80, 60, 94]
}

df_mhs = pd.DataFrame(data_mahasiswa)

# Bobot penilaian: 30% Kehadiran + 30% Tugas + 40% Ujian
df_mhs["Nilai_Akhir"] = (
    0.30 * df_mhs["Kehadiran (%)"] +
    0.30 * df_mhs["Nilai_Tugas"] +
    0.40 * df_mhs["Nilai_Ujian"]
).round(1)

# Status: Lulus jika Nilai Akhir >= 70.0
df_mhs["Status"] = np.where(df_mhs["Nilai_Akhir"] >= 70.0, "Lulus", "Tidak Lulus")
```

#### 2. Peluang Sederhana & Probabilitas Bersyarat

```python
# Peluang Dasar P(Lulus)
p_lulus = np.mean(df_mhs["Status"] == "Lulus")   # Hasil: 13 / 20 = 0.6500 (65%)

# Probabilitas Bersyarat Kehadiran Tinggi (>= 80%)
df_tinggi = df_mhs[df_mhs["Kehadiran (%)"] >= 80]
p_lulus_given_tinggi = np.mean(df_tinggi["Status"] == "Lulus")  # Hasil: 12 / 12 = 1.0000 (100%)

# Probabilitas Bersyarat Kehadiran Rendah (< 80%)
df_rendah = df_mhs[df_mhs["Kehadiran (%)"] < 80]
p_lulus_given_rendah = np.mean(df_rendah["Status"] == "Lulus")  # Hasil: 1 / 8 = 0.1250 (12.5%)
```

#### 3. Sensitivity Analysis (Variasi Batas Ambang Kehadiran)

```python
ambang_batas = [65, 70, 75, 80, 85, 90]
hasil_sensitivitas = []

for t in ambang_batas:
    kelompok = df_mhs[df_mhs["Kehadiran (%)"] >= t]
    p_kelulusan = np.mean(kelompok["Status"] == "Lulus") if len(kelompok) > 0 else 0
    hasil_sensitivitas.append((t, len(kelompok), p_kelulusan))
```

---

### 🔹 Bagian 6: Eksperimen Tambahan Rancang Sendiri

Untuk memperdalam analisis, dirancang 3 eksperimen lanjutan:

#### 1. Inversi Teorema Bayes: $P(\text{Kehadiran Tinggi} \mid \text{Lulus})$

Ingin dicari: Jika seorang mahasiswa dinyatakan **Lulus**, berapa peluang dia memiliki kehadiran tinggi?

```python
p_prior_tinggi = np.mean(df_mhs["Kehadiran (%)"] >= 80)     # P(Tinggi) = 12/20 = 0.60
p_likelihood   = p_lulus_given_tinggi                        # P(Lulus | Tinggi) = 1.00
p_evidence     = p_lulus                                     # P(Lulus) = 0.65

# Teorema Bayes
p_posterior_bayes = (p_likelihood * p_prior_tinggi) / p_evidence
# Hasil: (1.00 * 0.60) / 0.65 = 0.9231 (92.3%)
```

> **Kesimpulan:** 92.3% mahasiswa yang lulus berasal dari kelompok yang disiplin hadir ≥ 80%.

#### 2. Probabilitas Bersyarat Multikondisi

Mengevaluasi irisan dua syarat: **Kehadiran ≥ 80%** dan **Nilai Tugas ≥ 80**.

```python
kondisi_prima = (df_mhs["Kehadiran (%)"] >= 80) & (df_mhs["Nilai_Tugas"] >= 80)
p_lulus_multi = np.mean(df_mhs[kondisi_prima]["Status"] == "Lulus")       # 100% (11/11 mhs)
p_lulus_non_multi = np.mean(df_mhs[~kondisi_prima]["Status"] == "Lulus")   # 22.2% (2/9 mhs)
```

#### 3. Bootstrap Monte Carlo Resampling (1.000 Iterasi)

Menggunakan teknik _bootstrapping with replacement_ untuk mengukur ketahanan statistik (_robustness_) dan interval kepercayaan 95%:

```python
boot_p_lulus = []
boot_p_lulus_given_tinggi = []

for _ in range(1000):
    sample = df_mhs.sample(n=len(df_mhs), replace=True)
    boot_p_lulus.append(np.mean(sample["Status"] == "Lulus"))

    sub = sample[sample["Kehadiran (%)"] >= 80]
    if len(sub) > 0:
        boot_p_lulus_given_tinggi.append(np.mean(sub["Status"] == "Lulus"))

# 95% Confidence Interval
ci_lulus = np.percentile(boot_p_lulus, [2.5, 97.5])
ci_lulus_tinggi = np.percentile(boot_p_lulus_given_tinggi, [2.5, 97.5])
```

> **Hasil 95% Confidence Interval:**
> - $P(\text{Lulus})$: 45.0% - 85.0%
> - $P(\text{Lulus} \mid \text{Kehadiran Tinggi})$: 100.0% (Sangat stabil di 1.00).

---

## 📊 Analisis Hasil & Visualisasi

Notebook menyajikan 3 visualisasi utama yang disajikan secara terpisah untuk kemudahan analisis:

### 1. Diagram Batang Perbandingan Probabilitas (Bar Chart)

Membandingkan baseline probabilitas kelulusan terhadap dua sub-populasi:

- **Baseline $P(\text{Lulus})$**: 65.0%
- **Kehadiran Tinggi (≥ 80%)**: **100.0%**
- **Kehadiran Rendah (< 80%)**: **12.5%**

```
Probabilitas Kelulusan Berdasarkan Kondisi Kehadiran:
[Kehadiran >= 80%] ████████████████████ 1.00 (100%)
[Baseline Umum   ] █████████████░░░░░░░ 0.65 (65%)
[Kehadiran < 80% ] ██░░░░░░░░░░░░░░░░░░ 0.125 (12.5%)
```

### 2. Kurva Analisis Sensitivitas Ambang Batas (Line Chart)

Memetakan dinamika trade-off antara **ketelitian probabilitas kelulusan** vs **jumlah populasi mahasiswa yang terjaring**:

| Ambang Kehadiran | Jumlah Mahasiswa | Jumlah Lulus | P(Lulus \| Kehadiran ≥ T) |    Status Kelulusan     |
| :--------------: | :--------------: | :----------: | :-----------------------: | :---------------------: |
|    **≥ 65%**     |   17 mahasiswa   |   13 orang   |    **0.7647** (76.5%)     |       Belum murni       |
|    **≥ 70%**     |   15 mahasiswa   |   13 orang   |    **0.8667** (86.7%)     |     Meningkat pesat     |
|    **≥ 75%**     |   13 mahasiswa   |   13 orang   |   **1.0000** (100.0%)     | **Titik Jenuh Optimal** |
|    **≥ 80%**     |   12 mahasiswa   |   12 orang   |   **1.0000** (100.0%)     |    Standar Akademik     |
|    **≥ 85%**     |   10 mahasiswa   |   10 orang   |   **1.0000** (100.0%)     |     Sangat selektif     |
|    **≥ 90%**     |   6 mahasiswa    |   6 orang    |   **1.0000** (100.0%)     |    Populasi menyusut    |

> [!TIP]
> **Insight Batas Ambang:** Titik batas 75% hingga 80% merupakan _optimal decision threshold_. Di atas batas ini, model klasifikasi mencapai akurasi presisi 100% tanpa memangkas ukuran sampel secara berlebihan.

### 3. Scatter Plot Sebaran Kehadiran vs Nilai Akhir

- Sumbu X: Persentase Kehadiran (0 - 100%).
- Sumbu Y: Nilai Akhir Mahasiswa (0 - 100).
- Garis Referensi: Garis ambang kehadiran (80%) dan batas kelulusan (70.0).
- **Interpretasi:** Terlihat pola pemisahan linier (_linear separability_) yang sangat bersih. Kuadran kanan atas diduduki seluruhnya oleh titik hijau (**Lulus**), membuktikan fitur kehadiran memiliki _mutual information_ yang sangat tinggi terhadap label target.

---

## 🚀 Cara Menjalankan Proyek

### 1. Kloning Repositori

```bash
git clone https://github.com/G-than12/ML-05-Probabilitas.git
cd ML-05-Probabilitas
```

### 2. Buat dan Aktifkan Virtual Environment (Disarankan)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalasi Dependensi Pustaka

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

### 4. Eksekusi Jupyter Notebook

```bash
jupyter notebook 059_GathanHilabi_Pertemuan05.ipynb
```

_Atau buka file `.ipynb` langsung di Visual Studio Code dengan ekstensi Jupyter terpasang._

---

## 📂 Struktur Direktori

```plaintext
ML-05-Probabilitas/
│
├── .gitignore                          # Konfigurasi file build / cache yang diabaikan Git
├── 059_GathanHilabi_Pertemuan05.ipynb  # Notebook utama modul, kode praktikum & tugas mandiri
└── README.md                           # Dokumentasi komprehensif repositori (materi & kode)
```

---

## 💡 Ringkasan Temuan Kunci (_Key Takeaways_)

1. **Kehadiran sebagai Prediktor Kuat**: Kehadiran ≥ 80% melipatgandakan kepastian kelulusan mahasiswa dari baseline 65% menjadi 100%.
2. **Kekuatan Teorema Bayes**: Bayes membuktikan bahwa 92.3% mahasiswa yang lulus memiliki catatan kehadiran tinggi, memberikan dasar matematis untuk _early warning system_ akademik.
3. **Konvergensi LLN**: Simulasi Monte Carlo memvalidasi bahwa hukum bilangan besar menjamin akurasi estimasi model seiring dengan pertambahan volume data latih.

---

## 📄 Lisensi

Proyek ini didistribusikan di bawah lisensi **MIT License** — terbuka untuk keperluan akademik, studi mandiri, dan pengembangan lebih lanjut.

<div align="center">
  <sub>Praktikum Pembelajaran Mesin • Pertemuan 05 • 2026</sub><br>
  <sub>Dibuat oleh: <b>Gathan Hilabi</b> (NIM: 059)</sub>
</div>
