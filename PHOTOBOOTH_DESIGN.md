# Studio Sela Photobooth â€” rancangan dan peta implementasi

> **Untuk update berikutnya:** baca file ini sebelum mengubah pengalaman photobooth. Perbarui bagian "Keputusan produk" dan "Peta file" jika alur, pilihan style, endpoint, atau konfigurasi model berubah.

## Tujuan

Mengubah repo neural style transfer menjadi photobooth web lokal: pengunjung mengambil rangkaian lima foto atau mengunggah lima gambar, memilih style, lalu mengunduh GIF loop dan JPG frame.

## Alur pengunjung

1. **Layar 1 (Nama + Kamera + SIBI)**:
   - Pengguna melihat kamera live di sisi kiri (~62% lebar).
   - Di sisi kanan terdapat 3 kartu bertumpuk: (1) Input NAMAMU (default terisi "UNESA"; jika pengguna mulai mengetik atau menggunakan SIBI maka teks otomatis berganti dari awal, dan jika langsung menekan DONE maka otomatis memakai nama "UNESA"), (2) PREDIKSI SIBI (huruf besar, status keyakinan, bilah progres 3 detik), dan (3) Panduan cara pakai bahasa SIBI.
   - Di bagian bawah terdapat keyboard layar lengkap (QWERTY, baris kontrol: HAPUS, SPASI, BERSIHKAN) dengan ukuran tombol besar yang mudah dipencet.
   - Menekan tombol bilah **DONE** (lebar penuh di bagian paling bawah) mengunci nama dan beralih ke Layar 2.
2. **Layar 2 (Kamera + Style + Cara Pakai - 3 Kolom)**:
   - Kolom kiri: **CARA PAKAI (user flow)** dengan 4 kartu instruksi (1. Atur posisi, 2. Pilih style, 3. Ambil 5 foto, 4. Pilih 3 foto).
   - Kolom tengah: **KAMERA (80% frame)** dengan bilah status **AMBIL 5 FOTO 0 / 5** di bawahnya. Tombol shutter merah manual ditiadakan.
   - Kolom kanan: **PILIH STYLE** dalam grid 2 kolom x 4 baris.
3. **Trigger Otomatis Jepret Saat Memilih Style**:
   - Memilih salah satu kartu gaya langsung memicu hitung mundur (3, 2, 1) dan rangkaian 5 jepretan otomatis secara beruntun.
   - Bilah status memperbarui hitungan foto (`1 / 5` hingga `5 / 5`).
4. **Pemilihan 3 Foto & Auto-Generate**:
   - Setelah 5 foto selesai diambil, antarmuka langsung membuka layar pemilihan foto (*capture picker*).
   - Pengguna memilih tepat 3 dari 5 foto favorit untuk masuk ke dalam frame.
   - Begitu 3 foto dipilih dan dikonfirmasi, sistem langsung memulai proses neural style transfer menggunakan style yang sudah dipilih di awal, tanpa perlu memilih style ulang atau menekan tombol tambahan.
5. **Hasil Karya, Unduh Satu Tombol & QR Code Google Drive**:
   - Menampilkan preview GIF loop dan frame JPG bersanding rapi.
   - Tombol unduh disatukan menjadi 1 tombol utama: **`UNDUH & DAPATKAN QR CODE GOOGLE DRIVE`**.
   - Ketika ditekan, muncul popup **QR Code Google Drive**:
     - Pengunjung memindai (*scan*) QR Code menggunakan kamera ponsel untuk langsung membuka folder Google Drive dan mengunduh seluruh hasil ke ponsel mereka.
     - QR membuka folder hasil di Google Drive; tombol modal dan tombol hasil mengulang alur foto.
   - Paket karya yang diunggah ke Google Drive (dan di dalam ZIP):
     - `studio-sela-loop.gif` (GIF animasi 10 frame)
     - `studio-sela-frame.jpg` (Frame dengan 3 foto pilihan)
     - 5 foto asli (`foto_asli_01.jpg` .. `foto_asli_05.jpg`)
     - 5 foto style (`foto_style_01_dengan_frame.jpg` .. `foto_style_05_tanpa_frame.jpg`, mencakup baik foto yang masuk ke dalam frame maupun yang tidak dimasukkan ke dalam frame)
     - Total terdapat 10 foto individual + GIF + Frame.
   - Konfigurasi Google Drive memakai OAuth Desktop: simpan file unduhan OAuth sebagai `client_secret.json` di root proyek. Login Google pertama menghasilkan `token.json` lokal.
   - Salin `gdrive_config.example.json` menjadi `gdrive_config.json` dan isi `folder_id` opsional dengan folder yang dapat diedit oleh akun OAuth.
   - Opsi "Foto Lagi" untuk me-reset sesi ke layar awal.

## Keputusan produk

- Bahasa UI: Indonesia; nama gaya koleksi dipertahankan agar mudah dicocokkan dengan aset.
- Arah visual: studio editorial hangat, latar krem, tinta hijau gelap, aksen terakota, tipografi serif untuk headline dan sans-serif untuk kontrol.
- Area photobooth desktop mengisi satu layar dengan acuan 1920×1080: kontainer maksimal 1720 px, grid dua kolom tanpa spasi kosong di kiri (kamera ±62%, panel style ±34%) dan panel style di kanan. Tinggi tidak dipaksa lewat `aspect-ratio`; area kamera memakai sisa tinggi sehingga tidak ada scroll horizontal. Saat lebar ≤1180 px layout kembali satu kolom, dan pada tinggi layar ≤820 px jarak vertikal dirapatkan. Di ponsel tetap satu kolom.
- Kursor dikendalikan ujung ibu jari lewat MediaPipe Hands. Mencubit ibu jari dan telunjuk menjalankan klik; saat cubitan, posisi kursor dikunci agar klik tidak bergeser dari target. Zona gerak aktif memakai 85% dari rentang gerak sebelumnya dan dipetakan ke seluruh halaman agar tombol tepi lebih mudah dijangkau. Indikator tangan hanya tampil sebagai lingkaran di atas halaman; jangan menggambar landmark/garis pada preview kamera. Overlay kursor harus berada di DOM sebelum skrip pelacakan dijalankan.
- Setelah foto dijepret/diunggah, preview foto menutupi video kamera yang tetap berjalan di belakangnya supaya MediaPipe tetap menerima frame untuk kontrol tangan.
- Style yang ditawarkan: Starry Night, Candy, Mosaic, Udnie, The Wave, Ben Giles, La CafÃ©, Flowers, Batik, Batik 2, dan Mosaic Amber. Mosaic Amber memakai pola mosaik geometris bernuansa amber, ochre, krem, dan cokelat; asetnya `data/style-images/mosaic_amber.jpg`.
- Engine: neural style transfer Gatys yang sudah ada (`neural_style_transfer.py`), VGG19 (model disiapkan saat startup; representasi style hanya dihitung saat diminta), L-BFGS, inisialisasi content, tinggi 320 px, 200 iterasi per foto. Lima foto diproses berurutan sehingga durasi sekitar lima kali pemrosesan satu foto, ditambah waktu membuat GIF/frame. CLI tetap memakai default lama 1000 iterasi.
- Seluruh lima sumber foto menjalani neural style transfer secara berurutan menggunakan style yang sama. GIF loop berisi 10 frame (5 asli + 5 hasil style) yang diacak sebagai satu kumpulan tanpa mempertahankan urutan capture/render. JPG frame vertikal menampilkan tiga hasil style yang dipilih pengguna.
- Ukuran total unggahan dibatasi 60 MB; kamera menghasilkan JPEG untuk lima frame.
- Navigasi Gerakan Tangan (Gesture Control): Menggunakan MediaPipe Hands di frontend untuk mengendalikan kursor virtual via telunjuk dan aksi klik tanpa sentuh menggunakan gesture menjepit jempol-telunjuk (pinch).
- Model SIBI hasil training: `models/sibi_landmark_classifier.keras` memakai 21 landmark MediaPipe (x/y relatif, 42 fitur) dan MLP 24 kelas. Training memproses gambar penuh dengan MediaPipe; label diambil dari nama berkas. Evaluasi terakhir: akurasi validasi 95.03% dan tes 93.93% pada sampel yang berhasil dideteksi MediaPipe; MediaPipe menemukan tangan pada 4.197/6.120 train, 483/720 validasi, dan 247/360 tes. Angka akurasi bersyarat pada tangan berhasil dideteksi dan belum menjamin performa live. J dan Z tidak tercakup karena memerlukan gerakan. Model Keras, label, bobot JSON browser, dan laporan berada di `models/`.
- Dataset attribution: Teguh Andriyanto, â€œLabeled Official Indonesian Sign Language (SIBI) Alphabet,â€ Mendeley Data, CC BY 4.0: https://data.mendeley.com/datasets/yhsgmynccf/1.
- Input nama SIBI di browser memuat bobot JSON lewat `/api/sibi/model`, menilai landmark 21 titik secara lokal, dan menampilkan kandidat dengan keyakinan. Panel prediksi selalu terlihat pada layar awal, termasuk ketika belum ada tangan terdeteksi. Huruf otomatis dimasukkan setelah prediksi di atas ambang 72% bertahan selama tiga detik; untuk mengeja huruf yang sama lagi, lepaskan tangan sebentar lalu tunjukkan ulang. Keyboard fisik dan keyboard layar tetap tersedia.
- Urutan alfabet dan huruf bergerak mengikuti `data/sibi/sibi_alphabet_config.json`, berdasarkan gambar acuan SIBI dari pengguna. Model hanya memuat huruf yang memiliki contoh gambar berlabel; J dan Z tetap menggunakan keyboard karena memerlukan gerakan.
- Persentase optimasi L-BFGS adalah perkiraan berbasis panggilan closure (termasuk line search), bukan pengukuran waktu. Muat model dan persiapan style ditampilkan sebagai tahap terpisah; unduhan bobot model pertama kali masih dapat memakan waktu.
- Hasil dan foto sesi disimpan lokal di `data/photobooth/<job-id>/`. Server bind hanya ke `127.0.0.1`; ini prototipe lokal, bukan layanan publik multi-user.
- Koneksi kamera memerlukan localhost atau HTTPS sesuai aturan browser. Unggah foto adalah fallback.

## Peta file

| File | Tanggung jawab |
|---|---|
| `photobooth.py` | Flask app, validasi lima foto, NST berurutan, komposisi GIF/JPG dengan Pillow, status job, dan route hasil. |
| `templates/photobooth.html` | Form nama awal, kamera, pemilih tiga foto, kartu style, status proses, preview serta unduhan GIF/JPG. |
| `static/photobooth.css` | Sistem visual Studio Sela, tata letak responsif, seleksi foto, dan hasil. |
| `static/photobooth.js` | Form nama, prediksi dan konfirmasi SIBI tiga detik, keyboard layar, rangkaian lima jepretan, unggah, seleksi tiga foto, kontrol tangan, polling job, dan preview hasil. |
| `train_sibi_landmarks.py` | Ekstraksi landmark MediaPipe dari dataset SIBI dan training/evaluasi classifier 24 huruf statis. |
| `data/sibi/sibi_alphabet_config.json` | Urutan alfabet SIBI, label huruf bergerak, dan fallback keyboard sesuai gambar acuan. |
| `models/sibi_landmark_classifier.keras` | Classifier Keras hasil training. |
| `models/sibi_landmark_weights.json` | Bobot dense classifier untuk inferensi ringan di browser. |
| `neural_style_transfer.py` | Pipeline optimasi NST. `config['iterations']` memberi override untuk web; tanpa override perilaku CLI tetap 1000/3000. |
| `requirements-web.txt` | Flask/Werkzeug/Pillow untuk server web dan penyusunan aset hasil di samping dependensi NST. |
| `data/style-images/` | Gambar referensi untuk kartu style dan masukan style NST. |
| `data/photobooth/` | Direktori sesi foto dan output yang dibuat saat aplikasi digunakan. |

## Menjalankan lokal

1. Aktifkan environment proyek yang berisi PyTorch/torchvision/OpenCV dan pasang lapisan web: `pip install -r requirements-web.txt`.
2. Jalankan `python photobooth.py` dari root repository.
3. Buka `http://127.0.0.1:5000`.

Model VGG yang diperlukan dapat diunduh torchvision saat pertama kali dipakai; untuk hasil, perangkat sebaiknya memiliki model yang sudah tersedia. Perhitungan NST tradisional memang memerlukan waktu, sehingga halaman memperlihatkan status proses sebelum menampilkan hasil final.

## Catatan update

- Saat menambah/mengganti style, perbarui daftar `STYLES` di `photobooth.py` dan pastikan nama file style tepat. Samakan jumlah/style copy pada catatan ini.
- Jika path aset style berubah, perbarui route `/styles`, kartu pada template bila perlu, dan `Peta file` ini.
- Jika mengubah langkah, endpoint, atau status progres (`phase`, `progress`), jaga kesesuaian `photobooth.py`, `neural_style_transfer.py`, dan `static/photobooth.js`, lalu dokumentasikan kontrak API di sini.
- Jangan ubah default optimizer/iterasi CLI hanya untuk mempercepat web demo; gunakan override `iterations` pada konfigurasi web.
- Jika server dipindahkan dari localhost, tinjau autentikasi, privasi/retensi unggahan, batas ukuran, concurrency GPU/CPU, serta pembatasan akses sebelum membuka akses jaringan.
- Untuk melatih ulang classifier alfabet, jalankan `python train_sibi_landmarks.py` dengan TensorFlow, MediaPipe, NumPy, dan Pillow tersedia. Script membaca gambar penuh dari split train/val/test, mengambil label dari nama berkas, dan mengabaikan gambar yang tidak terdeteksi MediaPipe.
- Neural style transfer tidak membutuhkan training ulang saat referensi Batik diganti atau ditambah. Setiap style dipakai sebagai target tekstur dan dioptimalkan saat pengguna membuat hasil photobooth.

