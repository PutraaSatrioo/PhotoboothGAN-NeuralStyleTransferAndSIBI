# Studio Sela Photobooth â€” rancangan dan peta implementasi

> **Untuk update berikutnya:** baca file ini sebelum mengubah pengalaman photobooth. Perbarui bagian "Keputusan produk" dan "Peta file" jika alur, pilihan style, endpoint, atau konfigurasi model berubah.

## Tujuan

Mengubah repo neural style transfer menjadi photobooth web lokal: pengunjung mengambil rangkaian lima foto atau mengunggah lima gambar, memilih style, lalu mengunduh GIF loop dan JPG frame.

## Alur pengunjung

1. Pengguna mengisi nama (maksimal 32 karakter) lewat keyboard fisik atau keyboard layar, lalu menekan **Done**. Nama akan dicetak di kiri atas JPG frame.
2. Layar awal meminta izin kamera dan menampilkan preview live. MediaPipe Hands mendeteksi landmark untuk kontrol tangan dan classifier SIBI statis yang dilatih dari dataset lokal.
3. Setelah halaman photobooth terbuka, kamera terus berjalan; bila kamera tidak tersedia, pengguna tetap bisa memakai keyboard dan memilih gambar dari perangkat.
4. Tekan shutter untuk mengambil lima foto otomatis, masing-masing dengan hitung mundur tiga detik dan kilatan putih satu detik setelah jepret; alternatifnya, unggah lima foto.
5. Pilih tepat tiga dari lima foto untuk frame. Kelima foto tetap akan diberi style dan digunakan untuk GIF loop.
6. Pilih satu dari delapan kartu gaya, lalu tekan **Buat GIF & Frame**. Model memproses kelima foto berurutan di thread latar belakang; UI menampilkan tahap dan progres perkiraan.
7. Setelah selesai, pengguna melihat/mengunduh GIF loop berisi 10 frame before/after teracak dan JPG strip berisi tiga foto pilihan.

## Keputusan produk

- Bahasa UI: Indonesia; nama gaya koleksi dipertahankan agar mudah dicocokkan dengan aset.
- Arah visual: studio editorial hangat, latar krem, tinta hijau gelap, aksen terakota, tipografi serif untuk headline dan sans-serif untuk kontrol.
- Breakpoint utama: desktop dua kolom (kamera + pemilih style), ponsel satu kolom.
- Kursor dikendalikan ujung telunjuk lewat MediaPipe Hands. Mencubit ibu jari dan telunjuk menjalankan klik; saat cubitan, posisi kursor dikunci agar klik tidak bergeser mengikuti jari yang bergerak. Zona gerak aktif memakai 85% dari rentang gerak sebelumnya dan dipetakan ke seluruh halaman agar tombol tepi lebih mudah dijangkau. Indikator tangan hanya tampil sebagai lingkaran di atas halaman; jangan menggambar landmark/garis pada preview kamera. Overlay kursor harus berada di DOM sebelum skrip pelacakan dijalankan.
- Setelah foto dijepret/diunggah, preview foto menutupi video kamera yang tetap berjalan di belakangnya supaya MediaPipe tetap menerima frame untuk kontrol tangan.
- Style yang ditawarkan: Starry Night, Candy, Mosaic, Udnie, The Wave, Ben Giles, La CafÃ©, Flowers. Sumber gambar berada di `data/style-images/`.
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

