"""Small local web app for the neural style transfer photobooth."""
import os
import random
import subprocess
import threading
import time
import uuid
import zipfile

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageFont, ImageOps
from werkzeug.utils import secure_filename

import gdrive_uploader
import neural_style_transfer as nst


ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_DIR = os.path.join(ROOT, 'data', 'style-images')
JOB_DIR = os.path.join(ROOT, 'data', 'photobooth')
MUSIC_DIR = os.path.join(ROOT, 'music')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
STYLES = [
    {'id': 'vg_starry_night', 'name': 'Starry Night', 'mood': 'Swirling, luminous brushstrokes'},
    {'id': 'candy', 'name': 'Candy', 'mood': 'Bold color and playful texture'},
    {'id': 'mosaic', 'name': 'Mosaic', 'mood': 'Colorful, tiled geometry'},
    {'id': 'udnie', 'name': 'Udnie', 'mood': 'Energetic modernist shapes'},
    {'id': 'wave_crop', 'name': 'The Wave', 'mood': 'Rolling blues and expressive lines'},
    {'id': 'ben_giles', 'name': 'Ben Giles', 'mood': 'Dreamlike collage colors'},
    {'id': 'vg_la_cafe', 'name': 'La Café', 'mood': 'Warm post-impressionist city light'},
    {'id': 'flowers_crop', 'name': 'Flowers', 'mood': 'Soft floral color fields'},
    {'id': 'batik', 'name': 'Batik', 'mood': 'Cream, teal, cobalt and terracotta motifs'},
    {'id': 'contoh_hijau', 'name': 'Batik 2', 'mood': 'Green batik with cream linework'},
]
JOBS = {}
JOBS_LOCK = threading.Lock()
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 60 * 1024 * 1024
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS


def _font(size, bold=False):
    font_names = ['arialbd.ttf', 'arial.ttf'] if bold else ['arial.ttf']
    font_paths = [os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts', name)
                  for name in font_names]
    font_paths += ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
                   else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    for font_path in font_paths:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _cover_image(image, size):
    return ImageOps.fit(image.convert('RGB'), size, method=LANCZOS)


def _green_frame_slots(frame_image):
    """Find the three green photo placeholders in the supplied frame artwork."""
    pixels = frame_image.convert('RGB')
    width, height = pixels.size
    green_rows = []
    for y in range(height):
        count = sum(1 for x in range(width)
                    if (lambda rgb: rgb[1] > 140 and rgb[1] > rgb[0] * 1.35
                        and rgb[1] > rgb[2] * 1.2)(pixels.getpixel((x, y))))
        if count > width * 0.25:
            green_rows.append(y)

    row_groups = []
    for y in green_rows:
        if not row_groups or y > row_groups[-1][-1] + 1:
            row_groups.append([y])
        else:
            row_groups[-1].append(y)

    slots = []
    for rows in row_groups:
        ys = range(rows[0], rows[-1] + 1)
        xs = [x for y in ys for x in range(width)
              if (lambda rgb: rgb[1] > 140 and rgb[1] > rgb[0] * 1.35
                  and rgb[1] > rgb[2] * 1.2)(pixels.getpixel((x, y)))]
        if xs:
            slots.append((min(xs), rows[0], max(xs) + 1, rows[-1] + 1))
    if len(slots) != 3:
        raise RuntimeError('Frame harus memiliki tepat 3 area foto berwarna hijau.')
    return slots


def _make_music_video(job_path, gif_path):
    music_paths = []
    for folder, _, filenames in os.walk(MUSIC_DIR):
        music_paths.extend(os.path.join(folder, name) for name in filenames
                           if name.lower().endswith(('.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac')))
    if not music_paths:
        raise RuntimeError('Tidak ada file musik yang didukung di folder music/.')

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError('Dependensi imageio-ffmpeg belum terpasang. Jalankan pip install -r requirements-web.txt.') from exc

    music_path = random.choice(music_paths)
    mp4_path = os.path.join(job_path, 'studio-sela-musik.mp4')
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-ignore_loop', '0', '-i', gif_path,
        '-i', music_path,
        '-t', '15', '-map', '0:v:0', '-map', '1:a:0',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', mp4_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if exc.stderr else 'FFmpeg gagal membuat video.'
        raise RuntimeError('Gagal membuat MP4: {}'.format(detail[-500:])) from exc
    return mp4_path


_REMBG_SESSION = None


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is None:
        os.environ['U2NET_HOME'] = os.path.join(ROOT, 'models', 'u2net')
        import rembg
        _REMBG_SESSION = rembg.new_session('u2net')
    return _REMBG_SESSION


def _make_outputs(job_path, original_paths, styled_paths, selected_indices, guest_name, style=None):
    # Mix all before/after frames together; do not preserve capture or render order.
    gif_paths = list(original_paths) + list(styled_paths)
    random.shuffle(gif_paths)
    gif_frames = [_cover_image(Image.open(path), (480, 360)) for path in gif_paths]
    gif_path = os.path.join(job_path, 'studio-sela-loop.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=650, loop=0, optimize=True)
    mp4_path = _make_music_video(job_path, gif_path)

    frame_path = os.path.join(ROOT, 'frame', '1.png')
    if not os.path.isfile(frame_path):
        raise RuntimeError('Template frame tidak ditemukan: frame/1.png')
    source_frame = Image.open(frame_path).convert('RGB')
    slots = _green_frame_slots(source_frame)

    # 1. Frame dengan Background
    frame_bg = source_frame.copy()
    frame_pixels = source_frame.load()
    for photo_index, (left, top, right, bottom) in zip(selected_indices, slots):
        photo = _cover_image(Image.open(styled_paths[photo_index]),
                             (right - left, bottom - top))
        photo_pixels = photo.load()
        for y in range(top, bottom):
            for x in range(left, right):
                r, g, b = frame_pixels[x, y]
                if g > 140 and g > r * 1.35 and g > b * 1.2:
                    frame_bg.putpixel((x, y), photo_pixels[x - left, y - top])
    jpg_bg_path = os.path.join(job_path, 'studio-sela-frame.jpg')
    frame_bg.save(jpg_bg_path, quality=94, optimize=True)

    # 2. Frame latar tema (background foto subjek dihapus & diganti latar tema yang dipilih)
    import rembg
    rembg_sess = _get_rembg_session()

    theme_bg_img = None
    if style and isinstance(style, dict) and style.get('file'):
        style_img_path = os.path.join(STYLE_DIR, style['file'])
        if os.path.isfile(style_img_path):
            try:
                theme_bg_img = Image.open(style_img_path).convert('RGB')
            except Exception:
                theme_bg_img = None

    frame_nobg = source_frame.copy()
    for photo_index, (left, top, right, bottom) in zip(selected_indices, slots):
        slot_w, slot_h = right - left, bottom - top

        # Siapkan latar belakang tema yang dipilih (atau latar bersih jika tema belum termuat)
        if theme_bg_img is not None:
            slot_img = _cover_image(theme_bg_img, (slot_w, slot_h))
        else:
            slot_img = Image.new('RGB', (slot_w, slot_h), (255, 255, 255))

        orig_photo = _cover_image(Image.open(original_paths[photo_index]), (slot_w, slot_h))
        styled_photo = _cover_image(Image.open(styled_paths[photo_index]), (slot_w, slot_h))

        # Deteksi subjek dari foto asli agar potongan tepi rambut & tubuh sangat presisi
        try:
            orig_cutout = rembg.remove(orig_photo, session=rembg_sess)
            alpha_mask = orig_cutout.split()[-1]
            styled_cutout = styled_photo.convert('RGBA')
            styled_cutout.putalpha(alpha_mask)
        except Exception:
            styled_cutout = rembg.remove(styled_photo, session=rembg_sess)

        # Tempelkan subjek di atas latar belakang tema yang dipilih
        slot_img.paste(styled_cutout, (0, 0), styled_cutout)
        slot_pixels = slot_img.load()
        for y in range(top, bottom):
            for x in range(left, right):
                r, g, b = frame_pixels[x, y]
                if g > 140 and g > r * 1.35 and g > b * 1.2:
                    frame_nobg.putpixel((x, y), slot_pixels[x - left, y - top])
    jpg_nobg_path = os.path.join(job_path, 'studio-sela-frame-nobg.jpg')
    frame_nobg.save(jpg_nobg_path, quality=94, optimize=True)

    # Package GIF, music video, kedua frame JPG, dan seluruh foto ke file ZIP
    zip_path = os.path.join(job_path, 'studio-sela-semua-karya.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(gif_path, arcname='studio-sela-loop.gif')
        zipf.write(mp4_path, arcname='studio-sela-musik.mp4')
        zipf.write(jpg_bg_path, arcname='studio-sela-frame-dengan-bg.jpg')
        zipf.write(jpg_nobg_path, arcname='studio-sela-frame-latar-tema.jpg')
        # 5 foto asli
        for idx, orig_path in enumerate(original_paths):
            ext = os.path.splitext(orig_path)[1] or '.jpg'
            zipf.write(orig_path, arcname=f'foto_asli_{idx + 1:02d}{ext}')
        # 5 foto hasil style (3 foto dengan frame dan 2 foto tanpa frame)
        for idx, st_path in enumerate(styled_paths):
            ext = os.path.splitext(st_path)[1] or '.jpg'
            in_frame = "dengan_frame" if idx in selected_indices else "tanpa_frame"
            zipf.write(st_path, arcname=f'foto_style_{idx + 1:02d}_{in_frame}{ext}')

    return gif_path, mp4_path, jpg_bg_path, jpg_nobg_path, zip_path


def run_job(job_id, image_paths, selected_indices, style, guest_name):
    job_path = os.path.join(JOB_DIR, job_id)
    processed_dir = os.path.join(job_path, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    iterations = 200
    try:
        styled_paths = []
        style_extension = os.path.splitext(style['file'])[1]
        for index, image_path in enumerate(image_paths):
            photo_name = 'portrait_{:02d}'.format(index + 1)
            config = {
                'content_img_name': os.path.basename(image_path),
                'style_img_name': style['id'] + style_extension,
                'content_images_dir': os.path.dirname(image_path),
                'style_images_dir': STYLE_DIR,
                'output_img_dir': processed_dir,
                'img_format': (4, '.jpg'),
                'height': 320,
                'content_weight': 1e5,
                'style_weight': 3e4,
                'tv_weight': 1e0,
                'optimizer': 'lbfgs',
                'model': 'vgg19',
                'init_method': 'content',
                'saving_freq': -1,
                'iterations': iterations,
                'progress_callback': lambda phase, pct, idx=index: _update_progress(
                    job_id, 'Foto {} dari 5 · {}'.format(idx + 1, phase),
                    8 + int(86 * (idx + pct / 100.0) / 5)),
            }
            result_dir = nst.neural_style_transfer(config)
            result_names = [name for name in os.listdir(result_dir)
                            if name.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not result_names:
                raise RuntimeError('Foto {} selesai tanpa menghasilkan gambar.'.format(index + 1))
            styled_paths.append(os.path.join(result_dir, result_names[0]))

        _update_progress(job_id, 'Menyusun GIF, MP4 musik, 2 frame foto, dan paket unduhan...', 95)
        _make_outputs(job_path, image_paths, styled_paths, selected_indices, guest_name, style=style)
        with JOBS_LOCK:
            JOBS[job_id].update(status='done', gif='studio-sela-loop.gif',
                                mp4='studio-sela-musik.mp4',
                                frame='studio-sela-frame.jpg',
                                frame_nobg='studio-sela-frame-nobg.jpg',
                                zip='studio-sela-semua-karya.zip',
                                guest_name=guest_name,
                                progress=100,
                                phase='Selesai')
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(status='error', error=str(exc))


def _update_progress(job_id, phase, pct):
    with JOBS_LOCK:
        if job_id in JOBS and JOBS[job_id].get('status') == 'processing':
            JOBS[job_id].update(progress=int(min(99, max(0, pct))), phase=phase)


@app.get('/')
def index():
    return render_template('photobooth.html', styles=STYLES)


@app.get('/styles/<path:filename>')
def style_image(filename):
    return send_from_directory(STYLE_DIR, filename)


@app.get('/api/sibi/model')
def sibi_model_weights():
    model_path = os.path.join(ROOT, 'models', 'sibi_landmark_weights.json')
    if not os.path.isfile(model_path):
        return jsonify(error='Model SIBI belum dilatih.'), 404
    return send_from_directory(os.path.dirname(model_path), os.path.basename(model_path))


@app.post('/api/captures')
def create_capture():
    images = request.files.getlist('photos')
    style_id = request.form.get('style')
    selected_value = request.form.get('selected', '')
    guest_name = ' '.join(request.form.get('guest_name', '').split())[:32]
    style = next((item for item in STYLES if item['id'] == style_id), None)
    if len(images) != 5 or any(not image.filename for image in images):
        return jsonify(error='Sesi harus berisi tepat 5 foto.'), 400
    if style is None:
        return jsonify(error='Pilih salah satu style yang tersedia.'), 400
    if not guest_name:
        return jsonify(error='Isi nama sebelum membuat hasil photobooth.'), 400
    try:
        selected_indices = [int(value) for value in selected_value.split(',')]
    except ValueError:
        selected_indices = []
    if len(selected_indices) != 3 or len(set(selected_indices)) != 3 or any(
            index < 0 or index >= 5 for index in selected_indices):
        return jsonify(error='Pilih tepat 3 foto untuk frame.'), 400
    job_id = uuid.uuid4().hex
    job_path = os.path.join(JOB_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)
    image_paths = []
    for index, image in enumerate(images):
        extension = os.path.splitext(secure_filename(image.filename))[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            return jsonify(error='Gunakan gambar JPG, PNG, atau WebP.'), 400
        image_path = os.path.join(job_path, 'capture_{:02d}{}'.format(index + 1, extension))
        image.save(image_path)
        image_paths.append(image_path)
    style['file'] = next(name for name in os.listdir(STYLE_DIR)
                         if os.path.splitext(name)[0] == style['id'])
    with JOBS_LOCK:
        JOBS[job_id] = {
            'status': 'processing', 'style': style['name'], 'progress': 0,
            'phase': 'Menyiapkan job...',
        }
    threading.Thread(target=run_job,
                     args=(job_id, image_paths, selected_indices, style, guest_name),
                     daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.get('/api/jobs/<job_id>')
def job_status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error='Sesi photobooth tidak ditemukan.'), 404
        response = dict(job)
    if response.get('status') == 'done':
        response['gif_url'] = '/results/{}/{}'.format(job_id, response['gif'])
        response['mp4_url'] = '/results/{}/{}'.format(job_id, response['mp4'])
        response['frame_url'] = '/results/{}/{}'.format(job_id, response['frame'])
        if 'frame_nobg' in response:
            response['frame_nobg_url'] = '/results/{}/{}'.format(job_id, response['frame_nobg'])
        if 'zip' in response:
            response['zip_url'] = '/results/{}/{}'.format(job_id, response['zip'])
        if 'drive_url' in response:
            response['drive_url'] = response['drive_url']
            response['qr_base64'] = response.get('qr_base64')
    return jsonify(response)


@app.post('/api/jobs/<job_id>/upload-drive')
def upload_drive(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error='Sesi photobooth tidak ditemukan.'), 404
        if job.get('status') != 'done':
            return jsonify(error='Hasil foto belum siap diunggah.'), 409
        guest_name = job.get('guest_name', 'Tamu')
        # Return cached drive url if already uploaded
        if job.get('drive_url') and job.get('qr_base64'):
            return jsonify({
                'success': True,
                'needs_setup': False,
                'drive_url': job['drive_url'],
                'qr_base64': job['qr_base64']
            })

    job_path = os.path.join(JOB_DIR, job_id)
    result = gdrive_uploader.upload_job_to_drive(job_id, job_path, guest_name)
    with JOBS_LOCK:
        if job_id in JOBS and result.get('success'):
            JOBS[job_id]['drive_url'] = result['drive_url']
            JOBS[job_id]['qr_base64'] = result['qr_base64']
    return jsonify(result), (200 if result.get('success') or result.get('needs_setup') else 502)


@app.get('/results/<job_id>/<path:filename>')
def result_image(job_id, filename):
    folder = os.path.join(JOB_DIR, job_id)
    target_path = os.path.join(folder, filename)
    is_attachment = filename.lower().endswith('.zip')
    if os.path.isfile(target_path):
        return send_from_directory(os.path.dirname(target_path), os.path.basename(target_path), as_attachment=is_attachment)
    # Fallback search if filename was requested without its subfolder
    for root, _, files in os.walk(folder):
        if filename in files:
            return send_from_directory(root, filename, as_attachment=is_attachment)
    return jsonify(error='Gambar hasil tidak ditemukan.'), 404


def warmup_models():
    """Load model weights in the background; style features load only when requested."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Pre-warming model and styles on {device}...")
        nst.get_or_prepare_model('vgg19', device)
        _get_rembg_session()
        print("[*] Model & rembg ready. Style features will be prepared when a photo is submitted.")
    except Exception as err:
        print(f"[*] Pre-warm notice: {err}")


if __name__ == '__main__':
    os.makedirs(JOB_DIR, exist_ok=True)
    threading.Thread(target=warmup_models, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
