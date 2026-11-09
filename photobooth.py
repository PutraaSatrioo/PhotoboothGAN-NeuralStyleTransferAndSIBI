"""Small local web app for the neural style transfer photobooth."""
import os
import random
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont, ImageOps
from werkzeug.utils import secure_filename

import neural_style_transfer as nst


ROOT = os.path.dirname(os.path.abspath(__file__))
STYLE_DIR = os.path.join(ROOT, 'data', 'style-images')
JOB_DIR = os.path.join(ROOT, 'data', 'photobooth')
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


def _make_outputs(job_path, original_paths, styled_paths, selected_indices, guest_name):
    # Mix all before/after frames together; do not preserve capture or render order.
    gif_paths = list(original_paths) + list(styled_paths)
    random.shuffle(gif_paths)
    gif_frames = [_cover_image(Image.open(path), (480, 360)) for path in gif_paths]
    gif_path = os.path.join(job_path, 'studio-sela-loop.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=650, loop=0, optimize=True)

    width, margin, slot_h, gap, header_h, footer_h = 760, 48, 430, 26, 150, 130
    height = header_h + margin + 3 * slot_h + 2 * gap + footer_h
    frame = Image.new('RGB', (width, height), '#f5eee4')
    draw = ImageDraw.Draw(frame)
    name_font_size = 34
    title_font = _font(name_font_size, bold=True)
    while draw.textbbox((0, 0), guest_name, font=title_font)[2] > width - 2 * margin and name_font_size > 20:
        name_font_size -= 2
        title_font = _font(name_font_size, bold=True)
    caption_font = _font(18)
    footer_font = _font(16)
    draw.text((margin, 32), guest_name, fill='#272b25', font=title_font)
    draw.text((margin, 88), 'STUDIO SELA PHOTOBOOTH', fill='#db6847', font=caption_font)
    slot_w = width - 2 * margin
    for slot, photo_index in enumerate(selected_indices):
        top = header_h + slot * (slot_h + gap)
        draw.rectangle((margin - 8, top - 8, width - margin + 8, top + slot_h + 8),
                       fill='#fffdf9')
        image = _cover_image(Image.open(styled_paths[photo_index]), (slot_w, slot_h))
        frame.paste(image, (margin, top))
        draw.text((margin, top + slot_h + 8), '0{}'.format(slot + 1),
                  fill='#db6847', font=footer_font)
    draw.text((margin, height - 68), 'DIBUAT KHUSUS UNTUKMU  |  STUDIO SELA',
              fill='#64705a', font=footer_font)
    jpg_path = os.path.join(job_path, 'studio-sela-frame.jpg')
    frame.save(jpg_path, quality=94, optimize=True)
    return gif_path, jpg_path


def run_job(job_id, image_paths, selected_indices, style, guest_name):
    job_path = os.path.join(JOB_DIR, job_id)
    processed_dir = os.path.join(job_path, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    iterations = 80
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

        _update_progress(job_id, 'Menyusun GIF loop dan JPG frame...', 95)
        _make_outputs(job_path, image_paths, styled_paths, selected_indices, guest_name)
        with JOBS_LOCK:
            JOBS[job_id].update(status='done', gif='studio-sela-loop.gif',
                                frame='studio-sela-frame.jpg', progress=100,
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
        response['frame_url'] = '/results/{}/{}'.format(job_id, response['frame'])
    return jsonify(response)


@app.get('/results/<job_id>/<path:filename>')
def result_image(job_id, filename):
    folder = os.path.join(JOB_DIR, job_id)
    target_path = os.path.join(folder, filename)
    if os.path.isfile(target_path):
        return send_from_directory(os.path.dirname(target_path), os.path.basename(target_path))
    # Fallback search if filename was requested without its subfolder
    for root, _, files in os.walk(folder):
        if filename in files:
            return send_from_directory(root, filename)
    return jsonify(error='Gambar hasil tidak ditemukan.'), 404


def warmup_models():
    """Load model weights in the background; style features load only when requested."""
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Pre-warming model and styles on {device}...")
        nst.get_or_prepare_model('vgg19', device)
        print("[*] Model ready. Style features will be prepared when a photo is submitted.")
    except Exception as err:
        print(f"[*] Pre-warm notice: {err}")


if __name__ == '__main__':
    os.makedirs(JOB_DIR, exist_ok=True)
    threading.Thread(target=warmup_models, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
