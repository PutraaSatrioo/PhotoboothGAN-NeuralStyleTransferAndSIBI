"""Local photobooth app with classical neural style transfer."""
import os
import random
import base64
import json
import subprocess
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from requests.adapters import HTTPAdapter

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageFont, ImageOps
from werkzeug.utils import secure_filename

import gdrive_uploader
import ig_uploader
import neural_style_transfer as nst


ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_local_env():
    """Load KEY=value entries from .env without overriding process environment."""
    env_path = os.path.join(ROOT, '.env')
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding='utf-8') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()
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
app.config['TEMPLATES_AUTO_RELOAD'] = True
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
    import numpy as np

    arr = np.asarray(frame_image.convert('RGB'), dtype=np.float64)
    width = arr.shape[1]
    green_mask = (arr[:, :, 1] > 140) & (arr[:, :, 1] > arr[:, :, 0] * 1.35) & (
        arr[:, :, 1] > arr[:, :, 2] * 1.2
    )
    green_rows = np.flatnonzero(green_mask.sum(axis=1) > width * 0.25)

    row_groups = []
    for y in green_rows.tolist():
        if not row_groups or y > row_groups[-1][-1] + 1:
            row_groups.append([y])
        else:
            row_groups[-1].append(y)

    slots = []
    for rows in row_groups:
        xs = np.flatnonzero(green_mask[rows[0]:rows[-1] + 1].any(axis=0))
        if xs.size:
            slots.append((int(xs[0]), int(rows[0]), int(xs[-1]) + 1, int(rows[-1]) + 1))
    if len(slots) != 3:
        raise RuntimeError('Frame harus memiliki tepat 3 area foto berwarna hijau.')
    return slots


_FRAME_TEMPLATE_CACHE = None
_FRAME_TEMPLATE_LOCK = threading.Lock()


def _get_frame_template():
    """Load and cache frame/1.png, its 3 slots, and binary Pillow masks."""
    global _FRAME_TEMPLATE_CACHE
    with _FRAME_TEMPLATE_LOCK:
        if _FRAME_TEMPLATE_CACHE is None:
            import numpy as np

            frame_path = os.path.join(ROOT, 'frame', '1.png')
            if not os.path.isfile(frame_path):
                raise RuntimeError('Template frame tidak ditemukan: frame/1.png')
            source_frame = Image.open(frame_path).convert('RGB')
            slots = _green_frame_slots(source_frame)
            arr = np.asarray(source_frame, dtype=np.float64)
            green_mask = (arr[:, :, 1] > 140) & (arr[:, :, 1] > arr[:, :, 0] * 1.35) & (
                arr[:, :, 1] > arr[:, :, 2] * 1.2
            )
            slot_masks = [
                Image.fromarray((green_mask[top:bottom, left:right] * 255).astype(np.uint8), mode='L')
                for left, top, right, bottom in slots
            ]
            _FRAME_TEMPLATE_CACHE = (source_frame, slots, slot_masks)
        return _FRAME_TEMPLATE_CACHE


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
        _REMBG_SESSION = rembg.new_session('u2net', providers=['CPUExecutionProvider'])
    return _REMBG_SESSION



_HTTP_SESSION = None
_HTTP_SESSION_LOCK = threading.Lock()


def _get_http_session():
    global _HTTP_SESSION
    with _HTTP_SESSION_LOCK:
        if _HTTP_SESSION is None:
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
            session.mount('https://', adapter)
            session.mount('http://', adapter)
            _HTTP_SESSION = session
        return _HTTP_SESSION


def _encode_pil_data_url(image, quality=None):
    if quality is None:
        quality = int(os.environ.get('IMAGE_API_JPEG_QUALITY', '92'))
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded


def _encode_image_data_url(path, max_dim=None, quality=None):
    if max_dim is None:
        max_dim = int(os.environ.get('IMAGE_API_MAX_INPUT', '1024'))
    with Image.open(path) as image:
        contained = ImageOps.contain(image.convert('RGB'), (max_dim, max_dim), method=LANCZOS)
    return _encode_pil_data_url(contained, quality=quality)


def _call_image_api(prompt, source_data_url, style_data_url, out_width, out_height):
    api_key = os.environ.get('IMAGE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('IMAGE_API_KEY belum diisi di file .env.')
    endpoint = os.environ.get('IMAGE_API_URL', 'https://bifrost.hcm-lab.id/v1/chat/completions').strip()
    model = os.environ.get('IMAGE_MODEL', 'Qwen/Qwen-Image-2.1').strip()
    timeout = int(os.environ.get('IMAGE_API_TIMEOUT', '180'))

    extra_body = {'height': out_height, 'width': out_width}
    steps_env = os.environ.get('IMAGE_API_STEPS', '8').strip()
    if steps_env and int(steps_env) > 0:
        extra_body['num_inference_steps'] = int(steps_env)

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': source_data_url}},
            {'type': 'image_url', 'image_url': {'url': style_data_url}},
        ]}],
        'extra_body': extra_body,
    }
    session = _get_http_session()
    max_retries = 2
    result = None
    for attempt in range(max_retries + 1):
        try:
            response = session.post(
                endpoint,
                json=payload,
                headers={'Content-Type': 'application/json', 'x-api-key': api_key},
                timeout=timeout,
            )
            response.raise_for_status()
            result = response.json()
            break
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if code in (502, 503, 504) and attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            detail = exc.response.text[:600] if exc.response is not None else str(exc)
            raise RuntimeError('Image API HTTP {}: {}'.format(code or 'Error', detail)) from exc
        except requests.RequestException as exc:
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError('Tidak dapat menghubungi Image API: {}'.format(exc)) from exc
    try:
        content = result['choices'][0]['message']['content']
        parts = content if isinstance(content, list) else [{'type': 'text', 'text': content}]
        image_url = None
        for part in parts:
            if isinstance(part, dict) and (part.get('image_url') or part.get('image')):
                value = part.get('image_url') or part.get('image')
                image_url = value.get('url') if isinstance(value, dict) else value
                break
            text_value = part.get('text', '') if isinstance(part, dict) else ''
            if text_value.startswith(('data:image/', 'https://', 'http://')):
                image_url = text_value.strip()
                break
        if not image_url:
            raise ValueError('Response tidak berisi image_url.')
        if image_url.startswith('data:image/'):
            image_bytes = base64.b64decode(image_url.split(',', 1)[1])
        else:
            image_response = session.get(image_url, timeout=60)
            image_response.raise_for_status()
            image_bytes = image_response.content
        with Image.open(BytesIO(image_bytes)) as generated:
            generated.verify()
        return image_bytes
    except (KeyError, IndexError, TypeError, ValueError, requests.RequestException) as exc:
        raise RuntimeError('Format response Image API tidak dikenali: {}'.format(exc)) from exc


def _generate_image_via_api(image_path, style, output_path, style_data_url=None):
    out_width = int(os.environ.get('IMAGE_API_WIDTH', '512'))
    out_height = int(os.environ.get('IMAGE_API_HEIGHT', '384'))
    source_data_url = _encode_image_data_url(image_path)
    if style_data_url is None:
        style_data_url = _encode_image_data_url(os.path.join(STYLE_DIR, style['file']))

    prompt = (
        'Strictly apply strong neural style transfer from Image 2 (the style artwork) onto Image 1 (the photograph). '
        'Repaint 100% of Image 1—including the subject, face shading, hair, clothing, and entire background—using the '
        'exact color palette, brushwork, pattern motifs, surface texture, and artistic medium visible in Image 2. '
        'Do NOT leave Image 1 looking like a normal photograph and do NOT invent a generic style; match Image 2\'s visual '
        'texture and palette faithfully. Keep only the person\'s identity, facial features, pose, and composition from Image 1. '
        'Style reference: {} ({}). Return only the fully stylized image.'
    ).format(style['name'], style.get('mood', style['name']))
    image_bytes = _call_image_api(prompt, source_data_url, style_data_url, out_width, out_height)
    with open(output_path, 'wb') as output:
        output.write(image_bytes)


def _generate_batch_grid_via_api(image_paths, style, output_paths, style_data_url=None):
    """Combine up to 6 photos into a single 3x2 grid for 1 cloud render, then split back."""
    count = len(image_paths)
    if count == 0 or count > 6:
        raise ValueError('Grid batch hanya mendukung 1 sampai 6 foto.')

    grid_w = int(os.environ.get('IMAGE_API_GRID_WIDTH', '1280'))
    grid_h = int(os.environ.get('IMAGE_API_GRID_HEIGHT', '640'))
    cols = 3
    rows = 2 if count > 3 else 1
    pad = 8
    tile_w = (grid_w - pad * (cols + 1)) // cols
    tile_h = (grid_h - pad * (rows + 1)) // rows
    pad_x = max(4, (grid_w - tile_w * cols) // (cols + 1))
    pad_y = max(4, (grid_h - tile_h * rows) // (rows + 1))

    grid_img = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    boxes = []
    for idx, image_path in enumerate(image_paths):
        col = idx % cols
        row = idx // cols
        x0 = pad_x + col * (tile_w + pad_x)
        y0 = pad_y + row * (tile_h + pad_y)
        with Image.open(image_path) as src:
            tile = _cover_image(src, (tile_w, tile_h))
        grid_img.paste(tile, (x0, y0))
        boxes.append((x0, y0, x0 + tile_w, y0 + tile_h))

    source_data_url = _encode_pil_data_url(grid_img)
    if style_data_url is None:
        style_data_url = _encode_image_data_url(os.path.join(STYLE_DIR, style['file']))

    prompt = (
        'Image 1 is a collage containing {} separate photographs inside white grid borders. '
        'Image 2 is the style reference ({}: {}). '
        'Apply strong neural style transfer from Image 2 onto Image 1 while strictly preserving every subject, '
        'object, face, outline, position, and white grid border of all {} photos in Image 1. '
        'Repaint the surfaces, colors, and textures of all {} photos to strongly match Image 2 without removing, '
        'shifting, or merging the {} photos.'
    ).format(count, style['name'], style.get('mood', style['name']), count, count, count)

    image_bytes = _call_image_api(prompt, source_data_url, style_data_url, grid_w, grid_h)
    with Image.open(BytesIO(image_bytes)) as rendered:
        rendered_rgb = rendered.convert('RGB')
        if rendered_rgb.size != (grid_w, grid_h):
            rendered_rgb = rendered_rgb.resize((grid_w, grid_h), resample=LANCZOS)

    inset = 4
    for (x0, y0, x1, y1), output_path in zip(boxes, output_paths):
        crop_box = (x0 + inset, y0 + inset, max(x0 + inset + 1, x1 - inset), max(y0 + inset + 1, y1 - inset))
        rendered_rgb.crop(crop_box).save(output_path, format='PNG')


def _precompute_subject_masks(original_paths, selected_indices):
    """Compute rembg alpha masks for selected slots on CPU while cloud rendering runs."""
    import rembg

    _, slots, _ = _get_frame_template()
    rembg_sess = _get_rembg_session()
    masks = {}
    for photo_index, (left, top, right, bottom) in zip(selected_indices, slots):
        slot_size = (right - left, bottom - top)
        orig_photo = _cover_image(Image.open(original_paths[photo_index]), slot_size)
        orig_cutout = rembg.remove(orig_photo, session=rembg_sess)
        masks[photo_index] = orig_cutout.split()[-1]
    return masks


def _make_outputs(job_path, original_paths, styled_paths, selected_indices, guest_name, style=None, precomputed_masks=None):
    # Show each original immediately before its stylized result, in capture order.
    gif_paths = [path for pair in zip(original_paths, styled_paths) for path in pair]
    gif_frames = [_cover_image(Image.open(path), (480, 360)) for path in gif_paths]
    gif_path = os.path.join(job_path, 'studio-sela-loop.gif')
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:],
                       duration=650, loop=0, optimize=True)
    mp4_path = _make_music_video(job_path, gif_path)

    source_frame, slots, slot_masks = _get_frame_template()

    # 1. Frame dengan Background
    frame_bg = source_frame.copy()
    for photo_index, (left, top, right, bottom), mask_img in zip(selected_indices, slots, slot_masks):
        photo = _cover_image(Image.open(styled_paths[photo_index]),
                             (right - left, bottom - top))
        frame_bg.paste(photo, (left, top), mask_img)
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
    for photo_index, (left, top, right, bottom), mask_img in zip(selected_indices, slots, slot_masks):
        slot_w, slot_h = right - left, bottom - top

        # Siapkan latar belakang tema yang dipilih (atau latar bersih jika tema belum termuat)
        if theme_bg_img is not None:
            slot_img = _cover_image(theme_bg_img, (slot_w, slot_h))
        else:
            slot_img = Image.new('RGB', (slot_w, slot_h), (255, 255, 255))

        styled_photo = _cover_image(Image.open(styled_paths[photo_index]), (slot_w, slot_h))

        # Deteksi subjek dari foto asli agar potongan tepi rambut & tubuh sangat presisi
        try:
            if precomputed_masks and photo_index in precomputed_masks:
                alpha_mask = precomputed_masks[photo_index]
            else:
                orig_photo = _cover_image(Image.open(original_paths[photo_index]), (slot_w, slot_h))
                orig_cutout = rembg.remove(orig_photo, session=rembg_sess)
                alpha_mask = orig_cutout.split()[-1]
            styled_cutout = styled_photo.convert('RGBA')
            styled_cutout.putalpha(alpha_mask)
        except Exception:
            styled_cutout = rembg.remove(styled_photo, session=rembg_sess)

        # Tempelkan subjek di atas latar belakang tema yang dipilih
        slot_img.paste(styled_cutout, (0, 0), styled_cutout)
        frame_nobg.paste(slot_img, (left, top), mask_img)
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
    try:
        styled_paths = []
        precomputed_masks = None
        engine = os.environ.get('PHOTOBOOTH_ENGINE', 'qwen').strip().lower()
        if engine == 'qwen':
            total = len(image_paths)
            styled_paths = [os.path.join(processed_dir, 'portrait_{:02d}.png'.format(i + 1)) for i in range(total)]
            style_data_url = _encode_image_data_url(os.path.join(STYLE_DIR, style['file']))
            api_mode = os.environ.get('IMAGE_API_MODE', 'grid').strip().lower()
            if api_mode == 'grid' and 1 <= total <= 6:
                _update_progress(job_id, 'Mengirim {} foto sekaligus dalam 1 render server Qwen...'.format(total), 15)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    mask_future = executor.submit(_precompute_subject_masks, image_paths, selected_indices)
                    grid_future = executor.submit(
                        _generate_batch_grid_via_api, image_paths, style, styled_paths, style_data_url
                    )
                    grid_future.result()
                    _update_progress(job_id, 'Server Qwen selesai! Memisahkan 5 foto hasil...', 90)
                    try:
                        precomputed_masks = mask_future.result()
                    except Exception:
                        precomputed_masks = None
            else:
                styled_paths = [None] * total
                max_workers = max(1, min(total, int(os.environ.get('IMAGE_API_WORKERS', '2'))))
                _update_progress(job_id, 'Mengirim foto ke server Qwen (1/{})...'.format(total), 10)
                completed = 0
                with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
                    mask_future = executor.submit(_precompute_subject_masks, image_paths, selected_indices)
                    future_to_idx = {}
                    for index, image_path in enumerate(image_paths):
                        output_path = os.path.join(processed_dir, 'portrait_{:02d}.png'.format(index + 1))
                        fut = executor.submit(_generate_image_via_api, image_path, style, output_path, style_data_url)
                        future_to_idx[fut] = (index, output_path)
                    for fut in as_completed(future_to_idx):
                        idx, out_path = future_to_idx[fut]
                        fut.result()
                        styled_paths[idx] = out_path
                        completed += 1
                        _update_progress(
                            job_id,
                            'Server Qwen selesai mengolah {} dari {} foto'.format(completed, total),
                            10 + int(84 * completed / total),
                        )
                    try:
                        precomputed_masks = mask_future.result()
                    except Exception:
                        precomputed_masks = None
        elif engine == 'gatys_cpu':
            for index, image_path in enumerate(image_paths):
                config = {
                    'content_img_name': os.path.basename(image_path),
                    'style_img_name': style['file'],
                    'content_images_dir': os.path.dirname(image_path),
                    'style_images_dir': STYLE_DIR,
                    'output_img_dir': processed_dir,
                    'img_format': (4, '.jpg'),
                    'height': int(os.environ.get('NST_HEIGHT', '320')),
                    'content_weight': float(os.environ.get('NST_CONTENT_WEIGHT', '100000')),
                    'style_weight': float(os.environ.get('NST_STYLE_WEIGHT', '30000')),
                    'tv_weight': 1.0,
                    'optimizer': 'lbfgs',
                    'model': 'vgg19',
                    'init_method': 'content',
                    'saving_freq': -1,
                    'iterations': int(os.environ.get('NST_ITERATIONS', '200')),
                    'device': 'cpu',
                    'progress_callback': lambda phase, pct, idx=index: _update_progress(
                        job_id, 'Foto {} dari 5 · {}'.format(idx + 1, phase),
                        8 + int(86 * (idx + pct / 100.0) / 5)),
                }
                result_dir = nst.neural_style_transfer(config)
                result_names = sorted(name for name in os.listdir(result_dir)
                                      if name.lower().endswith(('.jpg', '.jpeg', '.png')))
                if not result_names:
                    raise RuntimeError('Foto {} selesai tanpa menghasilkan gambar.'.format(index + 1))
                output_path = os.path.join(processed_dir, 'portrait_{:02d}.jpg'.format(index + 1))
                with Image.open(os.path.join(result_dir, result_names[-1])) as result_image:
                    result_image.convert('RGB').save(output_path, quality=94, optimize=True)
                styled_paths.append(output_path)
                _update_progress(job_id, 'Foto {} dari 5 selesai diproses'.format(index + 1),
                                 8 + int(86 * (index + 1) / 5))
        else:
            raise RuntimeError('PHOTOBOOTH_ENGINE harus gatys_cpu atau qwen.')

        _update_progress(job_id, 'Menyusun GIF, MP4 musik, 2 frame foto, dan paket unduhan...', 95)
        _make_outputs(job_path, image_paths, styled_paths, selected_indices, guest_name,
                      style=style, precomputed_masks=precomputed_masks)
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


@app.post('/api/jobs/<job_id>/upload-ig')
def upload_ig(job_id):
    if os.environ.get('IG_UPLOAD_ENABLED', 'false').strip().lower() not in ('1', 'true', 'yes'):
        return jsonify(success=False, error='Upload Instagram sedang dinonaktifkan.'), 403
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error='Sesi photobooth tidak ditemukan.'), 404
        if job.get('status') != 'done':
            return jsonify(error='Hasil foto belum siap diunggah.'), 409
        # Return cached IG result if already uploaded
        if job.get('ig_url') and job.get('ig_qr_base64'):
            return jsonify({
                'success': True,
                'ig_url': job['ig_url'],
                'qr_base64': job['ig_qr_base64']
            })

    job_path = os.path.join(JOB_DIR, job_id)
    frame_path = os.path.join(job_path, 'studio-sela-frame.jpg')
    result = ig_uploader.upload_story_to_ig(frame_path)
    with JOBS_LOCK:
        if job_id in JOBS and result.get('success'):
            JOBS[job_id]['ig_url'] = result['ig_url']
            JOBS[job_id]['ig_qr_base64'] = result['qr_base64']
    return jsonify(result), (200 if result.get('success') else 502)


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
    """Prepare NST and background removal without using the local GPU."""
    try:
        if os.environ.get('PHOTOBOOTH_ENGINE', 'qwen').strip().lower() == 'gatys_cpu':
            import torch
            print('[*] Preparing VGG19 for Gatys NST on CPU...')
            nst.get_or_prepare_model('vgg19', torch.device('cpu'))
        _get_rembg_session()
        print('[*] Models ready; local GPU is not used by photobooth.')
    except Exception as err:
        print(f"[*] Pre-warm notice: {err}")


if __name__ == '__main__':
    os.makedirs(JOB_DIR, exist_ok=True)
    threading.Thread(target=warmup_models, daemon=True).start()
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
