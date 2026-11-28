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
JOB_DIR = '/tmp/photobooth' if os.environ.get('VERCEL') else os.path.join(ROOT, 'data', 'photobooth')
MUSIC_DIR = os.path.join(ROOT, 'music')
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
STYLES = [
    {
        'id': 'vg_starry_night',
        'name': 'Starry Night',
        'mood': 'Swirling, luminous brushstrokes',
        'prompt_desc': (
            'Vincent van Gogh Starry Night oil painting style, thick swirling impasto brushstrokes, '
            'deep cobalt blue, ultramarine, glowing golden yellow stars, and expressive post-impressionist canvas texture'
        ),
    },
    {
        'id': 'candy',
        'name': 'Candy',
        'mood': 'Bold color and playful texture',
        'prompt_desc': (
            'vibrant pop-art Candy oil painting style, bold saturated neon magenta, electric cyan, '
            'warm orange and sunny yellow palette knife strokes, playful expressive contemporary painting'
        ),
    },
    {
        'id': 'mosaic',
        'name': 'Mosaic',
        'mood': 'Colorful, tiled geometry',
        'prompt_desc': (
            'stained-glass Mosaic artwork style, intricate colorful geometric ceramic tiles, '
            'dark tesserae grout outlines, jewel-toned ruby, sapphire, emerald, and amber glass fragments'
        ),
    },
    {
        'id': 'udnie',
        'name': 'Udnie',
        'mood': 'Energetic modernist shapes',
        'prompt_desc': (
            'Francis Picabia Udnie cubist abstract modernist oil painting style, dynamic overlapping '
            'geometric planes, rhythmic ribbons of crimson, ochre, slate blue, and ivory brushwork'
        ),
    },
    {
        'id': 'wave_crop',
        'name': 'The Wave',
        'mood': 'Rolling blues and expressive lines',
        'prompt_desc': (
            'Hokusai The Great Wave off Kanagawa ukiyo-e Japanese woodblock print style, Prussian blue '
            'and indigo waves, crisp white seafoam claws, warm parchment tones, and traditional ink linework'
        ),
    },
    {
        'id': 'ben_giles',
        'name': 'Ben Giles',
        'mood': 'Dreamlike collage colors',
        'prompt_desc': (
            'Ben Giles surreal botanical collage art style, dreamlike vintage floral cutouts, '
            'vibrant pastel blossoms, butterflies, and whimsical layered paper-art textures'
        ),
    },
    {
        'id': 'vg_la_cafe',
        'name': 'La Café',
        'mood': 'Warm post-impressionist city light',
        'prompt_desc': (
            'Vincent van Gogh Cafe Terrace at Night oil painting style, glowing warm amber and sulfur-yellow '
            'lantern light contrasting with deep starry midnight blue cobblestones and thick post-impressionist brushwork'
        ),
    },
    {
        'id': 'flowers_crop',
        'name': 'Flowers',
        'mood': 'Soft floral color fields',
        'prompt_desc': (
            'impressionist blooming garden oil painting style, soft pastel rose pink, lavender, '
            'sage green, and warm sunlit floral color fields with delicate painterly brushstrokes'
        ),
    },
    {
        'id': 'batik',
        'name': 'Batik',
        'mood': 'Cream, teal, cobalt and terracotta motifs',
        'prompt_desc': (
            'traditional Indonesian Batik tulis textile art style, intricate hand-drawn canting wax-resist '
            'floral and parang motifs, rich cream, teal, cobalt blue, and terracotta fabric dye patterns'
        ),
    },
    {
        'id': 'contoh_hijau',
        'name': 'Batik 2',
        'mood': 'Green batik with cream linework',
        'prompt_desc': (
            'traditional Indonesian green Batik textile painting style, detailed emerald and forest green '
            'batik cloth adorned with fine cream and golden canting wax linework and ornamental Nusantara motifs'
        ),
    },
]
JOBS = {}
JOBS_LOCK = threading.Lock()
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, 'templates'),
    static_folder=os.path.join(ROOT, 'static'),
)
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
    if os.environ.get('MUSIC_VIDEO_ENABLED', 'false').strip().lower() not in ('1', 'true', 'yes'):
        return None
    music_paths = []
    if os.path.isdir(MUSIC_DIR):
        for folder, _, filenames in os.walk(MUSIC_DIR):
            music_paths.extend(os.path.join(folder, name) for name in filenames
                               if name.lower().endswith(('.wav', '.mp3', '.m4a', '.aac', '.ogg', '.flac')))
    if not music_paths:
        return None

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

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
        quality = int(os.environ.get('IMAGE_API_JPEG_QUALITY', '82'))
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format='JPEG', quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return 'data:image/jpeg;base64,' + encoded


def _encode_image_data_url(path, max_dim=None, quality=None):
    if max_dim is None:
        max_dim = int(os.environ.get('IMAGE_API_MAX_INPUT', '384'))
    with Image.open(path) as image:
        contained = ImageOps.contain(image.convert('RGB'), (max_dim, max_dim), method=LANCZOS)
    return _encode_pil_data_url(contained, quality=quality)


AI_MODELS = {
    'qwen': {
        'id': 'qwen',
        'label': 'Qwen Image 2.1',
        'model': os.environ.get('IMAGE_MODEL', 'Qwen/Qwen-Image-2.1').strip(),
        'steps': int(os.environ.get('IMAGE_API_STEPS', '8')),
        'guidance_scale': None,
        'single_image_input': False,
    },
    'bonsai': {
        'id': 'bonsai',
        'label': 'Bonsai Ternary 4B',
        'model': os.environ.get(
            'BONSAI_IMAGE_MODEL',
            'vllm-omni/prism-ml/bonsai-image-ternary-4B-unpacked',
        ).strip(),
        'steps': int(os.environ.get('BONSAI_API_STEPS', '8')),
        'guidance_scale': float(os.environ.get('BONSAI_GUIDANCE_SCALE', '1.5')),
        'single_image_input': True,
    },
}


def _resolve_ai_model(ai_model_key=None):
    key = (ai_model_key or 'qwen').strip().lower()
    return AI_MODELS.get(key, AI_MODELS['qwen'])


def _extract_image_bytes_from_chat_result(result, session):
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


def _call_image_api(prompt, source_data_url, style_data_url, out_width, out_height, ai_model_key='qwen', bonsai_prompt=None):
    api_key = os.environ.get('IMAGE_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError('IMAGE_API_KEY belum diisi di file .env.')
    endpoint = os.environ.get('IMAGE_API_URL', 'https://bifrost.hcm-lab.id/v1/chat/completions').strip()
    model_cfg = _resolve_ai_model(ai_model_key)
    timeout = int(os.environ.get('IMAGE_API_TIMEOUT', '180'))
    session = _get_http_session()

    def _invoke_model(cfg, active_prompt):
        extra_body = {'height': out_height, 'width': out_width}
        if cfg.get('steps') and cfg['steps'] > 0:
            extra_body['num_inference_steps'] = int(cfg['steps'])
        if cfg.get('guidance_scale') is not None:
            extra_body['guidance_scale'] = float(cfg['guidance_scale'])

        content_items = [
            {'type': 'text', 'text': active_prompt},
            {'type': 'image_url', 'image_url': {'url': source_data_url}},
        ]
        # Bonsai Ternary 4B performs true image-to-image style transfer when given 1 image_url + style text prompt.
        # Qwen Image 2.1 takes both source_data_url and style_data_url.
        if not cfg.get('single_image_input') and style_data_url:
            content_items.append({'type': 'image_url', 'image_url': {'url': style_data_url}})

        payload = {
            'model': cfg['model'],
            'messages': [{'role': 'user', 'content': content_items}],
            'extra_body': extra_body,
        }
        max_retries = 1 if cfg['id'] == 'qwen' else 2
        for attempt in range(max_retries + 1):
            try:
                response = session.post(
                    endpoint,
                    json=payload,
                    headers={'Content-Type': 'application/json', 'x-api-key': api_key},
                    timeout=timeout,
                )
                response.raise_for_status()
                return _extract_image_bytes_from_chat_result(response.json(), session)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else 0
                detail = exc.response.text[:600] if exc.response is not None else str(exc)
                # Fast-fail if backend port is down (connection refused) so we can immediately failover to Bonsai AI
                if 'connection refused' in detail.lower() or 'no healthy upstream' in detail.lower():
                    raise RuntimeError('Image API HTTP {}: {}'.format(code or 'Error', detail)) from exc
                if code in (502, 503, 504) and attempt < max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError('Image API HTTP {}: {}'.format(code or 'Error', detail)) from exc
            except requests.RequestException as exc:
                if attempt < max_retries:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError('Tidak dapat menghubungi Image API: {}'.format(exc)) from exc

    active_prompt = (bonsai_prompt or prompt) if model_cfg.get('single_image_input') else prompt
    try:
        return _invoke_model(model_cfg, active_prompt)
    except Exception as primary_exc:
        # If Qwen server is offline (e.g. HTTP 502 connection refused), automatically failover to live Bonsai Ternary 4B AI
        if model_cfg['id'] != 'bonsai':
            bonsai_cfg = _resolve_ai_model('bonsai')
            print(f'[*] {model_cfg["label"]} unavailable ({primary_exc}); failing over to {bonsai_cfg["label"]} cloud AI...')
            return _invoke_model(bonsai_cfg, bonsai_prompt or prompt)
        raise


def _stylize_fallback_pil(source_img, style_path):
    """Fast artistic color-palette & texture style transfer fallback if remote GPU server is down."""
    import numpy as np
    from PIL import ImageEnhance, ImageFilter

    src = source_img.convert('RGB')
    w, h = src.size
    with Image.open(style_path) as st_raw:
        st = ImageOps.fit(st_raw.convert('RGB'), (w, h), method=LANCZOS)

    src_arr = np.asarray(src, dtype=np.float32)
    st_arr = np.asarray(st, dtype=np.float32)

    # Reinhard color statistics transfer (blend 68% style palette + 32% original skin/lighting)
    src_mean = src_arr.mean(axis=(0, 1), keepdims=True)
    src_std = src_arr.std(axis=(0, 1), keepdims=True) + 1e-5
    st_mean = st_arr.mean(axis=(0, 1), keepdims=True)
    st_std = st_arr.std(axis=(0, 1), keepdims=True) + 1e-5

    recolored = (src_arr - src_mean) * (st_std / src_std) + st_mean
    recolored = np.clip(0.68 * recolored + 0.32 * src_arr, 0, 255)

    # Blend subtle high-frequency brushwork texture from the style image
    st_blur = np.asarray(st.filter(ImageFilter.GaussianBlur(radius=6)), dtype=np.float32)
    st_detail = st_arr - st_blur
    stylized_arr = np.clip(recolored + 0.42 * st_detail, 0, 255).astype(np.uint8)

    out = Image.fromarray(stylized_arr, mode='RGB')
    out = out.filter(ImageFilter.SMOOTH_MORE)
    out = ImageEnhance.Color(out).enhance(1.18)
    out = ImageEnhance.Contrast(out).enhance(1.08)
    return out


def _generate_image_via_api(image_path, style, output_path, style_data_url=None, ai_model_key='qwen'):
    out_width = int(os.environ.get('IMAGE_API_WIDTH', '512'))
    out_height = int(os.environ.get('IMAGE_API_HEIGHT', '384'))
    style_path = os.path.join(STYLE_DIR, style['file'])
    source_data_url = _encode_image_data_url(image_path)
    if style_data_url is None:
        style_data_url = _encode_image_data_url(style_path)

    style_desc = style.get('prompt_desc') or '{} ({})'.format(style['name'], style.get('mood', style['name']))
    prompt = (
        'Strictly apply strong neural style transfer from Image 2 (the style artwork) onto Image 1 (the photograph). '
        'Repaint 100% of Image 1—including the subject, face shading, hair, clothing, and entire background—using the '
        'exact color palette, brushwork, pattern motifs, surface texture, and artistic medium visible in Image 2. '
        'Do NOT leave Image 1 looking like a normal photograph and do NOT invent a generic style; match Image 2\'s visual '
        'texture and palette faithfully. Keep only the person\'s identity, facial features, pose, and composition from Image 1. '
        'Style reference: {} ({}). Return only the fully stylized image.'
    ).format(style['name'], style.get('mood', style['name']))
    bonsai_prompt = (
        'Transform this photograph into a vibrant {}. '
        'Repaint the entire image—including the subject, clothing, and background—with rich artistic brushwork, '
        'color palette, and surface texture while strictly preserving the person\'s exact face, identity, pose, and composition.'
    ).format(style_desc)
    try:
        image_bytes = _call_image_api(
            prompt, source_data_url, style_data_url, out_width, out_height,
            ai_model_key=ai_model_key, bonsai_prompt=bonsai_prompt,
        )
        with open(output_path, 'wb') as output:
            output.write(image_bytes)
    except Exception as exc:
        print(f'[*] AI API fallback triggered ({ai_model_key}) for {os.path.basename(image_path)}: {exc}')
        with Image.open(image_path) as src:
            fallback = _stylize_fallback_pil(_cover_image(src, (out_width, out_height)), style_path)
            fallback.save(output_path, format='PNG')


def _generate_batch_grid_via_api(image_paths, style, output_paths, style_data_url=None, ai_model_key='qwen'):
    """Combine up to 6 photos into a single 3x2 grid for 1 cloud render, then split back."""
    count = len(image_paths)
    if count == 0 or count > 6:
        raise ValueError('Grid batch hanya mendukung 1 sampai 6 foto.')

    default_grid_w = '768'
    default_grid_h = '512'
    grid_w = int(os.environ.get('IMAGE_API_GRID_WIDTH', default_grid_w))
    grid_h = int(os.environ.get('IMAGE_API_GRID_HEIGHT', default_grid_h))
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

    style_path = os.path.join(STYLE_DIR, style['file'])
    source_data_url = _encode_pil_data_url(grid_img)
    if style_data_url is None:
        style_data_url = _encode_image_data_url(style_path)

    style_desc = style.get('prompt_desc') or '{} ({})'.format(style['name'], style.get('mood', style['name']))
    prompt = (
        'Image 1 is a collage containing {} separate photographs inside white grid borders. '
        'Image 2 is the style reference ({}: {}). '
        'Apply strong neural style transfer from Image 2 onto Image 1 while strictly preserving every subject, '
        'object, face, outline, position, and white grid border of all {} photos in Image 1. '
        'Repaint the surfaces, colors, and textures of all {} photos to strongly match Image 2 without removing, '
        'shifting, or merging the {} photos.'
    ).format(count, style['name'], style.get('mood', style['name']), count, count, count)
    bonsai_prompt = (
        'Transform this 3x2 photo grid collage into a vibrant {} while strictly preserving every subject, '
        'person, face, pose, and white grid border of all {} panels. Repaint all {} photos with rich artistic '
        'brushwork, colors, and surface textures without shifting or merging the panels.'
    ).format(style_desc, count, count)

    try:
        image_bytes = _call_image_api(
            prompt, source_data_url, style_data_url, grid_w, grid_h,
            ai_model_key=ai_model_key, bonsai_prompt=bonsai_prompt,
        )
        with Image.open(BytesIO(image_bytes)) as rendered:
            rendered_rgb = rendered.convert('RGB')
            if rendered_rgb.size != (grid_w, grid_h):
                rendered_rgb = rendered_rgb.resize((grid_w, grid_h), resample=LANCZOS)

        inset = 4
        for (x0, y0, x1, y1), output_path in zip(boxes, output_paths):
            crop_box = (x0 + inset, y0 + inset, max(x0 + inset + 1, x1 - inset), max(y0 + inset + 1, y1 - inset))
            rendered_rgb.crop(crop_box).save(output_path, format='PNG')
    except Exception as exc:
        print(f'[*] Grid API fallback triggered ({ai_model_key}): {exc}')
        for image_path, output_path in zip(image_paths, output_paths):
            with Image.open(image_path) as src:
                fallback = _stylize_fallback_pil(_cover_image(src, (512, 384)), style_path)
                fallback.save(output_path, format='PNG')


def _precompute_subject_masks(original_paths, selected_indices):
    """Compute rembg alpha masks for selected slots on CPU while cloud rendering runs."""
    try:
        import rembg
    except ImportError:
        return None

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
    jpg_nobg_path = None
    try:
        import rembg
        rembg_sess = _get_rembg_session()
    except ImportError:
        rembg = None
        rembg_sess = None

    if rembg is not None:
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
        if mp4_path and os.path.isfile(mp4_path):
            zipf.write(mp4_path, arcname='studio-sela-musik.mp4')
        zipf.write(jpg_bg_path, arcname='studio-sela-frame-dengan-bg.jpg')
        if jpg_nobg_path and os.path.isfile(jpg_nobg_path):
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


def run_job(job_id, image_paths, selected_indices, style, guest_name, guest_email=None, ai_model_key='qwen'):
    job_path = os.path.join(JOB_DIR, job_id)
    processed_dir = os.path.join(job_path, 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    model_cfg = _resolve_ai_model(ai_model_key)
    ai_label = model_cfg['label']
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
                _update_progress(job_id, 'Mengirim {} foto sekaligus ke {}...'.format(total, ai_label), 15)
                with ThreadPoolExecutor(max_workers=2) as executor:
                    mask_future = executor.submit(_precompute_subject_masks, image_paths, selected_indices)
                    grid_future = executor.submit(
                        _generate_batch_grid_via_api, image_paths, style, styled_paths, style_data_url, model_cfg['id']
                    )
                    grid_future.result()
                    _update_progress(job_id, '{} selesai! Memisahkan 5 foto hasil...'.format(ai_label), 90)
                    try:
                        precomputed_masks = mask_future.result()
                    except Exception:
                        precomputed_masks = None
            else:
                styled_paths = [None] * total
                max_workers = max(1, min(total, int(os.environ.get('IMAGE_API_WORKERS', '2'))))
                _update_progress(job_id, 'Mengirim foto ke {} (1/{})...'.format(ai_label, total), 10)
                completed = 0
                with ThreadPoolExecutor(max_workers=max_workers + 1) as executor:
                    mask_future = executor.submit(_precompute_subject_masks, image_paths, selected_indices)
                    future_to_idx = {}
                    for index, image_path in enumerate(image_paths):
                        output_path = os.path.join(processed_dir, 'portrait_{:02d}.png'.format(index + 1))
                        fut = executor.submit(
                            _generate_image_via_api, image_path, style, output_path, style_data_url, model_cfg['id']
                        )
                        future_to_idx[fut] = (index, output_path)
                    for fut in as_completed(future_to_idx):
                        idx, out_path = future_to_idx[fut]
                        fut.result()
                        styled_paths[idx] = out_path
                        completed += 1
                        _update_progress(
                            job_id,
                            '{} selesai mengolah {} dari {} foto'.format(ai_label, completed, total),
                            10 + int(84 * completed / total),
                        )
                    try:
                        precomputed_masks = mask_future.result()
                    except Exception:
                        precomputed_masks = None
        elif engine == 'gatys_cpu':
            import neural_style_transfer as nst
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
        gif_path, mp4_path, jpg_bg_path, jpg_nobg_path, zip_path = _make_outputs(
            job_path, image_paths, styled_paths, selected_indices, guest_name,
            style=style, precomputed_masks=precomputed_masks,
        )
        done_payload = {
            'status': 'done',
            'gif': 'studio-sela-loop.gif',
            'frame': 'studio-sela-frame.jpg',
            'zip': 'studio-sela-semua-karya.zip',
            'guest_name': guest_name,
            'guest_email': guest_email or '',
            'progress': 100,
            'phase': 'Selesai',
        }
        if mp4_path and os.path.isfile(mp4_path):
            done_payload['mp4'] = 'studio-sela-musik.mp4'
        if jpg_nobg_path and os.path.isfile(jpg_nobg_path):
            done_payload['frame_nobg'] = 'studio-sela-frame-nobg.jpg'

        if guest_email and '@' in guest_email:
            _update_progress(job_id, 'Mengunggah ke Google Drive & mengirim link ke email...', 98)
            drive_res = gdrive_uploader.upload_job_to_drive(
                job_id, job_path, guest_name, guest_email=guest_email
            )
            if drive_res.get('success'):
                done_payload['drive_url'] = drive_res['drive_url']
                done_payload['qr_base64'] = drive_res['qr_base64']
                done_payload['email_sent'] = drive_res.get('email_sent', False)
                done_payload['email_sent_to'] = drive_res.get('email_sent_to')

        with JOBS_LOCK:
            JOBS[job_id].update(done_payload)
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
    ai_model_key = request.form.get('ai_model', 'qwen').strip().lower()
    if ai_model_key not in AI_MODELS:
        ai_model_key = 'qwen'
    selected_value = request.form.get('selected', '')
    guest_name = ' '.join(request.form.get('guest_name', '').split())[:32]
    guest_email = request.form.get('guest_email', '').strip()[:96]
    if not guest_name and guest_email and '@' in guest_name:
        guest_name = guest_email.split('@', 1)[0][:32]
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
            'phase': 'Menyiapkan job...', 'guest_email': guest_email,
            'ai_model': ai_model_key,
        }
    if os.environ.get('VERCEL'):
        run_job(job_id, image_paths, selected_indices, style, guest_name,
                guest_email=guest_email, ai_model_key=ai_model_key)
        with JOBS_LOCK:
            job = dict(JOBS.get(job_id, {}))
        if job.get('status') == 'error':
            return jsonify(error=job.get('error', 'Proses foto gagal.')), 500
        with open(os.path.join(job_path, job['gif']), 'rb') as f:
            gif_b64 = 'data:image/gif;base64,' + base64.b64encode(f.read()).decode('ascii')
        with open(os.path.join(job_path, job['frame']), 'rb') as f:
            frame_b64 = 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode('ascii')
        return jsonify(
            job_id=job_id,
            status='done',
            style=job.get('style', style['name']),
            ai_model=ai_model_key,
            gif_url=gif_b64,
            frame_url=frame_b64,
            drive_url=job.get('drive_url'),
            qr_base64=job.get('qr_base64'),
            email_sent=job.get('email_sent', False),
            email_sent_to=job.get('email_sent_to') or guest_email,
        ), 200
    threading.Thread(target=run_job,
                     args=(job_id, image_paths, selected_indices, style, guest_name, guest_email, ai_model_key),
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
        if 'mp4' in response:
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
        guest_email = job.get('guest_email', '')
        # Return cached drive url if already uploaded
        if job.get('drive_url') and job.get('qr_base64'):
            return jsonify({
                'success': True,
                'needs_setup': False,
                'drive_url': job['drive_url'],
                'qr_base64': job['qr_base64'],
                'email_sent': job.get('email_sent', False),
                'email_sent_to': job.get('email_sent_to') or guest_email,
            })

    job_path = os.path.join(JOB_DIR, job_id)
    result = gdrive_uploader.upload_job_to_drive(job_id, job_path, guest_name, guest_email=guest_email)
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
            import neural_style_transfer as nst
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
