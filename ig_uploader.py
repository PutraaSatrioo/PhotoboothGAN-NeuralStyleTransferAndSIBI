"""Instagram Story uploader untuk Studio Sela Photobooth.

Menggunakan instagrapi dengan penyimpanan sesi di ig_session.json agar
tidak berulang kali login. Kredensial dibaca dari .env (IG_USERNAME, IG_PASSWORD).

Cara pakai:
    result = upload_story_to_ig(image_path)
    # result = {'success': True, 'ig_url': 'https://instagram.com/username', 'qr_base64': '...'}
"""

from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
SESSION_FILE = os.path.join(ROOT, 'ig_session.json')
ENV_FILE = os.path.join(ROOT, '.env')


# ---------------------------------------------------------------------------
# Load .env sederhana (tanpa dependensi python-dotenv wajib)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Muat pasangan KEY=VALUE dari .env ke os.environ jika belum ada."""
    if not os.path.isfile(ENV_FILE):
        return
    with open(ENV_FILE, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()


# ---------------------------------------------------------------------------
# QR Code helper (sama seperti gdrive_uploader)
# ---------------------------------------------------------------------------

def _generate_qr_base64(url: str) -> str | None:
    try:
        import qrcode
        qr = qrcode.QRCode(version=1, box_size=8, border=3,
                            error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color='black', back_color='white')
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f'[IG Uploader] QR generation failed: {exc}', file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Instagrapi session management
# ---------------------------------------------------------------------------

def _get_client():
    """Kembalikan instagrapi.Client yang sudah login, dengan sesi tersimpan."""
    try:
        from instagrapi import Client
    except ImportError as exc:
        raise RuntimeError(
            'instagrapi belum terpasang. Jalankan: pip install instagrapi'
        ) from exc

    username = os.environ.get('IG_USERNAME', '').strip()
    password = os.environ.get('IG_PASSWORD', '').strip()

    if not username or not password:
        raise RuntimeError(
            'IG_USERNAME dan IG_PASSWORD belum diatur di file .env'
        )

    # Jangan pin app/device profile di sini. instagrapi merawat profile
    # terkininya sendiri; profile manual mudah kedaluwarsa dan ditolak IG.
    cl = Client()

    # Muat sesi tersimpan jika ada agar tidak perlu login ulang
    if os.path.isfile(SESSION_FILE):
        try:
            cl.load_settings(SESSION_FILE)
            # Gunakan cookie sesi yang tersimpan agar tidak memulai login
            # kata sandi lagi (yang sering terkena app_out_of_date/challenge).
            sessionid = cl.private.cookies.get("sessionid")
            if sessionid and hasattr(cl, "login_by_sessionid"):
                cl.login_by_sessionid(sessionid)
            else:
                cl.login(username, password)
            cl.dump_settings(SESSION_FILE)
            print('[IG Uploader] Sesi dimuat dari ig_session.json')
            return cl
        except Exception as exc:
            print(f'[IG Uploader] Gagal memakai sesi tersimpan ({exc}), login ulang...',
                  file=sys.stderr)

    # Login baru
    cl.login(username, password)
    cl.dump_settings(SESSION_FILE)
    print('[IG Uploader] Login berhasil, sesi disimpan ke ig_session.json')
    return cl


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_story_to_ig(image_path: str) -> dict:
    """Unggah gambar sebagai Instagram Story.

    Returns:
        dict dengan kunci:
        - success (bool)
        - ig_url (str): URL profil Instagram (stories tidak memiliki permalink publik)
        - qr_base64 (str | None): QR code PNG sebagai data URI
        - error (str | None): pesan kesalahan jika success=False
    """
    if not os.path.isfile(image_path):
        return {'success': False, 'error': f'File gambar tidak ditemukan: {image_path}'}

    try:
        cl = _get_client()
        username = os.environ.get('IG_USERNAME', '').strip()

        # Konversi ke JPEG yang kompatibel dengan Instagram (min 1080px lebar)
        from PIL import Image
        img = Image.open(image_path).convert('RGB')

        # Instagram Story: rasio 9:16, min 1080×1920 — resize + letterbox
        target_w, target_h = 1080, 1920
        orig_w, orig_h = img.size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        canvas = Image.new('RGB', (target_w, target_h), (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(img_resized, (paste_x, paste_y))

        tmp_path = image_path + '_ig_story.jpg'
        canvas.save(tmp_path, 'JPEG', quality=95)

        try:
            cl.photo_upload_to_story(tmp_path, caption='')
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        ig_profile_url = f'https://www.instagram.com/{username}/'
        qr = _generate_qr_base64(ig_profile_url)

        print(f'[IG Uploader] Story berhasil diunggah ke @{username}')
        return {
            'success': True,
            'ig_url': ig_profile_url,
            'qr_base64': qr,
        }

    except Exception as exc:
        detail = str(exc)
        if 'your version of instagram is out of date' in detail.lower():
            detail = (
                'Instagram menolak versi/profile aplikasi yang dikirim saat login. '
                'Perbarui dependensi dengan: python -m pip install -r requirements-web.txt '
                'lalu coba lagi. Jika tetap gagal, cek login akun melalui aplikasi resmi '
                'dan tunggu sebelum mencoba lagi; Instagram dapat membatasi login otomatis.'
            )
        print(f'[IG Uploader] Gagal: {detail}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {'success': False, 'error': detail[:500]}
