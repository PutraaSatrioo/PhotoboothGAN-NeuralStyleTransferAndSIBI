"""Google Drive OAuth uploader and QR Code generator for Studio Sela."""
import base64
import io
import json
import os
import sys
import traceback

import qrcode


ROOT = os.path.dirname(os.path.abspath(__file__))
OAUTH_CLIENT_FILE = os.path.join(ROOT, 'client_secret.json')
TOKEN_FILE = os.path.join(ROOT, 'token.json')
CONFIG_FILE = os.path.join(ROOT, 'gdrive_config.json')
SCOPES = ['https://www.googleapis.com/auth/drive']


def get_parent_folder_id():
    """Read a Drive folder ID, accepting UTF-8 files with or without a BOM."""
    if not os.path.isfile(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8-sig') as config_file:
            data = json.load(config_file)
        folder_id = data.get('folder_id') or data.get('parent_folder_id')
        return folder_id.strip() if isinstance(folder_id, str) and folder_id.strip() else None
    except Exception as exc:
        print(f'[Google Drive] Gagal membaca {CONFIG_FILE}: {exc}', file=sys.stderr)
        return None


def generate_qr_base64(url):
    """Return a high-quality QR code as a PNG data URL."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color='#222520', back_color='#ffffff')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def _get_oauth_credentials():
    """Load saved OAuth credentials or start a local OAuth consent flow."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    credentials = None
    if os.path.isfile(TOKEN_FILE):
        credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    if not credentials or not credentials.valid:
        if not os.path.isfile(OAUTH_CLIENT_FILE):
            raise FileNotFoundError(
                'File client_secret.json belum ada. Unduh OAuth Client ID jenis Desktop app '
                'dari Google Cloud Console dan simpan di root proyek.'
            )
        flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_FILE, SCOPES)
        # The local server receives Google's redirect while the user signs in.
        credentials = flow.run_local_server(host='127.0.0.1', port=0, open_browser=True)

    with open(TOKEN_FILE, 'w', encoding='utf-8') as token_file:
        token_file.write(credentials.to_json())
    return credentials


def _google_api_error(exc):
    """Format Google API errors with status and response details when available."""
    status = getattr(getattr(exc, 'resp', None), 'status', None)
    content = getattr(exc, 'content', None)
    if isinstance(content, bytes):
        content = content.decode('utf-8', errors='replace')
    detail = str(content).strip() if content else str(exc)
    if status:
        return f'Google Drive API HTTP {status}: {detail}'
    return detail


def send_smtp_email(recipient_email, guest_name, drive_url):
    """Send Google Drive link via SMTP if SMTP_USER and SMTP_PASSWORD are set in .env."""
    smtp_user = os.environ.get('SMTP_USER', '').strip()
    smtp_password = os.environ.get('SMTP_PASSWORD', '').strip()
    if not smtp_user or not smtp_password or not recipient_email:
        return False
    smtp_host = os.environ.get('SMTP_HOST', 'smtp.gmail.com').strip()
    smtp_port = int(os.environ.get('SMTP_PORT', '465'))
    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg['Subject'] = f'Hasil Foto Artistik Pixora Photobooth — {guest_name}'
        msg['From'] = smtp_user
        msg['To'] = recipient_email
        msg.set_content(
            f'Halo {guest_name},\n\n'
            f'Karya foto artistik Pixora Photobooth kamu sudah selesai dirender!\n'
            f'Silakan buka dan unduh seluruh foto melalui link Google Drive berikut:\n\n'
            f'{drive_url}\n\n'
            f'Salam hangat,\nPixora Photobooth'
        )
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=20) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception as exc:
        print(f'[Email SMTP] Gagal mengirim email ke {recipient_email}: {exc}', file=sys.stderr)
        return False


def upload_job_to_drive(job_id, job_path, guest_name, guest_email=None):
    """Upload a completed photo package, share/email link if guest_email given, and return share URL + QR."""
    parent_id = get_parent_folder_id()
    if not parent_id:
        return {
            'success': False,
            'needs_setup': True,
            'message': 'Folder tujuan Google Drive belum diatur. Periksa folder_id di gdrive_config.json.'
        }

    try:
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        service = build('drive', 'v3', credentials=_get_oauth_credentials())

        # Verify access before creating a session folder.
        service.files().get(fileId=parent_id, fields='id,name,mimeType').execute()

        folder_metadata = {
            'name': f'Pixora - {guest_name} ({job_id[:8]})',
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id],
        }
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        folder_id = folder['id']

        service.permissions().create(
            fileId=folder_id,
            body={'type': 'anyone', 'role': 'reader'},
        ).execute()

        files_to_upload = [
            ('studio-sela-frame.jpg', 'image/jpeg'),
            ('studio-sela-frame-nobg.jpg', 'image/jpeg'),
            ('studio-sela-loop.gif', 'image/gif'),
            ('studio-sela-musik.mp4', 'video/mp4'),
            ('studio-sela-semua-karya.zip', 'application/zip'),
        ]
        for filename, mimetype in files_to_upload:
            file_path = os.path.join(job_path, filename)
            if not os.path.isfile(file_path):
                continue
            metadata = {'name': filename, 'parents': [folder_id]}
            media = MediaFileUpload(file_path, mimetype=mimetype, resumable=True)
            service.files().create(body=metadata, media_body=media, fields='id').execute()

        drive_url = f'https://drive.google.com/drive/folders/{folder_id}?usp=sharing'
        email_sent = False
        clean_email = (guest_email or '').strip()
        if clean_email and '@' in clean_email:
            # 1. Try SMTP email if configured
            email_sent = send_smtp_email(clean_email, guest_name, drive_url)
            # 2. Always share via Google Drive notification email (uses existing Google OAuth, no SMTP needed)
            try:
                service.permissions().create(
                    fileId=folder_id,
                    body={'type': 'user', 'role': 'reader', 'emailAddress': clean_email},
                    sendNotificationEmail=True,
                    emailMessage=(
                        f'Halo {guest_name}, hasil foto artistik Pixora Photobooth kamu sudah selesai! '
                        f'Buka folder Google Drive ini untuk mengunduh semua fotomu: {drive_url}'
                    ),
                ).execute()
                email_sent = True
            except Exception as share_exc:
                print(f'[Google Drive] Notifikasi email Drive ke {clean_email} gagal: {share_exc}', file=sys.stderr)

        return {
            'success': True,
            'needs_setup': False,
            'drive_url': drive_url,
            'qr_base64': generate_qr_base64(drive_url),
            'email_sent': email_sent,
            'email_sent_to': clean_email if email_sent else None,
        }
    except Exception as exc:
        detail = _google_api_error(exc)
        print(f'[Google Drive] Upload gagal untuk job {job_id}: {detail}', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {
            'success': False,
            'needs_setup': isinstance(exc, (FileNotFoundError,)),
            'error': detail,
            'message': f'Upload Google Drive gagal: {detail}',
        }
