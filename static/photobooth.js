(() => {
  const video = document.getElementById('camera');
  const trackingVideo = document.getElementById('tracking-camera');
  const welcomeCameraPlaceholder = document.getElementById('welcome-camera-placeholder');
  const cameraFrame = document.getElementById('camera-frame');
  const welcomeCameraFrame = document.getElementById('welcome-camera-frame');
  const preview = document.getElementById('upload-preview');
  const cameraFlash = document.getElementById('camera-flash');
  const placeholder = document.getElementById('camera-placeholder');
  const countdown = document.getElementById('countdown');
  const fileInput = document.getElementById('photo-upload');
  const processing = document.getElementById('processing');
  const toast = document.getElementById('toast');
  const handCursor = document.getElementById('hand-cursor');
  const handPill = document.getElementById('hand-pill');
  const shotCounter = document.getElementById('shot-counter');
  const guestGreeting = document.getElementById('guest-greeting');

  let cameraStream = null;
  const maxPhotos = 5;
  const framePhotoCount = 3;
  let capturedPhotos = [];
  let selectedForFrame = new Set();
  let captureSequenceActive = false;
  let activeStyleId = 'vg_starry_night';
  let activeStyleName = 'Starry Night';
  let selectedAiModel = 'gemini';
  let guestName = 'UNESA';
  let guestEmail = '';
  let isDefaultName = true;
  let currentJobId = null;
  let currentZipUrl = null;
  let skipWaitingCurrentJob = false;
  const skippedJobIds = new Set();

  let sibiModel = null;
  let sibiCandidate = '';
  let sibiCandidateSince = 0;
  let committedSibiPose = '';
  const sibiConfirmMs = 3000;
  const sibiConfidenceThreshold = 0.72;
  let toastTimer = null;

  // Hand tracking state
  let isPinching = false;
  let pinchCooldown = false;
  let cursorX = window.innerWidth / 2;
  let cursorY = window.innerHeight / 2;

  function detectMobileDevice() {
    const byWidth = window.matchMedia('(max-width: 900px)').matches;
    const byTouch = window.matchMedia('(pointer: coarse)').matches && window.innerWidth <= 1024;
    const byUA = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent || '');
    return byWidth || byTouch || byUA;
  }

  let isMobile = detectMobileDevice();
  document.body.classList.toggle('is-mobile', isMobile);

  function loadMediaPipeScripts() {
    if (isMobile || document.getElementById('mp-hands-script')) return;
    const s1 = document.createElement('script');
    s1.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js';
    s1.crossOrigin = 'anonymous';
    const s2 = document.createElement('script');
    s2.id = 'mp-hands-script';
    s2.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
    s2.crossOrigin = 'anonymous';
    document.head.appendChild(s1);
    document.head.appendChild(s2);
  }

  // ---------------------------------------------------------------------------
  // SIBI Fingerspelling
  // ---------------------------------------------------------------------------

  async function loadSibiModel() {
    try {
      const response = await fetch('/api/sibi/model');
      if (!response.ok) throw new Error('model unavailable');
      sibiModel = await response.json();
      const status = document.getElementById('sibi-status');
      if (status) status.textContent = 'MODEL SIBI SIAP · 24 HURUF STATIS';
    } catch (_) {
      sibiModel = null;
      const status = document.getElementById('sibi-status');
      if (status) status.textContent = 'MODEL SIBI TIDAK TERSEDIA · GUNAKAN KEYBOARD';
    }
  }

  function denseLayer(input, kernel, bias, relu) {
    const output = new Array(bias.length);
    for (let col = 0; col < bias.length; col++) {
      let val = bias[col];
      for (let row = 0; row < input.length; row++) val += input[row] * kernel[row][col];
      output[col] = relu ? Math.max(0, val) : val;
    }
    return output;
  }

  function classifySibi(landmarks) {
    if (!sibiModel || !landmarks || landmarks.length !== 21) return null;
    const xs = landmarks.map(p => p.x);
    const ys = landmarks.map(p => p.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const features = [];
    landmarks.forEach(p => features.push(p.x - minX, p.y - minY));
    const weights = sibiModel.layers;
    let values = denseLayer(features, weights[0], weights[1], true);
    const gamma = weights[2];
    const beta = weights[3];
    const mean = weights[4];
    const variance = weights[5];
    values = values.map((val, idx) => ((val - mean[idx]) / Math.sqrt(variance[idx] + 0.001)) * gamma[idx] + beta[idx]);
    values = denseLayer(values, weights[6], weights[7], true);
    values = denseLayer(values, weights[8], weights[9], false);
    const maxLogit = Math.max(...values);
    const probabilities = values.map(val => Math.exp(val - maxLogit));
    const total = probabilities.reduce((sum, val) => sum + val, 0);
    const best = probabilities.reduce((bestIdx, val, idx) => val > probabilities[bestIdx] ? idx : bestIdx, 0);
    return { letter: sibiModel.classes[best], confidence: probabilities[best] / total };
  }

  function resetSibiPrediction(message) {
    sibiCandidate = '';
    sibiCandidateSince = 0;
    const conf = document.getElementById('sibi-confidence');
    const bar = document.getElementById('sibi-progress-bar');
    const hint = document.getElementById('sibi-confirm-hint');
    const status = document.getElementById('sibi-status');
    const cand = document.getElementById('sibi-candidate');
    if (cand) cand.textContent = '—';
    if (conf) conf.textContent = 'Menunggu tangan';
    if (bar) bar.style.width = '0%';
    if (hint) hint.textContent = message || 'Tahan huruf SIBI 3 detik untuk konfirmasi';
    if (status && message) status.textContent = message;
  }

  function updateSibiPrediction(landmarks) {
    if (document.getElementById('welcome-screen').hidden || !sibiModel) return;
    const prediction = classifySibi(landmarks);
    if (!prediction || prediction.confidence < sibiConfidenceThreshold) {
      resetSibiPrediction('TANGAN TERDETEKSI · POSISIKAN LEBIH JELAS');
      return;
    }

    if (prediction.letter === committedSibiPose) {
      document.getElementById('sibi-status').textContent = `HURUF ${prediction.letter} SUDAH MASUK · UBAH POSE UNTUK BERIKUTNYA`;
      return;
    }
    if (committedSibiPose) committedSibiPose = '';

    const now = performance.now();
    if (prediction.letter !== sibiCandidate) {
      sibiCandidate = prediction.letter;
      sibiCandidateSince = now;
    }
    const progress = Math.min(1, (now - sibiCandidateSince) / sibiConfirmMs);
    document.getElementById('sibi-candidate').textContent = prediction.letter;
    document.getElementById('sibi-confidence').textContent = `${Math.round(prediction.confidence * 100)}% YAKIN`;
    document.getElementById('sibi-progress-bar').style.width = `${progress * 100}%`;
    document.getElementById('sibi-confirm-hint').textContent = progress >= 1
      ? 'Huruf masuk! Lepas sejenak sebelum huruf yang sama.'
      : `Tahan: masuk otomatis dalam ${Math.ceil((1 - progress) * 3)} detik`;
    document.getElementById('sibi-status').textContent = `TERDETEKSI HURUF ${prediction.letter}`;

    if (progress >= 1) {
      insertNameText(prediction.letter);
      committedSibiPose = prediction.letter;
      sibiCandidate = '';
      sibiCandidateSince = 0;
    }
  }

  // ---------------------------------------------------------------------------
  // Virtual Keyboard & Text Input
  // ---------------------------------------------------------------------------

  function insertNameText(text) {
    const input = document.getElementById('guest-name');
    if (isDefaultName) {
      input.value = '';
      isDefaultName = false;
    }
    const start = input.selectionStart || 0;
    const end = input.selectionEnd || 0;
    input.setRangeText(text, start, end, 'end');
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.focus();
  }

  function setupPhysicalInputHandlers() {
    const input = document.getElementById('guest-name');
    if (!input) return;
    input.addEventListener('keydown', (e) => {
      // Clear default UNESA placeholder on first letter typed
      if (isDefaultName && e.key.length === 1 && !e.ctrlKey && !e.altKey && !e.metaKey) {
        input.value = '';
        isDefaultName = false;
      }
    });
    input.addEventListener('input', () => {
      if (input.value !== 'UNESA') {
        isDefaultName = false;
      }
    });
  }

  function buildVirtualKeyboard() {
    const keyboard = document.getElementById('virtual-keyboard');
    if (!keyboard) return;
    keyboard.replaceChildren();

    const rows = [
      ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
      ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
      ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
    ];

    rows.forEach(letters => {
      const rowEl = document.createElement('div');
      rowEl.className = 'key-row';
      letters.forEach(letter => {
        const key = document.createElement('button');
        key.type = 'button';
        key.className = 'letter-key';
        key.textContent = letter;
        key.addEventListener('click', (e) => {
          e.preventDefault();
          insertNameText(letter);
        });
        rowEl.appendChild(key);
      });
      keyboard.appendChild(rowEl);
    });

    // Control keys row: HAPUS, SPASI, BERSIHKAN
    const ctrlRow = document.createElement('div');
    ctrlRow.className = 'key-row key-controls';

    const hapusBtn = document.createElement('button');
    hapusBtn.type = 'button';
    hapusBtn.className = 'function-key key-hapus';
    hapusBtn.textContent = 'HAPUS';
    hapusBtn.addEventListener('click', (e) => {
      e.preventDefault();
      const input = document.getElementById('guest-name');
      if (isDefaultName) {
        input.value = '';
        isDefaultName = false;
        input.focus();
        return;
      }
      const start = input.selectionStart || 0;
      const end = input.selectionEnd || 0;
      if (start === end && start > 0) input.setSelectionRange(start - 1, end);
      input.setRangeText('', input.selectionStart, input.selectionEnd, 'end');
      input.focus();
    });

    const spasiBtn = document.createElement('button');
    spasiBtn.type = 'button';
    spasiBtn.className = 'function-key key-spasi';
    spasiBtn.textContent = 'SPASI';
    spasiBtn.addEventListener('click', (e) => {
      e.preventDefault();
      insertNameText(' ');
    });

    const bersihkanBtn = document.createElement('button');
    bersihkanBtn.type = 'button';
    bersihkanBtn.className = 'function-key key-bersihkan';
    bersihkanBtn.textContent = 'BERSIHKAN';
    bersihkanBtn.addEventListener('click', (e) => {
      e.preventDefault();
      const input = document.getElementById('guest-name');
      input.value = '';
      isDefaultName = false;
      input.focus();
    });

    ctrlRow.appendChild(hapusBtn);
    ctrlRow.appendChild(spasiBtn);
    ctrlRow.appendChild(bersihkanBtn);
    keyboard.appendChild(ctrlRow);
  }

  function notify(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
  }

  // ---------------------------------------------------------------------------
  // Camera & Video Setup
  // ---------------------------------------------------------------------------

  async function startCamera() {
    const statusTitle = document.getElementById('camera-status-title');
    const statusMessage = document.getElementById('camera-status-message');
    const retryButton = document.getElementById('camera-retry');
    if (retryButton) retryButton.hidden = true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (statusTitle) statusTitle.textContent = 'Akses kamera tidak tersedia';
      if (statusMessage) statusMessage.textContent = 'Buka aplikasi lewat localhost atau 127.0.0.1, lalu coba browser terbaru.';
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = false;
      return;
    }
    try {
      const videoConstraints = isMobile
        ? { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } }
        : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } };
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
      });
      video.srcObject = cameraStream;
      video.play().catch(err => console.warn('Camera preview notice:', err));
      video.hidden = false;
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = true;
      if (placeholder) placeholder.hidden = true;
      if (!isMobile) {
        trackingVideo.srcObject = cameraStream;
        trackingVideo.play().catch(err => console.warn('Background tracking video notice:', err));
        initMediaPipeHands();
      }
    } catch (err) {
      video.hidden = true;
      const messages = {
        NotAllowedError: 'Izin kamera ditolak. Klik ikon kamera di bilah alamat, ubah izinnya menjadi Izinkan, lalu coba lagi.',
        PermissionDeniedError: 'Izin kamera ditolak. Izinkan kamera untuk situs ini dari ikon di bilah alamat.',
        NotFoundError: 'Kamera tidak ditemukan. Sambungkan atau aktifkan kamera, lalu coba lagi.',
        NotReadableError: 'Kamera sedang dipakai aplikasi lain. Tutup aplikasi tersebut, lalu coba lagi.',
        AbortError: 'Kamera gagal dimulai. Periksa koneksi kamera lalu coba lagi.',
        SecurityError: 'Browser memblokir akses kamera. Buka halaman melalui localhost atau 127.0.0.1.'
      };
      const errMsg = messages[err.name] || `Kesalahan kamera (${err.name || 'tidak diketahui'}). Periksa izin dan sambungan kamera.`;
      if (statusTitle) statusTitle.textContent = 'Kamera belum bisa dibuka';
      if (statusMessage) statusMessage.textContent = errMsg;
      if (retryButton) retryButton.hidden = false;
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = false;
      if (placeholder) {
        placeholder.hidden = false;
        const pStrong = placeholder.querySelector('strong');
        const pSpan = placeholder.querySelector('span:not(.camera-icon)');
        if (pStrong) pStrong.textContent = 'Kamera belum bisa dibuka (ketuk untuk coba lagi)';
        if (pSpan) pSpan.textContent = errMsg;
      }
    }
  }

  document.getElementById('camera-retry')?.addEventListener('click', startCamera);
  placeholder?.addEventListener('click', () => { if (!cameraStream) startCamera(); });

  // ---------------------------------------------------------------------------
  // Screen 1: "DONE" -> Move to Screen 2
  // ---------------------------------------------------------------------------

  const doneButton = document.getElementById('name-done');
  doneButton.addEventListener('click', () => {
    const nameInput = document.getElementById('guest-name');
    let val = nameInput.value.trim().replace(/\s+/g, ' ').slice(0, 32);
    if (!val) {
      val = 'UNESA';
      nameInput.value = val;
    }
    guestEmail = '';
    guestName = val;
    guestGreeting.textContent = `Halo, ${guestName}`;

    // Seamlessly move camera preview video into Screen 2
    cameraFrame.prepend(video);
    video.hidden = !cameraStream;
    placeholder.hidden = !!cameraStream;

    document.getElementById('welcome-screen').hidden = true;
    document.getElementById('photobooth-app').hidden = false;
    notify(`Selamat datang, ${guestName}! Silakan pilih gaya seni untuk mulai jepret.`);
  });

  // ---------------------------------------------------------------------------
  // Screen 2: Style Selection & Immediate 5-Photo Burst Capture
  // ---------------------------------------------------------------------------

  function setupStyleCards() {
    const styleCards = document.querySelectorAll('.style-card');
    const styleGrid = document.getElementById('style-grid');
    styleCards.forEach((card, idx) => {
      card.style.setProperty('--card-idx', idx);
      card.addEventListener('click', () => {
        if (captureSequenceActive) return;
        const styleId = card.dataset.style;
        const styleName = card.dataset.name || card.querySelector('.style-name')?.textContent || styleId;
        activeStyleId = styleId;
        activeStyleName = styleName;

        styleCards.forEach(c => {
          c.classList.remove('selected', 'style-popping');
          c.setAttribute('aria-pressed', 'false');
        });
        card.classList.add('selected', 'style-popping');
        card.setAttribute('aria-pressed', 'true');

        if (isMobile) {
          if (styleGrid) styleGrid.classList.add('style-choosing');
          setTimeout(() => {
            card.classList.remove('style-popping');
            if (styleGrid) styleGrid.classList.remove('style-choosing');
            openMobileCameraStep();
          }, 260);
          return;
        }

        // Desktop/laptop: Immediately trigger 5-photo burst capture!
        startAutoBurstCapture();
      });
    });
  }

  function triggerMobileWebFullCapture() {
    if (captureSequenceActive) return;
    const camColumn = document.getElementById('camera-column');
    if (camColumn) {
      camColumn.classList.add('mobile-capture-fullscreen');
    }
    if (video && video.paused && cameraStream) {
      video.play().catch(() => {});
    }
    startAutoBurstCapture();
  }

  function setMobileStep(step) {
    document.body.dataset.mobileStep = step;
    const mobileWelcome = document.getElementById('mobile-welcome-screen');
    const appEl = document.getElementById('photobooth-app');
    const camTopbar = document.getElementById('mobile-camera-topbar');
    const camControls = document.getElementById('mobile-camera-controls');
    const camColumn = document.getElementById('camera-column');

    if (camColumn) camColumn.classList.remove('mobile-capture-fullscreen', 'mobile-manual-fullscreen');

    if (step === 'welcome') {
      if (mobileWelcome) mobileWelcome.hidden = false;
      if (appEl) appEl.hidden = true;
    } else if (step === 'style') {
      if (mobileWelcome) mobileWelcome.hidden = true;
      if (appEl) appEl.hidden = false;
      if (camTopbar) camTopbar.hidden = true;
      if (camControls) camControls.hidden = true;
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (step === 'camera') {
      if (mobileWelcome) mobileWelcome.hidden = true;
      if (appEl) appEl.hidden = false;
      if (camTopbar) camTopbar.hidden = false;
      if (camControls) camControls.hidden = false;
      const pill = document.getElementById('mobile-active-style-pill');
      if (pill) pill.textContent = `Style: ${activeStyleName}`;
      if (video.parentElement !== cameraFrame) {
        cameraFrame.prepend(video);
      }
      video.hidden = !cameraStream;
      placeholder.hidden = !!cameraStream;
      if (!cameraStream) {
        startCamera();
      } else if (video.paused) {
        video.play().catch(() => {});
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  }

  function openMobileCameraStep() {
    setMobileStep('camera');
  }

  async function startAutoBurstCapture() {
    if (captureSequenceActive) return;
    captureSequenceActive = true;
    capturedPhotos = [];
    selectedForFrame.clear();
    shotCounter.textContent = `0 / ${maxPhotos}`;
    preview.hidden = true;
    video.hidden = !cameraStream;

    if (!isMobile) {
      notify(`Gaya ${activeStyleName} dipilih! Bersiap di depan kamera...`);
    }

    try {
      const canvas = document.getElementById('capture-canvas');
      for (let shot = 1; shot <= maxPhotos; shot++) {
        // 3-2-1 countdown per shot
        for (const num of ['3', '2', '1']) {
          countdown.textContent = num;
          countdown.hidden = false;
          await new Promise(r => setTimeout(r, 650));
        }
        countdown.hidden = true;

        // Snap photo from video feed (cap resolution on mobile/serverless to keep upload fast & under 4.5MB)
        const srcW = video.videoWidth || (isMobile ? 960 : 1280);
        const srcH = video.videoHeight || 720;
        const maxCaptureW = isMobile ? 960 : 1280;
        const scale = srcW > maxCaptureW ? maxCaptureW / srcW : 1;
        canvas.width = Math.round(srcW * scale);
        canvas.height = Math.round(srcH * scale);
        const ctx = canvas.getContext('2d');
        ctx.setTransform(-1, 0, 0, 1, canvas.width, 0); // mirror horizontally
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const jpegQuality = isMobile ? 0.86 : 0.92;
        const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', jpegQuality));
        if (!blob) throw new Error('Gagal mengambil frame foto.');

        capturedPhotos.push({
          blob,
          url: URL.createObjectURL(blob),
          name: `capture_${shot}.jpg`
        });

        shotCounter.textContent = `${shot} / ${maxPhotos}`;

        // Flash animation
        cameraFlash.classList.remove('active');
        void cameraFlash.offsetWidth;
        cameraFlash.classList.add('active');

        await new Promise(r => setTimeout(r, 800));
        cameraFlash.classList.remove('active');
      }

      // Finish burst capture: navigate directly to photo picker
      finishBurstToPicker();
    } catch (err) {
      notify(err.message || 'Gagal mengambil 5 foto.');
    } finally {
      captureSequenceActive = false;
      countdown.hidden = true;
      const camColumn = document.getElementById('camera-column');
      if (camColumn) camColumn.classList.remove('mobile-capture-fullscreen', 'mobile-manual-fullscreen');
    }
  }

  function finishBurstToPicker() {
    if (isMobile) {
      document.body.dataset.mobileStep = 'picker';
    }
    document.querySelector('.booth-grid').hidden = true;
    const picker = document.getElementById('capture-picker');
    picker.hidden = false;
    renderCapturePicker();
    picker.scrollIntoView({ behavior: 'smooth', block: 'start' });
    if (!isMobile) {
      notify('Foto berhasil diambil! Pilih 3 foto terbaikmu untuk frame.');
    }
  }

  // ---------------------------------------------------------------------------
  // Photo Picker (PILIH 3 FOTO) & Auto-Generate
  // ---------------------------------------------------------------------------

  function renderCapturePicker() {
    const grid = document.getElementById('capture-grid');
    grid.replaceChildren();

    capturedPhotos.forEach((photo, idx) => {
      const choice = document.createElement('button');
      choice.type = 'button';
      choice.className = 'capture-choice';
      if (selectedForFrame.has(idx)) choice.classList.add('selected');
      choice.innerHTML = `
        <img alt="Foto ${idx + 1}" src="${photo.url}">
        <span class="capture-number">0${idx + 1}</span>
        <span class="capture-check">✓</span>
      `;

      choice.addEventListener('click', () => {
        if (selectedForFrame.has(idx)) {
          selectedForFrame.delete(idx);
        } else if (selectedForFrame.size < framePhotoCount) {
          selectedForFrame.add(idx);
        } else {
          notify('Kamu sudah memilih 3 foto. Klik foto lain untuk mengganti.');
          return;
        }

        choice.classList.toggle('selected', selectedForFrame.has(idx));
        updatePickerStatus();
      });

      grid.appendChild(choice);
    });

    updatePickerStatus();
  }

  function updatePickerStatus() {
    const count = selectedForFrame.size;
    const badge = document.getElementById('selection-count');
    const confirmBtn = document.getElementById('confirm-selection');
    badge.textContent = `${count} / ${framePhotoCount} DIPILIH`;

    if (count === framePhotoCount) {
      confirmBtn.disabled = false;
      confirmBtn.innerHTML = `<span>LANJUTKAN & PROSES SENI (${activeStyleName.toUpperCase()})</span> <span class="btn-arrow">&rarr;</span>`;
    } else {
      confirmBtn.disabled = true;
      confirmBtn.innerHTML = `<span>PILIH TEPAT 3 FOTO (${count}/3)</span> <span class="btn-arrow">&rarr;</span>`;
    }
  }

  const confirmBtn = document.getElementById('confirm-selection');

  confirmBtn.addEventListener('click', () => {
    if (selectedForFrame.size !== framePhotoCount) {
      notify('Silakan pilih tepat 3 foto.');
      return;
    }
    // Directly submit to generate artwork with style selected upfront
    submitCaptureJob();
  });

  // ---------------------------------------------------------------------------
  // Generate & Poll Artwork
  // ---------------------------------------------------------------------------

  async function submitCaptureJob() {
    skipWaitingCurrentJob = false;
    document.getElementById('capture-picker').hidden = true;
    document.getElementById('result-section').hidden = true;
    const bar = document.getElementById('progress-bar');
    const pctLabel = document.getElementById('processing-pct');
    if (bar) bar.style.width = '0%';
    if (pctLabel) pctLabel.textContent = '0%';
    document.getElementById('processing-phase').textContent = `Memproses 5 foto dengan Gemini AI (${activeStyleName})...`;
    const emailNote = document.getElementById('processing-email-note');
    if (emailNote) {
      emailNote.textContent = guestEmail
        ? `LINK GOOGLE DRIVE AKAN DIKIRIM OTOMATIS KE ${guestEmail.toUpperCase()}`
        : 'TUNGGU HASIL DI SINI ATAU LEWATI KE PEMILIHAN STYLE';
    }
    processing.hidden = false;

    try {
      const form = new FormData();
      capturedPhotos.forEach((photo, idx) => {
        form.append('photos', photo.blob, photo.name || `capture_${idx + 1}.jpg`);
      });
      form.append('style', activeStyleId);
      form.append('ai_model', selectedAiModel);
      form.append('selected', Array.from(selectedForFrame).sort((a, b) => a - b).join(','));
      form.append('guest_name', guestName);
      form.append('guest_email', guestEmail);

      const res = await fetch('/api/captures', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Gagal memulai pemrosesan foto.');

      currentJobId = data.job_id;
      if (skipWaitingCurrentJob) {
        skippedJobIds.add(data.job_id);
        return;
      }
      if (data.status === 'done') {
        renderCompletedJob(data.job_id, data);
      } else {
        pollJob(data.job_id);
      }
    } catch (err) {
      if (skipWaitingCurrentJob) return;
      processing.hidden = true;
      document.getElementById('capture-picker').hidden = false;
      notify(err.message || 'Terjadi kesalahan saat memproses foto.');
    }
  }

  document.getElementById('skip-loading-btn')?.addEventListener('click', () => {
    skipWaitingCurrentJob = true;
    if (currentJobId) skippedJobIds.add(currentJobId);
    processing.hidden = true;
    const targetEmail = guestEmail || 'email Google kamu';
    notify(`Render berjalan di server! Link Google Drive akan dikirim ke ${targetEmail}.`);
    resetForAnotherPhotoSession();
  });

  function withCacheBust(url) {
    return url && !url.startsWith('data:') ? `${url}?t=${Date.now()}` : url;
  }

  function renderCompletedJob(jobId, job) {
    if (skippedJobIds.has(jobId) || skipWaitingCurrentJob) return;
    currentJobId = jobId;
    currentZipUrl = job.zip_url || `/results/${jobId}/studio-sela-semua-karya.zip`;
    document.getElementById('gif-image').src = withCacheBust(job.gif_url);
    const mp4El = document.getElementById('mp4-video');
    if (mp4El) {
      if (job.mp4_url) {
        mp4El.src = withCacheBust(job.mp4_url);
        mp4El.closest('.deliverable-card')?.removeAttribute('hidden');
      } else {
        mp4El.closest('.deliverable-card')?.setAttribute('hidden', '');
      }
    }
    document.getElementById('frame-image').src = withCacheBust(job.frame_url);
    const nobgEl = document.getElementById('frame-nobg-image');
    if (nobgEl) {
      if (job.frame_nobg_url) {
        nobgEl.src = withCacheBust(job.frame_nobg_url);
        nobgEl.closest('.deliverable-card')?.removeAttribute('hidden');
      } else {
        nobgEl.closest('.deliverable-card')?.setAttribute('hidden', '');
      }
    }
    document.getElementById('result-style-name').textContent = job.style;

    const emailBanner = document.getElementById('result-email-banner');
    if (emailBanner) {
      const sentTo = job.email_sent_to || guestEmail;
      if (sentTo) {
        emailBanner.textContent = `✓ Link Google Drive otomatis dikirim ke ${sentTo}`;
        emailBanner.hidden = false;
      } else {
        emailBanner.hidden = true;
      }
    }

    processing.hidden = true;
    const btnIg = document.getElementById('btn-upload-ig');
    const btnDrive = document.getElementById('btn-upload-drive');
    if (isMobile) {
      document.body.dataset.mobileStep = 'result';
      if (btnIg) btnIg.hidden = false;
      if (btnDrive) btnDrive.hidden = true;
    } else {
      if (btnIg) btnIg.hidden = true;
      if (btnDrive) btnDrive.hidden = false;
    }
    document.getElementById('result-section').hidden = false;
    if (!isMobile) {
      notify('Karyamu sudah siap! Link Google Drive juga dikirim ke emailmu.');
    }
  }

  async function pollJob(jobId) {
    if (skippedJobIds.has(jobId) || skipWaitingCurrentJob) return;
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (skippedJobIds.has(jobId) || skipWaitingCurrentJob) return;
      const job = await res.json();
      if (!res.ok) throw new Error(job.error || 'Sesi tidak ditemukan.');

      if (job.status === 'error') throw new Error(job.error || 'Proses foto gagal.');

      if (job.status === 'done') {
        renderCompletedJob(jobId, job);
        return;
      }

      const pct = job.progress || 0;
      const bar = document.getElementById('progress-bar');
      const pctLabel = document.getElementById('processing-pct');
      const phaseLabel = document.getElementById('processing-phase');
      if (bar) bar.style.width = `${pct}%`;
      if (pctLabel) pctLabel.textContent = `${pct}%`;
      if (phaseLabel) phaseLabel.textContent = job.phase || 'Sedang melukis karya seni...';

      setTimeout(() => pollJob(jobId), 1500);
    } catch (err) {
      processing.hidden = true;
      document.getElementById('capture-picker').hidden = false;
      notify(err.message || 'Gagal memproses hasil foto.');
    }
  }

  // ---------------------------------------------------------------------------
  // Instagram Story + Google Drive Upload — Dual QR Code Modal
  // ---------------------------------------------------------------------------

  const qrModal = document.getElementById('qr-modal');
  const qrCloseBtn = document.getElementById('qr-close');
  const qrTakeAgainBtn = document.getElementById('qr-take-again');

  // Helpers: show/hide individual panel states
  function showIgLoading()  { document.getElementById('qr-ig-loading')?.removeAttribute('hidden'); document.getElementById('qr-ig-image-wrap')?.setAttribute('hidden',''); document.getElementById('qr-ig-error')?.setAttribute('hidden',''); }
  function showIgQr(src)    { document.getElementById('qr-ig-loading')?.setAttribute('hidden',''); const w=document.getElementById('qr-ig-image-wrap'); if(w){w.removeAttribute('hidden'); document.getElementById('qr-ig-img').src=src;} }
  function showIgError(msg) { document.getElementById('qr-ig-loading')?.setAttribute('hidden',''); document.getElementById('qr-ig-image-wrap')?.setAttribute('hidden',''); const e=document.getElementById('qr-ig-error'); if(e){e.textContent=msg; e.removeAttribute('hidden');} }

  function showDriveLoading()  { document.getElementById('qr-drive-loading')?.removeAttribute('hidden'); document.getElementById('qr-drive-image-wrap')?.setAttribute('hidden',''); document.getElementById('qr-drive-error')?.setAttribute('hidden',''); document.getElementById('qr-setup-notice')?.setAttribute('hidden',''); }
  function showDriveQr(src)    { document.getElementById('qr-drive-loading')?.setAttribute('hidden',''); const w=document.getElementById('qr-drive-image-wrap'); if(w){w.removeAttribute('hidden'); document.getElementById('qr-drive-img').src=src;} }
  function showDriveError(msg, isSetup) { document.getElementById('qr-drive-loading')?.setAttribute('hidden',''); document.getElementById('qr-drive-image-wrap')?.setAttribute('hidden',''); if(isSetup){ document.getElementById('qr-setup-notice')?.removeAttribute('hidden'); } else { const e=document.getElementById('qr-drive-error'); if(e){e.textContent=msg; e.removeAttribute('hidden');} } }

  async function doIgUpload() {
    showIgLoading();
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/upload-ig`, { method: 'POST' });
      const data = await res.json();
      if (res.ok && data.success && data.qr_base64) {
        showIgQr(data.qr_base64);
      } else {
        showIgError(data.error || 'Gagal upload ke Instagram.');
        notify(data.error || 'Gagal upload ke Instagram Story.');
      }
    } catch (err) {
      showIgError('Gagal upload ke Instagram: ' + err.message);
      notify('Gagal upload ke Instagram: ' + err.message);
    }
  }

  async function doDriveUpload() {
    showDriveLoading();
    try {
      const res = await fetch(`/api/jobs/${currentJobId}/upload-drive`, { method: 'POST' });
      const data = await res.json();
      if (res.ok && data.success && data.qr_base64) {
        showDriveQr(data.qr_base64);
      } else {
        showDriveError(data.message || data.error || 'Gagal upload ke Drive.', !!data.needs_setup);
        notify(data.message || data.error || 'Gagal upload ke Google Drive.');
      }
    } catch (err) {
      showDriveError('Gagal upload ke Google Drive: ' + err.message, false);
      notify('Gagal upload ke Google Drive: ' + err.message);
    }
  }

  async function shareFrameToInstagramStory() {
    const frameImg = document.getElementById('frame-image');
    const src = frameImg ? frameImg.src : '';
    if (!src) {
      notify('Gambar frame belum siap.');
      return;
    }
    try {
      const res = await fetch(src);
      const blob = await res.blob();
      const file = new File([blob], 'pixora-story.jpg', { type: 'image/jpeg' });

      // Web Share API Level 2 (Android Chrome & iOS Safari):
      // Pilih "Instagram -> Stories" untuk langsung membuka aplikasi Instagram di mode edit Story!
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        await navigator.share({
          files: [file],
          title: 'Pixora Story',
        });
        return;
      }

      // Fallback jika browser tidak mendukung file share: simpan foto & buka kamera Story Instagram
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'pixora-story.jpg';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => {
        window.location.href = 'instagram://story-camera';
      }, 450);
    } catch (err) {
      if (err && err.name === 'AbortError') return;
      window.location.href = 'instagram://story-camera';
    }
  }

  function openQrModal() {
    if (!currentJobId) return;
    if (qrModal) qrModal.hidden = false;
    doDriveUpload();
  }

  document.getElementById('btn-upload-ig')?.addEventListener('click', () => {
    if (isMobile) {
      shareFrameToInstagramStory();
    } else {
      openQrModal();
    }
  });
  document.getElementById('btn-upload-drive')?.addEventListener('click', openQrModal);

  if (qrCloseBtn && qrModal) {
    qrCloseBtn.addEventListener('click', () => {
      qrModal.hidden = true;
    });
  }

  // ---------------------------------------------------------------------------
  // Restart / Reset
  // ---------------------------------------------------------------------------

  function enterMobileDirectMode() {
    guestName = 'UNESA';
    if (guestGreeting) guestGreeting.textContent = 'Halo, Sobat Seni';
    if (handPill) handPill.hidden = true;
    if (handCursor) handCursor.style.display = 'none';
    document.getElementById('welcome-screen').hidden = true;
    setMobileStep('welcome');
  }

  document.getElementById('mobile-start-btn')?.addEventListener('click', (e) => {
    const btn = e.currentTarget;
    const content = document.getElementById('mobile-welcome-content');
    const mobileEmailInput = document.getElementById('mobile-guest-email');
    const emailVal = mobileEmailInput ? mobileEmailInput.value.trim() : '';
    if (!emailVal || !emailVal.includes('@') || !emailVal.includes('.')) {
      notify('Masukkan email Google kamu terlebih dahulu (contoh: nama@gmail.com).');
      if (mobileEmailInput) mobileEmailInput.focus();
      return;
    }
    guestEmail = emailVal;
    guestName = emailVal.split('@')[0].replace(/[^a-zA-Z0-9 _.-]/g, ' ').trim().slice(0, 32) || 'Tamu';
    if (guestGreeting) guestGreeting.textContent = `Halo, ${guestName}`;

    if (btn.classList.contains('is-starting')) return;
    btn.classList.add('is-starting');
    if (content) content.classList.add('is-leaving');
    setTimeout(() => {
      btn.classList.remove('is-starting');
      if (content) content.classList.remove('is-leaving');
      setMobileStep('style');
    }, 280);
  });

  document.getElementById('mobile-back-style-btn')?.addEventListener('click', () => {
    if (captureSequenceActive) return;
    setMobileStep('style');
  });

  document.getElementById('mobile-shutter-btn')?.addEventListener('click', () => {
    triggerMobileWebFullCapture();
  });

  function resetForAnotherPhotoSession() {
    capturedPhotos.forEach(p => URL.revokeObjectURL(p.url));
    capturedPhotos = [];
    selectedForFrame.clear();
    shotCounter.textContent = `0 / ${maxPhotos}`;

    document.getElementById('result-section').hidden = true;
    document.getElementById('capture-picker').hidden = true;
    if (qrModal) qrModal.hidden = true;
    document.querySelector('.booth-grid').hidden = false;

    // Reset styles selection
    document.querySelectorAll('.style-card').forEach(c => {
      c.classList.remove('selected', 'style-popping');
      c.setAttribute('aria-pressed', 'false');
    });

    if (isMobile) {
      setMobileStep('style');
      return;
    }

    // Keep user on Screen 2 (Style Selection & Camera) on desktop/laptop
    if (video && cameraFrame && video.parentElement !== cameraFrame) {
      cameraFrame.prepend(video);
    }
    if (video) video.hidden = !cameraStream;
    if (placeholder) placeholder.hidden = !!cameraStream;
    document.getElementById('welcome-screen').hidden = true;
    document.getElementById('photobooth-app').hidden = false;
  }

  document.getElementById('start-over').addEventListener('click', resetForAnotherPhotoSession);
  if (qrTakeAgainBtn) {
    qrTakeAgainBtn.addEventListener('click', resetForAnotherPhotoSession);
  }

  // ---------------------------------------------------------------------------
  // MediaPipe Hands & Full-Page Virtual Mouse Cursor
  // ---------------------------------------------------------------------------

  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      console.warn('MediaPipe Hands not yet loaded, retrying in 400ms...');
      setTimeout(initMediaPipeHands, 400);
      return;
    }

    try {
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      hands.onResults(onHandResults);

      let isBusy = false;
      async function sendFrame() {
        const source = trackingVideo && trackingVideo.readyState >= 2 && !trackingVideo.paused
          ? trackingVideo : video;
        if (source && source.readyState >= 2 && !source.paused && !isBusy) {
          isBusy = true;
          try {
            await hands.send({ image: source });
          } catch (err) {
            console.error('Hands send error:', err);
          } finally {
            isBusy = false;
          }
        }
        requestAnimationFrame(sendFrame);
      }

      sendFrame();
    } catch (e) {
      console.error('Failed to init MediaPipe Hands:', e);
    }
  }

  function onHandResults(results) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      isPinching = false;
      committedSibiPose = '';
      resetSibiPrediction();
      if (handCursor) handCursor.style.display = 'none';
      if (handCursor) handCursor.classList.remove('pinching');
      document.body.classList.remove('hand-control-active');
      if (handPill) handPill.classList.remove('active', 'pinching');
      return;
    }

    if (handPill) handPill.classList.add('active');
    document.body.classList.add('hand-control-active');

    const landmarks = results.multiHandLandmarks[0];
    updateSibiPrediction(landmarks);

    const thumbTip = landmarks[4];
    const indexTip = landmarks[8];
    const dx = thumbTip.x - indexTip.x;
    const dy = thumbTip.y - indexTip.y;
    const distance = Math.hypot(dx, dy);
    const pinchDetected = distance < 0.075;

    if (pinchDetected && !isPinching) {
      isPinching = true;
      if (handCursor) handCursor.classList.add('pinching');
      if (handPill) handPill.classList.add('pinching');
      triggerPinchClick(cursorX, cursorY);
    } else if (!pinchDetected && isPinching) {
      isPinching = false;
      if (handCursor) handCursor.classList.remove('pinching');
      if (handPill) handPill.classList.remove('pinching');
    }

    // Map hand position across the entire browser viewport (1920x1080)
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const handZoneStart = 0.05;
    const handZoneRange = 0.90;

    if (!pinchDetected) {
      const normX = clamp01((thumbTip.x - handZoneStart) / handZoneRange);
      const normY = clamp01((thumbTip.y - handZoneStart) / handZoneRange);

      // Inverted X because webcam is mirrored
      const rawX = (1 - normX) * window.innerWidth;
      const rawY = normY * window.innerHeight;

      // Exponential smoothing for natural jitter-free motion
      cursorX += (rawX - cursorX) * 0.22;
      cursorY += (rawY - cursorY) * 0.22;
    }

    if (handCursor) {
      handCursor.style.display = 'block';
      handCursor.style.left = `${cursorX}px`;
      handCursor.style.top = `${cursorY}px`;
    }
  }

  function triggerPinchClick(x, y) {
    if (pinchCooldown) return;
    pinchCooldown = true;
    setTimeout(() => { pinchCooldown = false; }, 320);

    const el = document.elementFromPoint(x, y);
    if (!el) return;

    const clickable = el.closest('button, .style-card, .capture-choice, a, input, [role="button"]');
    const target = clickable || el;

    const mouseInit = {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: x,
      clientY: y
    };

    target.dispatchEvent(new MouseEvent('mousedown', mouseInit));
    target.dispatchEvent(new MouseEvent('mouseup', mouseInit));
    if (clickable) {
      clickable.click();
      clickable.style.transform = 'scale(0.94)';
      setTimeout(() => { clickable.style.transform = ''; }, 160);
    } else {
      target.dispatchEvent(new MouseEvent('click', mouseInit));
    }
  }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  setupStyleCards();
  if (isMobile) {
    enterMobileDirectMode();
  } else {
    loadMediaPipeScripts();
    setupPhysicalInputHandlers();
    buildVirtualKeyboard();
    loadSibiModel();
  }
  startCamera();
})();
