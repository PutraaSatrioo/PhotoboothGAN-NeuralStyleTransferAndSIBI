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
  let guestName = 'UNESA';
  let isDefaultName = true;
  let currentJobId = null;
  let currentZipUrl = null;

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
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = false;
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      video.srcObject = cameraStream;
      trackingVideo.srcObject = cameraStream;
      trackingVideo.play().catch(err => console.warn('Background tracking video notice:', err));
      video.play().catch(err => console.warn('Camera preview notice:', err));
      video.hidden = false;
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = true;
      if (placeholder) placeholder.hidden = true;
      initMediaPipeHands();
    } catch (_) {
      video.hidden = true;
      if (welcomeCameraPlaceholder) welcomeCameraPlaceholder.hidden = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Screen 1: "DONE" -> Move to Screen 2
  // ---------------------------------------------------------------------------

  const doneButton = document.getElementById('name-done');
  doneButton.addEventListener('click', () => {
    const nameInput = document.getElementById('guest-name');
    let val = nameInput.value.trim().replace(/\s+/g, ' ').slice(0, 32);
    if (!val) {
      val = 'UNESA';
      nameInput.value = 'UNESA';
    }
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
    styleCards.forEach(card => {
      card.addEventListener('click', () => {
        if (captureSequenceActive) return;
        const styleId = card.dataset.style;
        const styleName = card.dataset.name || card.querySelector('.style-name')?.textContent || styleId;
        activeStyleId = styleId;
        activeStyleName = styleName;

        styleCards.forEach(c => {
          c.classList.remove('selected');
          c.setAttribute('aria-pressed', 'false');
        });
        card.classList.add('selected');
        card.setAttribute('aria-pressed', 'true');

        // Immediately trigger 5-photo burst capture!
        startAutoBurstCapture();
      });
    });
  }

  async function startAutoBurstCapture() {
    if (captureSequenceActive) return;
    captureSequenceActive = true;
    capturedPhotos = [];
    selectedForFrame.clear();
    shotCounter.textContent = `0 / ${maxPhotos}`;
    preview.hidden = true;
    video.hidden = !cameraStream;

    notify(`Gaya ${activeStyleName} dipilih! Bersiap di depan kamera...`);

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

        // Snap photo from video feed
        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;
        const ctx = canvas.getContext('2d');
        ctx.setTransform(-1, 0, 0, 1, canvas.width, 0); // mirror horizontally
        ctx.drawImage(video, 0, 0);
        const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.94));
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
    }
  }

  function finishBurstToPicker() {
    document.querySelector('.booth-grid').hidden = true;
    const picker = document.getElementById('capture-picker');
    picker.hidden = false;
    renderCapturePicker();
    picker.scrollIntoView({ behavior: 'smooth', block: 'start' });
    notify('Foto berhasil diambil! Pilih 3 foto terbaikmu untuk frame.');
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
      confirmBtn.innerHTML = `<span>LANJUTKAN & PROSES SENI (${activeStyleName.toUpperCase()})</span> <span class="btn-arrow">➔</span>`;
    } else {
      confirmBtn.disabled = true;
      confirmBtn.innerHTML = `<span>PILIH TEPAT 3 FOTO (${count}/3)</span> <span class="btn-arrow">➔</span>`;
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
    document.getElementById('capture-picker').hidden = true;
    document.getElementById('result-section').hidden = true;
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('processing-pct').textContent = '0%';
    document.getElementById('processing-phase').textContent = `Memproses 5 foto dengan gaya ${activeStyleName}...`;
    processing.hidden = false;

    try {
      const form = new FormData();
      capturedPhotos.forEach((photo, idx) => {
        form.append('photos', photo.blob, photo.name || `capture_${idx + 1}.jpg`);
      });
      form.append('style', activeStyleId);
      form.append('selected', Array.from(selectedForFrame).sort((a, b) => a - b).join(','));
      form.append('guest_name', guestName);

      const res = await fetch('/api/captures', { method: 'POST', body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Gagal memulai pemrosesan foto.');

      currentJobId = data.job_id;
      pollJob(data.job_id);
    } catch (err) {
      processing.hidden = true;
      document.getElementById('capture-picker').hidden = false;
      notify(err.message || 'Terjadi kesalahan saat memproses foto.');
    }
  }

  async function pollJob(jobId) {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();
      if (!res.ok) throw new Error(job.error || 'Sesi tidak ditemukan.');

      if (job.status === 'error') throw new Error(job.error || 'Proses foto gagal.');

      if (job.status === 'done') {
        currentJobId = jobId;
        currentZipUrl = job.zip_url || `/results/${jobId}/studio-sela-semua-karya.zip`;
        document.getElementById('gif-image').src = `${job.gif_url}?t=${Date.now()}`;
        document.getElementById('frame-image').src = `${job.frame_url}?t=${Date.now()}`;
        document.getElementById('result-style-name').textContent = job.style;

        processing.hidden = true;
        document.getElementById('result-section').hidden = false;
        notify('Karyamu sudah siap! Klik tombol untuk membuka QR Code Google Drive.');
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
  // Google Drive Upload & QR Code Modal
  // ---------------------------------------------------------------------------

  const allDownloadBtn = document.getElementById('all-download');
  const qrModal = document.getElementById('qr-modal');
  const qrCloseBtn = document.getElementById('qr-close');
  const qrLoading = document.getElementById('qr-loading');
  const qrImageWrap = document.getElementById('qr-image-wrap');
  const qrCodeImg = document.getElementById('qr-code-img');
  const qrTakeAgainBtn = document.getElementById('qr-take-again');
  const qrDirectDownload = document.getElementById('qr-direct-download');
  const qrSetupNotice = document.getElementById('qr-setup-notice');
  const qrErrorNotice = document.getElementById('qr-error-notice');

  if (allDownloadBtn) {
    allDownloadBtn.addEventListener('click', async () => {
      if (!currentJobId) return;
      if (qrModal) qrModal.hidden = false;
      if (qrLoading) qrLoading.hidden = false;
      if (qrImageWrap) qrImageWrap.hidden = true;
      if (qrSetupNotice) qrSetupNotice.hidden = true;
      if (qrErrorNotice) qrErrorNotice.hidden = true;
      if (qrDirectDownload) {
        qrDirectDownload.href = currentZipUrl || `/results/${currentJobId}/studio-sela-semua-karya.zip`;
      }

      try {
        const res = await fetch(`/api/jobs/${currentJobId}/upload-drive`, { method: 'POST' });
        const data = await res.json();
        if (qrLoading) qrLoading.hidden = true;
        if (!res.ok || !data.success || !data.qr_base64) {
          if (qrImageWrap) qrImageWrap.hidden = true;
          if (qrSetupNotice && data.needs_setup) qrSetupNotice.hidden = false;
          if (qrErrorNotice && !data.needs_setup) {
            qrErrorNotice.textContent = data.message || data.error || 'Foto tidak berhasil diunggah ke Google Drive.';
            qrErrorNotice.hidden = false;
          }
          notify(data.message || data.error || 'Foto tidak berhasil diunggah ke Google Drive.');
          return;
        }
        if (qrImageWrap) qrImageWrap.hidden = false;

        if (data.qr_base64 && qrCodeImg) {
          qrCodeImg.src = data.qr_base64;
        }
        if (data.needs_setup && qrSetupNotice) {
          qrSetupNotice.hidden = false;
        }
      } catch (err) {
        if (qrLoading) qrLoading.hidden = true;
        if (qrImageWrap) qrImageWrap.hidden = true;
        if (qrErrorNotice) {
          qrErrorNotice.textContent = 'Gagal mengunggah ke Google Drive: ' + err.message;
          qrErrorNotice.hidden = false;
        }
        notify('Gagal mengunggah ke Google Drive: ' + err.message);
      }
    });
  }

  if (qrCloseBtn && qrModal) {
    qrCloseBtn.addEventListener('click', () => {
      qrModal.hidden = true;
    });
  }

  // ---------------------------------------------------------------------------
  // Restart / Reset
  // ---------------------------------------------------------------------------

  function resetForAnotherPhotoSession() {
    capturedPhotos.forEach(p => URL.revokeObjectURL(p.url));
    capturedPhotos = [];
    selectedForFrame.clear();
    shotCounter.textContent = `0 / ${maxPhotos}`;

    // Return camera video to Screen 1
    welcomeCameraFrame.prepend(video);

    document.getElementById('result-section').hidden = true;
    document.getElementById('capture-picker').hidden = true;
    if (qrModal) qrModal.hidden = true;
    document.querySelector('.booth-grid').hidden = false;
    document.getElementById('photobooth-app').hidden = true;
    document.getElementById('welcome-screen').hidden = false;

    // Reset styles selection
    document.querySelectorAll('.style-card').forEach(c => {
      c.classList.remove('selected');
      c.setAttribute('aria-pressed', 'false');
    });

    const nameInput = document.getElementById('guest-name');
    nameInput.value = 'UNESA';
    isDefaultName = true;
    guestName = 'UNESA';
    nameInput.focus();
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

  setupPhysicalInputHandlers();
  buildVirtualKeyboard();
  setupStyleCards();
  loadSibiModel();
  startCamera();
})();
