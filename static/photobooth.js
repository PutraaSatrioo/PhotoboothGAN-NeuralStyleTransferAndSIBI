(() => {
  const video = document.getElementById('camera');
  const welcomeCameraPlaceholder = document.getElementById('welcome-camera-placeholder');
  const preview = document.getElementById('upload-preview');
  const cameraFlash = document.getElementById('camera-flash');
  const placeholder = document.getElementById('camera-placeholder');
  const shutter = document.getElementById('shutter');
  const countdown = document.getElementById('countdown');
  const fileInput = document.getElementById('photo-upload');
  const createButton = document.getElementById('create-button');
  const processing = document.getElementById('processing');
  const toast = document.getElementById('toast');
  const handCursor = document.getElementById('hand-cursor');
  const handPill = document.getElementById('hand-pill');
  const selectedStyle = () => document.querySelector('.style-card.selected').dataset.style;
  let cameraStream = null;
  const maxPhotos = 5;
  const framePhotoCount = 3;
  let capturedPhotos = [];
  let selectedForFrame = new Set();
  let captureSequenceActive = false;
  let guestName = '';
  let sibiModel = null;
  let sibiCandidate = '';
  let sibiCandidateSince = 0;
  let committedSibiPose = '';
  const sibiConfirmMs = 3000;
  const sibiConfidenceThreshold = 0.72;
  let toastTimer = null;

  let isPinching = false;
  let pinchCooldown = false;
  let cursorX = window.innerWidth / 2;
  let cursorY = window.innerHeight / 2;

  async function loadSibiModel() {
    try {
      const response = await fetch('/api/sibi/model');
      if (!response.ok) throw new Error('model unavailable');
      sibiModel = await response.json();
      const status = document.getElementById('sibi-status');
      status.textContent = 'MODEL SIBI SIAP · 24 HURUF STATIS';
      document.getElementById('sibi-status-dot').classList.add('active');
    } catch (_) {
      sibiModel = null;
      document.getElementById('sibi-status').textContent = 'MODEL SIBI TIDAK TERSEDIA · GUNAKAN KEYBOARD';
    }
  }

  function denseLayer(input, kernel, bias, relu) {
    const output = new Array(bias.length);
    for (let column = 0; column < bias.length; column++) {
      let value = bias[column];
      for (let row = 0; row < input.length; row++) value += input[row] * kernel[row][column];
      output[column] = relu ? Math.max(0, value) : value;
    }
    return output;
  }

  function classifySibi(landmarks) {
    if (!sibiModel || !landmarks || landmarks.length !== 21) return null;
    const xs = landmarks.map(point => point.x);
    const ys = landmarks.map(point => point.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const features = [];
    landmarks.forEach(point => features.push(point.x - minX, point.y - minY));
    const weights = sibiModel.layers;
    let values = denseLayer(features, weights[0], weights[1], true);
    const gamma = weights[2];
    const beta = weights[3];
    const mean = weights[4];
    const variance = weights[5];
    values = values.map((value, index) => ((value - mean[index]) / Math.sqrt(variance[index] + 0.001)) * gamma[index] + beta[index]);
    values = denseLayer(values, weights[6], weights[7], true);
    values = denseLayer(values, weights[8], weights[9], false);
    const maxLogit = Math.max(...values);
    const probabilities = values.map(value => Math.exp(value - maxLogit));
    const total = probabilities.reduce((sum, value) => sum + value, 0);
    const best = probabilities.reduce((bestIndex, value, index) => value > probabilities[bestIndex] ? index : bestIndex, 0);
    return {letter: sibiModel.classes[best], confidence: probabilities[best] / total};
  }

  function resetSibiPrediction(message) {
    sibiCandidate = '';
    sibiCandidateSince = 0;
    document.getElementById('sibi-confidence').textContent = 'menunggu tangan';
    document.getElementById('sibi-progress-bar').style.width = '0%';
    document.getElementById('sibi-confirm-hint').textContent = message || 'Tunjukkan satu huruf SIBI; prediksi akan tampil di sini.';
    if (message) document.getElementById('sibi-status').textContent = message;
  }

  function updateSibiPrediction(landmarks) {
    if (document.getElementById('welcome-screen').hidden || !sibiModel) return;
    const prediction = classifySibi(landmarks);
    if (!prediction || prediction.confidence < sibiConfidenceThreshold) {
      resetSibiPrediction('TANGAN TERDETEKSI · PREDIKSI BELUM YAKIN');
      return;
    }

    if (prediction.letter === committedSibiPose) {
      document.getElementById('sibi-status').textContent = `HURUF ${prediction.letter} SUDAH MASUK · UBAH POSE UNTUK HURUF BERIKUTNYA`;
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
    document.getElementById('sibi-confidence').textContent = `${Math.round(prediction.confidence * 100)}% yakin`;
    document.getElementById('sibi-progress-bar').style.width = `${progress * 100}%`;
    document.getElementById('sibi-confirm-hint').textContent = progress >= 1
      ? 'Huruf masuk. Lepaskan tangan sebentar sebelum mengeja huruf yang sama.'
      : `Pastikan hurufnya benar · masuk otomatis dalam ${Math.ceil((1 - progress) * 3)} detik`;
    document.getElementById('sibi-status').textContent = `PREDIKSI ${prediction.letter} · TAHAN POSE UNTUK KONFIRMASI`;

    if (progress >= 1) {
      insertNameText(prediction.letter);
      committedSibiPose = prediction.letter;
      sibiCandidate = '';
      sibiCandidateSince = 0;
    }
  }

  function notify(message) {
    toast.textContent = message;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
  }

  function updateShotCounter() {
    document.getElementById('shot-counter').textContent = `${capturedPhotos.length} / ${maxPhotos}`;
  }

  function insertNameText(text) {
    const input = document.getElementById('guest-name');
    const start = input.selectionStart || 0;
    const end = input.selectionEnd || 0;
    input.setRangeText(text, start, end, 'end');
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.focus();
  }

  function buildVirtualKeyboard() {
    const keyboard = document.getElementById('virtual-keyboard');
    [['QWERTYUIOP'], ['ASDFGHJKL'], ['ZXCVBNM']].forEach(row => {
      const rowElement = document.createElement('div');
      rowElement.className = 'key-row';
      [...row[0]].forEach(letter => {
        const key = document.createElement('button');
        key.type = 'button';
        key.className = 'letter-key';
        key.textContent = letter;
        key.addEventListener('click', () => insertNameText(letter));
        rowElement.appendChild(key);
      });
      keyboard.appendChild(rowElement);
    });
    const controls = document.createElement('div');
    controls.className = 'key-row key-controls';
    [['⌫', 'backspace'], ['SPASI', 'space'], ['BERSIHKAN', 'clear']].forEach(([label, action]) => {
      const key = document.createElement('button');
      key.type = 'button';
      key.className = 'function-key';
      key.textContent = label;
      key.addEventListener('click', () => {
        const input = document.getElementById('guest-name');
        if (action === 'space') insertNameText(' ');
        if (action === 'clear') {
          input.value = '';
          input.focus();
        }
        if (action === 'backspace') {
          const start = input.selectionStart || 0;
          const end = input.selectionEnd || 0;
          if (start === end && start > 0) input.setSelectionRange(start - 1, end);
          input.setRangeText('', input.selectionStart, input.selectionEnd, 'end');
          input.focus();
        }
      });
      controls.appendChild(key);
    });
    keyboard.appendChild(controls);
  }

  function updateSelectionCount() {
    const count = selectedForFrame.size;
    document.getElementById('selection-count').textContent = `${count} / ${framePhotoCount} DIPILIH`;
    const confirm = document.getElementById('confirm-selection');
    confirm.disabled = count !== framePhotoCount;
    confirm.querySelector('span').textContent = count === framePhotoCount
      ? 'LANJUTKAN DENGAN 3 FOTO' : `PILIH TEPAT 3 FOTO (${count}/3)`;
  }

  function renderCapturePicker() {
    const grid = document.getElementById('capture-grid');
    grid.replaceChildren();
    capturedPhotos.forEach((photo, index) => {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'capture-choice';
      card.setAttribute('aria-pressed', selectedForFrame.has(index) ? 'true' : 'false');
      if (selectedForFrame.has(index)) card.classList.add('selected');
      card.innerHTML = `<img alt="Foto ${index + 1}" src="${photo.url}"><span class="capture-number">0${index + 1}</span><span class="capture-check">✓</span>`;
      card.addEventListener('click', () => {
        if (selectedForFrame.has(index)) {
          selectedForFrame.delete(index);
        } else if (selectedForFrame.size < framePhotoCount) {
          selectedForFrame.add(index);
        } else {
          notify('Pilih tepat 3 foto untuk frame.');
          return;
        }
        card.classList.toggle('selected', selectedForFrame.has(index));
        card.setAttribute('aria-pressed', selectedForFrame.has(index) ? 'true' : 'false');
        updateSelectionCount();
      });
      grid.appendChild(card);
    });
    updateSelectionCount();
  }

  function finishCaptureSet() {
    if (capturedPhotos.length !== maxPhotos) return;
    shutter.disabled = true;
    document.getElementById('capture-picker').hidden = false;
    document.getElementById('create-button').hidden = true;
    document.getElementById('edit-selection').hidden = true;
    renderCapturePicker();
    document.getElementById('capture-picker').scrollIntoView({behavior: 'smooth', block: 'start'});
  }

  function addUploadedPhotos(files) {
    const available = maxPhotos - capturedPhotos.length;
    const validFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    validFiles.slice(0, available).forEach(file => {
      capturedPhotos.push({blob: file, url: URL.createObjectURL(file), name: file.name});
    });
    updateShotCounter();
    if (validFiles.length > available) notify('Sesi ini hanya menerima 5 foto.');
    if (capturedPhotos.length) {
      preview.src = capturedPhotos[capturedPhotos.length - 1].url;
      preview.hidden = false;
      video.hidden = !cameraStream;
      if (cameraStream) video.classList.add('tracking-in-background');
      placeholder.hidden = true;
    }
    if (capturedPhotos.length === maxPhotos) finishCaptureSet();
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      document.getElementById('camera-state').lastChild.textContent = ' UNGGAH FOTO';
      document.getElementById('sibi-status').textContent = 'KAMERA TIDAK TERSEDIA · GUNAKAN KEYBOARD';
      welcomeCameraPlaceholder.hidden = false;
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}, audio: false});
      video.srcObject = cameraStream;
      video.hidden = false;
      welcomeCameraPlaceholder.hidden = true;
      placeholder.hidden = true;
      document.getElementById('camera-state').classList.add('active');
      document.getElementById('sibi-status').textContent = sibiModel
        ? 'KAMERA LIVE · MODEL SIBI 24 HURUF SIAP' : 'KAMERA LIVE · MODEL SIBI DIPERSIAPKAN';
      document.getElementById('sibi-status-dot').classList.add('active');
      initMediaPipeHands();
    } catch (_) {
      video.hidden = true;
      document.getElementById('sibi-status').textContent = 'IZINKAN KAMERA UNTUK INPUT SIBI';
      document.getElementById('sibi-status-dot').classList.remove('active');
      welcomeCameraPlaceholder.hidden = false;
    }
  }

  shutter.addEventListener('click', async () => {
    if (captureSequenceActive || capturedPhotos.length >= maxPhotos) return;
    if (!cameraStream || video.readyState < 2) {
      fileInput.click();
      return;
    }
    captureSequenceActive = true;
    shutter.disabled = true;
    preview.hidden = true;
    video.classList.remove('tracking-in-background');
    video.hidden = false;
    try {
      const canvas = document.getElementById('capture-canvas');
      for (let shot = capturedPhotos.length; shot < maxPhotos; shot++) {
        for (const number of ['3', '2', '1']) {
          countdown.textContent = number;
          countdown.hidden = false;
          await new Promise(resolve => setTimeout(resolve, 650));
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const context = canvas.getContext('2d');
        context.setTransform(-1, 0, 0, 1, canvas.width, 0);
        context.drawImage(video, 0, 0);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .92));
        if (!blob) throw new Error('Foto tidak berhasil diambil. Coba lagi.');
        capturedPhotos.push({blob, url: URL.createObjectURL(blob), name: `capture_${shot + 1}.jpg`});
        updateShotCounter();
        cameraFlash.classList.remove('active');
        void cameraFlash.offsetWidth;
        cameraFlash.classList.add('active');
        await new Promise(resolve => setTimeout(resolve, 1000));
        cameraFlash.classList.remove('active');
      }
      countdown.hidden = true;
      preview.src = capturedPhotos[capturedPhotos.length - 1].url;
      preview.hidden = false;
      video.classList.add('tracking-in-background');
      finishCaptureSet();
    } catch (error) {
      notify(error.message || 'Tidak bisa mengambil rangkaian foto.');
      shutter.disabled = false;
    } finally {
      captureSequenceActive = false;
      countdown.hidden = true;
    }
  });

  fileInput.addEventListener('change', event => {
    addUploadedPhotos(event.target.files);
    fileInput.value = '';
  });
  document.querySelectorAll('.style-card').forEach(card => card.addEventListener('click', () => {
    document.querySelectorAll('.style-card').forEach(item => {
      item.classList.remove('selected');
      item.setAttribute('aria-pressed', 'false');
    });
    card.classList.add('selected');
    card.setAttribute('aria-pressed', 'true');
  }));

  async function poll(jobId) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || 'Sesi photobooth tidak ditemukan.');
    if (job.status === 'error') throw new Error(job.error || 'Proses foto gagal.');
    if (job.status === 'done') {
      document.getElementById('gif-image').src = `${job.gif_url}?t=${Date.now()}`;
      document.getElementById('gif-download').href = job.gif_url;
      document.getElementById('frame-image').src = `${job.frame_url}?t=${Date.now()}`;
      document.getElementById('frame-download').href = job.frame_url;
      document.getElementById('result-style-name').textContent = job.style;
      processing.hidden = true;
      document.getElementById('result-section').hidden = false;
      document.getElementById('result-section').scrollIntoView({behavior: 'smooth'});
      return;
    }
    // Update progress bar with real percentage from server
    const pct = job.progress || 0;
    const bar = document.getElementById('progress-bar');
    const pctLabel = document.getElementById('processing-pct');
    const phaseLabel = document.getElementById('processing-phase');
    if (bar) bar.style.width = pct + '%';
    if (pctLabel) pctLabel.textContent = pct + '%';
    if (phaseLabel) phaseLabel.textContent = job.phase || 'Sedang memproses foto...';
    setTimeout(() => poll(jobId).catch(handleError), 2000);
  }

  function handleError(error) {
    processing.hidden = true;
    createButton.disabled = false;
    notify(error.message || 'Tidak bisa memproses foto.');
  }

  createButton.addEventListener('click', async () => {
    if (capturedPhotos.length !== maxPhotos || selectedForFrame.size !== framePhotoCount) {
      notify('Siapkan 5 foto dan pilih tepat 3 untuk frame.');
      return;
    }
    createButton.disabled = true;
    document.getElementById('result-section').hidden = true;
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('processing-pct').textContent = '0%';
    document.getElementById('processing-phase').textContent = 'Menyiapkan 5 foto...';
    processing.hidden = false;
    try {
      const form = new FormData();
      capturedPhotos.forEach((photo, index) => {
        form.append('photos', photo.blob, photo.name || `capture_${index + 1}.jpg`);
      });
      form.append('style', selectedStyle());
      form.append('selected', Array.from(selectedForFrame).sort((a, b) => a - b).join(','));
      form.append('guest_name', guestName);
      const response = await fetch('/api/captures', {method: 'POST', body: form});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Foto tidak dapat diproses.');
      poll(data.job_id).catch(handleError);
    } catch (error) {
      handleError(error);
    }
  });

  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      console.warn('MediaPipe Hands library not loaded yet, retrying in 500ms...');
      setTimeout(initMediaPipeHands, 500);
      return;
    }

    try {
      const hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 0, // 0 is much faster for real-time webcam tracking
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      hands.onResults(onHandResults);

      let isProcessing = false;
      async function sendFrame() {
        if (video && video.readyState >= 2 && !video.paused && !isProcessing) {
          isProcessing = true;
          try {
            await hands.send({image: video});
          } catch (err) {
            console.error('Hands send error:', err);
          } finally {
            isProcessing = false;
          }
        }
        requestAnimationFrame(sendFrame);
      }

      sendFrame();
      console.log('MediaPipe Hands initialized successfully.');
    } catch (e) {
      console.error('Failed to initialize MediaPipe Hands:', e);
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
      if (!sibiModel) document.getElementById('sibi-status').textContent = cameraStream
        ? 'KAMERA LIVE · MODEL SIBI TIDAK TERSEDIA' : 'IZINKAN KAMERA UNTUK INPUT SIBI';
      else if (!document.getElementById('welcome-screen').hidden) {
        document.getElementById('sibi-status').textContent = 'MODEL SIBI SIAP · TUNJUKKAN SATU HURUF';
      }
      if (handPill) {
        handPill.classList.remove('active', 'pinching');
      }
      return;
    }

    if (handPill) handPill.classList.add('active');
    document.body.classList.add('hand-control-active');

    const landmarks = results.multiHandLandmarks[0];
    updateSibiPrediction(landmarks);
    const thumbTip = landmarks[4];   // Thumb tip
    const indexTip = landmarks[8];   // Index finger tip
    const dx = thumbTip.x - indexTip.x;
    const dy = thumbTip.y - indexTip.y;
    const pinchDetected = Math.hypot(dx, dy) < 0.075;

    // Keep the last cursor position while pinching. The index finger moves
    // toward the thumb during a click, so following it here causes click drift.
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

    // Use 85% of the previous hand-motion zone, then map it to the full page.
    // The cursor reaches page edges with less hand travel (especially the bottom controls).
    const clamp01 = (v) => Math.max(0, Math.min(1, v));
    const handReachScale = 0.85;
    const handZoneStart = 0.15;
    const handZoneRange = 0.70 * handReachScale;
    if (!pinchDetected) {
      const normX = clamp01((indexTip.x - handZoneStart) / handZoneRange);
      const normY = clamp01((indexTip.y - handZoneStart) / handZoneRange);

      // Mirroring X coordinate because webcam view is flipped horizontally like a mirror
      const rawX = (1 - normX) * window.innerWidth;
      const rawY = normY * window.innerHeight;

      // Smooth cursor movement (Lerp)
      cursorX += (rawX - cursorX) * 0.40;
      cursorY += (rawY - cursorY) * 0.40;
    }

    if (handCursor) {
      handCursor.removeAttribute('hidden');
      handCursor.style.display = 'block';
      handCursor.style.left = `${cursorX}px`;
      handCursor.style.top = `${cursorY}px`;
    }

    // Auto-scroll page when hand cursor moves near top or bottom edges
    if (!pinchDetected && cursorY > window.innerHeight - 80) {
      window.scrollBy({ top: 12, behavior: 'auto' });
    } else if (!pinchDetected && cursorY < 80) {
      window.scrollBy({ top: -12, behavior: 'auto' });
    }
  }

  function triggerPinchClick(x, y) {
    if (pinchCooldown) return;
    pinchCooldown = true;
    setTimeout(() => { pinchCooldown = false; }, 350);

    // Find any element underneath the hand cursor
    const element = document.elementFromPoint(x, y);
    if (!element) return;

    const clickable = element.closest('button, .style-card, .capture-choice, .upload-button, .download-button, a, input, [role="button"]');
    const target = clickable || element;

    // Send one click to the nearest control. Dispatching on a child and then
    // calling click() on its parent would toggle selection twice.
    const mouseEventInit = {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: x,
      clientY: y
    };
    target.dispatchEvent(new MouseEvent('mousedown', mouseEventInit));
    target.dispatchEvent(new MouseEvent('mouseup', mouseEventInit));
    if (clickable) {
      clickable.click();
      clickable.style.transform = 'scale(0.92)';
      setTimeout(() => {
        clickable.style.transform = '';
      }, 180);
    } else {
      target.dispatchEvent(new MouseEvent('click', mouseEventInit));
    }
  }

  document.getElementById('confirm-selection').addEventListener('click', () => {
    if (selectedForFrame.size !== framePhotoCount) return;
    document.getElementById('capture-picker').hidden = true;
    createButton.hidden = false;
    createButton.disabled = false;
    document.getElementById('edit-selection').hidden = false;
    document.querySelector('.style-column').scrollIntoView({behavior: 'smooth', block: 'center'});
  });

  document.getElementById('edit-selection').addEventListener('click', () => {
    document.getElementById('capture-picker').hidden = false;
    createButton.hidden = true;
    document.getElementById('edit-selection').hidden = true;
    document.getElementById('capture-picker').scrollIntoView({behavior: 'smooth', block: 'start'});
  });

  document.getElementById('start-over').addEventListener('click', () => {
    capturedPhotos.forEach(photo => URL.revokeObjectURL(photo.url));
    capturedPhotos = [];
    selectedForFrame = new Set();
    updateShotCounter();
    document.getElementById('capture-picker').hidden = true;
    document.getElementById('capture-grid').replaceChildren();
    document.getElementById('selection-count').textContent = '0 / 3 DIPILIH';
    document.getElementById('confirm-selection').disabled = true;
    document.getElementById('confirm-selection').querySelector('span').textContent = 'PILIH TEPAT 3 FOTO (0/3)';
    createButton.hidden = true;
    createButton.disabled = false;
    document.getElementById('edit-selection').hidden = true;
    shutter.disabled = false;
    preview.hidden = true;
    preview.removeAttribute('src');
    fileInput.value = '';
    video.classList.remove('tracking-in-background');
    video.hidden = !cameraStream;
    placeholder.hidden = !!cameraStream;
    document.getElementById('result-section').hidden = true;
    createButton.disabled = false;
    window.scrollTo({top: 0, behavior: 'smooth'});
  });
  document.getElementById('name-form').addEventListener('submit', event => {
    event.preventDefault();
    const nameInput = document.getElementById('guest-name');
    guestName = nameInput.value.trim().replace(/\s+/g, ' ').slice(0, 32);
    if (!guestName) {
      nameInput.focus();
      return;
    }
    nameInput.value = guestName;
    document.getElementById('camera-frame').insertBefore(video, preview);
    video.hidden = !cameraStream;
    placeholder.hidden = !!cameraStream;
    document.getElementById('welcome-screen').hidden = true;
    document.getElementById('photobooth-app').hidden = false;
  });
  buildVirtualKeyboard();
  loadSibiModel();
  startCamera();
})();
