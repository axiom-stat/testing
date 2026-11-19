(() => {
  'use strict';

  const DEFECT_LABELS_RU = {
    crack: 'Трещина / цек',
    chip: 'Скол кромки',
    glaze_bare: 'Плешина / сборка глазури',
    pinhole_speck: 'Накол / мушка',
    glaze_drip: 'Натек / прыщ',
    shape_warp: 'Перекос контура',
  };

  const ZONE_LABELS_RU = {
    rim: 'Кромка',
    interior: 'Поле',
  };

  const STAGE_LABELS_RU = {
    bisque: 'Утиль 900°C',
    glazed: 'Глазурь 1350°C',
  };

  const state = {
    activeMode: 'inspect',
    stage: 'bisque',
    inspectFile: null,
    pendingFilePromise: null,
    inspectRequestSeq: 0,
    demoFetchSeq: 0,
    originalObjectUrl: null,
    passportAction: 'verify',
    passportProbeFile: null,
    passportRequestSeq: 0,
    verifyRequestSeq: 0,
    analyticsRequestSeq: 0,
    probeObjectUrl: null,
    isInspecting: false,
    isVerifying: false,
    isRegistering: false,
  };

  const countUpRafIds = new WeakMap();

  const $ = (id) => document.getElementById(id);

  function announce(message) {
    const el = $('status-announcer');
    if (el) {
      el.textContent = message;
    }
  }

  function escapeText(value, fallback = '') {
    if (value === null || value === undefined) {
      return fallback;
    }
    return String(value);
  }

  function formatTime(isoString) {
    if (!isoString) {
      return '-';
    }
    const parsed = new Date(isoString);
    if (Number.isNaN(parsed.getTime())) {
      return String(isoString).slice(0, 16);
    }
    return parsed.toLocaleTimeString('ru-RU', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  }

  function prefersReducedMotion() {
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function cancelCountUp(el) {
    if (!el) {
      return;
    }
    const prevRafId = countUpRafIds.get(el);
    if (prevRafId !== undefined) {
      cancelAnimationFrame(prevRafId);
      countUpRafIds.delete(el);
    }
  }

  function animateCountUp(el, targetValue, durationMs, suffix, decimals) {
    if (!el) {
      return;
    }
    cancelCountUp(el);
    const finalText = `${targetValue.toFixed(decimals)}${suffix}`;
    if (prefersReducedMotion() || durationMs <= 0) {
      el.textContent = finalText;
      return;
    }
    const startTime = performance.now();
    function tick(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / durationMs);
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = targetValue * eased;
      el.textContent = `${current.toFixed(decimals)}${suffix}`;
      if (progress < 1) {
        const nextId = requestAnimationFrame(tick);
        countUpRafIds.set(el, nextId);
      } else {
        countUpRafIds.delete(el);
        el.textContent = finalText;
      }
    }
    const startId = requestAnimationFrame(tick);
    countUpRafIds.set(el, startId);
  }

  async function extractErrorDetail(response, fallbackMessage) {
    try {
      const payload = await response.json();
      if (payload && typeof payload.detail === 'string' && payload.detail.trim()) {
        return payload.detail.trim();
      }
    } catch (_) {
      /* ignore non-JSON error body */
    }
    return `${fallbackMessage} (HTTP ${response.status})`;
  }

  async function fetchImageAsFile(url, filename) {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Не удалось загрузить изображение (${resp.status})`);
    }
    const blob = await resp.blob();
    return new File([blob], filename, { type: blob.type || 'image/png' });
  }

  function createVerdictIcon(isAuthentic) {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('width', '18');
    svg.setAttribute('height', '18');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.75');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');

    if (isAuthentic) {
      const circle = document.createElementNS(ns, 'circle');
      circle.setAttribute('cx', '12');
      circle.setAttribute('cy', '12');
      circle.setAttribute('r', '9');
      const poly = document.createElementNS(ns, 'polyline');
      poly.setAttribute('points', '9 12 11 14 15 10');
      svg.appendChild(circle);
      svg.appendChild(poly);
    } else {
      const l1 = document.createElementNS(ns, 'line');
      l1.setAttribute('x1', '18');
      l1.setAttribute('y1', '6');
      l1.setAttribute('x2', '6');
      l1.setAttribute('y2', '18');
      const l2 = document.createElementNS(ns, 'line');
      l2.setAttribute('x1', '6');
      l2.setAttribute('y1', '6');
      l2.setAttribute('x2', '18');
      l2.setAttribute('y2', '18');
      svg.appendChild(l1);
      svg.appendChild(l2);
    }
    return svg;
  }

  /* -----------------------------------------------------------------------
     Health & Demo Catalog
     ----------------------------------------------------------------------- */
  async function loadHealth() {
    const pill = $('api-health-pill');
    const textEl = $('api-health-text');
    if (!pill || !textEl) {
      return;
    }
    try {
      const resp = await fetch('/health');
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }
      const data = await resp.json();
      if (data.status === 'ok') {
        pill.dataset.status = 'ok';
        textEl.textContent = data.model_mode === 'onnx' ? 'ONNX CPU' : 'OpenCV CPU';
      } else {
        pill.dataset.status = 'offline';
        textEl.textContent = 'Сервис недоступен';
      }
    } catch (_) {
      pill.dataset.status = 'offline';
      textEl.textContent = 'Сервис недоступен';
    }
  }

  async function loadDemoSamples() {
    const sampleList = $('demo-sample-list');
    const passportList = $('demo-passport-list');
    try {
      const resp = await fetch('/api/demo-samples');
      if (!resp.ok) {
        return;
      }
      const data = await resp.json();

      if (sampleList && Array.isArray(data.samples)) {
        sampleList.textContent = '';
        for (const sample of data.samples) {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'chip-btn';
          btn.dataset.sampleId = sample.id;
          btn.dataset.stage = sample.stage || 'glazed';
          btn.dataset.imageUrl = sample.image_url;
          btn.setAttribute('aria-pressed', 'false');
          btn.title = sample.label;
          btn.textContent = sample.label;
          btn.addEventListener('click', () => handleDemoSampleClick(btn, sample));
          sampleList.appendChild(btn);
        }
      }

      if (passportList && Array.isArray(data.passports)) {
        passportList.textContent = '';
        for (const item of data.passports) {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'chip-btn';
          btn.dataset.passportId = item.item_id;
          btn.dataset.probeUrl = item.probe_image_url;
          btn.setAttribute('aria-pressed', 'false');
          btn.title = item.label;
          btn.textContent = item.label;
          btn.addEventListener('click', () => handleDemoPassportClick(btn, item));
          passportList.appendChild(btn);
        }
      }
    } catch (_) {
      /* Demo catalog stays optional if offline */
    }
  }

  /* -----------------------------------------------------------------------
     Navigation Tabs
     ----------------------------------------------------------------------- */
  function switchMode(mode) {
    const modes = ['inspect', 'passport', 'dashboard'];
    const idx = modes.indexOf(mode);
    if (idx === -1) {
      return;
    }
    state.activeMode = mode;

    const tabsEl = $('mode-tabs');
    if (tabsEl) {
      tabsEl.style.setProperty('--tab-index', String(idx));
    }

    const tabButtons = document.querySelectorAll('#mode-tabs [role="tab"]');
    tabButtons.forEach((btn) => {
      const selected = btn.dataset.mode === mode;
      btn.setAttribute('aria-selected', selected ? 'true' : 'false');
    });

    const panels = {
      inspect: $('panel-inspect'),
      passport: $('panel-passport'),
      dashboard: $('panel-dashboard'),
    };
    Object.entries(panels).forEach(([key, panel]) => {
      if (!panel) {
        return;
      }
      const active = key === mode;
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
      if (active) {
        panel.querySelectorAll('.reveal').forEach((el) => el.classList.add('is-visible'));
      }
    });

    if (mode === 'dashboard') {
      loadAnalytics();
    }
  }

  function initTabs() {
    const tabButtons = Array.from(document.querySelectorAll('#mode-tabs [role="tab"]'));
    tabButtons.forEach((btn, idx) => {
      btn.addEventListener('click', () => switchMode(btn.dataset.mode));
      btn.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          const nextIdx = e.key === 'ArrowRight'
            ? (idx + 1) % tabButtons.length
            : (idx - 1 + tabButtons.length) % tabButtons.length;
          tabButtons[nextIdx].focus();
          switchMode(tabButtons[nextIdx].dataset.mode);
        }
      });
    });
  }

  /* -----------------------------------------------------------------------
     Mode 1: Inspect & Split Comparison
     ----------------------------------------------------------------------- */
  function setStage(stage) {
    state.stage = stage === 'bisque' ? 'bisque' : 'glazed';
    const bisqueBtn = $('stage-bisque');
    const glazedBtn = $('stage-glazed');
    if (bisqueBtn) {
      bisqueBtn.setAttribute('aria-checked', state.stage === 'bisque' ? 'true' : 'false');
    }
    if (glazedBtn) {
      glazedBtn.setAttribute('aria-checked', state.stage === 'glazed' ? 'true' : 'false');
    }
  }

  function applyInspectFile(file, previewUrlOverride) {
    if (!file) {
      return;
    }
    if (state.originalObjectUrl) {
      URL.revokeObjectURL(state.originalObjectUrl);
      state.originalObjectUrl = null;
    }
    state.inspectFile = file;
    state.originalObjectUrl = URL.createObjectURL(file);

    const dropzone = $('inspect-dropzone');
    if (dropzone) {
      dropzone.dataset.state = 'ready';
    }
    const nameEl = $('inspect-file-name');
    if (nameEl) {
      nameEl.textContent = file.name || 'Выбранный снимок';
    }
    const thumbEl = $('inspect-preview-thumb');
    if (thumbEl) {
      thumbEl.src = previewUrlOverride || state.originalObjectUrl;
      thumbEl.hidden = false;
    }
    const errEl = $('inspect-error');
    if (errEl) {
      errEl.hidden = true;
      errEl.textContent = '';
    }
  }

  function resetInspectFlightState() {
    state.isInspecting = false;
    const runBtn = $('btn-run-inspect');
    if (runBtn) {
      runBtn.disabled = false;
    }
    const skeletonEl = $('inspect-skeleton');
    if (skeletonEl) {
      skeletonEl.hidden = true;
    }
  }

  function setInspectFile(file, previewUrlOverride) {
    if (!file) {
      return;
    }
    state.inspectRequestSeq += 1;
    state.demoFetchSeq += 1;
    state.pendingFilePromise = null;
    resetInspectFlightState();
    applyInspectFile(file, previewUrlOverride);
  }

  function handleDemoSampleClick(btn, sample) {
    state.inspectRequestSeq += 1;
    const fetchSeq = ++state.demoFetchSeq;
    resetInspectFlightState();
    document.querySelectorAll('#demo-sample-list .chip-btn').forEach((chip) => {
      chip.setAttribute('aria-pressed', chip === btn ? 'true' : 'false');
    });
    if (sample.stage === 'bisque' || sample.stage === 'glazed') {
      setStage(sample.stage);
    }

    const dropzone = $('inspect-dropzone');
    if (dropzone) {
      dropzone.dataset.state = 'ready';
    }
    const nameEl = $('inspect-file-name');
    if (nameEl) {
      nameEl.textContent = sample.label;
    }
    const thumbEl = $('inspect-preview-thumb');
    if (thumbEl) {
      thumbEl.src = sample.image_url;
      thumbEl.hidden = false;
    }
    const errEl = $('inspect-error');
    if (errEl) {
      errEl.hidden = true;
      errEl.textContent = '';
    }

    state.inspectFile = null;
    const pendingPromise = fetchImageAsFile(sample.image_url, `${sample.id}.png`)
      .then((file) => {
        if (fetchSeq !== state.demoFetchSeq) {
          return null;
        }
        applyInspectFile(file, sample.image_url);
        if (state.pendingFilePromise === pendingPromise) {
          state.pendingFilePromise = null;
        }
        return file;
      })
      .catch((err) => {
        if (fetchSeq !== state.demoFetchSeq) {
          return null;
        }
        if (state.pendingFilePromise === pendingPromise) {
          state.pendingFilePromise = null;
        }
        state.inspectFile = null;
        if (dropzone) {
          dropzone.dataset.state = 'idle';
        }
        if (nameEl) {
          nameEl.textContent = 'Перетащите кадр или выберите файл';
        }
        if (thumbEl) {
          thumbEl.hidden = true;
          thumbEl.removeAttribute('src');
        }
        const resultEl = $('inspect-result');
        if (resultEl) {
          resultEl.hidden = true;
        }
        if (errEl) {
          errEl.textContent = err.message || 'Ошибка загрузки образца';
          errEl.hidden = false;
        }
        return null;
      });
    state.pendingFilePromise = pendingPromise;
  }

  function setCompareView(viewMode) {
    const viewport = $('compare-viewport');
    if (viewport) {
      viewport.dataset.view = viewMode;
    }
    document.querySelectorAll('#compare-mode-group [data-compare-mode]').forEach((btn) => {
      btn.setAttribute('aria-pressed', btn.dataset.compareMode === viewMode ? 'true' : 'false');
    });
  }

  function setWipePercent(percent) {
    const clamped = Math.max(0, Math.min(100, percent));
    const viewport = $('compare-viewport');
    const range = $('compare-range');
    if (viewport) {
      viewport.style.setProperty('--wipe', `${clamped}%`);
    }
    if (range) {
      range.value = String(Math.round(clamped));
    }
  }

  function renderDefectList(defects) {
    const listEl = $('inspect-defect-list');
    const countEl = $('inspect-defect-count');
    if (!listEl) {
      return;
    }
    listEl.textContent = '';
    const items = Array.isArray(defects) ? defects : [];
    if (countEl) {
      countEl.textContent = String(items.length);
    }

    if (items.length === 0) {
      const li = document.createElement('li');
      li.className = 'defect-row';
      const nameSpan = document.createElement('span');
      nameSpan.className = 'defect-name';
      nameSpan.textContent = 'Дефекты не выявлены';
      const zoneSpan = document.createElement('span');
      zoneSpan.className = 'defect-zone';
      zoneSpan.textContent = 'ГОСТ 28390-89';
      const measureSpan = document.createElement('span');
      measureSpan.className = 'defect-measure mono-num';
      measureSpan.textContent = '1 сорт';
      li.append(nameSpan, zoneSpan, measureSpan);
      listEl.appendChild(li);
      return;
    }

    for (const defect of items) {
      const li = document.createElement('li');
      li.className = 'defect-row';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'defect-name';
      nameSpan.textContent = defect.gost_name_ru || DEFECT_LABELS_RU[defect.defect_class] || defect.defect_class;

      const zoneSpan = document.createElement('span');
      zoneSpan.className = 'defect-zone';
      zoneSpan.textContent = ZONE_LABELS_RU[defect.zone] || escapeText(defect.zone, '-');

      const measureSpan = document.createElement('span');
      measureSpan.className = 'defect-measure mono-num';
      const lengthMm = Number(defect.length_mm || 0);
      const areaMm2 = Number(defect.area_mm2 || 0);
      if (lengthMm > 0 && defect.defect_class === 'crack') {
        measureSpan.textContent = `${lengthMm.toFixed(1)} мм`;
      } else if (areaMm2 > 0) {
        measureSpan.textContent = `${areaMm2.toFixed(1)} мм²`;
      } else {
        measureSpan.textContent = `${Math.round(Number(defect.confidence || 0) * 100)}%`;
      }

      li.append(nameSpan, zoneSpan, measureSpan);
      listEl.appendChild(li);
    }
  }

  async function runInspect() {
    if (state.isInspecting) {
      return;
    }
    const runBtn = $('btn-run-inspect');
    const emptyEl = $('inspect-empty');
    const skeletonEl = $('inspect-skeleton');
    const errorEl = $('inspect-error');
    const resultEl = $('inspect-result');

    const reqSeq = state.inspectRequestSeq;
    state.isInspecting = true;
    if (runBtn) {
      runBtn.disabled = true;
    }

    try {
      while (state.pendingFilePromise) {
        const pending = state.pendingFilePromise;
        await pending;
        if (reqSeq !== state.inspectRequestSeq) {
          return;
        }
        if (state.pendingFilePromise === pending) {
          state.pendingFilePromise = null;
        }
      }

      if (!state.inspectFile) {
        if (errorEl && errorEl.hidden) {
          errorEl.textContent = 'Сначала выберите снимок изделия или демонстрационный образец';
          errorEl.hidden = false;
        }
        return;
      }

      if (emptyEl) {
        emptyEl.hidden = true;
      }
      if (errorEl) {
        errorEl.hidden = true;
        errorEl.textContent = '';
      }
      if (resultEl) {
        resultEl.hidden = true;
      }
      if (skeletonEl) {
        skeletonEl.hidden = false;
      }

      const formData = new FormData();
      formData.append('file', state.inspectFile);
      formData.append('stage', state.stage);
      formData.append('save_to_db', 'true');

      const resp = await fetch('/api/inspect', {
        method: 'POST',
        body: formData,
      });
      if (reqSeq !== state.inspectRequestSeq) {
        return;
      }

      if (!resp.ok) {
        const msg = await extractErrorDetail(resp, 'Ошибка анализа изображения');
        throw new Error(msg);
      }

      const data = await resp.json();
      if (reqSeq !== state.inspectRequestSeq) {
        return;
      }

      const badgeEl = $('inspect-verdict-badge');
      if (badgeEl) {
        badgeEl.dataset.verdict = data.verdict;
        badgeEl.textContent = data.verdict;
      }

      const summaryEl = $('inspect-gost-summary');
      if (summaryEl) {
        const stageText = STAGE_LABELS_RU[data.stage] || data.stage;
        summaryEl.textContent = `${data.gost_grade} (${stageText})`;
      }

      const latencyEl = $('inspect-latency-ms');
      if (latencyEl) {
        const ms = Number(data.metrics && data.metrics.latency_ms ? data.metrics.latency_ms : 0);
        latencyEl.textContent = `${ms.toFixed(1)} мс`;
      }

      const origImg = $('img-original');
      const overImg = $('img-overlay');
      if (origImg && state.originalObjectUrl) {
        origImg.src = state.originalObjectUrl;
      }
      if (overImg) {
        if (data.overlay_url) {
          overImg.src = data.overlay_url;
        } else if (data.overlay_base64) {
          overImg.src = `data:image/png;base64,${data.overlay_base64}`;
        }
      }
      setCompareView('split');
      setWipePercent(50);

      renderDefectList(data.defects);

      if (skeletonEl) {
        skeletonEl.hidden = true;
      }
      if (resultEl) {
        resultEl.hidden = false;
      }
      announce(`Проверка завершена: ${data.verdict}`);
    } catch (err) {
      if (reqSeq !== state.inspectRequestSeq) {
        return;
      }
      if (skeletonEl) {
        skeletonEl.hidden = true;
      }
      if (errorEl) {
        errorEl.textContent = err.message || 'Не удалось выполнить проверку';
        errorEl.hidden = false;
      }
      announce('Ошибка проверки изделия');
    } finally {
      if (reqSeq === state.inspectRequestSeq) {
        state.isInspecting = false;
        if (runBtn) {
          runBtn.disabled = false;
        }
      }
    }
  }

  function initMode1() {
    const bisqueBtn = $('stage-bisque');
    const glazedBtn = $('stage-glazed');
    if (bisqueBtn) {
      bisqueBtn.addEventListener('click', () => {
        state.inspectRequestSeq += 1;
        resetInspectFlightState();
        setStage('bisque');
      });
    }
    if (glazedBtn) {
      glazedBtn.addEventListener('click', () => {
        state.inspectRequestSeq += 1;
        resetInspectFlightState();
        setStage('glazed');
      });
    }

    const fileInput = $('inspect-file-input');
    if (fileInput) {
      fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files[0]) {
          document.querySelectorAll('#demo-sample-list .chip-btn').forEach((chip) => {
            chip.setAttribute('aria-pressed', 'false');
          });
          setInspectFile(fileInput.files[0]);
        }
      });
    }

    const cameraBtn = $('btn-camera-capture');
    const cameraInput = $('inspect-camera-input');
    if (cameraBtn && cameraInput) {
      cameraBtn.addEventListener('click', () => cameraInput.click());
      cameraInput.addEventListener('change', () => {
        if (cameraInput.files && cameraInput.files[0]) {
          document.querySelectorAll('#demo-sample-list .chip-btn').forEach((chip) => {
            chip.setAttribute('aria-pressed', 'false');
          });
          setInspectFile(cameraInput.files[0]);
        }
      });
    }

    const dropzone = $('inspect-dropzone');
    if (dropzone) {
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.dataset.state = 'dragover';
      });
      dropzone.addEventListener('dragleave', () => {
        dropzone.dataset.state = state.inspectFile ? 'ready' : 'idle';
      });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files[0] : null;
        if (droppedFile) {
          document.querySelectorAll('#demo-sample-list .chip-btn').forEach((chip) => {
            chip.setAttribute('aria-pressed', 'false');
          });
          setInspectFile(droppedFile);
        } else {
          dropzone.dataset.state = state.inspectFile ? 'ready' : 'idle';
        }
      });
    }

    const runBtn = $('btn-run-inspect');
    if (runBtn) {
      runBtn.addEventListener('click', runInspect);
    }

    document.querySelectorAll('#compare-mode-group [data-compare-mode]').forEach((btn) => {
      btn.addEventListener('click', () => setCompareView(btn.dataset.compareMode));
    });

    const range = $('compare-range');
    if (range) {
      range.addEventListener('input', () => setWipePercent(Number(range.value)));
    }

    const handle = $('compare-handle');
    const viewport = $('compare-viewport');
    if (handle && viewport) {
      let dragging = false;
      let cachedRect = null;
      let rafPending = false;
      let nextPercent = 50;

      const applyDragFrame = () => {
        rafPending = false;
        setWipePercent(nextPercent);
      };

      handle.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        dragging = true;
        cachedRect = viewport.getBoundingClientRect();
        handle.classList.add('is-dragging');
        handle.setPointerCapture(e.pointerId);
      });

      handle.addEventListener('pointermove', (e) => {
        if (!dragging || !cachedRect || cachedRect.width <= 0) {
          return;
        }
        nextPercent = ((e.clientX - cachedRect.left) / cachedRect.width) * 100;
        if (!rafPending) {
          rafPending = true;
          requestAnimationFrame(applyDragFrame);
        }
      });

      const stopDrag = (e) => {
        if (!dragging) {
          return;
        }
        dragging = false;
        handle.classList.remove('is-dragging');
        if (handle.hasPointerCapture(e.pointerId)) {
          handle.releasePointerCapture(e.pointerId);
        }
      };

      handle.addEventListener('pointerup', stopDrag);
      handle.addEventListener('pointercancel', stopDrag);
    }
  }

  /* -----------------------------------------------------------------------
     Mode 2: 289-FZ Passport Registration & Verification
     ----------------------------------------------------------------------- */
  function clearPassportResult() {
    const resultEl = $('passport-result');
    if (resultEl) {
      resultEl.hidden = true;
    }
    const bannerEl = $('passport-verdict-banner');
    if (bannerEl) {
      bannerEl.removeAttribute('data-authentic');
    }
    const iconSlot = $('passport-verdict-icon');
    if (iconSlot) {
      iconSlot.textContent = '';
    }
    const verdictText = $('passport-verdict-text');
    if (verdictText) {
      verdictText.textContent = '';
    }
    const verdictMeta = $('passport-verdict-meta');
    if (verdictMeta) {
      verdictMeta.textContent = '';
    }
    const scoreEl = $('passport-score-value');
    if (scoreEl) {
      cancelCountUp(scoreEl);
      scoreEl.textContent = '0.0%';
    }
    const barIds = ['passport-meter-fill', 'metric-orb-bar', 'metric-phash-bar', 'metric-cobalt-bar'];
    for (const id of barIds) {
      const bar = $(id);
      if (bar) {
        bar.style.setProperty('--score', '0');
      }
    }
    const valIds = ['metric-orb-val', 'metric-phash-val', 'metric-cobalt-val'];
    for (const id of valIds) {
      const val = $(id);
      if (val) {
        val.textContent = '0%';
      }
    }
    const refImg = $('passport-ref-img');
    if (refImg) {
      refImg.removeAttribute('src');
    }
    const probeImg = $('passport-probe-img');
    if (probeImg) {
      probeImg.removeAttribute('src');
    }
  }

  function showPassportError(message) {
    clearPassportResult();
    const emptyEl = $('passport-empty');
    if (emptyEl) {
      emptyEl.hidden = true;
    }
    const skeletonEl = $('passport-skeleton');
    if (skeletonEl) {
      skeletonEl.hidden = true;
    }
    const qrCard = $('register-result-card');
    if (qrCard) {
      qrCard.hidden = true;
    }
    const errEl = $('passport-error');
    if (errEl) {
      errEl.textContent = message;
      errEl.hidden = false;
    }
  }

  function setPassportAction(action) {
    state.passportAction = action === 'register' ? 'register' : 'verify';
    if (state.passportAction === 'register') {
      state.passportRequestSeq += 1;
      state.verifyRequestSeq += 1;
      state.isVerifying = false;
      const verifyBtn = $('btn-verify-passport');
      if (verifyBtn) {
        verifyBtn.disabled = false;
      }
    }
    document.querySelectorAll('#passport-action-tabs [data-action]').forEach((btn) => {
      btn.setAttribute('aria-selected', btn.dataset.action === state.passportAction ? 'true' : 'false');
    });
    const verifyForm = $('form-passport-verify');
    const registerForm = $('form-passport-register');
    if (verifyForm) {
      verifyForm.hidden = state.passportAction !== 'verify';
    }
    if (registerForm) {
      registerForm.hidden = state.passportAction !== 'register';
    }
    const errEl = $('passport-error');
    if (errEl) {
      errEl.hidden = true;
      errEl.textContent = '';
    }
  }

  function applyPassportProbeFile(file) {
    if (!file) {
      return;
    }
    if (state.probeObjectUrl) {
      URL.revokeObjectURL(state.probeObjectUrl);
      state.probeObjectUrl = null;
    }
    state.passportProbeFile = file;
    state.probeObjectUrl = URL.createObjectURL(file);
  }

  function setPassportProbeFile(file) {
    if (!file) {
      return;
    }
    state.passportRequestSeq += 1;
    applyPassportProbeFile(file);
  }

  async function registerPassport() {
    if (state.isRegistering) {
      return;
    }
    const btn = $('btn-register-passport');
    const emptyEl = $('passport-empty');
    const skeletonEl = $('passport-skeleton');
    const errEl = $('passport-error');
    const qrCard = $('register-result-card');
    const fileInput = $('register-file-input');
    const refFile = fileInput && fileInput.files ? fileInput.files[0] : null;

    if (!refFile) {
      showPassportError('Выберите эталонный снимок изделия для паспорта');
      return;
    }

    state.passportRequestSeq += 1;
    state.verifyRequestSeq += 1;
    state.isVerifying = false;
    const verifyBtn = $('btn-verify-passport');
    if (verifyBtn) {
      verifyBtn.disabled = false;
    }

    state.isRegistering = true;
    if (btn) {
      btn.disabled = true;
    }
    clearPassportResult();
    if (emptyEl) {
      emptyEl.hidden = true;
    }
    if (skeletonEl) {
      skeletonEl.hidden = true;
    }
    if (qrCard) {
      qrCard.hidden = true;
    }
    if (errEl) {
      errEl.hidden = true;
      errEl.textContent = '';
    }

    try {
      const formData = new FormData();
      const customId = ($('register-item-id')?.value || '').trim();
      if (customId) {
        formData.append('item_id', customId);
      }
      formData.append('sku', ($('register-sku')?.value || 'GZHEL-SKU-01').trim());
      formData.append('master_name', ($('register-master-name')?.value || 'Мастер Гжель').trim());
      formData.append('batch_note', ($('register-batch-note')?.value || '').trim());
      formData.append('stage', 'glazed');
      formData.append('file', refFile);

      const resp = await fetch('/api/passport/register', {
        method: 'POST',
        body: formData,
      });
      if (!resp.ok) {
        const msg = await extractErrorDetail(resp, 'Ошибка регистрации паспорта');
        throw new Error(msg);
      }
      const data = await resp.json();

      const qrImg = $('register-qr-img');
      if (qrImg) {
        qrImg.src = data.qr_image_url;
      }
      const qrMeta = $('register-qr-meta');
      if (qrMeta) {
        qrMeta.textContent = '';
        const l1 = document.createElement('strong');
        l1.className = 'mono-num';
        l1.textContent = data.item_id;
        const l2 = document.createElement('span');
        l2.className = 'mono-num';
        l2.textContent = data.qr_payload;
        const l3 = document.createElement('span');
        l3.textContent = `Признаков ORB: ${data.orb_count}`;
        qrMeta.append(l1, l2, l3);
      }
      if (qrCard) {
        qrCard.hidden = false;
      }
      if (emptyEl) {
        emptyEl.hidden = false;
      }
      const idInput = $('passport-id-input');
      if (idInput) {
        idInput.value = data.item_id;
      }
      announce(`Паспорт ${data.item_id} создан`);
    } catch (err) {
      showPassportError(err.message || 'Не удалось создать паспорт');
    } finally {
      state.isRegistering = false;
      if (btn) {
        btn.disabled = false;
      }
    }
  }

  async function verifyPassport(options = {}) {
    const allowSupersede = Boolean(options && options.allowSupersede);
    if (state.isVerifying && !allowSupersede) {
      return;
    }
    const btn = $('btn-verify-passport');
    const idInput = $('passport-id-input');
    const fileInput = $('passport-file-input');
    const emptyEl = $('passport-empty');
    const skeletonEl = $('passport-skeleton');
    const errorEl = $('passport-error');
    const resultEl = $('passport-result');
    const bannerEl = $('passport-verdict-banner');

    const itemId = (idInput ? idInput.value : '').trim();
    const probeFile = state.passportProbeFile || (fileInput && fileInput.files ? fileInput.files[0] : null);

    if (!itemId) {
      showPassportError('Введите код паспорта изделия');
      return;
    }
    if (!probeFile) {
      showPassportError('Загрузите проверяемый кадр или выберите демо-пару');
      return;
    }
    if (!state.probeObjectUrl) {
      applyPassportProbeFile(probeFile);
    }

    const verifySeq = ++state.verifyRequestSeq;
    const probeUrl = state.probeObjectUrl;

    state.isVerifying = true;
    if (btn) {
      btn.disabled = true;
    }
    if (emptyEl) {
      emptyEl.hidden = true;
    }
    if (errorEl) {
      errorEl.hidden = true;
      errorEl.textContent = '';
    }
    clearPassportResult();
    if (skeletonEl) {
      skeletonEl.hidden = false;
    }

    try {
      const formData = new FormData();
      formData.append('file', probeFile);

      const resp = await fetch(`/api/passport/verify/${encodeURIComponent(itemId)}`, {
        method: 'POST',
        body: formData,
      });
      if (verifySeq !== state.verifyRequestSeq) {
        return;
      }
      if (!resp.ok) {
        const msg = await extractErrorDetail(resp, 'Ошибка проверки паспорта');
        throw new Error(msg);
      }

      const data = await resp.json();
      if (verifySeq !== state.verifyRequestSeq) {
        return;
      }
      const isAuthentic = data.verdict === 'Подлинник';

      if (bannerEl) {
        bannerEl.dataset.authentic = isAuthentic ? 'true' : 'false';
      }
      const iconSlot = $('passport-verdict-icon');
      if (iconSlot) {
        iconSlot.textContent = '';
        iconSlot.appendChild(createVerdictIcon(isAuthentic));
      }
      const verdictText = $('passport-verdict-text');
      if (verdictText) {
        verdictText.textContent = data.verdict;
      }
      const verdictMeta = $('passport-verdict-meta');
      if (verdictMeta) {
        const ms = Number(data.latency_ms || 0).toFixed(1);
        verdictMeta.textContent = `${data.registered_sku || itemId}, ${data.registered_master || 'Мастер Гжель'}, ${ms} мс`;
      }

      const refImg = $('passport-ref-img');
      if (refImg && data.reference_image_url) {
        refImg.src = data.reference_image_url;
      }
      const probeImg = $('passport-probe-img');
      if (probeImg && probeUrl) {
        probeImg.src = probeUrl;
      }

      const matchPct = Math.max(0, Math.min(100, Number(data.match_percent || 0)));
      const orbScore = Math.max(0, Math.min(1, Number(data.orb_score || 0)));
      const phashScore = Math.max(0, Math.min(1, Number(data.phash_score || 0)));
      const colorScore = Math.max(0, Math.min(1, Number(data.color_score || 0)));

      const meterFill = $('passport-meter-fill');
      const orbBar = $('metric-orb-bar');
      const phashBar = $('metric-phash-bar');
      const cobaltBar = $('metric-cobalt-bar');
      [meterFill, orbBar, phashBar, cobaltBar].forEach((bar) => {
        if (bar) {
          bar.style.setProperty('--score', '0');
        }
      });

      const orbVal = $('metric-orb-val');
      const phashVal = $('metric-phash-val');
      const cobaltVal = $('metric-cobalt-val');
      if (orbVal) {
        orbVal.textContent = `${Math.round(orbScore * 100)}%`;
      }
      if (phashVal) {
        phashVal.textContent = `${Math.round(phashScore * 100)}%`;
      }
      if (cobaltVal) {
        cobaltVal.textContent = `${Math.round(colorScore * 100)}%`;
      }

      if (skeletonEl) {
        skeletonEl.hidden = true;
      }
      if (resultEl) {
        resultEl.hidden = false;
      }

      animateCountUp($('passport-score-value'), matchPct, 360, '%', 1);
      requestAnimationFrame(() => {
        if (meterFill) {
          meterFill.style.setProperty('--score', String(matchPct / 100));
        }
        if (orbBar) {
          orbBar.style.setProperty('--score', String(orbScore));
        }
        if (phashBar) {
          phashBar.style.setProperty('--score', String(phashScore));
        }
        if (cobaltBar) {
          cobaltBar.style.setProperty('--score', String(colorScore));
        }
      });

      announce(`Сверка завершена: ${data.verdict}, ${matchPct.toFixed(1)}%`);
    } catch (err) {
      if (verifySeq !== state.verifyRequestSeq) {
        return;
      }
      showPassportError(err.message || 'Не удалось выполнить сверку');
      announce('Ошибка сверки паспорта');
    } finally {
      if (verifySeq === state.verifyRequestSeq) {
        state.isVerifying = false;
        if (btn) {
          btn.disabled = false;
        }
      }
    }
  }

  async function handleDemoPassportClick(btn, item) {
    const reqSeq = ++state.passportRequestSeq;
    document.querySelectorAll('#demo-passport-list .chip-btn').forEach((chip) => {
      chip.setAttribute('aria-pressed', chip === btn ? 'true' : 'false');
    });
    if (state.passportAction !== 'verify') {
      setPassportAction('verify');
    }
    const idInput = $('passport-id-input');
    if (idInput) {
      idInput.value = item.item_id;
    }

    const errorEl = $('passport-error');
    if (errorEl) {
      errorEl.hidden = true;
      errorEl.textContent = '';
    }

    try {
      const file = await fetchImageAsFile(item.probe_image_url, `${item.item_id}_probe.png`);
      if (reqSeq !== state.passportRequestSeq) {
        return;
      }
      applyPassportProbeFile(file);
      await verifyPassport({ allowSupersede: true });
    } catch (err) {
      if (reqSeq !== state.passportRequestSeq) {
        return;
      }
      showPassportError(err.message || 'Не удалось загрузить демо-кадр');
    }
  }

  function initMode2() {
    document.querySelectorAll('#passport-action-tabs [data-action]').forEach((btn) => {
      btn.addEventListener('click', () => setPassportAction(btn.dataset.action));
    });

    const verifyFileInput = $('passport-file-input');
    if (verifyFileInput) {
      verifyFileInput.addEventListener('change', () => {
        if (verifyFileInput.files && verifyFileInput.files[0]) {
          document.querySelectorAll('#demo-passport-list .chip-btn').forEach((chip) => {
            chip.setAttribute('aria-pressed', 'false');
          });
          setPassportProbeFile(verifyFileInput.files[0]);
        }
      });
    }

    const verifyBtn = $('btn-verify-passport');
    if (verifyBtn) {
      verifyBtn.addEventListener('click', () => verifyPassport());
    }

    const registerBtn = $('btn-register-passport');
    if (registerBtn) {
      registerBtn.addEventListener('click', registerPassport);
    }
  }

  /* -----------------------------------------------------------------------
     Mode 3: OTK Analytics Dashboard
     ----------------------------------------------------------------------- */
  function createBreakdownGroup(title, rows, labelFn, totalFn) {
    const group = document.createElement('div');
    group.className = 'breakdown-group';

    const h = document.createElement('h2');
    h.className = 'breakdown-title';
    h.textContent = title;
    group.appendChild(h);

    const list = document.createElement('ul');
    list.className = 'breakdown-list';

    if (!Array.isArray(rows) || rows.length === 0) {
      const emptyLi = document.createElement('li');
      emptyLi.className = 'helper-note';
      emptyLi.textContent = 'Нет данных';
      list.appendChild(emptyLi);
      group.appendChild(list);
      return group;
    }

    const maxVal = Math.max(1, ...rows.map((r) => Number(totalFn(r) || 0)));
    for (const row of rows) {
      const count = Number(totalFn(row) || 0);
      const ratio = Math.max(0, Math.min(1, count / maxVal));

      const li = document.createElement('li');
      li.className = 'breakdown-item';

      const head = document.createElement('div');
      head.className = 'breakdown-row-head';
      const nameSpan = document.createElement('span');
      nameSpan.textContent = labelFn(row);
      const valSpan = document.createElement('span');
      valSpan.className = 'mono-num';
      valSpan.textContent = String(count);
      head.append(nameSpan, valSpan);

      const track = document.createElement('div');
      track.className = 'breakdown-track';
      const fill = document.createElement('div');
      fill.className = 'breakdown-fill';
      fill.style.setProperty('--score', String(ratio));
      track.appendChild(fill);

      li.append(head, track);
      list.appendChild(li);
    }

    group.appendChild(list);
    return group;
  }

  function renderLedgerRows(rows) {
    const tbody = $('ledger-tbody');
    const table = $('ledger-table');
    const emptyEl = $('dashboard-empty');
    if (!tbody) {
      return;
    }
    tbody.textContent = '';

    if (!Array.isArray(rows) || rows.length === 0) {
      if (table) {
        table.hidden = true;
      }
      if (emptyEl) {
        emptyEl.hidden = false;
      }
      return;
    }

    if (table) {
      table.hidden = false;
    }
    if (emptyEl) {
      emptyEl.hidden = true;
    }

    for (const row of rows) {
      const tr = document.createElement('tr');

      const tdTime = document.createElement('td');
      tdTime.dataset.label = 'Время';
      tdTime.className = 'mono-num';
      tdTime.textContent = formatTime(row.created_at);

      const tdId = document.createElement('td');
      tdId.dataset.label = 'ID кадра';
      tdId.className = 'mono-num';
      tdId.textContent = row.item_id || `#${row.id}`;

      const tdStage = document.createElement('td');
      tdStage.dataset.label = 'Этап';
      tdStage.textContent = STAGE_LABELS_RU[row.stage] || escapeText(row.stage);

      const tdDefects = document.createElement('td');
      tdDefects.dataset.label = 'Дефекты';
      if (row.primary_defect) {
        const defectName = DEFECT_LABELS_RU[row.primary_defect] || row.primary_defect;
        tdDefects.textContent = `${defectName} (${row.defect_count})`;
      } else {
        tdDefects.textContent = String(row.defect_count ?? 0);
      }

      const tdLatency = document.createElement('td');
      tdLatency.dataset.label = 'Цикл, мс';
      tdLatency.className = 'mono-num num-col';
      tdLatency.textContent = Number(row.latency_ms || 0).toFixed(1);

      const tdVerdict = document.createElement('td');
      tdVerdict.dataset.label = 'Вердикт';
      const badge = document.createElement('span');
      badge.className = 'verdict-badge';
      badge.dataset.verdict = row.verdict;
      badge.textContent = row.verdict;
      tdVerdict.appendChild(badge);

      tr.append(tdTime, tdId, tdStage, tdDefects, tdLatency, tdVerdict);
      tbody.appendChild(tr);
    }
  }

  async function loadAnalytics() {
    const seq = ++state.analyticsRequestSeq;
    const skeletonEl = $('dashboard-skeleton');
    const errorEl = $('dashboard-error');
    const emptyEl = $('dashboard-empty');
    const stageVal = $('filter-stage')?.value || '';
    const verdictVal = $('filter-verdict')?.value || '';

    const params = new URLSearchParams({ limit: '25' });
    if (stageVal) {
      params.set('stage', stageVal);
    }
    if (verdictVal) {
      params.set('verdict', verdictVal);
    }

    if (errorEl) {
      errorEl.hidden = true;
      errorEl.textContent = '';
    }
    if (emptyEl) {
      emptyEl.hidden = true;
    }
    if (skeletonEl) {
      skeletonEl.hidden = false;
    }

    try {
      const resp = await fetch(`/api/analytics?${params.toString()}`);
      if (seq !== state.analyticsRequestSeq) {
        return;
      }
      if (!resp.ok) {
        const msg = await extractErrorDetail(resp, 'Ошибка загрузки аналитики');
        throw new Error(msg);
      }
      const data = await resp.json();
      if (seq !== state.analyticsRequestSeq) {
        return;
      }
      const summary = data.summary || {};

      const total = Number(summary.total_inspections || 0);
      const rework = Number(summary.rework_count || 0);
      const reject = Number(summary.reject_count || 0);
      const defectCount = rework + reject;
      const defectShare = total > 0 ? (defectCount / total) * 100 : 0;

      const kpiTotal = $('kpi-total');
      const kpiPass = $('kpi-pass-rate');
      const kpiDefect = $('kpi-defect-rate');
      const kpiAvgMs = $('kpi-avg-ms');
      if (kpiTotal) {
        kpiTotal.textContent = String(total);
      }
      if (kpiPass) {
        kpiPass.textContent = `${Number(summary.pass_rate_pct || 0).toFixed(1)}%`;
      }
      if (kpiDefect) {
        kpiDefect.textContent = `${defectCount} (${defectShare.toFixed(1)}%)`;
      }
      if (kpiAvgMs) {
        kpiAvgMs.textContent = Number(summary.avg_latency_ms || 0).toFixed(1);
      }

      const breakdownEl = $('dashboard-breakdown');
      if (breakdownEl) {
        breakdownEl.textContent = '';
        breakdownEl.appendChild(
          createBreakdownGroup(
            'По этапам',
            data.by_stage,
            (r) => STAGE_LABELS_RU[r.stage] || r.stage,
            (r) => r.total
          )
        );
        breakdownEl.appendChild(
          createBreakdownGroup(
            'Частые дефекты',
            data.by_defect_class,
            (r) => DEFECT_LABELS_RU[r.defect_class] || r.defect_class,
            (r) => r.count
          )
        );
        breakdownEl.appendChild(
          createBreakdownGroup(
            'По сменам',
            data.by_shift,
            (r) => `Смена ${r.shift}`,
            (r) => r.total
          )
        );
      }

      if (skeletonEl) {
        skeletonEl.hidden = true;
      }
      renderLedgerRows(data.recent_inspections);
    } catch (err) {
      if (seq !== state.analyticsRequestSeq) {
        return;
      }
      if (skeletonEl) {
        skeletonEl.hidden = true;
      }
      if (errorEl) {
        errorEl.textContent = err.message || 'Не удалось обновить дашборд';
        errorEl.hidden = false;
      }
    }
  }

  function initMode3() {
    const refreshBtn = $('btn-refresh-analytics');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', loadAnalytics);
    }
    const filterStage = $('filter-stage');
    if (filterStage) {
      filterStage.addEventListener('change', loadAnalytics);
    }
    const filterVerdict = $('filter-verdict');
    if (filterVerdict) {
      filterVerdict.addEventListener('change', loadAnalytics);
    }
  }

  function initRevealObserver() {
    const targets = document.querySelectorAll('.reveal');
    if (!targets.length || !('IntersectionObserver' in window)) {
      targets.forEach((el) => el.classList.add('is-visible'));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            io.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.14, rootMargin: '0px 0px -40px 0px' }
    );
    targets.forEach((el) => io.observe(el));
  }

  function boot() {
    initTabs();
    initMode1();
    initMode2();
    initMode3();
    initRevealObserver();
    loadHealth();
    loadDemoSamples();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
