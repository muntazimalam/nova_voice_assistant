/**
 * AURA AI Voice Assistant — Phase 2 Client Controller
 *
 * Additions over Phase 1:
 *  - Real microphone capture via getUserMedia + AudioWorklet (16 kHz PCM Int16)
 *  - PCM frames streamed to the backend as binary WebSocket messages
 *  - TTS MP3 playback: binary chunks reassembled → AudioContext.decodeAudioData → play
 *  - Binary / text message demuxing on onmessage
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── DOM refs ───────────────────────────────────────────────────────────────
  const connectionPill     = document.getElementById('connectionPill');
  const connectionLabel    = document.getElementById('connectionLabel');
  const latencyMetric      = document.getElementById('latencyMetric');
  const btnReconnect       = document.getElementById('btnReconnect');

  const sttMetric          = document.getElementById('sttMetric');
  const llmMetric          = document.getElementById('llmMetric');
  const ttsMetric          = document.getElementById('ttsMetric');
  const totalMetric        = document.getElementById('totalMetric');
  const chkFollowUp        = document.getElementById('chkFollowUp');
  const selVoiceEngine     = document.getElementById('selVoiceEngine');

  const orbCard            = document.querySelector('.orb-card');
  const voiceOrb           = document.getElementById('voiceOrb');
  const assistantStateChip = document.getElementById('assistantStateChip');
  const orbStatusHeadline  = document.getElementById('orbStatusHeadline');
  const orbStatusSub       = document.getElementById('orbStatusSub');
  const btnVoiceTrigger    = document.getElementById('btnVoiceTrigger');
  const micIcon            = document.getElementById('micIcon');

  const btnStartListening  = document.getElementById('btnStartListening');
  const btnSimulateTurn    = document.getElementById('btnSimulateTurn');
  const btnPing            = document.getElementById('btnPing');
  const txtCustomUtterance = document.getElementById('txtCustomUtterance');
  const btnSendUtterance   = document.getElementById('btnSendUtterance');

  const terminalOutput     = document.getElementById('terminalOutput');
  const chkAutoScroll      = document.getElementById('chkAutoScroll');
  const btnClearLogs       = document.getElementById('btnClearLogs');
  const logCount           = document.getElementById('logCount');
  const filterTabs         = document.querySelectorAll('.filter-tab');

  // ── SVG icons ──────────────────────────────────────────────────────────────
  const ICON_MIC = `
    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/>
    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
    <line x1="12" x2="12" y1="19" y2="22"/>
  `;
  const ICON_STOP = `
    <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>
  `;
  const ICON_SPINNER = `
    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
  `;

  // ── App state ──────────────────────────────────────────────────────────────
  let socket            = null;
  let currentState      = 'DISCONNECTED';
  let reconnectAttempts = 0;
  const MAX_RECONNECT   = 10000; // ms cap

  let totalLogs  = 0;
  let activeFilter = 'all';

  // ── Audio capture state ────────────────────────────────────────────────────
  let audioCtx        = null;   // AudioContext (created on first mic use)
  let mediaStream     = null;   // MediaStream from getUserMedia
  let workletNode     = null;   // AudioWorkletNode running the downsampler
  let legacyProcessor = null;   // ScriptProcessorNode fallback (worklet unavailable)
  let sourceNode      = null;   // MediaStreamSourceNode
  let isMicActive     = false;

  // ── High-Performance Web Audio Playback Queue (Zero-Gap Scheduled) ────────
  let playbackCtx           = null;   // AudioContext for scheduled playback & chimes
  let currentSegmentBuffers = [];     // Raw MP3 chunks for currently arriving segment
  let activeSources         = [];     // List of scheduled AudioBufferSourceNodes
  let nextPlayTime          = 0;       // Timeline cursor for seamless gapless playback
  let isAudioPlaying        = false;
  let allSegmentsReceived   = false;
  let followUpTimer         = null;

  // ==========================================================================
  // AudioWorklet inline source (downsampler: browser sample rate → 16 kHz Int16)
  // ==========================================================================

  /**
   * We register a worklet processor inline via a Blob URL so we don't need an
   * extra static file.  The processor receives float32 frames at the browser's
   * native sample rate and downsamples to 16 kHz by averaging blocks, then
   * converts to Int16 PCM and posts the buffer back to the main thread.
   */
const WORKLET_CODE = `
class DownsampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const inRate  = options.processorOptions.inputSampleRate || 44100;
    const outRate = 16000;
    // Exact fractional ratio (e.g. 44.1k -> 2.75625, 48k -> 3, 96k -> 6).
    this._ratio = inRate / outRate;
    // Fractional accumulator so the 16 kHz cadence is exact for non-multiple rates.
    this._phase = 0;
    this._acc   = 0;
    this._n     = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const out = [];

    for (let i = 0; i < ch.length; i++) {
      this._acc += ch[i];
      this._n   += 1;
      this._phase += 1;

      if (this._phase >= this._ratio) {
        // One output sample per ratio input samples (windowed average
        // doubles as a crude anti-alias low-pass).
        out.push(this._acc / this._n);
        this._phase -= this._ratio;
        this._acc = 0;
        this._n   = 0;
      }
    }

    if (out.length) {
      const buf = new Int16Array(out.length);
      for (let k = 0; k < out.length; k++) {
        buf[k] = Math.max(-32768, Math.min(32767, Math.round(out[k] * 32767)));
      }
      this.port.postMessage(buf);
    }
    return true;
  }
}
registerProcessor('downsample-processor', DownsampleProcessor);
`;

  // ==========================================================================
  // Mic capture
  // ==========================================================================

  async function startMicCapture() {
    if (isMicActive) return;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      logTerminal('error', 'getUserMedia not supported in this browser or context (use HTTPS / localhost).', 'system');
      return;
    }

    // Create + resume the AudioContext SYNCHRONOUSLY (before any await): the
    // browser only runs it "live" when the constructor call and resume() happen
    // inside the user gesture (the button click). If we awaited getUserMedia
    // first, Chrome starts the context suspended and resume() can hang forever,
    // leaving the worklet silent — the classic "mic does nothing" failure.
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }
    } catch (err) {
      logTerminal('error', `AudioContext could not resume (${err.message}); capture may stay silent.`, 'system');
    }
    logTerminal('system', `AudioContext state: ${audioCtx.state} (${audioCtx.sampleRate} Hz)`, 'system');

    logTerminal('system', 'Requesting microphone access...', 'system');
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    } catch (err) {
      logTerminal('error', `Microphone access denied: ${err.message}`, 'system');
      return;
    }
    logTerminal('system', 'Microphone granted.', 'system');

    // Register the downsampler worklet via Blob URL (no extra static file needed)
    let blobUrl = null;
    try {
      const blob    = new Blob([WORKLET_CODE], { type: 'application/javascript' });
      blobUrl        = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(blobUrl);
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      logTerminal('error', `AudioWorklet failed to load (${err.message}); falling back to ScriptProcessorNode.`, 'system');
      startLegacyCapture();
      return;
    }
    logTerminal('system', 'AudioWorklet registered.', 'system');

    try {
      sourceNode  = audioCtx.createMediaStreamSource(mediaStream);
      workletNode = new AudioWorkletNode(audioCtx, 'downsample-processor', {
        processorOptions: { inputSampleRate: audioCtx.sampleRate },
      });

      // Collect Int16 samples into 320-sample chunks (20 ms at 16 kHz = 640 bytes),
      // then push each frame to the server. Shared by the worklet path.
      const pushPcm = buildPcmFrameSender();
      workletNode.port.onmessage = (ev) => {
        // The worklet posts an Int16Array (or falls back to a single sample).
        const payload = ev.data;
        const values  = (payload && typeof payload.length === 'number') ? Array.from(payload) : [payload];
        pushPcm(values);
      };

      sourceNode.connect(workletNode);
      // Do NOT connect workletNode to audioCtx.destination — we don't want echo.

      isMicActive = true;
      playChime('wake');
      logTerminal('system', `Mic capture started (native ${audioCtx.sampleRate} Hz → 16 kHz PCM Int16, 20 ms frames).`, 'system');

      // Tell the server we're now sending audio
      sendWebSocketPayload({ type: 'audio_start' });
    } catch (err) {
      logTerminal('error', `Mic node setup failed (${err.message}); falling back to ScriptProcessorNode.`, 'system');
      try { if (sourceNode) sourceNode.disconnect(); } catch (_) {}
      sourceNode = null;
      startLegacyCapture();
      return;
    }
  }

  /**
   * Returns a push(values) accumulator that batches Int16 samples into
   * 320-sample (640-byte) frames and streams them over WebSocket. Logs the
   * first frame so the dashboard terminal proves mic bytes are flowing.
   */
  function buildPcmFrameSender() {
    const CHUNK_SAMPLES = 320;
    const pendingSamples = new Int16Array(CHUNK_SAMPLES);
    let pendingIdx = 0;
    let firstFrameSent = false;

    return function pushPcm(values) {
      for (const v of values) {
        pendingSamples[pendingIdx++] = v;
        if (pendingIdx < CHUNK_SAMPLES) continue;

        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(pendingSamples.buffer.slice(0));
          if (!firstFrameSent) {
            firstFrameSent = true;
            logTerminal('system', 'First PCM frame sent to server — mic bytes are flowing.', 'system');
          }
        } else {
          logTerminal('system', 'Mic active but WebSocket not open; dropping frame.', 'system');
        }
        pendingIdx = 0;
      }
    };
  }

  /**
   * Fallback when AudioWorklet (blob URL) is unavailable. ScriptProcessorNode
   * is deprecated but supported everywhere; we downsample inline the same way
   * and push the same 640-byte Int16 frames.
   */
  async function startLegacyCapture() {
    try {
      const CHUNK_SAMPLES = 320;
      const ratio = audioCtx.sampleRate / 16000;
      const grab = audioCtx.createScriptProcessor(4096, 1, 1);
      // Rolling fractional downsampler state.
      let phase = 0, acc = 0, n = 0;
      const pushPcm = buildPcmFrameSender();

      grab.onaudioprocess = (ev) => {
        const ch = ev.inputBuffer.getChannelData(0);
        const downsample = new Int16Array(Math.floor(ch.length / ratio) + 1);
        let outIdx = 0;
        for (let i = 0; i < ch.length; i++) {
          acc += ch[i]; n += 1; phase += 1;
          if (phase >= ratio) {
            downsample[outIdx++] = Math.max(-32768, Math.min(32767, Math.round((acc / n) * 32767)));
            phase -= ratio; acc = 0; n = 0;
          }
        }
        pushPcm(downsample.subarray(0, outIdx));
      };

      sourceNode = audioCtx.createMediaStreamSource(mediaStream);
      sourceNode.connect(grab);
      // ScriptProcessor only fires while connected to a graph; route to a
      // muted gain so we don't echo the mic back through the speakers.
      const mute = audioCtx.createGain();
      mute.gain.value = 0;
      grab.connect(mute);
      mute.connect(audioCtx.destination);
      legacyProcessor = grab;

      isMicActive = true;
      playChime('wake');
      logTerminal('system', `Mic capture started via ScriptProcessorNode fallback (${audioCtx.sampleRate} Hz → 16 kHz PCM).`, 'system');
      sendWebSocketPayload({ type: 'audio_start' });
    } catch (err) {
      logTerminal('error', `Microphone capture failed entirely: ${err.message}`, 'system');
      stopMicCapture(false);
    }
  }

  function stopMicCapture(notifyServer = true) {
    clearTimeout(followUpTimer);
    if (!isMicActive) return;

    if (legacyProcessor) { legacyProcessor.disconnect(); legacyProcessor = null; }
    if (workletNode)  { workletNode.disconnect(); workletNode = null; }
    if (sourceNode)   { sourceNode.disconnect();  sourceNode = null; }
    if (mediaStream)  { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    if (audioCtx)     { audioCtx.close(); audioCtx = null; }

    isMicActive = false;
    logTerminal('system', 'Mic capture stopped.', 'system');

    if (notifyServer && socket && socket.readyState === WebSocket.OPEN) {
      sendWebSocketPayload({ type: 'audio_end' });
    }
  }

  // ==========================================================================
  // High-Performance Web Audio Playback Queue & Siri/Alexa Chimes
  // ==========================================================================

  function getPlaybackContext() {
    if (!playbackCtx || playbackCtx.state === 'closed') {
      playbackCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (playbackCtx.state === 'suspended') {
      playbackCtx.resume();
    }
    return playbackCtx;
  }

  function playChime(type = 'wake') {
    try {
      const ctx = getPlaybackContext();
      const now = ctx.currentTime;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';

      if (type === 'wake') {
        // High-tech two-tone Siri/Alexa chime (F#5 740Hz -> A5 880Hz)
        osc.frequency.setValueAtTime(740, now);
        osc.frequency.exponentialRampToValueAtTime(880, now + 0.08);
        gain.gain.setValueAtTime(0.001, now);
        gain.gain.linearRampToValueAtTime(0.18, now + 0.03);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.35);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.36);
      } else if (type === 'thinking') {
        osc.frequency.setValueAtTime(880, now);
        osc.frequency.exponentialRampToValueAtTime(660, now + 0.06);
        gain.gain.setValueAtTime(0.001, now);
        gain.gain.linearRampToValueAtTime(0.10, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.19);
      }
    } catch (e) {
      console.debug('Chime error', e);
    }
  }

  function stopAudioPlayback() {
    for (const s of activeSources) {
      try { s.stop(); } catch (_) {}
    }
    activeSources = [];
    nextPlayTime = 0;
    isAudioPlaying = false;
    allSegmentsReceived = false;
    currentSegmentBuffers = [];
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  async function handleSegmentEnd(segmentIndex) {
    if (currentSegmentBuffers.length === 0) return;
    const totalBytes = currentSegmentBuffers.reduce((sum, b) => sum + b.byteLength, 0);
    const combined = new Uint8Array(totalBytes);
    let offset = 0;
    for (const b of currentSegmentBuffers) {
      combined.set(new Uint8Array(b), offset);
      offset += b.byteLength;
    }
    currentSegmentBuffers = [];

    // If in Instant Local mode, skip playing cloud audio chunks
    if (selVoiceEngine && selVoiceEngine.value === 'instant') {
      return;
    }

    try {
      const ctx = getPlaybackContext();
      const audioBuffer = await ctx.decodeAudioData(combined.buffer);
      const now = ctx.currentTime;
      const startTime = Math.max(now + 0.02, nextPlayTime);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.start(startTime);
      nextPlayTime = startTime + audioBuffer.duration;
      activeSources.push(source);
      isAudioPlaying = true;

      const durMs = Math.round(audioBuffer.duration * 1000);
      logTerminal('system', `Pipelined audio clause #${segmentIndex} ready (${durMs}ms). Gapless playback scheduled at +${Math.round((startTime - now)*1000)}ms.`, 'system');

      source.onended = () => {
        const idx = activeSources.indexOf(source);
        if (idx !== -1) activeSources.splice(idx, 1);
        if (activeSources.length === 0 && allSegmentsReceived) {
          isAudioPlaying = false;
          onResponsePlaybackFinished();
        }
      };
    } catch (err) {
      logTerminal('error', `Segment decode failed: ${err.message}`, 'system');
    }
  }

  function onResponsePlaybackFinished() {
    isAudioPlaying = false;
    // Check if hands-free follow-up mode is enabled
    if (chkFollowUp && chkFollowUp.checked && !isMicActive) {
      logTerminal('system', 'Follow-Up Mode: Listening for next conversational turn…', 'system');
      triggerFollowUpListening();
    }
  }

  function triggerFollowUpListening() {
    clearTimeout(followUpTimer);
    sendWebSocketPayload({ type: 'start_listening' });
    startMicCapture();
    // Auto-close follow-up after 5 seconds if no speech detected
    followUpTimer = setTimeout(() => {
      if (currentState === 'LISTENING') {
        stopMicCapture();
        sendWebSocketPayload({ type: 'stop_listening' });
        logTerminal('system', 'Follow-up listening window closed.', 'system');
      }
    }, 5000);
  }

  function speakLocal(text) {
    if (!('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.15;
    const voices = window.speechSynthesis.getVoices();
    const enVoice = voices.find(v => v.name.includes('Natural') || v.name.includes('Samantha') || v.lang.startsWith('en'));
    if (enVoice) u.voice = enVoice;
    window.speechSynthesis.speak(u);
  }

  function updateTelemetryHUD(metrics) {
    if (!metrics) return;
    if (sttMetric && metrics.stt_ms !== undefined) sttMetric.textContent = `${metrics.stt_ms} ms`;
    if (llmMetric && metrics.llm_ttft_ms !== undefined) llmMetric.textContent = `${metrics.llm_ttft_ms} ms`;
    if (ttsMetric && metrics.tts_first_ms !== undefined) ttsMetric.textContent = `${metrics.tts_first_ms} ms`;
    if (totalMetric && metrics.total_roundtrip_ms !== undefined) {
      totalMetric.textContent = `${metrics.total_roundtrip_ms} ms`;
      latencyMetric.textContent = `${metrics.total_roundtrip_ms} ms`;
    }
    logTerminal('system', `Telemetry: STT ${metrics.stt_ms || 0}ms | LLM TTFT ${metrics.llm_ttft_ms || 0}ms | TTS ${metrics.tts_first_ms || 0}ms | Total Roundtrip: ${metrics.total_roundtrip_ms || 0}ms`, 'system');
  }

  // ==========================================================================
  // WebSocket Manager
  // ==========================================================================

  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl    = `${protocol}//${window.location.host}/ws`;

    updateConnectionStatus('connecting', 'Connecting…');
    logTerminal('system', `Connecting to WebSocket: ${wsUrl}`);

    try {
      socket = new WebSocket(wsUrl);
      // Tell the browser we want to receive binary data as ArrayBuffers (not Blobs)
      socket.binaryType = 'arraybuffer';
    } catch (err) {
      logTerminal('error', `WebSocket constructor failed: ${err.message}`);
      scheduleReconnect();
      return;
    }

    socket.onopen = () => {
      reconnectAttempts = 0;
      updateConnectionStatus('connected', 'Live Connected');
      setAssistantState('IDLE');
      logTerminal('system', 'WebSocket established (HTTP 101 Switching Protocols)');
      sendPing();
    };

    socket.onmessage = (event) => {
      // ── Binary frame: MP3 audio chunk for current segment ───────────────────
      if (event.data instanceof ArrayBuffer) {
        currentSegmentBuffers.push(event.data);
        return;
      }

      // ── Text frame: JSON control message ──────────────────────────────────
      try {
        const data = JSON.parse(event.data);
        handleIncomingMessage(data);
      } catch (e) {
        logTerminal('ws-recv', `Raw (non-JSON): ${event.data}`, 'ws');
      }
    };

    socket.onclose = (event) => {
      updateConnectionStatus('disconnected', 'Disconnected');
      setAssistantState('DISCONNECTED');
      stopMicCapture(false); // mic cleanup on disconnect, don't send audio_end
      logTerminal('system', `WebSocket closed (code: ${event.code})`);
      scheduleReconnect();
    };

    socket.onerror = () => {
      logTerminal('error', 'WebSocket error occurred.', 'system');
    };
  }

  function scheduleReconnect() {
    reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), MAX_RECONNECT);
    logTerminal('system', `Reconnecting in ${(delay / 1000).toFixed(1)}s (attempt ${reconnectAttempts})…`);
    setTimeout(initWebSocket, delay);
  }

  function sendWebSocketPayload(payload) {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      logTerminal('error', 'Cannot send: WebSocket not open', 'system');
      return false;
    }
    socket.send(JSON.stringify(payload));
    logTerminal('ws-send', `Sent: ${payload.type} frame`, 'ws');
    return true;
  }

  function sendPing() {
    sendWebSocketPayload({ type: 'ping', client_timestamp: Date.now() });
  }

  // ==========================================================================
  // Message & State Handlers
  // ==========================================================================

  function handleIncomingMessage(data) {
    const msgType = data.type || 'unknown';

    if (msgType === 'pong') {
      if (data.client_timestamp) {
        const rtt = Date.now() - data.client_timestamp;
        latencyMetric.textContent = `${rtt} ms`;
        logTerminal('ws-recv', `Pong received (RTT: ${rtt}ms)`, 'ws');
      }
      return;
    }

    if (msgType === 'state_change') {
      const targetState = (data.status || 'idle').toUpperCase();
      if (targetState === 'LISTENING' && currentState !== 'LISTENING') {
        playChime('wake');
      } else if (targetState === 'PROCESSING') {
        playChime('thinking');
      }
      setAssistantState(targetState, data.message);
      logTerminal('state', `Assistant state → [${targetState}]: ${data.message || ''}`, 'state');
      return;
    }

    if (msgType === 'system_event') {
      logTerminal('system', `${data.message}${data.details ? ' (' + data.details + ')' : ''}`, 'system');
      return;
    }

    if (msgType === 'transcript_final') {
      logTerminal('ws-recv', `Speech recognized: "${data.content}" (${data.stt_ms || 0} ms)`, 'ws');
      if (sttMetric && data.stt_ms !== undefined) sttMetric.textContent = `${data.stt_ms} ms`;
      return;
    }

    if (msgType === 'chat_reply') {
      logTerminal('ws-recv', `Reply: ${data.content}`, 'ws');
      if (data.metrics) updateTelemetryHUD(data.metrics);
      return;
    }

    if (msgType === 'transcript_partial') {
      logTerminal('ws-recv', `Speech Token: ${data.content}`, 'ws');
      if (selVoiceEngine && selVoiceEngine.value === 'instant') {
        speakLocal(data.content);
      }
      return;
    }

    // ── Audio stream control frames ──────────────────────────────────────────
    if (msgType === 'audio_start') {
      stopAudioPlayback();
      allSegmentsReceived = false;
      currentSegmentBuffers = [];
      return;
    }

    if (msgType === 'audio_segment_start') {
      currentSegmentBuffers = [];
      return;
    }

    if (msgType === 'audio_segment_end') {
      handleSegmentEnd(data.segment_index);
      return;
    }

    if (msgType === 'audio_end') {
      allSegmentsReceived = true;
      if (data.metrics) {
        updateTelemetryHUD(data.metrics);
      }
      if (activeSources.length === 0) {
        onResponsePlaybackFinished();
      }
      return;
    }

    if (msgType === 'pipeline_metrics') {
      if (data.metrics) {
        updateTelemetryHUD(data.metrics);
      }
      return;
    }

    logTerminal('ws-recv', `Frame: ${JSON.stringify(data)}`, 'ws');
  }

  // ==========================================================================
  // UI State Machine
  // ==========================================================================

  function updateConnectionStatus(state, label) {
    connectionPill.className = `status-indicator-pill ${state}`;
    connectionLabel.textContent = label;
  }

  function setAssistantState(state, message) {
    currentState = state;
    orbCard.classList.remove('state-idle', 'state-listening', 'state-processing', 'state-speaking', 'state-disconnected');

    switch (state) {
      case 'IDLE':
        orbCard.classList.add('state-idle');
        assistantStateChip.textContent = 'STANDBY';
        orbStatusHeadline.textContent  = 'Assistant Ready';
        orbStatusSub.textContent       = message || 'Tap microphone or send a prompt to begin.';
        btnStartListening.innerHTML    = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-svg">
            <circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>
          </svg> Start Listening`;
        micIcon.innerHTML = ICON_MIC;
        break;

      case 'LISTENING':
        orbCard.classList.add('state-listening');
        assistantStateChip.textContent = 'LISTENING';
        orbStatusHeadline.textContent  = 'Listening…';
        orbStatusSub.textContent       = message || 'Capturing microphone in real-time.';
        btnStartListening.innerHTML    = `
          <svg viewBox="0 0 24 24" fill="currentColor" class="btn-svg">
            <rect x="6" y="6" width="12" height="12" rx="2"/>
          </svg> Stop Listening`;
        micIcon.innerHTML = ICON_STOP;
        break;

      case 'CAPTURING':
        orbCard.classList.add('state-listening');
        assistantStateChip.textContent = 'CAPTURING';
        orbStatusHeadline.textContent  = 'Capturing Command…';
        orbStatusSub.textContent       = message || 'Recording your command.';
        micIcon.innerHTML = ICON_STOP;
        break;

      case 'PROCESSING':
        orbCard.classList.add('state-processing');
        assistantStateChip.textContent = 'PROCESSING';
        orbStatusHeadline.textContent  = 'Thinking…';
        orbStatusSub.textContent       = message || 'STT + LLM inference underway.';
        micIcon.innerHTML = ICON_SPINNER;
        break;

      case 'SPEAKING':
        orbCard.classList.add('state-speaking');
        assistantStateChip.textContent = 'SPEAKING';
        orbStatusHeadline.textContent  = 'Assistant Responding';
        orbStatusSub.textContent       = message || 'Streaming synthesized audio.';
        micIcon.innerHTML = ICON_MIC;
        break;

      case 'DISCONNECTED':
      default:
        orbCard.classList.add('state-disconnected');
        assistantStateChip.textContent = 'OFFLINE';
        orbStatusHeadline.textContent  = 'Engine Disconnected';
        orbStatusSub.textContent       = 'Awaiting WebSocket reconnection…';
        latencyMetric.textContent      = '-- ms';
        micIcon.innerHTML = ICON_MIC;
        break;
    }
  }

  // ==========================================================================
  // Terminal Logger
  // ==========================================================================

  function logTerminal(type, message, category = 'all') {
    totalLogs++;
    logCount.textContent = `${totalLogs} event${totalLogs === 1 ? '' : 's'} logged`;

    const row       = document.createElement('div');
    row.className   = 'terminal-line';
    row.dataset.category = category;

    const timeStr = new Date().toTimeString().split(' ')[0];

    const TAG_MAP = {
      'ws-recv' : ['tag-ws-recv', 'WS RECV'],
      'ws-send' : ['tag-ws-send', 'WS SENT'],
      'state'   : ['tag-state',   'STATE'],
      'error'   : ['tag-error',   'ERROR'],
    };
    const [tagClass, tagLabel] = TAG_MAP[type] || ['tag-system', 'SYS'];

    row.innerHTML = `
      <span class="log-time">${timeStr}</span>
      <span class="log-tag ${tagClass}">[${tagLabel}]</span>
      <span class="log-message">${escapeHtml(message)}</span>
    `;

    if (activeFilter !== 'all' && category !== activeFilter && category !== 'all') {
      row.style.display = 'none';
    }

    terminalOutput.appendChild(row);
    if (chkAutoScroll.checked) {
      terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ==========================================================================
  // User Interactions
  // ==========================================================================

  async function toggleListening(forceStart = false) {
    clearTimeout(followUpTimer);
    if (currentState === 'SPEAKING' || isAudioPlaying) {
      stopAudioPlayback();
      sendWebSocketPayload({ type: 'interrupt' });
      stopMicCapture();
      if (!forceStart) {
        setAssistantState('IDLE', 'Assistant stopped.');
        return;
      }
    }
    if (forceStart || (!isMicActive && currentState !== 'LISTENING' && currentState !== 'CAPTURING')) {
      sendWebSocketPayload({ type: 'start_listening' });
      await startMicCapture();
    } else {
      stopMicCapture();
      sendWebSocketPayload({ type: 'stop_listening' });
    }
  }

  btnVoiceTrigger.addEventListener('click', toggleListening);
  btnStartListening.addEventListener('click', toggleListening);

  btnSimulateTurn.addEventListener('click', () => {
    const utterance = txtCustomUtterance.value.trim() || 'Simulated voice prompt from operator.';
    sendWebSocketPayload({ type: 'simulate_cycle', utterance });
  });

  btnPing.addEventListener('click', sendPing);

  btnReconnect.addEventListener('click', () => {
    if (socket) socket.close();
    initWebSocket();
  });

  function submitUtterance() {
    const text = txtCustomUtterance.value.trim();
    if (!text) return;
    sendWebSocketPayload({ type: 'chat_message', content: text });
    txtCustomUtterance.value = '';
  }

  btnSendUtterance.addEventListener('click', submitUtterance);
  txtCustomUtterance.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') submitUtterance();
  });

  btnClearLogs.addEventListener('click', () => {
    terminalOutput.innerHTML = '';
    totalLogs = 0;
    logCount.textContent = '0 events logged';
    logTerminal('system', 'Terminal buffer cleared.');
  });

  filterTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      filterTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      activeFilter = tab.dataset.filter;

      terminalOutput.querySelectorAll('.terminal-line').forEach((line) => {
        const cat = line.dataset.category;
        line.style.display = (activeFilter === 'all' || cat === activeFilter || cat === 'all')
          ? 'flex' : 'none';
      });
    });
  });

  // Heartbeat ping every 15 seconds
  setInterval(() => {
    if (socket && socket.readyState === WebSocket.OPEN) sendPing();
  }, 15000);

  // ── Boot ───────────────────────────────────────────────────────────────────
  initWebSocket();
});
