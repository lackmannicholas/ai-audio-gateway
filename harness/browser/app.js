// Audio Gateway POC — browser client.
// Connects to the gateway over WebRTC for audio. Structured UI events are
// streamed over SSE from the same gateway session.

let peerConnection = null;
let eventSource = null;
let agent = "cafe_single";
let micStream = null;
let remoteAudio = null;
let sessionId = null;
let ttsEnabled = false;
let cleaningUp = false;
let turnId = 0;
let liveTurnEl = null;

const $ = (id) => document.getElementById(id);

// ---- visualizer bars ----
(function initViz() {
  const viz = $("viz");
  for (let i = 0; i < 22; i++) {
    const b = document.createElement("div");
    b.style.height = (4 + Math.random() * 18) + "px";
    viz.appendChild(b);
  }
})();
let vizTimer = null;
function vizActive(on) {
  if (on && !vizTimer) {
    vizTimer = setInterval(() => {
      [...$("viz").children].forEach(b => b.style.height = (4 + Math.random() * 18) + "px");
    }, 120);
  } else if (!on && vizTimer) {
    clearInterval(vizTimer); vizTimer = null;
    [...$("viz").children].forEach(b => b.style.height = "4px");
  }
}

function setAgent(which) {
  if (peerConnection) return; // can't switch mid-call
  agent = which;
  $("sw-single").classList.toggle("active", which === "cafe_single");
  $("sw-rt").classList.toggle("active", which === "cafe_responder_thinker");
}

function logEvent(text, cls) {
  const d = document.createElement("div");
  d.className = "ev " + (cls || "");
  const t = new Date().toLocaleTimeString("en-US", { hour12: false });
  d.textContent = t + " " + text;
  const log = $("evlog");
  log.insertBefore(d, log.firstChild);
  while (log.children.length > 40) log.removeChild(log.lastChild);
}

function addMessage(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + (role === "user" ? "user" : "assistant");
  if (role !== "user") {
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = agent === "cafe_responder_thinker" ? "café responder" : "café agent";
    d.appendChild(who);
  }
  d.appendChild(document.createTextNode(text));
  $("convo").appendChild(d);
  $("convo").scrollTop = $("convo").scrollHeight;
}

function flash(el, cls = "active", ms = 700) {
  if (!el) return;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), ms);
}

function animatePacket(id) {
  const p = $(id);
  p.style.transition = "none"; p.style.opacity = "1"; p.style.left = "0%";
  requestAnimationFrame(() => {
    p.style.transition = "all .7s ease"; p.style.left = "65%";
    setTimeout(() => p.style.opacity = "0", 650);
  });
}

function newLiveTurn(label) {
  const tl = $("timeline");
  const row = document.createElement("div");
  row.className = "turn live";
  row.innerHTML = `<span class="id">t${turnId}</span><div class="bar">${label}<span class="tag">● live</span></div>`;
  tl.appendChild(row);
  if (liveTurnEl) liveTurnEl.classList.remove("live");
  liveTurnEl = row;
  while (tl.children.length > 5) tl.removeChild(tl.firstChild);
}

function markStale() {
  if (!liveTurnEl) return;
  liveTurnEl.classList.remove("live");
  liveTurnEl.classList.add("stale");
  const tag = liveTurnEl.querySelector(".tag");
  if (tag) { tag.textContent = "✕ stale"; tag.style.color = "var(--red)"; }
}

function resetSessionUi() {
  turnId = 0;
  liveTurnEl = null;
  $("turnval").textContent = "0";
  $("proxct").textContent = "×0";
  $("agentLabel").textContent = "—";
  $("timeline").replaceChildren();
  $("convo").replaceChildren();
  $("evlog").replaceChildren();
  document.querySelectorAll(".comp").forEach(el => {
    el.classList.remove("active", "fire", "firep");
  });
  document.querySelectorAll(".pkt").forEach(el => {
    el.style.opacity = "0";
    el.style.left = "0%";
  });
}

function speakAssistant(text) {
  if (!ttsEnabled || !("speechSynthesis" in window) || !text) return;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.03;
  utterance.pitch = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}

// ---- event stream from the gateway ----
function handleEvent(ev) {
  switch (ev.kind) {
    case "connected":
      ttsEnabled = !!ev.mock_tts;
      logEvent("connected · " + (ev.transport || "webrtc"));
      break;
    case "session_ended":
      cleanupConnection({ notify: false });
      break;
    case "error": {
      const message = ev.message || "session error";
      cleanupConnection({ notify: false });
      logEvent("error · " + message, "barge");
      break;
    }
    case "session_configured": {
      const tools = ev.tools || [];
      $("agentLabel").textContent = agent === "cafe_responder_thinker" ? "responder" : "single agent";
      const isRT = agent === "cafe_responder_thinker";
      $("rtWrap").style.display = isRT ? "block" : "none";
      $("toolsWrap").style.display = isRT ? "none" : "block";
      $("proxct").textContent = "×" + (tools.length || (isRT ? 1 : 4));
      logEvent("session.configure · " + (ev.agent || agent), "tool");
      break;
    }
    case "user_speech_started":
      flash($("gw-vad")); logEvent("user.speech_started"); break;
    case "user_speech_stopped":
      logEvent("user.speech_stopped"); break;
    case "transcript":
      addMessage(ev.role, ev.text);
      if (ev.role === "assistant") speakAssistant(ev.text);
      logEvent("transcript · " + ev.role); break;
    case "tool_call_requested":
      $("gw-proxy").classList.add("active");
      flash(agent === "cafe_responder_thinker" ? $("bz-thinker") : $("bz-tools"),
            agent === "cafe_responder_thinker" ? "firep" : "fire", 1400);
      animatePacket("pkt-lr");
      logEvent("tool_call.requested · " + ev.name, "tool");
      break;
    case "local_tool_call": {
      const el = $("th-" + ev.name);
      flash(el, "firep", 900);
      logEvent("  └ " + ev.name + " (local)", "local");
      break;
    }
    case "tool_call_output":
      animatePacket("pkt-rl");
      $("gw-proxy").classList.remove("active");
      logEvent("tool_call.output · " + (ev.name || ""), "tool");
      break;
    case "tool_call_stale":
      $("gw-proxy").classList.remove("active");
      logEvent("tool_call.stale · " + (ev.name || ""), "barge");
      break;
    case "audio_delta":
      $("gw-playback").classList.add("active");
      setTimeout(() => $("gw-playback").classList.remove("active"), 120);
      break;
    case "response_done":
      logEvent("response.done"); break;
    case "turn_latency":
      logEvent("turn latency · " + ev.ms + "ms (commit → first audio)");
      break;
    case "barge_in":
      turnId = ev.turn_id;
      $("turnval").textContent = turnId;
      flash($("gw-turn")); flash($("bz-stale"), "fire");
      markStale();
      logEvent("barge_in · turn_id→" + turnId, "barge");
      logEvent("response.cancel", "barge");
      break;
  }
}

function cleanupConnection(options = {}) {
  if (cleaningUp) return;
  cleaningUp = true;

  const notify = options.notify !== false;
  const sid = sessionId;
  sessionId = null;

  if (notify && sid) {
    try {
      navigator.sendBeacon(
        "/api/rtc/disconnect",
        new Blob([JSON.stringify({ session_id: sid })], { type: "application/json" }),
      );
    } catch (_) { }
  }

  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  if (peerConnection) {
    const pc = peerConnection;
    peerConnection = null;
    pc.onconnectionstatechange = null;
    pc.ontrack = null;
    pc.close();
  }
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (remoteAudio) {
    remoteAudio.pause();
    remoteAudio.srcObject = null;
    remoteAudio = null;
  }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  micStream = null;
  ttsEnabled = false;
  $("dot").classList.remove("on");
  $("statusText").textContent = "disconnected";
  $("connectBtn").textContent = "▶ Connect";
  vizActive(false);
  resetSessionUi();
  cleaningUp = false;
}

async function toggleConnect() {
  if (peerConnection) {
    cleanupConnection();
    return;
  }

  $("statusText").textContent = "connecting";
  $("connectBtn").textContent = "■ Disconnect";
  resetSessionUi();

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    peerConnection = new RTCPeerConnection();
    const pc = peerConnection;

    micStream.getTracks().forEach(track => pc.addTrack(track, micStream));

    pc.ontrack = (event) => {
      remoteAudio = new Audio();
      remoteAudio.autoplay = true;
      remoteAudio.srcObject = event.streams[0];
      remoteAudio.play().catch(() => logEvent("audio autoplay blocked", "barge"));
    };

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState;
      logEvent("rtc · " + state);
      if (state === "connected") {
        $("dot").classList.add("on");
        $("statusText").textContent = "connected";
      } else if (["failed", "closed", "disconnected"].includes(state)) {
        cleanupConnection();
      }
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await new Promise((resolve) => {
      if (pc.iceGatheringState === "complete") {
        resolve();
        return;
      }
      pc.addEventListener("icegatheringstatechange", () => {
        if (pc.iceGatheringState === "complete") resolve();
      });
    });

    const response = await fetch("/api/rtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, agent }),
    });
    if (!response.ok) throw new Error(`rtc offer failed: ${response.status}`);
    const answer = await response.json();
    sessionId = answer.session_id;

    eventSource = new EventSource(`/api/events/${sessionId}`);
    eventSource.onmessage = (event) => handleEvent(JSON.parse(event.data));
    eventSource.onerror = () => logEvent("event stream interrupted", "barge");

    await pc.setRemoteDescription({ type: "answer", sdp: answer.sdp });

    $("dot").classList.add("on");
    $("statusText").textContent = "connected";
    vizActive(true);
    newLiveTurn("listening…");
  } catch (e) {
    const message = e.message;
    cleanupConnection({ notify: false });
    logEvent("connect error: " + message, "barge");
  }
}

function sendBargeIn() {
  if (!sessionId) return;
  fetch("/api/rtc/barge-in", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  }).catch(() => logEvent("barge-in request failed", "barge"));
}

window.addEventListener("beforeunload", cleanupConnection);
