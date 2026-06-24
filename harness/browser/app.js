// Audio Gateway POC — browser client.
// Connects to the gateway over a websocket, streams mic audio in, and renders
// the live architecture view from the UI event stream the gateway emits.

let ws = null;
let agent = "cafe_single";
let audioCtx = null;
let micStream = null;
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
  if (ws) return; // can't switch mid-call
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

// ---- event stream from the gateway ----
function handleEvent(ev) {
  switch (ev.kind) {
    case "connected":
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
    case "audio_delta":
      $("gw-playback").classList.add("active");
      setTimeout(() => $("gw-playback").classList.remove("active"), 120);
      break;
    case "response_done":
      logEvent("response.done"); break;
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

// ---- mic capture: downsample to 8kHz 16-bit PCM, 20ms frames ----
async function startMic() {
  micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 8000 });
  const src = audioCtx.createMediaStreamSource(micStream);
  const proc = audioCtx.createScriptProcessor(2048, 1, 1);
  src.connect(proc); proc.connect(audioCtx.destination);
  proc.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== 1) return;
    const f32 = e.inputBuffer.getChannelData(0);
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    const bytes = new Uint8Array(i16.buffer);
    let bin = ""; for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    ws.send(JSON.stringify({ kind: "audio", pcm_b64: btoa(bin) }));
  };
}

function stopMic() {
  if (micStream) micStream.getTracks().forEach(t => t.stop());
  if (audioCtx) audioCtx.close();
  micStream = null; audioCtx = null;
}

async function toggleConnect() {
  if (ws) { ws.close(); return; }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws?agent=${agent}`);
  ws.onopen = async () => {
    $("dot").classList.add("on");
    $("statusText").textContent = "connected";
    $("connectBtn").textContent = "■ Disconnect";
    vizActive(true);
    newLiveTurn("listening…");
    try { await startMic(); } catch (e) { logEvent("mic error: " + e.message, "barge"); }
  };
  ws.onmessage = (m) => handleEvent(JSON.parse(m.data));
  ws.onclose = () => {
    $("dot").classList.remove("on");
    $("statusText").textContent = "disconnected";
    $("connectBtn").textContent = "▶ Connect";
    vizActive(false); stopMic(); ws = null;
  };
}

function sendBargeIn() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ kind: "barge_in" }));
}
