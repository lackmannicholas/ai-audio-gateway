# Audio Gateway POC

**A two-plane architecture for real-time voice AI — the split that keeps a
voice agent responsive under load, shown in ~3,000 lines you can run with no API
key.**

Voice agents fail in tells: they talk over you, they pause a beat too long, they
stutter when they should be listening. Most of that isn't the model being dumb —
it's a *systems* problem. The audio path is hard real time: a 20 ms frame every
20 ms, forever, or the caller hears a glitch. Reasoning is the opposite —
slow, bursty, variable; a tool call might take two seconds. Put both in one
process and the slow thing janks the fast thing: an LLM call blocks the event
loop and the audio hiccups, or a barge-in lags because you're busy awaiting a
completion.

The production answer is to stop pretending they're the same workload. Split the
system into two planes:

- a **media plane** — the *audio gateway* — that owns hard real-time work:
  browser WebRTC, audio I/O, VAD, pacing, barge-in;
- a **business plane** that owns meaning: prompts, tools, multi-step reasoning.

Between them runs a small, typed contract over a bidirectional gRPC stream. The
realtime model only ever holds **hollow proxy tools**: it calls them like local
functions, but every call relays across the wire to the business plane, which is
where execution actually happens. The media plane never blocks on business work;
audio keeps flowing while a tool call is in flight.

This repo demonstrates the whole pattern end-to-end with a café ordering
assistant, and runs on a **mock realtime model** by default — no API key, no
cost — so you can watch the architecture work before you spend a token.

```
  browser   ◀─WebRTC audio──▶  ┌───────────────────────┐  ──gRPC bidi stream──▶  ┌───────────────────┐
 ("phone")  ◀───SSE events───  │     AUDIO GATEWAY     │  ◀───────────────────   │   BUSINESS PLANE  │
                               │     media plane       │                         │   meaning plane   │
  realtime  ──WebSocket─────▶  │                       │                         │                   │
   model    ◀────────────────  │  VAD · pacing ·       │                         │  agents · tools · │
 (mock|AI)                     │  barge-in · turn_id · │                         │  thinker ·        │
                               │  hollow proxy tools   │                         │  turn staleness   │
                               └───────────────────────┘                         └───────────────────┘
```

## Run it

```bash
# Option A: docker — both planes, mock backend, zero deps on your machine
docker compose up --build
# then open http://localhost:8001

# Option B: local Python (3.13+)
make install
make run            # business plane on :8002, gateway on :8001
# open http://localhost:8001, click Connect, allow the mic, and order a coffee
```

Want a real speech-to-speech conversation instead of the mock? Set an OpenAI key
and flip the backends:

```bash
export OPENAI_API_KEY=sk-...
# Optional: mirrors the responder-thinker regional endpoint.
export OPENAI_BASE_URL=https://us.api.openai.com/v1
make run-openai      # REALTIME_BACKEND=openai THINKER_BACKEND=openai
```

The responder-thinker agent's thinker uses the OpenAI SDK in this mode, so the
`openai` extra must be installed. `make install` and the Docker image already
include it; if you installed some other way, add `pip install -e ".[openai]"`.
Note the repo defaults to the regional host `us.api.openai.com` for both the
realtime and thinker paths; on a standard global account set
`OPENAI_BASE_URL=https://api.openai.com/v1`.

Local settings can live in a repo-root `.env` (see `.env.example`); values
already set in your shell take precedence. The realtime websocket URL resolves in
order: `OPENAI_REALTIME_URL`, then `REALTIME_API_URL`, then derived from
`OPENAI_BASE_URL`, otherwise `wss://us.api.openai.com/v1/realtime`.

Optional mutual TLS on the bridge:

```bash
make certs           # generates a local CA + server/client certs
make run             # both planes now auto-detect the certs and use mTLS
```

## What you're looking at

The demo UI has an agent switcher, and flipping it shows the entire point of the
proxy boundary. Watch the architecture view as you order:

- **Single agent** exposes the four café tools (`get_menu`, `get_store_hours`,
  `place_order`, `check_order_status`) directly. The gateway builds **four**
  proxies; *every* tool call crosses the wire. Flat and simple — fine for
  straightforward ordering.
- **Responder–Thinker** exposes **one** tool: the thinker. The gateway builds a
  single proxy. When the model calls it, one envelope crosses the wire, the
  thinker lights up, and *its* tool calls (`get_menu`, `place_order`, …) fire
  **inside** the business plane, logged as `(local)` because they never cross
  back. One envelope out, one back — but a whole tree happened behind it that the
  gateway is blind to.

That asymmetry is the architecture in one picture. The media plane draws the
boundary exactly where it stops caring: it relays *shapes*, and where the real
fan-out happens — one round trip or a dozen nested calls — is the business
plane's business.

Then hit **Barge-in** mid-response. Watch `turn_id` increment, the live turn go
stale, and — if a thinker run was in flight — the business plane abandon its
result rather than speak it over a conversation that moved on. More on why that
matters below.

## Hollow proxy tools

A tool has two halves that usually live together: a *schema* (name, description,
JSON args — what the model needs to decide to call it) and an *implementation*
(the code that does the work). This architecture splits them across the wire.

At call setup the business plane sends the gateway a list of `ToolSpec`s —
schema only, no behavior. The gateway builds one **proxy** per spec: an object
the realtime model can call, whose entire body is *"relay this call across the
wire and await the result."* No business logic, no data access. Strip the
business plane away and the proxies relay to nothing.

That's what lets the media plane stay dumb and fast. It holds the shapes the
model reasons over; execution — and everything execution needs (menus, a
database, other services) — stays in the meaning plane, off the audio hot path.

## The contract

There is no `.proto`. The wire format is the Pydantic envelopes in
[`proto_contract/envelopes.py`](proto_contract/envelopes.py), serialized as JSON
over a gRPC `stream_stream` with identity serializers — you get HTTP/2
multiplexing and bidirectional streaming, with a human-readable, schema-validated
payload you can log and read.

Two enums *are* the contract; read them top to bottom and you know the whole
vocabulary of the boundary:

- `GatewayEventType` — things the gateway tells the business plane *happened*.
  Past tense. Reports of reality (`call.started`, `user.speech_stopped`,
  `tool_call.requested`, `barge_in`).
- `GatewayCommandType` — things the business plane tells the gateway to *do*.
  Imperative (`session.configure`, `tool_call.output`, `response.cancel`).

The asymmetry is deliberate: the media plane owns reality and reports it; the
business plane owns meaning and directs the media plane through a small command
vocabulary.

## Barge-in and turn staleness — coordinating two processes with one integer

This is the part that's genuinely hard, and the reason the split has to be done
carefully. When the caller interrupts, two things must happen fast: the assistant
has to stop talking *now*, and any reasoning already in flight for the old turn
has to not come back and speak over the new one.

The gateway owns a `turn_id` that increments on every barge-in. That number rides
on each `tool_call.requested` frame. The business plane mirrors it, snapshots it
before slow work, and checks it after: if the live turn has moved, the work is
stale and its result is discarded instead of spoken. Two processes, no shared
memory, coordinating real-time state through **one integer on the wire**.

Stopping the audio is its own small performance story. The gateway paces
assistant audio out at real time (one 20 ms frame per 20 ms) even though the
model streams it faster, so the queue *is* the backlog of not-yet-heard speech.
Barge-in clears that queue instantly. It also has to fire while audio is still
draining — not just while the model is generating — because a fast model finishes
generating long before the caller has heard it all. In OpenAI mode the gateway
additionally cancels the response, drops in-flight audio deltas from the
cancelled response, and truncates the model's conversation item to roughly what
was actually played, so the next turn isn't grounded in words the caller never
heard.

## Endpointing: one clock, one authority

Deciding *when the caller stopped talking* is where responsiveness is won or
lost, so exactly one component owns it — never zero, never two. Preferred is
local **TEN VAD** in the gateway: when it's active, the gateway disables the
realtime provider's server-side turn detection, and its own VAD commits the audio
buffer and asks for a response. (Run both and every utterance triggers duplicate
responses.) On platforms with no TEN VAD build — notably Linux `aarch64`, i.e.
Docker on Apple Silicon — the gateway falls back to a passthrough that does *not*
gate, and leaves the provider's server VAD on so something still endpoints.

The single most important tuning knob lives here: `VAD__HANGOVER_FRAMES` is how
much trailing silence must accumulate before the utterance is committed, counted
in ~20 ms chunks (15 = 300 ms). That's a hard floor on response latency, and the
exact parameter background noise attacks — noise holds the gate open and the
agent appears to stall. To make it measurable, the gateway emits a
**`turn_latency`** event per turn (utterance commit → first assistant audio),
visible in the browser event log and the gateway log.

## How a tool gets runtime context

A tool's job is `invoke(arguments, ctx) -> result`. The `arguments` come from the
model; the `ctx` is a [`ToolContext`](business/tools/base.py) the business
session builds per call and threads to *every* tool — carrying the call id, the
turn this invocation belongs to, and a live `is_stale()` check. The thinker
passes the same `ctx` down into its nested café-tool calls, so a barge-in
detected three levels into the fan-out abandons the whole tree. It's the same
`turn_id`-staleness mechanism as above, surfaced as a first-class parameter
rather than something bolted onto one special tool.

## Layout

```
proto_contract/   the wire contract (envelopes, channel, auth) — shared by both planes
gateway/          media plane: ASGI app, gRPC client (proxies), realtime backends, audio
business/         business plane: gRPC server (relay), agents, tools, the thinker loop
harness/          browser audio client + mTLS cert generator
tests/            contract, proxy relay, turn staleness, endpointing, full stack, mTLS
```

## Notes & honest caveats

- The **mock realtime model** is a scripted state machine, not an LLM. It speaks
  enough of a realtime event protocol to drive a full turn (transcript → tool
  call → audio) so the architecture runs deterministically without a key. Browser
  speech synthesis stands in as audible output in mock mode; use `make run-openai`
  for real speech-to-speech.
- The agent loop (the **thinker**) is **hand-rolled** on purpose — no agent SDK —
  so the tool-calling loop is visible in ~40 lines. Ask the model what to do; run
  any tools it wants; feed results back; repeat until it answers. The model behind
  it is swappable (mock | OpenAI).
- Domain state (the order book) is an in-memory singleton for the demo. A real
  deployment would key it by call and inject it through the `ToolContext` rather
  than reach a global.
- **mTLS + a shared token** secure the bridge. In production you'd use rotated,
  short-lived signed tokens from a secrets manager rather than a static secret.
- The browser's WebRTC jitter buffer leaves an unavoidable sub-~200 ms audio tail
  after a barge-in; the seconds-long, server-side backlog is what the queue-clear
  eliminates.

This is a teaching artifact: small enough to read in a sitting, real enough to
run, and structured to make the architecture legible rather than to be a
framework. Fork it, flip the backends, and watch the wire.
```
