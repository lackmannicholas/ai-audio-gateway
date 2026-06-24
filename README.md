# Audio Gateway POC

A small, **runnable** reference for the two-plane architecture behind production
voice AI: split the system into a **media plane** (the audio gateway) and a
**business plane** (agents, tools, reasoning), and bridge them with a
bidirectional gRPC stream.

The gateway owns hard real-time work — browser WebRTC, audio I/O, VAD, pacing,
barge-in. The
business plane owns meaning — prompts, tools, multi-step reasoning. Between them
runs a small, typed contract. The realtime model only ever holds **hollow proxy
tools**: it calls them like local functions, but every call relays across the
wire to the business plane, which is where execution actually happens.

This repo demonstrates the pattern end-to-end with a café ordering assistant,
and runs entirely on a **mock realtime model** — no API key, no cost.

```
┌────────────────┐ WebRTC  ┌─────────────────────┐  gRPC   ┌────────────────┐
│ browser        │ ──────▶ │  AUDIO GATEWAY      │ ◀─────▶ │ BUSINESS PLANE │
│ (plays "phone")│         │  media plane         │  bidi   │ meaning        │
└────────────────┘         │  VAD · pacing ·      │ stream  │ agents · tools │
┌────────────────┐   WS    │  barge-in · proxies  │         │ thinker ·      │
│ realtime model │ ◀─────▶ │  turn_id             │         │ staleness      │
│ (mock | openai)│         └─────────────────────┘         └────────────────┘
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

Want it real instead of mocked? Set an OpenAI key and flip the backends:

```bash
export OPENAI_API_KEY=sk-...
# Optional: mirrors the responder-thinker regional endpoint.
export OPENAI_BASE_URL=https://us.api.openai.com/v1
make run-openai      # REALTIME_BACKEND=openai THINKER_BACKEND=openai
```

You can also put local settings in a repo-root `.env` file. Values already set
in your shell take precedence over `.env`; see `.env.example` for the common
knobs. The realtime websocket uses `OPENAI_REALTIME_URL` if set, then
`REALTIME_API_URL`, then derives from `OPENAI_BASE_URL`, and otherwise defaults
to `wss://us.api.openai.com/v1/realtime`.

Optional mutual TLS on the bridge:

```bash
make certs           # generates a local CA + server/client certs
make run             # both planes now auto-detect the certs and use mTLS
```

## What to look for

Flip the agent switcher in the header and watch the architecture view:

- **Single agent** exposes the four café tools directly. The gateway builds
  **four** proxies; *every* tool call crosses the wire. Flat.
- **Responder–Thinker** exposes **one** tool — the thinker. The gateway builds a
  single proxy. When the model calls it, one envelope crosses the wire, the
  thinker lights up, and its own tool calls (`get_menu`, `place_order`) fire
  *inside* the business plane, logged as `(local)` because they never cross
  back. **One envelope out, one back — but a whole tree happened behind it that
  the gateway is blind to.** That asymmetry is the entire point of the proxy
  boundary.

Then hit **Barge-in** mid-response and watch `turn_id` increment and the live
turn go stale. If a thinker run was in flight, the business plane detects the
turn moved and abandons the result rather than speaking it over a conversation
that moved on — coordinated across two processes through nothing but the
`turn_id` on the wire.

## The contract

There is no `.proto`. The wire format is the Pydantic envelopes in
[`proto_contract/envelopes.py`](proto_contract/envelopes.py), serialized as JSON
over a gRPC `stream_stream` with identity serializers — HTTP/2 multiplexing and
bidirectional streaming, with a human-readable, schema-validated payload. The
two enums (`GatewayEventType`, `GatewayCommandType`) *are* the contract: read
them top to bottom and you know the whole vocabulary of the boundary.

## Layout

```
proto_contract/   the wire contract (envelopes, channel, auth) — shared by both planes
gateway/          media plane: ASGI app, gRPC client (proxies), realtime backends, audio
business/         business plane: gRPC server (relay), agents, tools, the thinker loop
harness/          browser audio client + mTLS cert generator
tests/            contract, proxy relay, turn staleness, full stack, mTLS
```

## Notes & honest caveats

- The **mock realtime model** is a scripted state machine, not an LLM. It speaks
  enough of a realtime event protocol to drive a full turn (transcript → tool
  call → audio) so the architecture runs deterministically without a key.
  Browser speech synthesis is used as an audible stand-in in mock mode; use
  `make run-openai` for a real speech-to-speech conversation.
- The **VAD** is a trivial energy gate — good enough to demo barge-in, not
  production-grade. Swap in Silero/WebRTC VAD behind the same interface.
- **mTLS + a shared token** secure the bridge. In production you'd use rotated,
  short-lived signed tokens from a secrets manager rather than a static secret.
- The agent loop (the "thinker") is **hand-rolled** on purpose — no agent SDK —
  so the tool-calling loop is visible in ~40 lines. The model behind it is
  swappable (mock | OpenAI).

This is a teaching artifact: small enough to read in a sitting, real enough to
run, and structured to make the architecture legible rather than to be a
framework.
```
