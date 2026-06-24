"""Auth for the bridge: a shared token in gRPC metadata, plus mTLS helpers.

Two layers, both deliberately lightweight but production-shaped:

  1. mTLS — the channel itself is mutually authenticated. The gateway presents
     a client cert; the business plane presents a server cert; each verifies the
     other against a shared CA. This is what stops anything on the network from
     opening the bridge. See ``harness/certs/gen_certs.sh`` to generate a local
     CA + certs for the demo.

  2. Token-in-metadata — a shared secret sent as the ``x-bridge-token`` header
     on the stream, checked once when the stream opens. This is the
     application-level "are you allowed to talk to me" check on top of "are you
     who you say you are" from mTLS.

For the POC the token comes from the BRIDGE_TOKEN env var (default provided so
``docker compose up`` just works). In production this would be a rotated secret
from a secrets manager, and you'd likely use short-lived signed tokens rather
than a static shared secret — noted in the README as the obvious hardening step.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import grpc

_DEFAULT_TOKEN = "poc-dev-bridge-token"  # fine for a local demo; rotate in prod


def bridge_token() -> str:
    return os.getenv("BRIDGE_TOKEN", _DEFAULT_TOKEN)


def verify_token(presented: str | None) -> bool:
    """Constant-time comparison against the expected bridge token."""
    if not presented:
        return False
    return hmac.compare_digest(presented, bridge_token())


def auth_metadata() -> list[tuple[str, str]]:
    """Metadata the gateway attaches when opening the stream."""
    return [("x-bridge-token", bridge_token())]


# --------------------------------------------------------------------------- #
# mTLS credential helpers. If the cert files are absent, callers fall back to
# insecure (local dev) — the README explains how to generate them.
# --------------------------------------------------------------------------- #
def _read(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return None


def _mtls_disabled() -> bool:
    return os.getenv("BRIDGE_INSECURE", "").lower() in ("1", "true", "yes")


def server_credentials(cert_dir: str = "harness/certs") -> grpc.ServerCredentials | None:
    if _mtls_disabled():
        return None
    d = Path(cert_dir)
    ca = _read(d / "ca.crt")
    crt = _read(d / "server.crt")
    key = _read(d / "server.key")
    if not (ca and crt and key):
        return None
    return grpc.ssl_server_credentials(
        [(key, crt)],
        root_certificates=ca,
        require_client_auth=True,  # mutual TLS
    )


def channel_credentials(cert_dir: str = "harness/certs") -> grpc.ChannelCredentials | None:
    if _mtls_disabled():
        return None
    d = Path(cert_dir)
    ca = _read(d / "ca.crt")
    crt = _read(d / "client.crt")
    key = _read(d / "client.key")
    if not (ca and crt and key):
        return None
    return grpc.ssl_channel_credentials(
        root_certificates=ca, private_key=key, certificate_chain=crt
    )


__all__ = ["bridge_token", "verify_token", "auth_metadata",
           "server_credentials", "channel_credentials"]
