"""The gRPC transport, defined without a ``.proto`` / protoc step.

We register a single bidirectional ``stream_stream`` method whose request and
response are raw ``bytes``. The serializers are identity functions, so the
bytes we put on the wire are exactly our orjson-serialized envelopes from
``proto_contract``. gRPC gives us HTTP/2 multiplexing, flow control, and
bidirectional streaming; we supply the message schema ourselves.

This is the same shape as the real production gateway: gRPC for the pipe,
JSON envelopes for the payload. Defining the method by hand (rather than via
generated stubs) keeps the whole transport legible in one file — there is no
hidden generated code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import grpc

# The fully-qualified method the business plane serves and the gateway calls.
# Business plane is the gRPC *server*; gateway is the gRPC *client* that opens
# the stream. (The stream is bidirectional, so "who calls whom" is just about
# who listens — events and commands both flow over the one open stream.)
SERVICE_NAME = "voicepoc.VoiceBridge"
METHOD_NAME = "Bridge"
FULL_METHOD = f"/{SERVICE_NAME}/{METHOD_NAME}"


def _identity(x: bytes) -> bytes:
    return x


# --------------------------------------------------------------------------- #
# Server side: register the handler on an aio server.
# --------------------------------------------------------------------------- #
def add_bridge_handler(
    server: grpc.aio.Server,
    handler: Callable[[AsyncIterator[bytes], grpc.aio.ServicerContext], AsyncIterator[bytes]],
) -> None:
    """Register ``handler`` as the bidirectional Bridge method on ``server``.

    ``handler`` receives an async iterator of inbound request bytes (events
    from the gateway) and the servicer context, and yields outbound response
    bytes (commands to the gateway).
    """
    rpc_method = grpc.stream_stream_rpc_method_handler(
        handler,
        request_deserializer=_identity,
        response_serializer=_identity,
    )
    generic_handler = grpc.method_handlers_generic_handler(
        SERVICE_NAME, {METHOD_NAME: rpc_method}
    )
    server.add_generic_rpc_handlers((generic_handler,))


# --------------------------------------------------------------------------- #
# Client side: open the stream from the gateway.
# --------------------------------------------------------------------------- #
def open_bridge_stream(
    channel: grpc.aio.Channel,
    metadata: list[tuple[str, str]] | None = None,
) -> grpc.aio.StreamStreamCall:
    """Open the bidirectional Bridge stream from the gateway side.

    Returns a call object you can ``write()`` events to and async-iterate for
    commands. ``metadata`` carries the auth token and trace context.
    """
    multicallable = channel.stream_stream(
        FULL_METHOD,
        request_serializer=_identity,
        response_deserializer=_identity,
    )
    return multicallable(metadata=metadata or [])


__all__ = ["SERVICE_NAME", "METHOD_NAME", "FULL_METHOD",
           "add_bridge_handler", "open_bridge_stream"]
