FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY proto_contract ./proto_contract
COPY gateway ./gateway
COPY business ./business
COPY harness ./harness

# Install the openai extra too: in OpenAI mode the responder-thinker agent's
# thinker uses the openai SDK. (The gateway's realtime backend only needs
# websockets, but this shared image also serves the business plane.)
RUN pip install --no-cache-dir -e ".[openai]"

# Default; overridden per-service in compose.
CMD ["python", "-m", "business.server"]
