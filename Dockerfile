FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY proto_contract ./proto_contract
COPY gateway ./gateway
COPY business ./business
COPY harness ./harness

RUN pip install --no-cache-dir -e .

# Default; overridden per-service in compose.
CMD ["python", "-m", "business.server"]
