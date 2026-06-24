# Audio Gateway POC

.PHONY: install certs business gateway run test clean

install:
	pip install -e ".[dev]"

# Generate local mTLS certs (optional — without them the bridge runs insecure).
certs:
	bash harness/certs/gen_certs.sh

# Run the two planes in separate terminals:
business:
	python -m business.server

gateway:
	python -m gateway.server

# Convenience: run both (business in background) for a quick local demo.
run:
	@echo "Starting business plane on :8002 and gateway on :8001"
	@python -m business.server & echo $$! > .business.pid
	@sleep 1
	@python -m gateway.server || true
	@kill `cat .business.pid` 2>/dev/null || true
	@rm -f .business.pid

# Run with the real OpenAI Realtime + thinker backends (needs OPENAI_API_KEY).
run-openai:
	@REALTIME_BACKEND=openai THINKER_BACKEND=openai $(MAKE) run

test:
	python -m pytest -q

clean:
	rm -f harness/certs/*.crt harness/certs/*.key harness/certs/*.srl .business.pid
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
