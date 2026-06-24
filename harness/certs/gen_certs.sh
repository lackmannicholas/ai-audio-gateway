#!/usr/bin/env bash
# Generate a local CA plus server (business plane) and client (gateway) certs
# for mutual TLS on the bridge. For local dev only — do not ship these.
#
#   ./harness/certs/gen_certs.sh
#
# Produces, in this directory: ca.crt, server.crt, server.key, client.crt,
# client.key. The business plane and gateway auto-detect these (see
# proto_contract/auth.py); if absent, they fall back to an insecure channel.
set -euo pipefail
cd "$(dirname "$0")"

SUBJ_CA="/CN=audio-gateway-poc-ca"
SUBJ_SRV="/CN=localhost"
SUBJ_CLI="/CN=gateway-client"

echo "==> CA"
openssl genrsa -out ca.key 2048 2>/dev/null
openssl req -x509 -new -nodes -key ca.key -sha256 -days 825 \
  -subj "$SUBJ_CA" -out ca.crt 2>/dev/null

gen_cert () {
  local name=$1 subj=$2
  openssl genrsa -out "$name.key" 2048 2>/dev/null
  openssl req -new -key "$name.key" -subj "$subj" -out "$name.csr" 2>/dev/null
  # SAN localhost so the gateway can verify the server cert by hostname.
  openssl x509 -req -in "$name.csr" -CA ca.crt -CAkey ca.key -CAcreateserial \
    -days 825 -sha256 -out "$name.crt" \
    -extfile <(printf "subjectAltName=DNS:localhost,IP:127.0.0.1") 2>/dev/null
  rm -f "$name.csr"
}

echo "==> server cert (business plane)"
gen_cert server "$SUBJ_SRV"
echo "==> client cert (gateway)"
gen_cert client "$SUBJ_CLI"

rm -f ca.srl
echo "==> done. mTLS certs in $(pwd)"
