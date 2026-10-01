#!/usr/bin/env bash
# Triton has no macOS wheel, so the local venv can't import it and the IDE can't
# resolve `import triton`. This downloads the Linux wheel (~250 MB, NOT installed) and
# extracts only its Python sources into .triton-src/ for go-to-definition / completion.
# The version matches what the Modal image installs. Kernels still run on Modal.
set -euo pipefail
VERSION="${1:-3.8.0}"
cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
.venv/bin/pip download "triton==$VERSION" --no-deps --only-binary=:all: \
    --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 \
    --python-version 3.11 -d "$tmp"
rm -rf .triton-src && mkdir .triton-src
unzip -q "$tmp"/triton-*.whl 'triton/*.py' -d .triton-src
echo "extracted triton $VERSION sources to .triton-src/"
