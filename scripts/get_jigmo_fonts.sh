#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ZIP_URL="https://kamichikoichi.github.io/jigmo/Jigmo-20250912.zip"

DOWNLOAD_DIR="${1:-${REPO_ROOT}/downloads}"
FONTS_DIR="${2:-${REPO_ROOT}/fonts}"
ZIP_PATH="${DOWNLOAD_DIR}/Jigmo-20250912.zip"

mkdir -p "${DOWNLOAD_DIR}" "${FONTS_DIR}"
curl -fsSL "${ZIP_URL}" -o "${ZIP_PATH}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

for name in Jigmo.ttf Jigmo2.ttf Jigmo3.ttf; do
  src="$(find "${TMP_DIR}" -type f -name "${name}" -print -quit)"
  if [[ -z "${src}" ]]; then
    echo "Missing ${name} in archive." >&2
    exit 1
  fi
  cp "${src}" "${FONTS_DIR}/${name}"
done

echo "Installed Jigmo fonts into ${FONTS_DIR}"
