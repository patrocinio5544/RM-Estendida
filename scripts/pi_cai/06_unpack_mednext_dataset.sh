#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ARCHIVE="${1:-}"

fail() {
    echo "ERRO: $*" >&2
    exit 1
}

[ -n "$ARCHIVE" ] || fail "Uso: ./scripts/pi_cai/06_unpack_mednext_dataset.sh caminho/do/pacote.tar.zst"
[ -f "$ARCHIVE" ] || fail "Arquivo não encontrado: $ARCHIVE"

if [ -f "${ARCHIVE}.sha256" ]; then
    EXPECTED="$(awk '{print $1}' "${ARCHIVE}.sha256")"
    ACTUAL="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    [ "$EXPECTED" = "$ACTUAL" ] || fail "Checksum inválido"
    echo "Checksum OK"
fi

case "$ARCHIVE" in
    *.tar.zst)
        command -v zstd >/dev/null 2>&1 || fail "Instale zstd: sudo apt install -y zstd"
        tar -I zstd -xf "$ARCHIVE" -C "$ROOT"
        ;;
    *.tar.gz|*.tgz)
        tar -xzf "$ARCHIVE" -C "$ROOT"
        ;;
    *.tar)
        tar -xf "$ARCHIVE" -C "$ROOT"
        ;;
    *)
        fail "Formato não suportado: $ARCHIVE"
        ;;
esac

echo "Dataset extraído."
echo "Treino recomendado:"
echo "python nnunet_mednext/run/run_training.py 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 220 0"
