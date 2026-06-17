#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RAW_TASK="data/nnUNet_raw/nnUNet_raw_data/Task220_PI-CAI"
PREP_TASK="data/nnUNet_preprocessed/Task220_PI-CAI"
OUT_DIR="artifacts/datasets"
STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="rm-estendida_picai220_mednext_medium_ready_${STAMP}"
MANIFEST="${OUT_DIR}/${BASE}_manifest.txt"

fail() {
    echo "ERRO: $*" >&2
    exit 1
}

[ -d "$RAW_TASK" ] || fail "Não encontrei $RAW_TASK"
[ -f "$RAW_TASK/dataset.json" ] || fail "Não encontrei dataset.json"
[ -d "$RAW_TASK/imagesTr" ] || fail "Não encontrei imagesTr"
[ -d "$RAW_TASK/labelsTr" ] || fail "Não encontrei labelsTr"
[ -d "$PREP_TASK" ] || fail "Não encontrei $PREP_TASK. Rode o preprocessing v1 antes."

mkdir -p "$OUT_DIR"

IMAGES_COUNT="$(find "$RAW_TASK/imagesTr" -type f -name '*.nii.gz' | wc -l)"
LABELS_COUNT="$(find "$RAW_TASK/labelsTr" -type f -name '*.nii.gz' | wc -l)"

[ "$IMAGES_COUNT" -eq 4500 ] || fail "imagesTr deveria ter 4500 arquivos, mas tem $IMAGES_COUNT"
[ "$LABELS_COUNT" -eq 1500 ] || fail "labelsTr deveria ter 1500 arquivos, mas tem $LABELS_COUNT"

cat > "$MANIFEST" <<EOF
RM-Estendida — PI-CAI 220 MedNeXt Medium-ready package

Raw task:
$RAW_TASK

Preprocessed task:
$PREP_TASK

Counts:
imagesTr: $IMAGES_COUNT
labelsTr: $LABELS_COUNT

Recommended trainer:
python nnunet_mednext/run/run_training.py 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel5 220 0

Fallback:
python nnunet_mednext/run/run_training.py 3d_fullres nnUNetTrainerV2_MedNeXt_M_kernel3 220 0
EOF

if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="${OUT_DIR}/${BASE}.tar.zst"
    tar -I "zstd -T0 -6" -cf "$ARCHIVE" "$RAW_TASK" "$PREP_TASK" "$MANIFEST"
else
    ARCHIVE="${OUT_DIR}/${BASE}.tar.gz"
    tar -czf "$ARCHIVE" "$RAW_TASK" "$PREP_TASK" "$MANIFEST"
fi

sha256sum "$ARCHIVE" > "${ARCHIVE}.sha256"

echo "Pacote criado:"
du -h "$ARCHIVE"
cat "${ARCHIVE}.sha256"
