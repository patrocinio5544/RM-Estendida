import json
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk


# ============================================================
# PATHS
# ============================================================

EXTRACTED_IMAGES_DIR = Path("data/01_extracted")

LABELS_ROOT = Path("data/00_raw/picai_labels/csPCa_lesion_delineations/human_expert")
LABELS_RESAMPLED_DIR = LABELS_ROOT / "resampled"
LABELS_POOCH25_DIR = LABELS_ROOT / "Pooch25"

TASK_ID = 220
TASK_NAME = "PI-CAI"
NNUNET_TASK_DIR = Path(f"data/nnUNet_raw/nnUNet_raw_data/Task{TASK_ID}_{TASK_NAME}")

IMAGES_TR = NNUNET_TASK_DIR / "imagesTr"
LABELS_TR = NNUNET_TASK_DIR / "labelsTr"
IMAGES_TS = NNUNET_TASK_DIR / "imagesTs"

QA_DIR = NNUNET_TASK_DIR / "_qa"

MODALITIES = {
    "0000": "t2w",
    "0001": "adc",
    "0002": "hbv",
}


# ============================================================
# BASIC IO
# ============================================================

def reset_output_dir():
    if NNUNET_TASK_DIR.exists():
        print(f"Removing old dataset directory: {NNUNET_TASK_DIR}")
        shutil.rmtree(NNUNET_TASK_DIR)

    IMAGES_TR.mkdir(parents=True, exist_ok=True)
    LABELS_TR.mkdir(parents=True, exist_ok=True)
    IMAGES_TS.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    return sitk.ReadImage(str(path))


def write_image(img, path: Path):
    sitk.WriteImage(img, str(path))


# ============================================================
# CASE PARSING
# ============================================================

def get_case_id_from_mha(path: Path):
    """
    Exemplo:
    11404_1001428_t2w.mha -> 11404_1001428
    """
    name = path.name.lower()

    for suffix in ["_t2w.mha", "_adc.mha", "_hbv.mha", "_cor.mha", "_sag.mha"]:
        if name.endswith(suffix):
            return path.name[:-len(suffix)]

    return None


def index_image_cases():
    """
    Retorna:
    {
        case_id: {
            "fold": "picai_public_images_fold0",
            "patient_dir": Path(...),
            "t2w": Path(...),
            "adc": Path(...),
            "hbv": Path(...)
        }
    }
    """

    cases = {}

    fold_dirs = sorted(EXTRACTED_IMAGES_DIR.glob("picai_public_images_fold*"))

    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in {EXTRACTED_IMAGES_DIR}")

    for fold_dir in fold_dirs:
        if not fold_dir.is_dir():
            continue

        print(f"Indexing {fold_dir.name}")

        mha_files = sorted(fold_dir.rglob("*.mha"))

        for f in mha_files:
            case_id = get_case_id_from_mha(f)

            if case_id is None:
                continue

            name = f.name.lower()

            if case_id not in cases:
                cases[case_id] = {
                    "fold": fold_dir.name,
                    "patient_dir": f.parent,
                    "t2w": None,
                    "adc": None,
                    "hbv": None,
                }

            if name.endswith("_t2w.mha"):
                cases[case_id]["t2w"] = f
            elif name.endswith("_adc.mha"):
                cases[case_id]["adc"] = f
            elif name.endswith("_hbv.mha"):
                cases[case_id]["hbv"] = f

    return cases


# ============================================================
# LABEL HANDLING
# ============================================================

def find_label(case_id: str):
    """
    Prioridade:
    1. human_expert/resampled
    2. human_expert/Pooch25

    Se não houver label, o caso é considerado negativo e recebe máscara zero.
    """

    candidates = [
        LABELS_RESAMPLED_DIR / f"{case_id}.nii.gz",
        LABELS_POOCH25_DIR / f"{case_id}.nii.gz",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def binarize_label(label_img):
    arr = sitk.GetArrayFromImage(label_img)
    arr = (arr > 0).astype(np.uint8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(label_img)

    return out


def create_empty_label(reference_img):
    arr = sitk.GetArrayFromImage(reference_img)
    arr = np.zeros_like(arr, dtype=np.uint8)

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(reference_img)

    return out


# ============================================================
# RESAMPLING / ALIGNMENT
# ============================================================

def same_geometry(img, ref):
    return (
        img.GetSize() == ref.GetSize()
        and np.allclose(img.GetSpacing(), ref.GetSpacing())
        and np.allclose(img.GetOrigin(), ref.GetOrigin())
        and np.allclose(img.GetDirection(), ref.GetDirection())
    )


def resample_to_reference(img, reference_img, is_label=False):
    if same_geometry(img, reference_img):
        return img

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)

    return resampler.Execute(img)


def normalize_intensity(img):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)

    mean = arr.mean()
    std = arr.std()

    if std < 1e-8:
        arr = arr * 0.0
    else:
        arr = (arr - mean) / std

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)

    return out


# ============================================================
# NNUNET EXPORT
# ============================================================

def export_case(case_id, case_info):
    required = ["t2w", "adc", "hbv"]

    missing = [m for m in required if case_info[m] is None]

    if missing:
        return {
            "case_id": case_id,
            "status": "skipped_missing_modality",
            "missing": missing,
            "fold": case_info["fold"],
        }

    t2w_img = read_image(case_info["t2w"])
    adc_img = read_image(case_info["adc"])
    hbv_img = read_image(case_info["hbv"])

    # t2w é a referência espacial
    ref_img = t2w_img

    images = {
        "0000": t2w_img,
        "0001": resample_to_reference(adc_img, ref_img, is_label=False),
        "0002": resample_to_reference(hbv_img, ref_img, is_label=False),
    }

    # normaliza cada canal separadamente
    for channel, img in images.items():
        img = normalize_intensity(img)
        out_path = IMAGES_TR / f"{case_id}_{channel}.nii.gz"
        write_image(img, out_path)

    # label
    label_path = find_label(case_id)

    if label_path is not None:
        label_img = read_image(label_path)
        label_img = binarize_label(label_img)
        label_img = resample_to_reference(label_img, ref_img, is_label=True)
        label_source = str(label_path)
        label_type = "positive_or_annotated"
    else:
        label_img = create_empty_label(ref_img)
        label_source = None
        label_type = "empty_negative"

    out_label_path = LABELS_TR / f"{case_id}.nii.gz"
    write_image(label_img, out_label_path)

    # sanity check label geometry
    final_label = read_image(out_label_path)

    if final_label.GetSize() != ref_img.GetSize():
        raise RuntimeError(
            f"Label size mismatch for {case_id}: "
            f"label={final_label.GetSize()} ref={ref_img.GetSize()}"
        )

    return {
        "case_id": case_id,
        "status": "exported",
        "fold": case_info["fold"],
        "label_type": label_type,
        "label_source": label_source,
        "image_files": [
            f"{case_id}_0000.nii.gz",
            f"{case_id}_0001.nii.gz",
            f"{case_id}_0002.nii.gz",
        ],
        "label_file": f"{case_id}.nii.gz",
    }


# ============================================================
# DATASET JSON
# ============================================================

def write_dataset_json(exported_cases):
    training_entries = []

    for case_id in sorted(exported_cases):
        training_entries.append({
            "image": f"./imagesTr/{case_id}.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz",
        })

    dataset = {
        "name": "PI-CAI",
        "description": "PI-CAI prostate MRI dataset converted to nnU-Net v1 format. Channels: T2W, ADC, HBV.",
        "reference": "PI-CAI Challenge / Zenodo 6624726 / DIAGNijmegen picai_labels",
        "licence": "See original PI-CAI dataset and label repositories.",
        "release": "RM-Estendida preprocessing v1",
        "tensorImageSize": "4D",
        "modality": {
            "0": "T2W",
            "1": "ADC",
            "2": "HBV",
        },
        "labels": {
            "0": "background",
            "1": "csPCa_lesion",
        },
        "numTraining": len(training_entries),
        "numTest": 0,
        "training": training_entries,
        "test": [],
    }

    with open(NNUNET_TASK_DIR / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)


# ============================================================
# SPLITS WITHOUT LEAKAGE
# ============================================================

def write_splits(export_records):
    """
    Gera split 5-fold preservando os folds originais.
    Cada fold original vira validação uma vez.
    """

    exported_by_fold = {}

    for rec in export_records:
        if rec["status"] != "exported":
            continue

        fold = rec["fold"]
        case_id = rec["case_id"]

        exported_by_fold.setdefault(fold, []).append(case_id)

    all_case_ids = sorted([
        rec["case_id"]
        for rec in export_records
        if rec["status"] == "exported"
    ])

    splits = []

    for fold in sorted(exported_by_fold.keys()):
        val = sorted(exported_by_fold[fold])
        train = sorted([c for c in all_case_ids if c not in set(val)])

        splits.append({
            "fold": fold,
            "train": train,
            "val": val,
        })

    with open(NNUNET_TASK_DIR / "splits_final.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    with open(QA_DIR / "fold_case_counts.json", "w", encoding="utf-8") as f:
        json.dump(
            {fold: len(cases) for fold, cases in sorted(exported_by_fold.items())},
            f,
            indent=4
        )


# ============================================================
# QA REPORT
# ============================================================

def write_qa_report(export_records):
    exported = [r for r in export_records if r["status"] == "exported"]
    skipped = [r for r in export_records if r["status"] != "exported"]

    positive_or_annotated = [
        r for r in exported
        if r.get("label_type") == "positive_or_annotated"
    ]

    empty_negative = [
        r for r in exported
        if r.get("label_type") == "empty_negative"
    ]

    report = {
        "total_records_seen": len(export_records),
        "exported_cases": len(exported),
        "skipped_cases": len(skipped),
        "cases_with_label_file": len(positive_or_annotated),
        "cases_with_empty_negative_label": len(empty_negative),
        "skipped": skipped[:200],
    }

    with open(QA_DIR / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n==================== QA REPORT ====================")
    print(f"Total records seen:              {report['total_records_seen']}")
    print(f"Exported cases:                  {report['exported_cases']}")
    print(f"Skipped cases:                   {report['skipped_cases']}")
    print(f"Cases with label file:           {report['cases_with_label_file']}")
    print(f"Cases with empty negative label: {report['cases_with_empty_negative_label']}")
    print("===================================================")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=== RM-Estendida | PI-CAI -> nnU-Net Dataset Builder ===")

    reset_output_dir()

    cases = index_image_cases()

    print(f"\nIndexed cases: {len(cases)}")

    export_records = []

    for idx, case_id in enumerate(sorted(cases.keys()), start=1):
        if idx % 50 == 0:
            print(f"Processing {idx}/{len(cases)} cases...")

        rec = export_case(case_id, cases[case_id])
        export_records.append(rec)

    exported_case_ids = sorted([
        r["case_id"]
        for r in export_records
        if r["status"] == "exported"
    ])

    write_dataset_json(exported_case_ids)
    write_splits(export_records)
    write_qa_report(export_records)

    print("\n✅ DONE - nnU-Net dataset created")
    print(f"Dataset path: {NNUNET_TASK_DIR}")


if __name__ == "__main__":
    main()
