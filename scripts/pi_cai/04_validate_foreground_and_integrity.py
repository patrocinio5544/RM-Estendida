from pathlib import Path
import json
import SimpleITK as sitk
import numpy as np
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

TASK_DIR = Path("/workspace/data/nnUNet_raw/nnUNet_raw_data/Task220_PI-CAI")

IMAGES_TR = TASK_DIR / "imagesTr"
LABELS_TR = TASK_DIR / "labelsTr"
SPLITS_FILE = TASK_DIR / "splits_final.json"
QA_DIR = TASK_DIR / "_qa"
QA_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = QA_DIR / "foreground_integrity_report.json"
REPORT_TXT = QA_DIR / "foreground_integrity_report.txt"

MODALITIES = {
    "0000": "T2W",
    "0001": "ADC",
    "0002": "HBV",
}

GEOMETRY_TOL = 1e-4


# ============================================================
# HELPERS
# ============================================================

def load_nifti(path: Path):
    return sitk.ReadImage(str(path))


def np_unique(path: Path):
    img = load_nifti(path)
    arr = sitk.GetArrayFromImage(img)
    return np.unique(arr), arr


def geometry_diff(img_a, img_b):
    spacing_a = np.array(img_a.GetSpacing())
    spacing_b = np.array(img_b.GetSpacing())

    origin_a = np.array(img_a.GetOrigin())
    origin_b = np.array(img_b.GetOrigin())

    direction_a = np.array(img_a.GetDirection())
    direction_b = np.array(img_b.GetDirection())

    return {
        "same_size": img_a.GetSize() == img_b.GetSize(),
        "spacing_max_diff": float(np.max(np.abs(spacing_a - spacing_b))),
        "origin_max_diff": float(np.max(np.abs(origin_a - origin_b))),
        "direction_max_diff": float(np.max(np.abs(direction_a - direction_b))),
    }


def geometry_ok(diff):
    return (
        diff["same_size"]
        and diff["spacing_max_diff"] <= GEOMETRY_TOL
        and diff["origin_max_diff"] <= GEOMETRY_TOL
        and diff["direction_max_diff"] <= GEOMETRY_TOL
    )


def get_case_ids_from_labels():
    return sorted([
        p.name.replace(".nii.gz", "")
        for p in LABELS_TR.glob("*.nii.gz")
    ])


def load_splits():
    if not SPLITS_FILE.exists():
        return None

    with open(SPLITS_FILE, "r") as f:
        return json.load(f)


# ============================================================
# MAIN VALIDATION
# ============================================================

def main():
    print("\n================ VALIDATING PI-CAI TASK ================\n")
    print(f"TASK_DIR: {TASK_DIR}")

    case_ids = get_case_ids_from_labels()
    splits = load_splits()

    report = {
        "task_dir": str(TASK_DIR),
        "total_cases_from_labels": len(case_ids),
        "modalities": MODALITIES,
        "counts": {
            "positive_cases": 0,
            "negative_cases": 0,
            "missing_labels": 0,
            "missing_modalities": 0,
            "invalid_label_values": 0,
            "geometry_errors": 0,
            "read_errors": 0,
        },
        "positive_cases": [],
        "negative_cases": [],
        "missing_labels": [],
        "missing_modalities": [],
        "invalid_label_values": [],
        "geometry_errors": [],
        "read_errors": [],
        "split_summary": {},
    }

    for idx, case_id in enumerate(case_ids, start=1):
        if idx % 50 == 0 or idx == 1:
            print(f"Checking {idx}/{len(case_ids)}: {case_id}")

        label_path = LABELS_TR / f"{case_id}.nii.gz"

        if not label_path.exists():
            report["counts"]["missing_labels"] += 1
            report["missing_labels"].append(case_id)
            continue

        try:
            label_img = load_nifti(label_path)
            label_values, label_arr = np_unique(label_path)
        except Exception as e:
            report["counts"]["read_errors"] += 1
            report["read_errors"].append({
                "case_id": case_id,
                "file": str(label_path),
                "error": str(e),
            })
            continue

        label_values_list = [int(v) for v in label_values.tolist()]

        # Validate binary labels
        if not set(label_values_list).issubset({0, 1}):
            report["counts"]["invalid_label_values"] += 1
            report["invalid_label_values"].append({
                "case_id": case_id,
                "values": label_values_list,
            })

        # Foreground check
        foreground_voxels = int(np.sum(label_arr > 0))

        if foreground_voxels > 0:
            report["counts"]["positive_cases"] += 1
            report["positive_cases"].append({
                "case_id": case_id,
                "foreground_voxels": foreground_voxels,
            })
        else:
            report["counts"]["negative_cases"] += 1
            report["negative_cases"].append(case_id)

        # Check all modalities
        missing = []
        modality_imgs = {}

        for suffix, modality_name in MODALITIES.items():
            img_path = IMAGES_TR / f"{case_id}_{suffix}.nii.gz"

            if not img_path.exists():
                missing.append(f"{suffix}:{modality_name}")
                continue

            try:
                modality_imgs[suffix] = load_nifti(img_path)
            except Exception as e:
                report["counts"]["read_errors"] += 1
                report["read_errors"].append({
                    "case_id": case_id,
                    "file": str(img_path),
                    "error": str(e),
                })

        if missing:
            report["counts"]["missing_modalities"] += 1
            report["missing_modalities"].append({
                "case_id": case_id,
                "missing": missing,
            })
            continue

        # Geometry check: label vs T2W reference
        t2w_img = modality_imgs["0000"]
        diff = geometry_diff(t2w_img, label_img)

        if not geometry_ok(diff):
            report["counts"]["geometry_errors"] += 1
            report["geometry_errors"].append({
                "case_id": case_id,
                "reference": "T2W_vs_LABEL",
                "diff": diff,
            })

        # Geometry check: ADC/HBV vs T2W
        for suffix in ["0001", "0002"]:
            diff_mod = geometry_diff(t2w_img, modality_imgs[suffix])

            if not geometry_ok(diff_mod):
                report["counts"]["geometry_errors"] += 1
                report["geometry_errors"].append({
                    "case_id": case_id,
                    "reference": f"T2W_vs_{MODALITIES[suffix]}",
                    "diff": diff_mod,
                })

    # ========================================================
    # SPLIT VALIDATION
    # ========================================================

    if splits is not None:
        for fold_idx, fold in enumerate(splits):
            train_cases = set(fold.get("train", []))
            val_cases = set(fold.get("val", []))

            overlap = sorted(train_cases.intersection(val_cases))

            val_positive = 0
            val_negative = 0

            positive_ids = set([x["case_id"] for x in report["positive_cases"]])
            negative_ids = set(report["negative_cases"])

            for case_id in val_cases:
                if case_id in positive_ids:
                    val_positive += 1
                elif case_id in negative_ids:
                    val_negative += 1

            report["split_summary"][f"fold_{fold_idx}"] = {
                "train_count": len(train_cases),
                "val_count": len(val_cases),
                "overlap_count": len(overlap),
                "overlap_cases": overlap[:20],
                "val_positive_cases": val_positive,
                "val_negative_cases": val_negative,
            }

    # ========================================================
    # SAVE REPORTS
    # ========================================================

    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=4)

    lines = []
    lines.append("================ PI-CAI FOREGROUND + INTEGRITY REPORT ================")
    lines.append(f"TASK_DIR: {TASK_DIR}")
    lines.append("")
    lines.append(f"Total cases from labels:      {report['total_cases_from_labels']}")
    lines.append(f"Positive cases:               {report['counts']['positive_cases']}")
    lines.append(f"Negative/background-only:     {report['counts']['negative_cases']}")
    lines.append(f"Missing labels:               {report['counts']['missing_labels']}")
    lines.append(f"Missing modalities:           {report['counts']['missing_modalities']}")
    lines.append(f"Invalid label values:         {report['counts']['invalid_label_values']}")
    lines.append(f"Geometry errors:              {report['counts']['geometry_errors']}")
    lines.append(f"Read errors:                  {report['counts']['read_errors']}")
    lines.append("")

    if report["split_summary"]:
        lines.append("================ SPLITS ================")
        for fold_name, fold_data in report["split_summary"].items():
            lines.append(
                f"{fold_name}: "
                f"train={fold_data['train_count']} | "
                f"val={fold_data['val_count']} | "
                f"val_pos={fold_data['val_positive_cases']} | "
                f"val_neg={fold_data['val_negative_cases']} | "
                f"overlap={fold_data['overlap_count']}"
            )
        lines.append("")

    lines.append("================ FIRST NEGATIVE CASES ================")
    for case_id in report["negative_cases"][:50]:
        lines.append(case_id)

    lines.append("")
    lines.append("================ FIRST POSITIVE CASES ================")
    for item in report["positive_cases"][:50]:
        lines.append(f"{item['case_id']} | foreground_voxels={item['foreground_voxels']}")

    lines.append("")
    lines.append(f"JSON report saved to: {REPORT_JSON}")
    lines.append("======================================================================")

    REPORT_TXT.write_text("\n".join(lines))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
