import SimpleITK as sitk
import numpy as np
from pathlib import Path


DATASET = Path("data/nnUNet_raw/nnUNet_raw_data/Task220_PI-CAI")

images_dir = DATASET / "imagesTr"
labels_dir = DATASET / "labelsTr"


def load_case(case_id):
    img = sitk.ReadImage(str(images_dir / f"{case_id}_0000.nii.gz"))
    label = sitk.ReadImage(str(labels_dir / f"{case_id}.nii.gz"))
    return img, label


def check_alignment(img, label):
    checks = {}

    checks["size_match"] = img.GetSize() == label.GetSize()
    checks["spacing_diff"] = np.abs(np.array(img.GetSpacing()) - np.array(label.GetSpacing()))
    checks["origin_diff"] = np.abs(np.array(img.GetOrigin()) - np.array(label.GetOrigin()))
    checks["direction_diff"] = np.abs(np.array(img.GetDirection()) - np.array(label.GetDirection()))

    return checks


def main():

    cases = [p.name.replace("_0000.nii.gz", "") for p in images_dir.glob("*_0000.nii.gz")]

    print(f"Checking {len(cases)} cases...\n")

    issues = 0

    for i, case_id in enumerate(cases[:200]):  # sample first 200

        img, label = load_case(case_id)
        checks = check_alignment(img, label)

        if not checks["size_match"]:
            print(f"[❌ SIZE MISMATCH] {case_id}")
            issues += 1
            continue

        if np.any(checks["spacing_diff"] > 1e-3):
            print(f"[⚠️ SPACING DIFF] {case_id} -> {checks['spacing_diff']}")
            issues += 1

        if np.any(checks["origin_diff"] > 1e-3):
            print(f"[⚠️ ORIGIN DIFF] {case_id} -> {checks['origin_diff']}")
            issues += 1

    print("\n====================")
    print(f"Issues found: {issues}")
    print("====================")


if __name__ == "__main__":
    main()
