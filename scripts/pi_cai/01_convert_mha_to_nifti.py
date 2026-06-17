import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path

# =========================
# PATHS
# =========================
INPUT_DIR = Path("data/01_extracted")
OUTPUT_DIR = Path("data/02_preprocessed")

OUTPUT_IMAGES = OUTPUT_DIR / "imagesTr"
OUTPUT_LABELS = OUTPUT_DIR / "labelsTr"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

# =========================
# MODALITIES (nnU-Net INPUT)
# =========================
MODALITIES = ["t2w", "adc", "hbv"]

# =========================
# LOAD MHA
# =========================
def load_mha(path):
    return sitk.ReadImage(str(path))

# =========================
# NORMALIZATION (z-score)
# =========================
def normalize(img_array):
    img_array = img_array.astype(np.float32)
    mean = np.mean(img_array)
    std = np.std(img_array) + 1e-8
    return (img_array - mean) / std

# =========================
# RESAMPLE TO REFERENCE
# =========================
def resample_to_reference(img, reference_img):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)

# =========================
# FIND MODALITIES (ROBUST PI-CAI)
# =========================
def find_modality_files(patient_dir):
    files = {
        "t2w": None,
        "adc": None,
        "hbv": None
    }

    all_files = list(patient_dir.glob("*.mha"))

    for f in all_files:
        name = f.name.lower()

        if "t2w" in name:
            files["t2w"] = f
        elif "adc" in name:
            files["adc"] = f
        elif "hbv" in name:
            files["hbv"] = f

    # validation (critical)
    required = ["t2w", "adc", "hbv"]

    if any(files[m] is None for m in required):
        print(f"⚠️ Skipping {patient_dir.name} missing modalities: {files}")
        return None

    return files

# =========================
# PROCESS PATIENT
# =========================
def process_patient(patient_path, patient_id):

    modality_files = find_modality_files(patient_path)

    if modality_files is None:
        return

    # reference geometry = t2w
    ref_img = load_mha(modality_files["t2w"])

    volumes = []

    for m in MODALITIES:
        img = load_mha(modality_files[m])

        # align geometry
        img = resample_to_reference(img, ref_img)

        arr = sitk.GetArrayFromImage(img)
        arr = normalize(arr)

        volumes.append(arr)

    # stack channels (C, D, H, W)
    stacked = np.stack(volumes, axis=-1)  # 🔥 vector dimension LAST

    sitk_img = sitk.GetImageFromArray(stacked, isVector=True)

    sitk_img.CopyInformation(ref_img)

    out_path = OUTPUT_IMAGES / f"{patient_id}.nii.gz"

    sitk.WriteImage(sitk_img, str(out_path))

    print(f"✔ Processed {patient_id}")

# =========================
# MAIN LOOP
# =========================
def main():

    folds = sorted(INPUT_DIR.glob("picai_public_images_fold*"))

    if not folds:
        print("❌ No folds found")
        return

    for fold in folds:

        patients = [p for p in fold.iterdir() if p.is_dir()]

        print(f"\n📦 Processing {fold.name} ({len(patients)} patients)")

        for patient in patients:
            process_patient(patient, patient.name)

    print("\n✅ DONE - ALL FOLDS PROCESSED")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()