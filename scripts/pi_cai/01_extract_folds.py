import zipfile
from pathlib import Path

RAW_DIR = Path("data/00_raw")
OUT_DIR = Path("data/01_extracted")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract(zip_path: Path, out_dir: Path):
    folder_name = zip_path.stem
    target_dir = out_dir / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    print(f"✔ Extracted to {target_dir}")

def main():
    zips = sorted(RAW_DIR.glob("*.zip"))

    if not zips:
        print("❌ No ZIP files found")
        return

    print(f"Found {len(zips)} folds")

    for z in zips:
        extract(z, OUT_DIR)

    print("\n✅ ALL FOLDS EXTRACTED")

if __name__ == "__main__":
    main()
