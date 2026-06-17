import os
import zipfile
from pathlib import Path

RAW_DIR = Path("data/00_raw")
OUT_DIR = Path("data/01_extracted")

OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_zip(zip_path, out_path):
print(f"Extracting {zip_path}")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
zip_ref.extractall(out_path)

def main():
zips = sorted(RAW_DIR.glob("*.zip"))

```
if not zips:
    print("No ZIP files found in data/00_raw")
    return

for z in zips:
    fold_name = z.stem
    out_path = OUT_DIR / fold_name
    out_path.mkdir(parents=True, exist_ok=True)

    extract_zip(z, out_path)

print("\nDone extracting all folds.")
```

if **name** == "**main**":
main()
