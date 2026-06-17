import gdown
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1xIbd_l-lU8KSQ3WRTWqvTe6BBhlCTxHN"

OUT_DIR = Path("data/00_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== PI-CAI FOLDER DOWNLOAD START ===")

    gdown.download_folder(
        url=FOLDER_URL,
        output=str(OUT_DIR),
        quiet=False,
        use_cookies=False
    )

    print("\n=== DOWNLOAD COMPLETE ===")
    print(f"Saved in: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
