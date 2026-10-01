"""Download and unzip MovieLens ml-latest-small into data/raw/."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"

if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as r:
        zipfile.ZipFile(io.BytesIO(r.read())).extractall(DEST)
    print("Done:", DEST / "ml-latest-small")
