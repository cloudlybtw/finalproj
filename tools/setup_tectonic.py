"""Download a pinned official Tectonic binary without a system installation."""
import hashlib
import io
from pathlib import Path
import platform
import tarfile
import urllib.request

VERSION = "0.17.0"
ASSETS = {
    ("Darwin", "arm64"): ("aarch64-apple-darwin", "a3f1cac7c5678f01661a92212f58480ae3b0634115d880dbc59e2953ded45667"),
    ("Darwin", "x86_64"): ("x86_64-apple-darwin", "7c90ef5b6ddb1eb1937e4337add5237b79338e4b9676459fa91187d24d6cdf80"),
    ("Linux", "x86_64"): ("x86_64-unknown-linux-musl", "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"),
    ("Linux", "aarch64"): ("aarch64-unknown-linux-musl", "b10954a95404f3ab2328d2fa59a5ebab8e657f893fab096f98be8db7c0c979b8"),
}
key = (platform.system(), platform.machine())
if key not in ASSETS:
    raise SystemExit("Install Tectonic 0.17.0 for this platform from https://github.com/tectonic-typesetting/tectonic/releases/tag/tectonic%400.17.0")
target, digest = ASSETS[key]
name = f"tectonic-{VERSION}-{target}.tar.gz"
url = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{VERSION}/{name}"
print(f"Downloading {url}")
request = urllib.request.Request(url, headers={"User-Agent": "ROBT407-reproducible-report"})
with urllib.request.urlopen(request) as response:
    data = response.read()
if hashlib.sha256(data).hexdigest() != digest:
    raise SystemExit("Release SHA-256 mismatch; refusing to unpack.")
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    member = next(m for m in archive.getmembers() if Path(m.name).name == "tectonic" and m.isfile())
    payload = archive.extractfile(member).read()
path = Path(__file__).resolve().parent / "tectonic"
path.write_bytes(payload)
path.chmod(0o755)
print(f"Installed {path}; run {path} --version")
