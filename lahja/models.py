"""Model catalog and one-time, checksum-verified download.

Nothing is downloaded on import or during transcription: the user runs ``hermes lahja setup``
once. Every URL names an exact Hugging Face repository commit (``resolve/<sha>``), every file has a
SHA-256 and a size cap; a mismatch deletes the partial file and fails. huggingface.co redirects the
download to its CDN (``*.cdn.hf.co`` / ``*.hf.co``). Files land in this plugin's profile-scoped data
dir, outside the install dir, so ``hermes plugins update`` never re-downloads them.

The dialect model comes from a community ggml mirror (JoaoZaokk/…-ggml) of oddadmix's official
weights; ``bench/convert.sh`` reproduces it byte for byte from the official repository.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


@dataclass(frozen=True)
class ModelFile:
    id: str
    display: str
    filename: str
    url: str
    sha256: str
    size_mb: int
    license: str
    source: str


MODELS: dict[str, ModelFile] = {
    m.id: m for m in (
        ModelFile(
            id="turbo-dialectal-v2-q5",
            display="Whisper large-v3-turbo, Arabic dialectal v2 (q5_0, 574 MB) — recommended",
            filename="ggml-whisper-large-v3-turbo-arabic-dialectal-v2-q5_0.bin",
            url=("https://huggingface.co/JoaoZaokk/whisper-large-v3-turbo-arabic-dialectal-v2-ggml/"
                 "resolve/da87605d9107bd2145166ede51fe3d7a24b1beb0/"
                 "ggml-whisper-large-v3-turbo-arabic-dialectal-v2-q5_0.bin"),
            sha256="325b7dc691ac04174d91abbd58af74543236a146b87ea2f2f9ecfd9b260a9b6a",
            size_mb=574,
            license="Apache-2.0",
            source=("community ggml mirror (JoaoZaokk) of oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2; "
                    "byte-identical to bench/convert.sh"),
        ),
        ModelFile(
            id="turbo-stock-q5",
            display="Whisper large-v3-turbo, stock OpenAI weights (q5_0, 574 MB) — for comparison",
            filename="ggml-large-v3-turbo-q5_0.bin",
            url=("https://huggingface.co/ggerganov/whisper.cpp/resolve/"
                 "5359861c739e955e79d9a303bcbc70fb988958b1/ggml-large-v3-turbo-q5_0.bin"),
            sha256="394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
            size_mb=574,
            license="MIT",
            source="ggerganov/whisper.cpp",
        ),
    )
}

VAD_MODEL = ModelFile(
    id="silero-v5.1.2",
    display="Silero VAD v5.1.2 (ggml, 0.9 MB)",
    filename="ggml-silero-v5.1.2.bin",
    url=("https://huggingface.co/ggml-org/whisper-vad/resolve/"
         "9ffd54a1e1ee413ddf265af9913beaf518d1639b/ggml-silero-v5.1.2.bin"),
    sha256="29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    size_mb=1,
    license="MIT",
    source="ggml-org/whisper-vad",
)

DEFAULT_MODEL = "turbo-dialectal-v2-q5"


def models_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(data_dir: Path, m: ModelFile) -> Path:
    """Where ``m`` lives (no side effects: never creates directories)."""
    return Path(data_dir) / "models" / m.filename


def is_installed(data_dir: Path, m: ModelFile) -> bool:
    p = path_for(data_dir, m)
    return p.is_file() and p.stat().st_size > 0


def _remove_stale_partials(d: Path) -> None:
    """A killed download leaves ``.download-*`` behind (up to 574 MB); clear them on the next run."""
    for p in d.glob(".download-*"):
        try:
            p.unlink()
        except OSError:
            pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(data_dir: Path, m: ModelFile, progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Download ``m`` (https only), verify SHA-256, then atomically move it into place."""
    if not m.url.startswith("https://"):
        raise ValueError(f"refusing non-https model URL: {m.url}")
    d = models_dir(data_dir)
    dest = d / m.filename
    if dest.is_file() and sha256_of(dest) == m.sha256:
        return dest
    _remove_stale_partials(d)
    cap = int(m.size_mb * 1024 * 1024 * 1.10) + (1 << 20)
    fd, tmp = tempfile.mkstemp(prefix=".download-", dir=str(d))
    try:
        req = urllib.request.Request(m.url, headers={"User-Agent": "hermes-lahja"})
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            h = hashlib.sha256()
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if done > cap:
                    raise ValueError(f"{m.filename}: download exceeds its expected size ({m.size_mb} MB)")
                if progress:
                    progress(done, total)
        if h.hexdigest() != m.sha256:
            raise ValueError(f"checksum mismatch for {m.filename}: got {h.hexdigest()}, expected {m.sha256}")
        os.replace(tmp, dest)
        return dest
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
