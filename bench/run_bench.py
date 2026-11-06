"""Reproduce Lahja's README numbers.

    python3 bench/run_bench.py --dialect <dialect q5_0 .bin> --stock <stock large-v3-turbo q5_0 .bin> \
        --vad <ggml-silero-v5.1.2.bin>

Needs whisper-cli and ffmpeg on PATH. Downloads the 13 held-out dialect clips published on the model
card (oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2, pinned revision) into bench/audio/,
generates 4 non-speech clips (digital silence, pink/brown/white noise) with ffmpeg, and prints:

  1. word error rate per dialect, stock Whisper vs the dialect model (with Lahja's VAD)
  2. what each model writes for audio that contains no speech
  3. decoding speed
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wer import edits, norm  # noqa: E402

REPO = "oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2"
REV = "b12b09bc6601f2116410ccc9716596de0301c24a"
NOISE = {
    "silence": "anullsrc=r=16000:cl=mono",
    "pink-noise": "anoisesrc=r=16000:color=pink:amplitude=0.02",
    "brown-noise": "anoisesrc=r=16000:color=brown:amplitude=0.05",
    "white-hiss": "anoisesrc=r=16000:color=white:amplitude=0.005",
}


def whisper(model: str, wav: Path, vad: str | None) -> tuple[str, float]:
    argv = ["whisper-cli", "-m", model, "-f", str(wav), "-l", "ar", "-nt", "-np"]
    if vad:
        argv += ["--vad", "-vm", vad]
    t = time.time()
    out = subprocess.run(argv, capture_output=True, text=True, check=True).stdout
    return " ".join(out.split()), time.time() - t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", required=True)
    ap.add_argument("--stock", required=True)
    ap.add_argument("--vad", required=True)
    ap.add_argument("--json", type=Path, default=HERE / "out" / "results.json")
    a = ap.parse_args()

    audio = HERE / "audio"
    audio.mkdir(exist_ok=True)
    refs = [line.rstrip("\n").split("\t") for line in open(HERE / "oddadmix-v2-samples.tsv", encoding="utf-8")
            if line.strip()]
    for key, _, _ in refs:
        dst = audio / f"{key}.wav"
        if not dst.exists():
            urllib.request.urlretrieve(f"https://huggingface.co/{REPO}/resolve/{REV}/samples/{key}.wav", dst)
    for name, src in NOISE.items():
        dst = audio / f"{name}.wav"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", src, "-t", "10", str(dst)],
                           check=True)

    systems = {"stock": (a.stock, None), "lahja": (a.dialect, a.vad)}
    res: dict = {"wer": {}, "per_dialect": {}, "non_speech": {}, "seconds": {}}
    audio_seconds = 0.0
    for sysname, (model, vad) in systems.items():
        E = N = 0
        secs = 0.0
        per = {}
        for key, dialect, ref in refs:
            hyp, dt = whisper(model, audio / f"{key}.wav", vad)
            secs += dt
            r, h = norm(ref), norm(hyp)
            e = edits(r, h)
            E, N = E + e, N + len(r)
            per[dialect] = round(e / len(r), 3)
        res["wer"][sysname] = round(E / N, 3)
        res["per_dialect"][sysname] = per
        res["seconds"][sysname] = round(secs, 1)
        res["non_speech"][sysname] = {n: whisper(model, audio / f"{n}.wav", vad)[0] for n in NOISE}
    for key, _, _ in refs:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(audio / f"{key}.wav")], capture_output=True, text=True, check=True).stdout
        audio_seconds += float(out)
    res["audio_seconds"] = round(audio_seconds, 1)

    a.json.parent.mkdir(exist_ok=True)
    a.json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WER (13 dialects, {res['audio_seconds']}s of audio): "
          f"stock {res['wer']['stock']:.1%}  lahja {res['wer']['lahja']:.1%}")
    print("dialect  stock  lahja")
    for _, d, _ in refs:
        print(f"{d:7s} {res['per_dialect']['stock'][d]:6.2f} {res['per_dialect']['lahja'][d]:6.2f}")
    print("non-speech audio:")
    for n in NOISE:
        print(f"  {n:12s} stock: {res['non_speech']['stock'][n] or '(empty)'!s:24s} "
              f"lahja: {res['non_speech']['lahja'][n] or '(empty)'}")
    print(f"decode time for all 13 clips (incl. model load per clip): "
          f"stock {res['seconds']['stock']}s, lahja {res['seconds']['lahja']}s")


if __name__ == "__main__":
    main()
