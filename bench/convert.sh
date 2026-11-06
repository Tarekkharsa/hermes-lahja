#!/usr/bin/env bash
# Reproduce Lahja's default model from the OFFICIAL oddadmix weights and compare its SHA-256 with
# the file `hermes lahja setup` downloads. Everything is pinned:
#   weights   oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2 @ b12b09bc6601f2116410ccc9716596de0301c24a
#   converter ggml-org/whisper.cpp v1.9.4 (7d75b14994ae7f59623e2471445e2355fe506ed2), models/convert-h5-to-ggml.py
#   assets    openai/whisper @ 86098128c0b4f24f0e2aa2994de830614b474227 (mel filters)
#   tokenizer openai/whisper-large-v3-turbo @ 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 (vocab.json,
#             added_tokens.json; checked identical to the fine-tune's tokenizer.json)
#   quantizer whisper-quantize from whisper.cpp 1.9.4 (brew install whisper-cpp)
#
# Usage: bench/convert.sh <work_dir>        (needs ~10 GB free, python3, git, whisper-quantize)
set -euo pipefail

WORK="${1:?usage: convert.sh <work_dir>}"
HF_REPO="oddadmix/whisper-large-v3-turbo-arabic-dialectal-v2"
HF_REV="b12b09bc6601f2116410ccc9716596de0301c24a"
WCPP_REV="7d75b14994ae7f59623e2471445e2355fe506ed2"
OAI_REV="86098128c0b4f24f0e2aa2994de830614b474227"
EXPECTED_Q5="325b7dc691ac04174d91abbd58af74543236a146b87ea2f2f9ecfd9b260a9b6a"

mkdir -p "$WORK"/{hf,src,out}
cd "$WORK"

echo "== 1/5 official weights @ $HF_REV"
for f in config.json generation_config.json model.safetensors processor_config.json tokenizer.json tokenizer_config.json; do
  [ -s "hf/$f" ] || curl -fL --retry 3 -o "hf/$f" "https://huggingface.co/$HF_REPO/resolve/$HF_REV/$f"
done

# The fine-tune ships only tokenizer.json; the converter reads vocab.json + added_tokens.json.
# Take them from the base model it was fine-tuned from, then prove they match the fine-tune's own
# tokenizer.json before using them.
BASE_REPO="openai/whisper-large-v3-turbo"
BASE_REV="41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
for f in vocab.json added_tokens.json; do
  [ -s "hf/$f" ] || curl -fL --retry 3 -o "hf/$f" "https://huggingface.co/$BASE_REPO/resolve/$BASE_REV/$f"
done
python3 -I - hf <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
tok = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
added = json.loads((d / "added_tokens.json").read_text(encoding="utf-8"))
assert tok["model"]["vocab"] == vocab, "vocab.json differs from the fine-tune's tokenizer"
assert {t["content"]: t["id"] for t in tok["added_tokens"]} == added, "added tokens differ"
print("tokenizer check: base vocab + added tokens identical to the fine-tune's tokenizer.json")
PY

echo "== 2/5 converter sources"
fetch() {  # repo rev dir
  [ -d "src/$3/.git" ] || git clone -q --filter=blob:none "https://github.com/$1.git" "src/$3"
  git -C "src/$3" -c advice.detachedHead=false checkout -q "$2"
}
fetch ggml-org/whisper.cpp "$WCPP_REV" whisper.cpp
fetch openai/whisper "$OAI_REV" whisper

echo "== 3/5 python env"
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip -q install "torch>=2.4,<3" "transformers>=4.45,<5" "numpy>=1.26,<3" "safetensors>=0.4,<1"

echo "== 4/5 convert to ggml f16"
venv/bin/python -I src/whisper.cpp/models/convert-h5-to-ggml.py hf src/whisper out
F16=out/ggml-model.bin
shasum -a 256 "$F16"

echo "== 5/5 quantize q5_0"
Q5=out/ggml-whisper-large-v3-turbo-arabic-dialectal-v2-q5_0.bin
whisper-quantize "$F16" "$Q5" q5_0 >/dev/null
GOT=$(shasum -a 256 "$Q5" | cut -d' ' -f1)
echo "q5_0 sha256: $GOT"
if [ "$GOT" = "$EXPECTED_Q5" ]; then
  echo "MATCH: the downloaded model is byte-identical to this conversion of the official weights."
else
  echo "DIFFERENT from the downloaded file ($EXPECTED_Q5)."
  exit 3
fi
