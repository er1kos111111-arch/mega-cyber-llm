#!/usr/bin/env bash
# run_tpu.sh — bootstrap and launch MC-LLM training on a Google TPU VM.
#
# Run this ON the TPU VM (Linux). It:
#   1. installs JAX with TPU support if missing;
#   2. verifies the TPU is visible;
#   3. runs a quick JAX smoke test, then real training.
#
# Usage:
#   bash scripts/run_tpu.sh smoke              # verify TPU works
#   bash scripts/run_tpu.sh train 100m         # train configs/100m.yaml
#   bash scripts/run_tpu.sh prepare            # tokenizer + shards (on VM)
set -euo pipefail

MODE="${1:-smoke}"
CONFIG_NAME="${2:-100m}"

echo "==> Python/PIP"
python3 --version
pip --version

# ---------------------------------------------------------------------------
# 1. install JAX for TPU
# ---------------------------------------------------------------------------
if ! python3 -c "import jax; print(jax.__version__)" >/dev/null 2>&1; then
  echo "==> Installing JAX[TPU]"
  pip install --upgrade "jax[tpu]" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

echo "==> Installing remaining deps"
pip install -q pyyaml numpy

# ---------------------------------------------------------------------------
# 2. verify TPU is visible
# ---------------------------------------------------------------------------
echo "==> Detecting TPU devices"
python3 - <<'PY'
import jax
devices = jax.devices()
print(f"jax: {jax.__version__}")
print(f"devices: {len(devices)}")
for d in devices:
    print("  -", d)
if not devices:
    raise SystemExit("ERROR: no TPU devices found — is the TPU VM configured correctly?")
PY

# ---------------------------------------------------------------------------
# 3. prepare data + tokenizer (on the VM, using the CPU cores)
# ---------------------------------------------------------------------------
if [ "$MODE" = "prepare" ]; then
  echo "==> Training tokenizer"
  python3 scripts/train_tokenizer.py --input data/raw --vocab-size 32768 --out tokenizer
  echo "==> Preparing shards"
  python3 scripts/prepare_data.py --input data/raw --tokenizer tokenizer \
      --out data/shards --vocab-size 32768
  exit 0
fi

# ---------------------------------------------------------------------------
# 4. run
# ---------------------------------------------------------------------------
if [ "$MODE" = "smoke" ]; then
  echo "==> Running JAX smoke test (random tokens)"
  python3 -m training.tpu_backend --config configs/100m.yaml --smoke --steps 20 --batch 8
else
  echo "==> Training configs/${CONFIG_NAME}.yaml"
  python3 -m training.tpu_backend --config "configs/${CONFIG_NAME}.yaml" \
      --data-dir data/shards --steps 2000 --batch 8 --out checkpoints_tpu
fi

echo "==> Done"
