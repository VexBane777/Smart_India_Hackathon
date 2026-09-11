"""
ONNX -> TFLite conversion, the final step to produce
voice_guard/assets/models/voice_detector.tflite.

Deliberately NOT run on the dev laptop this was written on — it needs
`tensorflow` + `onnx2tf`, both heavy installs (~1GB+) not worth adding to
a CPU-only box just for this one conversion. Run it wherever train.py ran:

    pip install tensorflow onnx2tf onnx-graphsurgeon sng4onnx
    python export_tflite.py --onnx runs/voice_guard_mlp/model.onnx \
        --out ../assets/models/voice_detector.tflite

Verifies the converted model's input/output shapes match what
lib/services/src/tflite_io.dart expects: input (1, 66) float32,
output (1, 2) float32 raw logits — see TFLiteService.infer's reshape
and softmax, which assumes exactly this shape.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print(
            "tensorflow not installed. This step needs it (and onnx2tf) — "
            "see this file's module docstring for the install command.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    work_dir = args.out.parent / "_onnx2tf_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(args.onnx),
            "-o",
            str(work_dir),
            "-osd",  # output signaturedefs off; keep plain concrete function
        ],
        check=True,
    )

    tflite_candidates = sorted(work_dir.glob("*.tflite"))
    if not tflite_candidates:
        raise SystemExit(f"onnx2tf produced no .tflite file in {work_dir}")
    # Prefer the plain float32 model (not the dynamic-range/int8 variants
    # onnx2tf also emits) for the first drop-in — quantization is a later,
    # separate accuracy/latency tradeoff, not a blocker for "a real model
    # exists at all".
    chosen = next((p for p in tflite_candidates if p.stem.endswith("_float32")), tflite_candidates[0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(chosen.read_bytes())

    _verify_shapes(args.out)
    print(f"Wrote {args.out}")


def _verify_shapes(tflite_path: Path) -> None:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_shape = interpreter.get_input_details()[0]["shape"].tolist()
    out_shape = interpreter.get_output_details()[0]["shape"].tolist()
    assert in_shape[-1] == 66, f"expected input dim 66, got {in_shape}"
    assert out_shape[-1] == 2, f"expected output dim 2 (real/synthetic logits), got {out_shape}"
    print(f"Verified shapes: input={in_shape} output={out_shape}")


if __name__ == "__main__":
    main()
