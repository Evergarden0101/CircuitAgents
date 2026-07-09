"""Convert an exported actor ONNX from channels-first to channels-last input.

The notebook's torch.onnx export produces input "obs" [batch, 10, 12, 12]
(NCHW), but the engine-side observation builders (RacetrackObservation.cs /
HighwayObservationBuilder.cs) emit a flat H*W*C sequence. Feeding that
sequence into the NCHW input scrambles every channel and the policy
degenerates to near-constant outputs around its bias (throttle ~0.4,
steering ~0.1) — it cannot corner.

This script rewrites the graph so the input is [batch, 12, 12, 10] (NHWC)
and a Transpose(0,3,1,2) node restores the layout the network was trained
on. The C# HWC buffer then feeds the model directly.

Usage:
    python convert_onnx_nhwc.py <in.onnx> [out.onnx]
    (default out: <in>_nhwc.onnx)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper


def convert(src: Path, dst: Path) -> None:
    model = onnx.load(str(src))
    graph = model.graph
    inp = graph.input[0]
    name = inp.name

    dims = inp.type.tensor_type.shape.dim
    if len(dims) != 4:
        raise SystemExit(f"expected 4-D input, got {len(dims)}-D")
    c, h, w = dims[1].dim_value, dims[2].dim_value, dims[3].dim_value

    # Reroute every consumer of the input through a transpose node
    internal = name + "_nchw"
    for node in graph.node:
        for i, n in enumerate(node.input):
            if n == name:
                node.input[i] = internal
    graph.node.insert(
        0, helper.make_node("Transpose", [name], [internal], perm=[0, 3, 1, 2])
    )

    # Input declaration becomes channels-last
    dims[1].dim_value, dims[2].dim_value, dims[3].dim_value = h, w, c

    onnx.checker.check_model(model)
    onnx.save(model, str(dst))

    # Equivalence check: NHWC model on transposed input == original on NCHW
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    nchw = rng.standard_normal((1, c, h, w)).astype(np.float32)
    nhwc = nchw.transpose(0, 2, 3, 1).copy()
    a = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(
        None, {name: nchw}
    )[0]
    b = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(
        None, {name: nhwc}
    )[0]
    err = float(np.abs(a - b).max())
    print(f"{dst}  input [{h},{w},{c}] NHWC  max|diff| vs original = {err:.2e}")
    if err > 1e-6:
        raise SystemExit("outputs differ — conversion is wrong")


if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_nhwc.onnx")
    convert(src, dst)
