"""Export a trained SB3 PPO checkpoint to ONNX.

Extracted from the task2/task3 notebooks so exports can run from the
command line against any checkpoint:

    python export_onnx.py runs/<EXP>/models/best/phase2/best_model.zip
    python export_onnx.py model.zip -m 3 -o ppo_actor_nhwc.onnx
    python export_onnx.py model.zip -m 0          # all three variants

Modes:
    1  full policy  (obs -> action_mean + value), channels-first [1,10,12,12]
    2  actor only   (obs -> action_mean),         channels-first [1,10,12,12]
    3  actor only NHWC (obs -> action_mean), channels-LAST [1,12,12,10] —
       the layout the C# builders emit; deploy THIS one in the engine.
    0  export all three (output name is used as a base, suffixes added)

Every export is verified: onnx.checker plus an onnxruntime run on random
observations compared against the torch policy (max|diff| must be ~1e-6).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RacetrackCNN(torch.nn.Module):
    """Must match the notebook's features extractor exactly; registered
    into __main__ below so SB3 can unpickle checkpoints trained there."""


def _make_cnn_class():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class RacetrackCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim=512):
            super().__init__(observation_space, features_dim)
            n_channels = observation_space.shape[0]
            self.cnn = nn.Sequential(
                nn.Conv2d(n_channels, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256 * 5 * 5, features_dim), nn.ReLU(),
            )

        def forward(self, obs):
            return self.cnn(obs.float())

    return RacetrackCNN


class ActorOnlyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.features_extractor(obs.float())
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)


class ActorOnlyNHWCWrapper(ActorOnlyWrapper):
    def forward(self, obs):
        return super().forward(obs.permute(0, 3, 1, 2))   # NHWC -> NCHW


class FullPolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        features = self.policy.features_extractor(obs.float())
        latent_pi, latent_vf = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi), self.policy.value_net(latent_vf)


MODES = {
    1: ("full policy (CHW)", FullPolicyWrapper, "chw", ["action_mean", "value"], "_full"),
    2: ("actor only (CHW)", ActorOnlyWrapper, "chw", ["action_mean"], "_actor"),
    3: ("actor only (NHWC, engine layout)", ActorOnlyNHWCWrapper, "nhwc", ["action_mean"], "_actor_nhwc"),
}


def export_one(policy, obs_shape, mode, out_path):
    import onnx
    import onnxruntime as ort

    label, wrapper_cls, layout, output_names, _ = MODES[mode]
    c, h, w = obs_shape
    shape = (1, c, h, w) if layout == "chw" else (1, h, w, c)
    dummy = torch.zeros(*shape, dtype=torch.float32)
    wrapper = wrapper_cls(policy).eval()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(out_path),
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["obs"], output_names=output_names,
            dynamic_axes={"obs": {0: "batch_size"},
                          **{n: {0: "batch_size"} for n in output_names}},
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(out_path)))

    # verify against the torch policy on random observations
    rng = np.random.default_rng(0)
    test = rng.uniform(-1, 1, size=(4, *shape[1:])).astype(np.float32)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"obs": test})
    with torch.no_grad():
        ref = wrapper(torch.as_tensor(test))
    refs = [t.numpy() for t in (ref if isinstance(ref, tuple) else (ref,))]
    err = max(float(np.abs(a - b).max()) for a, b in zip(ort_out, refs))
    print(f"  mode {mode} {label}: {out_path}  input {list(shape)}  "
          f"max|onnx-torch| = {err:.2e}")
    if err > 1e-5:
        raise SystemExit("export mismatch — aborting")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="SB3 PPO checkpoint (.zip)")
    ap.add_argument("-o", "--output", default=None,
                    help="output .onnx path (mode 0: used as base name)")
    ap.add_argument("-m", "--mode", type=int, default=3, choices=[0, 1, 2, 3],
                    help="1=full(CHW) 2=actor(CHW) 3=actor NHWC (default) 0=all")
    args = ap.parse_args()

    from stable_baselines3 import PPO

    # notebooks define RacetrackCNN in __main__; make unpickling find it here
    cnn = _make_cnn_class()
    sys.modules["__main__"].RacetrackCNN = cnn
    globals()["RacetrackCNN"] = cnn

    model_path = Path(args.model)
    model = PPO.load(str(model_path), device="cpu")
    policy = model.policy.eval().to("cpu")
    obs_shape = model.observation_space.shape   # (C, H, W)
    print(f"loaded {model_path} — {model.num_timesteps} steps, obs {obs_shape}")

    base = Path(args.output) if args.output else model_path.with_suffix("")
    modes = [1, 2, 3] if args.mode == 0 else [args.mode]
    for m in modes:
        if args.mode == 0 or not args.output:
            out = base.with_name(base.stem + MODES[m][4] + ".onnx")
        else:
            out = base if base.suffix == ".onnx" else base.with_suffix(".onnx")
        export_one(policy, obs_shape, m, out)


if __name__ == "__main__":
    main()
