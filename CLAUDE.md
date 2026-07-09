# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. The same file is picked up by the Claude Code VS Code extension chat — keep all repo guidance here rather than in editor-specific files.

## What this repo is

RL racetrack agents built on highway-env, deployed into a C# game engine (RE Engine) via ONNX. Two generations coexist:

- **Active stack** (`racetrack-agents/race_env.py` + the task2/task3 notebooks + `racetrack-agents/re_engine/`): gymnasium + `highway-env>=1.8` + Stable-Baselines3 PPO. This is where all current work happens. Two training notebooks share the same pipeline: `task2_laning_overtaking_sb3_ppo.ipynb` (fast track only) and `task3_generalization_sb3_ppo.ipynb` (random track per episode + per-track Generalization Report) — behavioural changes usually belong in BOTH.
- **Legacy stack** (`racetrack-agents/main.py`, `racetrack_env.py`, `agent/`, `models/`, and the root-level `*.ipynb` / `AIDrive*.cs`): TensorFlow 2.6 / gym 0.21 / highway-env 1.4. `racetrack-agents/requirements.txt` pins THIS legacy stack, not the active one. Don't modernize it; don't take API conventions from it.

## Commands

```bash
# Active-stack dependencies (requirements.txt is legacy — do not use it)
pip install "highway-env>=1.8" "stable-baselines3[extra]>=2.0" gymnasium torch tensorboard moviepy onnx onnxruntime

# Training: run a notebook (task2 = fast track, task3 = multi-track generalization).
# FAST_DEV_RUN (a variable in the config cell) toggles a shortened run vs the real
# one; it now uses 120k+ steps at production hyperparameters, the smallest budget
# that reliably shows LAPS (lap competence emerges at ~50k–100k steps, measured).

# C# observation-port verification (the closest thing to a test suite):
cd racetrack-agents/re_engine/verify && dotnet run -- ..
# expected: PASS ... maxDiff=0.00E+000 for every scenario, then ALL SCENARIOS MATCH
# Do not commit the Debug build artifacts this generates (repo tracks Release ones).

# Regenerate C#-side ground truth after ANY change to track geometry or observation config:
python racetrack-agents/re_engine/export_track_json.py   # rewrites track.json + reference_obs.json

# Convert a channels-first actor ONNX to the engine's channels-last input:
python racetrack-agents/re_engine/convert_onnx_nhwc.py <actor.onnx>

# Legacy stack demo (TF 2.6 env required):
python racetrack-agents/main.py --mode test --agent PPO --load_model ./models/PPO2.model --spawn_vehicles 3
```

There is no linter or Python test suite. Verification is: the `verify/` harness for the C# port, and short scripted env probes (reset/step, reward bounds in [0,1], wall-geometry spot checks) for `race_env.py` changes.

## Architecture

### The training environment (`race_env.py`, class `RacetrackFast`)

Registered by the notebook as `racetrack-v0`. Continuous throttle+steering control (`dynamical: True` bicycle model), OccupancyGrid observation `(10, 12, 12)`.

- **Tracks**: `config["track"]` = `"fast"` (9-segment circuit a…i), `"oval"`, `"stadium"`, `"rect"`, `"chicane"`, or `"random"` (resampled per episode — used for generalization training). Every track has a C# lane preset in `HighwayObservationBuilder` and a `track_<name>.json`; the verify harness checks preset/JSON parity, so adding a track means: builder method in `race_env.py` + C# preset + rerun `export_track_json.py` + verify. Each `_make_road_*` sets `self.segment_sequence` (checkpoint lap order) and `self.spawn_options`. All tracks are 2 lanes × 5 m wide.
- **Walls**: only the corridor's OUTER edges are walls — the painted line between the two lanes is not. `_wall_state()` measures clearance against the first/last boundary lanes of the closest edge and everything wall-related (`_is_at_wall`, `_off_track`, `_apply_wall_behavior` inside the overridden `_simulate`) flows through it. Relies on lanes of an edge being stacked toward +lateral (true for all current tracks). `wall_mode: "stop"` pins the car at the wall; escape is by reversing. Sustained wall contact (35 steps) terminates like a crash.
- **Reward pipeline** (`_reward`): raw shaped terms → checkpoint/lap bonus **added in raw units** → `lmap([-3, 3] → [0, 1])` → clip. The lmap input range must cover the true bounds of the composed reward or bad states saturate at 0 and lose their gradient — the bounds are derived in a comment at the lmap call; **re-derive them whenever any reward weight changes**. Milestone steps clipping to 1.0 is intentional.
- **Wall-escape economics**: within `wall_escape_zone` (1.0 m clearance) reverse/idle penalties are waived and clearance gained is rewarded (`wall_escape_progress`). This exists because penalizing reverse one step after it pulls the car off the wall teaches wall-creeping instead of escaping. Don't re-tighten it to contact-only.
- Speed rewards are zeroed at wall contact (anti wall-riding); `action_penalty` defaults to 0 (redundant with `reverse_penalty`, and it taxed escape commands).

### Training pipeline (task2/task3 notebooks)

Two-phase PPO curriculum: phase 1 trains lane-following with `other_vehicles=0`, phase 2 fine-tunes with NPCs via `PPO.load(..., custom_objects=...)`. Custom `RacetrackCNN` features extractor; the throttle action bias is manually shifted forward after model construction. Best models land in `runs/<EXP_ID>/models/best/phase1|phase2/best_model.zip` (per-phase — there is no top-level `best_model.zip`). PPO can regress late in training; trust `EvalCallback` best models over the last checkpoint. Exports three ONNX files: actor-only (CHW), full policy (CHW), and **actor-only NHWC** — see below.

### C# deployment (`re_engine/`)

A faithful C# port of the OccupancyGrid observation builder, verified bit-identical against Python (`reference_obs.json`). Two interchangeable implementations: `RacetrackObservation.cs` (multi-class, JSON track loading) and `HighwayObservationBuilder.cs` (single file; lanes are constructed by the caller via `Lane.Straight`/`Lane.Arc` factories, `RacetrackFastLanes()` is the built-in preset). Its README documents the observation contract rules that naive ports get wrong — read it before touching anything there.

**Invariants that break silently if violated:**

- **Tensor layout**: the C# builders emit channels-LAST (HWC) flat sequences, index `(ix*12 + iy)*10 + f`. The engine must load `ppo_actor_only_nhwc.onnx` (input `[1,12,12,10]`). Feeding the HWC sequence into the channels-first export degrades the policy to near-constant outputs (throttle ≈0.4, steering ≈0.2) — that symptom means layout mismatch.
- **Heading convention**: counter-clockwise from track +x (the a→b straight). From a forward vector: `Atan2(forward.Z, forward.X)` (preferred — mesh yaw offsets are baked in). From engine Euler yaw: `heading = 90° − yaw`. Feeding raw yaw rotates every observation ~90° ("grid points sideways"). Same conversion applies to NPCs.
- **Constant sync**: `acceleration_range` / `steering_range` / `policy_frequency` / grid config in `race_env.py` are duplicated by design into the C# `ActionDecoder` / builder constants. Changing them requires: update C# constants, re-run `export_track_json.py`, re-run the verify harness, re-export ONNX.
- `on_road` is a lane-centerline waypoint trace, NOT a filled road mask; the observation never uses lane width anywhere. Do not "fix" either.

### Empirical baselines (measured in-session, default config, production hyperparameters)

- Steering competence emerges between 30k–60k steps with the current reward (32 laps at 14.4 m/s by 60k; 37 laps at 16 m/s by 90k). The fast-dev recipe (120k, ent 0.02, lr 2e-4, batch 256, bias +0.40) was validated END-TO-END including the 16k phase-2 fine-tune at lr 3e-4 — no regression. Fast-track-only policies zero-shot transfer to all four simpler tracks at the speed limit.
- `acceleration_range ±2.5` + `steering π/8` is learnable (~22 laps at ~10 m/s by 100k) but caps cruise speed: braking distance at 2.5 m/s² from top speed (51 m) exceeds the 27 m grid preview. Prefer asymmetric `[-5.0, 2.5]` if limiting engine power.
