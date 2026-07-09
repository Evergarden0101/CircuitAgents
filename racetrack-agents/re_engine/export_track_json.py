# -*- coding: utf-8 -*-
"""
Export the RacetrackFast lane geometry to track.json (for the C# Track
loader) and dump reference observations to reference_obs.json so the C#
OccupancyGridBuilder can be unit-tested against the real environment.

Run:  python export_track_json.py   (from the re_engine folder, circuit env)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from highway_env.road.lane import StraightLane, CircularLane
from highway_env.vehicle.kinematics import Vehicle
from race_env import RacetrackFast

OUT_DIR = Path(__file__).resolve().parent


def export_track(env) -> dict:
    lanes = []
    net = env.road.network
    for _from in net.graph:
        for _to in net.graph[_from]:
            for lane in net.graph[_from][_to]:
                if isinstance(lane, StraightLane):
                    lanes.append({
                        "type": "straight",
                        "start": [float(lane.start[0]), float(lane.start[1])],
                        "end": [float(lane.end[0]), float(lane.end[1])],
                        "width": float(lane.width),
                    })
                elif isinstance(lane, CircularLane):
                    lanes.append({
                        "type": "circular",
                        "center": [float(lane.center[0]), float(lane.center[1])],
                        "radius": float(lane.radius),
                        "start_phase": float(lane.start_phase),
                        "end_phase": float(lane.end_phase),
                        "clockwise": bool(lane.direction == 1),
                        "width": float(lane.width),
                    })
                else:
                    raise TypeError(f"unsupported lane type {type(lane)}")
    return {"lanes": lanes}


def vehicle_record(v) -> dict:
    return {
        "x": float(v.position[0]),
        "y": float(v.position[1]),
        "vx": float(v.velocity[0]),
        "vy": float(v.velocity[1]),
        "heading": float(v.heading),
    }


def place(vehicle, lane, s, lat, speed, heading_offset=0.0):
    vehicle.position = np.array(lane.position(s, lat), dtype=float)
    vehicle.heading = float(lane.heading_at(s) + heading_offset)
    vehicle.speed = float(speed)
    if hasattr(vehicle, "lateral_velocity"):
        vehicle.lateral_velocity = 0.0
    vehicle.on_state_update()   # refresh vehicle.lane -> defines lat/ang_off


def make_scenario(env, name, ego_spec, npc_specs) -> dict:
    env.reset(seed=0)
    net = env.road.network
    ego = env.vehicle
    env.road.vehicles.clear()
    env.road.vehicles.append(ego)

    lane_key, s, lat, speed, hoff = ego_spec
    place(ego, net.get_lane(lane_key), s, lat, speed, hoff)

    npcs = []
    for lane_key, s, lat, speed, hoff in npc_specs:
        lane = net.get_lane(lane_key)
        npc = Vehicle(env.road, lane.position(s, lat),
                      heading=lane.heading_at(s) + hoff, speed=speed)
        env.road.vehicles.append(npc)
        npcs.append(npc)

    obs = env.observation_type.observe()
    assert obs.shape == (10, 12, 12), obs.shape
    return {
        "name": name,
        "ego": vehicle_record(ego),
        "npcs": [vehicle_record(n) for n in npcs],
        "obs_shape": list(obs.shape),
        "obs": [float(x) for x in obs.astype(np.float32).ravel()],
    }


TRACKS = ("fast", "oval", "stadium", "rect", "chicane")


def main():
    # Per-track lane geometry for the C# side: track.json stays the "fast"
    # track (backward compatible), track_<name>.json covers every variant.
    for name in TRACKS:
        tenv = RacetrackFast(config={"other_vehicles": 0, "duration": 10_000,
                                     "track": name})
        tenv.reset(seed=0)
        tdata = export_track(tenv)
        (OUT_DIR / f"track_{name}.json").write_text(
            json.dumps(tdata, indent=1), encoding="utf-8")
        print(f"track_{name}.json: {len(tdata['lanes'])} lanes")

    env = RacetrackFast(config={"other_vehicles": 0, "duration": 10_000})
    env.reset(seed=0)

    track = export_track(env)
    (OUT_DIR / "track.json").write_text(json.dumps(track, indent=1), encoding="utf-8")
    print(f"track.json: {len(track['lanes'])} lanes")

    scenarios = [
        make_scenario(env, "straight_ab_alone",
                      (("a", "b", 0), 30.0, 0.0, 8.0, 0.0), []),
        make_scenario(env, "arc_bc_with_two_npcs",
                      (("b", "c", 1), 10.0, 0.5, 6.0, 0.15),
                      [(("b", "c", 0), 22.0, -0.3, 3.0, 0.0),
                       (("c", "d", 0), 2.0, 0.0, 5.0, -0.1)]),
        make_scenario(env, "fg_overtake_and_far_npc",
                      (("f", "g", 0), 12.0, 0.4, 10.0, -0.05),
                      [(("f", "g", 1), 17.0, 0.0, 4.0, 0.0),
                       (("a", "b", 0), 10.0, 0.0, 6.0, 0.0)]),   # far -> off-grid
        make_scenario(env, "hairpin_de",
                      (("d", "e", 0), 20.0, -0.6, 7.0, 0.2),
                      [(("d", "e", 1), 30.0, 0.0, 3.0, 0.0)]),
    ]
    (OUT_DIR / "reference_obs.json").write_text(
        json.dumps({"scenarios": scenarios}), encoding="utf-8")
    print(f"reference_obs.json: {len(scenarios)} scenarios")


if __name__ == "__main__":
    main()
