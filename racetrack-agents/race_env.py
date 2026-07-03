from __future__ import annotations

import numpy as np
import numpy.random as random
from copy import deepcopy

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, CircularLane, LineType
from highway_env.vehicle.behavior import IDMVehicle

NEXT_ROAD = {
    ("a", "b"): ("b", "c"),
    ("b", "c"): ("c", "d"),
    ("c", "d"): ("d", "e"),
    ("d", "e"): ("e", "f"),
    ("e", "f"): ("f", "g"),
    ("f", "g"): ("g", "h"),
    ("g", "h"): ("h", "i"),
    ("h", "i"): ("i", "a"),
    ("i", "a"): ("a", "b"),
}


class RacetrackFast(AbstractEnv):
    """A full racetrack environment tuned for throttle + steering control

    This is a self-contained environment similar in style to
    `racetrack_env.RaceTrackEnv` but simplified and focused on the
    throttle/steering action space and a speed-focused reward term.
    """

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": [
                    "presence",
                    "on_road",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "cos_h",
                    "sin_h",
                    "long_off",
                    "lat_off",
                    "ang_off",
                ],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                    "long_off": [0, 500],
                    "lat_off": [-4, 4],
                    "ang_off": [-3.14159, 3.14159],
                },
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-3.0, 5.0],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "dynamical": False,
            },
            # Simulation
            "duration": 300,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            # Visuals
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            # Vehicles
            "controlled_vehicles": 1,
            "other_vehicles": 3,
            "speed_limit": 12.0,
            "terminate_off_road": True,
            "length": 100,
            "no_lanes": 3,
            # Reward weights (tunable)
            "collision_reward": -5.0,
            "lane_centering_cost": 4.0,
            "action_penalty": 0.1,
            "speed_reward": 0.6,
            "steering_penalty": 0.15,
            # Misc
            "show_trajectories": False,
        })
        return deepcopy(config)

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        super().__init__(config)
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0

    def _reset(self) -> None:
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # A compact oval made of straights and arcs (inspired by racetrack_env)
        speedlimits = [None, 12, 12, 12, 12, 12, 12, 12, 12]

        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c", CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[2]))
        net.add_lane("b", "c", CircularLane(center1, radii1 + 5, np.deg2rad(90), np.deg2rad(-1), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[2]))

        net.add_lane("c", "d", StraightLane([120, -19], [120, -30], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -19], [125, -30], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[3]))

        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e", CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[4]))
        net.add_lane("d", "e", CircularLane(center2, radii2 + 5, np.deg2rad(0), np.deg2rad(-181), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[4]))

        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f", CircularLane(center3, radii3 + 5, np.deg2rad(0), np.deg2rad(136), width=5, clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED), speed_limit=speedlimits[5]))
        net.add_lane("e", "f", CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5, clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS), speed_limit=speedlimits[5]))

        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[6]))

        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h", CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[7]))
        net.add_lane("g", "h", CircularLane(center4, radii4 + 5, np.deg2rad(315), np.deg2rad(165), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[7]))
        net.add_lane("h", "i", CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=speedlimits[7]))
        net.add_lane("h", "i", CircularLane(center4, radii4 + 5, np.deg2rad(170), np.deg2rad(58), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=speedlimits[7]))

        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a", CircularLane(center5, radii5 + 5, np.deg2rad(240), np.deg2rad(270), width=5, clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED), speed_limit=speedlimits[8]))
        net.add_lane("i", "a", CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5, clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS), speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self) -> None:
        self.controlled_vehicles = []
        road = self.road
        ego_lane = np.random.randint(2) if self.config.get("random_lane", False) else 0

        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", ego_lane)).position(0, 0),
            heading=road.network.get_lane(("a", "b", ego_lane)).heading_at(0),
            speed=min(9, self.config["speed_limit"]),
        )
        ego_vehicle.MAX_SPEED = self.config["speed_limit"]
        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        # Add a few traffic vehicles
        if self.config["other_vehicles"] > 0:
            vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", 0), longitudinal=0, speed=4)
            self.road.vehicles.append(vehicle)

        while len(self.road.vehicles) < self.config["other_vehicles"] + 1:
            random_lane_index = self.road.network.random_lane_index(self.np_random)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index, longitudinal=random.uniform(low=0, high=self.road.network.get_lane(random_lane_index).length), speed=4)
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 15:
                    break
            else:
                self.road.vehicles.append(vehicle)

    def _reward(self, action: np.ndarray) -> float:
        # action is [acceleration, steering]
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        speed_ratio = np.clip(self.vehicle.speed / max(1e-6, self.config["speed_limit"]), 0.0, 1.0)
        steering = action[1] if len(action) > 1 else 0.0

        lane_centering_reward = 1.0 / (1.0 + self.config["lane_centering_cost"] * (lateral ** 2))
        action_penalty = - self.config["action_penalty"] * np.linalg.norm(action)
        speed_reward = self.config["speed_reward"] * speed_ratio
        steering_cost = - self.config["steering_penalty"] * abs(steering)

        reward = lane_centering_reward + action_penalty + speed_reward + steering_cost

        # Offroad penalty
        if not self.vehicle.on_road or not self._reward_laning():
            reward = self.config.get("collision_reward", -5.0)

        # Crash overrides everything
        if self.vehicle.crashed:
            reward = self.config.get("collision_reward", -5.0)

        # Track offroad steps
        if not self.vehicle.on_road:
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0

        # Map reward to [0,1] for stability (same mapping as other racetrack env)
        reward = utils.lmap(reward, [-3.0, 3.0], [0.0, 1.0])
        return float(reward)

    def _is_terminal(self) -> bool:
        return self.vehicle.crashed or self._is_goal() or self.steps >= self.config["duration"]

    def _reward_laning(self) -> bool:
        current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        # Allow reward when driving on any lane segment (conservative)
        return True

    def _is_goal(self) -> bool:
        # Keep simple: episode goal is reaching the final segment ['i','a']
        return self.vehicle.on_road and self.vehicle.lane_index[:2] == ["i", "a"]
