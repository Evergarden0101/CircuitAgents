# Mapping from the older DQN command:
#   --spawn_vehicles 3  -> config["other_vehicles"] = 3
#   --batch_size 256    -> PPO minibatch size 256
#   --lr 0.00005        -> PPO starting learning rate 5e-5
#   --lr_decay          -> SB3 linear learning-rate schedule
#   --arch Identity     -> flatten the occupancy grid, then use an MLP
#   --fc_layers 3       -> 3 hidden layers in the actor and critic
from highway_env.envs.racetrack_env import RacetrackEnvOval
import numpy as np
class RacetrackFast(RacetrackEnvOval):
    @classmethod
    def default_config(cls):
        # Ensure we call the parent classmethod with the class argument so
        # implementations that expect an explicit `cls` parameter work.
        try:
            config = super().default_config()
        except TypeError:
            # Fall back to calling the unbound parent method explicitly with cls
            config = RacetrackEnvOval.default_config(cls)
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
                "features_range": {          # ← CRITICAL: must be here or long_off is dead
                    "x":        [-100, 100],
                    "y":        [-100, 100],
                    "vx":       [-20,  20],
                    "vy":       [-20,  20],
                    "long_off": [0,    500], # your oval track circumference ≈ 500m
                    "lat_off":  [-4,   4],   # ±half lane width (lane width = 5m → ±2.5m tight)
                    "ang_off":  [-3.14159, 3.14159],
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
                "dynamical":False
            },
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 300,

            # Reward weights — all consumed by _reward()'s sum loop
            "collision_reward":     -1.0,
            "lane_centering_reward": 1.0,
            "lane_centering_cost":   4.0,
            "action_reward":        -0.1,   # lower penalty to allow throttle
            "speed_reward":          0.4,   # NEW — picked up automatically
            "steering_penalty":     -0.15,  # NEW — separate from throttle

            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "speed_limit": 10.0,
            "terminate_off_road": True,
            "length": 100,
            "no_lanes": 3,
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            # "block_lane": False,  # block middle lane
            # "force_decision": False,  # block 1st and 3rd lane
        })
        return config

    def _rewards(self, action: np.ndarray) -> dict:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        speed_ratio = np.clip(self.vehicle.speed / self.config["speed_limit"], 0, 1)
        steering = action[1]  # index 1 = lateral/steering component

        return {
            # Original terms (keep for lmap bounds to stay valid)
            "lane_centering_reward": 1 / (1 + self.config["lane_centering_cost"] * lateral**2),
            "action_reward":         np.linalg.norm(action),
            "collision_reward":      float(self.vehicle.crashed),
            "on_road_reward":        float(self.vehicle.on_road),
            # New terms — automatically weighted by config keys
            "speed_reward":          speed_ratio,        # 0→1 as speed→speed_limit
            "steering_penalty":      abs(steering),      # 0→1, penalise large steer alone
        }