from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, CircularLane, LineType
from highway_env.vehicle.behavior import IDMVehicle


class RacetrackFast(AbstractEnv):
    """A full racetrack environment tuned for throttle + steering control

    This is a self-contained environment similar in style to
    `racetrack_env.RaceTrackEnv` but simplified and focused on the
    throttle/steering action space and a speed-focused reward term.
    """
    SEGMENT_SEQUENCE = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

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
                    "lat_off",
                    "ang_off",
                ],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
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
                "acceleration_range": [-3.0, 6.0],
                "steering_range": [-np.pi / 4, np.pi / 4],
                "dynamical": False,
            },
            # Simulation
            "duration": 1500,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            # Visuals
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            # Vehicles
            "controlled_vehicles": 1,
            "other_vehicles": 0,
            "speed_limit": 16.0,
            "terminate_off_road":        True,
            "length": 0,
            "no_lanes": 3,
            # Reward weights (tunable)
            "collision_reward": -5.0,
            "lane_centering_reward":  0.6,   # weight on 1/(1+cost×lat²)
            "lane_centering_cost": 4.0,
            "action_penalty": 0.1,
            "speed_reward": 1.2,
            "steering_penalty": 0.15,
            # ── Forward motion terms ───────────────────────────────────
            "reverse_penalty":  0.80,        # extra discrete hit for any speed < -0.5 m/s
            "idle_penalty":     0.80,        # hit for abs(speed) < 0.5 m/s
            "idle_grace_steps":       15,   # first 15 policy steps (3 seconds) are exempt

            "checkpoint_bonus":       0.25,   # bonus per segment crossing (9 segments total)
            "lap_bonus":              0.50,   # bonus for completing a full lap
            "forward_velocity_reward": 1.5,
            # Misc
            "show_trajectories": False,
            # ── NPC spawn randomisation ─────────────────────────────────
            "initial_speed":    0.0,    # ego spawn speed [m/s]
            "ego_min_speed":    -8.0,   # ← prevents reversing; set to 0 not negative
            "ego_max_speed":    16.0,

            "other_vehicles_speed_low":     1.0,    # slowest NPC [m/s]
                                                    # below ego's typical speed →
                                                    # forces overtaking behaviour
            "other_vehicles_speed_high":    17.0,   # fastest NPC [m/s]
                                                    # above speed_limit →
                                                    # forces defensive behaviour

            "other_vehicles_min_separation": 20.0,  # absolute min gap [m]
            "other_vehicles_sep_time":        2.0,  # gap multiplier [s]
                                                    # safe_gap = max(20, speed × 2)

            # ── IDM behaviour diversity ─────────────────────────────────
            "idm_acc_range":   [1.5, 5.0],  # COMFORT_ACC_MAX range [m/s²]
            "idm_dec_range":   [2.0, 5.0],  # |COMFORT_ACC_MIN| range [m/s²]
            "idm_gap_range":   [2.0, 8.0],  # DISTANCE_WANTED range [m]
            "idm_time_range":  [0.8, 2.5],  # TIME_WANTED range [s]

            "verbose_spawn":   False,       # print warning if not all NPCs spawned
        })
        return config

    def _reset(self) -> None:
        self.last_segment_start = "a"    # ego starts on "a"→"b"
        self.checkpoint_count   = 0
        self.lap_count          = 0
        self.episode_step       = 0
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        net = RoadNetwork()

        # A compact oval made of straights and arcs (inspired by racetrack_env)
        speedlimits = [None, 16, 16, 16, 16, 16, 16, 16, 16]

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

        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_vehicles(self) -> None:
        """
        Populate the track with ego + randomised IDM traffic.

        Randomisation axes for generalisation:
        ① Lane          — any lane on the network, including first NPC
        ② Longitudinal  — uniform across full lane length
        ③ Speed         — wide range [speed_low, speed_high]; exposes
                            slow blockers AND fast tailgaters
        ④ IDM params    — per-vehicle acc / gap / headway variation;
                            agent sees cautious AND aggressive drivers
        ⑤ Separation    — dynamic, speed-proportional safety gap
        """
        rng = self.np_random

        # ── Ego vehicle ────────────────────────────────────────────────
        ego_lane   = self.road.network.get_lane(("a", "b", 0))

        ego = self.action_type.vehicle_class.make_on_lane(
            self.road,
            ("a", "b", 0),
            longitudinal = ego_lane.length * 0.2,   # start at 20% of lane length
            speed        = self.config.get("initial_speed", 0.0)  ,
        )
        ego.MIN_SPEED = self.config.get("ego_min_speed", -8.0)
        ego.MAX_SPEED = self.config.get("ego_max_speed", 16.0)
        self.road.vehicles.append(ego)
        self.controlled_vehicles = [ego]

        # ── NPC config keys (add these to default_config) ──────────────
        speed_low   = self.config.get("other_vehicles_speed_low",       1.0)
        speed_high  = self.config.get("other_vehicles_speed_high",      17.0)
        min_sep     = self.config.get("other_vehicles_min_separation",  20.0)
        max_sep_mul = self.config.get("other_vehicles_sep_time",         2.0)
        # Dynamic safe gap = max(min_sep, vehicle_speed × sep_time)
        # → fast NPCs need bigger gaps; slow NPCs can be closer

        # IDM parameter ranges
        # idm_acc_range  = self.config.get("idm_acc_range",  [1.5, 5.0])   # COMFORT_ACC_MAX
        # idm_dec_range  = self.config.get("idm_dec_range",  [2.0, 5.0])   # |COMFORT_ACC_MIN|
        # idm_gap_range  = self.config.get("idm_gap_range",  [2.0, 8.0])   # DISTANCE_WANTED
        # idm_time_range = self.config.get("idm_time_range", [0.8, 2.5])   # TIME_WANTED

        # ── NPC placement loop ─────────────────────────────────────────
        target_count  = self.config["other_vehicles"] + 1   # +1 for ego already in list
        max_attempts  = target_count * 30                   # give up after this many tries
        attempts      = 0

        while len(self.road.vehicles) < target_count and attempts < max_attempts:
            attempts += 1

            # ① Random lane — covers whole network, not just ("b","c",0)
            lane_index = self.road.network.random_lane_index(rng)
            lane       = self.road.network.get_lane(lane_index)

            # ② Uniform longitudinal over full lane length
            longitudinal = float(rng.uniform(low=0.0, high=lane.length))

            # ③ Wide speed range — slow blockers and fast tailgaters
            speed = float(rng.uniform(low=speed_low, high=speed_high))

            candidate = IDMVehicle.make_on_lane(
                self.road,
                lane_index,
                longitudinal = longitudinal,
                speed        = speed,
            )

            # ④ Randomise IDM behaviour — per vehicle
            candidate.COMFORT_ACC_MAX = float(rng.uniform(*self.config["idm_acc_range"]))
            candidate.COMFORT_ACC_MIN = -float(rng.uniform(*self.config["idm_dec_range"]))
            candidate.DISTANCE_WANTED = float(rng.uniform(*self.config["idm_gap_range"]))
            candidate.TIME_WANTED     = float(rng.uniform(*self.config["idm_time_range"]))

            # ⑤ Dynamic separation — larger gap for faster vehicles
            safe_sep = max(min_sep, speed * max_sep_mul)

            too_close = any(
                np.linalg.norm(candidate.position - v.position) < safe_sep
                for v in self.road.vehicles
            )

            if not too_close:
                self.road.vehicles.append(candidate)

        if attempts >= max_attempts and self.config.get("verbose_spawn", False):
            print(f"[_make_vehicles] Warning: only spawned {len(self.road.vehicles) - 1} / "
                f"{self.config['other_vehicles']} NPCs after {max_attempts} attempts. "
                f"Track may be too crowded.")

    def _checkpoint_bonus(self) -> float:
            """
            Returns a one-time bonus when the ego crosses into the next
            segment in lap order. Returns 0.0 on every other step.
            Mutates self.last_segment_start, self.checkpoint_count, self.lap_count.
            """
            if self.vehicle.lane_index is None:
                return 0.0

            current_start = self.vehicle.lane_index[0]
            if current_start == self.last_segment_start:
                return 0.0   # still on the same segment, no bonus

            seq = self.SEGMENT_SEQUENCE
            try:
                last_idx    = seq.index(self.last_segment_start)
                current_idx = seq.index(current_start)
            except ValueError:
                self.last_segment_start = current_start
                return 0.0

            # Forward progress: moved to the NEXT segment in sequence
            expected_next = (last_idx + 1) % len(seq)
            bonus = 0.0

            if current_idx == expected_next:
                self.checkpoint_count += 1
                bonus = self.config.get("checkpoint_bonus", 0.25)

                # Completed a full lap (returned to segment 0 = "a")
                if current_idx == 0:
                    self.lap_count += 1
                    bonus += self.config.get("lap_bonus", 0.50)

            self.last_segment_start = current_start
            return bonus

    def _reward(self, action: np.ndarray) -> float:
        self.episode_step += 1
        speed       = self.vehicle.speed
        speed_limit = max(1e-6, self.config["speed_limit"])
        ego_min     = self.config.get("ego_min_speed", -8.0)

        # ── 1. Lane centering ──────────────────────────────────────────
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering = 1.0 / (1.0 + self.config["lane_centering_cost"] * lateral ** 2)

        # ── 2. Speed reward — negative for reversing, not zero ─────────
        # Forward:  [0, speed_limit] → [0.0, +1.0]
        # Reverse:  [ego_min, 0]     → [-1.0, 0.0]
        if speed >= 0:
            speed_ratio = np.clip(speed / speed_limit, 0.0, 1.0)
        else:
            speed_ratio = np.clip(speed / abs(ego_min), -1.0, 0.0)

        speed_reward = self.config["speed_reward"] * speed_ratio

        # Projects ego velocity onto the lane forward direction
        # Gives per-step incentive independent of speed magnitude
        forward_vel_reward = 0.0
        fwd_coef = self.config.get("forward_velocity_reward", 0.0)
        if fwd_coef > 0:
            try:
                lane_heading = self.vehicle.lane.heading_at(
                    self.vehicle.lane.local_coordinates(self.vehicle.position)[0]
                )
                vx, vy = self.vehicle.velocity
                # Component of velocity along lane tangent
                forward_proj = vx * np.cos(lane_heading) + vy * np.sin(lane_heading)
                forward_ratio = np.clip(forward_proj / speed_limit, -1.0, 1.0)
                forward_vel_reward = fwd_coef * forward_ratio
            except Exception:
                forward_vel_reward = 0.0

        # ── 3. Action penalties — separated, no double counting ─────────
        throttle = float(action[0])
        steering  = float(action[1]) if len(action) > 1 else 0.0

        # Throttle penalty: asymmetric — reverse throttle costs more
        # This replaces the old np.linalg.norm(action) which mixed both
        throttle_magnitude = abs(throttle) * (2.0 if throttle < 0.0 else 1.0)
        action_penalty = -self.config["action_penalty"] * throttle_magnitude

        # Steering penalty: independent from throttle penalty
        # Still needed — large steering through slow corners causes instability
        steering_cost = -self.config["steering_penalty"] * abs(steering)

        # ── Reverse and idle penalties ──────────────────────────────
        # Reverse: discrete signal so policy cannot rationalize away gradual drift
        reverse_penalty = (
            -self.config.get("reverse_penalty", 0.8)
            if speed < -0.5 else 0.0
        )

        # Idle: punish standing still so "do nothing" is not optimal
        grace = self.config.get("idle_grace_steps", 15)
        idle_penalty = (
            -self.config.get("idle_penalty", 0.8)
            if abs(speed) < 0.5 and self.episode_step > grace
            else 0.0
        )

        # ──  Compose base reward ─────────────────────────────────────
        reward = (
            lane_centering   # [0, 1]        stay on road, stay centered
            + speed_reward   # [-0.6, +0.6]  go forward, don't reverse
            + forward_vel_reward
            + action_penalty # [-0.3, 0]     don't over-throttle backward
            + steering_cost  # [-0.15, 0]    smooth steering
            + reverse_penalty# [-0.8, 0]     discrete reverse hit
            + idle_penalty   # [-0.4, 0]     discrete idle hit
        )

        # ── Crash penalty — overrides everything including offroad ───
        # Still needed: crash is terminal, must be the strongest signal.
        # Without this: policy learns that grazing walls is acceptable
        # because offroad_penalty and crash_penalty would be equal.
        # Keeping them the same value is fine since crash terminates the episode —
        # no future reward means the terminal signal IS the crash signal.
        if self.vehicle.crashed:
            reward = self.config.get("collision_reward", -5.0)

        # ── Map to [0, 1] ────────────────────────────────────────────
        # clip_range covers normal operation: base reward ∈ [-3, 3]
        # collision_reward=-5.0 gets clipped to 0.0 (minimum)
        reward = float(utils.lmap(reward, [-3.0, 3.0], [0.0, 1.0]))

        # Fires once per new segment entered in the correct lap direction.
        # Pays in [0,1] space directly so it is always meaningful regardless of base reward.
        reward = float(np.clip(reward + self._checkpoint_bonus(), 0.0, 1.0))
        return reward
    
    def _is_terminated(self) -> bool:
        if self.vehicle.crashed:
            return True
        if self.config["terminate_off_road"] and not self.vehicle.on_road:
            return True
            
        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
    
    def _reward_laning(self) -> bool:
        # current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        # Allow reward when driving on any lane segment (conservative)
        return True
