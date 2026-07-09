from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, ActionType, action_factory
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
                # Asymmetric ego-frame FOV: 27 m ahead / 9 m behind (x),
                # ±18 m lateral (y). Still 36 m / 12 cells per axis, so the
                # obs shape stays (10, 12, 12) and the CNN is unchanged;
                # the ego just sits off-centre at cell (3, 6). The forward
                # bias gives ~2-3 s of corner preview at racing speed.
                "grid_size": [[-9, 27], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "acceleration_range": [-5.0, 5.0],
                "steering_range": [-np.pi / 6, np.pi / 6],
                "dynamical": True,
            },
            # Simulation
            "duration": 800,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            # Visuals
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            # Track selection: "fast" (9-segment circuit), "oval"
            # (2 straights + 2 half-circles), "stadium" (long oval, high
            # speed), "rect" (rounded rectangle, four 90° corners),
            # "chicane" (oval with a left-right S kink — the only track
            # besides "fast" with both turn directions), or "random"
            # (new choice every episode — use for training policies that
            # must generalize across tracks instead of memorizing one)
            "track": "fast",
            # Vehicles
            "controlled_vehicles": 1,
            "other_vehicles": 0,
            "speed_limit": 16.0,
            "terminate_off_road":        True,
            "length": 0,
            "no_lanes": 3,
            "car_length": 4.5,
            # Reward weights (tunable)
            "collision_reward": -5.0,
            "lane_centering_reward":  1.0,   # weight on 1/(1+cost×lat²)
            "lane_centering_cost": 6.0,
            # Throttle-command magnitude penalty — DISABLED by default:
            # its forward part taxes acceleration (counterproductive when
            # speed is the objective) and its doubled reverse part just
            # duplicates reverse_penalty while also taxing the reverse
            # command during wall escapes (it is not escape-zone waived).
            # Steering smoothness is handled by steering_penalty +
            # steering_jerk_penalty, not by this term.
            "action_penalty": 0.0,
            "speed_reward": 1.0,
            # Kept small: taxing steering too hard makes "hold throttle and
            # let the wall steer the car" score better than learning to corner
            "steering_penalty": 0.10,
            "steering_jerk_penalty":  0.20,   # penalise direction reversals specifically
            # ── Forward motion terms ───────────────────────────────────
            "reverse_penalty":  0.80,        # extra discrete hit for any speed < -0.5 m/s
            "idle_penalty":     0.80,        # hit for abs(speed) < 0.5 m/s
            "idle_grace_steps":       15,   # first 15 policy steps (3 seconds) are exempt

            # Paid in RAW reward units (same scale as the weights above),
            # added BEFORE the lmap to [0, 1]
            "checkpoint_bonus":       1.0,    # bonus per segment crossing (9 segments total)
            "lap_bonus":              2.0,    # bonus for completing a full lap
            "forward_velocity_reward": 0.8,
            # Misc
            "show_trajectories": False,
            # ── NPC spawn randomisation ─────────────────────────────────
            "initial_speed":    0.0,    # ego spawn speed [m/s] //TODO: 4.0
            "ego_min_speed":    -8.0,   # ← prevents reversing; set to 0 not negative
            "ego_max_speed":    16.0,

            "other_vehicles_speed_low":     1.0,    # slowest NPC [m/s]
                                                    # below ego's typical speed →
                                                    # forces overtaking behaviour
            "other_vehicles_speed_high":    14.0,   # fastest NPC [m/s]
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

            # Wall collision behaviour
            "wall_mode":           "stop",   # pin at wall; escape by reversing
            "wall_restitution":     0.1,     # only used in bounce mode
            "wall_friction":        0.6,     # high tangential friction, matches the real env
            "wall_escape_reward":   0.4,     # bonus for actively reversing from wall
            "wall_stuck_penalty":   0.4,     # penalty for sitting still at wall
            "wall_ride_penalty":    0.6,     # extra penalty ∝ speed while grinding the wall
            # Escape zone: within this clearance of a wall, reverse/idle
            # penalties are waived and clearance GAINED is rewarded.
            # Without it the escape only paid while touching the wall
            # (~0.15 m band): one step into a successful reverse, the
            # -0.8 reverse penalty resumed — so policies learned to creep
            # along the wall instead of backing away.
            "wall_escape_zone":      1.0,    # [m] clearance treated as "escaping"
            "wall_escape_progress":  0.8,    # reward per m of clearance gained in
                                             # the zone (clipped to [-0.2, +0.4]/step)
            # Terminate after this many consecutive steps of wall contact at
            # ANY speed (7 s @ 5 Hz). Covers both the parked-against-the-wall
            # case and the grinding-along-the-wall exploit with one rule.
            # Stop mode pins the car at the wall (contact persists), so the
            # escape manoeuvre gets a longer budget than bounce mode needed.
            "max_wall_contact_steps": 35,
        })
        return config

    def _reset(self) -> None:
        self._prev_steering     = 0.0   # ← add this
        self.last_segment_start = "a"    # ego starts on "a"→"b"
        self.checkpoint_count   = 0
        self.lap_count          = 0
        self.episode_step       = 0
        self.wall_contact_steps = 0     # consecutive steps touching a wall (any speed)
        self._prev_wall_gap     = None  # for escape-progress shaping
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """Build the configured track. Sets, per track:
        self.segment_sequence — lap order for the checkpoint bonus
        self.spawn_options    — lane keys the ego may spawn on
        """
        track = self.config.get("track", "fast")
        if track == "random":
            options = ("fast", "oval", "stadium", "rect", "chicane")
            track = options[int(self.np_random.integers(0, len(options)))]
        self.active_track = track
        builders = {
            "oval": self._make_road_oval,
            "stadium": self._make_road_stadium,
            "rect": self._make_road_rect,
            "chicane": self._make_road_chicane,
        }
        builders.get(track, self._make_road_fast)()

    def _make_road_oval(self) -> None:
        """Two-lane oval: 80 m straights joined by 25/30 m half-circles.
        Gentler than the fast circuit — used as the generalization probe
        (train on "random", evaluate here) or as an easier curriculum."""
        net = RoadNetwork()
        sl = 16

        net.add_lane("a", "b", StraightLane([0, 0], [80, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=sl))
        net.add_lane("a", "b", StraightLane([0, 5], [80, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center1 = [80, -25]
        net.add_lane("b", "c", CircularLane(center1, 25, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("b", "c", CircularLane(center1, 30, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("c", "d", StraightLane([80, -50], [0, -50], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("c", "d", StraightLane([80, -55], [0, -55], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center2 = [0, -25]
        net.add_lane("d", "a", CircularLane(center2, 25, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("d", "a", CircularLane(center2, 30, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        self.segment_sequence = ["a", "b", "c", "d"]
        self.spawn_options = [
            ("a", "b", 0), ("a", "b", 1),
            ("c", "d", 0), ("c", "d", 1),
        ]
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_road_stadium(self) -> None:
        """High-speed oval: 120 m straights joined by 30/35 m half-circles.
        The gentlest track — corners are takeable near the speed limit."""
        net = RoadNetwork()
        sl = 16

        net.add_lane("a", "b", StraightLane([0, 0], [120, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=sl))
        net.add_lane("a", "b", StraightLane([0, 5], [120, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center1 = [120, -30]
        net.add_lane("b", "c", CircularLane(center1, 30, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("b", "c", CircularLane(center1, 35, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("c", "d", StraightLane([120, -60], [0, -60], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("c", "d", StraightLane([120, -65], [0, -65], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center2 = [0, -30]
        net.add_lane("d", "a", CircularLane(center2, 30, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("d", "a", CircularLane(center2, 35, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        self.segment_sequence = ["a", "b", "c", "d"]
        self.spawn_options = [
            ("a", "b", 0), ("a", "b", 1),
            ("c", "d", 0), ("c", "d", 1),
        ]
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_road_rect(self) -> None:
        """Rounded rectangle: 60/20 m straights joined by four 20/25 m
        quarter-circles — teaches repeated identical 90° corners."""
        net = RoadNetwork()
        sl = 16

        net.add_lane("a", "b", StraightLane([0, 0], [60, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=sl))
        net.add_lane("a", "b", StraightLane([0, 5], [60, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center1 = [60, -20]
        net.add_lane("b", "c", CircularLane(center1, 20, np.deg2rad(90), np.deg2rad(0), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("b", "c", CircularLane(center1, 25, np.deg2rad(90), np.deg2rad(0), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("c", "d", StraightLane([80, -20], [80, -40], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("c", "d", StraightLane([85, -20], [85, -40], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center2 = [60, -40]
        net.add_lane("d", "e", CircularLane(center2, 20, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("d", "e", CircularLane(center2, 25, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("e", "f", StraightLane([60, -60], [0, -60], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("e", "f", StraightLane([60, -65], [0, -65], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center3 = [0, -40]
        net.add_lane("f", "g", CircularLane(center3, 20, np.deg2rad(-90), np.deg2rad(-180), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("f", "g", CircularLane(center3, 25, np.deg2rad(-90), np.deg2rad(-180), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("g", "h", StraightLane([-20, -40], [-20, -20], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("g", "h", StraightLane([-25, -40], [-25, -20], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center4 = [0, -20]
        net.add_lane("h", "a", CircularLane(center4, 20, np.deg2rad(-180), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("h", "a", CircularLane(center4, 25, np.deg2rad(-180), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        self.segment_sequence = ["a", "b", "c", "d", "e", "f", "g", "h"]
        self.spawn_options = [
            ("a", "b", 0), ("a", "b", 1),
            ("e", "f", 0), ("e", "f", 1),
        ]
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_road_chicane(self) -> None:
        """Oval with a left-right S kink in the bottom straight — besides
        "fast" this is the only track with BOTH turn directions, and its
        kink (10/15 m radii) is the tightest geometry in the pool."""
        net = RoadNetwork()
        sl = 16

        net.add_lane("a", "b", StraightLane([0, 0], [80, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=sl))
        net.add_lane("a", "b", StraightLane([0, 5], [80, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center1 = [80, -25]
        net.add_lane("b", "c", CircularLane(center1, 25, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("b", "c", CircularLane(center1, 30, np.deg2rad(90), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("c", "d", StraightLane([80, -50], [55, -50], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("c", "d", StraightLane([80, -55], [55, -55], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        # Left half of the S: clockwise arc, so lane 0 is the OUTER radius
        # (+lateral points radially inward on clockwise lanes)
        center2 = [55, -65]
        net.add_lane("d", "e", CircularLane(center2, 15, np.deg2rad(90), np.deg2rad(180), width=5, clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED), speed_limit=sl))
        net.add_lane("d", "e", CircularLane(center2, 10, np.deg2rad(90), np.deg2rad(180), width=5, clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS), speed_limit=sl))

        # Right half of the S
        center3 = [25, -65]
        net.add_lane("e", "f", CircularLane(center3, 15, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("e", "f", CircularLane(center3, 20, np.deg2rad(0), np.deg2rad(-90), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        net.add_lane("f", "g", StraightLane([25, -80], [0, -80], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=sl))
        net.add_lane("f", "g", StraightLane([25, -85], [0, -85], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=sl))

        center4 = [0, -40]
        net.add_lane("g", "a", CircularLane(center4, 40, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE), speed_limit=sl))
        net.add_lane("g", "a", CircularLane(center4, 45, np.deg2rad(-90), np.deg2rad(-270), width=5, clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS), speed_limit=sl))

        self.segment_sequence = ["a", "b", "c", "d", "e", "f", "g"]
        self.spawn_options = [
            ("a", "b", 0), ("a", "b", 1),
            ("f", "g", 0),
        ]
        self.road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

    def _make_road_fast(self) -> None:
        net = RoadNetwork()

        # A compact circuit made of straights and arcs (inspired by racetrack_env)
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

        # Second lane must sit at EXACTLY one lane width (5 m) of lateral
        # offset: 5/sqrt(2) = 3.53553 on each axis. The original offsets
        # (5.09 m apart) left a ~9 cm strip between the lanes where
        # vehicle.on_road was False, killing episodes mid-track.
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7], line_types=(LineType.CONTINUOUS, LineType.NONE), width=5, speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.23553, -19.23553], [39.23553, -39.23553], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[6]))

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

        self.segment_sequence = list(self.SEGMENT_SEQUENCE)
        self.spawn_options = [
            ("a", "b", 0), ("a", "b", 1),
            ("c", "d", 0),
            ("f", "g", 0),
            ("h", "i", 0),
        ]
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
        spawn_options = self.spawn_options   # set by the track builder
        spawn_idx = int(rng.integers(0, len(spawn_options)))
        spawn_lane_key = spawn_options[spawn_idx]

        # ── Ego vehicle ────────────────────────────────────────────────
        # ego_lane   = self.road.network.get_lane(("a", "b", 0))
        ego_lane  = self.road.network.get_lane(spawn_lane_key)
        spawn_long = float(rng.uniform(
            low  = ego_lane.length * 0.1,
            high = ego_lane.length * 0.9,
        ))

        ego = self.action_type.vehicle_class.make_on_lane(
            self.road,
            # ("a", "b", 0),
            # longitudinal = ego_lane.length * 0.2,   # start at 20% of lane length
            spawn_lane_key,
            longitudinal = spawn_long,
            speed        = self.config.get("initial_speed", 0.0)  ,
        )
        ego.MIN_SPEED = self.config.get("ego_min_speed", -8.0)
        ego.MAX_SPEED = self.config.get("ego_max_speed", 16.0)
        ego.LENGTH = self.config.get("car_length", 2.5)
        self.road.vehicles.append(ego)
        self.controlled_vehicles = [ego]
        self.last_segment_start = spawn_lane_key[0]

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
            candidate.LENGTH = self.config.get("car_length", 2.5)

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

            seq = getattr(self, "segment_sequence", self.SEGMENT_SEQUENCE)
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
    
    def _wall_state(self, vehicle):
        """
        Clearance between the car body and the physical track walls.

        Only the two edges of the whole multi-lane corridor are walls —
        the painted lines between adjacent lanes are NOT. Lateral position
        is therefore measured against the first lane (one track edge) and
        the last lane (the other track edge) of the closest segment,
        instead of whichever single lane happens to be closest.
        (On this track, lane i sits at lateral +i*width relative to lane 0
        on every segment, so lanes[0]/lanes[-1] bound the corridor.)

        Returns (clearance, wall_side, lane, longitudinal) where
          clearance    gap [m] between car body and the nearest wall
                       (negative = car overlaps the wall),
          wall_side    -1 for the wall on lane 0's outer side,
                       +1 for the wall on the last lane's outer side,
          lane         boundary lane whose frame defines the wall geometry,
          longitudinal position along that lane [m],
        or None when the geometry lookup fails.
        """
        try:
            lane_idx = self.road.network.get_closest_lane_index(vehicle.position)
            if lane_idx is None:
                return None
            lanes = self.road.network.graph[lane_idx[0]][lane_idx[1]]
            first, last = lanes[0], lanes[-1]
            long_f, lat_f = first.local_coordinates(vehicle.position)
            long_l, lat_l = last.local_coordinates(vehicle.position)
        except Exception:
            return None

        half_car = vehicle.WIDTH / 2.0
        # Corridor spans [-w/2] in the first lane's frame to [+w/2] in the
        # last lane's frame; lanes are stacked toward positive lateral.
        low_clearance  = (lat_f + first.width / 2.0) - half_car
        high_clearance = (last.width / 2.0 - lat_l) - half_car
        if low_clearance <= high_clearance:
            return low_clearance, -1.0, first, long_f
        return high_clearance, +1.0, last, long_l

    def _off_track(self) -> bool:
        """
        True when the car centre has genuinely left the track corridor.
        Safety net only — wall physics bounces the car back every sim frame,
        so this should almost never fire (e.g. tunnelling through a junction
        gap at high speed). Replaces vehicle.on_road, which is per-lane and
        reports False on lane stripes / tiny gaps between segments.
        """
        state = self._wall_state(self.vehicle)
        if state is None:
            return not self.vehicle.on_road
        clearance, _, _, _ = state
        # clearance is measured from the car BODY edge; centre crosses the
        # wall line when clearance < -half_width
        return clearance < -self.vehicle.WIDTH / 2.0

    def _is_at_wall(self) -> bool:
        """
        True when car body is within 10% of the free play from a track wall.
        Uses lateral clearance — correct for BOTH stop and bounce modes.
        In bounce mode, car is nudged back to the wall line; on_road becomes
        True, but clearance is still ~0 → this method catches it.
        """
        state = self._wall_state(self.vehicle)
        if state is None:
            return False
        clearance, _, lane, _ = state
        free_play = lane.width / 2.0 - self.vehicle.WIDTH / 2.0
        return clearance <= 0.10 * free_play

    def _reward(self, action: np.ndarray) -> float:
        self.episode_step += 1
        speed       = self.vehicle.speed
        speed_limit = max(1e-6, self.config["speed_limit"])
        ego_min     = self.config.get("ego_min_speed", -8.0)

        # ── 1. Lane centering ──────────────────────────────────────────
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering = 1.0 / (1.0 + self.config["lane_centering_cost"] * lateral ** 2)

        # Wall contact state — needed before the speed terms so wall-riding
        # cannot collect speed payouts
        at_wall = self._is_at_wall() and not self.vehicle.crashed
        if at_wall:
            self.wall_contact_steps += 1
        else:
            self.wall_contact_steps = 0

        # Escape zone: a band of clearance around the walls (wider than the
        # ~0.15 m contact band) in which the escape manoeuvre is protected —
        # reverse/idle penalties are waived and clearance gained is rewarded
        wall_state = self._wall_state(self.vehicle)
        wall_gap   = wall_state[0] if wall_state is not None else None
        escape_zone = self.config.get("wall_escape_zone", 1.0)
        near_wall = (
            wall_gap is not None and wall_gap < escape_zone
            and not self.vehicle.crashed
        )

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

        # No speed payout while touching the wall — otherwise wall-riding
        # (holding throttle while the bounce physics re-aligns the heading
        # and steers the car around corners for free) nets positive reward
        # and becomes a stable local optimum.
        if at_wall:
            speed_reward       = 0.0
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

        # Jerk penalty: targets CHANGE in steering specifically
        # This is what the original norm() was accidentally doing
        prev_steer   = getattr(self, "_prev_steering", 0.0)
        steer_delta  = steering - prev_steer
        jerk_penalty = -self.config["steering_jerk_penalty"] * abs(steer_delta)
        self._prev_steering = steering

        # ── Reverse and idle penalties ──────────────────────────────
        # Reverse: discrete signal so policy cannot rationalize away gradual drift
        # Waived in the whole escape zone, not just at contact: punishing
        # reverse one step after it pulls the car off the wall taught the
        # policy to creep along the wall instead of backing away.
        if near_wall:
            reverse_penalty = 0.0
            idle_penalty    = 0.0
        else:
            reverse_penalty = (
                -self.config.get("reverse_penalty", 0.8) if speed < -0.5 else 0.0
            )
            grace = self.config.get("idle_grace_steps", 15)
            idle_penalty = (
                -self.config.get("idle_penalty", 0.8)
                if abs(speed) < 0.5 and self.episode_step > grace else 0.0
            )

        # ── Wall escape reward ──────────────────────────────────────────
        # When stuck at wall, reward reverse throttle specifically
        # This overrides the normal idle/reverse penalty temporarily

        # Detect wall contact — car is stopped but not crashed, not off-road
        # (off-road is handled separately; wall contact means speed≈0 after _apply_wall_behavior)
        # mode = self.config.get("wall_mode", "stop")
        wall_escape_reward = 0.0
        if at_wall:
            actively_reversing = speed < -0.3   # vehicle IS moving backward
            if actively_reversing:
                # Positive reward: proportional to reverse speed
                escape_ratio       = abs(speed) / max(1.0, abs(ego_min))
                wall_escape_reward = self.config.get("wall_escape_reward", 0.6) * escape_ratio
            elif throttle < -0.2 and speed < 1.0:
                # Policy commands reverse but speed not negative yet
                # (car just started reversing this step). The speed guard
                # stops wall-grinding at speed from dodging the ride penalty
                # by merely HOLDING reverse throttle while momentum carries it
                wall_escape_reward = self.config.get("wall_escape_reward", 0.6) * 0.3
            else:
                # At wall, not reversing → penalty; grows with speed so
                # grinding ALONG the wall is worse than stopping at it
                ride = float(np.clip(speed / speed_limit, 0.0, 1.0))
                wall_escape_reward = -(
                    self.config.get("wall_stuck_penalty", 0.6)
                    + self.config.get("wall_ride_penalty", 0.6) * ride
                )

        # ── Escape-progress shaping ─────────────────────────────────────
        # Pays for clearance GAINED while inside the escape zone (and mildly
        # charges clearance lost), so the reward keeps flowing through the
        # whole reverse-out manoeuvre, not only while touching the wall.
        # Asymmetric clip: escaping earns up to +0.4/step, approaching costs
        # at most -0.2 (contact itself is already penalised above).
        if near_wall and wall_gap is not None:
            prev_gap = self._prev_wall_gap if self._prev_wall_gap is not None else wall_gap
            gap_delta = (float(np.clip(wall_gap, 0.0, escape_zone))
                         - float(np.clip(prev_gap, 0.0, escape_zone)))
            wall_escape_reward += float(np.clip(
                self.config.get("wall_escape_progress", 0.8) * gap_delta,
                -0.2, 0.4))
        self._prev_wall_gap = wall_gap


        # ──  Compose base reward ─────────────────────────────────────
        reward = (
            lane_centering   # [0, 1]        stay on road, stay centered
            + speed_reward   # [-1, +1]      go forward, don't reverse
            + forward_vel_reward  # [-0.4, +0.8]  (proj capped by |ego_min|=8 in reverse)
            + action_penalty # 0 by default (redundant with reverse_penalty)
            + steering_cost  # [-0.1, 0]     smooth steering
            + jerk_penalty   # [-0.4, 0]     no steering direction flips
            + reverse_penalty# [-0.8, 0]     discrete reverse hit
            + idle_penalty   # [-0.8, 0]     discrete idle hit (exclusive with reverse)
            + wall_escape_reward  # [-1.2, +0.8] incl. escape-progress shaping
        )

        # ── Crash penalty — overrides everything including offroad ───
        # Still needed: crash is terminal, must be the strongest signal.
        # Without this: policy learns that grazing walls is acceptable
        # because offroad_penalty and crash_penalty would be equal.
        # Keeping them the same value is fine since crash terminates the episode —
        # no future reward means the terminal signal IS the crash signal.
        if self.vehicle.crashed:
            reward = self.config.get("collision_reward", -5.0)
        elif self.wall_contact_steps >= self.config.get("max_wall_contact_steps", 25):
            # Sustained wall contact terminates like a crash — a lap must
            # not be completable by riding the walls
            reward = self.config.get("collision_reward", -5.0)
        elif self._off_track():
            # Centre fully outside the track corridor (deeper than wall
            # contact, which only means clearance ≈ 0) → terminate signal
            reward = self.config.get("collision_reward", -5.0)

        # Fires once per new segment entered in the correct lap direction.
        # Paid in RAW reward units so the config values are comparable to the
        # other weights. (Previously added after the lmap, in [0,1] space,
        # which silently made each unit of bonus worth 6 raw units.)
        reward += self._checkpoint_bonus()

        # ── Map to [0, 1] ────────────────────────────────────────────
        # Range must cover the true bounds of the composed reward, otherwise
        # distinct bad states saturate to 0 after the final clip and lose
        # their gradient. With the current weights: worst ordinary step
        # ≈ -2.8 (full reverse + jerk + reverse penalty, off-center), best
        # ≈ +2.8 — re-derive these bounds whenever a weight changes.
        # collision_reward=-5.0 maps below 0 and clips to 0.0; checkpoint/
        # lap steps may exceed +3 and clip to 1.0 — intentional, the
        # milestone step is allowed to saturate.
        reward = float(utils.lmap(reward, [-3.0, 3.0], [0.0, 1.0]))
        return float(np.clip(reward, 0.0, 1.0))
    
    def _is_terminated(self) -> bool:
        if self.vehicle.crashed:
            return True

        # Sustained wall contact at ANY speed → terminate. One rule covers
        # both being parked against the wall and grinding along it (the
        # wall-riding exploit). Counter is maintained in _reward, which
        # runs immediately before this method every step.
        if self.wall_contact_steps >= self.config.get("max_wall_contact_steps", 25):
            return True

        # Safety net: car centre escaped the corridor entirely (deeper than
        # mere wall contact, so no at-wall exemption here).
        if self._off_track() and self.config["terminate_off_road"]:
            return True

        return False

    def _is_truncated(self) -> bool:
        return self.time >= self.config["duration"]
    
    def _reward_laning(self) -> bool:
        # current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        # Allow reward when driving on any lane segment (conservative)
        return True
    

    # ── Fast dev specific wall behavior ──────────────────────────────────
    # For fast dev, use lighter pre-clamp so exploration can still
    # output small forward actions (which then bounce), giving more
    # signal than complete freezing

    def _pre_step_wall_clamp(self, vehicle) -> None:
        state = self._wall_state(vehicle)
        if state is None:
            return
        clearance, _, _, _ = state

        if clearance >= 0.0:
            return

        if hasattr(vehicle, "action") and isinstance(vehicle.action, dict):
            current_acc = vehicle.action.get("acceleration", 0.0)
            if self.config.get("fast_dev_wall_clamp", False):
                # Soft clamp: allow up to 0.5 m/s² forward even at wall
                # This lets exploration produce some movement signal
                # rather than complete freezing
                vehicle.action["acceleration"] = min(
                    0.5, float(current_acc)
                )
            else:
                # Hard clamp: no forward acceleration at wall (production)
                vehicle.action["acceleration"] = min(
                    0.0, float(current_acc)
                )

    def _simulate(self, action: Action | None = None) -> None:
        """
        Override of AbstractEnv._simulate.
        Matches the EXACT parent signature: (self, action=None).
        Adds wall collision response after each physics step.
        All other logic copied verbatim from the parent source.
        """
        frames = int(
            self.config["simulation_frequency"] // self.config["policy_frequency"]
        )
        for frame in range(frames):
            # FIX 3: action_type.act() must be called so vehicle receives commands
            if (
                action is not None
                and not self.config["manual_control"]
                and self.steps % frames == 0
            ):
                self.action_type.act(action)

            # Step 2: PRE-step wall clamp — prevent acceleration INTO wall
            # before road.step() can integrate it
            # if self.config.get("wall_mode", "none") != "none":
            #     self._pre_step_wall_clamp(self.vehicle)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])

            # ── Wall behavior injected HERE, after physics step ──────
            # Must run before observation is taken and before rendering
            if self.config.get("wall_mode", "none") != "none":
                self._apply_wall_behavior(self.vehicle)

            # FIX 4: steps counter must be incremented
            self.steps += 1

            # FIX 5: intermediate frame rendering for video recording
            if frame < frames - 1:
                self._automatic_rendering()

        self.enable_auto_render = False

    def _apply_wall_behavior(self, vehicle) -> None:
        """
        Detect wall contact via lateral offset and apply stop or bounce.
        Operates on vehicle.speed and vehicle.heading (NOT velocity property
        which is computed and cannot be set directly in HighwayEnv kinematics).
        """
        mode = self.config.get("wall_mode", "stop")

        # Corridor-level wall geometry: only the two track edges are walls,
        # never the boundary between adjacent lanes
        state = self._wall_state(vehicle)
        if state is None:
            return
        clearance, wall_side, lane, longitudinal = state

        if clearance >= 0.0:
            return   # inside the track corridor, no wall contact

        # ── Wall contact detected ──────────────────────────────────
        overshoot = -clearance

        # Boundary lane heading at this longitudinal position
        heading  = lane.heading_at(longitudinal)
        tangent  = np.array([ np.cos(heading),  np.sin(heading)])
        normal   = np.array([-np.sin(heading),  np.cos(heading)])
        # normal points toward positive lateral (left of travel direction)

        # Current velocity from kinematic model
        current_vel = vehicle.speed * np.array([
            np.cos(vehicle.heading), np.sin(vehicle.heading)
        ])
        v_along = float(np.dot(current_vel, tangent))  # tangential component
        v_perp  = float(np.dot(current_vel, normal))   # normal component (into wall)

        if mode == "stop":
            # BicycleVehicle has lateral velocity from tire slip
            # Must clear it or the vehicle will continue sliding laterally
            if hasattr(vehicle, "lateral_velocity"):
                vehicle.lateral_velocity = 0.0
            if hasattr(vehicle, "yaw_rate"):
                vehicle.yaw_rate = 0.0

            friction    = self.config.get("wall_friction", 0.5)
            new_v_along = v_along * (1.0 - friction)

            # Cancel ONLY the velocity component directed INTO the wall;
            # an outward component (the reverse-escape manoeuvre) must
            # survive or the car could never leave the wall
            new_v_perp = 0.0 if wall_side * v_perp > 0.0 else v_perp

            new_vel   = new_v_along * tangent + new_v_perp * normal
            new_speed = float(np.linalg.norm(new_vel))

            if new_speed > 0.05:
                vehicle.heading = float(np.arctan2(new_vel[1], new_vel[0]))
            vehicle.speed = float(np.clip(new_speed, 0.0, vehicle.MAX_SPEED))

            if hasattr(vehicle, "action") and isinstance(vehicle.action, dict):
                vehicle.action["acceleration"] = min(
                    0.0, float(vehicle.action.get("acceleration", 0.0))
                )
            vehicle.position -= wall_side * overshoot * normal

        elif mode == "bounce":

            restitution = self.config.get("wall_restitution", 0.1)
            friction    = self.config.get("wall_friction",    0.3)

            # Bounce: reflect normal, damp tangential by friction
            new_v_perp  = -restitution * v_perp           # reflect + energy loss
            new_v_along = v_along * (1.0 - friction)      # friction slows slide

            new_vel   = new_v_along * tangent + new_v_perp * normal
            new_speed = float(np.linalg.norm(new_vel))

            vehicle.speed = float(np.clip(
                new_speed, vehicle.MIN_SPEED, vehicle.MAX_SPEED
            ))
            if new_speed > 0.1:
                vehicle.heading = float(np.arctan2(new_vel[1], new_vel[0]))

            # Nudge back inside
            vehicle.position -= wall_side * overshoot * normal