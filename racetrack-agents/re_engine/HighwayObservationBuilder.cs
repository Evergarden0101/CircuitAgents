// ============================================================================
// HighwayObservationBuilder.cs  —  SINGLE-FILE observation builder for the
// RacetrackFast PPO ONNX model (highway-env 1.11 OccupancyGridObservation).
//
// Self-contained: the track lane geometry (from race_env._make_road) is
// embedded, so lat_off / ang_off / on_road are fully implemented — no
// spline system, no other files, no engine dependencies.
//
// Output: float[1440] = float32 tensor [1, 10, 12, 12] CHANNELS-FIRST,
//         flat index = feature*144 + cellX*12 + cellY. Feed directly to the
//         ONNX input "obs".
//
// !! Corrections vs the "naive" template this replaces (each of these would
//    silently break the model, verified against the real environment):
//    1. Grid is NOT symmetric: x spans -9 m (behind) .. +27 m (ahead),
//       y spans -18 .. +18. Ego sits at cell (3, 6), not the center.
//    2. x, y, vx, vy VALUES stay in TRACK/world axes (only the CELL INDEX
//       is rotated into the ego frame). Never Dot() them onto forward/right.
//    3. vx, vy are RELATIVE to the ego's velocity (subtract it).
//    4. cos_h / sin_h use the vehicle's ABSOLUTE heading, not ego-relative.
//    5. Normalization maps to [-1, 1] (value/range, clamped), not [0, 1].
//    6. on_road is its own LAYER: a trace of lane-centerline waypoints
//       (every 3 m, ±100 m around the ego), independent of vehicles. It is
//       NOT "is this vehicle on the road".
//    7. The EGO must be written too — last, after all NPCs, so it wins its
//       own cell. Its x/y/vx/vy are 0, its lat/ang offsets are its own.
//    8. Closest lane uses |lat| + longitudinal-overrun + |heading diff|
//       (highway-env distance_with_heading), evaluated over ALL 18 lanes.
//
// 2D mapping: the track plane is (Position.X, Position.Y). If your game
// ground plane is X/Z (Y-up engine), fill Vehicle with
//   Position = (world.X, world.Z, 0), Velocity = (vel.X, vel.Z, 0),
//   Heading  = Atan2(forward.Z, forward.X)
// and author/convert the game track in race_env coordinates.
//
// Cadence: run the model at 5 Hz (every 0.2 s) and hold the action between.
// ============================================================================

using System;
using System.Collections.Generic;
using System.Numerics;

namespace RacetrackSingle
{
    public class HighwayObservationBuilder
    {
        // ============================================================
        // Configuration (must match race_env.py exactly)
        // ============================================================

        public const int CellsX = 12;          // longitudinal cells
        public const int CellsY = 12;          // lateral cells
        public const int FeatureCount = 10;
        public const int ObservationLength = FeatureCount * CellsX * CellsY; // 1440

        const double GridMinX = -9.0;          // "grid_size": [[-9, 27], [-18, 18]]
        const double GridMinY = -18.0;
        const double GridStep = 3.0;           // "grid_step": [3, 3]

        // features_range — all symmetric, so normalized = value / range
        const double XRange = 100.0;
        const double YRange = 100.0;
        const double VxRange = 20.0;
        const double VyRange = 20.0;
        const double LatRange = 4.0;
        const double AngRange = 3.14159;       // python config literal (not PI)

        const double LanePerception = 100.0;   // on_road: ±100 m of centerline
        const double WaypointSpacing = 3.0;    // on_road: min(grid_step)

        public enum Feature
        {
            Presence = 0, OnRoad = 1, X = 2, Y = 3, VX = 4,
            VY = 5, CosHeading = 6, SinHeading = 7, LatOffset = 8, AngOffset = 9
        }

        // ============================================================
        // Vehicle (fill from your car components / transforms)
        // ============================================================

        public class Vehicle
        {
            public Vector3 Position;   // track plane = (X, Y); Z ignored
            public Vector3 Velocity;   // world velocity, same plane
            public float Heading;      // radians, Atan2 convention, track frame
        }

        // ============================================================
        // Public entry
        // ============================================================

        public float[] BuildObservation(Vehicle ego, IList<Vehicle> others)
        {
            float[] obs = new float[ObservationLength];
            BuildObservationInto(obs, ego, others);
            return obs;
        }

        // Allocation-free variant for per-frame use.
        public void BuildObservationInto(float[] obs, Vehicle ego, IList<Vehicle> others)
        {
            if (obs.Length != ObservationLength)
                throw new ArgumentException("obs must have length " + ObservationLength);
            Array.Clear(obs, 0, obs.Length);   // empty cells = 0

            double cosH = Math.Cos(ego.Heading);
            double sinH = Math.Sin(ego.Heading);

            //--------------------------------------------
            // on_road layer: lane-centerline waypoint trace
            //--------------------------------------------
            for (int l = 0; l < Lanes.Length; l++)
            {
                Lane lane = Lanes[l];
                double s0, latUnused;
                lane.Local(ego.Position.X, ego.Position.Y, out s0, out latUnused);
                for (double s = s0 - LanePerception; s < s0 + LanePerception; s += WaypointSpacing)
                {
                    double sc = s < 0.0 ? 0.0 : (s > lane.Length ? lane.Length : s);
                    double wx, wy;
                    lane.Point(sc, out wx, out wy);
                    int gx, gy;
                    if (CellOf(wx - ego.Position.X, wy - ego.Position.Y, cosH, sinH, out gx, out gy))
                        obs[(int)Feature.OnRoad * CellsX * CellsY + gx * CellsY + gy] = 1f;
                }
            }

            //--------------------------------------------
            // vehicles: all NPCs first, EGO LAST (wins its own cell)
            //--------------------------------------------
            for (int i = 0; i < others.Count; i++)
                WriteVehicle(obs, others[i], ego, cosH, sinH);
            WriteVehicle(obs, ego, ego, cosH, sinH);
        }

        // ============================================================
        // Per-vehicle feature write
        // ============================================================

        void WriteVehicle(float[] obs, Vehicle v, Vehicle ego, double cosH, double sinH)
        {
            // ego-relative, in TRACK axes — deliberately NOT rotated (see header)
            double dx = v.Position.X - ego.Position.X;
            double dy = v.Position.Y - ego.Position.Y;
            double dvx = v.Velocity.X - ego.Velocity.X;
            double dvy = v.Velocity.Y - ego.Velocity.Y;

            // only the CELL uses the ego-aligned frame
            int gx, gy;
            if (!CellOf(dx, dy, cosH, sinH, out gx, out gy))
                return;                        // outside the 36 m window

            // lane offsets of THIS vehicle relative to ITS closest lane
            double latOff, angOff;
            LaneOffsets(v.Position.X, v.Position.Y, v.Heading, out latOff, out angOff);

            const int L = CellsX * CellsY;
            int cell = gx * CellsY + gy;

            obs[(int)Feature.Presence   * L + cell] = 1f;
            obs[(int)Feature.X          * L + cell] = Norm(dx, XRange);
            obs[(int)Feature.Y          * L + cell] = Norm(dy, YRange);
            obs[(int)Feature.VX         * L + cell] = Norm(dvx, VxRange);
            obs[(int)Feature.VY         * L + cell] = Norm(dvy, VyRange);
            obs[(int)Feature.CosHeading * L + cell] = (float)Math.Cos(v.Heading); // ABSOLUTE
            obs[(int)Feature.SinHeading * L + cell] = (float)Math.Sin(v.Heading);
            obs[(int)Feature.LatOffset  * L + cell] = Norm(latOff, LatRange);
            obs[(int)Feature.AngOffset  * L + cell] = Norm(angOff, AngRange);
        }

        // ============================================================
        // Lane offsets (the finished lat_off / ang_off part)
        // ============================================================

        // Exactly highway-env: closest lane by
        //   |lat| + max(s-len,0) + max(-s,0) + 1.0*|wrap(heading - laneHeading(s))|
        // then lat_off = lateral offset, ang_off = wrap(heading - laneHeading).
        public void LaneOffsets(double px, double py, double heading,
                                out double latOff, out double angOff)
        {
            int best = 0;
            double bestScore = double.PositiveInfinity;
            double bestS = 0, bestLat = 0;

            for (int i = 0; i < Lanes.Length; i++)
            {
                double s, lat;
                Lanes[i].Local(px, py, out s, out lat);
                double overrun = Math.Max(s - Lanes[i].Length, 0.0) + Math.Max(-s, 0.0);
                double angle = Math.Abs(WrapToPi(heading - Lanes[i].HeadingAt(s)));
                double score = Math.Abs(lat) + overrun + angle;
                if (score < bestScore)
                {
                    bestScore = score; best = i; bestS = s; bestLat = lat;
                }
            }

            latOff = bestLat;
            angOff = WrapToPi(heading - Lanes[best].HeadingAt(bestS));
        }

        // ============================================================
        // Grid binning (align_to_vehicle_axes = true)
        // ============================================================

        static bool CellOf(double dx, double dy, double cosH, double sinH,
                           out int gx, out int gy)
        {
            double xr = cosH * dx + sinH * dy;    // rotate world -> ego frame
            double yr = -sinH * dx + cosH * dy;
            gx = (int)Math.Floor((xr - GridMinX) / GridStep);
            gy = (int)Math.Floor((yr - GridMinY) / GridStep);
            return gx >= 0 && gx < CellsX && gy >= 0 && gy < CellsY;
        }

        // ============================================================
        // Utilities
        // ============================================================

        // lmap(value, [-range, +range], [-1, 1]) followed by the global
        // clip to [-1, 1]  ==  value/range clamped (all ranges symmetric).
        static float Norm(double value, double range)
        {
            double t = value / range;
            if (t < -1.0) t = -1.0;
            if (t > 1.0) t = 1.0;
            return (float)t;
        }

        // numpy wrap_to_pi: result in [-pi, pi)
        static double WrapToPi(double a)
        {
            return a - 2.0 * Math.PI * Math.Floor((a + Math.PI) / (2.0 * Math.PI));
        }

        // ============================================================
        // Action decoding (ONNX "action_mean" [1,2], UNCLIPPED)
        // ============================================================

        public static void DecodeAction(float[] actionMean,
                                        out float accelerationMs2, out float steeringRad)
        {
            float a0 = Math.Clamp(actionMean[0], -1f, 1f);
            float a1 = Math.Clamp(actionMean[1], -1f, 1f);
            accelerationMs2 = a0 * 5.0f;                  // acceleration_range [-5, 5]
            steeringRad = a1 * (float)(Math.PI / 6.0);    // steering_range ±30°
        }

        // ============================================================
        // Track geometry — copy of race_env._make_road() (18 lanes).
        // Straights: start/end points. Arcs: center, radius, phases,
        // clockwise flag (highway-env direction = cw ? 1 : -1).
        // ============================================================

        sealed class Lane
        {
            bool _arc;
            // straight
            double _sx, _sy, _dirX, _dirY, _latX, _latY, _heading;
            // arc
            double _cx, _cy, _r, _phase0;
            int _dir;

            public double Length;

            public static Lane Straight(double x1, double y1, double x2, double y2)
            {
                Lane l = new Lane { _arc = false, _sx = x1, _sy = y1 };
                double len = Math.Sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
                l.Length = len;
                l._dirX = (x2 - x1) / len; l._dirY = (y2 - y1) / len;
                l._latX = -l._dirY; l._latY = l._dirX;
                l._heading = Math.Atan2(l._dirY, l._dirX);
                return l;
            }

            public static Lane Arc(double cx, double cy, double r,
                                   double phase0Deg, double phase1Deg, bool clockwise)
            {
                Lane l = new Lane { _arc = true, _cx = cx, _cy = cy, _r = r };
                l._dir = clockwise ? 1 : -1;
                double p0 = phase0Deg * Math.PI / 180.0;
                double p1 = phase1Deg * Math.PI / 180.0;
                l._phase0 = p0;
                l.Length = r * (p1 - p0) * l._dir;
                return l;
            }

            public void Local(double px, double py, out double s, out double lat)
            {
                if (!_arc)
                {
                    double dx = px - _sx, dy = py - _sy;
                    s = dx * _dirX + dy * _dirY;
                    lat = dx * _latX + dy * _latY;
                }
                else
                {
                    double dx = px - _cx, dy = py - _cy;
                    double phi = Math.Atan2(dy, dx);
                    phi = _phase0 + WrapToPi(phi - _phase0);
                    double r = Math.Sqrt(dx * dx + dy * dy);
                    s = _dir * (phi - _phase0) * _r;
                    lat = _dir * (_r - r);
                }
            }

            public void Point(double s, out double px, out double py)   // centerline
            {
                if (!_arc)
                {
                    px = _sx + s * _dirX;
                    py = _sy + s * _dirY;
                }
                else
                {
                    double phi = _dir * s / _r + _phase0;
                    px = _cx + _r * Math.Cos(phi);
                    py = _cy + _r * Math.Sin(phi);
                }
            }

            public double HeadingAt(double s)
            {
                if (!_arc) return _heading;
                double phi = _dir * s / _r + _phase0;
                return WrapToPi(phi + Math.PI / 2.0 * _dir);
            }
        }

        static readonly Lane[] Lanes = new Lane[]
        {
            // a -> b
            Lane.Straight(42, 0, 100, 0),
            Lane.Straight(42, 5, 100, 5),
            // b -> c
            Lane.Arc(100, -20, 20, 90, -1, false),
            Lane.Arc(100, -20, 25, 90, -1, false),
            // c -> d
            Lane.Straight(120, -19, 120, -30),
            Lane.Straight(125, -19, 125, -30),
            // d -> e
            Lane.Arc(105, -30, 15, 0, -181, false),
            Lane.Arc(105, -30, 20, 0, -181, false),
            // e -> f  (lane 0 is the OUTER radius on this segment)
            Lane.Arc(70, -30, 20, 0, 136, true),
            Lane.Arc(70, -30, 15, 0, 137, true),
            // f -> g
            Lane.Straight(55.7, -15.7, 35.7, -35.7),
            Lane.Straight(59.23553, -19.23553, 39.23553, -39.23553),
            // g -> h
            Lane.Arc(18.1, -18.1, 25, 315, 170, false),
            Lane.Arc(18.1, -18.1, 30, 315, 165, false),
            // h -> i
            Lane.Arc(18.1, -18.1, 25, 170, 56, false),
            Lane.Arc(18.1, -18.1, 30, 170, 58, false),
            // i -> a  (lane 0 is the OUTER radius on this segment)
            Lane.Arc(43.2, 23.4, 23.5, 240, 270, true),
            Lane.Arc(43.2, 23.4, 18.5, 238, 268, true),
        };
    }
}
