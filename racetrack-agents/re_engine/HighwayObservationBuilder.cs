// ============================================================================
// HighwayObservationBuilder.cs  —  SINGLE-FILE observation builder for the
// RacetrackFast PPO ONNX model (highway-env 1.11 OccupancyGridObservation).
//
// Self-contained: no spline system, no other files, no engine dependencies.
// The LANES ARE SUPPLIED BY THE CALLER — build them with Lane.Straight
// (two points) / Lane.Arc (center, radius, phases, direction) and pass the
// list to the constructor:
//
//   var lanes = new List<HighwayObservationBuilder.Lane> {
//       HighwayObservationBuilder.Lane.Straight(42, 0, 100, 0),
//       HighwayObservationBuilder.Lane.Arc(100, -20, 20,
//           Math.PI / 2, -0.0175, clockwise: false),   // phases in RADIANS
//       ...
//   };
//   var builder = new HighwayObservationBuilder(lanes);
//
// The geometry MUST replicate race_env._make_road() exactly (same lanes,
// same order not required, but same centerlines) or lat_off/ang_off/on_road
// diverge from what the model saw in training.
// RacetrackFastLanes() returns the built-in copy of the current track.
//
// Output: float[1440] = float32 tensor [1, 12, 12, 10] CHANNELS-LAST (HWC),
//         flat index = (cellX*12 + cellY)*10 + feature. Feed directly to a
//         model input expecting an H*W*C sequence.
//         (The notebook's raw torch.onnx export is channels-first
//          [1, 10, 12, 12]; reorder with chw[(f*12+ix)*12+iy] =
//          hwc[(ix*12+iy)*10+f] if you target that layout instead.)
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
        // Construction — lanes come from the caller
        // ============================================================

        private readonly Lane[] _lanes;

        public HighwayObservationBuilder(IList<Lane> lanes)
        {
            if (lanes == null || lanes.Count == 0)
                throw new ArgumentException("at least one lane is required");
            _lanes = new Lane[lanes.Count];
            for (int i = 0; i < lanes.Count; i++) _lanes[i] = lanes[i];
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
            for (int l = 0; l < _lanes.Length; l++)
            {
                Lane lane = _lanes[l];
                double s0, latUnused;
                lane.Local(ego.Position.X, ego.Position.Y, out s0, out latUnused);
                for (double s = s0 - LanePerception; s < s0 + LanePerception; s += WaypointSpacing)
                {
                    double sc = s < 0.0 ? 0.0 : (s > lane.Length ? lane.Length : s);
                    double wx, wy;
                    lane.Point(sc, out wx, out wy);
                    int gx, gy;
                    if (CellOf(wx - ego.Position.X, wy - ego.Position.Y, cosH, sinH, out gx, out gy))
                        obs[(gx * CellsY + gy) * FeatureCount + (int)Feature.OnRoad] = 1f;
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

            // HWC: all 10 features of one cell are contiguous
            int cell = (gx * CellsY + gy) * FeatureCount;

            obs[cell + (int)Feature.Presence]   = 1f;
            obs[cell + (int)Feature.X]          = Norm(dx, XRange);
            obs[cell + (int)Feature.Y]          = Norm(dy, YRange);
            obs[cell + (int)Feature.VX]         = Norm(dvx, VxRange);
            obs[cell + (int)Feature.VY]         = Norm(dvy, VyRange);
            obs[cell + (int)Feature.CosHeading] = (float)Math.Cos(v.Heading); // ABSOLUTE
            obs[cell + (int)Feature.SinHeading] = (float)Math.Sin(v.Heading);
            obs[cell + (int)Feature.LatOffset]  = Norm(latOff, LatRange);
            obs[cell + (int)Feature.AngOffset]  = Norm(angOff, AngRange);
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

            for (int i = 0; i < _lanes.Length; i++)
            {
                double s, lat;
                _lanes[i].Local(px, py, out s, out lat);
                double overrun = Math.Max(s - _lanes[i].Length, 0.0) + Math.Max(-s, 0.0);
                double angle = Math.Abs(WrapToPi(heading - _lanes[i].HeadingAt(s)));
                double score = Math.Abs(lat) + overrun + angle;
                if (score < bestScore)
                {
                    bestScore = score; best = i; bestS = s; bestLat = lat;
                }
            }

            latOff = bestLat;
            angOff = WrapToPi(heading - _lanes[best].HeadingAt(bestS));
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

        // INVERSE of CellOf, for debug-drawing the grid: world-frame centre
        // of cell (gx, gy). The grid's +x axis IS the ego heading — gx grows
        // toward where the car FACES (-9 m behind .. +27 m ahead), gy toward
        // the car's LEFT. A grid that always points world-east ("right")
        // means the draw code skipped this rotation; a mirrored one means it
        // used CellOf's world->ego rotation (transposed sin signs) instead.
        public static void CellCenterWorld(int gx, int gy, Vehicle ego,
                                           out double wx, out double wy)
        {
            double lx = GridMinX + (gx + 0.5) * GridStep;   // forward, ego frame
            double ly = GridMinY + (gy + 0.5) * GridStep;   // left,    ego frame
            double c = Math.Cos(ego.Heading), s = Math.Sin(ego.Heading);
            wx = ego.Position.X + c * lx - s * ly;
            wy = ego.Position.Y + s * lx + c * ly;
        }

        // ============================================================
        // Heading helpers — heading is COUNTER-clockwise from track +x
        // (the a->b straight); engine yaw is CLOCKWISE from +Z. With the
        // Y-up mapping (track.x = world.X, track.y = world.Z):
        //   heading = Atan2(forward.Z, forward.X) = 90° - yaw.
        // A car spawned with yRotation = 90 faces a->b -> heading 0. If you
        // pass raw yaw as heading, every observation is rotated ~90° and
        // the grid preview points sideways.
        // ============================================================

        public static float HeadingFromForward(Vector3 forward)
        {
            return (float)Math.Atan2(forward.Z, forward.X);
        }

        // Row-major/row-vector matrices (System.Numerics, DirectX): the
        // world-space forward (local +Z) axis is row 3 = (M31, M32, M33).
        public static float HeadingFromWorldMatrix(Matrix4x4 m)
        {
            return (float)Math.Atan2(m.M33, m.M31);
        }

        public static float HeadingFromEulerYawDegrees(float yawDegrees)
        {
            return (float)WrapToPi((90.0 - yawDegrees) * Math.PI / 180.0);
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
        // Lane geometry — construct these OUTSIDE and pass them to the
        // builder's constructor. Two shapes, mirroring highway-env:
        //   Lane.Straight(x1, y1, x2, y2)          — centerline endpoints
        //   Lane.Arc(cx, cy, radius, p0, p1, cw)   — phases in RADIANS
        //   Lane.ArcDegrees(...)                   — same, phases in degrees
        // clockwise maps to highway-env direction = cw ? 1 : -1; length is
        // derived (straight: point distance, arc: radius * swept angle).
        // ============================================================

        public sealed class Lane
        {
            bool _arc;
            // straight
            double _sx, _sy, _dirX, _dirY, _latX, _latY, _heading;
            // arc
            double _cx, _cy, _r, _phase0;
            int _dir;

            public double Length { get; private set; }

            Lane() { }

            public static Lane Straight(double x1, double y1, double x2, double y2)
            {
                double len = Math.Sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
                if (len <= 0.0)
                    throw new ArgumentException("straight lane needs two distinct points");
                Lane l = new Lane { _arc = false, _sx = x1, _sy = y1 };
                l.Length = len;
                l._dirX = (x2 - x1) / len; l._dirY = (y2 - y1) / len;
                l._latX = -l._dirY; l._latY = l._dirX;
                l._heading = Math.Atan2(l._dirY, l._dirX);
                return l;
            }

            // phases in RADIANS (highway-env / track.json convention)
            public static Lane Arc(double cx, double cy, double radius,
                                   double startPhase, double endPhase, bool clockwise)
            {
                if (radius <= 0.0)
                    throw new ArgumentException("arc lane needs a positive radius");
                Lane l = new Lane { _arc = true, _cx = cx, _cy = cy, _r = radius };
                l._dir = clockwise ? 1 : -1;
                l._phase0 = startPhase;
                l.Length = radius * (endPhase - startPhase) * l._dir;
                if (l.Length <= 0.0)
                    throw new ArgumentException(
                        "arc phases sweep against the clockwise flag (negative length)");
                return l;
            }

            // phases in degrees (convenience; race_env authors arcs in degrees)
            public static Lane ArcDegrees(double cx, double cy, double radius,
                                          double startPhaseDeg, double endPhaseDeg,
                                          bool clockwise)
            {
                return Arc(cx, cy, radius,
                           startPhaseDeg * Math.PI / 180.0,
                           endPhaseDeg * Math.PI / 180.0, clockwise);
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

        // ============================================================
        // Built-in preset — copy of race_env._make_road() (18 lanes).
        // Use it as-is, or as the template for authoring your own list:
        //   new HighwayObservationBuilder(
        //       HighwayObservationBuilder.RacetrackFastLanes())
        // ============================================================

        public static Lane[] RacetrackFastLanes()
        {
            return new Lane[]
            {
                // a -> b
                Lane.Straight(42, 0, 100, 0),
                Lane.Straight(42, 5, 100, 5),
                // b -> c
                Lane.ArcDegrees(100, -20, 20, 90, -1, false),
                Lane.ArcDegrees(100, -20, 25, 90, -1, false),
                // c -> d
                Lane.Straight(120, -19, 120, -30),
                Lane.Straight(125, -19, 125, -30),
                // d -> e
                Lane.ArcDegrees(105, -30, 15, 0, -181, false),
                Lane.ArcDegrees(105, -30, 20, 0, -181, false),
                // e -> f  (lane 0 is the OUTER radius on this segment)
                Lane.ArcDegrees(70, -30, 20, 0, 136, true),
                Lane.ArcDegrees(70, -30, 15, 0, 137, true),
                // f -> g
                Lane.Straight(55.7, -15.7, 35.7, -35.7),
                Lane.Straight(59.23553, -19.23553, 39.23553, -39.23553),
                // g -> h
                Lane.ArcDegrees(18.1, -18.1, 25, 315, 170, false),
                Lane.ArcDegrees(18.1, -18.1, 30, 315, 165, false),
                // h -> i
                Lane.ArcDegrees(18.1, -18.1, 25, 170, 56, false),
                Lane.ArcDegrees(18.1, -18.1, 30, 170, 58, false),
                // i -> a  (lane 0 is the OUTER radius on this segment)
                Lane.ArcDegrees(43.2, 23.4, 23.5, 240, 270, true),
                Lane.ArcDegrees(43.2, 23.4, 18.5, 238, 268, true),
            };
        }
    }
}
