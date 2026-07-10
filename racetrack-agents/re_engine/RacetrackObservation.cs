// ============================================================================
// RacetrackObservation.cs
//
// Faithful C# port of highway-env 1.11 OccupancyGridObservation, configured
// exactly like race_env.RacetrackFast, for running the exported PPO ONNX
// model (ppo_actor_only.onnx) inside RE Engine / Unity-like C# games.
//
// Engine-agnostic: only System math, no engine types. Adapt with the thin
// wrappers at the bottom of this header.
//
// ---------------------------------------------------------------------------
// OBSERVATION CONTRACT (must match training EXACTLY)
// ---------------------------------------------------------------------------
// Tensor: float32 [1, 12, 12, 10], HWC (channels-LAST) sequence,
//   flat index = (ix*12 + iy)*10 + f
//   ix = longitudinal cell (grid x: -9 m .. +27 m ahead, 3 m cells)  = H
//   iy = lateral cell      (grid y: -18 m .. +18 m,       3 m cells) = W
//   f  = feature/channel                                             = C
//   ego sits at cell (3, 6)
// Deploy ppo_actor_only_nhwc.onnx (input [1, 12, 12, 10]) with this builder —
// exported by the notebook's NHWC cell, or converted from an existing
// channels-first export with convert_onnx_nhwc.py. Feeding this HWC sequence
// into the channels-FIRST export (ppo_actor_only.onnx, [1, 10, 12, 12])
// scrambles every channel: the policy degenerates to near-constant outputs
// (throttle ~0.4, steering ~0.2) and cannot corner.
// HwcToChw() is provided for the channels-first path if ever needed.
//
// Features (channel order):
//   0 presence  1 at each vehicle's cell, else 0
//   1 on_road   1 at cells crossed by LANE CENTERLINE waypoints (see below)
//   2 x         (v.x - ego.x)   / 100, TRACK axes (NOT rotated to ego frame)
//   3 y         (v.y - ego.y)   / 100, TRACK axes
//   4 vx        (v.vx - ego.vx) / 20,  TRACK axes
//   5 vy        (v.vy - ego.vy) / 20,  TRACK axes
//   6 cos_h     cos(v.heading)  ABSOLUTE heading in track frame
//   7 sin_h     sin(v.heading)
//   8 lat_off   v's lateral offset from ITS closest lane centerline, / 4
//   9 ang_off   wrap(v.heading - lane heading at v), / 3.14159
//
// Cell placement: relative position (dx,dy) rotated INTO the ego frame
// (align_to_vehicle_axes=True). The VALUES of x,y,vx,vy stay in track axes —
// only the cell index is rotated. This asymmetry is how highway-env works
// and how the model was trained; do not "fix" it.
//
// Write order: NPCs first, ego LAST (highway-env iterates road.vehicles
// reversed, ego is vehicles[0]) — so ego wins its own cell. Cells without a
// vehicle stay 0 (numpy fills NaN then nan_to_num->0). All written values
// are clipped to [-1, 1].
//
// on_road layer: for every lane, project ego onto the lane to get origin s0,
// then walk waypoints s = s0-100 .. s0+100 in steps of 3 m (clamped to
// [0, lane.length]) and mark the cell containing centerline point(s).
// It is a CENTERLINE TRACE, not a filled-road mask and not a coverage
// percentage. Do NOT raycast a filled corridor — the model never saw that.
//
// lat_off / ang_off: need lane geometry in-game. Two options provided:
//   1. Track.BuildRacetrackFast()      — hardcoded copy of race_env._make_road
//   2. Track.FromData(TrackData)       — load track.json exported from python
//      (see export_track_json.py; parse with your engine's JSON lib)
// Closest lane = argmin over lanes of:
//      |lat| + max(s-len,0) + max(-s,0) + 1.0*|wrap(heading - laneHeading(s))|
// (this is highway-env's distance_with_heading used by vehicle.on_state_update)
//
// ---------------------------------------------------------------------------
// COORDINATE MAPPING (game -> track frame)
// ---------------------------------------------------------------------------
// highway-env is 2D. Map your game ground plane to it consistently, e.g.
// Unity-style Y-up:   track.x = world.X,  track.y = world.Z
//   position: (t.x, t.y)     velocity: rigidbody velocity (X, Z)
//   heading:  Math.Atan2(forward.Z, forward.X)
// The track JSON / hardcoded track is authored in race_env coordinates, so
// either build your game track at those coordinates or convert with Frame2D.
//
// ---------------------------------------------------------------------------
// RUNNING THE MODEL
// ---------------------------------------------------------------------------
// - Inference cadence: policy_frequency = 5 Hz. Run the model every 0.2 s of
//   game time and HOLD the action in between.
// - Model input "obs":         [1, 12, 12, 10] float32 HWC (this builder's
//                              output; use HwcToChw for a channels-first model)
// - ONNX output "action_mean": [1, 2] float32, UNCLIPPED. Decode:
//       a = clamp(action_mean, -1, 1)
//       acceleration = a[0] * 5.0          //  acceleration_range [-5, 5] m/s^2
//       steering     = a[1] * (PI / 6.0)   //  steering_range [-30deg, +30deg]
//   (see ActionDecoder below; positive steering turns toward +lateral, i.e.
//    the +y side in track axes — verify the sign once in-game.)
// ============================================================================

using System;
using System.Collections.Generic;

namespace RacetrackAgents
{
    // ------------------------------------------------------------------ math
    public static class HwMath
    {
        public const double TwoPi = 2.0 * Math.PI;

        // numpy-compatible wrap to [-pi, pi)
        public static double WrapToPi(double x)
        {
            return x - TwoPi * Math.Floor((x + Math.PI) / TwoPi);
        }

        // highway_env.utils.lmap: linear map WITHOUT clipping
        public static double Lmap(double v, double inLow, double inHigh, double outLow, double outHigh)
        {
            return outLow + (v - inLow) * (outHigh - outLow) / (inHigh - inLow);
        }

        // ALL speeds/velocities in this contract are METERS PER SECOND in
        // track axes. If your engine hands you km/h (e.g. a kph Vector3),
        // convert at the boundary — feeding kph scales every velocity 3.6x:
        // the vx/vy observation channels clip at +-20 m/s and the model
        // silently degrades with no error anywhere.
        public static double KphToMs(double kph) { return kph / 3.6; }

        public static float Clamp1(double v)
        {
            if (v < -1.0) return -1f;
            if (v > 1.0) return 1f;
            return (float)v;
        }
    }

    public struct Vec2
    {
        public double X, Y;
        public Vec2(double x, double y) { X = x; Y = y; }
        public static Vec2 operator +(Vec2 a, Vec2 b) { return new Vec2(a.X + b.X, a.Y + b.Y); }
        public static Vec2 operator -(Vec2 a, Vec2 b) { return new Vec2(a.X - b.X, a.Y - b.Y); }
        public double Norm() { return Math.Sqrt(X * X + Y * Y); }
    }

    // One vehicle's state, already expressed in TRACK-frame 2D coordinates.
    public struct VehicleState
    {
        public double X, Y;        // position [m]
        public double Vx, Vy;      // velocity [m/s] (world/track axes) —
                                   // km/h sources: HwMath.KphToMs first!
        public double Heading;     // [rad], atan2 convention in track axes

        public VehicleState(double x, double y, double vx, double vy, double heading)
        {
            X = x; Y = y; Vx = vx; Vy = vy; Heading = heading;
        }
    }

    // Optional helper to convert a game frame into the track frame:
    // track = R(-yaw) * (world - origin), headings shifted by -yaw,
    // with optional Y mirror for handedness mismatches.
    public struct Frame2D
    {
        public double OriginX, OriginY; // track-frame position of game origin
        public double Yaw;              // rotation from game axes to track axes [rad]
        public bool MirrorY;            // flip lateral axis (left/right-handed fix)

        public VehicleState ToTrack(double px, double py, double vx, double vy, double heading)
        {
            double c = Math.Cos(-Yaw), s = Math.Sin(-Yaw);
            double tx = c * px - s * py + OriginX;
            double ty = s * px + c * py + OriginY;
            double tvx = c * vx - s * vy;
            double tvy = s * vx + c * vy;
            double th = heading + (-Yaw);
            if (MirrorY) { ty = -ty + 2 * OriginY; tvy = -tvy; th = -th; }
            return new VehicleState(tx, ty, tvx, tvy, HwMath.WrapToPi(th));
        }
    }

    // ----------------------------------------------------------------- lanes
    public abstract class Lane
    {
        public double Width = 5.0;
        public abstract double Length { get; }
        public abstract void LocalCoordinates(Vec2 p, out double s, out double lat);
        public abstract Vec2 PositionAt(double s, double lat);
        public abstract double HeadingAt(double s);

        // highway-env AbstractLane.distance (L1)
        public double Distance(Vec2 p)
        {
            double s, lat;
            LocalCoordinates(p, out s, out lat);
            return Math.Abs(lat) + Math.Max(s - Length, 0.0) + Math.Max(-s, 0.0);
        }

        // highway-env AbstractLane.distance_with_heading (heading_weight = 1.0)
        public double DistanceWithHeading(Vec2 p, double heading)
        {
            double s, lat;
            LocalCoordinates(p, out s, out lat);
            double angle = Math.Abs(HwMath.WrapToPi(heading - HeadingAt(s)));
            return Math.Abs(lat) + Math.Max(s - Length, 0.0) + Math.Max(-s, 0.0) + angle;
        }
    }

    public sealed class StraightLane : Lane
    {
        public Vec2 Start, End;
        private Vec2 _dir, _dirLat;
        private double _length, _heading;

        public StraightLane(Vec2 start, Vec2 end, double width)
        {
            Start = start; End = end; Width = width;
            _length = (end - start).Norm();
            _dir = new Vec2((end.X - start.X) / _length, (end.Y - start.Y) / _length);
            _dirLat = new Vec2(-_dir.Y, _dir.X);
            _heading = Math.Atan2(_dir.Y, _dir.X);
        }

        public override double Length { get { return _length; } }

        public override void LocalCoordinates(Vec2 p, out double s, out double lat)
        {
            Vec2 d = p - Start;
            s = d.X * _dir.X + d.Y * _dir.Y;
            lat = d.X * _dirLat.X + d.Y * _dirLat.Y;
        }

        public override Vec2 PositionAt(double s, double lat)
        {
            return new Vec2(Start.X + s * _dir.X + lat * _dirLat.X,
                            Start.Y + s * _dir.Y + lat * _dirLat.Y);
        }

        public override double HeadingAt(double s) { return _heading; }
    }

    public sealed class CircularLane : Lane
    {
        public Vec2 Center;
        public double Radius, StartPhase, EndPhase;
        public bool Clockwise;
        private int _direction;      // 1 if clockwise else -1 (highway-env convention)
        private double _length;

        public CircularLane(Vec2 center, double radius, double startPhase, double endPhase,
                            bool clockwise, double width)
        {
            Center = center; Radius = radius;
            StartPhase = startPhase; EndPhase = endPhase;
            Clockwise = clockwise; Width = width;
            _direction = clockwise ? 1 : -1;
            _length = radius * (endPhase - startPhase) * _direction;
        }

        public override double Length { get { return _length; } }

        public override void LocalCoordinates(Vec2 p, out double s, out double lat)
        {
            Vec2 delta = p - Center;
            double phi = Math.Atan2(delta.Y, delta.X);
            phi = StartPhase + HwMath.WrapToPi(phi - StartPhase);
            double r = delta.Norm();
            s = _direction * (phi - StartPhase) * Radius;
            lat = _direction * (Radius - r);
        }

        public override Vec2 PositionAt(double s, double lat)
        {
            double phi = _direction * s / Radius + StartPhase;
            double r = Radius - lat * _direction;
            return new Vec2(Center.X + r * Math.Cos(phi), Center.Y + r * Math.Sin(phi));
        }

        public override double HeadingAt(double s)
        {
            double phi = _direction * s / Radius + StartPhase;
            return HwMath.WrapToPi(phi + Math.PI / 2.0 * _direction);
        }
    }

    // ----------------------------------------------------- track (lane list)
    // Plain DTOs for JSON loading (fill them with your engine's JSON parser,
    // or System.Text.Json with IncludeFields=true). Schema = track.json from
    // export_track_json.py.
    [Serializable]
    public class TrackLaneData
    {
        public string type;          // "straight" | "circular"
        public double[] start;       // straight: [x, y]
        public double[] end;         // straight: [x, y]
        public double[] center;      // circular: [x, y]
        public double radius;        // circular
        public double start_phase;   // circular [rad]
        public double end_phase;     // circular [rad]
        public bool clockwise;       // circular
        public double width;
    }

    [Serializable]
    public class TrackData
    {
        public TrackLaneData[] lanes;
    }

    public sealed class Track
    {
        public readonly List<Lane> Lanes = new List<Lane>();

        public static Track FromData(TrackData data)
        {
            Track t = new Track();
            foreach (TrackLaneData ld in data.lanes)
            {
                if (ld.type == "straight")
                    t.Lanes.Add(new StraightLane(new Vec2(ld.start[0], ld.start[1]),
                                                 new Vec2(ld.end[0], ld.end[1]), ld.width));
                else if (ld.type == "circular")
                    t.Lanes.Add(new CircularLane(new Vec2(ld.center[0], ld.center[1]),
                                                 ld.radius, ld.start_phase, ld.end_phase,
                                                 ld.clockwise, ld.width));
                else
                    throw new ArgumentException("unknown lane type: " + ld.type);
            }
            return t;
        }

        // Hardcoded copy of race_env.RacetrackFast._make_road (same lane order).
        public static Track BuildRacetrackFast()
        {
            const double W = 5.0;
            Func<double, double> deg = d => d * Math.PI / 180.0;
            Track t = new Track();
            // a -> b
            t.Lanes.Add(new StraightLane(new Vec2(42, 0), new Vec2(100, 0), W));
            t.Lanes.Add(new StraightLane(new Vec2(42, 5), new Vec2(100, 5), W));
            // b -> c   (center [100,-20])
            t.Lanes.Add(new CircularLane(new Vec2(100, -20), 20, deg(90), deg(-1), false, W));
            t.Lanes.Add(new CircularLane(new Vec2(100, -20), 25, deg(90), deg(-1), false, W));
            // c -> d
            t.Lanes.Add(new StraightLane(new Vec2(120, -19), new Vec2(120, -30), W));
            t.Lanes.Add(new StraightLane(new Vec2(125, -19), new Vec2(125, -30), W));
            // d -> e   (center [105,-30])
            t.Lanes.Add(new CircularLane(new Vec2(105, -30), 15, deg(0), deg(-181), false, W));
            t.Lanes.Add(new CircularLane(new Vec2(105, -30), 20, deg(0), deg(-181), false, W));
            // e -> f   (center [70,-30]; lane 0 is the OUTER radius here)
            t.Lanes.Add(new CircularLane(new Vec2(70, -30), 20, deg(0), deg(136), true, W));
            t.Lanes.Add(new CircularLane(new Vec2(70, -30), 15, deg(0), deg(137), true, W));
            // f -> g
            t.Lanes.Add(new StraightLane(new Vec2(55.7, -15.7), new Vec2(35.7, -35.7), W));
            t.Lanes.Add(new StraightLane(new Vec2(59.23553, -19.23553), new Vec2(39.23553, -39.23553), W));
            // g -> h   (center [18.1,-18.1])
            t.Lanes.Add(new CircularLane(new Vec2(18.1, -18.1), 25, deg(315), deg(170), false, W));
            t.Lanes.Add(new CircularLane(new Vec2(18.1, -18.1), 30, deg(315), deg(165), false, W));
            // h -> i
            t.Lanes.Add(new CircularLane(new Vec2(18.1, -18.1), 25, deg(170), deg(56), false, W));
            t.Lanes.Add(new CircularLane(new Vec2(18.1, -18.1), 30, deg(170), deg(58), false, W));
            // i -> a   (center [43.2, 23.4]; lane 0 is the OUTER radius here)
            t.Lanes.Add(new CircularLane(new Vec2(43.2, 23.4), 23.5, deg(240), deg(270), true, W));
            t.Lanes.Add(new CircularLane(new Vec2(43.2, 23.4), 18.5, deg(238), deg(268), true, W));
            return t;
        }

        // highway-env RoadNetwork.get_closest_lane_index with heading
        // (this is what vehicle.on_state_update uses -> defines lat/ang_off)
        public Lane ClosestLane(Vec2 p, double heading)
        {
            Lane best = null;
            double bestDist = double.PositiveInfinity;
            foreach (Lane lane in Lanes)
            {
                double d = lane.DistanceWithHeading(p, heading);
                if (d < bestDist) { bestDist = d; best = lane; }
            }
            return best;
        }

        // Wall clearance — position-based "am I at a wall?". Mirror of
        // race_env._wall_state(): the corridor is PAIRS of parallel lanes
        // (lane 0 then lane 1 of each segment, consecutive in Lanes — true
        // for BuildRacetrackFast and every track_<name>.json) and only the
        // corridor's OUTER edges are walls. Returns the gap [m] between the
        // car BODY edge and the nearest wall (~1.5 centered in a boundary
        // lane, ~4.0 on the divider, <= 0 touching). Gate StuckRecovery
        // with this so a slow standing start mid-road never reverses.
        public double WallClearance(Vec2 p, double vehicleWidth)
        {
            int best = 0;
            double bestScore = double.PositiveInfinity;
            for (int i = 0; i < Lanes.Count; i++)
            {
                double d = Lanes[i].Distance(p);
                if (d < bestScore) { bestScore = d; best = i; }
            }
            int pairBase = best - (best % 2);
            Lane first = Lanes[pairBase];
            Lane last = Lanes[Math.Min(pairBase + 1, Lanes.Count - 1)];
            double sF, latF, sL, latL;
            first.LocalCoordinates(p, out sF, out latF);
            last.LocalCoordinates(p, out sL, out latL);
            double halfCar = vehicleWidth / 2.0;
            double lowClear = (latF + first.Width / 2.0) - halfCar;
            double highClear = (last.Width / 2.0 - latL) - halfCar;
            return Math.Min(lowClear, highClear);
        }

        // lat_off / ang_off of a vehicle, exactly like Vehicle.lane_offset
        public void LaneOffsets(VehicleState v, out double latOff, out double angOff)
        {
            Lane lane = ClosestLane(new Vec2(v.X, v.Y), v.Heading);
            double s, lat;
            lane.LocalCoordinates(new Vec2(v.X, v.Y), out s, out lat);
            latOff = lat;
            angOff = HwMath.WrapToPi(v.Heading - lane.HeadingAt(s));
        }
    }

    // ------------------------------------------------- occupancy grid builder
    public sealed class OccupancyGridBuilder
    {
        // ---- constants copied from race_env.RacetrackFast.default_config ----
        public const int CellsX = 12, CellsY = 12, FeatureCount = 10;
        public const int TensorLength = FeatureCount * CellsX * CellsY;   // 1440
        const double GridMinX = -9.0, GridMinY = -18.0;                    // grid_size
        const double StepX = 3.0, StepY = 3.0;                             // grid_step
        const double XRange = 100.0, YRange = 100.0;                       // features_range
        const double VxRange = 20.0, VyRange = 20.0;
        const double LatOffRange = 4.0;
        const double AngOffRange = 3.14159;      // config uses this literal, not PI
        const double LanePerceptionDistance = 100.0;                       // on_road layer
        const double LaneWaypointSpacing = 3.0;                            // min(grid_step)

        // feature/channel indices
        public const int F_PRESENCE = 0, F_ON_ROAD = 1, F_X = 2, F_Y = 3,
                         F_VX = 4, F_VY = 5, F_COS_H = 6, F_SIN_H = 7,
                         F_LAT_OFF = 8, F_ANG_OFF = 9;

        private readonly Track _track;

        public OccupancyGridBuilder(Track track) { _track = track; }

        public float[] Build(VehicleState ego, IList<VehicleState> npcs)
        {
            float[] grid = new float[TensorLength];
            BuildInto(grid, ego, npcs);
            return grid;
        }

        // Allocation-free variant for per-frame use. The flat array is HWC
        // (channels-last): feed it to a model input of shape [1, 12, 12, 10].
        public void BuildInto(float[] grid, VehicleState ego, IList<VehicleState> npcs)
        {
            if (grid.Length != TensorLength)
                throw new ArgumentException("grid must have length " + TensorLength);
            Array.Clear(grid, 0, grid.Length);   // empty cells = 0 (numpy nan_to_num)

            double cosH = Math.Cos(ego.Heading), sinH = Math.Sin(ego.Heading);

            // ---- on_road layer: centerline waypoint trace (NOT a filled mask)
            Vec2 egoPos = new Vec2(ego.X, ego.Y);
            foreach (Lane lane in _track.Lanes)
            {
                double origin, latUnused;
                lane.LocalCoordinates(egoPos, out origin, out latUnused);
                double start = origin - LanePerceptionDistance;
                double stop = origin + LanePerceptionDistance;
                for (double s = start; s < stop; s += LaneWaypointSpacing)
                {
                    double sc = s; // numpy .clip(0, length)
                    if (sc < 0.0) sc = 0.0;
                    if (sc > lane.Length) sc = lane.Length;
                    Vec2 wp = lane.PositionAt(sc, 0.0);
                    int ix, iy;
                    if (PosToIndex(wp.X - ego.X, wp.Y - ego.Y, cosH, sinH, out ix, out iy))
                        grid[(ix * CellsY + iy) * FeatureCount + F_ON_ROAD] = 1f;
                }
            }

            // ---- vehicle layers: NPCs first, ego LAST so ego wins its own cell
            for (int i = 0; i < npcs.Count; i++)
                WriteVehicle(grid, npcs[i], ego, cosH, sinH);
            WriteVehicle(grid, ego, ego, cosH, sinH);
        }

        private void WriteVehicle(float[] grid, VehicleState v, VehicleState ego,
                                  double cosH, double sinH)
        {
            // to_dict(origin_vehicle=ego): x,y,vx,vy are ego-relative in
            // TRACK axes; cos_h/sin_h/lat_off/ang_off are the vehicle's own.
            double dx = v.X - ego.X, dy = v.Y - ego.Y;
            double dvx = v.Vx - ego.Vx, dvy = v.Vy - ego.Vy;

            int ix, iy;
            if (!PosToIndex(dx, dy, cosH, sinH, out ix, out iy))
                return;                                   // outside the grid

            double latOff, angOff;
            _track.LaneOffsets(v, out latOff, out angOff);

            // HWC: all 10 features of one cell are contiguous
            int cell = (ix * CellsY + iy) * FeatureCount;
            grid[cell + F_PRESENCE] = 1f;
            grid[cell + F_X] = HwMath.Clamp1(HwMath.Lmap(dx, -XRange, XRange, -1, 1));
            grid[cell + F_Y] = HwMath.Clamp1(HwMath.Lmap(dy, -YRange, YRange, -1, 1));
            grid[cell + F_VX] = HwMath.Clamp1(HwMath.Lmap(dvx, -VxRange, VxRange, -1, 1));
            grid[cell + F_VY] = HwMath.Clamp1(HwMath.Lmap(dvy, -VyRange, VyRange, -1, 1));
            grid[cell + F_COS_H] = HwMath.Clamp1(Math.Cos(v.Heading));
            grid[cell + F_SIN_H] = HwMath.Clamp1(Math.Sin(v.Heading));
            grid[cell + F_LAT_OFF] = HwMath.Clamp1(HwMath.Lmap(latOff, -LatOffRange, LatOffRange, -1, 1));
            grid[cell + F_ANG_OFF] = HwMath.Clamp1(HwMath.Lmap(angOff, -AngOffRange, AngOffRange, -1, 1));
        }

        // pos_to_index with align_to_vehicle_axes=True: rotate the RELATIVE
        // position into the ego frame, then bin.
        private static bool PosToIndex(double dx, double dy, double cosH, double sinH,
                                       out int ix, out int iy)
        {
            double xr = cosH * dx + sinH * dy;
            double yr = -sinH * dx + cosH * dy;
            ix = (int)Math.Floor((xr - GridMinX) / StepX);
            iy = (int)Math.Floor((yr - GridMinY) / StepY);
            return ix >= 0 && ix < CellsX && iy >= 0 && iy < CellsY;
        }

        // INVERSE of PosToIndex, for debug-drawing the grid: world-frame
        // centre of cell (ix, iy). The grid's +x axis IS the ego heading —
        // ix grows toward where the car FACES (-9 m behind .. +27 m ahead),
        // iy grows toward the car's LEFT. If your overlay shows the 27 m
        // preview pointing world-east ("right") regardless of the car's
        // heading, your draw code skipped this ego rotation; if it is
        // mirrored/rotated the wrong way, it used the world->ego rotation
        // (PosToIndex above) instead of this ego->world one — note the
        // transposed signs on sinH.
        public static void CellCenterWorld(int ix, int iy, VehicleState ego,
                                           out double wx, out double wy)
        {
            double lx = GridMinX + (ix + 0.5) * StepX;   // forward, ego frame
            double ly = GridMinY + (iy + 0.5) * StepY;   // left,    ego frame
            double c = Math.Cos(ego.Heading), s = Math.Sin(ego.Heading);
            wx = ego.X + c * lx - s * ly;
            wy = ego.Y + s * lx + c * ly;
        }

        // Reorder HWC (this builder's native layout) into CHW
        // [C=10, H=CellsX, W=CellsY] for channels-first consumers — e.g. the
        // notebook's torch.onnx export, whose "obs" input is [1, 10, 12, 12].
        // WARNING: pick the layout your inference API actually expects;
        // feeding one layout into the other's input scrambles the grid.
        public static void HwcToChw(float[] hwc, float[] chw)
        {
            for (int f = 0; f < FeatureCount; f++)
                for (int ix = 0; ix < CellsX; ix++)
                    for (int iy = 0; iy < CellsY; iy++)
                        chw[(f * CellsX + ix) * CellsY + iy] =
                            hwc[(ix * CellsY + iy) * FeatureCount + f];
        }
    }

    // ------------------------------------------------ per-frame update driver
    // Owns the observation buffers and the 5 Hz policy cadence.
    //
    // Create ONCE at level load — the Track (lane geometry) is static data,
    // built a single time via Track.BuildRacetrackFast() or Track.FromData();
    // Update() only READS it, nothing is re-created per frame.
    //
    // Typical game loop (Unity-style pseudo-code):
    //
    //   Track track = Track.BuildRacetrackFast();          // once
    //   ObservationUpdater updater = new ObservationUpdater(track);
    //
    //   void FixedUpdate() {
    //       VehicleState ego = new VehicleState(
    //           tf.position.x, tf.position.z,               // track x,y
    //           rb.velocity.x, rb.velocity.z,
    //           Math.Atan2(tf.forward.z, tf.forward.x));
    //       npcBuffer.Clear();
    //       foreach (var car in otherCars)                  // transforms of NPCs
    //           npcBuffer.Add(new VehicleState(...same fields...));
    //
    //       if (updater.Update(Time.fixedDeltaTime, ego, npcBuffer)) {
    //           onnx.Run(updater.ObsHWC);                   // [1,12,12,10]
    //           ActionDecoder.Decode(onnx.Output, out accel, out steer);
    //       }
    //       ApplyAction(accel, steer);   // hold latest action every tick
    //   }
    public sealed class ObservationUpdater
    {
        public const double PolicyPeriod = 0.2;   // policy_frequency = 5 Hz

        private readonly OccupancyGridBuilder _builder;
        private readonly float[] _hwc = new float[OccupancyGridBuilder.TensorLength];
        private readonly float[] _chw = new float[OccupancyGridBuilder.TensorLength];
        private double _timer = PolicyPeriod;     // fire on the first call

        public ObservationUpdater(Track track)
        {
            _builder = new OccupancyGridBuilder(track);
        }

        // Model-ready tensor, channels-last [Height=12 (longitudinal),
        // Width=12 (lateral), Channels=10]. Feed THIS to the model input.
        public float[] ObsHWC { get { return _hwc; } }

        // Same observation reordered channels-first [1, 10, 12, 12] — the
        // layout of the notebook's raw torch.onnx export. See HwcToChw
        // warning before using.
        public float[] ObsCHW { get { return _chw; } }

        // Call every game tick with the CURRENT vehicle states (track frame).
        // Returns true when a fresh observation was produced (every 0.2 s of
        // accumulated game time) — run inference then; hold the previous
        // action on all other ticks.
        public bool Update(double dt, VehicleState ego, IList<VehicleState> npcs)
        {
            _timer += dt;
            if (_timer < PolicyPeriod)
                return false;
            _timer -= PolicyPeriod;   // keep the remainder -> exact 5 Hz average
            Refresh(ego, npcs);
            return true;
        }

        // Rebuild both layouts immediately, ignoring the timer.
        public void Refresh(VehicleState ego, IList<VehicleState> npcs)
        {
            _builder.BuildInto(_hwc, ego, npcs);
            OccupancyGridBuilder.HwcToChw(_hwc, _chw);
        }
    }

    // ------------------------------------------------------- heading helpers
    // The track-frame heading is the angle of the car's forward direction in
    // the (track.x, track.y) plane, atan2 convention: 0 = +x (the a->b
    // straight), +pi/2 = +y, measured COUNTER-clockwise. With the usual
    // Y-up mapping (track.x = world.X, track.y = world.Z):
    //
    //   * From the FORWARD VECTOR (preferred — convention-proof): any
    //     initial model rotation (e.g. the ego spawning with yRotation=90
    //     to face a->b) is already baked into the vector, nothing to add.
    //   * From a WORLD MATRIX: the forward basis vector is the local +Z
    //     axis transformed to world; row-major/row-vector layout
    //     (System.Numerics, DirectX) puts it in row 3 = (M31, M32, M33).
    //   * From EULER ANGLES: engine yaw is measured CLOCKWISE from +Z
    //     (viewed from above), while heading is counter-clockwise from +X —
    //     so heading = 90° - yaw. A car with yRotation = 90 faces a->b:
    //     heading = 0. Feeding raw yaw as heading rotates every observation
    //     by ~90° (the classic "grid points sideways" symptom).
    public static class Heading
    {
        // forward = the car's forward unit vector in world space (X, Z used)
        public static double FromForward(double forwardX, double forwardZ)
        {
            return Math.Atan2(forwardZ, forwardX);
        }

        // m31/m33 = X and Z of the matrix row (or column, if column-major)
        // holding the transformed local +Z axis
        public static double FromWorldMatrix(double m31, double m33)
        {
            return Math.Atan2(m33, m31);
        }

        // yawDegrees = Euler Y rotation, clockwise-from-+Z convention
        public static double FromEulerYawDegrees(double yawDegrees)
        {
            return HwMath.WrapToPi((90.0 - yawDegrees) * Math.PI / 180.0);
        }
    }

    // ------------------------------------------------- deployment assists
    // Steering low-pass. The deterministic policy dithers the steering
    // command around center (the 3 m occupancy-grid quantisation makes its
    // input jump between cells), which shows as shaking on straights.
    // An EMA on the DECODED steering removes the dither without retraining.
    // Measured in the training env: Alpha 0.6 cuts steering sign-flips from
    // 34 to 12 per 100 steps and improves centering on the straights
    // (mean |lat| 1.8 -> 1.3 m) at identical lap count; 0.75 smooths
    // further (8 flips/100) and is still lap-neutral.
    // Call Smooth() once per policy tick (5 Hz); Reset() on respawn.
    public sealed class SteeringSmoother
    {
        public double Alpha = 0.6;   // 0 = off; higher = smoother but laggier
        private double _y;

        public double Smooth(double steeringRad)
        {
            _y = Alpha * _y + (1.0 - Alpha) * steeringRad;
            return _y;
        }

        public void Reset() { _y = 0.0; }
    }

    // Stuck detection + reverse indicator — MOVEMENT-BASED, no lane
    // geometry needed. While the policy commands forward, the car must
    // cover at least MinMoveDistance within WindowSeconds; if it doesn't,
    // it is pinned (wall, obstacle) and Reversing is raised for
    // ReverseSeconds so the caller overrides the action with a straight
    // reverse. A standing start never triggers: even a sluggish 1 m/s^2
    // launch covers 0.5 m in the first second, which resets the window.
    //
    //   var recovery = new StuckRecovery();
    //   ...per policy tick, after ActionDecoder.Decode(...):
    //   if (recovery.Update(0.2, ego.X, ego.Y, accel)) {
    //       accel = recovery.ReverseAccel;   // back out
    //       steer = 0.0;                     // keep wheels straight
    //   }
    //
    // Position is any consistent ground-plane coordinate pair in METERS
    // (world or track frame — only differences are used).
    public sealed class StuckRecovery
    {
        public double MinMoveDistance = 0.3;   // must move this far ... [m]
        public double WindowSeconds   = 1.0;   // ... within this long, while
                                               // commanding forward
        public double ReverseSeconds  = 1.6;   // how long to back up
        public double ReverseAccel    = -3.0;  // override acceleration [m/s^2]

        private bool _windowActive;
        private double _windowTime, _startX, _startY;
        private double _reverseLeft;

        public bool Reversing { get { return _reverseLeft > 0.0; } }

        // Call every policy tick with the tick duration, the car position
        // [m] and the acceleration the policy commanded (pre-override).
        // True while the caller should OVERRIDE the action with reverse.
        public bool Update(double dt, double posX, double posY,
                           double commandedAccel)
        {
            if (_reverseLeft > 0.0)
            {
                _reverseLeft -= dt;
                if (_reverseLeft <= 0.0) _windowActive = false;
                return _reverseLeft > 0.0;
            }
            if (commandedAccel <= 0.0)         // not trying to go forward
            {
                _windowActive = false;
                return false;
            }
            if (!_windowActive)                // start measuring from here
            {
                _windowActive = true;
                _windowTime = 0.0;
                _startX = posX; _startY = posY;
                return false;
            }
            _windowTime += dt;
            double dx = posX - _startX, dy = posY - _startY;
            if (dx * dx + dy * dy >= MinMoveDistance * MinMoveDistance)
            {
                _windowTime = 0.0;             // progressing — slide the window
                _startX = posX; _startY = posY;
                return false;
            }
            if (_windowTime >= WindowSeconds)  // commanded forward, went nowhere
            {
                _reverseLeft = ReverseSeconds;
                _windowActive = false;
                return true;
            }
            return false;
        }

        public void Reset() { _windowActive = false; _reverseLeft = 0.0; }
    }

    // Car-dynamics limits for the REAL game vehicle. The sim car steers
    // instantly and corners at full grip; an engine car cannot. Two guards,
    // applied AFTER SteeringSmoother, BEFORE the StuckRecovery override:
    //   1. steering RATE limit — the wheel moves at most MaxSteerRate rad/s
    //      toward the commanded angle (no instant lock-to-lock snaps).
    //   2. cornering brake — bicycle-model curvature k = |tan(steer)| / Wheelbase
    //      implies lateral acceleration v^2 * k. Above the grip budget
    //      (MaxLateralAccel) the limiter lifts the throttle and applies a
    //      SMALL brake (BrakeAccel), so the car slows into tight turns
    //      instead of understeering into the wall.
    // Defaults: full lock (30 deg) caps cornering at ~6.8 m/s; a gentle
    // 0.1 rad command doesn't brake below ~16 m/s — straights unaffected.
    public sealed class CorneringLimiter
    {
        public double Wheelbase        = 4.5;  // [m] match the game car
        public double MaxSteerRate     = 2.0;  // [rad/s] ~0.5 s lock-to-lock
        public double MaxLateralAccel  = 6.0;  // [m/s^2] grip budget
        // The limiter slows the car INTO corners but can never stop it:
        // vLimit is floored at MinCorneringSpeed, and the brake force is
        // PROPORTIONAL to the overspeed (BrakeGain per m/s over, capped at
        // MaxBrakeAccel) — so it triggers on any overspeed yet fades to zero
        // as the limit is reached. Below vLimit the policy's throttle passes
        // through untouched (no lift zone), so the car re-accelerates out of
        // the corner instead of decaying to a stop under game drag.
        public double MinCorneringSpeed = 5.0; // [m/s] hard floor for vLimit
        public double BrakeGain         = 1.0; // [1/s] brake per m/s overspeed
        public double MaxBrakeAccel     = -3.0;// [m/s^2] strongest brake

        private double _steer;                 // current rate-limited wheel angle

        // Call once per policy tick with the SMOOTHED command; modifies
        // accel/steer in place. speed in m/s; steer in RADIANS (feeding
        // degrees makes Tan() explode and drags the car to a standstill).
        public void Apply(double dt, double speed, ref double accel, ref double steer)
        {
            double maxDelta = MaxSteerRate * dt;
            double delta = steer - _steer;
            if (delta > maxDelta) delta = maxDelta;
            if (delta < -maxDelta) delta = -maxDelta;
            _steer += delta;
            steer = _steer;

            double k = Math.Abs(Math.Tan(_steer)) / Wheelbase;
            if (k <= 1e-6) return;

            double vLimit = Math.Sqrt(MaxLateralAccel / k);
            if (vLimit < MinCorneringSpeed) vLimit = MinCorneringSpeed;

            double over = speed - vLimit;
            if (over > 0.0)
            {
                double brake = -BrakeGain * over;      // gentle near the limit
                if (brake < MaxBrakeAccel) brake = MaxBrakeAccel;
                accel = Math.Min(accel, brake);
            }
        }

        public void Reset() { _steer = 0.0; }
    }

    // ------------------------------------------------------- action decoding
    public static class ActionDecoder
    {
        // Keep in sync with race_env config:
        //   "acceleration_range": [-5.0, 5.0], "steering_range": [-pi/6, pi/6]
        public const double AccelMin = -5.0, AccelMax = 5.0;
        public const double SteerMax = Math.PI / 6.0;

        // actionMean = raw ONNX "action_mean" output (UNCLIPPED Gaussian mean)
        public static void Decode(float[] actionMean,
                                  out double accelerationMs2, out double steeringRad)
        {
            double a0 = Math.Max(-1.0, Math.Min(1.0, actionMean[0]));
            double a1 = Math.Max(-1.0, Math.Min(1.0, actionMean[1]));
            accelerationMs2 = HwMath.Lmap(a0, -1, 1, AccelMin, AccelMax);
            steeringRad = a1 * SteerMax;   // front-wheel angle
        }
    }
}
