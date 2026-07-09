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
// NOTE: the notebook's torch.onnx export is channels-FIRST [1, 10, 12, 12];
// this builder emits the HWC sequence the engine-side model input expects.
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
        public double Vx, Vy;      // velocity [m/s] (world/track axes)
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
