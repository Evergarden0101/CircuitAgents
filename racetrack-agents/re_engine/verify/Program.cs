// Verification harness: builds observations with the C# OccupancyGridBuilder
// and compares them cell-by-cell against reference_obs.json dumped from the
// real highway-env environment. Not needed in the game.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using RacetrackAgents;

class RefVehicle { public double x, y, vx, vy, heading; }
class RefScenario
{
    public string name;
    public RefVehicle ego;
    public RefVehicle[] npcs;
    public int[] obs_shape;
    public float[] obs;
}
class RefFile { public RefScenario[] scenarios; }

static class Program
{
    static VehicleState ToState(RefVehicle v)
    {
        return new VehicleState(v.x, v.y, v.vx, v.vy, v.heading);
    }

    static int Main(string[] args)
    {
        string dir = args.Length > 0 ? args[0] : "..";
        var opts = new JsonSerializerOptions { IncludeFields = true };

        var trackData = JsonSerializer.Deserialize<TrackData>(
            File.ReadAllText(Path.Combine(dir, "track.json")), opts);
        var reference = JsonSerializer.Deserialize<RefFile>(
            File.ReadAllText(Path.Combine(dir, "reference_obs.json")), opts);

        var tracks = new (string label, Track track)[]
        {
            ("json-loaded", Track.FromData(trackData)),
            ("hardcoded  ", Track.BuildRacetrackFast()),
        };

        bool allPass = true;
        foreach (var (label, track) in tracks)
        {
            var builder = new OccupancyGridBuilder(track);
            Console.WriteLine($"--- track source: {label} ---");
            foreach (var sc in reference.scenarios)
            {
                var npcs = new VehicleState[sc.npcs.Length];
                for (int i = 0; i < npcs.Length; i++) npcs[i] = ToState(sc.npcs[i]);
                float[] obs = builder.Build(ToState(sc.ego), npcs);   // HWC

                // reference_obs.json is dumped CHW from python; the builder
                // now emits HWC — compare via hwc[(ix*12+iy)*10+f] vs
                // chw[(f*12+ix)*12+iy]
                double maxDiff = 0;
                int worst = -1;   // HWC index of the worst cell
                for (int f = 0; f < 10; f++)
                    for (int ix = 0; ix < 12; ix++)
                        for (int iy = 0; iy < 12; iy++)
                        {
                            int iHwc = (ix * 12 + iy) * 10 + f;
                            int iChw = (f * 12 + ix) * 12 + iy;
                            double d = Math.Abs(obs[iHwc] - sc.obs[iChw]);
                            if (d > maxDiff) { maxDiff = d; worst = iHwc; }
                        }
                bool pass = maxDiff <= 1e-5;

                // CHW reorder round-trip: converter output must equal the
                // python CHW reference element-for-element
                float[] chw = new float[obs.Length];
                OccupancyGridBuilder.HwcToChw(obs, chw);
                bool chwOk = pass;
                for (int i = 0; i < chw.Length && chwOk; i++)
                    if (Math.Abs(chw[i] - sc.obs[i]) > 1e-5) chwOk = false;
                pass &= chwOk;
                allPass &= pass;
                string detail = pass ? "" :
                    $"  worst hwc idx {worst} (cell {worst / 120},{(worst / 10) % 12}, feature {worst % 10}): " +
                    $"c#={obs[worst]:G9}" + (chwOk ? "" : "  [CHW reorder BROKEN]");
                Console.WriteLine($"  {(pass ? "PASS" : "FAIL")}  {sc.name,-28} maxDiff={maxDiff:E2} chw={(chwOk ? "ok" : "BAD")}{detail}");
            }
        }
        // ---- single-file builder (HighwayObservationBuilder.cs) ----
        // Lanes are now constructed OUTSIDE the builder: once from the
        // built-in preset, once from track.json via the Lane factories
        // (points for straights; center/radius/phases[rad] for arcs).
        var jsonLanes = new List<RacetrackSingle.HighwayObservationBuilder.Lane>();
        foreach (var ld in trackData.lanes)
            jsonLanes.Add(ld.type == "straight"
                ? RacetrackSingle.HighwayObservationBuilder.Lane.Straight(
                      ld.start[0], ld.start[1], ld.end[0], ld.end[1])
                : RacetrackSingle.HighwayObservationBuilder.Lane.Arc(
                      ld.center[0], ld.center[1], ld.radius,
                      ld.start_phase, ld.end_phase, ld.clockwise));

        var singles = new (string label, RacetrackSingle.HighwayObservationBuilder builder)[]
        {
            ("preset lanes", new RacetrackSingle.HighwayObservationBuilder(
                RacetrackSingle.HighwayObservationBuilder.RacetrackFastLanes())),
            ("json lanes  ", new RacetrackSingle.HighwayObservationBuilder(jsonLanes)),
        };
        foreach (var (label, single) in singles)
        {
            Console.WriteLine($"--- single-file HighwayObservationBuilder ({label}) ---");
            foreach (var sc in reference.scenarios)
            {
                var npcs = new List<RacetrackSingle.HighwayObservationBuilder.Vehicle>();
                foreach (var n in sc.npcs) npcs.Add(ToSingle(n));
                float[] obs = single.BuildObservation(ToSingle(sc.ego), npcs);   // HWC

                double maxDiff = 0;
                for (int f = 0; f < 10; f++)
                    for (int ix = 0; ix < 12; ix++)
                        for (int iy = 0; iy < 12; iy++)
                            maxDiff = Math.Max(maxDiff, Math.Abs(
                                obs[(ix * 12 + iy) * 10 + f] - sc.obs[(f * 12 + ix) * 12 + iy]));
                bool pass = maxDiff <= 1e-5;
                allPass &= pass;
                Console.WriteLine($"  {(pass ? "PASS" : "FAIL")}  {sc.name,-28} maxDiff={maxDiff:E2}");
            }
        }

        // ---- track preset parity: C# preset lanes vs python-exported JSON ----
        // Guards against typos in the hardcoded presets: every lane's
        // centerline is sampled along its length and compared to the lane
        // built from track_<name>.json (exported by export_track_json.py).
        Console.WriteLine("--- track preset geometry parity ---");
        var presets = new (string name, RacetrackSingle.HighwayObservationBuilder.Lane[] lanes)[]
        {
            ("fast",    RacetrackSingle.HighwayObservationBuilder.RacetrackFastLanes()),
            ("oval",    RacetrackSingle.HighwayObservationBuilder.OvalLanes()),
            ("stadium", RacetrackSingle.HighwayObservationBuilder.StadiumLanes()),
            ("rect",    RacetrackSingle.HighwayObservationBuilder.RectLanes()),
            ("chicane", RacetrackSingle.HighwayObservationBuilder.ChicaneLanes()),
        };
        foreach (var (name, preset) in presets)
        {
            string path = Path.Combine(dir, $"track_{name}.json");
            if (!File.Exists(path))
            {
                Console.WriteLine($"  SKIP  {name,-8} ({path} missing — run export_track_json.py)");
                continue;
            }
            var td = JsonSerializer.Deserialize<TrackData>(File.ReadAllText(path), opts);
            bool pass = td.lanes.Length == preset.Length;
            double worst = 0;
            if (pass)
            {
                for (int i = 0; i < preset.Length; i++)
                {
                    var ld = td.lanes[i];
                    var refLane = ld.type == "straight"
                        ? RacetrackSingle.HighwayObservationBuilder.Lane.Straight(
                              ld.start[0], ld.start[1], ld.end[0], ld.end[1])
                        : RacetrackSingle.HighwayObservationBuilder.Lane.Arc(
                              ld.center[0], ld.center[1], ld.radius,
                              ld.start_phase, ld.end_phase, ld.clockwise);
                    for (int k = 0; k <= 4; k++)
                    {
                        double s = refLane.Length * k / 4.0;
                        refLane.Point(s, out double rx, out double ry);
                        preset[i].Point(s, out double px, out double py);
                        double d = Math.Sqrt((rx - px) * (rx - px) + (ry - py) * (ry - py));
                        worst = Math.Max(worst, d);
                        worst = Math.Max(worst, Math.Abs(refLane.Length - preset[i].Length));
                    }
                }
                pass = worst <= 1e-4;
            }
            allPass &= pass;
            Console.WriteLine($"  {(pass ? "PASS" : "FAIL")}  {name,-8} lanes={preset.Length} maxDiff={worst:E2}");
        }

        // ---- WallClearance parity: known geometry on the a->b straight ----
        // (car width 2.0, matching training: lane-0 centre -> 1.5 m free,
        // divider -> 4.0 m, 0.7 m past either wall -> -0.7)
        Console.WriteLine("--- wall clearance (car width 2.0) ---");
        var wcTrack = Track.BuildRacetrackFast();
        var wcSingle = new RacetrackSingle.HighwayObservationBuilder(
            RacetrackSingle.HighwayObservationBuilder.RacetrackFastLanes());
        (double y, double expect)[] wcCases = {
            (0.0, 1.5), (2.5, 4.0), (5.0, 1.5), (7.2, -0.7), (-2.2, -0.7),
        };
        foreach (var (y, expect) in wcCases)
        {
            double a = wcTrack.WallClearance(new Vec2(70, y), 2.0);
            double b = wcSingle.WallClearance(70, y, 2.0);
            bool ok = Math.Abs(a - expect) < 1e-9 && Math.Abs(b - expect) < 1e-9;
            allPass &= ok;
            Console.WriteLine($"  {(ok ? "PASS" : "FAIL")}  y={y,5:F1} -> track={a:F2} single={b:F2} (expect {expect:F2})");
        }

        // ---- CarController physics parity vs highway-env BicycleVehicle ----
        // Reference generated in python: start (10, 5, heading .3, speed 8),
        // 15 steps accel 4, 20 steps (2, steer .3), 10 steps (-3, -.2), dt 1/15.
        Console.WriteLine("--- CarController vs BicycleVehicle ---");
        var car = new RacetrackSingle.CarController();
        car.SetPose(10.0, 5.0, 0.3, 8.0);
        double carDt = 1.0 / 15.0;
        for (int i = 0; i < 15; i++) car.Step(carDt, 4.0, 0.0);
        for (int i = 0; i < 20; i++) car.Step(carDt, 2.0, 0.3);
        for (int i = 0; i < 10; i++) car.Step(carDt, -3.0, -0.2);
        double[] got = { car.X, car.Y, car.Heading, car.Speed,
                         car.Velocity.X, car.Velocity.Z };
        double[] want = { 35.252217994, 28.278207476, 1.068805490, 12.666666667,
                          6.061697451, 11.122126504 };
        double carWorst = 0;
        for (int i = 0; i < got.Length; i++)
            carWorst = Math.Max(carWorst, Math.Abs(got[i] - want[i]));
        bool carOk = carWorst < 1e-5;
        allPass &= carOk;
        Console.WriteLine($"  {(carOk ? "PASS" : "FAIL")}  45-step scripted drive maxDiff={carWorst:E2} " +
                          $"(x={car.X:F3} y={car.Y:F3} v=({car.Velocity.X:F3},{car.Velocity.Z:F3}))");

        Console.WriteLine(allPass ? "ALL SCENARIOS MATCH" : "MISMATCH DETECTED");
        return allPass ? 0 : 1;
    }

    static RacetrackSingle.HighwayObservationBuilder.Vehicle ToSingle(RefVehicle v)
    {
        return new RacetrackSingle.HighwayObservationBuilder.Vehicle
        {
            Position = new System.Numerics.Vector3((float)v.x, (float)v.y, 0),
            Velocity = new System.Numerics.Vector3((float)v.vx, (float)v.vy, 0),
            Heading = (float)v.heading,
        };
    }
}
