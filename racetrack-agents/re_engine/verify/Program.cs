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
