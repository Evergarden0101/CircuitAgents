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
                float[] obs = builder.Build(ToState(sc.ego), npcs);

                double maxDiff = 0;
                int worst = -1;
                for (int i = 0; i < obs.Length; i++)
                {
                    double d = Math.Abs(obs[i] - sc.obs[i]);
                    if (d > maxDiff) { maxDiff = d; worst = i; }
                }
                bool pass = maxDiff <= 1e-5;

                // HWC reorder round-trip: hwc[(ix*12+iy)*10+f] == chw[(f*12+ix)*12+iy]
                float[] hwc = new float[obs.Length];
                OccupancyGridBuilder.ChwToHwc(obs, hwc);
                bool hwcOk = true;
                for (int f = 0; f < 10 && hwcOk; f++)
                    for (int ix = 0; ix < 12 && hwcOk; ix++)
                        for (int iy = 0; iy < 12; iy++)
                            if (hwc[(ix * 12 + iy) * 10 + f] != obs[(f * 12 + ix) * 12 + iy])
                            { hwcOk = false; break; }
                pass &= hwcOk;
                allPass &= pass;
                string detail = pass ? "" :
                    $"  worst idx {worst} (feature {worst / 144}, cell {(worst % 144) / 12},{worst % 12}): " +
                    $"c#={obs[worst]:G9} py={sc.obs[worst]:G9}" + (hwcOk ? "" : "  [HWC reorder BROKEN]");
                Console.WriteLine($"  {(pass ? "PASS" : "FAIL")}  {sc.name,-28} maxDiff={maxDiff:E2} hwc={(hwcOk ? "ok" : "BAD")}{detail}");
            }
        }
        // ---- single-file builder (HighwayObservationBuilder.cs) ----
        Console.WriteLine("--- single-file HighwayObservationBuilder ---");
        var single = new RacetrackSingle.HighwayObservationBuilder();
        foreach (var sc in reference.scenarios)
        {
            var npcs = new List<RacetrackSingle.HighwayObservationBuilder.Vehicle>();
            foreach (var n in sc.npcs) npcs.Add(ToSingle(n));
            float[] obs = single.BuildObservation(ToSingle(sc.ego), npcs);

            double maxDiff = 0;
            for (int i = 0; i < obs.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(obs[i] - sc.obs[i]));
            bool pass = maxDiff <= 1e-5;
            allPass &= pass;
            Console.WriteLine($"  {(pass ? "PASS" : "FAIL")}  {sc.name,-28} maxDiff={maxDiff:E2}");
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
