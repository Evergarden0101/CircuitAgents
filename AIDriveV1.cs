
///////////////////////////////////////////////////V1
public class OccupancyGridBuilder
{
    // ── MUST match training config ─────────────────────────────
    const int W = 11, H = 11, F = 11;
    const float STEP_X = 5f, STEP_Y = 5f;
    const float MIN_X = -27.5f, MIN_Y = -27.5f;

    // features_range entries (adjust to match training)
    static readonly (float min, float max)[] RANGES = new (float, float)[F]
    {
        (0f,   1f),     // 0 presence   – binary, no lmap needed
        (0f,   1f),     // 1 on_road    – binary
        (-100f,100f),   // 2 x
        (-100f,100f),   // 3 y
        (-20f, 20f),    // 4 vx
        (-20f, 20f),    // 5 vy
        (-1f,  1f),     // 6 cos_h      – already bounded
        (-1f,  1f),     // 7 sin_h      – already bounded
        (0f,   500f),   // 8 long_off   ← SET YOUR TRACK LENGTH, or (0,1) if dead
        (-4f,  4f),     // 9 lat_off
        (-3.14159f, 3.14159f),  // 10 ang_off
    };

    // Output tensor: grid[f, i, j]  (channel-first, matching ONNX)
    public float[,,] BuildGrid(
        Vehicle ego,
        IEnumerable<Vehicle> otherVehicles,
        RoadNetwork road)
    {
        var grid = new float[F, W, H];

        // ── Pass 1: on_road layer (independent of vehicles) ────
        for (int i = 0; i < W; i++)
        for (int j = 0; j < H; j++)
        {
            Vector2 cellWorldPos = CellToWorld(ego, i, j);
            grid[1, i, j] = road.IsOnRoad(cellWorldPos) ? 1f : 0f;
        }

        // ── Pass 2: vehicles ───────────────────────────────────
        var allVehicles = otherVehicles.Prepend(ego); // ego included
        foreach (var v in allVehicles)
        {
            Vector2 relPos = v.WorldPos - ego.WorldPos;
            (int ci, int cj) = WorldOffsetToCell(relPos);

            if (ci < 0 || ci >= W || cj < 0 || cj >= H) continue;

            LaneOffset laneOff = road.GetNearestLaneOffset(v);

            float[] raw = new float[F]
            {
                1f,                             // 0 presence
                grid[1, ci, cj],                // 1 on_road (already filled)
                relPos.x,                       // 2 x
                relPos.y,                       // 3 y
                v.Velocity.x,                   // 4 vx (world, absolute)
                v.Velocity.y,                   // 5 vy (world, absolute)
                MathF.Cos(v.HeadingRad),        // 6 cos_h
                MathF.Sin(v.HeadingRad),        // 7 sin_h
                laneOff.Longitudinal,           // 8 long_off
                laneOff.Lateral,                // 9 lat_off
                v.HeadingRad - laneOff.LaneHeadingRad,  // 10 ang_off
            };

            for (int f = 0; f < F; f++)
                grid[f, ci, cj] = Normalize(f, raw[f]);
        }

        return grid;
    }

    Vector2 CellToWorld(Vehicle ego, int i, int j)
    {
        float ox = MIN_X + (i + 0.5f) * STEP_X;
        float oy = MIN_Y + (j + 0.5f) * STEP_Y;
        // If align_to_vehicle_axes=True, rotate (ox,oy) by ego.HeadingRad here
        return ego.WorldPos + new Vector2(ox, oy);
    }

    (int, int) WorldOffsetToCell(Vector2 relPos)
    {
        int i = (int)MathF.Floor((relPos.x - MIN_X) / STEP_X);
        int j = (int)MathF.Floor((relPos.y - MIN_Y) / STEP_Y);
        return (i, j);
    }

    static float Normalize(int f, float raw)
    {
        var (min, max) = RANGES[f];
        float n = 2f * (raw - min) / (max - min) - 1f;
        return Math.Clamp(n, -1f, 1f);        // clip
    }
}