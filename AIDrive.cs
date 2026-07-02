/// <summary>
/// OccupancyGrid observer — 11 features, 12×12 grid, align_to_vehicle_axes=True
///
/// ONNX input  shape : [1, 11, 12, 12]  (NCHW)
/// ONNX output shape : [1, 2]           → [throttle, steering] both ∈ [-1, 1]
///
/// Channel order matches Python feature list:
///   ch 0  presence   binary (no normalization)
///   ch 1  on_road    binary (no normalization)
///   ch 2  x          lmap([-100,100] → [-1,1])
///   ch 3  y          lmap([-100,100] → [-1,1])
///   ch 4  vx         lmap([-20,20]   → [-1,1])
///   ch 5  vy         lmap([-20,20]   → [-1,1])
///   ch 6  cos_h      already in [-1,1], no lmap
///   ch 7  sin_h      already in [-1,1], no lmap
///   ch 8  long_off   lmap([0,500]    → [-1,1])
///   ch 9  lat_off    lmap([-4,4]     → [-1,1])
///   ch 10 ang_off    lmap([-π,π]     → [-1,1])
/// 
/// F  = 11  (channels)
/// W  = (18 - (-18)) / 3 = 12
/// H  = 12
/// Tensor:      [11, 12, 12]   →  1584 floats
/// ONNX input:  [1, 11, 12, 12]   (NCHW: batch=1, C=11, H=12, W=12)
/// ONNX output: [1, 2]            (throttle, steering in [-1,1])
/// </summary>
public class OccupancyGridObserver11
{
    // ── Grid constants — MUST match training config ──────────────
    private const int F       = 11;
    private const int GRID_N  = 12;         // cells per axis
    private const float STEP  = 3f;         // meters per cell
    private const float GMIN  = -18f;       // grid_size min

    // Action output ranges — match your training action config
    private const float ACCEL_MIN = -3f;   private const float ACCEL_MAX = 5f;
    private const float STEER_MIN = -0.4f; private const float STEER_MAX = 0.4f;

    // ── Per-channel normalization: (rawMin, rawMax)
    //    Binary/trig channels have (0,0) sentinel → skip lmap, pass through
    private static readonly (float min, float max)[] RANGES = {
        (  0f,       0f),   // ch 0  presence   — binary, skip
        (  0f,       0f),   // ch 1  on_road    — binary, skip
        (-100f,    100f),   // ch 2  x
        (-100f,    100f),   // ch 3  y
        ( -20f,     20f),   // ch 4  vx
        ( -20f,     20f),   // ch 5  vy
        (  0f,       0f),   // ch 6  cos_h      — already in [-1,1], skip
        (  0f,       0f),   // ch 7  sin_h      — already in [-1,1], skip
        (  0f,      500f),  // ch 8  long_off   ← match your track circumference
        ( -4f,       4f),   // ch 9  lat_off
        (-(float)Math.PI, (float)Math.PI), // ch 10 ang_off
    };

    // Sentinel: if min==max==0 → skip normalization (already bounded or binary)
    private static bool SkipNorm(int ch) => RANGES[ch].min == 0f && RANGES[ch].max == 0f;

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _outputName;

    public OccupancyGridObserver11(string onnxPath)
    {
        _session    = new InferenceSession(onnxPath);
        _inputName  = _session.InputMetadata.Keys.First();   // typically "obs"
        _outputName = _session.OutputMetadata.Keys.First();  // typically "continuous_actions"
    }

    // ─────────────────────────────────────────────────────────────
    // Main inference entry point — call at 5 Hz (every 3 physics steps)
    // Returns (acceleration [m/s²], steeringAngle [rad])
    // ─────────────────────────────────────────────────────────────
    public (float accel, float steer) Infer(
        VehicleState ego,
        IReadOnlyList<VehicleState> npcs,
        IRoadNetwork road)
    {
        float[] flat = BuildAndFlatten(ego, npcs, road);

        // ONNX tensor: [1, 11, 12, 12] NCHW
        var tensor = new DenseTensor<float>(flat, new[] { 1, F, GRID_N, GRID_N });
        using var results = _session.Run(new[] {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        });

        float[] raw = results[0].AsEnumerable<float>().ToArray();
        // raw[0] = throttle ∈ [-1,1], raw[1] = steering ∈ [-1,1]

        float accel = Lmap(raw[0], -1f, 1f, ACCEL_MIN, ACCEL_MAX);
        float steer = Lmap(raw[1], -1f, 1f, STEER_MIN, STEER_MAX);
        return (accel, steer);
    }

    // ─────────────────────────────────────────────────────────────
    // Build grid[ch, i, j] and flatten to float[] in NCHW order
    //   i = longitudinal (vehicle forward), j = lateral (vehicle left-right)
    //   Vehicle with multiple pixels: we use its centre only (no sub-cell splitting)
    // ─────────────────────────────────────────────────────────────
    private float[] BuildAndFlatten(
        VehicleState ego,
        IReadOnlyList<VehicleState> npcs,
        IRoadNetwork road)
    {
        var grid = new float[F, GRID_N, GRID_N];

        float cosH = MathF.Cos(ego.HeadingRad);
        float sinH = MathF.Sin(ego.HeadingRad);

        // ── Pass 1: on_road channel (ch 1) ────────────────────
        // Sample every cell centre in world space and query road
        for (int i = 0; i < GRID_N; i++)
        for (int j = 0; j < GRID_N; j++)
        {
            // Cell centre in vehicle frame (forward=i-axis, lateral=j-axis)
            float fwd = GMIN + (i + 0.5f) * STEP;
            float lat = GMIN + (j + 0.5f) * STEP;

            // Rotate vehicle→world (inverse of WorldToCell rotation below)
            Vector2 worldPos = ego.WorldPos + new Vector2(
                fwd * cosH - lat * sinH,
                fwd * sinH + lat * cosH
            );
            grid[1, i, j] = road.IsOnRoad(worldPos) ? 1f : 0f;
        }

        // ── Pass 2: all vehicles (ego + npcs) ─────────────────
        void WriteVehicle(VehicleState v)
        {
            // Project world position into ego vehicle frame
            float dx = v.WorldPos.X - ego.WorldPos.X;
            float dy = v.WorldPos.Y - ego.WorldPos.Y;

            // align_to_vehicle_axes = True:
            // rotate world-relative offset into vehicle frame
            float fwd =  dx * cosH + dy * sinH;  // longitudinal
            float lat = -dx * sinH + dy * cosH;  // lateral

            int ci = (int)MathF.Floor((fwd - GMIN) / STEP);
            int cj = (int)MathF.Floor((lat - GMIN) / STEP);

            if ((uint)ci >= GRID_N || (uint)cj >= GRID_N) return; // out of grid

            // Lane offset for this vehicle
            LaneOffset lo = road.GetNearestLaneOffset(v.WorldPos, v.HeadingRad);

            // Write all 11 channels
            float[] raw = {
                1f,                                     // ch 0  presence
                grid[1, ci, cj],                        // ch 1  on_road (already filled)
                v.WorldPos.X - ego.WorldPos.X,          // ch 2  x  (relative world X)
                v.WorldPos.Y - ego.WorldPos.Y,          // ch 3  y  (relative world Y)
                v.Velocity.X,                           // ch 4  vx (absolute world X)
                v.Velocity.Y,                           // ch 5  vy (absolute world Y)
                MathF.Cos(v.HeadingRad),                // ch 6  cos_h
                MathF.Sin(v.HeadingRad),                // ch 7  sin_h
                lo.Longitudinal,                        // ch 8  long_off  [0, track len]
                lo.Lateral,                             // ch 9  lat_off   [m from centre]
                v.HeadingRad - lo.LaneHeadingRad,       // ch 10 ang_off   [rad]
            };

            for (int ch = 0; ch < F; ch++)
                grid[ch, ci, cj] = NormAndClip(ch, raw[ch]);
        }

        WriteVehicle(ego);
        foreach (var npc in npcs)
            WriteVehicle(npc);

        // ── Flatten [F, W, H] → float[] in C-order (channel-first NCHW) ─
        // Index order: flat[ch * GRID_N*GRID_N  +  i * GRID_N  +  j]
        var flat = new float[F * GRID_N * GRID_N];
        for (int ch = 0; ch < F;      ch++)
        for (int i  = 0; i  < GRID_N; i++)
        for (int j  = 0; j  < GRID_N; j++)
            flat[ch * GRID_N * GRID_N + i * GRID_N + j] = grid[ch, i, j];

        return flat;
    }

    // lmap [rawMin,rawMax] → [-1,1], then clip
    private static float NormAndClip(int ch, float raw)
    {
        if (SkipNorm(ch))
            return Math.Clamp(raw, -1f, 1f);   // binary/trig: just clip

        var (mn, mx) = RANGES[ch];
        float n = 2f * (raw - mn) / (mx - mn) - 1f;
        return Math.Clamp(n, -1f, 1f);
    }

    private static float Lmap(float v, float i0, float i1, float o0, float o1)
        => o0 + (v - i0) * (o1 - o0) / (i1 - i0);
}

// ── Data structures ─────────────────────────────────────────────
public struct VehicleState
{
    public Vector2 WorldPos;
    public Vector2 Velocity;
    public float   HeadingRad;
}

public struct LaneOffset
{
    public float Longitudinal;   // distance along lane from lane start [m]
    public float Lateral;        // perpendicular offset from centerline [m]
    public float LaneHeadingRad; // heading of the lane tangent at closest point
}

public interface IRoadNetwork
{
    bool        IsOnRoad(Vector2 worldPos);
    LaneOffset  GetNearestLaneOffset(Vector2 worldPos, float vehicleHeadingRad);
}
