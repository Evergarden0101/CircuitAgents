using System;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// Reconstructs the OccupancyGrid observation for the racetrack-v0 model.
/// Config: features=["presence","on_road"], grid_size=[[-18,18],[-18,18]],
///         grid_step=[3,3], align_to_vehicle_axes=True
///
/// ONNX input shape: [1, 2, 12, 12]  (NCHW: batch, channels, height, width)
///   channel 0 = presence (1 if vehicle in cell, else 0)
///   channel 1 = on_road  (1 if cell center is on any lane, else 0)
///
/// ONNX output shape: [1, 2]
///   output[0] = throttle ∈ [-1,1]  → maps to accel ∈ [-3, 5] m/s²
///   output[1] = steering ∈ [-1,1]  → maps to angle ∈ [-0.35, 0.35] rad
/// </summary>
public class OccupancyGridObserver
{
    // ── Must match training config exactly ──────────────────────
    private const int F = 2;          // channels: presence, on_road
    private const int GRID_N = 12;    // cells per axis (= 36m / 3m)
    private const float STEP = 3f;    // meters per cell
    private const float GRID_MIN = -18f; // grid_size min (same on both axes)

    // Mapped action ranges  (match your training action config)
    private const float ACCEL_MIN = -3f;  private const float ACCEL_MAX = 5f;
    private const float STEER_MIN = -0.35f; private const float STEER_MAX = 0.35f;

    private readonly InferenceSession _session;

    public OccupancyGridObserver(string onnxPath)
    {
        _session = new InferenceSession(onnxPath);
    }

    // ────────────────────────────────────────────────────────────
    // Call this every policy tick (5 Hz).
    // Returns (acceleration m/s², steeringAngle rad)
    // ────────────────────────────────────────────────────────────
    public (float accel, float steer) Infer(
        Vector2 egoWorldPos,
        float   egoHeadingRad,
        IReadOnlyList<Vector2> npcWorldPositions,
        IRoadNetwork road)
    {
        float[,,] grid = BuildGrid(egoWorldPos, egoHeadingRad, npcWorldPositions, road);

        // Flatten [C, H, W] → float[] in C-order (channel-first)
        // ONNX expects NCHW: [1, 2, 12, 12]
        float[] flat = FlattenGrid(grid);

        // Build input tensor
        var inputTensor = new DenseTensor<float>(flat, new[] { 1, F, GRID_N, GRID_N });
        var inputs = new[] { NamedOnnxValue.CreateFromTensor("obs", inputTensor) };

        using var results = _session.Run(inputs);
        float[] raw = results[0].AsEnumerable<float>().ToArray();
        // raw[0] = throttle ∈ [-1,1], raw[1] = steering ∈ [-1,1]

        float accel = Lmap(raw[0], -1f, 1f, ACCEL_MIN, ACCEL_MAX);
        float steer = Lmap(raw[1], -1f, 1f, STEER_MIN, STEER_MAX);
        return (accel, steer);
    }

    // ────────────────────────────────────────────────────────────
    // Grid builder
    // grid[c, i, j]:
    //   c=0  presence: 1 if any NPC or ego is in cell (i,j)
    //   c=1  on_road : 1 if cell center is on a road lane
    //   i = longitudinal cell index (vehicle forward axis)
    //   j = lateral cell index      (vehicle left/right axis)
    // ────────────────────────────────────────────────────────────
    private float[,,] BuildGrid(
        Vector2 egoPos,
        float   heading,
        IReadOnlyList<Vector2> npcs,
        IRoadNetwork road)
    {
        var grid = new float[F, GRID_N, GRID_N];

        float cosH = MathF.Cos(heading);
        float sinH = MathF.Sin(heading);

        // ── Pass 1: on_road layer ────────────────────────────
        // Sample each cell centre; mark if it falls within any lane.
        for (int i = 0; i < GRID_N; i++)
        for (int j = 0; j < GRID_N; j++)
        {
            // Cell centre in vehicle frame (forward, lateral)
            float fwd = GRID_MIN + (i + 0.5f) * STEP;
            float lat = GRID_MIN + (j + 0.5f) * STEP;

            // Rotate from vehicle frame to world frame
            Vector2 worldPos = egoPos + new Vector2(
                fwd * cosH - lat * sinH,
                fwd * sinH + lat * cosH
            );

            grid[1, i, j] = road.IsOnRoad(worldPos) ? 1f : 0f;
        }

        // ── Pass 2: presence layer ───────────────────────────
        // Mark cells occupied by ANY vehicle (ego counts too — it's always
        // at the grid centre cell, but mark it anyway for correctness).
        void MarkVehicle(Vector2 worldPos)
        {
            (int ci, int cj) = WorldToCell(worldPos, egoPos, cosH, sinH);
            if (ci >= 0 && ci < GRID_N && cj >= 0 && cj < GRID_N)
                grid[0, ci, cj] = 1f;
        }

        MarkVehicle(egoPos);
        foreach (var npc in npcs)
            MarkVehicle(npc);

        return grid;
    }

    // ────────────────────────────────────────────────────────────
    // Convert world position → grid cell (i=longitudinal, j=lateral)
    // align_to_vehicle_axes = True: rotate world offset into vehicle frame first
    // ────────────────────────────────────────────────────────────
    private static (int i, int j) WorldToCell(
        Vector2 worldPos,
        Vector2 egoPos,
        float cosH, float sinH)
    {
        float dx = worldPos.X - egoPos.X;
        float dy = worldPos.Y - egoPos.Y;

        // Rotate world-relative offset into vehicle frame
        // (same rotation matrix as HighwayEnv pos_to_index)
        float fwd = dx * cosH + dy * sinH;   // longitudinal (vehicle X)
        float lat = -dx * sinH + dy * cosH;  // lateral      (vehicle Y)

        int i = (int)MathF.Floor((fwd - GRID_MIN) / STEP);
        int j = (int)MathF.Floor((lat - GRID_MIN) / STEP);
        return (i, j);
    }

    // ────────────────────────────────────────────────────────────
    // Flatten float[C, H, W] → float[] in channel-first (NCHW) row-major order
    // ────────────────────────────────────────────────────────────
    private static float[] FlattenGrid(float[,,] grid)
    {
        var flat = new float[F * GRID_N * GRID_N];
        for (int c = 0; c < F; c++)
        for (int h = 0; h < GRID_N; h++)
        for (int w = 0; w < GRID_N; w++)
            flat[c * GRID_N * GRID_N + h * GRID_N + w] = grid[c, h, w];
        return flat;
    }

    // Linear map: value in [inMin,inMax] → [outMin,outMax]
    private static float Lmap(float v, float inMin, float inMax, float outMin, float outMax)
        => outMin + (v - inMin) * (outMax - outMin) / (inMax - inMin);
}