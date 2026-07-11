# CircuitAgents プロジェクト概要

強化学習 (PPO) によるサーキット走行エージェントを、highway-env 上で学習し、
ONNX 経由で C# ゲームエンジン (RE Engine) にデプロイするプロジェクトです。
本書は結果・環境設計・セットアップ・エクスポート手順をまとめた日本語ドキュメントです。

---

## 1. 結果ハイライト

すべて本リポジトリの環境で実測した数値です。

| 項目 | 結果 |
|---|---|
| fast トラック走行 | **37 周 / 800 秒エピソード**、16 m/s (制限速度)、無衝突 |
| 周回能力の獲得 | **3〜6 万ステップ** (6 万で 32 周、9 万で 36 周) |
| fast-dev (12 万ステップ) | フェーズ 2 込みで 37 周 — 短縮設定でも確実に周回 |
| 汎化モデル (random 学習 15 万) | **1 つの方策で 5 トラックすべて** を 2 分間ノーミス走行 (平均 15.7 m/s) |
| ゼロショット転移 | fast のみで学習した方策が他 4 トラックも制限速度で走行 |
| C# 観測ポート | Python 環境と **ビット一致** (maxDiff = 0.0、検証ハーネスで自動確認) |
| C# 車両物理 (CarController) | highway-env BicycleVehicle と **1e-7 一致** |

学習曲線・走行軌跡・5 トラック同時走行の動画は、
`task2` / `task3` ノートブックの Trajectory Plot / Per-Track Videos セルで再現できます。

---

## 2. オリジナル highway-env との違いと理由

ベースは highway-env の racetrack 環境です。まずデフォルト設定を正確に押さえます
(インストール版 `highway_env.envs.racetrack_env.RacetrackEnv` から抽出):

| 項目 | highway-env デフォルト |
|---|---|
| 行動 | **ステアリングのみ** (`longitudinal: False`)、速度は離散 `target_speeds [0, 5, 10]` |
| 観測 | OccupancyGrid **2ch** (presence / on_road)、±18 m の対称視野 |
| 報酬 | 車線中央維持 `1/(1+4·lat²)` + 操作量ペナルティ `−0.3·‖action‖` + 衝突 −1 |
| 速度 | speed_limit 10 m/s — 「速く走る」動機なし |
| 壁 | **存在しない** — 路外は終了 (または放置)、すり抜け可能 |
| 車両 | 運動学 Bicycle (タイヤスリップなし)、NPC 1 台、300 秒 |

車線をなぞる研究には十分ですが、「アクセルを踏み、壁があり、速さを競う」ゲームの
レースとは前提が異なります。ゲームデプロイを目的に、
`racetrack-agents/race_env.py` (`RacetrackFast`) で以下を段階的に再設計しました
(なし→反発→停止と進化した壁、wall-riding や壁這いの局所解を潰した報酬の経済設計など、
変遷の詳細はスライド版を参照)。

### 2.1 壁の物理 (最重要の変更)

| | オリジナル | 本環境 |
|---|---|---|
| 路外 | 罰則のみ (すり抜け可能) | **コース外周のみ物理的な壁** |
| 車線境界線 | — | 壁ではない (2 車線を自由に使える) |
| 壁接触 | — | 停止 (`wall_mode: "stop"`)、**バックでのみ脱出可能** |
| 壁接触の継続 | — | 35 ステップ (7 秒) でクラッシュ扱い終了 |

**理由:** ゲーム側には物理的な壁があるため、学習時に同じ制約を課さないと
デプロイ後に挙動が崩れます。また「壁に擦りながら曲がる」(wall-riding) が
最速解になってしまうのを、壁接触中の速度報酬ゼロ化と接触継続終了で遮断しています。

### 2.2 報酬設計

生の報酬項 (速度・車線中央維持・前進速度・各種ペナルティ) を合成し、
チェックポイント/ラップボーナスを**生の単位で加算**したあと、
`lmap([-3, 3] → [0, 1])` で正規化します。

**理由 (重要):** lmap の入力範囲が合成報酬の真の下限を覆っていないと、
悪い状態がすべて 0 に飽和して「悪い」と「最悪」の区別 (勾配) が消えます。
範囲の導出は `race_env.py` の lmap 呼び出しに併記しており、
**報酬の重みを変えたら必ず再導出**してください。

### 2.3 壁脱出の経済設計

壁から 1 m 以内 (`wall_escape_zone`) ではバック走行・停止のペナルティを免除し、
壁からの距離増加に報酬 (`wall_escape_progress`) を与えます。

**理由:** 脱出の一歩目 (壁から 20 cm 離れた瞬間) にバックペナルティが復活すると、
「壁沿いを這う」局所解を学習することを実測で確認済みです。ゾーンを接触時のみに
戻さないでください。

### 2.4 観測と行動

| 項目 | 設定 | 理由 |
|---|---|---|
| 観測 | OccupancyGrid 10ch × 12×12、**前方 27 m / 後方 9 m** の非対称視野 | 16 m/s からの制動距離 25.6 m がぎりぎり視野内に収まる |
| 行動 | 連続スロットル [−5, 5] m/s² + ステア ±π/6 | ±2.5 m/s² では制動距離 51 m > 視野 27 m となり巡航速度が落ちる (実測 10 m/s)。出力を絞る場合は非対称 [−5, 2.5] を推奨 |
| 動力学 | `dynamical: True` (タイヤスリップ付き Bicycle model) | ゲームの実車挙動に近い |
| 自己位置 | **観測に含まれない** (自車の x/y は常に 0) | 反射的レーン追従を強制 → トラック暗記を防ぎ汎化の鍵に |

### 2.5 トラック

`config["track"]` で 5 種 + random を選択: `fast` (S 字 + ヘアピン×2)、`oval`、
`stadium` (高速)、`rect` (90°×4)、`chicane` (左右切り返し、最小半径 10 m)。
`"random"` はエピソード毎に再抽選され、汎化学習 (task3) に使います。
全トラックは 2 車線 × 5 m 幅。C# プリセットと `track_<name>.json` に複製され、
検証ハーネスが幾何一致を自動チェックします。

### 2.6 学習設定 (2 フェーズ PPO)

1. **フェーズ 1**: NPC なしでレーン追従を学習 (`other_vehicles=0`)
2. **フェーズ 2**: NPC を加えて追い越しをファインチューニング (学習率を下げる)

- カスタム CNN 特徴抽出器 (`RacetrackCNN`) + 初期スロットルバイアス +0.4
  (序盤の探索で前進が出やすくなる)
- **fast-dev = 12 万ステップ・本番ハイパラ (ent 0.02 / lr 2e-4 / batch 256)**。
  旧設定 (6.5 万・ent 0.10) は方策がランダムに留まり永遠に曲がれません —
  エントロピー報酬が「ランダムでいること」に支払われるためです。
- PPO は終盤に退行することがあるため、最終チェックポイントではなく
  **EvalCallback のベストモデル** (`runs/<EXP_ID>/models/best/phase1|phase2/`) を使います。

---

## 3. セットアップとノートブックの使い方

```bash
# 依存パッケージ (リポジトリの requirements.txt は旧 TF スタック用 — 使わない)
pip install "highway-env>=1.8" "stable-baselines3[extra]>=2.0" gymnasium torch \
            tensorboard moviepy onnx onnxruntime
```

| ノートブック | 用途 |
|---|---|
| `task2_laning_overtaking_sb3_ppo.ipynb` | fast トラック専用の学習 |
| `task3_generalization_sb3_ppo.ipynb` | random トラック学習 + トラック別評価レポート + 5 トラック動画グリッド |

使い方:

1. 設定セルの `FAST_DEV_RUN` を選ぶ — `True` で 12〜15 万ステップの短縮学習
   (周回まで到達することを実測保証)、`False` で本番学習 (100 万+)。
2. **Reward yardstick** セルが「1 周の報酬価値 ≈ 101」「アイドリングの床 ≈ 0.53/step」を
   表示します。TensorBoard の `eval/mean_reward` がアイドリング床に張り付いていたら
   局所解、周回相当値を超えていたら周回獲得です。
3. Trajectory Plot セルはエピソードを走り終えてから描画します (進捗が 100 ステップ毎に
   表示され、`max_steps` で短縮可能、PNG も `VIDEO_DIR` に保存されます)。
4. 録画は 2 分間またはエピソード終了まで。クラッシュや壁スタック終了で動画は自然に切れます。

C# ポートの検証 (テストスイート相当):

```bash
cd racetrack-agents/re_engine/verify && dotnet run -- ..
# 期待: 全シナリオ PASS → ALL SCENARIOS MATCH
```

---

## 4. エクスポートツール

### 4.1 ONNX エクスポート (`racetrack-agents/export_onnx.py`)

```bash
python export_onnx.py <model.zip> -m 3 [-o out.onnx]
```

| モード | 出力 | 用途 |
|---|---|---|
| `1` | full policy (action + value)、CHW | デバッグ・蒸留 |
| `2` | actor のみ、CHW `[1,10,12,12]` | 参考 |
| `3` | **actor のみ、NHWC `[1,12,12,10]`** | **エンジン用 (推奨・既定)** |
| `0` | 上記 3 つすべて (`_full` / `_actor` / `_actor_nhwc` サフィックス) | 一括 |

各エクスポートは onnxruntime で PyTorch 方策と突き合わせ検証されます (≤ 4e-6)。

**エンジンは必ず NHWC 版を使うこと。** C# ビルダーは H×W×C の平坦配列
(`(ix*12+iy)*10+f`) を出力するため、CHW 版に入れるとチャネルが混ざり、
方策は「スロットル ≈ 0.4・ステア ≈ 0.2 のほぼ一定出力」に退化します
(この症状が出たらレイアウト不一致を疑ってください)。

### 4.2 その他のツール

```bash
# 既存の CHW actor を NHWC 化 (Transpose ノードを前置、出力一致を自動検証)
python racetrack-agents/re_engine/convert_onnx_nhwc.py <actor.onnx>

# トラック形状・観測設定を変更したら JSON + 参照観測を再生成
python racetrack-agents/re_engine/export_track_json.py
```

### 4.3 C# デプロイ (`re_engine/`)

- 観測ビルダー 2 種 (`RacetrackObservation.cs` / 単一ファイル版 `HighwayObservationBuilder.cs`)
  — Python とビット一致。レーンは `Lane.Straight` / `Lane.Arc` で外部から構築可能、
  5 トラック分のプリセット内蔵。
- 補助クラス: `SteeringSmoother` (直線の蛇行を除去)、`CorneringLimiter`
  (ステア速度制限 + 旋回時の比例ブレーキ、速度床つきで停止しない)、
  `StuckRecovery` (移動距離ベースのスタック検出 → バック指示)、
  `CarController` (学習時と同一の Bicycle 動力学、速度を Vector3 で出力)。
- **単位は全て m/s・メートル・ラジアン**。km/h の Vector3 は `KphToMs` で変換して
  から渡すこと (kph のまま入れると観測が 3.6 倍になり無警告で劣化します)。
- 進行方向 (heading) は「トラック +x から反時計回り」。前方ベクトルから
  `Atan2(forward.Z, forward.X)`、エンジンの Euler yaw からは `90° − yaw`。

### 4.4 定数同期の掟

`acceleration_range` / `steering_range` / `policy_frequency` / グリッド設定は
C# 側 (`ActionDecoder` / ビルダー定数) に**意図的に複製**されています。
変更したら: C# 定数更新 → `export_track_json.py` 再実行 → 検証ハーネス → ONNX 再エクスポート。

---

## 5. リポジトリ構成 (アクティブスタック)

```
racetrack-agents/
├── race_env.py                        # 学習環境 RacetrackFast (トラック5種/壁/報酬)
├── task2_laning_overtaking_sb3_ppo.ipynb   # fast トラック学習
├── task3_generalization_sb3_ppo.ipynb      # 汎化学習 + トラック別レポート
├── export_onnx.py                     # ONNX エクスポート CLI
└── re_engine/                         # C# デプロイ一式 (README に契約の詳細)
    ├── HighwayObservationBuilder.cs   # 単一ファイル観測ビルダー + 補助クラス
    ├── RacetrackObservation.cs        # フル版観測ビルダー
    ├── CarController.cs               # 学習と同一物理の車両コントローラ
    ├── export_track_json.py / convert_onnx_nhwc.py
    ├── track_<name>.json              # トラック幾何 (5種)
    └── verify/                        # dotnet 検証ハーネス
```

※ `main.py` / `racetrack_env.py` / `agent/` / `models/` は旧 TF2.6 スタックです。
アクティブスタックとは互換性がないため、参照・改修しないでください。
