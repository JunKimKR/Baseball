# OpenBiomechanics (OBP) pitching data → `norms_side_v1.json`

The packaged 2D side-view norms (`engine/src/pitchcoach/data/norms_side_v1.json`) are built from
Driveline Baseball's [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics)
by `engine/scripts/build_obp_norms.py`. This file documents the data, the checks made on it, and
how the norms were derived. Only this README is committed. The downloads live in git-ignored
directories:

| Path (git-ignored) | Contents |
|---|---|
| `raw/pitching_landmarks.zip` | Release `dataset-v1` asset, one member `landmarks.csv` (110 MB unzipped) |
| `repo/` | Shallow, sparse clone of the OBP repository pinned to commit `331af0d196437cfc28c011a859a45543291099b5`, holding only the 7 files listed below |

## Fetching (from `engine/`)

```bash
mkdir -p data/obp/raw
curl -sSL --fail -o data/obp/raw/pitching_landmarks.zip \
  https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/pitching_landmarks.zip
sha256sum data/obp/raw/pitching_landmarks.zip
# efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22 (the build script checks it)
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/drivelineresearch/openbiomechanics.git data/obp/repo
git -C data/obp/repo sparse-checkout set --no-cone /LICENSE-DATA.md /DATASHEET.md /README.md \
  /baseball_pitching/README.md /baseball_pitching/data/metadata.csv \
  /baseball_pitching/data/poi/poi_metrics.csv /baseball_pitching/data/data_dictionary.csv
git -C data/obp/repo fetch --depth 1 origin 331af0d196437cfc28c011a859a45543291099b5
git -C data/obp/repo checkout --detach 331af0d196437cfc28c011a859a45543291099b5
```

- The clone is pinned to the commit the norms were built from.
- The sparse set lists files one by one, so the repository's agent-instruction files
  (`CLAUDE.md`, `AGENTS.md`) are never checked out.
- Only these CSV and Markdown files are used, as data. No code from the OBP repository is run.

## License

The OBP data and everything derived from it are licensed
**[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)** (`LICENSE-DATA.md` in the
OBP repository). The derived `norms_side_v1.json` is distributed under the same license, and its
`license`, `license_url` and `source` fields say so.

Changes made to the data:
- projection of the 3D joint centers to 2D
- resampling and filtering
- event detection and metric computation by pitchcoach
- aggregation into a mean and sd per metric

`LICENSE-DATA.md` adds one restriction: anyone employed by, associated with, or a significant
shareholder of a professional sports organization or financial analysis firm may not use the
data at all. This project is personal and non-commercial.

## What the data contain

**`landmarks.csv`** (in the zip):
- 62 columns, 411 trials, 306–1114 rows per trial (0.85–3.09 s).
- `session_pitch`: trial id; the rows of each trial are contiguous.
- `time`: seconds from the trial's first frame. It is a 360 Hz grid to within the file's 0.1 ms
  rounding, checked for every trial.
- 18 landmarks × `_x/_y/_z`, in meters, in the lab frame: +X = 2nd base → home, +Y = 1st-base side,
  +Z = up.
- 6 event times, in seconds: `pkh_time`, `fp_10_time`, `fp_100_time`, `MER_time`, `BR_time`,
  `MIR_time`. `fp_10_time` and `fp_100_time` are empty in 8 trials.
- No coordinate values are missing.
- Per the OBP datasheet, marker-derived signals are already low-passed (4th-order Butterworth,
  20 Hz).

The 18 landmarks:
- `rear_ankle_jc`, `rear_knee_jc`, `rear_hip`
- `lead_ankle_jc`, `lead_knee_jc`, `lead_hip`
- `shoulder_jc`, `elbow_jc`, `wrist_jc`, `hand_jc` (throwing arm)
- `glove_shoulder_jc`, `glove_elbow_jc`, `glove_wrist_jc`, `glove_hand_jc`
- `thorax_ap`, `thorax_dist`, `thorax_prox`, `centerofmass`

Arms are named throwing (no prefix) / glove and legs lead / rear, not left / right.

**`baseball_pitching/data/metadata.csv`**: 411 rows, one per trial. Columns: `user`, `session`,
`session_pitch`, `session_mass_kg`, `session_height_m`, `age_yrs`, `playing_level`,
`pitch_speed_mph`, `filename_new`, `modelname_new`.
- Height is in **meters**: 1.676–2.032.
- `playing_level`:

  | Level | Trials | Athletes |
  |---|---|---|
  | `college` | 314 | 75 |
  | `independent` | 42 | 12 |
  | `milb` | 23 | 6 |
  | `high_school` | 32 | 7 |

- 100 athletes, each with one session.

**`baseball_pitching/data/poi/poi_metrics.csv`**: the throwing hand, `p_throws`, which
metadata.csv does not have. R: 328 trials, L: 83.

### Checks on the data (all 411 trials)

- `wrist_jc` is the throwing wrist. Its 3D peak speed is 17–23 m/s, against 3–10 m/s for
  `glove_wrist_jc`, and it is faster in every trial.
- `lead_ankle_jc` is the stride foot. It moves +1.0 to +2.2 m in X from `pkh_time` to `BR_time`,
  against +0.2 to +0.8 m for the rear ankle. So **+X is the throw direction**: our
  `throw_direction` is +1 for every trial.
- The `lead_knee_jc` height peaks at `pkh_time`, within 6 ms.
- Z is up: the ankles are at about 0 m, the shoulders at about 0.79 × height.
- `p_throws` matches the geometry, and the lab frame is not mirrored for left-handers. At ball
  release:
  - R: the throwing shoulder is at −Y (3rd-base side), 0.17–0.38 m from the glove shoulder.
  - L: the throwing shoulder is at +Y, 0.14–0.35 m from the glove shoulder.

## Joint mapping (OBP → MediaPipe slot)

| OBP landmark | RHP | LHP |
|---|---|---|
| `shoulder_jc`, `elbow_jc`, `wrist_jc` | `right_shoulder/elbow/wrist` | `left_…` |
| `glove_shoulder_jc`, `glove_elbow_jc`, `glove_wrist_jc` | `left_shoulder/elbow/wrist` | `right_…` |
| `rear_hip`, `rear_knee_jc`, `rear_ankle_jc` | `right_hip/knee/ankle` | `left_…` |
| `lead_hip`, `lead_knee_jc`, `lead_ankle_jc` | `left_hip/knee/ankle` | `right_…` |

- The other 21 MediaPipe slots have no OBP equivalent: nose, eyes, ears, mouth, hand points,
  heels and toes. Their xy is NaN and their vis is 0.
- `hand_jc`, `glove_hand_jc`, `thorax_*` and `centerofmass` are not used.
- The metrics depend only on these roles, so they do not depend on handedness.
- Every trial is also checked in 2D: at our release, the throwing wrist must be the fastest
  tracked joint. All 379 trials passed.

## How the norms are built

For each trial:
1. Projection: (x, y) = (X, −Z), in meters, with ppm = 1.
2. The 12 landmarks go into the MediaPipe slots above.
3. Linear resampling from 360 Hz to 240 Hz.
4. `smooth()` with its defaults (15 Hz), as for a 240 fps video.
5. Our `detect_events()` and `throw_direction()`.
6. `compute_metrics()`.

A trial is skipped, and counted, if any of these hold:
- an event is not found
- the direction is not +1
- the throwing wrist is not the fastest tracked joint at release

OBP's own events are used only for the agreement table below.

**Population**: playing level college or above (`college`, `independent`, `milb`); `high_school`
is excluded. That gives **379 trials, 93 athletes**: 310 trials RHP, 69 LHP; heights
167.6–203.2 cm. No trials were skipped.

**Norms**: every metric whose default confidence is not `low` (12 metrics), with the mean and
sample sd over the trials. `direction` is `higher` for speeds, angular velocities, stride and
lead-knee extension, and `target` for everything else. Three `low` metrics get no norm and no
score; they are reference only:
- the pelvis and shoulder rotation proxies
- `elbow_ext_peak_vel_dps` (see below)

| Metric | Mean | SD | Direction |
|---|---|---|---|
| `stride_length_pct_height` | 72.75 % | 4.68 | higher |
| `lead_knee_flexion_fp_deg` | 46.4° | 8.5 | target |
| `lead_knee_extension_deg` | 2.6° | 12.6 | higher |
| `pelvis_peak_speed_ms` | 2.39 m/s | 0.35 | higher |
| `trunk_flexion_br_deg` | 26.3° | 8.3 | target |
| `trunk_flexion_peak_vel_dps` | 342°/s | 59 | higher |
| `shoulder_peak_speed_ms` | 3.87 m/s | 0.46 | higher |
| `wrist_peak_speed_ms` | 19.45 m/s | 1.09 | higher |
| `fp_to_br_ms` | 110.1 ms | 22.1 | target |
| `gap_pelvis_shoulder_ms` | 32.2 ms | 64.2 | target |
| `gap_shoulder_elbow_ms` | 86.5 ms | 19.5 | target |
| `gap_elbow_wrist_ms` | 43.3 ms | 3.4 | target |

The projected metrics track OBP's own 3D values: stride r = 0.82 (OBP's
`stride_length` mean is 83.5 %), trunk flexion at release r = 0.79 (OBP mean 34.3°), and lead-knee
extension r = 0.74.

### Our events vs OBP's events (379 trials; ours − OBP, ms)

| Comparison | Median | IQR | n |
|---|---|---|---|
| foot plant vs `fp_100_time` (100 % body weight) | +12.5 | +6.9 … +18.1 | 371 |
| foot plant vs `fp_10_time` (first contact) | +32.0 | +26.4 … +44.4 | 371 |
| release vs `BR_time` | −12.5 | −13.9 … −11.1 | 379 |
| leg-lift peak vs `pkh_time` (all 411 trials) | 0.0 | −1.4 … +1.4 | 411 |

What follows from these differences:

- **Release.** Our release is the 2D wrist-speed peak. The wrist joint center peaks before ball
  release in 3D too (median −14 ms in the raw data), so the offset is part of the definition,
  not the projection.
- **Foot plant.** Our foot plant is when the lead ankle has stopped (< 0.5 m/s for ≥ 3 frames).
  The ankle center keeps moving for a while after force-plate contact.
- **Short FP→BR.** Together these make `fp_to_br_ms` about 25 ms shorter than OBP's
  `BR_time − fp_100_time` (median 136 ms). The norm uses the same definitions as the video
  analysis.
- **Large disagreements.** 32 trials disagree with `fp_100_time` by more than 40 ms; the build
  prints this count.
  - 9 are OBP force-plate events that cannot be right: OBP's own FP→BR is −72 to 661 ms
    (athletes 2918 and 2923).
  - 4 are one athlete (2990) for whom our foot plant is within 5 ms of first contact
    (`fp_10_time`); this pitcher takes 50–70 ms to reach 100 % body weight. This is a definition
    difference, not a miss.
  - 12 are **late misses** by our detector. The foot bounces or slides about 4 cm on the ground
    after contact and stays above 0.5 m/s, so our FP→BR is only 38–88 ms.
  - 7 are **early misses**. The ankle stops moving forward 6–8 cm above the ground while it
    still moves sideways, which a side view cannot see; the knee is nearly straight (7–25°
    flexion).
- **Effect of the 19 misses (5 %).** They are kept, as the build plan specifies. Excluding them
  would move no mean by more than 0.7 units. It would tighten three sds:

  | Metric | SD with misses | SD without |
  |---|---|---|
  | `fp_to_br_ms` | 22.1 ms | 17.4 ms |
  | `lead_knee_flexion_fp_deg` | 8.5° | 7.4° |
  | `lead_knee_extension_deg` | 12.6° | 11.2° |

  The ~110 ms mean of `fp_to_br_ms` comes from the definitions above, not from the misses.

### Why `elbow_ext_peak_vel_dps` has no norm

Its default confidence is `low`, a controller ruling on the numbers below. It is still computed
and shown, but not scored.

- It is the rate of the **projected** elbow angle, and it mostly measures forearm
  foreshortening.
- In OBP its 2D peak comes when the projected forearm is a median 41 % of its true length
  (5th percentile 11 %).
- The projected rate averaged 6333 ± 1867°/s over the 379 trials (median 6026°/s). The true
  3D elbow extension velocity from the same landmarks, with the same 15 Hz filter, was a median
  2460°/s (5th–95th percentile 2100–2850°/s; every 4th of the 411 trials).
- So the side view inflates it about 2.5×, and its spread mostly reflects how the arm is turned
  relative to the camera.
- With direction `higher`, a norm would reward foreshortening, and the spec forbids overstating
  precision.

### Timing scores at video frame rates

**Why.** The `gap_elbow_wrist_ms` norm (sd 3.45 ms) is narrower than one frame: 4.17 ms at
240 fps, 16.7 ms at 60 fps. Without a correction, ±1 frame of jitter in the peak frame decides
its score.

**Formula.** A timing value is the difference of two frame times, each quantized to 1000/fps ms;
each error is uniform, with variance q²/12. So `score_metrics(metrics, norms, fps)` scores every
metric with unit `ms` against sd_eff = √(sd² + (1000/fps)²/6). The attached norm keeps the
population sd. The OBP sd already contains the 240 Hz quantum, so the correction is slightly
conservative at 240 fps.

**Effect on `gap_elbow_wrist_ms`** (each reachable value: score without the correction → with
it):

| fps | sd_eff | Scores |
|---|---|---|
| 240 | 3.85 ms | 37.5 ms: 9 → 13; 41.7 ms: 63 → 66; 45.8 ms: 47 → 52; 50.0 ms: 5 → 8 |
| 120 | 4.84 ms | 41.7 ms: 63 → 73; 50.0 ms: 5 → 17 |
| 60 | 7.63 ms | 33.3 ms: 0 → 19; 50.0 ms: 5 → 38 |

### Caveats for using the norms

- `gap_pelvis_shoulder_ms` has a wide spread (sd 64 ms, robust sd 31 ms). The 2D pelvis-center
  speed has a broad plateau during the stride, and its peak frame falls anywhere from 130 ms
  before to 40 ms after foot plant (5th–95th percentile).
- The norms come from orthographic projections of marker-based joint centers. A video adds
  perspective, keypoint noise and MediaPipe joint definitions on top.
- Peak wrist speed loses about 1 % to the projection and resampling (3D raw 21.0 m/s → 2D raw
  20.7 m/s, medians) and 5 % to our 15 Hz smoothing (→ 19.6 m/s). OBP had already low-passed
  the data at 20 Hz, which a video does not do, so the norms' peak speeds may be slightly lower
  than a video of the same pitch would give.

## Rebuilding (from `engine/`)

```bash
../.venv/bin/python -I scripts/build_obp_norms.py \
  --landmarks data/obp/raw/pitching_landmarks.zip \
  --metadata data/obp/repo/baseball_pitching/data/metadata.csv \
  --out src/pitchcoach/data/norms_side_v1.json
../.venv/bin/python -m pytest tests/test_build_obp_norms.py -v
```

- The script checks the zip's sha256. It reads the handedness from
  `<metadata dir>/poi/poi_metrics.csv` (override with `--poi`) and the repository commit from the
  clone (`--obp-commit`). It prints the population, skips, event agreement and the norms table.
- Run as shown, it should reproduce the committed file byte for byte.
