# 투수 코칭 시스템 — 측면 분석 + 웹앱 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 측면 투구 영상 1개를 넣으면 구간 분할·체크포인트·에너지 흐름 오버레이·부위 점수·원인별 운동 제안을 보여주는 로컬 웹앱을 만든다 (설계 문서 7절의 1~2단계).

**Architecture:** 분석 엔진 Python 패키지 `pitchcoach`(포즈 → 스무딩 → 이벤트 → 지표 → 기준값 점수 → 에너지 흐름 → 진단)가 결과 JSON 하나를 만든다. 로컬 FastAPI 서버가 업로드를 받아 엔진을 백그라운드로 돌리고 JSON·영상을 제공한다. React 프런트엔드가 영상 위 Canvas에 결과를 그린다.

**Tech Stack:** Python 3.13, numpy, scipy, opencv-python, mediapipe 1.x (Tasks API `PoseLandmarker`), pyyaml, FastAPI, uvicorn, pytest, httpx · Node 22, React 18, TypeScript, Vite, Vitest.

**Spec:** `docs/superpowers/specs/2026-11-24-pitching-coach-design.md`

**범위 밖 (별도 계획):** 후면 영상 분석, Rapsodo/TrackMan CSV(설계 5.7, 7절 3단계), 3D/사운드/LLM(4단계).

## Global Constraints

- 개인·비상업 사용, localhost 전용. 계정·외부 공개·업로드 자동 삭제 없음.
- UI 문자열은 한국어만.
- 화면의 "힘/에너지"는 운동학 기반 추정 지표이며 N·N·m를 표시하지 않는다.
- 모든 지표는 `confidence ∈ {"high","medium","low"}`를 가진다. low 지표는 오버레이에서 흐리게(불투명도 0.35)·점선으로 그린다.
- fps < 120 → 경고(`LOW_FPS`), fps < 60 → 타이밍 지표 값 `null` (`TIMING_DISABLED`).
- 카메라는 투구 팔 쪽 측면만 지원. `camera_side != "arm"`이면 경고(`GLOVE_SIDE_CAMERA`), 분석은 진행.
- 모든 기준값·운동 항목은 `source` 필드를 가진다. 출처 없는 항목은 넣지 않는다.
- 운동은 "제안"으로 표기하고 모든 결과 화면에 문구 `트레이닝 코치와 상의 후 적용하세요.`를 표시한다.
- 3D 기준값과 2D 측정값을 직접 비교하지 않는다. 기준값은 OpenBiomechanics 3D 데이터를 시상면(측면)으로 투영해 만든다. 투영 불가 지표(회전 proxy)는 `score: null`(참고값).
- v1 완료 기준: 수작업 라벨 클립 20개에서 이벤트 오차 120fps 기준 ±2프레임 이내 비율 ≥ 80%.
- 개발 환경: 저장소 루트의 Python 가상환경 `.venv`(Task 1이 생성, git 무시). 엔진 명령은 `engine/`에서 `../.venv/bin/python -m pytest ...` 처럼 venv 파이썬으로 실행.
- 테스트용 실제 영상: `samples/pitch_side_30fps_handheld.mp4` (git 무시, 커밋 금지 — 타인이 촬영한 영상). 우투수, 30fps, 720×1280 세로, 관중석 핸드헬드 촬영(카메라가 투수를 따라 패닝), 투구 팔 쪽(3루 쪽) 사선 각도, 화면 오른쪽으로 던짐, 릴리스 구간 모션 블러, 투구 후 마운드를 걸어 다님.
- OpenBiomechanics 데이터 라이선스: CC BY-NC-SA 4.0 (`LICENSE-DATA.md`로 확인). 이 데이터로 만든 기준값 파일에도 같은 라이선스를 표기한다.

## Review Focus

1. **좌투수 / 왼쪽으로 던지는 영상**: 투구 방향(화면 좌→우, 우→좌)과 투구 손(L/R)의 네 조합 모두 같은 이벤트·지표를 내야 한다. → Task 3 `test_events_all_orientations`.
2. **키포인트 누락 프레임**(가림, visibility < 0.5): NaN 보간 후 계산, 연속 누락이 0.1초를 넘으면 경고 `KEYPOINTS_MISSING`, 크래시 없음. → Task 2 `test_smooth_fills_short_gaps`, `test_quality_flags_long_gap`.
3. **세로 촬영(회전 메타데이터) 영상**: 아이폰 세로 영상은 회전 정보대로 프레임을 돌려 읽어야 한다. → Task 9 `test_read_frames_applies_rotation`.
4. **한 클립에 투구가 여러 번**: 첫 번째 투구만 분석하고 경고 `MULTIPLE_PITCHES`. → Task 3 `test_events_first_pitch_only`.
5. **사람 미검출·짧은 영상**: 검출률 < 80% 또는 길이 < 1초면 `error` 경고를 담은 결과를 반환하고 지표는 계산하지 않는다(서버는 실패가 아닌 완료 상태로 응답). → Task 2 `test_quality_rejects_no_person`, Task 10 `test_analysis_with_error_warning_completes`.
6. **카메라가 움직이는 영상**(핸드헬드·패닝 — 위 샘플 영상): 설계 5.8의 '흔들림 경고'. 배경 움직임을 추정해 `CAMERA_MOTION` 경고를 내고, 픽셀 선속도 지표(단위 m/s)는 confidence "low"로 낮춘다. 각도·각속도는 카메라 평행이동에 영향받지 않으므로 유지. → Task 9 `test_camera_motion_detects_pan`, `test_camera_motion_static_is_zero`, `test_analyze_adds_camera_motion_warning`, Task 4 `test_metrics_camera_moving_lowers_linear_speeds`.
7. **투구 전후에 걷는 동작이 있는 클립**(샘플 영상): 걷기를 두 번째 투구로, 준비 동작을 스트라이드로 오인하지 않는다. → Task 3 `test_events_walking_after_pitch_no_multiple`, `test_events_no_motion`.

---

## File Structure

```
engine/
  pyproject.toml
  src/pitchcoach/
    __init__.py
    joints.py        # 관절 이름·인덱스, 좌우 매핑
    models.py        # dataclass 모델 + to_dict (결과 JSON 스키마의 단일 출처)
    smoothing.py     # 누락 보간 + Butterworth
    scale.py         # 픽셀→미터
    quality.py       # 입력 품질 경고
    events.py        # 투구 방향, 이벤트, 구간
    kinematics.py    # 속도·각도 시계열 기본 함수
    metrics.py       # 고정 지표 목록 계산
    norms.py         # 기준값 로드 + 점수
    energy.py        # 에너지 흐름 강도, 누수, 부위 점수
    diagnosis.py     # 규칙 엔진
    pose.py          # PoseEstimator 프로토콜, MediaPipe 구현, 영상 읽기
    camera.py        # 카메라 움직임(패닝·흔들림) 추정
    analyze.py       # 파이프라인
    cli.py           # `pitchcoach analyze`
    data/norms_side_v1.json   # Task 6이 생성
    data/rules_v1.yaml        # Task 8
    server/app.py    # FastAPI
  scripts/build_obp_norms.py
  scripts/eval_events.py
  tests/... (모듈별 test_*.py), tests/synth.py
web/
  package.json, vite.config.ts, index.html
  src/types.ts       # 결과 JSON 타입 (models.py와 1:1)
  src/api.ts
  src/render/frame.ts, src/render/skeleton.ts, src/render/energy.ts
  src/components/UploadPage.tsx, AnalysisPage.tsx, VideoOverlay.tsx,
                 PhaseTimeline.tsx, SegmentScores.tsx, FindingsPanel.tsx
  src/**/*.test.ts
```

## 결과 JSON 스키마 (v1) — 엔진·서버·웹의 공통 계약

```jsonc
{
  "schema_version": 1,
  "meta": {"fps": 240.0, "width": 1080, "height": 1920, "n_frames": 600, "duration_s": 2.5},
  "subject": {"height_cm": 185.0, "hand": "R", "camera_side": "arm"},
  "warnings": [{"code": "LOW_FPS", "severity": "warn", "message": "..."}],   // severity: info|warn|error
  "throw_direction": 1,            // +1 = 화면 오른쪽으로 던짐, -1 = 왼쪽
  "keypoints": {"joints": ["nose", "..."], "frames": [[[x, y, vis], ...], ...]},  // 스무딩된 픽셀 좌표
  "events": {"leg_lift_peak": 120, "foot_plant": 300, "mer_est": 330, "release": 342},  // 프레임 번호 또는 null
  "phases": [{"name": "leg_lift", "label": "레그 리프트", "start": 0, "end": 120}, ...],
  "metrics": [{"name": "stride_length_pct_height", "label": "보폭(키 대비)", "value": 82.1, "unit": "%",
               "confidence": "high", "segment": "legs", "score": 64, "z": 0.36,
               "norm": {"mean": 80.0, "sd": 6.0, "direction": "higher", "source": "..."}}],
  "segments": [{"segment": "legs", "label": "다리", "score": 71, "confidence": "high"}, ...],  // legs, pelvis, trunk, arm
  "energy": {"segments": ["legs", "pelvis", "trunk", "arm"],
             "intensity": [[0.1, 0.0, 0.05, 0.0], ...],   // 프레임별, 0..1
             "leaks": [{"segment": "trunk", "frame": 318, "kind": "out_of_order", "severity": 0.6, "message": "..."}],
             "peaks": {"pelvis_center": 302, "shoulder_center": 314, "elbow": 330, "wrist": 342}},   // 선속도 최고 프레임 또는 null
  "findings": [{"issue_id": "late_trunk_peak", "title": "...", "evidence": "...",
                "causes": [{"id": "...", "title": "...",
                            "self_test": {"id": "...", "instruction": "...",
                                          "options": [{"id": "pass", "label": "..."}, {"id": "fail", "label": "..."}],
                                          "implies_cause_on": "fail", "source": "..."},
                            "exercises": [{"id": "...", "name": "...", "sets_reps": "3×8", "description": "...", "source": "..."}]}]}]
}
```

---

### Task 1: 엔진 골격 + 모델 + 관절 정의

**Files:**
- Create: `engine/pyproject.toml`, `engine/src/pitchcoach/__init__.py`, `joints.py`, `models.py`
- Create: `.gitignore` — `.venv/`, `.superpowers/`, `samples/`, `engine/data/obp/repo/`, `engine/data/obp/raw/`, `web/node_modules/`, `web/dist/`, `data/`, `__pycache__/`, `*.egg-info/`, `.pytest_cache/`
- Test: `engine/tests/test_models.py`

**Interfaces:**
- Produces:
  - `joints.JOINTS: list[str]` — MediaPipe Pose 33개 이름 순서 그대로(`nose`, `left_eye_inner`, …, `right_foot_index`). `joints.idx(name) -> int`.
  - `joints.side_joints(hand: str) -> dict` — `{"throw_shoulder","throw_elbow","throw_wrist","lead_hip","lead_knee","lead_ankle","lead_heel","lead_toe","rear_hip","rear_knee","rear_ankle"}` → 인덱스. hand="R"이면 throw=right, lead=left.
  - `models.py` dataclass: `VideoMeta(fps, width, height, n_frames)` + `duration_s` 프로퍼티; `Subject(height_cm: float, hand: Literal["R","L"], camera_side: Literal["arm","glove"])`; `Keypoints(xy: np.ndarray (F,33,2) float, vis: np.ndarray (F,33))`; `AnalysisWarning(code, severity, message)` (파이썬 내장 `Warning`을 가리지 않도록 이 이름); `Events(leg_lift_peak, foot_plant, mer_est, release: int|None)`; `Phase(name, label, start, end)`; `Norm(mean, sd, direction: Literal["higher","lower","target"], source)`; `Metric(name, label, value: float|None, unit, confidence, segment, score: int|None=None, z: float|None=None, norm: Norm|None=None)`; `SegmentScore(segment, label, score: int|None, confidence)`; `Leak(segment, frame, kind: Literal["out_of_order","speed_drop"], severity: float, message)`; `Energy(segments, intensity: np.ndarray (F,4), leaks, peaks: dict[str, int | None])` (peaks 키: `pelvis_center`, `shoulder_center`, `elbow`, `wrist`); `SelfTest`, `Exercise`, `Cause`, `Finding`(스키마 필드 그대로); `AnalysisResult`(스키마 최상위 필드 그대로, 미계산 필드는 빈 값).
  - `AnalysisResult.to_dict() -> dict` (스키마 v1과 정확히 일치, numpy → list, NaN → None).

- [ ] **Step 1: 실패 테스트 작성** — `test_side_joints_right_hand` (`side_joints("R")["throw_wrist"] == idx("right_wrist")`, `["lead_knee"] == idx("left_knee")`), `test_side_joints_left_hand`(반대), `test_result_to_dict_schema` (최소 AnalysisResult → `to_dict()`의 최상위 키 집합이 스키마와 같고 `schema_version == 1`, NaN 값이 `None`, `json.dumps` 성공).
- [ ] **Step 2: 실행해 실패 확인** — `python3 -m venv .venv && .venv/bin/pip install -e 'engine[dev]' && cd engine && ../.venv/bin/python -m pytest tests/test_models.py -v` → ImportError.
- [ ] **Step 3: 구현** — pyproject에 의존성(numpy, scipy, opencv-python, mediapipe, pyyaml, fastapi, uvicorn, python-multipart; dev: pytest, httpx)과 console script `pitchcoach = pitchcoach.cli:main`.
- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS.
- [ ] **Step 5: 커밋** — `git commit -m "feat(engine): models, joints, result schema"`

### Task 2: 스무딩 · 크기 보정 · 품질 검사

**Files:**
- Create: `smoothing.py`, `scale.py`, `quality.py`
- Test: `tests/test_smoothing.py`, `tests/test_scale.py`, `tests/test_quality.py`

**Interfaces:**
- Consumes: `Keypoints`, `VideoMeta`, `Subject`, `AnalysisWarning`.
- Produces:
  - `smooth(kp: Keypoints, fps: float, cutoff_hz: float = 15.0, vis_threshold: float = 0.5) -> Keypoints` — vis < 0.5 좌표를 NaN 처리 → 관절별 선형 보간(양 끝은 최근값 유지) → 4차 Butterworth `filtfilt`. `cutoff_hz`는 `min(cutoff_hz, 0.4 * fps / 2)`로 제한.
  - `longest_gap_s(kp: Keypoints, fps: float, joints: list[int]) -> float`
  - `pixels_per_meter(kp: Keypoints, height_cm: float, n_frames: int = 15) -> float` — 첫 `n_frames` 프레임의 (양 발목 중점 → 코) 거리 중앙값 ÷ (0.89 × 키[m]).
  - `check_quality(meta: VideoMeta, kp_raw: Keypoints, subject: Subject, key_joints: list[int]) -> list[AnalysisWarning]` — 코드: `LOW_FPS`(warn, fps<120), `TIMING_DISABLED`(warn, fps<60), `GLOVE_SIDE_CAMERA`(warn), `KEYPOINTS_MISSING`(warn, key_joints 연속 누락 > 0.1s), `NO_PERSON`(error, 프레임의 80% 미만에서 key_joints 평균 vis ≥ 0.5), `TOO_SHORT`(error, duration < 1.0s). 메시지는 한국어.

- [ ] **Step 1: 실패 테스트** — `test_smooth_removes_noise`(사인파 + 가우시안 노이즈 → 원 신호와 RMSE가 노이즈 RMSE의 50% 미만), `test_smooth_fills_short_gaps`(5프레임 vis=0 → 결과에 NaN 없음, 보간값이 앞뒤 값 사이), `test_ppm_from_standing_pose`(코–발목 거리 890px, 키 100cm → ppm == 1000 ± 1), `test_quality_low_fps`(fps=90 → `LOW_FPS`만, fps=50 → `LOW_FPS`와 `TIMING_DISABLED`), `test_quality_flags_long_gap`(240fps에서 30프레임 연속 누락 → `KEYPOINTS_MISSING`), `test_quality_rejects_no_person`(vis 전부 0 → severity "error" `NO_PERSON`), `test_quality_too_short`.
- [ ] **Step 2: 실패 확인** — `pytest tests/test_smoothing.py tests/test_scale.py tests/test_quality.py -v` → FAIL.
- [ ] **Step 3: 구현** (위 시그니처).
- [ ] **Step 4: 통과 확인** → PASS.
- [ ] **Step 5: 커밋** — `feat(engine): smoothing, scale, quality checks`

### Task 3: 합성 투구 데이터 + 이벤트 · 구간 검출

**Files:**
- Create: `tests/synth.py`, `src/pitchcoach/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: `Keypoints`, `Subject`, `side_joints`, `Events`, `Phase`, `AnalysisWarning`.
- Produces:
  - `tests/synth.py: make_pitch(fps=240, hand="R", direction=1, ppm=500.0, height_cm=185.0, leg_lift_peak_s=0.5, foot_plant_s=1.25, peak_offsets_ms=None, n_pitches=1, walk_after=False) -> tuple[Keypoints, Events]` — 33관절 전부 vis=1인 단순 막대 인형. 관절 위치는 해부학적으로 정확할 필요 없이 시간 함수로 각각 정의해도 된다.
    - 처음 0.2초는 서 있는 자세: (양 발목 중점 → 코) 거리 = 0.89 × 키[m] × ppm 픽셀 (Task 2 `pixels_per_meter`와 맞물림).
    - 앞무릎이 leg_lift_peak_s에 최고점이고 이때 앞엉덩이보다 위(화면 y가 더 작음). 앞발목은 foot_plant_s까지 투구 방향으로 전진(최고 2 m/s 이상) 후 정지.
    - 선속도 최고 시점 = foot_plant + `peak_offsets_ms`(기본 `{"pelvis": 10, "shoulder": 60, "elbow": 125, "wrist": 150}`): 골반 중심, 어깨 중심, 투구 팔꿈치, 투구 손목 각각 가우시안 속도 프로파일. 손목 최고 속도(≥ 15 m/s)가 클립 전체 최댓값. 정답 release = foot_plant + wrist 오프셋(기본 1.40초), 정답 mer_est = release − 30ms(손목이 어깨보다 가장 뒤에 있는 시점).
    - direction=-1은 x 좌우 반전, hand="L"은 좌우 관절 교체.
    - n_pitches=2면 첫 릴리스 1.5초 뒤 같은 동작을 반복. walk_after=True면 릴리스 0.6초 뒤부터 1.5초간 걷기(발목 속도 약 1.4 m/s로 교대 전진, 무릎은 엉덩이 아래, 손목 ≤ 2 m/s).
    - 클립 길이는 마지막 동작 뒤 0.6초까지 자동. 반환 Events는 첫 투구의 정답 프레임. Task 4·6·7·9 테스트도 이 함수를 쓴다.
  - `detect_events(kp: Keypoints, fps: float, ppm: float, subject: Subject) -> tuple[Events, list[AnalysisWarning]]` — **릴리스를 먼저 찾고 거꾸로** 찾는다(투구 전후의 걷기·준비 동작에 강하도록):
    - release: 투구 손목 속도 s_w의 최댓값 M이 3 m/s 미만이면 None. 아니면 s_w ≥ 0.5·M 인 지역 최대 중 **첫 번째** 프레임.
    - foot_plant: [release − 0.4s, release] 구간 안에서 앞발목 속도 ≥ 0.5 m/s 인 프레임이 한 번 나온 뒤, 앞발목 속도 < 0.5 m/s 가 3프레임 이상 이어지는 첫 프레임.
    - leg_lift_peak: [foot_plant − 2.0s, foot_plant] 구간에서 앞무릎 y 최소(화면상 최고) 프레임.
    - mer_est: foot_plant~release 사이 (손목 x − 어깨 x) × direction 최소 프레임 (direction은 아래 `throw_direction`으로 내부 계산).
    - `MULTIPLE_PITCHES`(warn): release 1.0초 뒤에 s_w ≥ 0.5·M 인 지역 최대가 또 있으면. 첫 투구만 사용.
    - 검출 실패 이벤트는 None, 경고 `EVENT_NOT_FOUND`(warn, 메시지에 이벤트 이름). release가 None이면 나머지도 None.
  - `throw_direction(kp: Keypoints, hand: str, events: Events) -> int` — 앞발목 x(foot_plant) − x(leg_lift_peak)의 부호. 둘 중 하나라도 None이면 release 프레임의 손목 x속도 부호, 그것도 없으면 +1.
  - `build_phases(events: Events, n_frames: int, fps: float) -> list[Phase]` — 이름/라벨: `leg_lift` 레그 리프트 [0, LLP], `stride` 스트라이드 [LLP, FP], `arm_cocking` 팔 코킹 [FP, MER], `acceleration` 가속·릴리스 [MER, BR], `follow_through` 팔로스루 [BR, min(BR+0.5s, 끝)]. None 이벤트가 있으면 해당 구간 생략.

- [ ] **Step 1: 실패 테스트** — `test_events_match_synthetic`(240fps, 정답과 각 이벤트 차이 ≤ 2프레임), `test_events_all_orientations`(hand∈{R,L} × direction∈{1,-1} 4조합 모두 ≤ 2프레임, `throw_direction` 정확), `test_events_first_pitch_only`(n_pitches=2 → 첫 투구 정답과 일치 + `MULTIPLE_PITCHES`), `test_events_walking_after_pitch_no_multiple`(walk_after=True → 정답과 일치, `MULTIPLE_PITCHES` 없음), `test_events_120fps`(fps=120에서도 ≤ 2프레임), `test_events_no_motion`(서 있기만 하는 클립 → 모든 이벤트 None + `EVENT_NOT_FOUND`, 예외 없음), `test_phases_order`(start ≤ end, 연속).
- [ ] **Step 2: 실패 확인** — `pytest tests/test_events.py -v`.
- [ ] **Step 3: synth.py와 events.py 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋** — `feat(engine): event and phase detection with synthetic pitch`

### Task 4: 운동학 + 지표

**Files:**
- Create: `kinematics.py`, `metrics.py`
- Test: `tests/test_kinematics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Keypoints`, `Events`, `Subject`, `side_joints`, `make_pitch`.
- Produces:
  - `kinematics.speed(points_px: np.ndarray (F,2), fps, ppm) -> np.ndarray (F,)` m/s (중앙차분).
  - `kinematics.angle(a, b, c) -> np.ndarray` b에서의 각도(°), `angular_velocity(angle_deg, fps) -> np.ndarray` °/s.
  - `kinematics.trunk_flexion(hip_c, shoulder_c, direction) -> np.ndarray` — 몸통선과 수직선 사이 각도, 투구 방향으로 기울면 +.
  - `kinematics.rotation_proxy(left_px, right_px) -> np.ndarray` — `degrees(arccos(clip(w / max(w), 0, 1)))`, w = 좌우 관절 x 거리.
  - `metrics.METRIC_DEFS: list[dict]` — 아래 표 그대로 (name, label, unit, segment, confidence 기본값, timing: bool).
  - `compute_metrics(kp, events, fps, ppm, subject, direction, camera_moving: bool = False) -> list[Metric]` — 이벤트 None이면 관련 지표 value None. fps < 120이면 timing 지표 confidence를 "medium"으로 낮춤. fps < 60이면 timing 지표 value None, 그리고 속도·각속도 지표(단위 m/s, °/s) confidence "low"(30fps에선 스무딩 차단 주파수가 6Hz로 묶여 최고 속도가 크게 깎이므로). camera_moving=True면 픽셀 선속도 지표(단위 m/s) confidence "low".

| name | label | unit | segment | 기본 confidence | 정의 |
|---|---|---|---|---|---|
| stride_length_pct_height | 보폭(키 대비) | % | legs | high | FP 프레임에서 앞발목 − 뒷발목 x 거리 / 키 (같은 프레임 안에서 재므로 카메라 이동에 영향 없음) |
| lead_knee_flexion_fp_deg | 착지 시 앞무릎 굴곡 | ° | legs | high | 180 − 무릎각 @FP |
| lead_knee_extension_deg | 앞다리 블록(무릎 펴짐) | ° | legs | high | 무릎각 @BR − 무릎각 @FP |
| pelvis_peak_speed_ms | 골반 최고 속도 | m/s | legs | medium | 골반 중심 속도 최대(LLP~BR) |
| pelvis_rotation_proxy_deg | 골반 회전(추정) | ° | pelvis | low | rotation_proxy(엉덩이) @FP (score 없음) |
| shoulder_rotation_proxy_deg | 어깨 회전(추정) | ° | pelvis | low | rotation_proxy(어깨) @FP (score 없음) |
| trunk_flexion_br_deg | 릴리스 시 몸통 숙임 | ° | trunk | high | trunk_flexion @BR |
| trunk_flexion_peak_vel_dps | 몸통 숙임 최고 각속도 | °/s | trunk | medium | FP~BR+0.1s 최대 |
| shoulder_peak_speed_ms | 어깨 중심 최고 속도 | m/s | trunk | medium | FP~BR 최대 |
| elbow_ext_peak_vel_dps | 팔꿈치 펴짐 최고 각속도 | °/s | arm | medium | FP~BR+0.05s 최대 |
| wrist_peak_speed_ms | 손목 최고 속도 | m/s | arm | medium | FP~BR+0.05s 최대 |
| fp_to_br_ms | 착지→릴리스 시간 | ms | arm | high | timing |
| gap_pelvis_shoulder_ms | 골반→어깨 최고속도 간격 | ms | trunk | medium | timing |
| gap_shoulder_elbow_ms | 어깨→팔꿈치 최고속도 간격 | ms | arm | medium | timing |
| gap_elbow_wrist_ms | 팔꿈치→손목 최고속도 간격 | ms | arm | medium | timing |

- [ ] **Step 1: 실패 테스트** — `test_speed_constant_motion`(1프레임에 ppm/fps 픽셀 이동 → 1.0 m/s ± 0.01), `test_angle_right_angle`(90° ± 0.01), `test_rotation_proxy_half_width`(w = max/2 → 60° ± 0.1), `test_metrics_names_fixed`(compute_metrics 결과 name 목록 == METRIC_DEFS 순서), `test_metrics_synthetic_gaps`(make_pitch 240fps → gap_pelvis_shoulder_ms ≈ 50 ± 9, gap_shoulder_elbow_ms ≈ 65 ± 9, gap_elbow_wrist_ms ≈ 25 ± 9, fp_to_br_ms ≈ 150 ± 9), `test_metrics_low_fps_disables_timing`(fps=50 → timing 지표 value None, m/s·°/s 지표 confidence "low"), `test_metrics_camera_moving_lowers_linear_speeds`(camera_moving=True → m/s 지표만 "low", 각도·°/s 지표는 그대로), `test_metrics_missing_event`(foot_plant None → FP 의존 지표 None, 예외 없음).
- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.**
- [ ] **Step 4: 통과 확인.**
- [ ] **Step 5: 커밋** — `feat(engine): kinematics and metric set`

### Task 5: 기준값 로드 + 점수

**Files:**
- Create: `norms.py`
- Test: `tests/test_norms.py`, `tests/fixtures/norms_test.json`

**Interfaces:**
- Consumes: `Metric`, `Norm`.
- Produces:
  - 기준값 파일 형식: `{"version": 1, "population": "...", "source": "...", "norms": {"<metric name>": {"mean", "sd", "direction", "source"}}}`.
  - `load_norms(path: Path | None = None) -> dict[str, Norm]` — None이면 패키지 `data/norms_side_v1.json`.
  - `score_metrics(metrics: list[Metric], norms: dict[str, Norm]) -> list[Metric]` — z = (value − mean)/sd. score: higher → `round(100·Φ(z))`, lower → `round(100·Φ(−z))`, target → `round(100·2·(1 − Φ(|z|)))`. value None이거나 기준값 없거나 confidence "low"면 score/z None(norm은 있으면 붙임).

- [ ] **Step 1: 실패 테스트** — `test_score_higher`(z=0 → 50, z=2 → 98), `test_score_target`(z=0 → 100, z=±1 → 32), `test_score_lower`, `test_score_none_for_low_confidence`, `test_score_none_without_norm`.
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 커밋** — `feat(engine): norm loading and scoring`

### Task 6: OpenBiomechanics 기반 2D 기준값 생성

**Files:**
- Create: `engine/scripts/build_obp_norms.py`, `engine/src/pitchcoach/data/norms_side_v1.json`, `engine/data/obp/README.md`
- Test: `tests/test_build_obp_norms.py`

**Interfaces:**
- Consumes: `metrics.compute_metrics`, `METRIC_DEFS`, `Keypoints`, `Events`.
- Produces: `norms_side_v1.json` (Task 5 형식). `project_sagittal(points_3d: np.ndarray (F,J,3), forward_axis: int, up_axis: int) -> np.ndarray (F,J,2)` (x = 전방축, y = −상방축, 픽셀 대신 미터이므로 ppm=1).

- [ ] **Step 1: 데이터 받기** — 컨트롤러가 미리 확인한 사실: 라이선스 CC BY-NC-SA 4.0(`LICENSE-DATA.md`). 3D 관절 중심 위치 시계열은 저장소 안이 아니라 GitHub Release `dataset-v1`의 `pitching_landmarks.zip`에 있다 — URL `https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/pitching_landmarks.zip` (다운로드 가능 확인), sha256 `efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22`. README 기준: 360 Hz, 좌표계 +x = 2루→홈, +y = 1루 쪽, +z = 위, trial은 `session_pitch` + `time`으로 구분, 이벤트(foot contact 10%BW, foot plant 100%BW, MER, BR, MIR) 시각이 열로 들어 있음. 선수 키·수준·투구 손은 저장소의 `baseball_pitching/data/metadata.csv`. zip은 `engine/data/obp/raw/`, 저장소는 `engine/data/obp/repo/`(둘 다 git 무시)에 둔다. 실제 열 이름, 관절 목록, 수준 값을 확인해 `engine/data/obp/README.md`(커밋)에 적는다. 필요한 관절(양 엉덩이·무릎·발목, 양 어깨, 투구 팔꿈치·손목)이 없으면 여기서 멈추고 BLOCKED로 보고.
- [ ] **Step 2: 실패 테스트** — `test_project_sagittal_drops_lateral`(측면축 이동만 있는 점 → 2D 좌표 불변), `test_build_norms_from_fake_trials`(가짜 3D trial 3개 — `make_pitch(fps=360, ppm=1.0)` 2D 좌표를 x→x, −y→z로 올리고 y축엔 임의 측면 값, trial마다 peak_offsets_ms를 조금씩 다르게 → 출력 JSON의 모든 키가 METRIC_DEFS 중 confidence≠low인 이름, sd > 0, source 비어 있지 않음).
- [ ] **Step 3: 구현** — trial마다: 3D → `project_sagittal(forward_axis=0, up_axis=2)` → OBP 관절 → MediaPipe 관절 이름 매핑으로 `Keypoints` 구성(단위 미터 → ppm=1.0, 없는 관절은 vis=0) → 360 Hz를 240 Hz로 리샘플 → Task 2 `smooth`(기본값) 적용(영상 파이프라인과 같은 필터 조건) → **이벤트는 우리 `detect_events`로 검출**(영상과 같은 정의를 쓰기 위해서다. OBP의 force-plate 이벤트는 정의가 달라 기준값 계산에 쓰지 않는다. 검출 실패 trial은 건너뛰고 개수를 기록하며, OBP FP/BR 대비 우리 이벤트의 중앙 차이(ms)를 README에 적는다) → `throw_direction` → `compute_metrics` → 지표별 mean/sd. direction: speed·velocity·stride·knee_extension = higher, 나머지 = target. 모집단: metadata 기준 수준이 대학 이상(college, independent, MiLB, MLB 등 실제 값)인 투수. JSON에 `population`(수준 목록, trial 수, 선수 수), `source`(OBP URL, 릴리스 태그 `dataset-v1`, 저장소 커밋 해시), `license`("CC BY-NC-SA 4.0") 필드. 지표별 `source`에도 같은 출처 문자열.
- [ ] **Step 4: 실제 데이터로 생성 후 통과 확인** — (`engine/`에서) `../.venv/bin/python scripts/build_obp_norms.py --landmarks data/obp/raw/pitching_landmarks.zip --metadata data/obp/repo/baseball_pitching/data/metadata.csv --out src/pitchcoach/data/norms_side_v1.json` 그리고 `pytest tests/test_build_obp_norms.py -v`. 생성된 mean 값이 물리적으로 말이 되는지 한 번 눈으로 확인(예: 손목 최고 속도가 수 m/s~수십 m/s 범위). 이상하면 축 방향부터 의심.
- [ ] **Step 5: 커밋** — `feat(engine): OBP-derived sagittal norms`

### Task 7: 에너지 흐름 · 누수 · 부위 점수

**Files:**
- Create: `energy.py`
- Test: `tests/test_energy.py`

**Interfaces:**
- Consumes: `Keypoints`, `Events`, scored `list[Metric]`, `kinematics`, `make_pitch`.
- Produces:
  - `SEGMENTS = ["legs","pelvis","trunk","arm"]`, 라벨 `다리/골반/몸통/팔`.
  - `energy_intensity(kp, fps, ppm, subject, direction, norms) -> np.ndarray (F,4)` — legs = 골반 중심 속도, pelvis = 골반 rotation_proxy 각속도(절댓값), trunk = 어깨 중심 속도, arm = 손목 속도. legs·trunk·arm 열은 각각 `pelvis_peak_speed_ms`·`shoulder_peak_speed_ms`·`wrist_peak_speed_ms` 기준값 mean으로, pelvis 열과 기준값이 없는 열은 클립 내 최댓값으로 나누고 [0,1]로 자름.
  - `peak_frames(kp, fps, ppm, subject, events) -> dict[str, int | None]` — 키 `pelvis_center`, `shoulder_center`, `elbow`, `wrist`: 각 점 선속도의 최고 프레임(구간 leg_lift_peak~release+0.05s, 이벤트 없으면 None). `Energy.peaks`에 들어가고 detect_leaks가 쓴다.
  - `detect_leaks(metrics: list[Metric], peaks: dict[str, int | None], fps: float) -> list[Leak]` — out_of_order: peaks의 골반 중심 → 어깨 중심 → 팔꿈치 → 손목 순서를 어기면 늦게 온 앞 단계 점의 부위에 Leak(점→부위: 골반 중심→legs, 어깨 중심→trunk, 팔꿈치·손목→arm; frame = 그 점의 최고 프레임, severity = min(1, 어긋난 ms/30)). **fps < 60이면 out_of_order는 만들지 않는다**(타이밍 비활성). speed_drop: `pelvis_peak_speed_ms`(legs, frame=peaks.pelvis_center), `shoulder_peak_speed_ms`(trunk, peaks.shoulder_center), `wrist_peak_speed_ms`(arm, peaks.wrist)의 z < −1이면 Leak(severity = min(1, |z|/3)); z가 None이면 없음. 메시지는 한국어(예: `몸통 최고 속도가 팔꿈치보다 12ms 늦습니다`).
  - `segment_scores(metrics) -> list[SegmentScore]` — 부위별 score 있는 지표의 평균(반올림), 없으면 None. confidence = 부위 지표 중 가장 낮은 값.

- [ ] **Step 1: 실패 테스트** — `test_intensity_shape_and_range`, `test_intensity_peaks_in_order`(make_pitch → 열별 최고 프레임이 legs ≤ trunk ≤ arm), `test_no_leak_on_ideal_sequence`, `test_out_of_order_leak`(make_pitch를 수정해 어깨 최고를 팔꿈치 뒤로 → trunk에 out_of_order 1개), `test_speed_drop_leak`(wrist z=−2 → arm speed_drop, severity≈0.67), `test_no_out_of_order_leak_below_60fps`, `test_peak_frames_synthetic`(make_pitch 240fps → 각 점 최고 프레임 = foot_plant + 오프셋 ± 2프레임), `test_segment_scores_average_and_none`.
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.** (어깨 최고를 팔꿈치 뒤로 보내는 데이터는 `make_pitch(peak_offsets_ms=...)`로 만든다)
- [ ] **Step 5: 커밋** — `feat(engine): energy flow, leaks, segment scores`

### Task 8: 진단 규칙 + 운동 라이브러리

**Files:**
- Create: `diagnosis.py`, `data/rules_v1.yaml`
- Test: `tests/test_diagnosis.py`

**Interfaces:**
- Consumes: scored `list[Metric]`, `list[Leak]`, `Finding`/`Cause`/`SelfTest`/`Exercise`.
- Produces:
  - YAML 형식: `issues: [{id, title, when: {any: [{metric, z_lt|z_gt|abs_z_gt: float} | {leak_segment, leak_kind}]}, evidence_template, causes: [{id, title, self_test: {id, instruction, options:[{id,label}], implies_cause_on}, exercises: [{id, name, sets_reps, description, source}]}]}]`.
  - `load_rules(path=None) -> list[dict]`, `diagnose(metrics, leaks, rules) -> list[Finding]` — 조건 충족 issue만, 지표 z 또는 leak severity가 큰 순서. evidence_template은 `{value}`, `{unit}`, `{z}` 치환.
  - `validate_rules(rules) -> list[str]` — 오류 목록: 모르는 metric 이름, source 없는 exercise, self_test 옵션 2개 미만.
- 초기 issue(최소 6개, 각 원인 2~3개): 보폭 부족, 앞다리 블록 약함, 골반→어깨 순서 어긋남, 몸통 속도 손실, 팔꿈치→손목 순서 어긋남, 팔 속도 손실. 운동 내용과 자가 테스트는 공개 문헌·공개 교육 자료에서 작성하고 각 항목 `source`에 문헌/URL을 적는다. 출처를 찾지 못한 항목은 넣지 않는다.

- [ ] **Step 1: 실패 테스트** — `test_rules_file_valid`(`validate_rules(load_rules()) == []`), `test_diagnose_triggers_on_z`, `test_diagnose_triggers_on_leak`, `test_diagnose_ignores_none_z`, `test_diagnose_sorted_by_severity`, `test_validate_rules_catches_missing_source`.
- [ ] **Step 2~4: 실패 확인 → 구현·규칙 작성 → 통과.**
- [ ] **Step 5: 커밋** — `feat(engine): rule-based diagnosis and exercise library`

### Task 9: 포즈 추정 + 파이프라인 + CLI

**Files:**
- Create: `pose.py`, `camera.py`, `analyze.py`, `cli.py`
- Test: `tests/test_pose.py`, `tests/test_camera.py`, `tests/test_analyze.py`

**Interfaces:**
- Consumes: 앞 Task 전부.
- Produces:
  - `read_frames(path) -> Iterator[np.ndarray]` + `read_meta(path) -> VideoMeta` — OpenCV, 회전 메타데이터 적용(`CAP_PROP_ORIENTATION_META` 또는 ffprobe로 확인 후 `cv2.rotate`), meta의 width/height는 회전 후 값.
  - `class PoseEstimator(Protocol): def estimate(self, video_path: Path) -> tuple[Keypoints, VideoMeta]`.
  - `MediaPipeEstimator(model_path: Path | None = None)` — 확인된 사실: mediapipe 1.2.3에는 예전 `solutions` 모듈이 없고 Tasks API(`mediapipe.tasks.python.vision.PoseLandmarker`)만 있다. VIDEO 모드, `num_poses=1`, heavy 모델. 모델 파일은 처음 실행 시 `~/.cache/pitchcoach/`에 내려받는다: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task` (접근 확인). 사람이 검출되지 않은 프레임은 vis=0. 정규화 좌표 → 픽셀.
  - `camera_motion_ratio(video_path: Path, kp: Keypoints) -> float` (camera.py) — 연속 프레임 사이 배경 특징점(사람 키포인트 bbox를 20% 넓힌 영역 밖; `cv2.goodFeaturesToTrack` + `cv2.calcOpticalFlowPyrLK`)의 이동량 중앙값을 프레임마다 구하고, 그 경로 길이 합 ÷ 프레임 폭을 반환. 영상을 열 수 없으면 0.0.
  - `analyze(video_path: Path, subject: Subject, estimator: PoseEstimator | None = None) -> AnalysisResult` — 순서: estimate → check_quality(key_joints = `side_joints(hand)`의 값들; error 있으면 warnings·meta만 채워 반환) → smooth → ppm → detect_events → throw_direction → phases → camera_motion_ratio(값 > 0.05면 `CAMERA_MOTION` warn 경고 `카메라가 움직였습니다. 속도 지표는 참고만 하세요. 삼각대로 고정해 촬영하세요.` 추가, camera_moving=True) → compute_metrics → score → peak_frames·energy_intensity·detect_leaks → segment_scores → diagnose.
  - CLI: `pitchcoach analyze VIDEO --height CM --hand R|L [--camera-side arm|glove] --out result.json`.

- [ ] **Step 1: 실패 테스트** — `test_read_frames_applies_rotation`(cv2로 만든 200×100 영상 + 회전 메타가 있는 테스트 파일이 없으면 ffmpeg으로 `-metadata:s:v rotate=90` 생성, 읽은 프레임 shape (200,100,3)), `test_analyze_with_fake_estimator`(make_pitch를 돌려주는 가짜 추정기 → to_dict이 스키마 통과, events not None, metrics 15개, findings는 list), `test_analyze_returns_early_on_error`(vis 0인 가짜 → NO_PERSON 경고, metrics 빈 list), `test_cli_writes_json`(가짜 추정기 주입 가능한 `main(argv, estimator=None)`), `test_camera_motion_detects_pan`(cv2로 만든 무늬 배경이 프레임마다 4px씩 평행이동하는 영상 → ratio > 0.05), `test_camera_motion_static_is_zero`(정지 배경 → ratio < 0.01), `test_analyze_adds_camera_motion_warning`(camera_motion_ratio가 0.2를 돌려주도록 monkeypatch → `CAMERA_MOTION` 경고 + m/s 지표 confidence "low").
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 실제 영상 스모크 테스트** — `samples/pitch_side_30fps_handheld.mp4`로 `../.venv/bin/pitchcoach analyze ../samples/pitch_side_30fps_handheld.mp4 --height 176 --hand R --out ../samples/pitch_side_result.json` 실행(키 176은 근사값). 30fps·핸드헬드라 `LOW_FPS`, `TIMING_DISABLED`, `CAMERA_MOTION` 경고가 예상된다. JSON이 만들어지는지, release·foot_plant·leg_lift_peak 프레임이 영상에서 눈으로 봐도 대략 맞는지(ffmpeg로 해당 프레임을 이미지로 뽑아 확인) 보고서에 적는다. 이벤트가 틀리면 원인(패닝·블러·사선 각도 등)을 적되, 이 영상 하나에 맞춘 임시 보정은 하지 않는다.
- [ ] **Step 6: 커밋** — `feat(engine): mediapipe pose, pipeline, CLI`

### Task 10: 로컬 서버

**Files:**
- Create: `engine/src/pitchcoach/server/__init__.py`, `app.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `analyze`, `Subject`, `AnalysisResult.to_dict`.
- Produces (web이 사용):
  - `create_app(data_dir: Path, estimator: PoseEstimator | None = None) -> FastAPI`.
  - `POST /api/analyses` multipart: `video`(파일), `height_cm`(float), `hand`(R|L), `camera_side`(arm|glove, 기본 arm) → `201 {"id": str}`. id = `uuid4().hex`. 저장 `data_dir/{id}/video.<ext>`, `status.json`.
  - `GET /api/analyses/{id}` → `{"id", "status": "processing"|"done"|"failed", "error": str|null, "result": <스키마>|null}`. 404 if 없음.
  - `GET /api/analyses/{id}/video` → 원본 영상(Range 요청 지원: `FileResponse`).
  - `GET /api/analyses` → `[{"id","created_at","status"}]` 최신순(개인용 기록 목록).
  - 빌드된 `web/dist`가 있으면 `/`에 정적 서빙.
  - 실행: `pitchcoach serve [--port 8000] [--data-dir ./data]` (cli에 추가), 바인딩 `127.0.0.1`.
  - 분석은 FastAPI `BackgroundTasks`. 예외 시 status failed + error 메시지.

- [ ] **Step 1: 실패 테스트** (httpx `TestClient`, 가짜 추정기) — `test_upload_then_done`, `test_get_unknown_404`, `test_video_range_request`(Range 헤더 → 206), `test_invalid_hand_422`, `test_analysis_with_error_warning_completes`(NO_PERSON → status "done", result.warnings에 error), `test_list_analyses_newest_first`.
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 커밋** — `feat(server): local FastAPI server`

### Task 11: 웹 골격 · 타입 · API · 업로드 화면

**Files:**
- Create: `web/` (Vite React TS), `src/types.ts`, `src/api.ts`, `src/components/UploadPage.tsx`, `src/App.tsx`
- Test: `src/api.test.ts`

**Interfaces:**
- Consumes: Task 10 엔드포인트, 결과 스키마.
- Produces:
  - `types.ts`: 스키마의 모든 객체를 TS 타입으로(`AnalysisResult`, `Metric`, `Leak`, `Finding`, … 이름은 models.py와 동일).
  - `api.ts`: `uploadVideo(file: File, opts: {heightCm: number; hand: "R"|"L"; cameraSide: "arm"|"glove"}): Promise<string>`, `getAnalysis(id): Promise<{status; error; result: AnalysisResult|null}>`, `pollAnalysis(id, onUpdate, intervalMs = 1000): () => void`(취소 함수 반환), `videoUrl(id): string`, `listAnalyses()`.
  - 라우팅: `#/` 업로드 + 기록 목록, `#/a/{id}` 분석 화면 (해시 라우팅, 라이브러리 없이).
  - UploadPage: 파일, 키, 투구 손, 카메라 방향(기본 "투구 팔 쪽"), 촬영 가이드 문구(설계 5.8: 투수 90° 옆, 삼각대 고정, 전신이 프레임 안, 슬로모션 120~240fps).
  - Vite dev 서버는 `/api`를 `http://127.0.0.1:8000`으로 프록시.

- [ ] **Step 1: 실패 테스트** — `uploadVideo sends multipart fields`(fetch mock: FormData에 video/height_cm/hand/camera_side), `pollAnalysis stops on done`, `pollAnalysis cancel stops polling`.
- [ ] **Step 2~4: `npm test` 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 커밋** — `feat(web): scaffold, API client, upload page`

### Task 12: 영상 오버레이 · 에너지 흐름 렌더링

**Files:**
- Create: `src/render/frame.ts`, `src/render/skeleton.ts`, `src/render/energy.ts`, `src/components/VideoOverlay.tsx`
- Test: `src/render/frame.test.ts`, `src/render/energy.test.ts`

**Interfaces:**
- Consumes: `AnalysisResult`.
- Produces:
  - `frameAt(timeS: number, fps: number, nFrames: number): number` — `clamp(round(timeS·fps), 0, nFrames−1)`.
  - `toCanvas(x, y, video: {w,h}, canvas: {w,h}): [number, number]` — object-fit: contain 레터박스 고려.
  - `SEGMENT_JOINTS: Record<Segment, [string, string][]>` — legs: 앞·뒷다리 엉덩이–무릎–발목–발끝, pelvis: 좌우 엉덩이, trunk: 엉덩이 중심–어깨 중심 + 좌우 어깨, arm: 투구 어깨–팔꿈치–손목.
  - `segmentStyle(intensity: number, confidence: Confidence, leaking: boolean): {color: string; width: number; alpha: number; dash: number[]}` — 색: 강도 0 → 짙은 파랑, 1 → 밝은 노랑(선형 보간), 굵기 2 + 10·강도 px, low confidence → alpha 0.35, dash [6,4], leaking → 빨강 `#ff3b30`.
  - `activeLeaks(leaks, frame, windowFrames): Leak[]` — |leak.frame − frame| ≤ windowFrames(= 0.05s·fps).
  - `particlesFor(segment, intensity, frame, seed): {x,y,r,alpha}[]` — 부위 선분을 따라 흐르는 입자(근위 → 원위 방향, 개수 = 3 + 12·강도). 누수 중이면 입자가 선분 밖으로 흩어짐. seed 고정 시 결정적.
  - `VideoOverlay`: `<video>` + 같은 크기 `<canvas>`, `requestVideoFrameCallback`(미지원 시 rAF)으로 현재 프레임을 그림. props: `result`, `videoSrc`, `seekToFrame?: number`, `onFrame(frame)`. 키보드 ←/→ 한 프레임 이동, 스페이스 재생/정지, 재생 속도 선택(0.1×, 0.25×, 0.5×, 1×).

- [ ] **Step 1: 실패 테스트** — `frameAt rounds and clamps`, `toCanvas letterbox`(1080×1920 영상, 800×600 캔버스 → 좌우 여백 반영), `segmentStyle low confidence dims`, `segmentStyle leaking is red`, `activeLeaks window`, `particlesFor deterministic and count scales with intensity`.
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 브라우저 확인** — Task 9에서 만든 결과로 서버+웹을 띄워 Playwright로 분석 화면 스크린샷(deviceScaleFactor 1)을 찍어 스켈레톤이 사람 위에 겹치는지 확인.
- [ ] **Step 6: 커밋** — `feat(web): video overlay with energy flow`

### Task 13: 구간 타임라인 · 부위 점수 · 진단 패널 · 분석 화면 조립

**Files:**
- Create: `src/components/PhaseTimeline.tsx`, `SegmentScores.tsx`, `FindingsPanel.tsx`, `AnalysisPage.tsx`, `src/render/timeline.ts`
- Test: `src/render/timeline.test.ts`, `src/components/FindingsPanel.test.tsx`

**Interfaces:**
- Consumes: `AnalysisResult`, `VideoOverlay`, `api.pollAnalysis`.
- Produces:
  - `timeline.ts`: `phaseRects(phases, nFrames, widthPx): {name,label,x,w}[]`, `ghostMarkers(metrics, peaks, fps): {label, actualFrame, idealFrame}[]` — 실제 = `energy.peaks`의 어깨 중심·팔꿈치·손목 최고 프레임. 이상 = 실제 골반 중심 최고 프레임에서 시작해 골반→어깨, 어깨→팔꿈치, 팔꿈치→손목 간격 지표의 norm.mean을 차례로 누적(설계 5.5 타이밍 고스트; 골반 최고 시점 자체에는 기준값이 없으므로 거기를 기준점으로 삼는다). peaks.pelvis_center가 null이거나 해당 간격 지표의 value·norm이 null이면 그 마커와 이후 마커는 생략. `phaseStatus(phase, metrics, leaks): "ok"|"warn"` — 구간 안에 leak가 있거나, `METRIC_PHASE` 상수(stride_length→stride, lead_knee_*→arm_cocking, trunk_flexion_br·wrist·elbow·gap_elbow_wrist→acceleration, 나머지 gap·shoulder·pelvis_peak→arm_cocking)로 그 구간에 속한 지표 중 score < 30이 있으면 "warn" (설계 5.5 체크포인트 아이콘).
  - PhaseTimeline: 구간 막대 + 이벤트 점, 클릭 시 `seekToFrame`, 현재 프레임 커서, 고스트 마커(반투명)와 실제 마커 사이 ms 차이 표시, 누수 지점 빨간 점.
  - SegmentScores: 다리/골반/몸통/팔 점수 막대 + 신뢰도 배지(높음/중간/낮음), score null이면 "참고값".
  - FindingsPanel: finding별 제목·근거, 원인 후보마다 자가 테스트(옵션 라디오). `implies_cause_on`과 같은 옵션을 고르면 그 원인만 펼치고 다른 원인은 접음. 아무것도 안 고르면 모든 원인을 "가능성"으로 나열. 상단 고정 문구 `트레이닝 코치와 상의 후 적용하세요.`. 자가 테스트 응답은 `localStorage`(키 `selftest:{analysisId}`, try/catch).
  - AnalysisPage: 처리 중 → 진행 표시, 경고 목록(severity 색), error 경고면 오버레이 대신 경고와 촬영 가이드만, 아니면 VideoOverlay + PhaseTimeline + SegmentScores + 지표 표(라벨·값·단위·점수·신뢰도·출처 툴팁) + FindingsPanel.

- [ ] **Step 1: 실패 테스트** — `phaseRects proportional widths`, `ghostMarkers cumulative from pelvis peak`, `ghostMarkers skips null`, `phaseStatus warns on leak or low score`, `FindingsPanel shows disclaimer`, `FindingsPanel narrows causes after self-test`(testing-library + jsdom).
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 브라우저 확인** — 전체 흐름(업로드 → 처리 중 → 결과) Playwright로 실행, 폭 390px에서도 가로 스크롤 없음 확인, 스크린샷을 사용자에게 보여줄 경우 deviceScaleFactor 2.
- [ ] **Step 6: 커밋** — `feat(web): timeline, scores, findings, analysis page`

### Task 14: 이벤트 정확도 평가 + 실행 안내

**Files:**
- Create: `engine/scripts/eval_events.py`, `engine/eval/labels_template.csv`, `README.md`(저장소 루트)
- Test: `tests/test_eval_events.py`

**Interfaces:**
- Consumes: `analyze`.
- Produces:
  - 라벨 CSV 열: `video,fps,height_cm,hand,leg_lift_peak,foot_plant,release` (프레임 번호).
  - `evaluate(labels_csv: Path, results: dict[str, Events]) -> dict` — 이벤트별·전체 `within_2_frames_at_120fps` 비율. 허용 오차 = `2 × fps / 120` 프레임(240fps면 4프레임).
  - CLI: `python scripts/eval_events.py labels.csv` → 표 출력, 전체 비율 ≥ 0.80이면 종료 코드 0.
  - README: 설치(`pip install -e engine[dev]`, `npm --prefix web ci && npm --prefix web run build`), 실행(`pitchcoach serve`), 촬영 가이드, 한계(2D·추정 지표·회전 신뢰도 낮음·운동은 제안), OBP 비상업 라이선스 표기.

- [ ] **Step 1: 실패 테스트** — `test_tolerance_scales_with_fps`, `test_evaluate_ratio`.
- [ ] **Step 2~4: 실패 확인 → 구현 → 통과.**
- [ ] **Step 5: 실측** — 사용자가 클립 20개와 라벨을 준비하면 실행. 준비 전이면 이 단계는 "사용자 데이터 대기"로 보고한다(완료로 표시하지 않음).
- [ ] **Step 6: 커밋** — `feat: event accuracy evaluation and README`
