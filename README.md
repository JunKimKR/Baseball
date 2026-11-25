# pitchcoach

투수의 **측면 영상**을 올리면 투구 동작을 **구간(단계)**으로 나누고, 체크포인트·에너지 흐름(에너지 누수 지점)·부위별 점수를 보여 주고, 문제의 가능한 원인과 **원인별 운동 제안**을 알려 주는 개인용 분석 도구입니다. 본인 컴퓨터에서만 돌아가며(Python 엔진 + 로컬 FastAPI 서버 + React 화면, 서버는 `127.0.0.1`에만 열립니다), **개인·비상업 용도**입니다. 의료 진단 도구가 아닙니다.

## 준비물

- Python 3.11 이상 (3.13에서 개발)
- Node 22 이상
- Linux에서는 MediaPipe가 시스템 GL 라이브러리를 요구합니다. 개발 중에는 아래 명령이 필요했습니다.
  ```bash
  sudo apt-get install -y libegl1 libgles2 libgl1
  ```
- 첫 분석 때 MediaPipe 포즈 모델(약 30 MB)을 내려받아 `~/.cache/pitchcoach/`에 저장합니다(인터넷 필요, 이후에는 재사용).

## 설치

저장소 루트에서:

```bash
python3 -m venv .venv && .venv/bin/pip install -e 'engine[dev]'
npm --prefix web ci && npm --prefix web run build
```

## 실행

```bash
.venv/bin/pitchcoach serve
```

브라우저에서 http://127.0.0.1:8000 을 엽니다(빌드된 `web/dist`를 `/`에서 서빙). 옵션: `--port`(기본 8000), `--data-dir`(기본 `~/.pitchcoach/data`), `--web-dir`(기본 `web/dist`).

화면 없이 명령줄로만 분석할 수도 있습니다.

```bash
.venv/bin/pitchcoach analyze VIDEO --height 182 --hand R [--camera-side arm|glove] --out result.json
```

- `--height`: 투수 키(cm), `--hand`: 투구하는 손(`R`/`L`), `--camera-side`: 카메라가 투구 팔 쪽(`arm`, 기본)인지 글러브 쪽(`glove`)인지.
- 오류 수준 경고가 있으면 종료 코드 1, 영상을 읽을 수 없으면 2입니다.

### 개발 모드

```bash
.venv/bin/pitchcoach serve                # 터미널 1: API (127.0.0.1:8000)
npm --prefix web run dev                  # 터미널 2: Vite 개발 서버 (/api 는 8000번으로 프록시)
```

### 테스트

```bash
cd engine && ../.venv/bin/python -m pytest      # 엔진 (MediaPipe/네트워크가 필요한 slow 테스트는 기본 제외)
npm --prefix web test                           # 웹
```

## 촬영 가이드

1. **투구 팔 쪽 측면**에서 찍습니다. 우투수는 3루 쪽, 좌투수는 반대쪽입니다. 반대편(글러브 쪽)에서 찍었다면 업로드 화면의 "카메라 방향"에서 글러브 쪽을 선택하세요(CLI는 `--camera-side glove`).
2. 카메라는 투수와 **90°**(투구 방향에 수직)가 되게 둡니다.
3. **삼각대로 고정**하고 촬영 중 카메라를 움직이지 않습니다.
4. 머리부터 발끝까지 **전신이 프레임 안**에 계속 들어와야 합니다.
5. **슬로모션 120~240 fps**로 찍습니다.

지키지 않으면 이렇게 됩니다.

- **120 fps 미만**: 정확도가 떨어진다는 경고(`LOW_FPS`)가 표시됩니다. 이벤트 시각이 프레임 단위로 뭉개지기 때문입니다.
- **60 fps 미만**: 타이밍 지표와 에너지 순서 누수 판정이 꺼지고, 속도 지표는 참고값입니다.
- **카메라가 움직임(패닝·손으로 든 촬영)**: `CAMERA_MOTION` 경고가 뜨고 선속도 지표는 참고값이 됩니다. 이 경우 발 착지(foot plant) 검출이 대개 실패합니다. 저장소의 30 fps 손떨림 샘플로 확인했을 때 릴리스만 검출되었습니다.

## 한계 (꼭 읽어 주세요)

- **2D 단일 카메라**입니다. 깊이 방향 움직임은 보이지 않고, 원근·키포인트 잡음이 섞입니다.
- "힘/에너지"는 **운동학적 추정**입니다. 뉴턴(N)이나 N·m 같은 실제 힘·토크가 아닙니다.
- 측면 영상에서 **골반·어깨 회전은 신뢰도가 낮아** 점수에 넣지 않고 참고로만 보여 줍니다.
- **팔꿈치 신전 각속도**는 참고값 전용입니다. 2D 측면에서는 약 2.5배 부풀려집니다(원리와 수치는 `engine/data/obp/README.md`).
- **기준값(norm)**은 OpenBiomechanics Project의 모션캡처 투수(대학·독립리그·마이너, 379구, 93명)를 2D로 투영해 만든 것입니다. **MLB 선수 기준이 아닙니다.** 타이밍 점수는 영상의 프레임 양자화만큼 기준 폭을 넓혀서 계산합니다.
- **운동 라이브러리는 작습니다(4개).** 각 항목은 PubMed에서 초록을 직접 확인한 문헌을 근거로 달았지만, 근거 대부분은 상관(연관) 수준이며 인과를 뜻하지 않습니다. 일부 문제는 아직 근거 있는 원인이 없어 원인 목록이 비어 있습니다.
- 운동은 **제안**일 뿐입니다. 적용 전에 트레이닝 코치와 상의하고, 통증이 있으면 멈추고 전문가와 상담하세요.

## 데이터와 라이선스

- `engine/src/pitchcoach/data/norms_side_v1.json`은 Driveline Baseball의 [OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics) 데이터에서 파생되었으며 **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**(출처 표시 + 비상업 + 동일 조건 변경 허락)을 따릅니다. 이 파일과 파생물을 상업적으로 쓸 수 없습니다. 변경 내용과 재생성 절차는 [`engine/data/obp/README.md`](engine/data/obp/README.md)를 보세요.
- 업로드한 영상은 `~/.pitchcoach/data`에 **로컬로 저장**되며 자동으로 지워지지 않습니다. 필요 없으면 직접 삭제하세요. 영상은 외부로 전송되지 않습니다(첫 실행의 모델 다운로드 제외).

## 이벤트 정확도 평가

이벤트(레그 리프트 정점 `leg_lift_peak`, 발 착지 `foot_plant`, 릴리스 `release`) 검출이 사람이 찍은 프레임과 얼마나 맞는지 측정하는 도구입니다.

1. **라벨 만들기**: 앱에서 영상을 열고 ←/→ 키로 한 프레임씩 움직이며 화면의 `프레임 N / M` 숫자를 읽어 각 이벤트의 프레임 번호를 적습니다.
2. [`engine/eval/labels_template.csv`](engine/eval/labels_template.csv)를 복사해 채웁니다. 열: `video,fps,height_cm,hand,leg_lift_peak,foot_plant,release`. 영상 경로는 CSV가 있는 폴더 기준 상대경로를 쓸 수 있고, 라벨을 모르는 칸은 비워 두면(그 이벤트는 분모에서 제외) 되며, `#`로 시작하는 줄은 무시됩니다.
3. 실행:
   ```bash
   .venv/bin/python engine/scripts/eval_events.py labels.csv
   ```
   실제 MediaPipe로 각 영상을 분석한 뒤 이벤트별·전체 표를 출력합니다. 검출값이 없으면(None) 틀린 것으로 셉니다.
4. **허용 오차는 `2 × fps / 120` 프레임**입니다(120 fps에서 ±2, 240 fps에서 ±4). 전체 일치율이 **0.80 이상이면 종료 코드 0**, 아니면 1입니다.

**v1 목표**는 라벨링한 20개 클립에서 ±2프레임(120 fps 기준) 일치율 80% 이상입니다. **아직 측정하지 않았습니다.** 라벨된 클립 20개가 준비되면 위 명령으로 측정해야 하며, 그 전에는 이 도구의 이벤트 정확도를 보증할 수 없습니다.
