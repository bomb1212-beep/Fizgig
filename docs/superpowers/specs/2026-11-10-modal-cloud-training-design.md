# Modal 클라우드 학습 백엔드 설계

- 작성일: 2026-11-10
- 대상: 툴1 `Fizgig_auto/automation`의 `runner.py`로 실행되는 LoRA 학습(시드 학습, 최종 학습)
- 상태: 설계 승인됨(대화에서 섹션 1~4 승인), 구현 전

---

## 1. 목적과 범위

### 사용자가 정한 것

| 항목 | 결정 |
|---|---|
| 목적 | **PC를 꺼도 학습이 끝까지 진행**되고, **로컬 GPU가 고장·부족할 때 대체 수단**이 된다 |
| 선택 방식 | **학습 작업마다** 로컬/Modal을 고른다. 기본값은 로컬 |
| 결과 회수 | PC가 켜져 있으면 끝나는 즉시 자동으로 받는다. 꺼져 있었으면 **다음에 GUI·runner를 켤 때 자동으로** 받는다 |
| 비용 확인 | **확인창 없이 바로 제출**한다. 예상 비용은 로그에만 남긴다. 안전장치는 Modal 계정의 지출 한도($0)다 |
| 방식 | 로컬에서 데이터만 준비해 올리고, **Modal 함수가 같은 Fizgig 학습 코드로** 캐시 2단계와 학습을 실행한다 |

### 설계가 가정하는 것 (사용자가 말하지 않았지만 이 설계가 기대는 것)

- 동시에 도는 클라우드 학습은 한 번에 하나다(로컬 큐와 같은 직렬 처리).
- 클라우드 학습 결과는 로컬 학습 결과와 **같은 폴더 구조**로 들어와야 하고, 뒤 단계(홀드아웃·에포크 비교)는 바뀌지 않는다.
- Modal 토큰은 사용자가 직접 만든다. 이 도구는 토큰을 만들거나 저장소에 쓰지 않는다.

### 범위 밖

- **Kaggle.** 무료 GPU(P100 16GB, T4 16GB×2)는 krea2 학습의 최대 VRAM(학습 24.6GB, 미리보기 27.8GB, RTX 5090 실측)에 모자라고, 학습 코드가 쓰는 bf16/fp8 경로를 제대로 지원하지 않는다.
- **클라우드에서 그림 생성.** ComfyUI가 필요한 홀드아웃 그리기·에포크 그리드·P-H3 편집은 여전히 로컬 GPU에서 돈다.
- **여러 클라우드 학습 동시 실행**, 자동 재시도, 다른 클라우드(RunPod 등).

---

## 2. 바탕이 된 사실 (2026-11-10 확인)

- **학습 실행 구조.** `runner.execute()`가 `launch_plan()`으로 만든 세 단계(`Cache Preparation` → `Text Encoder Caching` → `Training`)를 차례로 subprocess로 실행한다. 각 단계 명령에 Windows 절대 경로가 들어간다(파이썬, `src/fizgig/families/*.py`, 모델 파일, 실행 폴더).
- **Fizgig는 리눅스에서 돈다.** 원 저장소에 리눅스용 `docker/Dockerfile`과 `uv_install_deps.py`가 있고, 코드에 리눅스 분기가 있다. torch.compile은 리눅스에서 gcc와 Python 헤더가 필요하다(Dockerfile 주석).
- **학습에 필요한 모델은 약 30GB이고 전부 HuggingFace `Comfy-Org/Krea-2`에 있다(계정 불필요).**
  - `diffusion_models/krea2_raw_bf16.safetensors` 24.5GB
  - `text_encoders/qwen3vl_4b_fp8_scaled.safetensors` 4.9GB
  - `vae/qwen_image_vae.safetensors` 0.24GB
  - `loras/krea2_turbo_lora_rank_64_bf16.safetensors` 0.44GB (미리보기용 `--speed_lora`)
  - 터보 DiT 12.2GB는 학습 명령에 없다.
- **데이터는 작다.** export 하나가 약 46MB(이미지+캡션 73개 파일).
- **결과물.** `output/`에 최종 LoRA(56MB), 마지막 학습 상태 폴더(약 280MB), 미리보기, 손실 로그가 생긴다.
- **Modal 계정.** Starter 요금제, 월 $30 크레딧, 지출 한도 $0(크레딧 소진 시 정지), GPU 동시 10개. 요금(초당 → 시간당): L40S $1.95, A100 80GB $2.50, RTX PRO 6000 $3.03, H100 $3.95. CPU 물리 코어당 $0.047/h, 메모리 GiB당 $0.008/h. Volume은 월 1TiB까지 무료.
- **원 저장소 README의 GPU 권장.** krea2는 32GB 이상이면 블록 스왑이 없고, 스왑은 스텝 시간을 약 4배로 늘린다. 48GB인 L40S가 가성비 기준 권장 카드다.

---

## 3. 구성 요소

```
automation/cloud/
  __init__.py
  requirements.txt     # modal (Fizgig venv에 추가 설치하는 선택 의존성)
  spec.py              # 배포와 로컬이 함께 쓰는 상수 (modal을 import하지 않음)
  modal_app.py         # Modal에 배포되는 앱: 이미지, 저장소, 원격 함수
  remote_exec.py       # 원격 컨테이너 안에서 단계를 돌리는 순수 파이썬 (로컬 테스트 가능)
  modal_backend.py     # 내 PC 쪽: 제출, 추적, 회수, 취소, sync, 비용 추정
  client.py            # Modal API를 감싼 얇은 계층. `import modal`은 이 파일에만 있다
  test_remote_exec.py
  test_modal_backend.py
```

### 3.0 `spec.py` (공용 상수)

배포된 앱과 내 PC 쪽이 어긋나면 안 되는 값을 한 곳에 둔다. `modal_app.py`와 `modal_backend.py`가 둘 다 이 파일을 읽고, `config.json`으로는 바꾸지 않는다(바꾸면 다시 배포해야 하는 값이기 때문).

- 앱 이름 `fizgig-train`, 저장소 이름 `fizgig-models`·`fizgig-runs`
- GPU 허용 목록과 함수 이름 대응: `L40S`(기본)·`A100-80GB`·`RTX-PRO-6000`·`H100` → `train_run_l40s` 등
- 학습 함수 자원: CPU 4코어, 메모리 64GiB, 타임아웃 6시간
- 원격 경로: `/models`, `/runs`, 코드 `/tmp/code`, 파이썬 `/opt/venv/bin/python`
- 받을 모델 파일 4개의 HF 저장소·경로·크기

### 3.1 `modal_app.py` (Modal 쪽)

- **앱 이름** `fizgig-train`(`spec.py`).
- **이미지.** `debian_slim`(Python 3.12) + apt `build-essential`, `libgl1`, `libglib2.0-0` + 저장소의 `requirements.txt`를 `uv_install_deps.py`로 `/opt/venv`에 설치(공식 Dockerfile과 같은 방식). `remote_exec.py`를 이미지에 포함한다.
- **저장소(Volume) 두 개.**
  - `fizgig-models` → `/models` (HF 경로 그대로: `/models/diffusion_models/...`)
  - `fizgig-runs` → `/runs`
- **함수.**
  - `version()` — CPU. 배포된 앱 버전 문자열(`modal_app.py`·`remote_exec.py`·`spec.py`의 sha256 앞 12자리)을 돌려준다.
  - `ensure_models()` — CPU. 위 4개 파일이 `/models`에 없거나 크기가 다르면 `hf_hub_download`로 받는다. 이미 있으면 아무것도 하지 않는다.
  - `train_run_<gpu>(run_key)` — GPU별로 하나씩 등록한다. 허용 목록: `L40S`(기본), `A100-80GB`, `RTX-PRO-6000`, `H100`. 각 함수는 CPU 4코어, 메모리 64GiB, 타임아웃 6시간으로 고정하고 `remote_exec.run(run_key)`를 부른다.

### 3.2 `remote_exec.py` (원격 실행기)

`/runs/<run_key>/`의 `plan.json`을 읽어 단계를 순서대로 실행한다. 로컬 `runner.execute()`의 단계 루프와 같은 규칙이다.

- 원격 `run/state.json`의 `completed_stages`에 있는 단계는 건너뛴다.
- `Training` 단계는 `output/<id>-*-state` 중 필요한 4개 파일(`training_state.json`, `lora.safetensors`, `optimizer.pt`, `rng.pt`)을 모두 가진 가장 최근 폴더가 있으면 `--resume`을 붙인다.
- 단계 출력은 `run/<Stage_Name>.log`에 쓴다.
- **60초마다** 저장소를 커밋한다(로그·체크포인트가 컨테이너 밖에서 보이게). 단계가 끝날 때와 종료할 때도 커밋한다.
- `status.json`을 단계 시작·종료·실패 때 갱신한다(§6.2).
- 코드 압축본(`code/fizgig_src.tar.gz`)을 `/tmp/code`에 풀고 그 코드로 실행한다.

### 3.3 `client.py` (Modal API 계층)

`modal_backend.py`가 쓰는 인터페이스. 테스트는 같은 인터페이스의 가짜 클래스를 쓴다.

| 메서드 | 하는 일 |
|---|---|
| `deployed_version()` | `version()` 원격 호출. 앱이 없으면 `None` |
| `deploy()` | `modal deploy automation/cloud/modal_app.py`를 subprocess로 실행 |
| `ensure_models()` | `ensure_models()` 원격 호출(완료까지 대기) |
| `upload(local_dir, remote_dir)` | `fizgig-runs`에 폴더 업로드 |
| `spawn(gpu, run_key)` | `train_run_<gpu>`를 분리 실행하고 호출 id를 돌려준다 |
| `poll(call_id)` | `running` / `done` / `failed` / `cancelled` / `unknown` 중 하나 |
| `read_text(remote_path)` | 원격 파일 읽기(없으면 `None`) |
| `download(remote_dir, local_dir, exclude)` | 원격 폴더 받기 |
| `delete(remote_dir)` | 원격 폴더 지우기 |
| `cancel(call_id)` | 호출 취소 요청 |

### 3.4 `modal_backend.py` (내 PC 쪽)

| 함수 | 하는 일 |
|---|---|
| `submit(run, cfg, n)` | §5.1의 제출 |
| `follow(run, cfg)` | PC가 켜져 있는 동안 상태 추적·로그 이어 붙이기·회수(§5.2) |
| `fetch(run, cfg)` | 결과 받기와 검증(§5.4) |
| `sync(roots)` | 여러 실행 폴더를 훑어 끝난 원격 작업을 회수(§5.3) |
| `cancel(run)` | 원격 호출 취소(§7.3) |
| `estimate(cfg, n, epochs)` | 예상 시간·비용(§8) |
| `remote_plan(run, cfg, n, run_key)` | `launch_plan()`에 원격 경로를 넣어 계획을 만든다 |

### 3.5 기존 파일 변경

- **`runner.py`**
  - `launch_plan(run, cfg, n, *, target=None)`: `target`이 있으면 파이썬 경로(`/opt/venv/bin/python`), 코드 위치(`/tmp/code`), 모델 경로(`/models/...`), 실행 폴더(`/runs/<run_key>/run`)를 그 값으로 바꾼다. 없으면 지금과 같다.
  - `execute()`: `cfg["backend"] == "modal"`이면 로컬 단계 대신 `modal_backend`로 넘긴다. `state.json`의 `remote.status`에 따라:

    | `remote.status` | `runner run`이 하는 일 |
    |---|---|
    | 없음 | `submit()` 후 `follow()` |
    | `submitted` / `running` | 제출하지 않고 `follow()`만 |
    | `done` | `fetch()` |
    | `fetched` (완료) | 지금의 "Already completed; skipped" 경로 |
    | `failed` / `cancelled` | 같은 `run_key`로 다시 `submit()` (체크포인트부터 이어짐) |
    | `stale` | 새 데이터로 새 `run_key`를 만들어 `submit()`, 옛 원격 폴더는 지운다 |
    | `unknown` | 거부하고 Modal 대시보드를 확인하라고 알린다. `runner cancel --job X`로 `cancelled`로 바꾼 뒤에만 다시 제출된다 |
  - `main()`의 `run`: 클라우드 작업에는 GPU 잠금, `free_comfy_before_training`, `vram_gate`, `model_paths.problems` 검사를 하지 않는다. 한 번의 실행에 로컬 작업과 클라우드 작업이 섞여 있으면 로컬 작업에만 GPU 잠금을 잡는다.
  - 클라우드 작업은 캡션이 빠진 이미지가 하나라도 있으면 **거부**한다(로컬 Gemma 서버를 띄우지 않는다).
  - 새 명령: `sync`(§5.3), `cancel --job X`(§7.3). `status`는 원격 상태를 함께 보여 준다.
  - `doctor`: 클라우드 작업이 있으면 `modal` 패키지, 토큰, `deployed_version()` 응답을 점검한다. 클라우드 작업만 있을 때 로컬 CUDA·로컬 모델 파일 부재는 실패로 세지 않는다.
- **`pipeline/seed.py`**: `runner_config(..., backend="local", modal_gpu="L40S")` 인자를 받아 설정에 넣는다. `--backend`, `--modal-gpu` CLI 인자.
- **`pipeline/export.py`**: `export todos --backend modal --modal-gpu L40S`. 작업 항목에 `backend`, `modal_gpu`를 쓴다.
- **`dataset_tab.py`**
  - "학습 위치: 로컬 / Modal" 드롭다운. `seed_train`과 `todos` 단계의 argv에 `--backend modal`(과 GPU)을 붙인다. 기본값은 로컬이고 로컬일 때는 아무것도 붙이지 않는다.
  - 탭이 처음 열릴 때 백그라운드로 `runner sync`를 한 번 실행하고 결과를 로그에 적는다.
  - 클라우드 단계의 취소는 §7.3, 창 닫기는 §7.4를 따른다.
- **`config.json`**: `cloud.modal` 블록(§9).

---

## 4. 원격 저장소 배치

```
fizgig-models (/models)
  diffusion_models/krea2_raw_bf16.safetensors
  text_encoders/qwen3vl_4b_fp8_scaled.safetensors
  vae/qwen_image_vae.safetensors
  loras/krea2_turbo_lora_rank_64_bf16.safetensors

fizgig-runs (/runs)
  <run_key>/
    plan.json                    # 원격 경로로 된 단계 명령과 써 둘 파일들
    code/fizgig_src.tar.gz       # 제출 시점의 src/ (캐시 파일 제외)
    code/sha256.txt
    run/                         # 원격 실행 폴더 = 로컬 runs/<id>/ 에 대응
      images/*.png, *.txt        # 업로드한 데이터
      dataset.toml               # plan.json의 files에서 생성
      cache/
      output/                    # 최종 LoRA, 학습 상태, sample/, loss_log/
      Cache_Preparation.log, Text_Encoder_Caching.log, Training.log
      state.json                 # 원격 단계 진행 기록
    status.json                  # 로컬이 읽는 요약(§6.2)
```

`run_key`는 `<작업 id>-<데이터셋 지문 앞 12자리>`다. 같은 작업·같은 데이터면 같은 키라서 실패 후 다시 실행하면 같은 폴더에서 이어 간다.

---

## 5. 데이터 흐름

### 5.1 제출 (`runner run`, backend = modal)

1. `prepare(cfg)`로 로컬 실행 폴더를 지금처럼 만든다(이미지·캡션·데이터셋 지문). 캡션이 빠진 이미지가 있으면 거부.
2. `client.deployed_version()`이 로컬 `modal_app.py`+`remote_exec.py` 해시와 다르거나 `None`이면 `client.deploy()`. 첫 배포는 이미지 빌드 때문에 10~20분 걸릴 수 있다고 로그에 쓴다.
3. `client.ensure_models()`.
4. `remote_plan()`으로 원격 경로 계획을 만들고, 업로드할 묶음을 임시 폴더에 꾸린다: `plan.json`, `code/`, `run/images/`.
5. `client.upload()` → `client.spawn(gpu, run_key)`.
6. 로컬 `state.json`에 `remote` 블록(§6.1)을 쓰고, 예상 비용을 로그에 남긴다.

### 5.2 PC가 켜져 있는 동안 (`follow`)

- `poll_seconds`(기본 30초)마다 `client.poll()`과 원격 `status.json`을 읽는다.
- 원격 단계 로그를 읽어 로컬 같은 이름 로그에 **새로 생긴 부분만** 이어 붙인다. 원격이 60초마다 커밋하므로 로그는 1~2분 늦게 따라온다.
- 원격 단계가 바뀌면 `emit_event("stage", ...)`를 내서 GUI 진행 표시가 로컬 학습과 같게 움직인다.
- `done`이면 `fetch()`. `failed`/`cancelled`면 §7.2.
- runner 프로세스가 끝나도(창 닫기, PC 종료) 원격은 계속 돈다.

### 5.3 다음에 켤 때 (`runner sync`)

- 훑는 곳: 기본 실행 폴더(`automation/runs/*`)와 모든 프로젝트의 시드 실행 폴더(`<projects_dir>/*/seed_lora/*/runs/*`).
- `state.json`에 `remote`가 있고 `remote.status`가 `submitted`/`running`인 실행만 본다.
- 원격이 끝났으면 `fetch()`, 아직 돌면 "진행 중(경과 시간)"을 출력만 한다. 아무것도 새로 제출하지 않는다.
- `runner run`과 `runner status`도 시작할 때 자기 작업에 대해 같은 확인을 한다.

### 5.4 결과 받기 (`fetch`)

1. 로컬 데이터셋 지문이 `remote.dataset_fingerprint`와 다르면 받지 않는다. "제출 후 데이터가 바뀌었다"고 알리고 `remote.status = "stale"`.
2. 원격 `run/output/`을 로컬 `run/output/`으로, 원격 단계 로그를 로컬로 받는다. `cache/`는 받지 않는다.
3. 기존 `output_valid(run, cfg, epochs)`를 통과해야 로컬 `state.json`을 `complete: true`로 바꾸고 `completed_stages`를 채운다.
4. 통과하면 원격 `/runs/<run_key>/`를 지우고 `remote.status = "fetched"`. 통과하지 못하면 원격은 남기고 실패로 기록한다.

---

## 6. 상태 기록

### 6.1 로컬 `state.json`의 `remote` 블록

```json
"remote": {
  "provider": "modal",
  "app_version": "a1b2c3d4e5f6",
  "call_id": "fc-…",
  "run_key": "suamom_873f40a8c0ad_krea2-76181ae5df38",
  "gpu": "L40S",
  "submitted_at": "2026-11-10T21:00:00",
  "code_sha256": "…",
  "dataset_fingerprint": "…",
  "estimate": {"hours": 1.6, "usd": 4.5},
  "status": "submitted | running | done | fetched | failed | cancelled | stale | unknown",
  "message": "실패·취소 사유 문장"
}
```

### 6.2 원격 `status.json`

```json
{"state": "running | done | failed | cancelled", "stage": "Training", "exit_code": null,
 "message": "", "updated_at": "…", "code_sha256": "…", "dataset_fingerprint": "…",
 "log_tail": "실패 시 마지막 40줄"}
```

---

## 7. 오류 처리와 취소

### 7.1 제출 전 실패 — 아무것도 올리거나 시작하지 않는다

| 상황 | 메시지 요지 |
|---|---|
| `modal` 패키지 없음 | `automation/cloud/requirements.txt` 설치 명령을 알려 준다 |
| 토큰 없음 | `python -m modal token new`를 알려 준다 |
| 워크스페이스 응답 없음 | 네트워크·토큰 확인 |
| 허용 목록에 없는 GPU | 허용 목록을 보여 준다 |
| 캡션 누락 | 누락 파일 수와 첫 몇 개 이름 |
| 업로드 중단 | 업로드만 다시 시도하라고 알린다. `spawn`은 하지 않는다 |

### 7.2 원격 실패

- 단계 종료 코드가 0이 아니면 원격 실행기가 `status.json`에 실패 단계·코드·로그 끝 40줄을 쓴다. 로컬은 로그를 받고 로컬 학습 실패와 같은 문장(`<Stage> exited <code>; see <run>`)으로 알린다.
- `poll()`이 `failed`인데 `status.json`이 `running`에 머물러 있으면(컨테이너가 강제 종료됨) "Modal이 작업을 중단했다 — 크레딧·지출 한도 소진이나 6시간 제한일 수 있다"고 적는다.
- 원격 폴더와 체크포인트는 남는다. 같은 작업을 다시 실행하면 같은 `run_key`로 다시 제출되고, 끝난 캐시 단계는 건너뛰고 마지막 체크포인트부터 이어 학습한다.
- **자동 재시도는 하지 않는다.**
- `poll()`이 `unknown`(호출 id를 Modal이 더 이상 모름)이면 `status.json`으로 판단한다. `done`이면 회수하고, 그 밖이면 `remote.status = "unknown"`으로 표시만 한다. 자동으로 다시 제출하지 않는다.

### 7.3 취소

- GUI 데이터셋 탭의 **취소** 버튼이 클라우드 학습 단계에서 눌리면 실행 폴더에 `cancel_remote.request`를 쓴다. `follow()`가 이 파일을 보면 `client.cancel(call_id)`를 부르고 `remote.status = "cancelled"`로 기록한 뒤 종료한다. 원격 폴더는 이어서 할 수 있게 남긴다.
- CLI: `runner cancel --job X`가 같은 일을 한다.
- 일시정지는 클라우드 작업에 제공하지 않는다(취소 후 다시 실행하면 체크포인트부터 이어진다).

### 7.4 창 닫기·runner 종료

- **원격을 취소하지 않는다.** 데이터셋 탭의 닫기 경로는 클라우드 단계에 대해 `cancel_remote.request`를 쓰지 않고 로컬 runner 프로세스만 끝낸다.
- 닫을 때 로그에 "원격에서 계속 진행 중 — 다음에 열 때 결과를 받는다"고 남긴다.

### 7.5 중복 제출 방지

`state.json`의 `remote.status`가 `submitted`/`running`이면 `runner run`은 새로 제출하지 않고 `follow()`만 한다.

---

## 8. 비용·시간 추정

```
steps   = n × epochs
hours   = overhead_minutes / 60 + steps × sec_per_step[gpu] / 3600
usd     = hours × (gpu_rate[gpu] + cpu_cores × cpu_rate + memory_gib × memory_rate)
```

`cpu_cores`, `memory_gib`는 `spec.py`의 학습 함수 자원 값이다(4, 64).

- 초기 값(실측 전 가정값, `config.json`에서 바꿀 수 있음): `overhead_minutes` 15, `sec_per_step` L40S 1.6 / A100-80GB 1.3 / RTX-PRO-6000 0.9 / H100 0.8. 로컬 RTX 5090 실측은 0.95초/스텝.
- 요금은 §2의 값을 `config.json`에 둔다.
- 결과는 제출 로그와 `remote.estimate`에만 남긴다. 확인창은 없다.
- 실기 확인(§10.2) 뒤 `sec_per_step`을 실측값으로 바꾼다.

---

## 9. 설정

### 9.1 작업(todo 항목 또는 시드 설정)

| 키 | 값 | 기본 |
|---|---|---|
| `backend` | `"local"` / `"modal"` | `"local"` |
| `modal_gpu` | `L40S`, `A100-80GB`, `RTX-PRO-6000`, `H100` | `"L40S"` |

### 9.2 `config.json`의 `cloud.modal`

```json
"cloud": {"modal": {
  "poll_seconds": 30,
  "overhead_minutes": 15,
  "sec_per_step": {"L40S": 1.6, "A100-80GB": 1.3, "RTX-PRO-6000": 0.9, "H100": 0.8},
  "gpu_rate": {"L40S": 1.95, "A100-80GB": 2.50, "RTX-PRO-6000": 3.03, "H100": 3.95},
  "cpu_rate": 0.047, "memory_rate": 0.008
}}
```

블록이 없으면 위 값을 기본으로 쓴다. 여기 있는 값은 폴링 간격과 비용 추정에만 쓰이고, 배포에 영향을 주는 값은 `spec.py`에 있다. 토큰은 여기에 두지 않는다.

### 9.3 인증

Modal 표준 토큰(`%USERPROFILE%\.modal.toml` 또는 `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` 환경 변수)을 그대로 쓴다. 저장소·todo·로그에 토큰 값을 쓰지 않는다.

---

## 10. 테스트와 검증

### 10.1 자동 테스트 (네트워크·GPU 없이)

- `cloud/test_remote_exec.py` — 가짜 단계 명령(짧은 파이썬 스크립트)으로:
  - 단계 순서대로 실행, `completed_stages` 건너뛰기
  - 완전한 학습 상태 폴더가 있을 때만 `--resume`, 불완전한 폴더는 무시
  - 실패 시 `status.json`에 단계·코드·로그 끝
  - 코드 압축본을 풀어 그 코드로 실행
- `cloud/test_modal_backend.py` — 가짜 `client`로:
  - `remote_plan()`에 Windows 경로(`G:\`, `\\`, `.exe`)가 남지 않는다
  - 제출하면 `remote` 블록이 생기고 `spawn`이 한 번 불린다
  - 진행 중인 `remote`가 있으면 다시 실행해도 `spawn` 0회
  - §3.5 표의 `remote.status`별 동작 7가지
  - 배포 버전이 다를 때만 `deploy()`
  - `sync()`: 끝난 작업만 받고, `output_valid`를 통과해야 완료, 통과하면 원격 삭제
  - 데이터셋 지문이 다르면 `stale`, 받지 않음
  - 실패·강제 종료·`unknown` 처리와 문장
  - `cancel_remote.request` → `cancel()` 호출, 창 닫기 경로는 `cancel()` 0회
  - 로그 이어 붙이기가 중복 없이 새 부분만 붙인다
- `test_runner.py` — 클라우드 작업에 GPU 잠금·VRAM 검사·ComfyUI 비우기·로컬 모델 검사를 하지 않음, 캡션 누락 거부, `doctor`의 클라우드 점검, 로컬/클라우드가 섞인 실행
- `test_dataset_flow.py` — 드롭다운이 `seed_train`·`todos` argv에 `--backend modal`을 붙이고 로컬일 때는 붙이지 않음, 탭이 열릴 때 `sync` 1회
- `pipeline/test_pipeline_*` — `export todos --backend`, `seed --backend`가 설정에 들어감

### 10.2 실기 확인 (구현 후, 사용자 승인을 받고 크레딧 사용)

1. 첫 제출: 배포(이미지 빌드)와 `ensure_models` 30GB 다운로드 시간을 잰다.
2. 짧은 학습: export `873f40a8c0ad`(36장) × 2에포크, L40S. 스텝당 시간과 실제 비용을 재서 `sec_per_step`을 갱신한다. 예상 $1 미만.
3. 받은 LoRA로 기존 홀드아웃 그리기(`evaluate holdout --draw --purpose final`)가 로컬 결과와 똑같이 동작하는지 본다.
4. 제출 직후 runner를 끝내고, 학습이 끝난 뒤 GUI를 열어 `sync`가 결과를 받는지 본다.
5. 학습 도중 취소 버튼 → 원격이 멈추고, 다시 실행하면 체크포인트부터 이어지는지 본다.

---

## 11. 사용자가 할 일 (구현 후 한 번)

```powershell
cd 'G:\데이터셋 통합작성툴\Fizgig_auto'
.\venv\Scripts\python.exe -m pip install -r automation\cloud\requirements.txt
.\venv\Scripts\python.exe -m modal token new
```

`modal token new`는 브라우저로 Modal에 로그인해 이 PC 전용 토큰을 만든다.

참고: 다른 세션이 만든 `modal-gui` 폴더의 `README.md`에 기존 토큰이 평문으로 적혀 있다. 그 폴더를 공유할 일이 있으면 Modal 설정 → API tokens에서 그 토큰을 지우는 것이 좋다.
