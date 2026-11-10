# Modal 클라우드 학습 구현 계획

작성일 2026-11-10 · 기준 브랜치 `dataset-pipeline` · 설계 커밋 `bf593bf5`

기준 문서 [Modal 클라우드 학습 백엔드 설계](../specs/2026-11-10-modal-cloud-training-design.md)

시드 학습과 최종 LoRA 학습을 작업별로 Modal에 제출한다. PC가 종료되어도 제출된 호출은 계속 실행되며, 다음 실행에서 같은 호출의 상태를 확인하고 기존 학습 폴더 형식으로 결과를 회수한다. 구현은 1~6단계의 오프라인 검증까지 진행한다. Modal 배포와 실기 검증은 별도 승인 후 7단계에서 진행한다.

## 1. 실행 계획과 입력 검증

대상 파일 `automation/runner.py`, `src/fizgig/families/launch.py`, `automation/cloud/spec.py`, `automation/test_runner_family.py`

- [x] `spec.py`에 앱·Volume 이름, GPU와 함수 이름 매핑, CPU 4코어·메모리 64GiB·학습 타임아웃 6시간, 원격 경로, 모델 역할과 HF 파일 매핑을 둔다. 모델 검증용 크기는 HF의 정확한 바이트 수와 고정 리비전으로 기록한다. 설계의 약 30GB는 다운로드 규모 안내에 쓴다. [Krea 2 모델 저장소](https://huggingface.co/Comfy-Org/Krea-2/tree/main)
- [x] `runner.launch_plan(..., target=...)`에서 실행 파이썬·소스·모델·런 경로를 주입한다. 현재 `launch.plan()`은 Windows의 `os.path`로 경로를 조합하고 `exists()`로 Turbo LoRA 포함 여부를 정한다. 원격 대상일 때 POSIX 경로 연산을 쓰고, 준비된 로컬 데이터로 파일·캡션을 확인하며, 모델 역할은 선언된 매핑으로 확인한다. 원격 파일 실재 여부는 컨테이너에서 확인한다. 경로 조합과 모델 확인의 기본 동작은 기존 `plan()` 호출과 같다.
- [x] 공통 epoch 계산을 계획, 예상 비용, 결과 검증에서 함께 쓴다. 캐시와 프롬프트 파일을 포함한 `plan.json`에 단계 argv, 생성 파일, 작업 ID, family, 이미지 수, epoch, 데이터 지문, 코드 SHA-256을 넣는다. 명령은 인자 배열로 저장한다.
- [x] `settings()`에서 `backend`와 `modal_gpu`를 정규화하고, Modal은 `krea2`만 허용한다. 로컬과 Modal의 결과 폴더는 구분한다. 기존 로컬 작업과 명시적으로 `backend: local`을 고른 작업은 같은 버전 해시를 쓰며, `cloud` 비용·폴링 설정과 GPU 선택은 학습 폴더를 바꾸지 않는다.
- [x] 제출 전에 export 검증을 통과하고, 채택한 원본 이미지마다 비어 있지 않은 캡션이 있는지 확인한다. `prepare()`의 작업용 복사본을 쓰되, 클라우드 작업에서 과거 캡션 재사용이나 Gemma 호출로 누락을 채우지 않는다. 원본 이미지·캡션 인벤토리 지문과 작업용 `images/` 지문을 함께 보관한다.

통과 기준. 로컬 Qwen 2.1·Krea 2 명령과 기존 로컬 버전 해시가 유지된다. Windows에서 만든 Modal 계획의 argv와 생성 파일에 드라이브 문자, 역슬래시 경로, `.exe`가 없고, 세 단계 및 Turbo LoRA 경로가 포함된다. 누락 캡션과 미지원 family는 업로드 전에 거부된다.

## 2. 원격 실행과 체크포인트

대상 파일 `automation/cloud/remote_exec.py`, `automation/cloud/test_remote_exec.py`

- [x] 순수 Python 실행기는 명시적으로 전달된 런 경로, 코드 작업 디렉터리, 커밋 콜백을 받는다. 압축 파일의 SHA-256을 비교하고 허용된 `src/` 파일만 작업 디렉터리에 푼다. 압축의 절대 경로·상위 경로 이동·링크를 거부한다. 제출한 소스의 `src`를 `PYTHONPATH`에 두고 그 소스에서 각 명령을 실행한다.
- [x] `Cache Preparation` → `Text Encoder Caching` → `Training` 순서로 실행한다. 데이터·코드·계획이 같은 런에서 완료로 기록된 단계만 건너뛴다. 학습 전 기존 runner와 같은 기준으로 캐시의 실제 스텝 수를 확인한다. 네 필수 파일과 읽을 수 있는 상태 JSON이 있는 최신 체크포인트에만 `--resume`을 붙인다. `Training`의 완료 표시는 최종 LoRA와 마지막 학습 상태 검증 후에 남긴다.
- [x] 단계 로그는 이어 쓰고, 상태 JSON은 원자적으로 교체한다. 60초마다 진행 내용을 Volume에 커밋하며 단계 종료와 최종 상태에서도 커밋한다. `status.json`에 제출 시도 ID를 포함해 과거 시도의 `done`과 구분한다. 실패 시 단계·종료 코드·마지막 40줄을 저장한다. 이미 성공한 같은 시도가 플랫폼에서 다시 실행되면 커밋된 결과를 사용한다. Modal은 GPU 컨테이너 선점 시 같은 입력을 다시 실행할 수 있다. [Volume 커밋과 다시 읽기](https://modal.com/docs/guide/volumes), [Modal 선점 동작](https://modal.com/docs/guide/preemption)

통과 기준. 가짜 명령만으로 단계 순서, 완료 캐시 건너뛰기, 불완전한 체크포인트 제외, 제출 소스 사용, 실패 상태, 중단 후 재개와 완료 시도 재진입을 검증한다.

## 3. Modal 앱과 SDK 경계

대상 파일 `automation/cloud/__init__.py`, `requirements.txt`, `spec.py`, `modal_app.py`, `client.py` (`requirements.txt`는 `automation/cloud/` 아래의 선택 의존성)

- [x] CPU 함수 `version()`·`ensure_models()`와 GPU별 학습 함수를 등록한다. 원본 `requirements.txt`와 `uv_install_deps.py`로 Linux 이미지의 `/opt/venv`를 만들고, `gcc`와 Python 3.12에 맞는 `Python.h`를 확인한다. `PYTHONPATH`와 CUDA wheel 선택을 검증한다. 학습 함수는 `cpu=4`, `memory=65536`, `timeout=21600`으로 선언한다. 도구 차원의 자동 재제출은 설정하지 않는다. [Modal 리소스 단위](https://modal.com/docs/guide/resources)
- [x] 배포 버전은 `modal_app.py`, `remote_exec.py`, `spec.py`와 이미지 빌드 입력의 해시로 만든다. 최초 제출 또는 버전 불일치 때만 배포하고, 배포 후 버전을 다시 확인한다. `ensure_models()`는 파일 크기가 맞으면 재사용하고, 필요한 파일만 내려받아 검증 후 제 위치에 두고 커밋한다. 다운로드용 CPU 함수에 30GB를 받을 수 있는 별도 타임아웃을 둔다. 재사용 컨테이너는 새 업로드와 모델 커밋을 읽기 전에 해당 Volume을 다시 읽는다.
- [x] `client.py`는 로컬에서 Modal SDK를 불러오는 유일한 진입점이다. 설치된 Python으로 배포 CLI를 실행하고 같은 환경·프로필에서 함수와 Volume을 조회한다. `Function.spawn()`에서 받은 호출 ID와 `FunctionCall.from_id(...).get(timeout=0)`으로 재연결한다. 폴링 대기, 만료된 결과, 원격 실행 타임아웃, 취소, 통신 장애를 구분한다. 만료 시 해당 시도의 Volume 상태를 확인한다. 파일 업로드·읽기·회수·삭제와 호출 취소를 가짜 객체로 대체할 수 있게 한다. [분리 실행과 호출 ID](https://modal.com/docs/guide/function-invocation-methods), [호출 결과 만료](https://modal.com/docs/guide/job-queue), [Volume SDK](https://modal.com/docs/sdk/py/latest/Volume)

통과 기준. Modal 패키지가 없는 환경에서도 로컬 runner와 GUI를 불러올 수 있다. SDK 어댑터 테스트는 가짜 객체로만 실행한다. 모델 준비, CUDA wheel, Python 헤더는 실제 승인 후 이미지에서 확인한다.

## 4. 제출·추적·회수·취소

대상 파일 `automation/cloud/modal_backend.py`, `automation/cloud/test_modal_backend.py`, `automation/config.json`

- [x] `submit()`은 배포 및 모델 준비를 마친 뒤에 `plan.json`, 제출 시점의 `src/` 압축본, 작업용 이미지·캡션만 업로드한다. 같은 입력의 명시적 재시도는 저장된 원래 코드·계획과 체크포인트를 사용한다. 데이터가 같아도 설정이 다른 실행이 같은 `run_key`와 충돌하면 계획 식별자로 구분한다. `spawn()` 직전에 시도 ID와 제출 의도를 기록하고, 호출 ID를 받으면 즉시 원자적으로 기록한다. 호출이 생성됐는지 모른 채 끊기면 `unknown`으로 취급해 자동 중복 제출을 막는다.
- [x] `follow()`는 기본 30초 주기로 상태를 읽고 기존 `run`·`stage` 이벤트와 단계 로그를 연결한다. 이미 받은 로그는 바이트 단위로 이어 붙인다. 연결 오류는 보고하되 마지막으로 확인한 상태를 보존한다. 큐 잠금으로 제출을 직렬화하고, 로컬 추적 프로세스가 종료된 뒤에도 기존 미종료 원격 작업을 먼저 확인한다. 같은 실행의 `run`·`sync`·`cancel`이 상태나 로그를 동시에 덮어쓰지 않게 갱신을 직렬화한다.
- [x] `sync()`는 기본 `runs/`, 설정된 프로젝트의 시드 `runs/`, CLI `--runs-dir`을 읽는다. 저장된 `run_config.json`으로 제출 당시의 검증 조건을 복원한다. 진행 중인 호출은 한 번 조회하고, 완료된 호출과 앞서 회수에 실패한 `done`만 회수한다. 빈 큐나 동기화할 상태가 없으면 정상 종료하며 Modal SDK도 불러오지 않는다.
- [x] `fetch()`는 작업용 데이터와 현재 원본 인벤토리 지문을 각각 비교한다. 원본 캡션 편집으로 `context()`의 결과 폴더가 달라졌어도 이전 원격 결과를 채택하지 않는다. 결과와 로그를 임시 폴더로 받은 뒤 기존 `output_valid()`를 호출한다. 검증을 통과하고 게시 직전 지문이 여전히 일치할 때 기존 출력 레이아웃에 게시한다. 로컬 완료 상태를 기록한 다음에 해당 원격 런을 삭제한다. 삭제 실패는 정리 대상으로 남기고, 다운로드 실패는 `done`으로 남겨 다시 회수할 수 있게 한다. 검증 실패 시 원격 폴더를 보존한다.
- [x] `cancel()`은 대상 호출 ID를 지정한다. 호출이 이미 끝난 경우 정상 결과를 회수한다. 호출이 여전히 실행 중이거나 요청 성공 여부를 모르면 기존 상태를 유지한다. `unknown`의 중복 제출 제한은 취소가 확인되거나, 대시보드에서 중단을 확인한 사용자가 CLI로 명시적으로 해제할 때만 푼다.
- [x] 예상 비용은 `n × epochs × sec_per_step`에 준비 시간을 더하고 GPU·CPU·메모리 시간당 요금을 곱한다. 가정값은 `cloud.modal`에, 배포 자원 값은 `spec.py`에 둔다. 결과는 로그와 `remote.estimate`에 저장한다. [Modal 요금표](https://modal.com/pricing)

| 저장된 원격 상태 | `runner run`의 다음 동작 |
| --- | --- |
| 없음 | 입력 검증 후 제출하고 추적 |
| `submitted` / `running` | 기존 호출 추적 |
| `done` | 결과 회수 재시도 |
| `fetched` | 기존 출력 검증 후 완료 건으로 표시. 파일이 훼손됐으면 해당 사실을 표시 |
| `failed` / `cancelled` | 사용자가 다시 실행했을 때 같은 입력의 체크포인트부터 재개 |
| `stale` | 새 데이터로 새 런 제출. 이전 원격 폴더는 이전 호출이 종료된 것이 확인된 후 정리 |
| `unknown` | 해당 시도의 원격 완료 기록이 있으면 회수. 그 외에는 대시보드 확인 안내 |

통과 기준. 가짜 클라이언트로 표의 7가지 분기, 배포 일치 여부, 진행 중 재실행에서 `spawn` 0회, 제출 직후 프로세스 종료, 로그 중복 방지, 설정이 다른 런의 충돌 방지, 원본 편집, 회수 중단·재시도, 출력 검증 실패, 취소와 회수 경쟁, 통신 오류 및 강제 종료 안내를 확인한다.

## 5. runner와 작업 생성 경로

대상 파일 `automation/runner.py`, `automation/pipeline/seed.py`, `automation/pipeline/export.py` 및 기존 테스트

- [x] `runner.main()`에서 `sync`와 `cancel --job`을 일반 실행의 enabled 작업 선택과 분리한다. 완료 회수와 취소는 저장된 상태에서 작업을 찾으므로, 나중에 비활성화된 todo와 프로젝트별 시드 실행도 다룬다. 같은 ID가 여러 런에 해당하면 `--runs-dir` 등으로 대상을 좁힌다. `status`는 대상 호출을 한 번 확인한다. `plan`은 클라우드 작업에도 오프라인으로 동작한다.
- [x] `run`은 cloud 분기로 먼저 보낸다. cloud 전용 `doctor`는 Modal 설치·인증·배포 응답·지원 family·캡션을 확인한다. `run`과 `doctor`의 로컬 전용 검사와 Gemma 연결은 실제 로컬 작업에서만 수행한다. `run`과 `caption`은 `queue.lock`을 먼저 획득하는 순서로 통일하고, `gpu.lock`은 로컬 캡션·캐시·학습 구간만 잡는다. `sync`와 `cancel`은 학습 큐 잠금과 별도로 원격 상태를 갱신한다.
- [x] `seed.runner_config()`와 `seed.main()`에 `--backend`·`--modal-gpu`를 연결하고 `prepare_runner()`가 쓰는 프로젝트 설정에서 전역 `cloud.modal` 값을 이어받게 한다. `seed.train()`을 거쳐도 runner의 종료 이유와 단계 이벤트가 GUI에 도착하게 한다.
- [x] `export.plan_todos()`·`add_todos()`와 `export todos` CLI에 같은 옵션을 연결한다. Modal 선택 시 export가 지원하는 Krea 2 작업만 준비하고 GUI와 CLI에 그 범위를 표시한다. Krea 2가 없거나 호출자가 Modal과 미지원 family를 명시하면 이유를 반환한다. 기존 동일 ID의 todo는 그대로 두고, 선택한 위치와 기존 위치가 다르면 로그에서 구분한다. 준비 버튼으로 생성한 작업은 기존처럼 비활성 상태로 추가한다.

통과 기준. 기존 todo로 로컬 작업을 실행하는 경로가 보존된다. cloud 전용과 혼합 큐에서 호출 순서와 GPU 잠금 범위를 검사한다. export ID 선택, 기존 todo 보존, 프로젝트별 시드 설정, `sync` 빈 큐, `cancel` 대상 탐색을 회귀 테스트한다.

## 6. GUI 연결과 오프라인 회귀

대상 파일 `automation/dataset_tab.py`, `automation/dataset_gui.py`, `automation/automation_gui.py` 및 GUI·프로세스 종료 테스트

- [x] `DatasetTab.start()`에서 선택한 학습 위치를 `seed_train`과 `todos`에 전달한다. 로컬 선택은 기존 argv를 그대로 쓰고, Modal 선택은 `--backend modal --modal-gpu L40S`를 붙인다. 기존 `_with_picked_export()`의 export ID를 보존한다. Modal 시드 학습과 최종 학습 준비에 각각 실행과 todo 생성에 해당하는 안내를 표시한다.
- [x] 공통 `DatasetTab`에서 시작 동기화 작업을 한 번만 비동기로 실행한다. 선택한 프로젝트에 관계없이 설정된 모든 루트를 확인한다. 이벤트 큐로 결과를 기록하고, 회수된 시드 런은 목록에 나타나도록 갱신한다. 두 호스트에서 중복으로 `sync`를 예약하지 않는다.
- [x] 클라우드 시드 학습의 명시적 취소는 시도별 요청 파일을 통해 runner에 전달한다. 요청 경로를 제출 전에 연결해 호출 ID가 생기기 전의 취소도 처리한다. `dataset_gui.PipelineApp.close()`와 `automation_gui.App.close()`는 클라우드 작업에서 로컬 추적만 종료하는 경로로 연결한다. 최종 todo를 실행하는 `automation_gui.App.start()`·`pause()`도 현재 작업의 backend 이벤트를 사용해 Modal 실행에 일시정지를 제공하지 않고 원격 취소를 연결한다. 로컬 작업의 pause·프로세스 트리 종료 회귀를 유지한다.
- [x] 가짜 Modal로 정상 제출→프로세스 종료→새 프로세스 `sync`→결과 회수를 끝까지 검증한다. GUI 테스트에서 각 학습 버튼의 argv, 시작 `sync` 1회, 명시적 취소 시 원격 취소 1회, 두 창의 닫기에서 원격 취소 0회를 확인한다. 전역 `gpu.lock` 대신 테스트용 임시 루트를 쓴다.

저장소 루트에서 전체 회귀 테스트를 실행한다.

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s automation -p 'test_*.py'
```

계획 작성 시점 기준으로 `test_runner*.py`와 `test_dataset_flow.py` 75개가 통과했다. 기본 Windows 샌드박스 임시 폴더에서 권한 오류가 발생해 Python의 `tempfile.tempdir`을 워크스페이스의 `_scratch`로 지정해 재실행했다. 구현 후에도 같은 환경이면 이 설정으로 회귀 테스트를 실행한다.

## 7. 승인 후 실기 검증

1. Modal 인증과 계정의 잔여 크레딧·지출 한도를 확인한다. 앱을 배포하고 CPU 모델 준비 함수를 실행한다. 이미지 빌드와 모델 다운로드 시간을 따로 기록하고, 두 번째 모델 준비 호출에서 파일이 재사용되는지 확인한다.
2. export `873f40a8c0ad` 36장으로 L40S에서 2에포크, 72스텝을 실행한다. 제출 로그에 호출 ID가 기록된 뒤 로컬 runner를 종료한다. 원격 학습이 끝난 후 GUI를 열어 `sync`로 회수하고, 로컬 LoRA·최종 학습 상태·로그와 원격 폴더 정리를 확인한다. 설계의 `$1 미만`은 짧은 GPU 학습의 예상치다. 실기 승인 시 이미지 빌드와 CPU 모델 다운로드는 별도 비용으로 표시한다.
3. 받은 LoRA로 기존 로컬 홀드아웃 그리기를 실행한다. 로컬 GPU를 사용할 수 있는 때에 검증한다. 최종 LoRA, 마지막 epoch 상태와 홀드아웃이 같은 작업을 가리키는지 확인한다.
4. 별도 승인된 실기 작업에서 체크포인트가 저장된 다음 취소 버튼을 누른다. 원격 호출 종료를 확인하고 같은 입력으로 다시 실행해 최신 체크포인트를 읽는지 본다. 측정한 스텝당 시간과 실제 청구 사용량으로 비용 추정치를 갱신한다.

완료 기준. 오프라인 회귀가 통과하고, 승인된 실기에서 제출한 코드와 데이터의 결과가 PC 종료 후에도 기존 출력 경로로 회수된다. 취소된 작업은 같은 입력으로 재개되며, Krea 2 최종 LoRA는 기존 홀드아웃 도구에서 열린다.
