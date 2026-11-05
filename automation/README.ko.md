# Fizgig Qwen 2.1 자동화

`TODO → 원본 복사·정확히 같은 파일 중복 제거·손상 검사 → 로컬 Gemma 이미지 캡션 → Gemma 종료 → VAE 캐시 → 텍스트 캐시 → LoRA 학습 → 체크포인트·샘플 저장`을 순차 실행합니다.

Fizgig_auto 자체의 `venv/Scripts/python.exe`를 사용합니다. 기본 실행 방식은 한국어 GUI입니다.

## GUI 실행

- 루트의 `run_automation_gui.bat` 또는 `run_automation.bat`를 더블클릭합니다. 창을 여는 것만으로 학습을 새로 시작하지 않습니다.
- `check_automation.bat`도 GUI를 열어 환경 검사를 실행합니다. 검사 결과와 오류는 창의 로그에 계속 남습니다. 이미 실행 중인 큐가 있으면 그 작업을 모니터링합니다.
- **작업 & 학습:** 데이터 폴더, 스텝·rank·해상도·학습률을 저장하고 TODO를 추가·편집합니다. 실행 대상을 선택한 작업 또는 활성 작업 전체로 정합니다.
- **사진 · 캡션 검수:** 사진과 기존 캡션을 미리 보고 작업 사본의 캡션을 편집·저장합니다. 원본 사진·캡션은 수정하지 않습니다.
- **캡션 지침:** 영어 설명문 생성 지침을 편집합니다. 지침을 바꾸면 이전 지침의 자동 생성 캡션을 이관하지 않습니다.
- **LLM 설정:** 저장된 프로필, 로컬 서버 주소·모델 별칭, 배치 파일 또는 직접 실행할 llama-server/GGUF/mmproj, 토큰·대기 시간을 바꿉니다. ‘연결 확인’은 이미 실행 중인 서버를 검사합니다.
- **환경 검사 / 사진 준비 / 캡션 1장 테스트 / 전체 캡션 / 캡션 + LoRA 학습·재개** 버튼으로 각 단계를 실행합니다. 작업은 별도 프로세스에서 돌아가며 로그·진행률을 창에 표시합니다.
- **일시정지 요청:** 캡션은 진행 중인 요청이 끝난 뒤, 캐시는 현재 단계가 끝난 뒤, 학습은 에포크 경계에서 상태를 저장한 뒤 멈춥니다. 학습·재개 버튼은 선택한 작업의 중단 표식을 제거합니다.
- 기존 CLI 큐가 실행 중이면 GUI는 새 학습을 시작하지 않고 기존 로그에 연결합니다. 창을 실행 중에 닫으면 안전한 일시정지를 요청하고 작업이 멈춘 뒤 닫습니다.
- 프로세스가 종료됐는데 잠금만 남았으면 **실행 잠금 정리**를 사용합니다. 실행 중인 소유자가 확인되면 잠금을 지우지 않습니다.

학습 설정만 변경할 때는 같은 이미지·트리거·캡션 모델 설정·지침의 이전 캡션을 새 작업 버전으로 이관합니다. 실행 설정은 `run_config.json`에 저장합니다. 최초 GUI 저장 시 현재 설정과 정확히 일치하는 기존 작업 폴더의 설정도 기록해 기존 32장 캡션을 이어서 사용할 수 있게 합니다.

캡션·작업 사본이 바뀌면 캐시 완료 기록을 무효화합니다. 캐시 파일의 헤더·캡션 메타데이터도 확인합니다. 완료 판정에는 최종 LoRA의 헤더·에포크, 마지막 학습 상태 파일과 최종 에포크 샘플이 필요합니다. 입력이 바뀐 작업은 이전 입력의 학습 상태에서 자동 재개하지 않습니다.

## 최초 실행

1. 모델 다운로드 완료를 확인합니다. 중단되었으면 루트의 `download_qwen21.bat`를 다시 실행합니다. 이어받기를 지원합니다.
2. `check_automation.bat`에서 Qwen 모델들과 Gemma 배치·모델·mmproj 파일이 OK인지 확인합니다.
3. `run_automation.bat`로 GUI를 열고 작업을 선택한 뒤 원하는 실행 버튼을 누릅니다. ‘활성 작업 전체’를 선택하면 TODO의 활성 작업들을 차례대로 처리합니다. 캡션은 로컬 Gemma 서버에만 요청하며 OpenAI 키와 유료 API가 필요하지 않습니다. 각 단계 로그는 작업 폴더에도 기록됩니다.

자동화가 지정한 `C:/LLM/models/Gemma/start-gemma4-64k.bat`를 숨김 실행합니다. 모델은 `gemma4-31b-qat-uncen`, 주소는 `http://127.0.0.1:18080/v1`입니다. 모델·MTP·mmproj·64K 설정은 원래 배치 파일을 따르며 그 파일을 수정하지 않습니다. 개별 캡션 요청은 추론 과정을 끄고 최종 설명문만 저장합니다.

자동화가 시작한 Gemma 프로세스만 캡션 후 종료해서 Qwen 학습에 GPU를 넘깁니다. 이미 사용자가 켜 둔 Gemma 서버는 재사용하고 종료하지 않습니다. 그 경우 학습은 캡션 후 멈추며, 해당 서버를 직접 닫고 학습·재개 버튼을 누르면 저장된 캡션을 재사용합니다. Gemma가 꺼져 있으면 학습 버튼에서 서버 시작·캡션·서버 종료를 자동 처리합니다.

## 변경할 파일

- `config.json`: 기본 `dataset_root`, 목표 3000스텝, 해상도, rank, 학습률, Gemma 캡션 모델·주소·배치 파일, 샘플 프롬프트.
- `todo.json`: 작업 목록. 작업마다 `id`, `trigger`, `dataset_root`, `rank`, `target_steps`, `seed` 등을 덮어쓸 수 있습니다. `enabled: false`는 건너뜁니다.
- `caption_instructions.txt`: 영어 캡션 지침. 트리거는 코드가 앞에 붙입니다. 기존 `.txt` 캡션이 있으면 우선 사용합니다.

현재 기본 폴더는 `H:/data_실사/권은비/워터밤`입니다. 바꾸려면 `config.json`의 `dataset_root`만 수정합니다. JSON 경로에는 `/`를 쓰는 것이 편합니다. 원본 폴더 바로 아래의 이미지만 읽습니다.

TODO 예시:

```json
[
  {"id": "eunbi_waterbomb", "trigger": "eunbiwb", "seed": 42},
  {"id": "subject_b", "trigger": "subjectb", "dataset_root": "H:/other_dataset", "seed": 43}
]
```

같은 32장과 같은 설정으로 작업명만 바꾸면 서로 비슷한 LoRA가 나옵니다. 서로 다른 대상이나 스타일을 만들려면 데이터셋을 작업별로 구분합니다. 작업 4~5개를 등록하면 차례대로 실행하며, 하룻밤 완료 수는 실제 학습 속도와 캐시·샘플 시간에 따라 달라집니다.

## 스텝과 검수

이 버전의 Qwen 학습기는 에포크 수를 받습니다. 실행기는 batch=1, repeats=1, accumulation=1로 고정하고 이미지 수로 목표에 가장 가까운 에포크를 계산합니다. 32장이면 94에포크, 약 3008스텝입니다. 캐시된 항목 수가 이미지 수와 다르면 학습을 중단하므로 예상 스텝 수가 조용히 달라지지 않습니다.

기본값은 Fizgig의 Qwen Fast 프리셋(rank 8, Adaptive LR 2e-4~4e-4, 0.5MP, EMA 0.98, 학습 어댑터)입니다. 약 500스텝마다 중간 LoRA와 같은 시드의 샘플을 남깁니다. 3000스텝이 항상 최적이라는 뜻은 아니므로 중간 결과도 검수합니다. rank를 16으로 올리면 Adaptive LR 범위도 1e-4~2e-4 등으로 조정합니다.

결과: `automation/runs/<작업명>-<버전>/output/` 안의 `.safetensors`와 `sample/`.

설정·캡션 지침·원본 파일이 바뀌면 별도 작업 버전이 만들어집니다. 산출물 검사를 통과한 완료 버전은 재학습하지 않습니다. 중단된 작업은 현재 입력과 일치하고 정상인 캡션·캐시 단계를 재사용하며, 같은 입력으로 완전히 저장한 학습 상태에서 재개합니다. 체크포인트 사이의 진행은 다시 학습할 수 있습니다. 마지막 에포크가 샘플 간격과 맞지 않아도 최종 검수 샘플을 생성합니다.

## 개별 단계와 상태

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command prepare
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command caption
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command plan
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command run -Job eunbi_waterbomb
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command status
```

`prepare`는 원본 복사만 합니다. `plan`은 캡션·모델이 준비된 뒤 실행 명령을 저장합니다. `run`은 추가 승인 질문 없이 순차 실행하며 실패 시 다음 작업으로 넘어가지 않습니다. GUI나 ComfyUI의 GPU 작업과 동시에 실행하지 않는 편이 좋습니다.

한 장만 먼저 캡션을 확인하려면 루트에서 `venv\Scripts\python.exe automation\runner.py caption --limit 1`을 실행합니다. Gemma 로딩·실행 로그는 각 작업의 `gemma_server.log`, 로컬 토큰 사용 기록은 `local_caption_usage.jsonl`에 남습니다. 캡션 재개 시 기존 파일을 건너뜁니다.

진행 중 `output/.pause_requested`를 만들면 에포크가 끝난 뒤 멈추고 전체 큐도 중단됩니다. 재개하려면 해당 파일을 지우고 다시 실행합니다. 강제 종료 후 `automation/queue.lock`만 남았으면 실행 중인 runner가 없는 것을 확인한 후 지웁니다.

이번 구성은 기존 사진 캡션과 LoRA 학습입니다. 이미지 생성·일러스트 변환·모델 병합은 포함하지 않습니다. 병합은 검수한 LoRA와 비율을 정한 뒤 별도 단계로 추가하는 것이 좋습니다.

모델 출처: https://huggingface.co/Qwen/Qwen-Image-2.1

Fizgig 학습 어댑터: https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter

로컬 이미지 캡션 API 형식: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
