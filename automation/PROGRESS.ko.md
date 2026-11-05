# Fizgig Qwen Image 2.1 자동화 진행 기록

작성일: 2026-10-03 (한국 시간)  
최종 재검토일: 2026-11-05 (한국 시간)  
프로젝트: `G:\데이터셋 통합작성툴\Fizgig_auto`  
기록 범위: 모델 준비, 기존 사진 자동 캡션, 학습 자동화 코드, GUI 초안 및 로컬 파일·코드·검사 결과의 재점검.

**최신 갱신 — 데이터셋 파이프라인 M0 (2026-11-05):** 캐릭터 LoRA 데이터셋 통합 파이프라인(`TASK_SPEC_dataset_pipeline.md`)의 첫 마일스톤을 구현했다. 모델 경로 정리, ComfyUI 클라이언트, Krea2 그래프 빌더(a/b/c), 비교 하네스와 계보 기록이며, **14절**에 정리했다. 실제 GPU로 Krea2 구성 (a)(b)를 각 24장 생성하고 재실행 시 추가 생성 0건임을 확인했다. 학습은 실행하지 않았고 M1~M5는 착수하지 않았다. 아래 1~13절은 파이프라인 이전 시점의 기록이다.

**이전 갱신 — GUI 구현:** 한국어 GUI의 설정·TODO·사진/캡션 검수·실행·로그·일시정지/재개 연결과 실행 배치 파일을 완성했다. 실행기 검사 18개, 실제 Tk GUI 검사 9개가 통과했고 화면을 렌더링해 검수했다. 기존 CLI에서 진행 중인 실제 학습에도 읽기 전용으로 연결했다. 학습 완료·최종 품질 확인은 아직 남아 있다. 아래 1~12절은 GUI 완성 이전 시점의 기록이며, 변경된 현재 동작과 검증은 **13절** 및 `README.ko.md`를 따른다.

이 문서에서 **현재 확인**은 재검토 시 파일 또는 검사로 확인한 사실, **이전 기록**은 이번에 다시 실행하지 않은 과거 작업 기록, **미검증**은 실제 동작 확인이 남은 항목을 뜻한다. 최초 재검토에서는 문서만 수정했고, 이후 사용자의 실행 설정 복구 요청에 따라 같은 날 `config.json`에 저장된 Gemma 프로필을 적용했다. 복구 후 환경 검사·회귀 검사·기존 캡션 재사용을 확인했다. LLM 시작, 추가 캡션 생성, 캐시 생성, 학습은 수행하지 않았다.

## 1. 현재 상태

**Qwen 모델 다운로드와 기존 사진 32장의 로컬 Gemma 캡션 생성은 완료됐다. LoRA 실제 학습은 아직 실행하지 않았으며, GUI는 미완성이다.**

| 항목 | 상태 | 확인 내용 |
| --- | --- | --- |
| Fizgig 자체 Python | 확인 완료 | `venv\Scripts\python.exe`가 존재하며 이 실행 파일을 사용한다. |
| Qwen Image 2.1 모델 준비 | 완료 | 공식 가중치 7개와 Fizgig 학습 어댑터 1개가 존재한다. |
| 데이터 준비 | 실제 실행 완료 | 원본 32장을 작업 폴더로 복사하고 중복·손상 검사를 수행했다. |
| 로컬 이미지 캡션 | 실제 실행 완료 | Gemma로 사진 32장에 대한 `.txt` 캡션 32개를 만들었다. |
| TODO 기반 학습 실행기 | 코드 구현 | 준비·캡션·캐시·학습을 순차 처리하는 실행기와 실행 배치 파일을 작성했다. |
| 현재 실행 설정 | 복구 완료 | `config.json`을 저장된 Gemma 프로필의 기존 키·값으로 복구했다. 데이터·학습 설정은 유지했다. |
| 환경 검사 | 통과 | `doctor` 종료 코드 0. CUDA 사용 가능, Gemma 실행 파일·모델·mmproj와 Qwen 경로 모두 OK. |
| 자동화 회귀 검사 | 통과 | 복구한 실제 설정으로 기존 검사 11개 통과. 실제 GPU 학습 검사는 아님. |
| 기존 캡션 재사용 | 확인 완료 | 기존 작업 버전과 일치하며 캡션 32개가 있어 Gemma 로딩을 건너뜀. |
| VAE·텍스트 캐시 | 미실행 | 완료 산출물이 확인되지 않는다. |
| LoRA 학습·체크포인트 | 미실행 | 실제 학습 로그와 LoRA 출력 파일이 확인되지 않는다. |
| 사진·캡션 검수 GUI | 작성 중 | 화면과 LLM 설정창 초안을 작성했지만 실행 연결과 진입점이 미완성이다. |
| 모델 병합 | 미구현 | 이번 실행기에는 포함하지 않았다. |
| 하룻밤 LoRA 4~5개 생산 | 미검증 | 실제 학습 속도와 연속 학습 성공 여부를 아직 측정하지 않았다. |

이 문서는 현재 상태를 기록한 것이다. CLI를 막던 로컬 설정 누락은 복구됐다. 기존 `automation\README.ko.md`의 실행 방법을 사용할 수 있지만, 실제 캐시·학습과 GUI의 미검증·미완성 항목은 아래와 같이 남아 있다.

## 2. 사용자 요청 및 확정한 구성

원하는 흐름은 다음과 같다.

```text
TODO 작업 목록
  → 기존 사진 읽기·복사·검사
  → 사진을 보는 로컬 LLM으로 설명문 생성
  → 캡션 검수
  → Qwen Image 2.1 LoRA 학습
  → 체크포인트·샘플 검수
```

| 설정 | 선택한 값 |
| --- | --- |
| 이미지 기반 모델 | Qwen Image 2.1 |
| 모델 출처 | [Qwen 공식 Hugging Face 저장소](https://huggingface.co/Qwen/Qwen-Image-2.1) |
| 데이터 준비 방식 | 기존 사진 자동 캡션 → LoRA 학습 |
| 기본 데이터 폴더 | `H:\data_실사\권은비\워터밤` |
| 폴더 변경 방식 | 기본 `dataset_root` 또는 작업별 `dataset_root` 설정 |
| 목표 학습량 | 약 3,000스텝 |
| 캡션 모델 | 로컬 `gemma4-31b-qat-uncen` |
| Gemma 시작 파일 | `C:\LLM\models\Gemma\start-gemma4-64k.bat` |
| Gemma 모델 폴더 | `C:\LLM\models\Gemma\gemma4-31b-qat-uncen` |
| 로컬 연결 주소 | `http://127.0.0.1:18080/v1` |
| GUI 요구사항 | 사진 설명문 생성, 데이터 폴더 변경, 캡션 LLM을 바꾸는 별도 설정창 |

초기에는 OpenAI API를 사용하는 캡션 구성을 작성했으나, 사용자의 선택에 따라 로컬 Gemma 방식으로 전환했다. 실제 사진 캡션은 로컬 Gemma로 생성했다.

## 3. 실행 환경 확인

- Fizgig 자동화 폴더의 Python: `G:\fizgig\Fizgig_auto\venv\Scripts\python.exe`
- 이번 재검토의 실제 Python 버전은 **3.12.10**이다. 기존 문서의 3.12.9 표기를 정정했다.
- PyTorch 2.10.0+cu128, `torch.cuda.is_available() == True`를 현재 확인했다.
- GPU: NVIDIA GeForce RTX 5090. 이번 PyTorch `get_device_properties(0).total_memory` 조회값은 **32,579MiB**다. 이전 기록의 32,607MiB는 조회 방식이 남아 있지 않아 현재 값과 구분한다.
- Pillow 이미지 검사와 Tkinter 8.6 로딩을 현재 확인했다. Hugging Face Hub 사용 가능 여부는 이전 기록이며, 이번에는 다운로드를 다시 실행하지 않았다.
- 처음 Python 실행 파일이 없다고 판단했던 내용을 재확인하여 정정했다. 자체 Python 실행 파일은 존재한다.

환경 확인은 실제 Qwen GPU 학습 성공을 의미하지 않는다. 전체 학습 실행은 아직 남아 있다.

## 4. 모델 다운로드 및 연결

다운로드한 공식 모델 리비전:

```text
d26bb61231c349cf6b7896fa83353113880e1ba3
```

다운로드 목록은 `G:\fizgig\Fizgig_auto\automation\download_manifest.json`에 저장했다.

| 구성 요소 | 위치 | 가중치 파일 수 |
| --- | --- | ---: |
| Transformer / DiT | `G:\fizgig\Fizgig_auto\models\Qwen-Image-2.1\transformer` | 2 |
| Text encoder | `G:\fizgig\Fizgig_auto\models\Qwen-Image-2.1\text_encoder` | 4 |
| VAE | `G:\fizgig\Fizgig_auto\models\Qwen-Image-2.1\vae\diffusion_pytorch_model.safetensors` | 1 |
| Fizgig 학습 어댑터 | `G:\fizgig\Fizgig_auto\models\fizgig_adapter\fizgig_qwen_image_2.1_training_adapter.safetensors` | 1 |

8개 가중치 파일의 합계는 **33,199,583,432바이트, 약 30.92GiB**다. 설정 파일과 다운로드 캐시 등의 추가 용량은 이 합계에 포함하지 않았다.

Fizgig `prefs.json`에 Qwen 2.1 모델 경로를 연결했다. DiT는 폴더 경로를 사용하고, 텍스트 인코더는 첫 번째 샤드인 `model-00001-of-00004.safetensors` 파일을 지정하여 나머지 샤드를 찾도록 구성했다.

`verify_models.py`를 이번에 다시 실행해 네이티브 모델 구조와 텐서 이름·형태 일치 및 로컬 토크나이저 로딩을 확인했다. DiT 297개, 텍스트 인코더 750개 텐서가 일치했고 토크나이저 예문은 3토큰으로 처리됐다. 종료 코드는 0이다. 이 검사는 가중치 메타데이터와 모델 구조 확인이며, 전체 모델의 GPU 학습, 가중치 전체의 체크섬, VAE·학습 어댑터의 텐서 호환성을 검증하지 않는다. 별도의 검증 결과 로그 파일은 남겨두지 않았다.

학습 어댑터 출처: [Fizgig Qwen Image 2.1 Training Adapter](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter).

## 5. 데이터 준비와 실제 캡션 결과

원본 폴더에서 사진 **32장**을 확인했다. 확장자별로 JPG 7장, PNG 23장, WebP 2장이다. 이번 데이터에는 읽을 수 없는 이미지나 정확히 동일한 파일 중복이 발견되지 않았고, 제외된 항목은 0개다.

원본 사진은 수정하지 않고 작업 폴더에 복사했다. 복사본 이름에는 파일 내용의 SHA-256 해시를 사용한다. 현재 데이터 수집은 지정 폴더 바로 아래의 이미지를 대상으로 하며 하위 폴더를 재귀적으로 읽는 기능은 포함하지 않았다.

| 작업 폴더 | 사진 | 캡션 | 의미 |
| --- | ---: | ---: | --- |
| `G:\fizgig\Fizgig_auto\automation\runs\eunbi_waterbomb-34b6f399010671ab` | 32 | 0 | 초기 설정으로 준비한 버전 |
| `G:\fizgig\Fizgig_auto\automation\runs\eunbi_waterbomb-d3a9b48e1970278f` | 32 | 32 | 로컬 Gemma 캡션을 완료한 버전 |

완료된 사진과 캡션은 두 번째 작업 폴더의 `images` 폴더에 함께 있다. 이미지마다 같은 이름의 `.txt` 파일을 만들었고, 모든 캡션에 트리거 `eunbiwb,`를 앞에 붙였다. 캡션은 모두 비어 있지 않으며 확인한 파일에는 생각 과정 표식이 없다. 문장 내용과 LoRA 학습 적합성에 대한 사람의 최종 검수는 남아 있다.

각 작업의 `dataset_manifest.json`에는 원본과 복사본의 대응 관계 및 검사 결과를 저장했다. 로컬 실행 기록은 완료 작업 폴더의 다음 파일에 있다.

- `local_caption_usage.jsonl`: Gemma 캡션 요청 32건의 모델명과 토큰 사용량.
- `gemma_server.log`: 로컬 서버 시작, 멀티모달 모델 로딩, 요청 처리 기록.

로그 32건을 다시 합산한 총 토큰은 **13,535개**다. 입력 10,342개, 출력 3,193개이며 모두 `gemma4-31b-qat-uncen`으로 기록돼 있다. 이전 작업 기록상 유료 OpenAI 캡션 요청은 실행하지 않았다. 현재 코드와 로컬 로그는 Gemma 사용을 뒷받침하지만, OpenAI 계정의 청구 내역을 별도로 조회한 것은 아니다.

## 6. 캡션 지침과 로컬 LLM 실행 방식

캡션 지침은 `G:\fizgig\Fizgig_auto\automation\caption_instructions.txt`에 분리했다. 사진에서 확인할 수 있는 머리 모양, 옷과 색, 자세, 표정, 구도, 촬영 각도, 조명, 배경을 영어 약 60~100단어로 설명하도록 구성했다. 인물의 이름을 추측하거나 미적 평가를 붙이지 않도록 했다. 트리거는 실행기가 추가한다.

원본에 같은 이름의 비어 있지 않은 `.txt` 캡션이 있고 작업 사본의 캡션 파일이 없으면 우선 복사한다. 같은 작업 버전에서 이미 존재하는 비어 있지 않은 캡션은 재개 시 건너뛴다. 캡션 요청에는 EXIF 방향을 반영하고 최대 1024×1024 범위로 축소한 JPEG 이미지를 넣는다. 출력의 `<think>…</think>` 제거와 불완전한 응답 거부를 구현했지만, 모든 모델의 사고 과정 표식을 지원한다고 검증한 것은 아니다.

`local_caption.py`는 `http://127.0.0.1`, `http://localhost`, `http://[::1]`의 `/v1` 주소만 허용하고 프록시와 리다이렉트를 차단한다. 로컬 서버가 준비됐고 모델 별칭이 일치하면 재사용하고, 실행기가 직접 시작한 서버만 캡션 종료 후 닫도록 구현했다. 실제 자동 시작·캡션 생성·종료와 GPU 메모리 반환을 확인했다는 이전 기록이 있으며, 이번 재검토에서는 모델 서버를 다시 시작하지 않았다.

기존에 사용자가 실행한 서버는 자동으로 종료하지 않는다. `execute()`는 **설정한 캡션 주소의 포트가 열려 있으면** 캐시·학습 실행 전에 중단한다. GPU 사용량이나 전체 GPU 프로세스를 검사하는 것은 아니므로, 다른 포트의 LLM·ComfyUI 작업을 탐지하는 기능으로 해석하면 안 된다.

LLM 프로필에는 사용자 배치 파일, Q4_K_M GGUF 모델, BF16 mmproj 경로가 저장돼 있다. 최근에는 배치 파일 방식 외에 llama-server 실행 파일과 GGUF/mmproj를 직접 지정하는 방식도 추가했지만, 이 직접 실행 방식은 아직 검증하지 않았다.

## 7. TODO 및 학습 자동화 코드

`runner.py`에 다음 기능을 구현했다.

- `doctor`: Python·CUDA, 서버 준비 여부, 모델·실행 파일 경로와 데이터 목록 점검. 실제 캡션·캐시·학습 성공 검사는 아니다.
- `prepare`: 원본 사본 생성, 정확히 같은 파일 중복 제거, 손상 검사, 데이터 목록 저장.
- `caption`: 기존 캡션 재사용 및 부족한 사진의 로컬 캡션 생성. `--limit`으로 일부만 처리 가능.
- `plan`: 사진 준비 후 캡션·모델 조건을 검사해 네이티브 학습 명령을 `launch_plan.json`에 저장. GPU 학습은 실행하지 않지만 작업 폴더·manifest를 생성할 수 있다.
- `run`: 활성 TODO 작업의 캡션·캐시·학습을 순차 실행.
- `status`: 현재 설정·원본·지침으로 계산한 작업 버전의 `state.json` 조회. 과거 실행 폴더 전체를 나열하지 않으며 원본 폴더 접근과 정상 설정이 필요하다.

작업 버전은 병합된 설정 전체, 원본 경로·이미지 SHA-256·원본 캡션, 캡션 지침으로 계산한다. 따라서 학습률·스텝·LLM 대기 시간 같은 설정 변경도 새 작업 버전을 만들 수 있다. 이전 작업 폴더에서 캡션을 자동으로 찾아 이관하는 기능은 없다.

큐 잠금, 단계별 로그, 완료 단계 재사용, 마지막으로 완전히 저장한 학습 상태에서 재개하는 코드도 작성했다. 실제 캐시와 학습 재개는 아직 실행 검증하지 않았다. `state.json`은 캐시·학습 실행 과정에서 작성되므로, 캡션까지 완료돼 있어도 `status`는 `not started`를 표시할 수 있다. 캡션 완료 여부는 `images`의 파일과 사용량 로그로 따로 확인해야 한다.

**작업 사본의 캡션·이미지 수정은 버전 계산에 포함되지 않는다.** 완료 캐시 단계도 `state.json`의 이름만 보고 건너뛰므로, 캐시 이후 캡션을 편집하거나 캐시 파일이 손상·삭제됐을 때 자동으로 재생성된다고 보장할 수 없다. 캐시 무효화와 산출물 검증은 보완 과제다. `dataset_manifest.json`에는 전체 실행 설정 사본도 저장되지 않는다.

현재 TODO는 `eunbi_waterbomb` 1개이며, 트리거는 `eunbiwb`, 시드는 42다. 기본 데이터 폴더 외에 작업별 데이터 폴더와 목표 스텝·rank 등을 지정할 수 있다. 별도 승인 질문 없이 순차 처리하는 CLI 흐름을 구현했으며, 전용 승인·검수 상태를 관리하는 기능은 아직 없다.

### 학습 기본값

| 항목 | 설정 또는 예상값 |
| --- | --- |
| 목표 스텝 | 3,000 |
| 데이터 수 | 32장 |
| 배치 / 반복 / 누적 | 각각 1 |
| 계산한 에포크 | 94 |
| 예상 실제 스텝 | 32 × 94 = 3,008 |
| Rank | 8 |
| 이미지 해상도 | 0.5MP |
| Adaptive LR | 0.0002~0.0004 |
| EMA | 0.98 |
| 저장 간격 | 16에포크, 약 512스텝 |

네이티브 Qwen 학습기가 에포크 수를 받으므로 목표 스텝을 가장 가까운 에포크로 환산한다. 캐시 항목 수가 예상 이미지 수와 다르면 중단하도록 작성했다. 위 숫자는 코드가 계산하는 학습 계획이며 실제 학습 완료 수치가 아니다.

학습 성공 후 예정된 위치는 `automation\runs\<작업명>-<버전>\output`이다. 현재 두 작업 폴더에는 학습 출력, 학습 상태 파일, 캐시 완료 산출물 또는 학습 로그가 없다. 체크포인트와 검수 샘플 생성도 아직 검증하지 않았다.

### 일시정지·재개의 현재 한계

- `automation\stop_requested`: Gemma 시작 대기 중, 사진별 캡션 요청 전, 캐시·학습 단계 시작 전에 확인한다. 이미 진행 중인 HTTP 요청이나 `subprocess.call()`로 실행 중인 캐시·학습을 즉시 취소하지 않는다. `prepare` 전체를 중단시키는 검사도 없다.
- `output\.pause_requested`: 학습기가 에포크 경계에서 확인하는 별도 표식이다. 학습 중 큐 표식만 만들면 이 파일이 자동으로 생성되는 것은 아니다. GUI의 일시정지 연결은 미완성이다.
- 재개 전에는 사용한 중단 표식을 확인하고 제거해야 한다. `queue.lock`은 `doctor` 외의 CLI 명령에 적용되며 `status`도 잠금을 요구한다. 남은 잠금은 실행 중인 runner가 없음을 확인한 뒤 처리한다.

## 8. GUI 구현 상태

파일: `G:\fizgig\Fizgig_auto\automation\automation_gui.py`

Tkinter로 다음 화면 초안을 작성했다.

- 작업·학습 탭: 기본 데이터 폴더, 학습 설정, TODO 목록, 단계별 실행 버튼, 로그·상태 표시.
- 사진·캡션 검수 탭: 사진 미리보기, 캡션 편집 영역.
- 캡션 지침 탭: 지침 편집 영역.
- LLM 설정창: 프로필 선택·저장, 모델명, 로컬 주소, 배치 파일, llama-server, GGUF/mmproj, 컨텍스트, 토큰 제한, 대기 시간, 연결 검사.
- 작업 설정창: 작업명, 트리거, 데이터 폴더, 스텝·rank, 시드, 활성 여부.

현재 `LLMDialog`, `JobDialog`와 `App`의 화면 구성 코드는 있으나, `App`에 정의된 메서드는 `__init__`, `setup_style`, `button`, `build`, `build_work`, `build_review`, `build_instructions`뿐이다. 실행·설정 저장·작업 갱신·사진 선택·캡션 저장·일시정지·로그 처리 등의 연결 함수가 빠져 있다. 예를 들어 `build()`는 정의되지 않은 `self.open_llm`을 참조한다. `App().mainloop()` 실행 진입점과 GUI 실행 배치 파일도 없어, 진입점 추가만으로 완성되지 않는다.

따라서 **현재 GUI를 실행 가능한 완성 프로그램으로 볼 수 없다.** 화면을 실제로 띄워 조작하거나 레이아웃을 검수한 이력도 없다. 직접 모델 실행과 취소 요청 처리의 최근 코드 역시 추가 검증이 필요하다.

## 9. 검증 이력과 실행 설정 복구

### 수행한 검증

| 검증 | 결과와 한계 |
| --- | --- |
| Python·CUDA·GPU 확인 | 이번 재확인 완료. Python 버전과 GPU 메모리 조회값 정정 |
| 다운로드 파일 존재·모델 경로 연결 | 현재 파일 점검에서도 확인 |
| 텐서 구조·토크나이저 확인 | 이번 재실행 통과. 전체 GPU 학습 검사는 아님 |
| 원본 32장 준비 | 실제 산출물과 manifest 확인 |
| Gemma 이미지 캡션 | 32장 모두 실제 생성, 요청 로그 확인 |
| 자동 시작한 Gemma 종료 | 이전 실행에서 확인 |
| 복구 전 설정의 자동화 회귀 검사 | 11개 모두 `setUp()`에서 `KeyError('caption_base_url')`. 각 검사 본문은 실행되지 않음 |
| Gemma 프로필을 메모리에 적용한 회귀 검사 | 동일한 기존 검사 11개 통과. 디스크 설정 복구나 실서버 동작 성공을 뜻하지 않음 |
| 복구 후 실제 설정의 자동화 회귀 검사 | 메모리 설정 덮어쓰기 없이 기존 검사 11개 통과, 종료 코드 0 |
| 복구 후 `doctor` | 종료 코드 0. 모델·배치 경로 모두 OK, 원본 32장·제외 0개, Gemma는 실행 중이 아님 |
| 복구 후 기존 캡션 재사용 | 작업 버전 `d3a9b48e1970278f` 일치, 비어 있지 않은 캡션 32개 확인. `captions()`가 Gemma 로딩을 건너뜀 |
| 자동화 Python 구문 검사 | `automation` 바로 아래 `.py` 파일 모두 AST 파싱 통과. GUI 실행 검사는 아님 |
| 최근 직접 실행·취소·GUI 변경 | 해당 경로를 검증하는 테스트와 실제 동작 확인은 여전히 없음 |
| 실제 LoRA 학습 | 미실시 |

회귀 검사에는 원본 보존, 중복·손상 처리, **원본 캡션 변경** 시 작업 버전 변경, 로컬 주소 제한, 이미지 요청 구성, 캡션 재사용, 외부 서버를 종료하지 않는 동작 등이 포함돼 있다. 서버·응답은 모의 객체를 사용하며 실제 Gemma나 Qwen 학습을 실행하지 않는다. 직접 실행 모드, 취소, 캐시 무효화, 학습 재개, GUI는 이 11개 검사의 검증 범위에 포함되지 않는다.

이번 재검토에 사용한 기본 명령은 프로젝트 루트 기준으로 다음과 같다.

```powershell
.\venv\Scripts\python.exe -B automation\verify_models.py
.\venv\Scripts\python.exe -B -m unittest discover -s automation -p test_runner.py -v
powershell -NoProfile -ExecutionPolicy Bypass -File .\automation\run.ps1 -Command doctor
```

복구 전 첫 회귀 검사 시도는 샌드박스의 임시 폴더 접근 오류로 중단됐다. 접근 제한 없이 재실행한 결과가 위의 설정 오류 11건이며, 검사 과정에서 남은 빈 임시 폴더는 정리했다. 원인 분리를 위한 추가 실행에서는 `runner.read_json()`의 `config.json` 반환값에만 저장된 Gemma 프로필을 메모리에서 덮어써 같은 11개 검사가 통과함을 확인했다. 이후 실제 설정을 복구하고 위 회귀 검사 명령을 다시 실행해 **11개 모두 통과**했다. 복구 후 검사에서도 기존 임시 폴더 접근 제한을 피하기 위해 샌드박스 밖에서 실행했다.

### 복구 전 문제와 처리 결과

복구 전 `config.json`에는 다음 문제가 있었다.

```text
caption_model = gpt-4.1-mini
caption_base_url 등 로컬 연결 필수 설정 없음
```

복구 전 TODO를 `runner.settings()`에 넘겨 **`KeyError('caption_base_url')`를 실제 재현했다.** `doctor`, `prepare`, `caption`, `plan`, `run`, `status` 모두 이 설정 처리를 거치므로 환경 검사부터 정상 완료할 수 없었다. 설정 복구로 이 오류는 해소됐다. 다만 `runner.py`의 최상위 예외 처리는 여전히 `ValueError`, `RuntimeError`만 처리하므로, 필수 키가 다시 누락되면 친절한 설정 안내 대신 traceback으로 노출되는 코드 보완 과제는 남아 있다.

`G:\fizgig\Fizgig_auto\automation\llm_profiles.json`의 `Gemma 4 31B · 64K` 프로필을 기준으로 모델명을 `gemma4-31b-qat-uncen`으로 되돌리고, 로컬 주소·배치·GGUF·mmproj 경로·자동 시작·대기 시간을 `config.json`에 복구했다. 현재 프로필의 모든 키·값이 일치한다. 데이터 폴더, 목표 스텝, rank, 학습률, 샘플 프롬프트와 TODO는 유지했다. 과거 설정이 달라진 원인은 확인하지 않았다.

복구한 실제 설정과 현재 TODO로 `context()`를 계산한 결과, 기존 완료 폴더 `eunbi_waterbomb-d3a9b48e1970278f`와 정확히 일치했다. 사진 32장 각각에 비어 있지 않은 캡션이 있음을 확인하고 `captions()`를 호출해 `All captions already present; Gemma load skipped.`를 확인했다. 기존 산출물을 수정하거나 새 작업 버전을 만들지 않았다. `caption_launch_mode: "batch"` 같은 새 키를 추가하는 것만으로도 해시는 달라질 수 있어, 복구에는 프로필의 기존 키·값만 사용했다.

## 10. 작성한 주요 파일

아래 위치는 모두 `G:\fizgig\Fizgig_auto` 기준이다.

| 파일 | 역할 |
| --- | --- |
| `automation\download_models.py` | 공식 Qwen 모델·어댑터 다운로드와 경로 연결 |
| `automation\download_manifest.json` | 다운로드 출처·리비전·가중치 목록 |
| `automation\verify_models.py` | 모델 텐서 구조·토크나이저 확인 |
| `automation\runner.py` | TODO 기반 단계 실행 및 학습 연동 |
| `automation\local_caption.py` | 로컬 이미지 캡션 API 및 서버 생명주기 관리 |
| `automation\config.json` | 기본 데이터·학습·캡션 설정, Gemma 프로필 복구 완료 |
| `automation\todo.json` | 활성 작업 목록 |
| `automation\caption_instructions.txt` | 캡션 작성 지침 |
| `automation\llm_profiles.json` | 로컬 Gemma 설정 프로필 |
| `automation\automation_gui.py` | 미완성 GUI 초안 |
| `automation\test_runner.py` | 자동화 회귀 검사 |
| `automation\run.ps1` | 자체 Python을 사용하는 PowerShell 실행 래퍼 |
| `automation\README.ko.md` | 자동화 사용 방법 |
| `run_automation.bat` | CLI 전체 실행 진입 파일 |
| `check_automation.bat` | CLI 환경 점검 진입 파일 |
| `download_qwen21.bat` | 모델 다운로드 진입 파일 |
| `automation\.gitignore`, `models\.gitignore` | 실행 산출물·모델 등의 버전 관리 제외 |

초기 OpenAI 키 등록용 `set_api_key.ps1`은 로컬 방식 전환 후 제거했다. 기존 Fizgig 핵심 GUI·번역 파일에 있던 수정 사항은 이번 자동화 작업의 성과로 포함하지 않았다.

## 11. 남은 작업 순서

1. **완료:** `config.json`을 저장된 Gemma 프로필의 기존 키·값으로 복구했다. `doctor`와 실제 설정의 회귀 검사 11개가 통과했고, 작업 경로가 `eunbi_waterbomb-d3a9b48e1970278f`와 일치하며 캡션 32개가 재사용됨을 확인했다.
2. 필수 설정 누락에 대한 오류 안내와 설정 파일에 의존하지 않는 검사 준비를 보완한다. 실행 설정 사본 저장, 사본 캡션 변경 후 캐시 무효화, 캡션 완료 상태 표시도 보완한다.
3. GUI의 설정 저장, TODO 편집, 실행 작업 스레드, 진행 로그, 캡션 검수·저장, 안전한 중단·재개를 연결하고 실행 진입점·배치 파일을 추가한다.
4. 직접 LLM 실행·취소·외부 서버 보존·두 중단 표식·학습 재개의 검증을 보강하고 GUI를 실제로 띄워 검수한다. 사용 문서도 검증된 동작에 맞춘다.
5. 기존 캡션 32개의 내용을 검수한 뒤 짧은 Qwen 학습으로 캐시 생성, GPU 메모리, 체크포인트·샘플 저장을 확인한다. 짧은 학습용 설정은 새 작업 버전을 만들므로 캡션 재사용·이관 방식부터 정한다.
6. 전체 약 3,000스텝 학습을 수행하고 사람의 검수로 결과를 확인한다. 완료 기준은 최종 LoRA·학습 상태·샘플 파일의 존재와 실제 로딩·품질 확인이다.
7. 여러 TODO의 연속 실행 시간을 측정한 뒤 하룻밤 처리 가능한 작업 수를 판단한다.
8. 모델 병합이 필요하면 검수한 LoRA와 병합 방식·비율을 정한 뒤 별도 기능으로 구현한다.

현재까지 확인된 실제 결과물은 다운로드된 Qwen 가중치, 준비된 사진 사본, 로컬 Gemma 캡션 32개다. 완성된 LoRA 및 동작 검증을 마친 GUI는 아직 없다.

## 12. 작업 이력 — 2026-10-03 (한국 시간)

### 진행 기록 재검토

요청: 진행 기록 문서를 실제 상태와 대조하고 수정·보충한다.

1. `runner.py`, `local_caption.py`, GUI 초안, 설정·프로필, 모델 목록과 기존 실행 산출물을 확인했다.
2. Python 3.12.10과 CUDA 사용 가능 상태를 재확인하고 문서의 환경 수치를 정정했다. 모델 구조 검사는 DiT 297개·텍스트 인코더 750개 텐서와 로컬 토크나이저 확인을 통과했다.
3. 기존 작업 폴더의 사진·캡션 수와 로컬 요청 로그를 확인했다. 완료된 캡션은 32개이며, 기록된 토큰 합계는 13,535개였다.
4. 로컬 주소 설정 누락을 실제 재현했다. 기존 회귀 검사 11개가 모두 설정 준비 단계에서 실패했으며, Gemma 프로필을 메모리에 적용한 진단에서는 11개 모두 통과했다.
5. 캡션 재사용과 작업 버전 계산, 캐시 무효화 누락, 상태 표시, 일시정지 표식의 한계 및 GUI 미완성 범위를 문서에 보충했다.

### 실행 설정 복구

요청: 실행 설정을 복구한다.

복구 기준은 `automation\llm_profiles.json`의 `Gemma 4 31B · 64K` 프로필이다. `automation\config.json`에 다음 변경을 적용했다.

| 설정 | 복구 전 | 복구 후 |
| --- | --- | --- |
| `caption_model` | `gpt-4.1-mini` | `gemma4-31b-qat-uncen` |
| `caption_base_url` | 누락 | `http://127.0.0.1:18080/v1` |
| `caption_start_script` | 누락 | 프로필의 `start-gemma4-64k.bat` 경로 |
| `caption_model_file` | 누락 | 프로필의 Q4_K_M GGUF 경로 |
| `caption_mmproj_file` | 누락 | 프로필의 BF16 mmproj 경로 |
| `caption_auto_start` | 누락 | `true` |
| `caption_startup_timeout` | 누락 | 300초 |
| `caption_request_timeout` | 누락 | 180초 |

`caption_max_output_tokens`는 기존 300을 유지했다. 데이터 폴더, 목표 3,000스텝, rank 8, 학습률, 샘플 프롬프트, TODO도 유지했다. 기존 작업 버전이 달라지지 않도록 프로필에 없던 키는 추가하지 않았다.

복구 후에는 다음 순서로 확인했다.

1. 실제 `config.json`을 읽어 프로필의 모든 키·값과 일치하는지 확인했다.
2. 환경 검사(`doctor`)가 종료 코드 0으로 완료됐다. Gemma 배치·GGUF·mmproj 및 Qwen 모델 경로가 모두 OK였고, 원본은 읽을 수 있는 고유 사진 32장·제외 0개였다. Gemma 서버는 당시 실행 중이 아니었다.
3. 메모리 설정 덮어쓰기 없이 기존 회귀 검사 11개가 모두 통과했다.
4. 현재 설정으로 계산한 작업 폴더가 `automation\runs\eunbi_waterbomb-d3a9b48e1970278f`와 일치했다. 사진별 비어 있지 않은 캡션 32개를 확인한 뒤 캡션 처리 함수를 호출했고, `All captions already present; Gemma load skipped.`가 출력됐다.
5. 이 결과를 문서의 현재 상태·검증 이력·남은 작업에 반영했다.

복구 단계에서 수정한 파일은 `automation\config.json`과 이 진행 기록이다. 실행기·GUI 소스, 원본 사진, 기존 캡션은 수정하지 않았다. 위 결과는 직전 복구 작업의 실행 결과를 기록한 것이며, 문서화 요청에서 검사를 다시 실행한 것은 아니다.

### 종료 시점과 다음 작업

**실행 설정 복구와 기존 캡션 재사용 확인까지 완료했다.** 추가 LLM 추론, VAE·텍스트 캐시 생성, 실제 LoRA 학습은 실행하지 않았다. 현재 완성된 LoRA는 없고 GUI도 미완성이다.

후속 개발·검증 항목은 11절에 정리했다. 학습 단계로 진행할 때는 먼저 캡션 내용을 검수하고, 짧은 학습으로 캐시·GPU 메모리·체크포인트·샘플 저장을 확인해야 한다. 이번 기록은 전체 학습 성공을 의미하지 않는다.

## 13. GUI 완성 및 실행 연동 — 2026-10-03 (한국 시간)

요청: CLI 안내가 아니라 실제 GUI로 실행할 수 있게 만든다.

### 구현한 동작

- `automation_gui.py`의 미정의 연결 함수를 구현하고 실행 진입점을 추가했다. 한국어 작업·학습, 사진·캡션 검수, 캡션 지침 탭과 LLM·작업·학습률 설정창을 연결했다.
- `run_automation_gui.bat`를 추가했다. `run_automation.bat`도 GUI를 열고, `check_automation.bat`는 같은 GUI에서 환경 검사를 실행한다. 창을 여는 것만으로 새 학습을 시작하지 않는다.
- 별도 실행 프로세스의 로그를 GUI 이벤트 큐로 받고, 캐시·학습 로그 파일을 읽어 캡션 수·학습 스텝·진행률을 표시한다. 검사 성공·실패 후 창을 유지한다. Tk 위젯 갱신은 주 스레드에서만 수행한다.
- 기존 `queue.lock` 소유자의 실행 여부를 읽기 전용으로 검사하고, 실행 중인 기존 큐의 상태·로그를 모니터링한다. 새 학습을 중복 실행하지 않는다. 종료한 프로세스의 잠금만 GUI에서 정리할 수 있다.
- 일시정지 버튼이 큐 중단 표식과 현재 작업의 학습 중단 표식을 함께 기록한다. 학습/재개에서는 선택한 작업의 표식을 제거한다. 진행 중인 캐시나 HTTP 요청을 강제 종료하지 않는다. 실행 중 창을 닫으면 안전하게 멈춘 뒤 닫는다.
- 사진·기존 캡션 32개를 실제 화면에 표시했다. 편집은 작업 사본에만 저장하며, 미저장 캡션을 둔 채 작업·사진을 바꿀 때 저장/취소를 처리한다.
- 실행 설정 사본과 캡션 설정 식별자를 `run_config.json`에 저장한다. 동일 이미지·트리거·캡션 설정·지침의 캡션을 학습 설정이 달라진 새 버전으로 이관한다. 현재 설정과 정확히 일치하는 과거 폴더는 GUI 설정 변경 전에 설정 사본을 기록한다.
- 작업 사본의 실제 이미지·캡션 해시를 저장해 변경 시 캐시 단계를 다시 실행한다. 캐시 헤더·캡션 메타데이터를 검사한다. 입력이 바뀌면 그 이전 학습 상태에서 자동 재개하지 않는다. 지문이 없는 구버전 실행은 이미지 내용 해시와 캐시 캡션이 일치할 때만 호환 재개한다.
- 최종 LoRA 헤더·에포크, 최종 학습 상태 파일, 최종 에포크 샘플을 완료 조건으로 검사한다. 네이티브 학습기에 마지막 에포크가 샘플 주기와 맞지 않아도 최종 샘플을 생성하는 처리를 추가했다.
- `doctor`는 CUDA나 필수 모델·실행 경로가 없으면 실패로 종료한다. GUI에 오류를 남긴다.

### 검증과 현재 실행 상태

- `test_runner.py`: 18개 통과. 기존 11개에 캡션 이관/지침 변경, 사본 지문, 잘못된 완료 기록, 캐시 재생성, 구버전 재개, 환경 검사 실패 처리를 추가했다.
- `test_gui.py`: 실제 Tk 위젯과 가벼운 모의 하위 프로세스로 9개 통과. 사진·캡션 저장, 원본 보존, LLM 프로필 보존, TODO 편집, 비동기 성공/실패, 중단/재개, 기존 큐 모니터링, 미저장 캡션 보호, 작은 창의 로그 표시를 확인했다.
- 작업·사진 검수·LLM 설정 화면을 실제로 렌더링하고 캡처로 확인했다. 잘리던 하단 로그 영역과 종료 시 남는 Tk 콜백·파일 읽기 스레드 문제를 수정했다.
- GUI 구현 중 기존 CLI가 `eunbi_waterbomb-d3a9b48e1970278f`를 실제 학습 중인 것을 발견했다. 검사 시 34/94에포크, 이후 43/94에포크까지 진행했고 캐시 호환 검사가 통과했다. 기존 학습 프로세스나 중단 표식을 변경하지 않았다.
- 개발 검사는 새 실제 GPU 학습이나 LLM 추론을 시작하지 않았다. 현재 진행 중인 학습은 변경 전 로드한 코드를 사용하므로 새 최종 샘플·완료 검증 변경은 다음 실행/재개부터 적용된다.

최종 학습 완료, 생성된 LoRA 로딩·품질 및 여러 TODO의 실제 GPU 연속 실행 검증은 여전히 남아 있다. 직접 llama-server 실행 방식의 실제 모델 시작도 이번 GUI 검사 범위에는 포함하지 않았다.

## 14. 데이터셋 파이프라인 M0 — 사전 정리와 비교 하네스 — 2026-11-05 (한국 시간)

근거 명세: `TASK_SPEC_dataset_pipeline.md` (§4 M0, §7 확정 사항). 이번 절은 **M0만** 다룬다. M1~M5는 아직 구현하지 않았다.

### 14.1 모델 경로 정리 (M0-1)

`prefs.json`의 모델 경로 대부분이 실제로 없는 옛 위치(`G:\fizgig\Fizgig_auto\...`, `G:/ComfyUI_windows_portable_nvidia/...`)를 가리키고 있었다. Fizgig GUI 소유 파일이므로 고치지 않고, `automation/config.json`에 `models.<family>.<prefs_key>` 블록을 새로 두어 그 위에 덮어쓰는 방식으로 해결했다.

- `automation/pipeline/models.py`: `prefs.json`을 읽고 `config.json`의 해당 family 값으로 덮어쓴다. 상대 경로는 `Fizgig_auto` 기준으로 해석한다. `prefs.json`은 읽기만 한다.
- `runner.doctor`가 family별로 모든 모델 파일을 OK/MISSING + 출처(`config.json`/`prefs.json`)와 함께 출력하고, ComfyUI 연결·노드 등록·`CLIPLoader` type `krea2`·각 모델 파일이 ComfyUI에 보이는 이름까지 검사한다.
- **작업 버전 해시는 바뀌지 않았다.** `comfy`/`models`/`pipeline` 블록은 파이프라인 배관이므로 `version_settings()`가 해시에서 제외한다. 기존 `eunbi_waterbomb-d3a9b48e1970278f`가 그대로 나오는 것을 실제로 확인했다. family가 기본값(`qwen_image21`)이 아닐 때만 해시에 들어간다.

실제 `doctor` 실행 결과(확인 완료, 종료 코드 0): qwen_image21 4개·krea2 5개 전부 `config.json` 출처로 OK, ComfyUI `http://127.0.0.1:8190` 연결 OK, 필요한 노드 전부 OK, `CLIPLoader type krea2` OK, Krea2 모델 5개가 ComfyUI 이름으로 해석됨, 작업 데이터 32장 읽기 OK.

### 14.2 저장소 정리 (M0-2)

- 툴2(`qwen 2.1 edit`)에 git을 초기화하고 현 상태를 첫 커밋(`3e83174`)으로 보존했다. `.venv`, `dataset/`, `sheets/`, `logs/`는 제외하고 코드·설정·`characters/` 원본만 넣었다.
- 툴1(`Fizgig_auto`)은 `dataset-pipeline` 브랜치에서 작업한다.

### 14.3 ComfyUI 클라이언트 (M0-3)

`automation/pipeline/comfy.py`. 업로드·제출·대기·취소·`/free`를 지원하고, `automation/state/prompt_journal/<prompt_id>.json`에 **제출 직후** prompt_id를 원자적으로 기록한다. 재시작하면 `recover()`가 `/history`를 조회해 미결 항목을 done/failed/cancelled로 정리한다. 이미지는 임시 파일에 쓴 뒤 `os.replace`로 확정하므로 중간에 죽어도 반쪽 PNG가 남지 않는다. 모델 파일은 절대 경로로 설정되어 있으므로 `/object_info`가 나열하는 이름으로 변환하며, 그 폴더에 없으면 어느 폴더에 있어야 하는지 밝히며 실패한다.

### 14.4 Krea2 그래프 (M0-4)

`automation/pipeline/krea2_gen.py`. 세 구성을 API 형식으로 직접 만든다.

| 구성 | 내용 | 그래프 버전 |
| --- | --- | --- |
| (a) | Turbo DiT + 캐릭터 LoRA | `krea2_t2i_turbo@1` |
| (b) | RAW DiT + 가속 LoRA + 캐릭터 LoRA | `krea2_t2i_raw_turbo@1` |
| (c) | Turbo DiT + 스타일 레퍼런스 + 캐릭터 LoRA (레퍼런스 조건) | `krea2_ref_turbo@1` |

Turbo DiT에 가속 LoRA를 겹치는 구성은 코드가 거부한다. 샘플링은 family 정의(`src/fizgig/families/krea2.py`)대로 8스텝·CFG 1.0·euler·simple이며, mu 1.15는 ComfyUI의 `Krea2` 클래스가 `sampling_settings`로 이미 갖고 있어 별도 노드가 필요 없다.

**가속 LoRA 가시성 문제(실제 확인):** `krea2_turbo_lora_rank_64_bf16.safetensors`는 `models/diffusion_models/krea2/`에 있어 `LoraLoaderModelOnly`가 찾지 못한다. `G:\Lakis\ComfyUI\models\loras\`에 **하드링크**(0바이트 추가, `krea2_turbo_lora_rank_64_bf16.safetensors`)를 만들어 노출했다. 원본 파일은 이동·복사하지 않았다. `doctor`가 이 가시성을 계속 검사한다.

### 14.5 비교 하네스 (M0-5, M0-6)

`automation/pipeline/experiment.py`. `plan` → `run` → `sheet` → `report` 4단계이고, 캐릭터 카드는 `automation/experiments/suamom_card.json`(project.json의 축소판)이다.

- 고정 슬롯 **24개**(angles 8, framing 4, expressions 4, backgrounds 4, poses 4) × 시드. 슬롯 문구는 툴2 캡션이 실제로 쓰는 태그 어휘 안에서 만들었다 — `suamom_v1`가 그 언어로 학습됐기 때문에 다른 어휘로 물으면 모델이 아니라 프롬프트를 측정하게 된다.
- **A**: `qwen 2.1 edit/dataset/수아맘`의 기존 산출물에서 슬롯 태그로 매칭한다. 같은 카테고리(`angles_*` 등) 파일이 동점에서 우선한다.
- **B**: Krea2 구성 (a)(b) × 에포크. 체크포인트는 존재하는 에포크 중 가장 가까운 것을 쓴다.
- `run`은 파일이 이미 있으면 건너뛴다. 점수는 시트에서 1~5로 입력하고 `CSV 저장`으로 `scores.csv`에 쓴다(`--serve`면 그 자리에서 저장, 아니면 다운로드). `report`는 방식별 집계표와 **추천 구성 1개 + 근거**를 만든다.
- 계보는 M0 하네스부터 적용했다(§3.3). 툴2가 남긴 `manifest.jsonl`의 `refs`/`source`를 읽어 **실제 원본(세대 0)까지 parent_id로 연결**하고, Krea2 산출물은 `trained_on_snapshot`으로 LoRA 학습 스냅샷에 연결한다. `write_candidates`는 세대 상한과 편집 깊이 상한을 검사한다.

### 14.6 실제 실행 결과 (확인 완료)

`python -m pipeline.experiment plan --seeds 1003 --epochs 28 --variants a,b` → **72 작업**(슬롯 24 × (툴2 1 + Krea2 2) × 시드 1). 슬롯 3개(`ang_quarter_close`, `bg_classroom`, `frm_dutch`)는 툴2 산출물에 해당 태그가 없어 비어 있다(임의로 다른 그림을 끌어오지 않는다).

`run` 실제 실행 결과: **이미지 69장 생성**(Krea2 (a) 24장, (b) 24장, 툴2 복사 21장), 실패 0건, 3건은 위 사유로 비어 있음. Krea2 1장당 **12~16초**(RTX 5090, 1024×1536, 8스텝)이고 총 20분 정도 걸렸다.

**같은 명령 재실행 결과: `generated 0, skipped 69, failed 0, missing 3`** — 수용 기준 "추가 생성 0건"을 만족한다.

`candidates.jsonl` 69줄: 세대 전부 1, 편집 깊이는 툴2 21건이 1, Krea2 48건이 0, 상한 위반 경고 0건. 툴2 21건의 `parent_ids`는 5개 고유 원본 id로 모두 `characters/수아맘/`의 실제 파일 해시와 일치했다(원본까지 추적 확인). Krea2 48건은 `trained_on_snapshot: "suamom_v1@28"`로 스냅샷에 연결된다.

컨택트 시트(`sheet.html`, 67 KB)가 열리고 슬롯×방식으로 69칸+빈칸 3개를 보여준다. `--serve`로 띄운 로컬 서버에서 점수 저장이 `scores.csv`로 기록되는 것과, 실험 폴더 밖 경로 요청이 404로 막히는 것을 확인했다(자가 점수는 확인 후 삭제했다).

### 14.7 아직 하지 않은 것 / 미검증

- **구성 (c)는 실행하지 않았다.** 그래프 빌더와 단위 검사는 있지만, 슬롯별 레퍼런스 이미지 선택이 아직 없고 `krea2_style_reference.safetensors`가 설치되어 있지 않다(모델 다운로드는 사용자 승인 필요, §5.3). 사유는 하네스 매니페스트의 `variant_notes`와 리포트에 기록된다.
- **생성기 C(Qwen 보정)**는 툴2의 `--fix-jobs`가 필요해 M2로 넘겼다. M0 수용 기준에는 포함되지 않는다.
- **사람 점수는 아직 없다.** `sheet.html`에서 1~5점을 입력해야 집계표와 추천이 나온다. 위 14.6의 점수 저장은 자가 확인용 1줄이었고 삭제했다.
- **학습은 실행하지 않았다.** Krea2 소규모 학습·에포크 그리드는 M1/M3이다.
- 툴1 Tk GUI(`test_gui.py` 9개)에는 파이프라인 탭을 아직 넣지 않았다(M5).
- 툴2 `dataset_batch.py`는 이번에 **고치지 않았다**(M2). §1.2의 알려진 버그 2건과 재개 키 약점은 그대로다.

### 14.8 검사

명세 §5.7의 명령으로 **121개 통과**(기존 27개 + 파이프라인 94개), 실패·오류 0건. 파이프라인 검사는 ComfyUI와 GPU를 모의 객체로 대체하며 네트워크를 쓰지 않는다.

| 파일 | 내용 |
| --- | --- |
| `pipeline/test_pipeline_models.py` | 설정 우선순위, 상대 경로, 출처 표시, 누락 사유, 선택 파일 |
| `pipeline/test_pipeline_comfy.py` | 저널 기록/회수, 원자적 저장, 이름 해석(고유명·모호·실종), 취소, `/free` |
| `pipeline/test_pipeline_krea2.py` | 세 구성의 노드·체인·거부 규칙, 그래프 버전 |
| `pipeline/test_pipeline_lineage.py` | 빠른 해시 두 모드, 세대·편집 규칙, 상한, jsonl 중복 방지, 원본 추적 |
| `pipeline/test_pipeline_experiment.py` | 슬롯 24개, 프롬프트 조립 순서, 매칭, 재실행 0건, 계보 기록, 실패 격리, 시트·점수·집계·추천·리포트·서버 |

기존 `test_runner.py`의 계획 검사 1건은 모델 경로 출처가 `prefs.json`에서 `pipeline.models`로 바뀐 데 맞춰 모의 대상을 바꿨다. 검사가 확인하는 동작(스테이지 순서, 에포크 수, 어댑터 인자)은 그대로다.

### 14.9 다음 (M1 착수 전 확인 필요)

명세 §4에 따라 **다음 마일스톤으로 넘어가기 전에 사용자 확인을 받는다.** M1은 runner의 family 지원(qwen_image21 | krea2)과 캐시·샘플·완료 판정 일반화다. 착수 전에 확인할 것:

1. 위 시트를 보고 **M0 사람 점수**를 넣을지, 아니면 지금 규모(1 시드·1 에포크)로는 부족해 에포크를 늘릴지.
2. Krea2 추론 구성 추천을 받을 준비가 됐는지(§7 Q2). (c)를 포함하려면 `krea2_style_reference.safetensors` 다운로드 승인이 필요하다.
3. 툴2 이미지가 없는 슬롯 3개를 툴2 CLI로 생성할지, 비워 둘지.

