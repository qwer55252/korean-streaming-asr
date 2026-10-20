# Korean Streaming ASR

노트북에서 실행하는 **한국어 실시간 마이크 자막**. Whisper 모델은 로컬 서버에 한 번만 로드하며 음성은 외부 API로 전송하지 않습니다. 최초 설치에는 인터넷이 필요합니다.

- Apple Silicon: whisper.cpp + Metal GPU 가속
- Linux / Windows: 기본 CPU 실행 (성능은 CPU에 따라 다름)
- 한국어 고정, 임시 결과와 발화 종료 후 확정 결과 구분
- 마이크 입력과 추론을 분리하고 오래된 대기 중 임시 결과는 최신 것으로 교체
- WAV 재생 테스트, JSONL 출력, 마이크 선택 지원

**실행 기준:** M1 MacBook Air / 통합 메모리 8GB를 초기 개발 대상으로 삼습니다. 같은 RAM 용량만으로 다른 CPU에서 같은 지연을 보장하지 않습니다. 모델 학습이 아닌 사전 학습 모델의 추론 프로젝트입니다. 우선 `base`로 시작하고 한국어 정확도가 부족하면 `small`을 비교하세요. 영어 전용 `.en` 모델은 사용하지 않습니다.

## 설치

Python 3.11 이상(권장 3.12), Git, C/C++ 컴파일러가 필요합니다. 모델/빌드용 여유 공간은 최소 3GB를 권장합니다. whisper.cpp는 **v1.8.2의 고정 커밋**을 사용합니다.

### macOS (Apple Silicon 권장)

Command Line Tools가 없다면 `xcode-select --install`로 설치하세요. Homebrew를 쓰는 경우:

```bash
brew install python@3.12 git portaudio
git clone https://github.com/qwer55252/korean-streaming-asr.git
cd korean-streaming-asr
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/setup.py --model base
```

Apple Silicon에서는 ARM64 Python을 사용하세요 (`python -c 'import platform; print(platform.machine())'` → `arm64`). Metal은 macOS 빌드에서 활성화됩니다.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install git build-essential python3-venv python3-dev libportaudio2
git clone https://github.com/qwer55252/korean-streaming-asr.git
cd korean-streaming-asr
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/setup.py --model base
```

Python 3.11 이상인지 먼저 확인하세요. 기본 구성은 CPU이며 NVIDIA GPU 자동 설정은 현재 포함하지 않습니다.

### Windows (PowerShell)

Python 3.12, Git, Visual Studio 2022 Build Tools의 **Desktop development with C++**와 Windows SDK를 설치한 뒤:

```powershell
git clone https://github.com/qwer55252/korean-streaming-asr.git
cd korean-streaming-asr
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/setup.py --model base
```

아래 명령의 `python`과 `korean-asr`은 Windows에서는 각각 `.\.venv\Scripts\python.exe`와 `.\.venv\Scripts\korean-asr.exe`로 실행할 수 있습니다. Windows/Linux 실제 마이크 및 추론 성능은 별도 검증이 필요합니다. CI는 Python 처리 로직을 검사합니다.

## 실행: 터미널 두 개

첫 번째 터미널에서 모델 서버 실행:

```bash
source .venv/bin/activate
python scripts/server.py --model base
```

모델 로드가 끝나고 서버가 준비되면 두 번째 터미널에서:

```bash
source .venv/bin/activate
korean-asr
```

마이크 권한 팝업을 허용하고 한국어로 말하세요. macOS에서 입력이 안 되면 시스템 설정 → 개인정보 보호 및 보안 → 마이크에서 실행 중인 터미널 앱의 권한을 확인하세요. `Ctrl+C`로 클라이언트를 종료하면 처리 중인 발화의 확정 결과까지 기다립니다. 서버는 첫 번째 터미널에서 별도로 종료합니다.

```text
[임시 #0] 안녕하세요 ... (추론 시간)
[확정 #0] 안녕하세요. ... (추론 시간)
```

같은 번호의 임시 결과는 **전체 발화를 대체하는 가설**입니다. 이어 붙이지 마세요. 최종 전사에는 `final`만 사용합니다. 괄호 안 시간은 요청/추론 소요 시간이며 발화부터 확정까지의 전체 지연이 아닙니다.

```bash
korean-asr --list-devices
korean-asr --device 1
korean-asr --json
korean-asr --threshold 0.005 --silence 0.8 --interval 1.0
```

`--threshold`는 RMS 에너지 기준입니다. 조용한 목소리가 누락되면 낮추고 주변 소음으로 계속 인식되면 높이세요. 기본값은 0.008이며 신경망 VAD가 아닙니다.

정확도 비교:

```bash
python scripts/setup.py --model small
# 기존 서버를 먼저 종료한 뒤
python scripts/server.py --model small
```

## 재현 가능한 오디오 검증

PCM16 WAV를 같은 분할/추론 경로에 넣을 수 있습니다. `--realtime`은 100ms씩 입력을 공급합니다. 없으면 추론 대기열에 맞춰 처리하는 기능 테스트이며 실시간 성능 측정이 아닙니다.

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -c:a pcm_s16le test.wav
korean-asr --wav test.wav --realtime --json
python -m pytest -q
```

## 처리 구조와 한계

```text
마이크 (장치 기본 샘플레이트)
 → 100ms 블록 → 16kHz 모노 리샘플링
 → RMS 음성 구간 감지 (앞부분 300ms 보존)
 → 1초마다 누적 발화의 임시 인식
 → 800ms 침묵 또는 12초 길이에서 최종 인식
 → 한국어 임시/확정 이벤트
```

Whisper는 본래 스트리밍 모델이 아니므로 누적 오디오를 반복 인식합니다. 마지막 부분은 후속 입력에 따라 바뀔 수 있습니다. 이 버전은 LocalAgreement에 의한 단어 단위 확정을 구현하지 않습니다.

- 12초 연속 발화는 강제로 분할하므로 경계 단어가 잘릴 수 있습니다. 길이는 `--max-seconds`로 조정할 수 있지만 긴 버퍼는 연산량을 늘립니다.
- 에너지 감지는 배경음과 사람 목소리를 완벽하게 구분하지 않습니다. 소음/음악에서 오인식이나 환각이 발생할 수 있습니다.
- 임시 결과 대기열은 교체하며 확정 결과는 보존합니다. 확정 요청 대기가 4개에 이르면 명시적인 과부하 오류로 종료합니다. 무한 지연을 숨기지 않습니다.
- 기본 마이크 캡처 대기열은 3초로 제한합니다. 입력 오버플로는 오류로 표시합니다.
- `base`의 한국어 정확도는 발음·환경에 따라 부족할 수 있습니다. `small`은 비교 후보이며 이 기기에서의 저지연을 보장하지 않습니다.
- 서버는 인증 없는 개발용 서버이므로 실행 스크립트는 `127.0.0.1`에만 바인딩합니다.
- 기본적으로 오디오나 전사 파일을 저장하지 않습니다. JSONL을 파일로 리다이렉트하면 전사가 저장됩니다.

실측 결과는 [VALIDATION.md](VALIDATION.md)를 참고하세요.

## 라이선스

이 프로젝트 코드는 MIT입니다. whisper.cpp, Whisper 모델 및 Python 의존성은 각자의 라이선스를 따릅니다. [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [Whisper](https://github.com/openai/whisper).
