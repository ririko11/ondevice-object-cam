# 따라하기 가이드

**각 단계 끝의 "확인"이 되면 다음 단계로 넘어가세요.**

| 단계 | 내용 | 어디서 |
|---|---|---|
| 1 | Python 3.11 설치 | PC |
| 2 | 폴더 놓기 + VS Code로 열기 | PC |
| 3 | 프로젝트 전용 환경(.venv311) 만들기 | PC |
| 4 | 사진 준비 (자동, 할 일 없음) | PC |
| 5 | 모델 변환 (노트북 실행) | PC |
| 6 | 데모 검증 (사진/동영상) | PC |
| 7 | 라즈베리파이로 옮겨서 카메라 실시간 실행 | 연구실 |
| 8 | 진행 보고 + 깃허브 | - |

---

## 1. Python 3.11 설치

기존 3.14는 그대로 두고 옆에 따로 설치됩니다 (`Python311` 폴더). VS Code 터미널이나 PowerShell에서:

```powershell
winget install -e --id Python.Python.3.11
```

- 약관 동의 물어보면 `Y`, 허용 창 뜨면 **예**
- 끝나면 터미널을 닫고 새로 열어서:

```powershell
py -0
```

**확인:** 목록에 `3.11`과 `3.14`가 둘 다 보임

---

## 2. 폴더 놓기 + VS Code로 열기

바탕화면 경로(`OneDrive\바탕 화면`)에 한글이 있어서 TensorFlow가 가끔 파일을 못 엽니다. **C 드라이브 바로 아래**에 둡니다.
("로컬 디스크 (C:)"는 화면에 보이는 이름일 뿐, 실제 경로는 `C:\` 라서 한글이 없습니다.)

1. zip을 풀어서 나온 `ondevice-object-cam` 폴더를 **내 PC → 로컬 디스크 (C:)** 에 넣기
2. VS Code → **File → Open Folder** → `C:\ondevice-object-cam`
3. **Terminal → New Terminal** (Ctrl + `)

**확인:** 터미널 줄이 `PS C:\ondevice-object-cam>` 으로 시작

```
C:\ondevice-object-cam\
├── convert_models.ipynb    [PC] 모델 변환 + 비교
├── object_cam.py           [PC·Pi] 실시간 인식 데모
├── check_env.py            [PC·Pi] 버전 확인
├── requirements-pc.txt     PC 설치 목록
├── requirements-pi.txt     라즈베리파이 설치 목록 (참고)
├── README.md, GUIDE.md, .gitignore
└── images\                 ← 4단계에서 사진 넣는 곳
```

---

## 3. 프로젝트 전용 환경 만들기

VS Code 터미널에서 한 줄씩:

```powershell
py -3.11 -m venv .venv311
.venv311\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-pc.txt
python check_env.py
```

- activate에서 빨간 에러 → `Set-ExecutionPolicy -Scope Process Bypass` 입력 후 다시 activate
- `pip install`은 TensorFlow 때문에 몇 분 걸림

**확인:** 터미널 줄 앞에 `(.venv311)`, check_env 결과가 `3.11.x / 1.26.4 / tensorflow 2.14.1`

---

## 4. 사진 준비 (자동이라 할 일 없음)

평가/보정용 사진은 5단계 노트북이 **공개 데이터셋(ImageNet-V2)에서 일상 물건 23종 사진을 자동으로 받습니다** (머그컵, 물병, 키보드, 마우스, 노트북, 핸드폰, 시계, 선글라스, 백팩, 바나나 등, 종류마다 10장, 정답 라벨 있음).
- 처음 한 번만 약 1.2GB 다운로드 (인터넷 속도에 따라 몇 분), 필요한 23종만 `data\everyday`에 풀림
- 끝나면 `data\imagenetv2-matched-frequency.tar.gz`는 지워도 됨

- (선택) 폰으로 찍은 책상 위 물건 사진을 `images` 폴더에 넣으면 INT8 보정 데이터에 같이 쓰이고, 인식 결과도 보여줍니다.
- 안 넣어도 진행됩니다.

---

## 5. 모델 변환 (노트북)

1. 왼쪽에서 `convert_models.ipynb` 클릭
2. 오른쪽 위 **Select Kernel → Python Environments → .venv311** 선택
   (Jupyter 확장 설치하라고 하면 설치)
3. **Run All**

| 셀 | 하는 일 | 확인 |
|---|---|---|
| 1 | 버전 확인 | 3.11 / 1.26.4 / 2.14 |
| 2 | 일상 물건 사진 자동 다운로드 (처음 1회 약 1.2GB) | 물건 사진 썸네일이 보임 |
| 3 | MobileNetV2 불러오기 (첫 실행 시 자동 다운로드) | 원본 정확도 출력 |
| 4 | 보정 데이터 준비 | 보정 데이터 (92, 224, 224, 3) 정도 |
| 5 | TFLite 4종 변환 | 파일 4개 + 크기 출력 |
| 6 | 크기/속도/정확도 비교표 + 그래프 | ★ 발표용 |
| 7 | 예시 비교 + `sample.jpg` 저장 | |

**확인:** 폴더에 `mobilenet_*.tflite` 4개, `labels.txt`, `ptq_result.csv`, `ptq_result.png`, `sample.jpg` 생성

---

## 6. 데모 검증 (PC, 카메라 없이)

터미널(`(.venv311)` 상태)에서:

```powershell
python object_cam.py --source sample.jpg
python object_cam.py --bench sample.jpg
```

내 사진으로 해보려면 `--source images\사진이름.jpg`

- 첫 번째: 데모 창이 뜸. **창을 클릭하면 모델이 바뀜** (int8 → fp32 → dynamic → float16), q로 종료
- 두 번째: 모델 4개 크기와 1장당 추론 시간 표 출력 → `bench_result.csv`
- (선택) 폰으로 물건을 이리저리 비추는 10초 동영상 찍어서 `--source test.mp4` → 실시간처럼 확인

**확인:** 창에 사물 이름이 맞게 뜨고, 클릭하면 모델 이름과 ms가 바뀜

---

## 7. 라즈베리파이 (연구실)

1. MobaXterm 터미널: `mkdir -p ~/objcam`
2. MobaXterm 왼쪽 파일 창에서 `objcam` 폴더로 들어가 **8개 드래그**
   `object_cam.py`, `check_env.py`, `labels.txt`, `sample.jpg`, `mobilenet_fp32.tflite`, `mobilenet_dynamic.tflite`, `mobilenet_float16.tflite`, `mobilenet_int8.tflite`
3. 실행:

```bash
source ~/camera_test/.venv311/bin/activate
cd ~/objcam
python check_env.py
python object_cam.py --bench sample.jpg     # Pi에서 모델별 속도 측정
DISPLAY=:0 XAUTHORITY=/home/<본인계정>/.Xauthority python object_cam.py
```

**확인:** 터치스크린에 카메라 영상 + 사물 이름, 화면 터치 시 모델이 바뀌며 FPS가 달라짐 → 사진/영상 촬영

---

## 8. 진행 보고 + 깃허브

6단계 비교표가 나오면 Claude에게 보여주고 보고 문안 + 깃허브 업로드 방법 받기
