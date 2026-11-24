"""
[PC / 라즈베리파이 공용] 실시간 사물 인식 + 경량화 모델 비교 데모

실행 예
  라즈베리파이(카메라) : DISPLAY=:0 XAUTHORITY=/home/<계정>/.Xauthority python object_cam.py
  PC(사진)            : python object_cam.py --source images/cup.jpg
  PC(동영상)          : python object_cam.py --source test.mp4
  속도 측정(벤치마크)  : python object_cam.py --bench images/cup.jpg
  화면 없이 결과 저장  : python object_cam.py --source images/cup.jpg --save out.png

조작 (터치스크린 = 마우스 클릭)
  화면 터치 또는 m : 다음 모델로 바꾸기 (fp32 -> dynamic -> float16 -> int8)
  s : 화면 캡처 저장,  q : 종료

같은 폴더에 있는 mobilenet_*.tflite 파일과 labels.txt 를 자동으로 찾아서 사용합니다.
"""
import argparse
import csv
import glob
import os
import platform
import time

import cv2
import numpy as np

# 라즈베리파이: tflite_runtime / PC(Windows): tensorflow 안의 같은 TFLite 엔진(2.14)
try:
    from tflite_runtime.interpreter import Interpreter
    BACKEND = "tflite_runtime"
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    BACKEND = "tensorflow.lite"

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ORDER = ["fp32", "dynamic", "float16", "int8"]
W, H = 800, 480                 # 라즈베리파이 7인치 터치스크린 해상도
VIEW_W, VIEW_H = 600, 450       # 영상 표시 영역
FONT = cv2.FONT_HERSHEY_SIMPLEX


def find_models(folder):
    found = {}
    for path in glob.glob(os.path.join(folder, "mobilenet_*.tflite")):
        key = os.path.basename(path)[len("mobilenet_"):-len(".tflite")]
        found[key] = path
    ordered = [k for k in MODEL_ORDER if k in found] + sorted(k for k in found if k not in MODEL_ORDER)
    return [(k, found[k]) for k in ordered]


def load_labels(folder):
    with open(os.path.join(folder, "labels.txt"), encoding="utf-8") as f:
        return [line.strip() for line in f]


class Classifier:
    def __init__(self, name, path, threads=4):
        self.name = name
        self.path = path
        self.size_mb = os.path.getsize(path) / 1024 / 1024
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.out = self.it.get_output_details()[0]
        self.size = int(self.inp["shape"][1])

    def preprocess(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        s = min(h, w)                                   # 가운데를 정사각형으로 자르기
        crop = frame_bgr[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        img = cv2.resize(crop, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # 모델은 RGB로 학습됨
        x = img.astype(np.float32)[None]                # 0~255 그대로 (정규화는 모델 안에서 함)
        dtype = self.inp["dtype"]
        if dtype != np.float32:                         # 입력이 정수형인 모델 대비
            scale, zero = self.inp["quantization"]
            x = np.clip(np.round(x / scale + zero), np.iinfo(dtype).min, np.iinfo(dtype).max)
        return x.astype(dtype)

    def predict(self, frame_bgr):
        x = self.preprocess(frame_bgr)
        t0 = time.perf_counter()
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        y = self.it.get_tensor(self.out["index"])[0]
        ms = (time.perf_counter() - t0) * 1000
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y.astype(np.float32), ms


def open_source(src):
    """카메라 번호(0) / 사진 / 동영상을 같은 방식으로 다루기 위한 함수"""
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise SystemExit(f"카메라 {src}번을 열 수 없습니다.")
        return lambda: cap.read()[1], cap.release
    if src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"사진을 열 수 없습니다: {src}")
        return lambda: img.copy(), lambda: None
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"동영상을 열 수 없습니다: {src}")

    def read_loop():                                    # 동영상은 끝나면 처음부터 반복
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
        return f
    return read_loop, cap.release


def draw(frame, clf, prob, ms, fps, labels):
    canvas = np.full((H, W, 3), 25, np.uint8)
    h, w = frame.shape[:2]
    scale = min(VIEW_W / w, VIEW_H / h)
    view = cv2.resize(frame, (int(w * scale), int(h * scale)))
    vy, vx = (H - view.shape[0]) // 2, (VIEW_W - view.shape[1]) // 2
    canvas[vy:vy + view.shape[0], vx:vx + view.shape[1]] = view

    # 모델이 실제로 보는 가운데 정사각형 영역 표시
    s = int(min(h, w) * scale)
    cx, cy = vx + view.shape[1] // 2, vy + view.shape[0] // 2
    cv2.rectangle(canvas, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), (0, 255, 255), 1)

    top = np.argsort(prob)[::-1][:3]
    name = labels[top[0]]
    cv2.rectangle(canvas, (vx, vy), (vx + view.shape[1], vy + 44), (0, 0, 0), -1)
    cv2.putText(canvas, f"{name}  {prob[top[0]] * 100:.0f}%", (vx + 10, vy + 31), FONT, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    x = VIEW_W + 12
    cv2.putText(canvas, "MODEL", (x, 30), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, clf.name, (x, 62), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"size  {clf.size_mb:.1f} MB", (x, 95), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"infer {ms:.1f} ms", (x, 120), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"FPS   {fps:.1f}", (x, 145), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "TOP-3", (x, 190), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    for i, k in enumerate(top):
        y = 215 + i * 50
        cv2.putText(canvas, labels[k][:20], (x, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y + 8), (x + 170, y + 20), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x, y + 8), (x + int(170 * float(prob[k])), y + 20), (0, 200, 255), -1)

    cv2.putText(canvas, BACKEND, (x, 400), FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "tap: change model", (x, 430), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "q: quit  s: capture", (x, 455), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def bench(models, labels, image_path, runs=50):
    """모델별 크기와 이미지 1장 추론 시간 측정 (교수님 비교 항목)"""
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"사진을 열 수 없습니다: {image_path}")
    device = f"{platform.system()} {platform.machine()}"
    rows = []
    print(f"device: {device} / backend: {BACKEND} / image: {os.path.basename(image_path)}")
    print(f"{'model':<10}{'size(MB)':>10}{'ms/img':>10}  top-1")
    for name, path in models:
        clf = Classifier(name, path)
        for _ in range(5):                              # 예열
            clf.predict(img)
        times = []
        for _ in range(runs):
            prob, ms = clf.predict(img)
            times.append(ms)
        k = int(np.argmax(prob))
        rows.append([device, name, f"{clf.size_mb:.2f}", f"{np.mean(times):.2f}", labels[k], f"{prob[k] * 100:.1f}"])
        print(f"{name:<10}{clf.size_mb:>10.2f}{np.mean(times):>10.2f}  {labels[k]} ({prob[k] * 100:.0f}%)")
    out = os.path.join(HERE, "bench_result.csv")
    with open(out, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["device", "model", "size_mb", "ms_per_image", "top1", "top1_prob(%)"])
        w.writerows(rows)
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="카메라 번호(0) 또는 사진/동영상 파일 경로")
    ap.add_argument("--model", default=None, help="시작 모델 (fp32 / dynamic / float16 / int8)")
    ap.add_argument("--bench", default=None, help="사진 경로를 주면 모든 모델 속도 측정")
    ap.add_argument("--save", default=None, help="화면 없이 결과 이미지 저장")
    args = ap.parse_args()

    models = find_models(HERE)
    if not models:
        raise SystemExit("mobilenet_*.tflite 파일이 없습니다. PC에서 convert_models.ipynb 를 먼저 실행하세요.")
    labels = load_labels(HERE)

    if args.bench:
        bench(models, labels, args.bench)
        return

    names = [n for n, _ in models]
    idx = names.index(args.model) if args.model in names else (names.index("int8") if "int8" in names else 0)
    cache = {}

    def get(i):
        if i not in cache:
            cache[i] = Classifier(*models[i])
        return cache[i]

    read, release = open_source(args.source)

    if args.save:
        frame = read()
        prob, ms = get(idx).predict(frame)
        cv2.imwrite(args.save, draw(frame, get(idx), prob, ms, 1000 / max(ms, 1e-3), labels))
        print(f"saved {args.save} (model {names[idx]}, backend {BACKEND})")
        release()
        return

    state = {"switch": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["switch"] = True

    cv2.namedWindow("object_cam")
    cv2.setMouseCallback("object_cam", on_mouse)
    t_prev = time.time()
    fps = 0.0
    while True:
        frame = read()
        if frame is None:
            break
        clf = get(idx)
        prob, ms = clf.predict(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6)) if fps else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        canvas = draw(frame, clf, prob, ms, fps, labels)
        cv2.imshow("object_cam", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty("object_cam", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("s"):
            path = os.path.join(HERE, f"capture_{clf.name}_{int(now)}.png")
            cv2.imwrite(path, canvas)
            print("saved", path)
        if key == ord("m") or state["switch"]:
            state["switch"] = False
            idx = (idx + 1) % len(models)
            fps = 0.0
    release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
