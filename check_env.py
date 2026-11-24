"""PC와 라즈베리파이 환경 버전이 맞는지 확인: python check_env.py"""
import platform
import sys

print("python      ", sys.version.split()[0], "(3.11.x 이어야 함)")
print("platform    ", platform.system(), platform.machine())
import numpy
print("numpy       ", numpy.__version__, "(1.26.4 이어야 함)")
import cv2
print("opencv      ", cv2.__version__)
try:
    import tflite_runtime
    print("tflite      ", "tflite_runtime", getattr(tflite_runtime, "__version__", "?"))
except ImportError:
    import tensorflow as tf
    print("tflite      ", "tensorflow", tf.__version__, "(2.14.x 이어야 함)")
