import sys, platform, time
import cv2

print("Python:", sys.version.split()[0])
print("OS:", platform.system(), platform.release())
print("cv2 path:", cv2.__file__)
print("cv2 version:", cv2.__version__)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Failed to open camera index 0")

cv2.namedWindow("test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ read failed, continue...")
        time.sleep(0.05)
        continue
    cv2.imshow("test", frame)
    if (cv2.waitKey(1) & 0xFF) in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
