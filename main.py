import cv2
import numpy as np

cap = cv2.VideoCapture(0)

EDGE_THRESHOLD_RATIO = 10.0
count = 0
prev_above_threshold = False  # 이전 프레임 상태

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    edge_ratio = (edge_pixels / total_pixels) * 100

    # 현재 상태
    current_above_threshold = edge_ratio > EDGE_THRESHOLD_RATIO

    # 중복 카운트 방지 (상승 에지)
    if current_above_threshold and not prev_above_threshold:
        count += 1

    prev_above_threshold = current_above_threshold

    # 화면 표시
    cv2.putText(
        edges,
        f"Edge Ratio: {edge_ratio:.2f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        255,
        2
    )

    cv2.putText(
        edges,
        f"Count (>10%): {count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        255,
        2
    )

    cv2.imshow("Canny Edge Live", edges)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
