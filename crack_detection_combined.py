import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import requests
import base64
import json

# TensorFlow legacy Keras 환경 설정 (기존 app.py 참고)
os.environ['TF_USE_LEGACY_KERAS'] = '1'

# 라즈베리파이 설정
PI_IP = "192.168.0.100"  # 실제 라즈베리파이 IP 주소로 수정하세요
PI_URL = f"http://{PI_IP}:5000/receive"

# 모델 로드
MODEL_PATH = 'wallCrackDetector/models/fifth_model.h5'
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_to_pi(edge_ratio, ai_result, score, crack_count, frame, edges):
    try:
        payload = {
            "edge_ratio": edge_ratio,
            "ai_result": ai_result,
            "score": float(score),
            "crack_count": crack_count,
            "image_raw": encode_image(frame),
            "image_edge": encode_image(edges)
        }
        response = requests.post(PI_URL, json=payload, timeout=2)
        if response.status_code == 200:
            print(f"[WiFi] Data sent to Pi successfully. (Count: {crack_count})")
        else:
            print(f"[WiFi] Failed to send data. Status code: {response.status_code}")
    except Exception as e:
        print(f"[WiFi] Error sending data: {e}")

def predict_crack(frame):
    # 모델 입력 규격에 맞게 전처리 (128x128)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    return prediction[0][0]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    EDGE_THRESHOLD_RATIO = 2.0  # 사용자 요청에 따라 2%로 하향 조정
    crack_count = 0
    prev_crack_detected = False # 중복 카운트 방지용
    
    print(f"Starting Crack Detection... Sending to {PI_URL}")
    print("Press 'ESC' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Canny Edge Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        # 2. Edge Ratio 계산
        edge_pixels = np.count_nonzero(edges)
        total_pixels = edges.size
        edge_ratio = (edge_pixels / total_pixels) * 100

        status_text = f"Edge Ratio: {edge_ratio:.2f}%"
        ai_result = "Waiting..."
        color = (255, 255, 255) # White
        score = 0.0

        # 3. 비율이 2%를 넘으면 AI 판단 실행
        if edge_ratio > EDGE_THRESHOLD_RATIO:
            score = predict_crack(frame)
            if score > 0.5:
                ai_result = "CRACK DETECTED!"
                color = (0, 0, 255) # Red
                
                # 중복 카운트 방지 (새로운 균열이 감지된 경우만 카운트 증가)
                if not prev_crack_detected:
                    crack_count += 1
                    prev_crack_detected = True
            else:
                ai_result = "NO CRACK"
                color = (0, 255, 0) # Green
                prev_crack_detected = False
            
            print(f"[RESULT] {status_text} -> {ai_result} (Score: {score:.4f}, Total: {crack_count})")
            
            # 라즈베리파이로 데이터 전송 (카운트 포함)
            send_to_pi(edge_ratio, ai_result, score, crack_count, frame, edges)
        else:
            ai_result = "Below Threshold"
            prev_crack_detected = False

        # 화면에 정보 표시
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"AI: {ai_result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Crack Count: {crack_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Canny Edge 결과도 작은 창으로 표시 (디버깅용)
        edge_small = cv2.resize(edges, (frame.shape[1]//4, frame.shape[0]//4))
        edge_colored = cv2.cvtColor(edge_small, cv2.COLOR_GRAY2BGR)
        frame[0:edge_colored.shape[0], frame.shape[1]-edge_colored.shape[1]:frame.shape[1]] = edge_colored

        cv2.imshow("Crack Detection System", frame)

        if cv2.waitKey(1) & 0xFF == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
