from flask import Flask, request, jsonify
from flask_sock import Sock
import cv2
import numpy as np
import base64
import os
import json

app = Flask(__name__)
sock = Sock(app)

# 연결된 웹소켓 클라이언트 목록
clients = []

# 수신된 이미지를 저장할 폴더
SAVE_DIR = "received_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@sock.route('/ws')
def handle_ws(ws):
    clients.append(ws)
    print(f"[WS] Client connected. Total clients: {len(clients)}")
    try:
        while True:
            # 클라이언트로부터 메시지를 기다림 (연결 유지용)
            data = ws.receive()
            if data == 'PING':
                ws.send('PONG')
    except:
        pass
    finally:
        clients.remove(ws)
        print(f"[WS] Client disconnected. Total clients: {len(clients)}")

@app.route('/receive', methods=['POST'])
def receive_data():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    edge_ratio = data.get("edge_ratio")
    ai_result = data.get("ai_result")
    score = data.get("score")
    crack_count = data.get("crack_count", 0)
    image_raw_b64 = data.get("image_raw")
    image_edge_b64 = data.get("image_edge")

    print(f"\n[RECEIVED] Edge Ratio: {edge_ratio:.2f}% | AI: {ai_result} (Score: {score:.4f}) | Total Cracks: {crack_count}")

    # 웹소켓 클라이언트들에게 데이터 브로드캐스트
    message = json.dumps({
        "type": "crack_detection",
        "edge_ratio": edge_ratio,
        "ai_result": ai_result,
        "score": score,
        "crack_count": crack_count,
        "image_raw": image_raw_b64,
        "image_edge": image_edge_b64
    })
    
    for client in clients:
        try:
            client.send(message)
        except:
            pass

    # 이미지 저장 (필요 시)
    if image_raw_b64:
        img_raw = decode_image(image_raw_b64)
        cv2.imwrite(os.path.join(SAVE_DIR, "last_raw.jpg"), img_raw)
    
    if image_edge_b64:
        img_edge = decode_image(image_edge_b64)
        cv2.imwrite(os.path.join(SAVE_DIR, "last_edge.jpg"), img_edge)

    return jsonify({"status": "success", "message": "Data received and broadcasted"}), 200

if __name__ == '__main__':
    print("Raspberry Pi Receiver Server Starting with WebSocket support...")
    app.run(host='0.0.0.0', port=5000)
