from tkinter import messagebox
from flask import Flask, render_template, Response
from time import sleep
import cv2
from ultralytics import YOLO
import torch
import time
import threading

from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2,torch

app = Flask(__name__)

capture = cv2.VideoCapture(0)  # 웹캠으로부터 비디오 캡처 객체 생성
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

# YOLOv5 모델 불러오기

# Load a model

model = YOLO("yolo-Weights/yolov8n.pt")


#사람 감지 추적 플래그
person_detected = False
detection_start_time = None

# 비디오 나오게
def GenerateFrames():
    global person_detected  # 플래그를 글로벌로 선언
    global detection_start_time
    while True:
        sleep(0.1)  # 프레임 생성 간격을 잠시 지연시킵니다.
        ref, frame = capture.read()  # 비디오 프레임을 읽어옵니다.

        if not ref:  # 비디오 프레임을 제대로 읽어오지 못했다면 반복문을 종료합니다.
            break


        # YOLOv5로 객체 감지 수행
        results = model(frame)



        # 감지된 객체가 있는지 확인
        if len(results.pred[0]) > 0:
            detected_classes = results.names[int(results.pred[0][:, -1][0])]
            if detected_classes == "person":
                if not person_detected:
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elif time.time() - detection_start_time >= 3:
                        person_detected = True
                        show_warning()
            else:
                person_detected = False
                detection_start_time = None

        # 결과 화면에 출력
        frame = results.render()[0]


        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환


            # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
        yield ((b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   )
)



# 경고 팝업 창을 띄우는 함수
def show_warning():
    def _show_warning():
        messagebox.showwarning("경고", "3초 동안 사람이 감지되었습니다!")

    # 경고 메시지를 표시하는 쓰레드 시작
    t = threading.Thread(target=_show_warning)
    t.start()

@app.route('/')
def Index():
    video_url = "http://192.168.10.114:8080/stream_test"
    return render_template('index.html', video_url=video_url)  # index_origin.html 파일을 렌더링하여 반환합니다.


@app.route('/stream_test')
def Stream():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    # 라즈베리파이의 IP 번호와 포트 번호를 지정하여 Flask 앱을 실행합니다.
    app.run(host="192.168.10.114", port="8080")