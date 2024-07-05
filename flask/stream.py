from tkinter import messagebox
from flask import Flask, session, render_template, Response, redirect, request, url_for, make_response,\
    copy_current_request_context
from time import sleep
from ultralytics import YOLO
import torch
import time
import threading
import matplotlib.pyplot as plt
import cv2,torch
import winsound as sd
import pygame
from aloneperson_1 import GenerateFrames_person, stop_recording, start_recording
from flaskpose import generate_frames
import numpy as np

app = Flask(__name__)

app.secret_key="your_secret_key"

ID="jinji"
PW="guard"

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 웹캠으로부터 비디오 캡처 객체 생성
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

# YOLOv5 모델 불러오기
#model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s6')
model3 = torch.hub.load('ultralytics/yolov5:master', 'custom', path='best_knife.pt')

person_detected = False
detection_start_time = None

# 비디오 나오게
def GenerateFrames():
    global person_detected  # 플래그를 글로벌로 선언
    global detection_start_time

    while True:
        sleep(0.2)  # 프레임 생성 간격을 잠시 지연시킵니다.
        ref, frame = capture.read()  # 비디오 프레임을 읽어옵니다.

        if not ref:  # 비디오 프레임을 제대로 읽어오지 못했다면 반복문을 종료합니다.
            break


        # YOLOv5로 객체 감지 수행
        #results = model(frame)
        results3 = model3(frame)

        frame2 = frame
        frame3 = frame
        frame4 = frame

        # confidence 값을 출력하고, knife이면서 confidence가 0.1 이상인 경우에만 Bounding Box 그리기
        for det in results3.xyxy[0]:
            label = int(det[5])  # 클래스 레이블
            confidence = det[4]  # 신뢰도 값
            class_name = model3.names[label]  # 클래스 이름

            print(f'Class: {class_name}, Confidence: {confidence:.4f}')

            if class_name == "knife" and confidence > 0.3: #"Fall_Down"

                    # Bounding Box 좌표 추출
                    x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])

                    # Bounding Box 그리기
                    cv2.rectangle(frame3, (x, y), (x + w, y + h), (26, 140, 255), 5)

                    # 사각형의 가로, 세로 길이 및 중심 좌표 계산
                    width = 250
                    height = 80
                    center_x = frame4.shape[1] // 2  # 프레임의 너비의 중심
                    center_y = frame4.shape[0] // 2  # 프레임의 높이의 중심
                    x1, y1 = center_x - width // 2, center_y - height // 2  # 왼쪽 위 꼭지점 좌표
                    x2, y2 = center_x + width // 2, center_y + height // 2  # 오른쪽 아래 꼭지점 좌표

                    time.sleep(1)
                    cv2.rectangle(frame4, (x1, y1), (x2, y2), (255, 255, 255), -1)

                    # 텍스트를 사각형 내부 중앙에 배치하기 위한 좌표 계산
                    text = ('  [Warning]  ')
                    text2 = ('Knife Detection')

                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]

                    # 텍스트 위치 계산 (가운데 정렬)
                    text_x = center_x - text_size[0] // 2
                    text_y = center_y + text_size[1] // 2
                    cv2.putText(frame4, text, (text_x, text_y - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                    text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]

                    # 텍스트 위치 계산 (가운데 정렬)
                    text2_x = (center_x - text2_size[0] // 2)
                    text2_y = (center_y + text2_size[1] // 2)
                    cv2.putText(frame4, text2, (text2_x, text2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (26, 140, 255), 3)

                    # 클래스와 confidence 값을 화면에 표시
                    cv2.putText(frame3, f'{class_name} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (26, 140, 255), 3)

                    #beepsound()
                    play_soundk()
        #############################################################################################
        # confidence 값을 출력하고, person이면서 confidence가 0.1 이상인 경우에만 Bounding Box 그리기

        # 결과 화면에 출력
        #frame = results.render()[0]
        #frame1 = results1.render()[0]
        #frame2 = results2.render()[0]
        #frame3 = results3.render()[0]


        #frame = cv2.imencode('.jpg', frame)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame2)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame3)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame4)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환


            # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
        yield ((b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'))

#소리
def beepsound():
    fr = 1000    # range : 37 ~ 32767 #hz조정 커질수록 더 찢어지는 소리
    du = 3000     # 1000 ms ==1second #재생시간
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def play_soundk():
    pygame.mixer.init()
    pygame.mixer.music.load('static/knife1.mp3')
    pygame.mixer.music.play()

#guardian mode 웹캠 창
@app.route('/index') #index.html
def Index():
    video_url = "http://192.168.10.104:8080/index"
    return render_template('index.html', video_url=video_url)  # index_origin.html 파일을 렌더링하여 반환합니다.

@app.route('/stream') #stream.py
def Stream():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#mode창 + 로그인에서 넘기기
@app.route('/mode')
def Mode():
    return render_template('mode.html')

@app.route('/redirect3', methods=['POST'])
def redirect_page3():
    return redirect(url_for('Mode'))

# 버튼 누르면 guardian mode 버튼 -> index.html
@app.route('/redirect', methods=['POST'])
def redirect_page():
    return redirect(url_for('Index'))

# 버튼 누르면 aloneperson mode 버튼 -> aloneperson.html
@app.route('/redirect2', methods=['POST'])
def redirect_page2():
    return redirect(url_for('Aloneperson'))

@app.route('/redirect4', methods=['POST'])
def redirect_page4():
    return redirect(url_for('Pose'))

@app.route('/pose')
def Pose():
    video_url = "http://192.168.10.104:8080/pose"

    return render_template('pose.html', video_url=video_url)  # index_origin.html 파일을 렌더링하여


@app.route('/flaskpose')
def Flaskpose():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#aloneperson mode 웹캠 창
@app.route('/aloneperson')
def Aloneperson():
    video_url = "http://192.168.10.104:8080/alone"

    return render_template('aloneperson.html', video_url=video_url)  # index_origin.html 파일을 렌더링하여

@app.route('/aloneperson_1')
def Aloneperson_1():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(GenerateFrames_person(), mimetype='multipart/x-mixed-replace; boundary=frame')

#splash 창

@app.route('/')
def start_page():
    return render_template('splash.html')

#로그인 창

@app.route("/home")
def home():
    if "userID" in session:
        return render_template("home.html", username=session.get("userID"), login=True)
    else:
        return render_template("home.html",login=False)


@app.route("/login", methods=["get"])
def login():
    global ID, PW
    _id_ = request.args.get("loginId")
    _password_ = request.args.get("loginPw")

    if ID == _id_ and PW == _password_ :
        session["userID"]=_id_
        return redirect(url_for("home"))
    else:
        return redirect(url_for("home"))

@app.route("/logout")
def logout():
    session.pop("userID")
    return redirect(url_for("home"))

@app.route('/start_recording')
def start_recording_route():
    start_recording()
    return '녹화 시작됨'

# stop_recording 라우트 정의
@app.route('/stop_recording')
def stop_recording_route():
    stop_recording()
    return '녹화 중지됨'

if __name__ == "__main__":
    # 라즈베리파이의 IP 번호와 포트 번호를 지정하여 Flask 앱을 실행합니다.
    app.run(host="192.168.10.104", port=8080)






