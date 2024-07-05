from tkinter import messagebox
from flask import Flask, session, render_template, Response, redirect, request, url_for, copy_current_request_context
from time import sleep
import cv2
from ultralytics import YOLO
import torch
import time
import threading
import matplotlib.pyplot as plt
import cv2,torch
import winsound as sd
from gtts import gTTS
from playsound import playsound
import pyttsx3
import os
import pygame

app = Flask(__name__)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 웹캠으로부터 비디오 캡처 객체 생성
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

# YOLOv5 모델 불러오기
model_person = torch.hub.load('ultralytics/yolov5:master', 'yolov5s6')

static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(static_folder, exist_ok=True)

# 녹화된 파일 경로 설정
output_file_name = os.path.join(static_folder, 'output.avi')


detection_start_time = None
recording = False
out = None

# 비디오 나오게
def GenerateFrames_person():

    global detection_start_time

    while True:
        sleep(0.2)  # 프레임 생성 간격을 잠시 지연시킵니다.
        ref, frame = capture.read()  # 비디오 프레임을 읽어옵니다.

        if not ref:  # 비디오 프레임을 제대로 읽어오지 못했다면 반복문을 종료합니다.
            break

        frame1 = frame
        frame2 = frame
        frame4 = frame

        # YOLOv5로 객체 감지 수행
        results_person = model_person(frame)

        # confidence 값을 출력하고, knife이면서 confidence가 0.1 이상인 경우에만 Bounding Box 그리기
        for det in results_person.xyxy[0]:
            label = int(det[5])  # 클래스 레이블
            confidence = det[4]  # 신뢰도 값
            class_name = model_person.names[label]  # 클래스 이름

            print(f'Class: {class_name}, Confidence: {confidence:.4f}')

            if class_name == "person" :

                # Bounding Box 좌표 추출
                x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])

                # Bounding Box 그리기
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

                # 클래스와 confidence 값을 화면에 표시
                cv2.putText(frame, f'{class_name} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 3)

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
                text2 = ('person Detection')

                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]

                # 텍스트 위치 계산 (가운데 정렬)
                text_x = center_x - text_size[0] // 2
                text_y = center_y + text_size[1] // 2
                cv2.putText(frame4, text, (text_x, text_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)[0]

                # 텍스트 위치 계산 (가운데 정렬)
                text2_x = (center_x - text2_size[0] // 2)
                text2_y = (center_y + text2_size[1] // 2)
                cv2.putText(frame4, text2, (text2_x, text2_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

                #beepsound()
                play_sound()

            if class_name == "cat" :

                # Bounding Box 좌표 추출
                x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])

                # Bounding Box 그리기
                cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # 클래스와 confidence 값을 화면에 표시
                cv2.putText(frame1, f'{class_name} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 3)

            if class_name == "dog" :

                # Bounding Box 좌표 추출
                x, y, w, h = int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])

                # Bounding Box 그리기
                cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 5)

                # 클래스와 confidence 값을 화면에 표시
                cv2.putText(frame2, f'{class_name} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 3)

        # 결과 화면에 출력
        #frame = results_person.render()[0]


        frame = cv2.imencode('.jpg', frame)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame1)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame2)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환
        frame = cv2.imencode('.jpg', frame4)[1].tobytes()  # 프레임을 JPEG 이미지로 인코딩하고 바이트로 변환


            # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
        yield ((b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   ))
def beepsound():
    fr = 1000    # range : 37 ~ 32767 #hz조정 커질수록 더 찢어지는 소리
    du = 3000     # 1000 ms ==1second #재생시간
    sd.Beep(fr, du) # winsound.Beep(frequency, duration)

def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('static/oneperson_2.mp3')
    pygame.mixer.music.play()

# 경고 팝업 창을 띄우는 함수
def show_warning():
    def _show_warning():
        messagebox.showwarning("경고", "3초 동안 사람이 감지되었습니다!")

    # 경고 메시지를 표시하는 쓰레드 시작
    t = threading.Thread(target=_show_warning)
    t.start()
@app.route('/start_recording')
def start_recording():
    global recording, out
    try:
        if not recording:
            recording = True
            out = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
            threading.Thread(target=record_frames).start()
        return 'Recording Started'
    except Exception as e:
        return f'Error starting recording: {str(e)}'

@app.route('/stop_recording')
def stop_recording():
    global recording, out
    if recording:
        recording = False
        out.release()
    return 'Recording Stopped'

def record_frames():
    global out
    while recording:
        success, frame = capture.read()
        if not success:
            break
        out.write(frame)



if __name__ == "__main__":
    # 라즈베리파이의 IP 번호와 포트 번호를 지정하여 Flask 앱을 실행합니다.
    app.run(host="192.168.10.104", port=8080)






