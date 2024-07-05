from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time
import pygame

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pTime = 0
cnt = 0

def generate_frames():
    global pTime, cnt
    with (mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose):
        while True:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame4 = image

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image)

            keypoints = []
            if results.pose_landmarks:
                for data_point in results.pose_landmarks.landmark:
                    keypoints.append({
                        'X': data_point.x,
                        'Y': data_point.y,
                        'Z': data_point.z,
                        'Visibility': data_point.visibility,
                    })
                af = keypoints[10]['Y']
                bf = keypoints[24]['Y']
                diff = abs(bf - af)

                av = keypoints[5]['Y']
                bv = keypoints[16]['Y']
                cv = keypoints [2]['Y']
                dv = keypoints[15]['Y']
                diff2 = abs(bv - av)
                diff3 = abs(dv - cv)

                if diff < 0.1:
                    cnt += 1
                    print(f' {cnt}번 쓰러짐 감지!')
                    cv2.rectangle(image, (20,20), (570, 65), (255, 255, 255), -1)
                    cv2.putText(image, '[Warning] FallDown Detection', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (235,255,51), 3)
                    play_sound()

                if diff2 < 0.1 and diff3 < 0.1:
                    cnt += 1
                    print(f' {cnt}번 폭력 감지!')
                    cv2.rectangle(image, (20, 20), (600, 65), (255, 255, 255), -1)
                    cv2.putText(image, '[Warning] Violence Detection', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 0, 0), 3)
                    play_soundv()

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
            )

            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load('static/falldown.mp3')
    pygame.mixer.music.play()

def play_soundv():
    pygame.mixer.init()
    pygame.mixer.music.load('static/violence.mp3')
    pygame.mixer.music.play()


if __name__ == "__main__":
    app.run(debug=True)