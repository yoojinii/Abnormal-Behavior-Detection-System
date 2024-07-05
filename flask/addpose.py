import cv2
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    # 각 값을 받아 넘파이 배열로 변형
    a = np.array(a)  # 첫번째
    b = np.array(b)  # 두번째
    c = np.array(c)  # 세번째

    # 라디안을 계산하고 실제 각도로 변경한다.
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # 180도가 넘으면 360에서 뺀 값을 계산한다.
    if angle > 180.0:
        angle = 360 - angle

    # 각도를 리턴한다.
    return angle


# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.6) as pose:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape
    # Convert the BGR image to RGB before processing.
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if not results.pose_landmarks:
      continue
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = BG_COLOR
    annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
    # Plot pose world landmarks.
    mp_drawing.plot_landmarks(
        results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

#image_path= "C:\\Users\\youji\\OneDrive\\바탕 화면\\falldownf.jpg"
video_path=""

# For webcam input:
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)
cap = cv2.imread(video_path)
pTime = 0  # fps 계산을 위한 previous 타임
cnt = 0  # 몇 번 넘어졌는지 세기위한 변수

with mp_pose.Pose(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
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
        a=keypoints[10]['Y']
        b=keypoints[24]['Y']
        diff = abs(b-a)

        if diff < 0.1 : #얼굴과 몸통이 수평선
            cnt +=1
            print(f' {cnt}번 쓰러짐 감지!')

        a = keypoints[28]['Y']
        b = keypoints[26]['Y']
        c = keypoints[24]['Y']

        ang = calculate_angle(a, b, c)
        if  ang < 90 :  # 얼굴과 몸통이 수평선
            if ang < 90 :
                cnt += 1
                print(f' {cnt}번 폭력 감지!')

        a1 = keypoints[27]['Y']
        b1 = keypoints[25]['Y']
        c1 = keypoints[23]['Y']

        ang1 = calculate_angle(a1, b1, c1)
        if ang1 < 90:  # 얼굴과 몸통이 수평선
            if ang1 < 90:
                cnt += 1
                print(f' {cnt}번 폭력 감지!')

    cTime=time.time()
    fps= 1/(cTime-pTime)
    pTime=cTime

    cv2.putText(image, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

