import cv2
import mediapipe as mp
import pyautogui
import mouse

pyautogui.PAUSE = 0  # 設置每個pyautogui動作之間的暫停時間為0秒

# 初始化攝影機和Mediapipe模組
cam = cv2.VideoCapture(0)
facedetection = mp.solutions.face_detection.FaceDetection()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

capture_hands = mp.solutions.hands.Hands()
drawing_options = mp.solutions.drawing_utils

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 處理手部追蹤
    output_hands = capture_hands.process(rgb_frame)
    all_hands = output_hands.multi_hand_landmarks

    # 處理臉部網格
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # 臉部網格
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                screen_x = screen_w / frame_w * x
                screen_y = screen_h / frame_h * y
                mouse.move(screen_x, screen_y, absolute=True, duration=0.07)

    # 手部網格
    if all_hands:
        for hand in all_hands:
            drawing_options.draw_landmarks(frame, hand)
            one_hand_landmarks = hand.landmark
            img_h, img_w, _ = frame.shape
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)
                if id == 12:
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
                    x1, y1 = x, y
                if id == 8:
                    x2, y2 = x, y
                    cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
            dist = y2 - y1
            cv2.putText(frame, str(dist), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            if dist < 16:
                pyautogui.click()

    cv2.imshow('ouo', frame)
    if cv2.waitKey(1) == 27:  # 按下ESC鍵退出
        break

cam.release()
cv2.destroyAllWindows()