import cv2
import mediapipe as mp
import pyautogui
import time
import keyboard

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
n = False
yu=0

while True:
    _, frame = cam.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    faame_h, frame_w, _ = frame.shape

    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        for landmark in landmarks[4:5]:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if cv2.waitKey(1) == ord('a'):                        #繪製一條垂直基準線
                yu= y-25
            baseline_y = faame_h // 2  # 基準線的x坐標
            color = (0, 255, 0)  # 線的顏色，這裡是綠色
            thickness = 2  # 線的粗細
            cv2.line(frame, (0, yu), (frame_w, yu), color, thickness)
        if y < yu and n == False:
            n = True
            keyboard.press_and_release('space')
            time.sleep(0.01)
            print('up')
        if y > yu and n == True:
            n = False

    cv2.imshow('ouo', frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
