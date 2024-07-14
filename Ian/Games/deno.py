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


    if landmark_points: # 如果有偵測到臉
        landmarks = landmark_points[0].landmark # 取得所有的定位點
        for landmark in landmarks[4:5]: # 取得鼻子的定位點
            x = int(landmark.x * frame.shape[1]) #取得鼻子x座標
            y = int(landmark.y * frame.shape[0]) #取得鼻子y座標
            cv2.circle(frame, (x, y), 3, (0, 255, 0)) #畫出鼻子
            #繪製一條水平基準線
            if cv2.waitKey(1) == ord('a'):
                yu= y-25
            baseline_y = faame_h // 2  # 基準線的x坐標
            color = (0, 255, 0)  # 線的顏色，這裡是綠色
            thickness = 2  # 線的粗細
            cv2.line(frame, (0, yu), (frame_w, yu), color, thickness) # 繪製基準線
        if y < yu and n == False: # 如果鼻子的y座標小於基準線
            n = True # 防止一直按住空白鍵
            keyboard.press_and_release('space') # 按下空白鍵
            time.sleep(0.01) # 等待0.01秒
            # print('up')
        if y > yu and n == True: # 如果鼻子的y座標大於基準線
            n = False # 防止一直按住空白鍵

    cv2.imshow('ouo', frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()