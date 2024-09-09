import cv2
import mediapipe as mp
import pyautogui
import mouse

pyautogui.PAUSE = 0 # 設置每個pyautogui動作之間的暫停時間為0秒


# 初始化摄像头和手部、脸部检测模块
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 初始化区域和屏幕尺寸
rx1, ry1, rx2, ry2 = 1, 1, 2, 2
screen_w, screen_h = pyautogui.size()

# 定义一个函数来检测两点之间的欧氏距离
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # 翻转摄像头图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理脸部和手部关键点
    face_output = face_mesh.process(rgb_frame)
    hand_output = hands.process(rgb_frame)
    
    frame_h, frame_w, _ = frame.shape

    # 检测脸部关键点
    if face_output.multi_face_landmarks:
        for landmarks in face_output.multi_face_landmarks:
            # 获取眼睛左右两个点的坐标
            x1 = int(landmarks.landmark[151].x * frame_w)
            y1 = int(landmarks.landmark[151].y * frame_h)
            x2 = int(landmarks.landmark[366].x * frame_w)
            y2 = int(landmarks.landmark[366].y * frame_h)

            # 按下'a'键设置矩形区域
            if cv2.waitKey(1) == ord('a'):
                rx1, ry1 = x1, y1
                rx2, ry2 = x2, y2
                break

        # 画出矩形区域
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

        # 获取眼睛的关键点
        landmarks = face_output.multi_face_landmarks[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            if id == 1:
                # 将眼睛的位置映射到屏幕坐标
                if rx1 < x < rx2 and ry1 < y < ry2:
                    screen_x = (x - rx1) * screen_w / (rx2 - rx1)
                    screen_y = (y - ry1) * screen_h / (ry2 - ry1)
                    mouse.move(screen_x, screen_y, absolute=True, duration=0.07)
            

    # 检测手部关键点
    if hand_output.multi_hand_landmarks:
        for hand_landmarks in hand_output.multi_hand_landmarks:
            # 画出手部的关键点和连接线
            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # 获取食指指尖和拇指指尖的坐标
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

            index_x = int(index_finger_tip.x * frame_w)
            index_y = int(index_finger_tip.y * frame_h)
            thumb_x = int(thumb_tip.x * frame_w)
            thumb_y = int(thumb_tip.y * frame_h)

            # 画出食指指尖和拇指指尖的点
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)

            # 计算食指指尖和拇指指尖之间的距离
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)

            print(distance)

            # # 如果距离小于一定值，触发鼠标单击
            if 10 < distance < 22:  # 根据实际情况调整这个距离阈值
                pyautogui.leftClick()

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == 27:  # 按下ESC键退出
        break

cam.release()
cv2.destroyAllWindows()
