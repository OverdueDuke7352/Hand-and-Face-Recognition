import cv2
import mediapipe as mp
import pyautogui
import mouse

pyautogui.PAUSE = 0  # 设置每个pyautogui动作之间的暂停时间为0秒

# 初始化摄像头和MediaPipe模块以进行脸部和手部检测
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils

# 初始化屏幕尺寸
screen_w, screen_h = pyautogui.size()

# 定义一个函数来计算两点之间的欧氏距离
def calculate_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 初始化区域坐标和标志变量
rx1, ry1, rx2, ry2 = None, None, None, None
tracking_enabled = False

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 翻转摄像头图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 处理脸部和手部关键点
    face_output = face_mesh.process(rgb_frame)
    hand_output = hands.process(rgb_frame)
    
    frame_h, frame_w, _ = frame.shape

    # 脸部关键点检测和眼动跟踪
    if face_output.multi_face_landmarks:
        for landmarks in face_output.multi_face_landmarks:
            # 获取眼睛左右两个点的坐标
            x1 = int(landmarks.landmark[151].x * frame_w)
            y1 = int(landmarks.landmark[151].y * frame_h)
            x2 = int(landmarks.landmark[366].x * frame_w)
            y2 = int(landmarks.landmark[366].y * frame_h)

            # 按下'a'键设置跟踪区域
            if cv2.waitKey(1) == ord('a'):
                rx1, ry1 = x1, y1
                rx2, ry2 = x2, y2
                tracking_enabled = True
                break

        # 确保在尝试绘制之前跟踪区域已设置
        if tracking_enabled:
            # 画出跟踪矩形
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

            # 眼动跟踪和鼠标移动
            for id, landmark in enumerate(landmarks.landmark[474:478]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                if id == 1:
                    # 将眼睛位置映射到屏幕坐标
                    screen_x = (x - rx1) * screen_w / (rx2 - rx1)
                    screen_y = (y - ry1) * screen_h / (ry2 - ry1)
                    screen_x = max(0, min(screen_x, screen_w))  # 将值限制在屏幕尺寸内
                    screen_y = max(0, min(screen_y, screen_h))
                    mouse.move(screen_x, screen_y, absolute=True, duration=0.075)
    
    # 手部关键点检测和手势识别
    if hand_output.multi_hand_landmarks:
        for hand_landmarks in hand_output.multi_hand_landmarks:
            # mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]

            middle_x = int(middle_finger_tip.x * frame_w)
            middle_y = int(middle_finger_tip.y * frame_h)
            index_x = int(index_finger_tip.x * frame_w)
            index_y = int(index_finger_tip.y * frame_h)
            thumb_x = int(thumb_tip.x * frame_w)
            thumb_y = int(thumb_tip.y * frame_h)

            cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)

            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y)
            flip = calculate_distance(middle_x, middle_y, thumb_x, thumb_y)

            print(flip, end = '\r', flush=True)

            # 如果距离在阈值内，触发鼠标点击
            if tracking_enabled and 10 < distance < 22:
                pyautogui.leftClick()
            
            # 如果距离在阈值内，触发鼠标拖曳
            if  flip < 22:
                pyautogui.mouseDown(screen_x, screen_y, button='left')
            elif tracking_enabled:
                pyautogui.mouseUp(screen_x, screen_y, button='left')

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == 27:  # 按下ESC键退出
        break

cam.release()
cv2.destroyAllWindows()
