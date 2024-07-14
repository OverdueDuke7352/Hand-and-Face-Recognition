import cv2
import mediapipe as mp
import pyautogui
import mouse

# pyautogui.PAUSE = 0  # 设置每个pyautogui动作之间的暂停时间为0秒

# 初始化摄像头和MediaPipe模块以进行脸部和手部检测
cam = cv2.VideoCapture(2)
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


    if face_output.multi_face_landmarks: #如果偵測到臉部
        for landmarks in face_output.multi_face_landmarks: #遍歷每一個臉
            x1 = int(landmarks.landmark[151].x * frame_w) #取得左眼x座標
            y1 = int(landmarks.landmark[151].y * frame_h) #取得左眼y座標
            x2 = int(landmarks.landmark[366].x * frame_w) #取得右眼x座標
            y2 = int(landmarks.landmark[366].y * frame_h) #取得右眼y座標
            if cv2.waitKey(1) == ord('a'): # 按下a键，記錄可操作的範圍
                rx1, ry1 = x1, y1 #記錄左眼座標，作爲可操作範圍的邊界
                rx2, ry2 = x2, y2 #記錄右眼座標，作爲可操作範圍的邊界
                tracking_enabled = True
                break
        if tracking_enabled: #如果可操作範圍已經記錄
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2) #畫出可操作範圍
            for id, landmark in enumerate(landmarks.landmark[474:475]): #遍歷左眼的定位點
                x = int(landmark.x * frame_w) #取得左眼球x座標
                y = int(landmark.y * frame_h) #取得左眼球y座標
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1) #畫出左眼球
                if id == 1:
                    screen_x = (x - rx1) * screen_w / (rx2 - rx1) #計算滑鼠x座標，與螢幕對應的位置
                    screen_y = (y - ry1) * screen_h / (ry2 - ry1) #計算滑鼠y座標，與螢幕對應的位置
                    screen_x = max(0, min(screen_x, screen_w)) #確保滑鼠x座標在範圍內
                    screen_y = max(0, min(screen_y, screen_h)) #確保滑鼠y座標在範圍內
                    mouse.move(screen_x, screen_y, absolute=True, duration=0.075) #移動滑鼠到螢幕對應的位置

    if hand_output.multi_hand_landmarks: #如果偵測到手
        for hand_landmarks in hand_output.multi_hand_landmarks: #遍歷每一隻手
            #取得食指指尖定位點座標
            index_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            #取得中指指尖定位點座標
            middle_finger_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            #取得大拇指指尖定位點座標
            thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle_x = int(middle_finger_tip.x * frame_w) #取得中指指尖x座標
            middle_y = int(middle_finger_tip.y * frame_h) #取得中指指尖y座標
            index_x = int(index_finger_tip.x * frame_w) #取得食指指尖x座標
            index_y = int(index_finger_tip.y * frame_h) #取得食指指尖y座標
            thumb_x = int(thumb_tip.x * frame_w) #取得大拇指指尖x座標
            thumb_y = int(thumb_tip.y * frame_h) #取得大拇指指尖y座標
            cv2.circle(frame, (middle_x, middle_y), 10, (0, 255, 0), -1) #畫出中指指尖
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 0), -1) #畫出食指指尖
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1) #畫出大拇指指尖
            distance = calculate_distance(index_x, index_y, thumb_x, thumb_y) #計算食指和大拇指之間的距離
            flip = calculate_distance(middle_x, middle_y, thumb_x, thumb_y) #計算中指和大拇指之間的距離
            if tracking_enabled and 10 < distance < 22: #如果食指和大拇指之間的距離在某個範圍內
                pyautogui.leftClick() #執行滑鼠點擊
            if  flip < 22: #如果中指和大拇指之間的距離在某個範圍內
                pyautogui.mouseDown(screen_x, screen_y, button='left') #執行滑鼠按下（爲了滑動）
            elif tracking_enabled: #設定放開條件
                pyautogui.mouseUp(screen_x, screen_y, button='left') #執行滑鼠放開

    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) == 27:  # 按下ESC键退出
        break

cam.release()
cv2.destroyAllWindows()