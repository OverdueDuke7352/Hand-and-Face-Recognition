import cv2
import mediapipe as mp
import pyautogui



capture_hands = mp.solutions.hands.Hands()
drawing_options = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()


cam = cv2.VideoCapture(1)
x1 = y1 = x2 = y2 = 0
while True:
    _, image = cam.read()
    img_h, img_w, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks
    if all_hands: #如果偵測到手指
        for hand in all_hands: #遍歷每一隻手指
            drawing_options.draw_landmarks(image, hand) #畫出手指定位點
            one_hand_landmarks = hand.landmark #取得手指定位點
            for id,lm in enumerate(one_hand_landmarks):  #遍歷手指定位點
                x = int(lm.x * img_w) #取得x座標
                y = int(lm.y * img_h) #取得y座標
                if id == 4: #找到食指
                    mouse_x = int(screen_width / img_w * x *1)
                    mouse_y = int(screen_height / img_h * y *1)
                    cv2.circle(image, (x,y), 10, (0,255,255))
                    pyautogui.moveTo(mouse_x, mouse_y)
                    x1 = x
                    y1 = y
                if id == 8:
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x,y), 10, (0,255,255))
        dist = y2-y1
        print(dist)
        cv2.putText(image, str(dist), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        if abs(dist) < 16:
            pyautogui.click(mouse_x, mouse_y)

    cv2.imshow("Finger Mouse", image)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()