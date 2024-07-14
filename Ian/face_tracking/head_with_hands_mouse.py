import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

pyautogui.PAUSE = 0 # No delay between pyautogui functions

# Initialize MediaPipe Hands and Face Mesh
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands()
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize variables
x1 = y1 = x2 = y2 = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Flip and convert the image to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Process the image for face mesh
    face_results = face_mesh.process(image)
    # Process the image for hand landmarks
    hand_results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape

    # Process hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)

                if id == 4:
                    cv2.circle(image, (x, y), 10, (0, 255, 255))
                    x1 = x
                    y1 = y
                if id == 8:  # Index finger tip
                    x2 = x
                    y2 = y
                    cv2.circle(image, (x, y), 10, (0, 255, 255), -1)
            dist = y2 - y1
            if abs(dist) < 10:
                pyautogui.click()

    # Process face landmarks
    if face_results.multi_face_landmarks:
        face_3d = []
        face_2d = []
        for face_landmarks in face_results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [33, 263, 1, 61, 291, 199]:
                    if idx == 1:
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_3d.append([x, y, lm.z])
                    face_2d.append([x, y])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -7:
                text = 'Looking Left'
                pyautogui.moveRel(-25, 0, 0)
            elif y > 7:
                text = 'Looking Right'
                pyautogui.moveRel(25, 0, 0)
            elif x < -5:
                text = 'Looking Down'
                pyautogui.moveRel(0, 25, 0)
            elif x > 7:
                text = 'Looking Up'
                pyautogui.moveRel(0, -25, 0)
            else:
                text = 'Looking Forward'
                pyautogui.moveRel(0, 0, 0)

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'X: {x:.2f}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Y: {y:.2f}', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Z: {z:.2f}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime
    cv2.putText(image, f'FPS: {fps:.2f}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('MediaPipe Face Mesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
