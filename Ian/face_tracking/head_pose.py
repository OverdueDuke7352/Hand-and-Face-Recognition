import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

pyautogui.PAUSE = 0 # No delay between pyautogui functions

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

x1 = y1 = x2 = y2 = 0
capture_hands = mp.solutions.hands.Hands()
drawing_options = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()

    start = time.time()

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = True

    results = face_mesh.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    output_hands = capture_hands.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if all_hands:
        for hand in all_hands:
            drawing_options.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            for id,lm in enumerate(one_hand_landmarks):
                x = int(lm.x * img_w)
                y = int(lm.y * img_h)
                if id == 8:
                        x2 = x
                        y2 = y
                        cv2.circle(image, (x,y), 10, (0,255,255))
            dist = y2-y1
            # print(dist)
            if dist < 16:
                pyautogui.click(mouse_x, mouse_y)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*3000)
                        nose_2d = (lm.x*img_w, lm.y*img_h)

                    x ,y = int(lm.x*img_w), int(lm.y*img_h)
                    face_3d.append([x, y, lm.z])
                    face_2d.append([x, y])

            if len(face_3d) >= 4 and len(face_2d) >= 4:  # Ensure at least 4 points for solvePnP
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h/2],
                                       [0, focal_length, img_w/2],
                                       [0, 0, 1]])

                dist_matrix = np.zeros((4,1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                if y < -10:
                    text = 'Looking Left'
                    pyautogui.moveRel(-10, 0, 0.1)
                elif y > 10:
                    text = 'Looking Right'
                    pyautogui.moveRel(10, 0, 0.1)
                elif x < -10:
                    text = 'Looking Down'
                    pyautogui.moveRel(0, 10, 0.1)
                elif x > 10:
                    text = 'Looking UP'
                    pyautogui.moveRel(0, -10, 0.1)
                else:
                    text = 'Looking Forward'
                    pyautogui.moveRel(0, 0, 0.1)

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec,trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))
                cv2.line(image, p1, p2, (255, 0, 0), 3)

                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'X: {x:.2f}', (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Y: {y:.2f}', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Z: {z:.2f}', (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        print(f'FPS: {fps}')
        cv2.putText(image, f'FPS: {fps:.2f}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,  # Replace FACE_CONNECTIONS with FACEMESH_TESSELATION
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    cv2.imshow('MediaPipe Face Mesh', image)
    cv2.imshow("Finger Mouse", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()