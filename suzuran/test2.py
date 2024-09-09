import cv2
import mediapipe as mp
import pyautogui

pyautogui.PAUSE = 0 # Set the duration between each pyautogui action to 0 seconds

# Load the face detection and facial landmarks models
face_detection = mp.solutions.face_detection.FaceDetection()
face_mesh = mp.solutions.face_mesh.FaceMesh()
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Load the video or webcam feed
cap = cv2.VideoCapture(0)  # Change the parameter to the video file path if needed

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    # Convert the frame to RGB and pass it to the face detection model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    # Check if any faces are detected
    if results.detections:
        for detection in results.detections:
            # Get the bounding box coordinates of the face
            bbox = detection.location_data.relative_bounding_box
            h, w, c = frame.shape
            x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

            # Draw the bounding box around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Pass the RGB frame to the face mesh model to get the facial landmarks
            landmarks = face_mesh.process(rgb_frame)

            # Check if any facial landmarks are detected
            if landmarks.multi_face_landmarks:
                for face_landmarks in landmarks.multi_face_landmarks:
                    # Draw the facial landmarks on the frame
                    for landmark in face_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        # cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                        # Get the left eye coordinates
                        left_eye_x = int(face_landmarks.landmark[413].x * w)+5
                        left_eye_y = int(face_landmarks.landmark[413].y * h)-5

                        # Draw a rectangle around the left eye
                        left_eye_width = 40
                        left_eye_height = 20
                        cv2.rectangle(rgb_frame, (left_eye_x, left_eye_y), (left_eye_x + left_eye_width, left_eye_y + left_eye_height), (255, 0, 0), 2)
                        print(left_eye_x, left_eye_y, left_eye_width, left_eye_height)

                        eye_tracking = rgb_frame[left_eye_y:left_eye_y + left_eye_height, left_eye_x:left_eye_x + left_eye_width]
                        resized_eye = cv2.resize(eye_tracking, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
                        

                        

                        cv2.imshow('Eye Tracking', eye_tracking)
                        cv2.imshow('Resized_eye', resized_eye)

    

    # Display the frame with the face detection and facial landmarks
    cv2.imshow('Face Tracking', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()