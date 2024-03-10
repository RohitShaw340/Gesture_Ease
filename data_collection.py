import mediapipe as mp
import cv2
import numpy as np
import time


def detect_hand_landmarks(image):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    results = mp_hands.process(image)

    landmarks = []
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        for landmark in hand_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))

    return landmarks


def collect_data():
    # Load the Hand Landmark Detection model
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    file_name = input("Enter Gesture Name : ")
    frame_count = 0
    dataset = []
    frame_data = []

    # Open the webcam
    cap = cv2.VideoCapture(0)

    timer = 0

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = mp_hands.process(frame_rgb)
        # Draw landmarks on the frame
        if results.multi_hand_landmarks:
            temp = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x, y, z = landmark.x, landmark.y, landmark.z
                    # print(landmark)
                    temp.append((x, y, z))
                    # Draw a circle at each landmark position
                    cv2.circle(
                        frame,
                        (int(x * frame.shape[1]), int(y * frame.shape[0])),
                        5,
                        (0, 255, 0),
                        -1,
                    )

            frame_data.append(temp)
            frame_count += 1
            print(frame_count % 30, " Appended")
        # Show the frame with landmarks
        cv2.imshow("Hand Landmarks", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_count != 0 and frame_count % 30 == 0:
            dataset.append(frame_data)
            frame_data = []
            print("Data Collected : ", len(dataset))
            print("Data appended")
            # cv2.waitKey(1)

            print("Start collecting data again....")

        if len(dataset) == 30:
            break
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # print(dataset)
    arr = np.array(dataset)
    print(arr.shape)
    np.save(file_name, dataset)
    # landmarks = detect_hand_landmarks(image)
    # print(landmarks)

collect_data()