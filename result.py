import numpy as np
from keras.models import load_model
import mediapipe as mp
import cv2
import numpy as np
import time

model = load_model("hand_gesture_model.h5")
lable_map = np.load("Datasets/label_map.npy", allow_pickle=True).item()

def predict(data):
    # Find avg x,y,z
    avg_x = data[0, :, 0].mean()
    avg_y = data[0, :, 1].mean()
    avg_z = data[0, :, 2].mean()

    # Change the origin
    data[:, :, 0] -= avg_x
    data[:, :, 1] -= avg_y
    data[:, :, 2] -= avg_z
    # Reshape the data to match the input shape expected by the model
    # Here, we assume the model expects input shape (batch_size, timesteps, input_dim)
    # Adjust the reshape according to your model's input shape
    data = data.reshape(1, 30, -1)
    print(data.shape)
    # Make predictions
    predictions = model.predict(data)
    print(predictions)

    # Get the predicted label
    predicted_label = np.argmax(predictions)

    # Print the predicted label
    print("Predicted gesture:", predicted_label)
    return lable_map[predicted_label]

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
    output=""
    # Load the Hand Landmark Detection model
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )
    # frame_count = 0
    frame_data = []

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, output, (0, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
            print(len(frame_data), " Appended")
        # Show the frame with landmarks
        cv2.imshow("Hand Landmarks", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if len(frame_data) == 30:
            output = predict(np.array(frame_data))
            # text = f"Predicted gesture: {output}"
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 1
            # font_thickness = 2
            # text_color = (0, 0, 255)  # Red color in BGR format
            # text_position = (50, frame.shape[0] - 50)  # Bottom-left corner of the frame

            # # Get the size of the text
            # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            # # Adjust text position to ensure it fits within the frame
            # if text_position[0] + text_size[0] > frame.shape[1]:
            #     text_position = (frame.shape[1] - text_size[0] - 10, text_position[1])
            # if text_position[1] - text_size[1] < 0:
            #     text_position = (text_position[0], text_size[1] + 10)

            # # Draw the text on the frame
            # cv2.putText(frame, text, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            
            print("gesture : " , output)
            frame_data=[]

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

collect_data()
