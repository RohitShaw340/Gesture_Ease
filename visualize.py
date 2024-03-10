import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load data from file
numpy_data = np.load("Datasets/x.npy")
data = numpy_data[35]
colors = plt.cm.jet(np.linspace(0, 1, len(data)))

#show image

for i, frame in enumerate(data):
    x = frame[:, 0]  # X coordinates
    y = frame[:, 1]  # Y coordinates
    z = frame[:, 2]  # Z coordinates

    # Plot the points with different colors for each frame
    plt.scatter(x, y, c=[colors[i]], label=f"Frame {i}")

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Hand Landmarks')
plt.legend()
plt.show()









fps = 30
frame_width = 1280  # Increase the frame width
frame_height = 720  # Increase the frame height
frame_size = (frame_width, frame_height)

# Create a VideoWriter object
out = cv2.VideoWriter('hand_landmarks_video.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)
# OpenCV loop to display video in a loop until 'q' is pressed
while True:
    # Plot each frame and write to video
    for i, frame in enumerate(data):
        x = frame[:, 0]  # X coordinates
        y = frame[:, 1]  # Y coordinates
        z = frame[:, 2]  # Z coordinates
        
        # Create a blank frame
        img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        
        # Plot the points with different colors for each frame
        for j in range(len(x)):
            cv2.circle(img, (int(x[j]*frame_size[0]), int(y[j]*frame_size[1])), 5, (int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255)), -1)
        
        # Write frame to video
        out.write(img)

        # Display frame
        cv2.imshow('Hand Landmarks Video', img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Check if 'q' is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release VideoWriter and close OpenCV windows
out.release()
cv2.destroyAllWindows()
