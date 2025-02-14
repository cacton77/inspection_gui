import cv2
import numpy as np

# URLs of the live streams
url1 = "http://192.168.1.241:5000/video_feed"
url2 = "http://192.168.1.241:5000/video_feed2"

# Open connections to the video streams
cap1 = cv2.VideoCapture(url1)
cap2 = cv2.VideoCapture(url2)

while True:
    # Capture frame-by-frame from both streams
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    # If frames are read correctly, ret is True
    if not ret1 or not ret2:
        print("Failed to grab frame")
        break

    # Rotate the first frame by 90 degrees clockwise
    frame1_rotated = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)

    # Rotate the second frame by 90 degrees counterclockwise
    frame2_rotated = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Concatenate the two frames side by side
    combined_frame = np.concatenate((frame1_rotated, frame2_rotated), axis=1)

    # Display the resulting frame
    cv2.imshow('Live Stream', combined_frame)

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the captures
cap1.release()
cap2.release()
cv2.destroyAllWindows()
