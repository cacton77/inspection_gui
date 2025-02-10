import cv2
import numpy as np

# Generate random data
data = np.random.randint(0, 100, 100)

# Create a blank image
height, width = 400, 800
image = np.ones((height, width, 3), dtype=np.uint8) * 255

# Define the bounding box
bbox_top_left = (50, 50)
bbox_bottom_right = (750, 350)
cv2.rectangle(image, bbox_top_left, bbox_bottom_right, (0, 0, 0), 2)

# Plot the data as vertical bars
bar_width = (bbox_bottom_right[0] - bbox_top_left[0]) // len(data)
for i, value in enumerate(data):
    x1 = bbox_top_left[0] + i * bar_width
    y1 = bbox_bottom_right[1] - int((bbox_bottom_right[1] - bbox_top_left[1]) * (value / 100))
    x2 = x1 + bar_width - 1
    y2 = bbox_bottom_right[1]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

# Display the image
cv2.imshow('Minimalist Plot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

