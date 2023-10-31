import cv2
import numpy as np

input_image = cv2.imread(r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\IRL\Girl with da perl training data.jpg")

gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

_, thresholded = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

largest_contour = max(contours, key=cv2.contourArea)

mask = np.zeros_like(input_image)
cv2.drawContours(mask, [largest_contour], 0, (255, 255, 255), -1)

cropped_background = cv2.bitwise_and(input_image, mask)

# Define the color range for the frame (adjust the values as needed)
lower_white = np.array([211, 211, 211], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

# Create a mask for the white frame in the cropped_background
frame_mask = cv2.inRange(cropped_background, lower_white, upper_white)

# Find contours in the mask
contours, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (the frame)
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the frame
frame_mask = np.zeros_like(cropped_background)
cv2.drawContours(frame_mask, [largest_contour], 0, (255, 255, 255), -1)

# Invert the frame mask
frame_mask_inv = cv2.bitwise_not(frame_mask)

# Remove the frame from the cropped_background
image_without_frame = cv2.bitwise_and(cropped_background, frame_mask_inv)

# Perform noise reduction and remove thin pixel lines
kernel = np.ones((5, 5), np.uint8)

noise_reduced_image = cv2.morphologyEx(image_without_frame, cv2.MORPH_OPEN, kernel)

cv2.imshow("Cleaned Image", noise_reduced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
