import cv2 as cv
import numpy as np

# Load the original color image
image = cv.imread(r'100 Billeder cirka\Mona Lisa 7.jpg')
#image = cv.imread(r'Malerier\Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg')
# Convert the color image to the HSV color space
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define the lower and upper bounds for the color you are interested in (in HSV format)
lower_color = np.array([20, 20, 20])  # Set your lower HSV thresholds
upper_color = np.array([255, 255, 255])  # Set your upper HSV thresholds

# Threshold the image to create a binary mask based on the color
color_mask = cv.inRange(hsv, lower_color, upper_color)

# Find contours in the binary mask
contours, _ = cv.findContours(color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Define minimum contour area (10% of the total image area)
min_contour_area = 0.1 * 4000 * 1800

# Print the number of contours found
print(f'Number of contours found: {len(contours)}')

# Find the contour with the largest area
largest_contour = max(contours, key=cv.contourArea)

# Draw the largest contour in blue on the original image
cv.drawContours(image, [largest_contour], 0, (255, 0, 0), 2)

# Extract the largest contour from the original color image
x, y, w, h = cv.boundingRect(largest_contour)
cropped_object = image[y:y+h, x:x+w]

# Calculate the average HSV values of the object
average_hsv = np.mean(hsv[y:y+h, x:x+w], axis=(0, 1))

# Print the average HSV values
print(f'Average HSV: H={average_hsv[0]}, S={average_hsv[1]}, V={average_hsv[2]}')

# Display the cropped object
cv.imshow('Cropped Object', cropped_object)
cv.waitKey(0)
cv.destroyAllWindows()
