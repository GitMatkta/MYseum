import cv2
import numpy as np

# Section 1: Load and Preprocess the Image

# Load an image from your file system
image_path = cv2.imread(r'100 Billeder cirka\Almond Blossoms 8.jpg')
image = cv2.resize(image_path, (595, 842))

# Load a reference image
reference_image = cv2.imread(r'Malerier\Almond Blossoms.jpg')

# Convert the image to grayscale (required for edge detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Section 2: Edge Detection

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(gray_image, threshold1=340, threshold2=800)  # You can adjust the thresholds

# Section 3: Contour Detection

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the frame)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Section 4: Rectification

    # Get the orientation angle of the largest contour
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Find the four corners of the bounding box
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype='float32')

    # Define the target corners for the rectified image (a rectangle)
    target_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype='float32')

    # Compute the perspective transformation matrix (homography)
    M = cv2.getPerspectiveTransform(corners, target_corners)

    # Apply the perspective transformation to rectify the image
    rectified_image = cv2.warpPerspective(image, M, (w, h))

    # Rotate the rectified image to make it horizontal
    if angle < -45:
        angle += 90
    rotated_rectified_image = cv2.warpAffine(rectified_image, cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1), (w, h))

else:
    print("No contour found")

# Section 5: White Region Removal

# Define the upper threshold for white color
white_threshold = 180

# Threshold the image to remove white regions
thresholded_image = cv2.inRange(rotated_rectified_image, (white_threshold, white_threshold, white_threshold), (255, 255, 255))

# Invert the thresholded image (so white regions become black and vice versa)
thresholded_image = cv2.bitwise_not(thresholded_image)

# Apply the thresholded mask to your original image
removed_white = cv2.bitwise_and(rotated_rectified_image, rotated_rectified_image, mask=thresholded_image)

# Section 6: Display the Result

# Display the result image
cv2.imshow("Result Image", removed_white)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
