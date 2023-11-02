import cv2
import numpy as np

# Load an image from your file system
image_path = cv2.imread(r'C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\100 Billeder cirka\Almond Blossoms 8.jpg')
image = cv2.resize(image_path, (595, 842))

# Load a reference image
reference_image = cv2.imread(r'C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\Malerier\Almond Blossoms.jpg')

# Convert the image to grayscale (required for edge detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using the Canny algorithm
edges = cv2.Canny(gray_image, threshold1=340, threshold2=800)  # You can adjust the thresholds

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assumed to be the frame)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

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

    # Display the rotated and rectified image
    cv2.imshow("Rotated and Rectified Image", rotated_rectified_image)
else:
    print("No contour found")

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
