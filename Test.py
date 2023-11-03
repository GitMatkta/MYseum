import cv2

# Load an image from your file system
image_path = cv2.imread(r'C:\Users\nicol\Documents\GitHub\MYseum\100 Billeder cirka\Almond Blossoms 7.jpg')
image = cv2.resize(image_path, (595, 842))

# Convert the image to grayscale (required for edge detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection using the Canny algorithm
# For Images 1, 2, 3, 4, 5, 9, 10
# edges = cv2.Canny(gray_image, threshold1=100, threshold2=150)  # You can adjust the thresholds

# Images 6(340 _ 800), 7(380 _ 800), 8(380 _ 800) needs to have changed threshold
edges = cv2.Canny(gray_image, threshold1=340, threshold2=800)  # You can adjust the thresholds

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Find the largest contour (assumed to be the frame)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the original image based on the bounding box
    cropped_image = image[y:y+h, x:x+w]

    # Display the cropped image
    cv2.imshow("Cropped Image", cropped_image)
else:
    print("No contour found")

# Display the result
cv2.imshow("Edge Detection", image)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
