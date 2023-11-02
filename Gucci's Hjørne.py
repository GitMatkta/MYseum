import cv2

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

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
