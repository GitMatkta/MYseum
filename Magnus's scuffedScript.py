import cv2

# Load the image
image = cv2.imread(r'C:\University\P3\Project\MYseum\IRL\Girl with da perl training data.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Thresholding (adjust the threshold value as needed)
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours in the binary image
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image (for visualization)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Analyze contours to identify holes
for contour in contours:
    # If a contour has child contours, it contains a hole
    if len(contour) > 5:
        print("Found a hole!")
        # You can further process or draw the contours representing holes here

# Display the results
cv2.imshow('Original Image with Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
