import cv2
import numpy as np

# Load the image
image = cv2.imread(r'100 Billeder cirka\Almond Blossoms 1.jpg')
imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Convert the image to grayscale
gray = cv2.cvtColor(imageHSV, cv2.COLOR_BGR2GRAY)

# Apply histogram equalization to enhance contrast
equalized = cv2.equalizeHist(gray)

blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

# Apply Canny edge detection with adjusted threshold values
edges = cv2.Canny(equalized, 20, 100)  # You may need to adjust the threshold values

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours (you may need to adjust the size threshold)
min_contour_size = 1000
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_size]

filtered_contours2 = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)

    # Adjust aspect_ratio and area thresholds based on your specific case
    if 0.6 < aspect_ratio < 1.4 and 5000 < area < 50000:
        filtered_contours2.append(contour)
        cv2.rectangle(imageHSV, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangles around potential paintings
        cv2.putText(imageHSV, 'Potential Painting', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract potential painting ROI
        potential_painting_roi = gray[y:y+h, x:x+w]

        # Perform histogram matching between potential_painting_roi and reference_gray
        matched_painting = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(potential_painting_roi)
        matched_painting = cv2.cvtColor(matched_painting, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for visualization

        # Display the matched painting
        cv2.imshow('Matched Painting', matched_painting)
        cv2.waitKey(0)

# Now, you can analyze the remaining contours to identify the actual painting.

# For example, you can use bounding boxes or other techniques to locate the painting within the contours.

# Display the result with detected edges
edges2 = cv2.resize(edges, (360, 800))
cv2.imshow('Edge Detection Result', edges2)
contourImage = cv2.resize(imageHSV, (360, 800))
cv2.imshow('Contours Detection Result', contourImage)
cv2.waitKey(0)
cv2.destroyAllWindows()





