import cv2

# Load reference painting and wall photo
reference_painting = cv2.imread(r'C:\University\P3\Project\MYseum\Malerier\Girl_with_a_Pearl_Earring.jpg')
wall_photo = cv2.imread(r'C:\University\P3\Project\MYseum\IRL\Girl with da perl training data.jpg')

# Convert images to grayscale
reference_gray = cv2.cvtColor(reference_painting, cv2.COLOR_BGR2GRAY)
wall_gray = cv2.cvtColor(wall_photo, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to wall photo
binary_wall = cv2.adaptiveThreshold(wall_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Perform contour detection
contours, _ = cv2.findContours(binary_wall, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter Contours based on Aspect Ratio and Area
filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)

    # Adjust aspect_ratio and area thresholds based on your specific case
    if 0.6 < aspect_ratio < 1.4 and 5000 < area < 50000:
        filtered_contours.append(contour)
        cv2.rectangle(wall_photo, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangles around potential paintings
        cv2.putText(wall_photo, 'Potential Painting', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extract potential painting ROI
        potential_painting_roi = wall_gray[y:y+h, x:x+w]

        # Perform histogram matching between potential_painting_roi and reference_gray
        matched_painting = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(potential_painting_roi)
        matched_painting = cv2.cvtColor(matched_painting, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for visualization

        # Display the matched painting
        cv2.imshow('Matched Painting', matched_painting)
        cv2.waitKey(0)

# Display the results with potential paintings and their matches
cv2.imshow('Contours Detection Result', wall_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()