import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread(r"C:\Users\jakob\Desktop\Git Things\MYseum\IRL\Girl with da perl training data.jpg")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to separate the painting from the paper
_, thresh = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours (you may need to adjust the size threshold)
min_contour_size = 10
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_size]

# Initialize a list to store potential paintings
potential_paintings = []

# Iterate over the filtered contours and create bounding boxes
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    potential_paintings.append((x, y, w, h))

# Filter out small bounding boxes based on a threshold
min_painting_area = 10  # Adjust this threshold as needed
potential_paintings = [p for p in potential_paintings if p[2] * p[3] > min_painting_area]

# Now, analyze the content of the remaining potential paintings to identify the actual paintings

# For example, you can use color histogram analysis or other image features to distinguish paintings from the background.

# Load your training image
training_image = cv2.imread(r"Malerier\Girl_with_a_Pearl_Earring.jpg")

# Convert the image to the RGB color space (assuming it's in BGR)
training_image_rgb = cv2.cvtColor(training_image, cv2.COLOR_BGR2RGB)

# Separate the channels
r, g, b = cv2.split(training_image_rgb)

# Create histograms for each channel
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

# Plot the histograms using Matplotlib
plt.figure()
plt.title("Color Histogram")
plt.xlabel("Color Value")
plt.ylabel("Pixel Count")
plt.plot(hist_r, color='red', label='Red Channel')
plt.plot(hist_g, color='green', label='Green Channel')
plt.plot(hist_b, color='blue', label='Blue Channel')
plt.xlim([0, 256])
plt.legend()
plt.show()

# Once you've identified the actual paintings, you can draw bounding boxes around them and display the result.

# Display the result
for x, y, w, h in potential_paintings:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the result
#cv2.imshow('Detected Painting', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("prut")

# Grayscale histogram
# histogram = cv2.calcHist([imageGray], [0], None, [256], [0, 256])
# plt.figure()
# plt.axis("off")
# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(histogram)
# plt.xlim([0, 256])
# plt.show()