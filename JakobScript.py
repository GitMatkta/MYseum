import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\jakob\Desktop\Git Things\MYseum\IRL\Girl with da perl training data.jpg")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Apply thresholding to separate the painting from the paper
_, thresh = cv2.threshold(imageGray, 200, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded image
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out small contours (you may need to adjust the size threshold)
min_contour_size = 1000
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_size]

# Choose the largest remaining contour
largest_contour = max(filtered_contours, key=cv2.contourArea)

# Create a bounding box around the largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw the bounding box on the original image
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)



# Display the result
cv2.imshow('Detected Painting', image)
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