import cv2
import numpy as np

# Load the main image and the template image
main_image = cv2.imread(r"C:\Users\rikke\Git\MYseum\IRL\Girl with da perl training data.jpg")
template = cv2.imread(r"C:\Users\rikke\Git\MYseum\Malerier\Girl_with_a_Pearl_Earring.jpg")

# Convert images to grayscale
main_gray = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Find the dimensions of the template image
template_height, template_width = template_gray.shape[:2]

# Use template matching to find the location of the template in the main image
res = cv2.matchTemplate(main_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 8  # Adjust this threshold as needed
loc = np.where(res >= threshold)

# Iterate over the locations and draw a bounding box around the found templates
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + template_width, pt[1] + template_height)
    cv2.rectangle(main_image, pt, bottom_right, (0, 0, 255), 2)  # Red rectangle

# Save or display the result
cv2.imwrite("result_image.jpg", main_image)
cv2.imshow("Result", main_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
