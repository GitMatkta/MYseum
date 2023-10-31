import cv2
import numpy as np

input_image = cv2.imread(r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\IRL\Girl with da perl training data.jpg")

mask = np.zeros(input_image.shape[:2], np.uint8)

rect = (50, 50, input_image.shape[1] - 100, input_image.shape[0] - 100)

cv2.grabCut(input_image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)

sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

edge_image = np.uint8(edge_magnitude)

result = edge_image * mask2

contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result_image = input_image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()