import cv2
import os

input_image = cv2.imread(r"C:\Users\Tze Huo Gucci Ho\Desktop\Git Projects\MYseum\Malerier\Girl_with_a_Pearl_Earring.jpg")

# Extract BGR histogram
bgr_planes = cv2.split(input_image)
histSize = 256
histRange = (0, 256) # the upper boundary is exclusive
accumulate = False
b_hist = cv2.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
g_hist = cv2.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
r_hist = cv2.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)



cv2.imshow("Original", input_image)
cv2.waitKey(0)
cv2.destroyAllWindows()