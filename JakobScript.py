import cv2
import matplotlib.pyplot as plt

image = cv2.imread(r"C:\Users\jakob\Desktop\Git Things\MYseum\IRL\Girl with da perl training data.jpg")
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




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

cv2.waitKey(0)

print("prut")
