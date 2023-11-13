import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


image = cv.imread(r"100 Billeder cirka\Babel 1.jpg")

b = 0
g = 0
r = 0

for i in range(3):
    histr = cv.calcHist([image[:, :, i]], [0], None, [256], [0, 256])

    if i == 0:
        for j in range(len(histr)):
            b += histr[j][0]
        print(b)
    elif i == 1:
        for j in range(len(histr)):
            g += histr[j][0]
        print(g)
    else:
        for j in range(len(histr)):
            r += histr[j][0]
        print(r)

    plt.plot(histr, color='r')
    plt.xlim([0, 256])
    plt.show()
