import cv2
import numpy as np


def foo():
    # open image:
    path = "images/count1.jpg"
    img = cv2.imread(path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] += 0
    img_brighten = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    stacked = np.hstack((img, img_brighten))
    cv2.imshow("something", stacked)
    cv2.waitKey(-1)


foo()
