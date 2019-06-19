import numpy as np
import cv2

url = "https://raw.githubusercontent.com/mdavydov/ComputerVisionCourse/master/images/view.jpg"

img = cv2.imread(url)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(r, c) = img.shape[:2]
pts1 = np.float32([[0, 0], [r, 0], [0, c]])
pts2 = np.float32([[50, 0], [r, 50], [0, c - 50]])

# implement this function
M = cv2.getAffineTransform(pts1, pts2)
print(M)

dst = cv2.warpAffine(img, M, img.shape[:2])
cv2.imshow(img)
cv2.waitKey(-1)
cv2.imshow(dst)
cv2.waitKey(-1)
