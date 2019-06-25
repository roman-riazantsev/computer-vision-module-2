import numpy as np
import cv2
from skimage import io


def url_to_image(url):
    print("downloading %s" % (url))
    return cv2.cvtColor(io.imread(url), cv2.COLOR_BGR2RGB)


url = "https://raw.githubusercontent.com/mdavydov/ComputerVisionCourse/master/images/view.jpg"
img = url_to_image(url)

(r, c) = img.shape[:2]
pts1 = np.float32([[0, 0], [r, 0], [0, c]])
pts2 = np.float32([[50, 0], [r, 50], [0, c - 50]])

# M = np.array([pts[]])

# implement this function
M = cv2.getAffineTransform(pts1, pts2)
print(M)

dst = cv2.warpAffine(img, M, img.shape[:2])

# transformed_image = transform_matrix.dot(image)
pts1, pts2, pts3 = np.float32([[0, 0], [r, 0], [0, c]])
p1_2, p2_2, p3_2 = np.float32([[50, 0], [r, 50], [0, c - 50]])

matrix = np.array([[pts1[0], pts1[1], 1, 0, 0, 0],
                   [0, 0, 0, pts1[0], pts1[1], 1],
                   [pts2[0], pts2[1], 1, 0, 0, 0],
                   [0, 0, 0, pts2[0], pts2[1], 1],
                   [pts3[0], pts3[1], 1, 0, 0, 0],
                   [0, 0, 0, pts3[0], pts3[1], 1]])

final = [p1_2[0], p1_2[1], p2_2[0], p2_2[1], p3_2[0], p3_2[1]]

result = np.linalg.solve(matrix, final).reshape(2, 3)

print(result)

