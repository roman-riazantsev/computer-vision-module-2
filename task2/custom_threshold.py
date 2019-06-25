import numpy as np
import cv2


def tilted_threshold(img, left, right, middle=0.5, middle_pos=0):
    h, w = img.shape[:2]
    if middle_pos:
        left_part = w * middle_pos
        right_part = w - left_part

        plane_left = [np.linspace(left, middle, left_part) for _ in range(h)]
        plane_right = [np.linspace(middle, right, right_part) for _ in range(h)]
        plane = np.concatenate([plane_left, plane_right], axis=1)
    else:
        plane = [np.linspace(left, right, w) for _ in range(h)]
        plane = np.array(plane)

    plane = cv2.resize(plane, (int(w), int(h)))
    plane[plane < 0] = 0
    result = np.where(img > plane, 255, 0)
    result = result.astype(np.uint8)

    return result


path = "images/count2.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, None, fx=2, fy=2)
res = tilted_threshold(img, 90, 110, 30, 0)
cv2.imshow("subplot", res)
cv2.waitKey(-1)
