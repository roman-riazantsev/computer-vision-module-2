import cv2
import numpy as np

from task2.light_utils import reset_light


def process_photo(path):
    img = cv2.imread(path)
    # after = reset_light(img, 1, -20)
    # after = cv2.fastNlMeansDenoising(after, None, 7, 13, 100)
    # subplot = np.concatenate([img, after], axis=1)'
    after = img.copy()
    # cv2.imshow("img", subplot)
    # cv2.waitKey(-1)

    rect_img = img.copy()

    (h, w) = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    x1, y1, x2, y2 = 70, 20, w - 70, h
    rect = (x1, y1, w-140, h-20)
    pt1, pt2 = (x1, y1), (x2, y2)
    print(pt1, pt2)
    cv2.rectangle(rect_img, pt1, pt2, (255, 0, 0), 2)
    cv2.imshow("rect", rect_img)
    cv2.waitKey(-1)
    cv2.grabCut(after, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
    cv2.imshow("mask", mask2 * 255)
    cv2.waitKey(-1)
    cv2.imshow("img", img)
    cv2.waitKey(-1)


process_photo("images/obj4.jpg")
