import numpy as np
import cv2


def get_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img / 255.


def subplot_images(img1, img2):
    s1, s2 = img1.shape[:2], img2.shape[:2]

    if s1[0] != s2[0]:
        if s1[0] > s2[0]:
            img1 = cv2.resize(img1, s2)
        else:
            img2 = cv2.resize(img2, s1)

    img_both = np.concatenate((img1, img2), axis=1)
    img_both *= 255
    return img_both
