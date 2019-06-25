import cv2
import numpy as np

from task1.light_utils import reset_light


def tilted_threshold(img, threshold):
    h, w = img.shape[:2]

    left, middle, right, middle_pos = threshold

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


def double_threshold(img, threshold_bottom, threshold_top):
    h, w = img.shape[:2]

    plane_bottom = np.ones((h, w)) * threshold_bottom
    plane_top = np.ones((h, w)) * threshold_top

    result = np.where(np.logical_and(img > plane_bottom, img < plane_top), 255, 0)
    result = result.astype(np.uint8)
    return result


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def cv2clahe(img, clipLimit, tileGridSize):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def open_img_1(path):
    img = cv2.imread(path)
    shifted = cv2.pyrMeanShiftFiltering(img, 11, 21)
    img = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    return img


def open_img_8(path):
    img = cv2.imread(path)
    img = reset_light(img, 1.5, -100)
    img = cv2.pyrMeanShiftFiltering(img, 7, 21)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    return img


def open_img_9(path):
    img = cv2.imread(path)
    img = cv2.pyrMeanShiftFiltering(img, 3, 19)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def open_img_10(path):
    img = cv2.imread(path)
    return img


def extraction_function_1(img):
    threshold = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    res = cv2.fastNlMeansDenoising(threshold, None, 64, 7, 16)
    return res


def extraction_function_3(img):
    img = cv2.pyrMeanShiftFiltering(img, 3, 19)
    img = cv2.fastNlMeansDenoising(img, None, 3, 7, 8)
    img = reset_light(img, 2, -100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    img = tilted_threshold(img, [50, 100, 120, 0.9])
    return img


def extraction_function_5(img):
    img = cv2.bitwise_not(img)

    res, threshold = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    threshold = cv2.fastNlMeansDenoising(threshold, None, 31, 7, 8)
    return threshold


def extraction_function_6(img):
    img = cv2.bitwise_not(img)

    threshold = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    res = cv2.fastNlMeansDenoising(threshold, None, 63, 7, 16)
    return res


def extraction_function_7(img):
    threshold = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    res = cv2.fastNlMeansDenoising(threshold, None, 39, 11, 50)
    return res


def extraction_function_8(img):
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, None, fx=2, fy=2)
    # image1 = cv2.dilate(img, kernel, iterations=1)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 5)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    return img


def extraction_function_9(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = reset_light(img, 1.5, -100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = tilted_threshold(img, [110, 125, -200, 0.6])
    img = cv2.bitwise_not(img)
    return img


def extraction_function_10(img):
    img = cv2.fastNlMeansDenoising(img, None, 50, 3, 10)
    img = reset_light(img, 1.5, -100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = tilted_threshold(img, [10000, 180, 80, 0.16])
    img = cv2.bitwise_not(img)

    return img


def extraction_function_11(img):
    # img = cv2.blur(img, (2, 2))
    img = cv2.pyrMeanShiftFiltering(img, 21, 3)
    # img = reset_light(img, 2, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.bitwise_not(img)
    # res, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    img = tilted_threshold(img, [10000, 130, 130, 0.2])
    img = cv2.fastNlMeansDenoising(img, None, 31, 9, 31)
    img = cv2.bitwise_not(img)
    # threshold = cv2.fastNlMeansDenoising(threshold, None, 31, 7, 8)
    return img


extraction_functions = [extraction_function_1, extraction_function_1, extraction_function_3, extraction_function_1,
                        extraction_function_5, extraction_function_6, extraction_function_7, extraction_function_8,
                        extraction_function_9, extraction_function_10, extraction_function_11]

loading_functions = [open_img_1, open_img_1, open_img_10, open_img_1, open_img_1, open_img_1,
                     open_img_1, open_img_8, open_img_9, open_img_10, open_img_10]
