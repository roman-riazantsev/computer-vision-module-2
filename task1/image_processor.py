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
    # img = reset_light(img, 0., 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def extraction_function_1(img):
    threshold = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    res = cv2.fastNlMeansDenoising(threshold, None, 64, 7, 16)
    return res


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
    img = cv2.fastNlMeansDenoising(img, None, 11, 7, 10)
    img = cv2.adaptiveThreshold(img, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 5)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    return img


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def extraction_function_9(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = reset_light(img, 1.5, -100)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = tilted_threshold(img, [110, 125, -200, 0.6])
    return img


extraction_functions = [extraction_function_1, extraction_function_1, extraction_function_1, extraction_function_1,
                        extraction_function_5, extraction_function_6, extraction_function_7, extraction_function_8,
                        extraction_function_9, extraction_function_1, extraction_function_1]

loading_functions = [open_img_1, open_img_1, open_img_1, open_img_1, open_img_1, open_img_1,
                     open_img_1, open_img_8, open_img_9, open_img_1, open_img_1]

# def segment_function_2(img):
#     retval, threshold_otsu = cv2.threshold(img, 100, 120, cv2.THRESH_BINARY)
#     return threshold_otsu


# def segment_function_3(img):
#     img = cv2.Canny(img, 100, 200)
#     return img
