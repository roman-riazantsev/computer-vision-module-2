import cv2
import numpy as np

from task1.light_utils import reset_light


def open_img_1(path):
    img = cv2.imread(path)
    return img


def open_img_2(path):
    img = cv2.imread(path)
    return img


def check_rectangle(img, x1, y1, x2, y2):
    rect_img = img.copy()

    pt1, pt2 = (x1, y1), (x2, y2)
    cv2.rectangle(rect_img, pt1, pt2, (255, 0, 0), 2)
    cv2.imshow("rect", rect_img)
    cv2.waitKey(-1)


def crop_by_mask(img, mask2):
    img = img * mask2[:, :, np.newaxis]
    cv2.imshow("mask", mask2 * 255)
    cv2.waitKey(-1)
    cv2.imshow("img", img)
    cv2.waitKey(-1)
    return img


def get_grab_mask(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    after = img.copy()
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x1, y1, x2 - x1, y2 - y1)

    cv2.grabCut(after, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return mask2


def extraction_function_1(img):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 40, 20, w - 30, h - 20

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)
    result = crop_by_mask(img, mask)

    return result


def extraction_function_2(img):
    result = img.copy()
    img = reset_light(img, 0.8, 0)
    img = cv2.blur(img, (11, 11))
    cv2.imshow("img", img)
    cv2.waitKey(-1)
    img = cv2.pyrMeanShiftFiltering(img, 21, 21)

    h, w = img.shape[:2]
    x1, y1, x2, y2 = 100, 70, w - 50, h - 40

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.fastNlMeansDenoising(mask, None, 7, 21, 8)

    result = crop_by_mask(result, mask)

    return result


def extraction_function_3(img):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 60, 0, w - 200, h - 150

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)
    result = crop_by_mask(img, mask)

    return result


def extraction_function_4(img):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 70, 20, w - 70, h

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)
    result = crop_by_mask(img, mask)
    return result


def extraction_function_5(img):
    result = img.copy()
    h, w = img.shape[:2]
    # img = reset_light(img, 0.8, 0)
    x1, y1, x2, y2 = 40, 15, w - 30, h - 5
    img = cv2.pyrMeanShiftFiltering(img, 21, 21)

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)
    result = crop_by_mask(result, mask)

    return result


def extraction_function_6(img):
    result = img.copy()
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 55, 80, w - 80, h - 25

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(20, 20))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.pyrMeanShiftFiltering(img, 21, 14)

    check_rectangle(img, x1, y1, x2, y2)
    mask = get_grab_mask(img, x1, y1, x2, y2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.fastNlMeansDenoising(mask, None, 7, 21, 8)

    result = crop_by_mask(result, mask)

    return result


extraction_functions = [extraction_function_1, extraction_function_2, extraction_function_3, extraction_function_4,
                        extraction_function_5, extraction_function_6]

loading_functions = [open_img_1, open_img_2, open_img_1, open_img_1, open_img_1, open_img_1]
