import cv2


def open_img(path):
    img = cv2.imread(path)
    shifted = cv2.pyrMeanShiftFiltering(img, 11, 21)
    img = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
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
    threshold = cv2.fastNlMeansDenoising(threshold, None, 32, 7, 8)
    return threshold


def extraction_function_6(img):
    img = cv2.bitwise_not(img)

    threshold = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 5)
    res = cv2.fastNlMeansDenoising(threshold, None, 64, 7, 16)
    return res


extraction_functions = [extraction_function_1, extraction_function_1, extraction_function_1, extraction_function_1,
                        extraction_function_5, extraction_function_6]

# def segment_function_2(img):
#     retval, threshold_otsu = cv2.threshold(img, 100, 120, cv2.THRESH_BINARY)
#     return threshold_otsu


# def segment_function_3(img):
#     img = cv2.Canny(img, 100, 200)
#     return img
