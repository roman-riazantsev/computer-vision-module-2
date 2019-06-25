import cv2
import numpy as np
import imutils
from task2.light_utils import reset_light
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


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

    result = np.where(img > plane, 255, 0)
    result = result.astype(np.uint8)

    return result


def get_mask(img, threshold):
    if isinstance(threshold, list):
        img2 = tilted_threshold(img, threshold)
    else:
        img2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

    img = cv2.bitwise_not(img2)
    # after = cv2.fastNlMeansDenoising(img, None, 3, 7, 64)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    # image1 = cv2.dilate(img, kernel, iterations=1)
    image2 = cv2.erode(img, kernel, iterations=1)

    after = cv2.bitwise_not(image2)
    # subplot(img2, after)

    kernel = np.ones((3, 3), np.uint8)
    img2 = cv2.dilate(img2, kernel, iterations=1)

    return img2


def get_mask_2(img, threshold):
    if isinstance(threshold, list):
        img2 = tilted_threshold(img, threshold)
    else:
        img2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((3, 3), np.uint8)
    img2 = cv2.dilate(img2, kernel, iterations=1)

    return img2


def get_mask_3(img, threshold):
    if isinstance(threshold, list):
        img2 = tilted_threshold(img, threshold)
    else:
        img2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

    mask = cv2.fastNlMeansDenoising(img2, None, 7, 7, 10)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    return mask


def get_mask_4(img, threshold):
    if isinstance(threshold, list):
        img2 = tilted_threshold(img, threshold)
    else:
        img2 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]

    return img2


def find_contours(img, thresh):
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("[INFO] {} unique contours found".format(len(cnts)))

    image = img.copy()

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 1)

    return len(cnts), image


def analyse_connections(D, thresh):
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    res = len(np.unique(labels)) - 1
    print("[INFO] {} unique segments found".format(res))

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labels, labeled_img, res


def distance_transform(thresh):
    img2 = ndimage.distance_transform_edt(thresh)
    return img2


def subplot(img, mask):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    res = np.concatenate([img, mask], axis=1)
    cv2.imshow("subplot", res)
    cv2.waitKey(-1)


def draw_labels(img, mask, labels):
    # loop over the unique labels returned by the Watershed
    # algorithm
    image = img

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(mask.shape, dtype="uint8")
        mask[labels == label] = 255
        # cv2_imshow(mask)#[labels == label] = 255
        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return image


def preprocess(img, contrast, brightness):
    img = reset_light(img, contrast, brightness)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def open_img_1(path):
    img = cv2.imread(path)
    return img


def open_img_2(path):
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=2, fy=2)
    return img


def extraction_function_1(img):
    img = cv2.resize(img, None, fx=2, fy=2)
    img2 = preprocess(img, 2, 0)
    mask = get_mask_2(img2, threshold=[100, 50, 5, 0.7])
    cnts1, img2 = find_contours(img, mask)
    img2 = distance_transform(mask)
    labels, colorful, cnts2 = analyse_connections(img2, mask)
    img = draw_labels(img, mask, labels)
    return img, cnts2


def extraction_function_2(img):
    img2 = preprocess(img, 2, 32)
    mask = get_mask(img2, threshold=60)
    cnts, img2 = find_contours(img, mask)
    img2 = distance_transform(mask)
    labels, colorful, cnts = analyse_connections(img2, mask)
    img = draw_labels(img, mask, labels)

    return img, cnts


def extraction_function_3(img):
    img = cv2.resize(img, None, fx=2, fy=2)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(12, 12))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img2 = preprocess(img2, 1, 0)
    mask = get_mask_4(img2, threshold=130)

    kernel = np.ones((3, 3))
    mask = cv2.fastNlMeansDenoising(mask, None, 3, 7, 16)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    #
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=3)

    cnts, img2 = find_contours(img, mask)
    # img2 = distance_transform(mask)
    # labels, colorful, cnts2 = analyse_connections(img2, mask)
    # img2 = draw_labels(img, mask, labels)

    return img2, cnts


def extraction_function_4(img):
    img2 = img.copy()

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img2 = preprocess(img, 3, 0)
    mask = get_mask(img2, threshold=[170, 200, 150, 0.3])
    cnts, img2 = find_contours(img, mask)
    img2 = distance_transform(mask)
    labels, colorful, cnts2 = analyse_connections(img2, mask)
    img = draw_labels(img, mask, labels)

    return img, cnts2


def extraction_function_10(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img2 = cv2.equalizeHist(img2)

    mask = get_mask_4(img2, threshold=210)

    mask = cv2.fastNlMeansDenoising(mask, None, 21, 21, 10)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    cnts2, img2 = find_contours(img, mask)

    return img2, cnts2


extraction_functions = [extraction_function_1, extraction_function_2, extraction_function_3, extraction_function_4,
                        extraction_function_1, extraction_function_1, extraction_function_1, extraction_function_1,
                        extraction_function_1, extraction_function_10, extraction_function_1]

loading_functions = [open_img_1, open_img_2, open_img_1, open_img_1, open_img_1, open_img_1,
                     open_img_1, open_img_1, open_img_1, open_img_1, open_img_1]
