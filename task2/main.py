import cv2
import numpy as np
import imutils
from task2.light_utils import reset_light
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


def preprocess(img):
    img = reset_light(img, 2, 32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def get_mask(img):
    img2 = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]

    img = cv2.bitwise_not(img2)
    # after = cv2.fastNlMeansDenoising(img, None, 3, 7, 64)

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
    # image1 = cv2.dilate(img, kernel, iterations=1)
    image2 = cv2.erode(img, kernel, iterations=1)

    after = cv2.bitwise_not(image2)
    subplot(img2, after)

    kernel = np.ones((3, 3), np.uint8)
    img2 = cv2.dilate(img2, kernel, iterations=1)

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

    return image


def analyse_connections(D, thresh):
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    label_hue = np.uint8(179 * labels / np.max(labels))
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    return labels, labeled_img


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


def main():
    path = "images/count2.jpg"
    img = cv2.imread(path)
    img = cv2.resize(img, None, fx=2, fy=2)

    img2 = preprocess(img)
    mask = get_mask(img2)

    subplot(img, mask)

    img2 = find_contours(img, mask)
    subplot(img2, mask)
    img2 = distance_transform(mask)
    #
    # img2_c = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    labels, colorful = analyse_connections(img2, mask)

    img = draw_labels(img, mask, labels)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow("subplot", img)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
