import cv2

from lab_1_1.viz_utils import *
from lab_2_1.conv_ops import get_edges
from lab_2_1.viz_utils import subplot_images

import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


def get_result_1(img_1):
    return cv2.pyrMeanShiftFiltering(img_1, 11, 21)


def main():
    path = "images/count1.jpg"
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shifted = get_result_1(img)

    subplot = np.concatenate((img, shifted), axis=1)
    # cv2.imshow("shifted", subplot)
    # cv2.waitKey(-1)

    gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(-1)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # show the output image
    # cv2.imshow("name", image)
    # cv2.waitKey(-1)

    D = ndimage.distance_transform_edt(thresh)

    # print(D.max())
    # cv2.imshow("D", D)
    # cv2.waitKey(-1)
    plt.imshow(D)
    plt.show()

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

    # cv2_imshow(img)
    cv2.imshow("labeled_img", labeled_img)
    cv2.waitKey(-1)

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
        mask = np.zeros(gray.shape, dtype="uint8")
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

    # show the output image
    cv2.imshow("result", image)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
