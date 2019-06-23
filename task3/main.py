import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from matplotlib import pyplot as plt


def segmentation_function_1(img):
    (w, h) = img.shape[:2]
    mask = np.zeros((w, h), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (5, 5, w - 5, h - 5)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 2, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    result = img * mask[:, :, np.newaxis]
    mask = cv2.cvtColor(mask.astype('float32'), cv2.COLOR_GRAY2BGR) * 255

    subplot = np.concatenate([img, mask, result], axis=1) / 255.
    cv2.imshow("res", subplot)
    cv2.waitKey(-1)


def segmentation_function_2(img):
    img = rgb2gray(img)

    s = np.linspace(0, 2 * np.pi, 400)
    x = 290 + 200 * np.cos(s)
    y = 320 + 260 * np.sin(s)
    init = np.array([x, y]).T

    snake = active_contour(gaussian(img, 3),
                           init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()


def main():
    path = "images/pear.jpg"
    img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    shifted = cv2.pyrMeanShiftFiltering(img, 11, 100)
    cv2.imshow("shifter", shifted)
    cv2.waitKey(-1)
    segmentation_function_1(img)
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    # segmentation_function_2(img)


if __name__ == "__main__":
    main()
