import cv2

from lab_1_1.viz_utils import *
from lab_2_1.conv_ops import get_edges
from lab_2_1.viz_utils import subplot_images


def main():
    path = "images/text1.jpg"
    img = get_img(path, normalize=False)
    edges = cv2.Canny(img, 100, 200)
    subplot = np.concatenate((img, edges), axis=1)
    # sybplot = subplot_images(img, edges)
    cv2.imshow('img', subplot)
    cv2.waitKey(-1)


if __name__ == "__main__":
    main()
