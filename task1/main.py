import cv2
import numpy as np

from task1.image_processor import *


def save_result(images, i):
    result = np.concatenate(images, axis=1)
    cv2.imwrite("results/result{}.jpg".format(i), result)


def main():
    for i in range(9, 10):
        path = "images/text{}.jpg".format(i)
        img = loading_functions[i - 1](path)
        segmented_text = extraction_functions[i - 1](img)
        save_result([img, segmented_text], i)


if __name__ == "__main__":
    main()
