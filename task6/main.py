import cv2
import numpy as np

from task6.patch_match import PatchMatch


def reconstruct(matches, img1, img2):
    h, w = img1.shape[:2]
    result = np.zeros_like(img1)
    for i in range(h):
        for j in range(w):
            y, x = matches[i, j]
            result[i, j, :] = img2[y, x, :]

    return result


if __name__ == "__main__":
    img1 = cv2.imread("images/v001.jpg")
    img2 = cv2.imread("images/v100.jpg")

    img1 = img1 / 255.
    img2 = img2 / 255.

    img1 = cv2.resize(img1, None, fx=0.25, fy=0.25)
    img2 = cv2.resize(img2, None, fx=0.25, fy=0.25)

    patch_match = PatchMatch(img1, img2, kernel_size=3)
    t, matches = patch_match.compute_matches(iters=1)
    result = reconstruct(matches, img1, img2)
    cv2.imshow("res", result)
    cv2.waitKey(-1)

    cv2.imwrite("results/result.jpg", result * 255.)

    with open("results/time.txt", "w") as text_file:
        text_file.write("Top to bottom and back in :{}".format(t))
