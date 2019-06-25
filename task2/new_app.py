from task2.image_processor import *


def save_result(images, i):
    if len(images[0].shape) == 2:
        images[0] = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)
    if len(images[1].shape) == 2:
        images[1] = cv2.cvtColor(images[1], cv2.COLOR_GRAY2RGB)

    h, w = images[1].shape[:2]
    images[0] = cv2.resize(images[0], (int(w), int(h)))

    result = np.concatenate(images, axis=1)
    cv2.imwrite("results/result{}.jpg".format(i), result)


def main():
    n = 10
    for i in range(n, n + 1):
        path = "images/count{}.jpg".format(i)
        img1 = loading_functions[i - 1](path)
        img = img1.copy()
        result = extraction_functions[i - 1](img1)
        # img = cv2.resize(img, None, fx=0.5, fy=0.5)
        save_result([img, result], i)


if __name__ == "__main__":
    main()
