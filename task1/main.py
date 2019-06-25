from task1.image_processor import *


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
    for i in range(11, 12):
        path = "images/text{}.jpg".format(i)
        img = loading_functions[i - 1](path)
        segmented_text = extraction_functions[i - 1](img)
        save_result([img, segmented_text], i)


if __name__ == "__main__":
    main()
