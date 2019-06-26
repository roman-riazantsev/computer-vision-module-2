from task2.image_processor import *


def save_result(images, i, cnts):
    if len(images[0].shape) == 2:
        images[0] = cv2.cvtColor(images[0], cv2.COLOR_GRAY2RGB)
    if len(images[1].shape) == 2:
        images[1] = cv2.cvtColor(images[1], cv2.COLOR_GRAY2RGB)

    h, w = images[1].shape[:2]
    images[0] = cv2.resize(images[0], (int(w), int(h)))

    result = np.concatenate(images, axis=1)
    cv2.imwrite("results/result{}.jpg".format(i), result)

    with open("results/result{}.txt".format(i), "w") as text_file:
        text_file.write("# of objects :{}".format(cnts))


def main():
    for i in [7]:
        path = "images/count{}.jpg".format(i)
        img1 = loading_functions[i - 1](path)
        img = img1.copy()
        result, cnts = extraction_functions[i - 1](img1)
        save_result([img, result], i, cnts)


if __name__ == "__main__":
    main()
