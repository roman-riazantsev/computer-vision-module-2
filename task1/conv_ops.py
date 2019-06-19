import numpy as np


def conv2d(img, filter):
    img_h = img.shape[0]
    fil_h = filter.shape[0]

    img_w = img.shape[1]
    fil_w = filter.shape[1]

    out_dim_h = int(((img_h - fil_h + 2 * 0) / 1) + 1)
    out_dim_w = int(((img_w - fil_w + 2 * 0) / 1) + 1)

    result = np.empty([out_dim_h, out_dim_w])

    for row_i in range(out_dim_h):
        for col_i in range(out_dim_w):
            region = [row[col_i:col_i + fil_w] for row in img[row_i:row_i + fil_w]]
            product = np.multiply(region, filter)
            result[row_i][col_i] = np.sum(product)

    return result


def get_edges(image, treshhold):
    scharr_x = np.array([[-3, 0, +3],
                         [-10, 0, +10],
                         [-3, 0, +3]])
    scharr_y = scharr_x.T
    grad_x = conv2d(image, scharr_x)
    grad_y = conv2d(image, scharr_y)
    a = [grad_x, grad_y]
    grad = np.sqrt(np.sum(np.square(a), axis=0))

    g_max, g_min = grad.max(), grad.min()
    grad = (grad - g_min) / (g_max - g_min)

    grad[grad <= treshhold] = 0
    return grad


def gaussian_blur(img, kernel_shape=(3, 3), sigma=1.0, simple_blur=False):
    def get_gausian_filter(kernel_shape, sigma):
        m, n = [(ss - 1.) / 2. for ss in kernel_shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    if simple_blur:
        kernel = np.ones(kernel_shape)
        kernel = kernel / np.prod(kernel_shape)
    else:
        kernel = get_gausian_filter(kernel_shape, sigma)

    return conv2d(img, kernel)
