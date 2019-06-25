import time

import numpy as np


class PatchMatch:
    def __init__(self, img1, img2, kernel_size):
        self.kernel_size = kernel_size
        self.h, self.w = img1.shape[:2]
        self.pad_width = kernel_size // 2
        padding = (self.pad_width, self.pad_width), (self.pad_width, self.pad_width), (0, 0)
        self.img1 = np.pad(img1, padding, "mean")
        self.img2 = img2
        self.matches = self.get_random_matches()
        self.mses = self.get_mses(self.img1, self.img2)

    def get_random_matches(self):
        n_pixels = self.h * self.w

        index_grid = np.indices((self.h, self.w))

        max_h_idx = self.h - self.pad_width
        max_w_idx = self.w - self.pad_width

        crop_grid = index_grid[:, self.pad_width:max_h_idx, self.pad_width:max_w_idx]
        crop_grid = crop_grid.T.reshape(-1, 2)

        random_xs = np.random.choice(crop_grid.T[0], n_pixels).reshape(self.h, self.w, 1)
        random_ys = np.random.choice(crop_grid.T[0], n_pixels).reshape(self.h, self.w, 1)
        matches = np.concatenate((random_xs, random_ys), axis=2)

        return matches

    def get_mses(self, img1, img2):
        mses = np.ones([self.h, self.w])

        for row_idx in range(self.h):
            for col_idx in range(self.w):
                pos1 = np.array([row_idx, col_idx])
                pos2 = self.matches[row_idx, col_idx]

                patch_1, patch_2 = self.get_patches(pos1, pos2, img1, img2)
                mses[row_idx, col_idx] = self.mse(patch_1, patch_2)

        return mses

    def get_patches(self, pos_1, pos_2, img_1, img_2):
        def get_edges(pos):
            x, y = pos
            min_x = x
            max_x = x + self.kernel_size
            min_y = y
            max_y = y + self.kernel_size
            return min_x, max_x, min_y, max_y

        def get_edges_center(pos):
            x, y = pos
            min_x = x - self.pad_width
            max_x = x + self.pad_width + 1
            min_y = y - self.pad_width
            max_y = y + self.pad_width + 1
            return min_x, max_x, min_y, max_y

        min_x_1, max_x_1, min_y_1, max_y_1 = get_edges(pos_1)
        min_x_2, max_x_2, min_y_2, max_y_2 = get_edges_center(pos_2)

        patch_1 = img_1[min_x_1:max_x_1, min_y_1:max_y_1]
        patch_2 = img_2[min_x_2:max_x_2, min_y_2:max_y_2]

        return patch_1, patch_2

    def mse(self, patch_1, patch_2):
        mse = np.sum((patch_1 - patch_2) ** 2)
        mse /= self.kernel_size ** 2

        return mse

    def compute_matches(self, iters):
        for itr in range(iters):
            start = time.time()
            for i in range(self.h):
                for j in range(self.w):
                    pos = np.array([i, j])
                    self.match_patches(pos, True)
                    self.random_search(pos, self.img1, self.img2)
            for i in range(self.h - 1, -1, -1):
                for j in range(self.w - 1, -1, -1):
                    pos = np.array([i, j])
                    self.match_patches(pos, True)
                    self.random_search(pos, self.img1, self.img2)
            end = time.time()
            print("top to bottom and back in :{}".format(end - start))

        return self.matches

    def match_patches(self, pos, descent):
        h, w = np.array(self.img1.shape[:2]) - self.pad_width + 1
        x, y = pos

        if descent:
            positions = np.array([(x, x, max(x - 1, 0)), (y, max(y - 1, 0), y)])
        else:
            positions = np.array([(x, x, min(x + 1, h - 1)), (y, min(y + 1, w - 1), y)])

        distances = self.mses[positions[0], positions[1]]
        min_pos_idx = np.argmin(distances)
        x_new, y_new = pos2 = positions.T[min_pos_idx]
        self.matches[x, y] = self.matches[x_new, y_new]
        patch_1, patch_2 = self.get_patches(pos, self.matches[x, y], self.img1, self.img2)
        self.mses[x, y] = self.mse(patch_1, patch_2)

    def random_search(self, pos, img1, img2):
        x, y = pos
        h, w = img2.shape[:2]

        b_x, b_y = self.matches[x, y]

        for i in range(4, 10):
            search_h = h / 4 ** -i
            search_w = w / 4 ** -i

            search_min_r = max(b_x - search_h, self.pad_width)
            search_max_r = min(b_x + search_h, h - self.pad_width)
            random_b_x = np.random.randint(search_min_r, search_max_r)

            search_min_c = max(b_y - search_w, self.pad_width)
            search_max_c = min(b_y + search_w, w - self.pad_width)
            random_b_y = np.random.randint(search_min_c, search_max_c)

            new_pos = np.array([random_b_x, random_b_y])
            patch_1, patch_2 = self.get_patches(pos, new_pos, img1, img2)
            error = self.mse(patch_1, patch_2)
            if error < self.mses[x, y]:
                self.matches[x, y] = new_pos
                self.mses[x, y] = error
