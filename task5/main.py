import numpy as np
import cv2

p_1, p_2, p_3, p_4 = np.array([[241, 6], [726, 29], [6, 668], [627, 837]])
p_1_, p_2_, p_3_, p_4_ = np.array([[0, 0], [300, 0], [0, 400], [300, 400]]).astype('float32')

x_1 = np.array(p_1[0])
y_1 = np.array(p_1[1])
x_2 = np.array(p_2[0])
y_2 = np.array(p_2[1])
x_3 = np.array(p_3[0])
y_3 = np.array(p_3[1])
x_4 = np.array(p_4[0])
y_4 = np.array(p_4[1])

x_1_ = np.array(p_1_[0])
y_1_ = np.array(p_1_[1])
x_2_ = np.array(p_2_[0])
y_2_ = np.array(p_2_[1])
x_3_ = np.array(p_3_[0])
y_3_ = np.array(p_3_[1])
x_4_ = np.array(p_4_[0])
y_4_ = np.array(p_4_[1])

matrix = np.array([[x_1, y_1, 1, 0, 0, 0, -x_1_.dot(x_1), -x_1_.dot(y_1)],
                   [0, 0, 0, x_1, y_1, 1, -y_1_.dot(x_1), -y_1_.dot(y_1)],
                   [x_2, y_2, 1, 0, 0, 0, -x_2_.dot(x_2), -x_2_.dot(y_2)],
                   [0, 0, 0, x_2, y_2, 1, -y_2_.dot(x_2), -y_2_.dot(y_2)],
                   [x_3, y_3, 1, 0, 0, 0, -x_3_.dot(x_3), -x_3_.dot(y_3)],
                   [0, 0, 0, x_3, y_3, 1, -y_3_.dot(x_3), -y_3_.dot(y_3)],
                   [x_4, y_4, 1, 0, 0, 0, -x_4_.dot(x_4), -x_4_.dot(y_4)],
                   [0, 0, 0, x_4, y_4, 1, -y_4_.dot(x_4), -y_4_.dot(y_4)]])

p2 = p_1_, p_2_, p_3_, p_4_
p2 = np.array(p2).flatten()

result = np.linalg.solve(matrix, p2)
result = np.append(result, 1).reshape(3, 3)
print(result)

pts1 = np.float32([[241, 6], [726, 29], [6, 668], [627, 837]])
pts2 = np.float32([[0, 0], [300, 0], [0, 400], [300, 400]])
# implement this function
M = cv2.getPerspectiveTransform(pts1, pts2)

print(M)
