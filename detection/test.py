import cv2
import numpy as np

img = cv2.imread("test.jpg")
ori_shape = img.shape
ori_image = img
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
new_img = np.zeros(ori_shape)
arr = np.ndarray((480, 848))
arr = (ori_image[:, :, 0] + ori_image[:, :, 1] + ori_image[:, :, 2]) / 255
new_img[:, :, 0] = arr
new_img[:, :, 1] = arr
new_img[:, :, 2] = arr
print(new_img.shape)

while True:

    cv2.imshow("test", new_img)
    # print(shape)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
