import cv2 
import numpy as np
rgb = "/home/testbed/Projects/bop_toolkit/CNC-picking/real_d415/000000/rgb/000000.png"
img_path = "/home/testbed/Projects/bop_toolkit/CNC-picking/real_d415/000000/depth/000000.png"
rgb = cv2.imread(rgb)
cv2.imshow("RGB", rgb)
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
print(img.shape)
print(img.dtype)
print(np.max(img))
print(np.min(img))
print(np.unique(img, return_counts=True))

img[img > 1000] = 0
img = img/1000 * 255
cv2.imshow("Depth", img)
cv2.waitKey(0)