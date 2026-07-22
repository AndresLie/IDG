import cv2
import numpy as np

img = np.zeros((100, 100), dtype=np.uint8)
cv2.line(img, (40, 20), (60, 80), 255, 1)

coords = cv2.findNonZero(img)
rect = cv2.minAreaRect(coords)
width, height = rect[1]
angle = rect[2]
print(f"Original: width={width}, height={height}, angle={angle}")

if width < height:
    width, height = height, width
    angle += 90

print(f"Adjusted: width={width}, height={height}, angle={angle}")

radius = 10
kernel_size = radius * 2 + 1
kernel = np.zeros((kernel_size * 2, kernel_size * 2), dtype=np.uint8)
center = (kernel_size, kernel_size)
axes = (radius, max(1, radius // 3))
cv2.ellipse(kernel, center, axes, angle, 0, 360, 255, -1)

cv2.imwrite("test_kernel.png", kernel)
dilated = cv2.dilate(img, kernel)
cv2.imwrite("test_dilated.png", dilated)
