import cv2
import numpy as np

img = np.zeros((100, 100), dtype=np.uint8)
cv2.line(img, (40, 20), (60, 80), 255, 1)

coords = cv2.findNonZero(img)
rect = cv2.minAreaRect(coords)
width, height = rect[1]
angle = rect[2]

if width < height:
    width, height = height, width
    angle += 90

radius = 10
kernel_size = radius * 2 + 1
kernel = np.zeros((kernel_size * 2, kernel_size * 2), dtype=np.uint8)
center = (kernel_size, kernel_size)
axes = (radius, max(1, radius // 3))
cv2.ellipse(kernel, center, axes, angle, 0, 360, 255, -1)

kx, ky, kw, kh = cv2.boundingRect(kernel)
cropped_kernel = kernel[ky:ky+kh, kx:kx+kw]

dilated_crop = cv2.dilate(img, cropped_kernel)
dilated_full = cv2.dilate(img, kernel)

# Check centroids
M_crop = cv2.moments(dilated_crop)
M_full = cv2.moments(dilated_full)
M_orig = cv2.moments(img)

print(f"Orig centroid: {M_orig['m10']/M_orig['m00']}, {M_orig['m01']/M_orig['m00']}")
print(f"Crop centroid: {M_crop['m10']/M_crop['m00']}, {M_crop['m01']/M_crop['m00']}")
print(f"Full centroid: {M_full['m10']/M_full['m00']}, {M_full['m01']/M_full['m00']}")

