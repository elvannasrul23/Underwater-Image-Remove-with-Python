import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image 
img = cv2.imread("D:\Elvan\Kuliah\Semester 4\Pengolahan Citra Digital\Tugas\penyu.jpeg")

# Extract ROI from original image
roi = img[1:500, 120:590]

# Rotate ROI by 90 degrees 
angle = 90
rows, cols, _ = roi.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
roi_rotated = cv2.warpAffine(roi, M, (cols, rows))

# Calculate average color of ROI
mean = cv2.mean(roi_rotated)[0:3]

# Create average color image
average = np.full_like(roi_rotated, mean, dtype=np.uint8)

# Create opposite (inverted) average color image
opposite = 255 - average

# Convert both images to HSV
roi_hsv = cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2HSV)
opposite_hsv = cv2.cvtColor(opposite, cv2.COLOR_BGR2HSV)

# Copy the opposite hue and saturation channels to roi_hsv
roi_hsv[..., 0:2] = opposite_hsv[..., 0:2]

# Convert roi_hsv back to BGR
img2 = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)

# Blend 50-50 with original ROI
blend = cv2.addWeighted(img2, 0.5, roi_rotated, 0.5, 0)

# Apply sharpening filter
sharp_kernel = np.array([[0,-1, 0],
                              [-1, 5,-1],
                              [0,-1,0]])

sharpened = cv2.filter2D(blend, -1, sharp_kernel)

# Original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.xlabel('source : instagram/oasisatgracebay', fontsize=7)

# ROI
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(roi_rotated, cv2.COLOR_BGR2RGB))
plt.title('ROI')
plt.xlabel('90 Rotation', fontsize=7)

# Blend
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
plt.title('Enhancement')
plt.xlabel('Blending', fontsize=7)

# Sharpened
plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
plt.title('Filtering')
plt.xlabel('Sharpening', fontsize=7)

plt.show()
