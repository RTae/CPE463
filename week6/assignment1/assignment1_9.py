import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img_T1 = cv2.imread("./images/T1.jpg")
img_T1 = cv2.cvtColor(img_T1, cv2.COLOR_BGR2RGB)

img_T2 = cv2.imread("./images/T2.bmp")
img_T2 = cv2.cvtColor(img_T2, cv2.COLOR_BGR2RGB)

# Define range for value to filter green and yello out
lower_hue = np.array([0,0,0]) 
upper_hue = np.array([10,255,255])

# Convert image to HSV color space
img_hsv_T1 = cv2.cvtColor(img_T1, cv2.COLOR_RGB2HSV)
img_hsv_T2 = cv2.cvtColor(img_T2, cv2.COLOR_RGB2HSV)

# Filter the image
mask_hsv_T1 = cv2.inRange(img_hsv_T1, lower_hue, upper_hue)
mask_hsv_T2 = cv2.inRange(img_hsv_T2, lower_hue, upper_hue)

# Masking the image
masked_image_T1 = np.copy(img_T1)
masked_image_T2 = np.copy(img_T2)
masked_image_T1[mask_hsv_T1==0] = [0,0,0]
masked_image_T2[mask_hsv_T2==0] = [0,0,0]

# Convert mask image to gray scale image
masked_image_T1 = cv2.cvtColor(masked_image_T1,cv2.COLOR_RGB2GRAY)
masked_image_T2 = cv2.cvtColor(masked_image_T2,cv2.COLOR_RGB2GRAY)

# Use color tresholding to create binary image
_, binary_T1 = cv2.threshold(masked_image_T1, 40, 215, cv2.THRESH_BINARY)
_, binary_T2 = cv2.threshold(masked_image_T2, 40, 215, cv2.THRESH_BINARY)

# Find contours of image
contours_T1, _ = cv2.findContours(binary_T1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_T2, _ = cv2.findContours(binary_T2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Remove not red tomato contours by size
temp_array_T1 = []
temp_array_T2 = []

for idx_contour in range(len(contours_T1)):
  temp_array_T1.append(not (contours_T1[idx_contour].shape[0] < 70))

for idx_contour in range(len(contours_T2)):
  temp_array_T2.append(not (contours_T2[idx_contour].shape[0] < 100))

contours_image_T1 = np.copy(img_T1)
contours_T1 = np.array(contours_T1)
temp_array_T1 = np.array(temp_array_T1)
contours_T1 = contours_T1[temp_array_T1]

contours_image_T2 = np.copy(img_T2)
contours_T2 = np.array(contours_T2)
temp_array_T2 = np.array(temp_array_T2)
contours_T2 = contours_T2[temp_array_T2]

# Draw contours on image
contours_image_T1 = cv2.drawContours(contours_image_T1, contours_T1, -1, (0,255,0), 3)
contours_image_T2 = cv2.drawContours(contours_image_T2, contours_T2, -1, (0,255,0), 3)

# Show draw image
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(contours_image_T1, cmap="gray")
axes[0].set_title("Number of red tomato: "+str(len(contours_T1)))
axes[0].axis('off')

axes[1].imshow(contours_image_T2, cmap="gray")
axes[1].set_title("Number of red tomato: "+str(len(contours_T2)))
axes[1].axis('off')

plt.show()