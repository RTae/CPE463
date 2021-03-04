import cv2
import numpy as np
import matplotlib.pyplot as plt

img_table = cv2.imread("./images/circles.bmp")
img_table = cv2.cvtColor(img_table, cv2.COLOR_BGR2RGB)

# Define range for value to filter back and white out
lower_hue = np.array([0,0,40]) 
upper_hue = np.array([255,255,215])

# Convert image to HSV color space
img_hsv = cv2.cvtColor(img_table, cv2.COLOR_RGB2HSV)

# Filter the image
mask_hsv = cv2.inRange(img_hsv, lower_hue, upper_hue)

# Masking the image
masked_image = np.copy(img_table)
masked_image[mask_hsv==0] = [0,0,0]

# Convert mask image to gray scale image
masked_image = cv2.cvtColor(masked_image,cv2.COLOR_RGB2GRAY)

# Use color tresholding to create binary image
_, binary = cv2.threshold(masked_image, 40, 215, cv2.THRESH_BINARY)

# Find contours of image
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Remove not circle contours by size
temp_array = []
for idx_contour in range(len(contours)):
  temp_array.append(not (contours[idx_contour].shape[0] < 20))

# Draw contours on image
contours_image = np.copy(img_table)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)

# Show draw image
plt.imshow(contours_image, cmap="gray")
plt.title("Number of circle: "+str(sum(temp_array))) 
plt.axis('off')
plt.show()
