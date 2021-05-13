import cv2
import numpy as np
import matplotlib.pyplot as plt

def harrisCorners(img, window_size, k):

    # Find x and y derivatives
    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    # Get height and width of image
    height = img.shape[0]
    width = img.shape[1]

    # Copy new image and covert to color domain to add point color corner
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    offset = window_size/2

    #Loop through image and find our corners
    for y in range(int(offset), int(height-offset)):
        for x in range(int(offset), int(width-offset)):
            # Moving window by offset + 1
            # | IxIx IxIy |
            # | IxIy IyIy |
            windowIxx = Ixx[int(y-offset):int(y+offset+1), int(x-offset):int(x+offset+1)]
            windowIxy = Ixy[int(y-offset):int(y+offset+1), int(x-offset):int(x+offset+1)]
            windowIyy = Iyy[int(y-offset):int(y+offset+1), int(x-offset):int(x+offset+1)]

            # Calculate sum on each window in gradient direction
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            # Find cornerness value
            r = det - k*(trace**2)

            #If corner response is over zero, color the point
            if r > 0:
                color_img.itemset((y, x, 0), 255)
                color_img.itemset((y, x, 1), 0)
                color_img.itemset((y, x, 2), 0)
    return color_img

window_size = 2
k = 0.04

img = cv2.imread("./chessboard.png", 0)
finalImg = harrisCorners(img, int(window_size), float(k))
plt.imshow(finalImg, cmap="gray")
plt.show()