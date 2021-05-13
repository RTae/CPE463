import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def readImage(filename):
    cap = cv2.VideoCapture(filename)
    _, img = cap.read()
    cap.release()

    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    return img

def findDescriptor(img, N=3):

    contour_array = []
    # Edge detection by canny
    img = cv2.Canny(img, 100,200)

    # Get point of edge and sampling 1/N image
    indices = np.where(img != [0])
    indices = zip(indices[0], indices[1])
    c=0
    for indice in indices:
        if c % N == 0:
            contour_array.append([indice[0],indice[1]])
        c+=1
    contour_array = np.array(contour_array)

    # Convert in form of Fourier descriptors
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]

    # Fourier transform
    fourier_result = np.fft.fft(contour_complex)

    # Plot
    samping_rate = contour_array.shape[0]
    abs_fourier_transform = np.abs(fourier_result)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, samping_rate/2, len(power_spectrum))

    return frequency,power_spectrum

filename_A = './picture/A.gif'
filename_C = './picture/C.gif'
img_A = readImage(filename_A)
img_C = readImage(filename_C)
  
# resize image and rotate
scale_percent = 40 
width = int(img_A.shape[1] * scale_percent / 100)
height = int(img_A.shape[0] * scale_percent / 100)
dim = (width, height)
resize = cv2.resize(img_A, dim, interpolation = cv2.INTER_AREA)
h_o, w_o = img_A.shape
h_r, w_r = resize.shape
h_e = int((h_o-h_r)/2)
w_e = int((w_o-w_r)/2)
resize = cv2.copyMakeBorder(resize, h_e, h_e, w_e, w_e, cv2.BORDER_CONSTANT, value=(255,255,255))
resize = Image.fromarray(resize)
rotated = resize.rotate(320, fillcolor='white')
rotated = np.asarray(rotated)

# FDS
frequency_A, power_spectrum_A = findDescriptor(img_A, N=3)
frequency_B, power_spectrum_B = findDescriptor(rotated, N=3)
frequency_C, power_spectrum_C = findDescriptor(img_C, N=3)

fig, axes = plt.subplots(2, 1, figsize=(10, 10))

axes[0].plot(frequency_A, power_spectrum_A, label = "Origin A")
axes[0].plot(frequency_B, power_spectrum_B, label = "Modify A")
axes[0].set_title("Compare A and B")
axes[0].legend()

axes[1].plot(frequency_A, power_spectrum_A, label = "Origin A")
axes[1].plot(frequency_C, power_spectrum_C, label = "Origin C")
axes[1].set_title("Compare A and C")
axes[1].legend()

plt.show()
