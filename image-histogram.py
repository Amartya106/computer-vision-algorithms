import cv2
import numpy as np
import matplotlib.pyplot as plt

def GrayHistogram():
    img = cv2.imread('pictures/tiger.png')
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imggray], [0], None, [256], [0, 256]) #256 is not included in range 0 in inculded though

    plt.subplot(121)
    plt.imshow(imggray, cmap='gray')

    plt.subplot(122)
    plt.plot(hist)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")


    plt.show()

def RGBHistogram():
    img = cv2.imread('pictures/tiger.png')
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(121)
    plt.imshow(imgrgb)

    colors = ['r', 'g', 'b']
    for i in range(len(colors)):
        hist = cv2.calcHist([imgrgb], [i], None, [256], [0, 256])
        plt.subplot(122)
        plt.plot(hist, colors[i])
 

    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")

    plt.show()

def RGBRegionHistogram():
    img = cv2.imread('pictures/tiger.png')
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgrgb = imgrgb[300:500, 350:600]

    plt.subplot(121)
    plt.imshow(imgrgb)

    colors = ['r', 'g', 'b']
    for i in range(len(colors)):
        hist = cv2.calcHist([imgrgb], [i], None, [256], [0, 256])
        plt.subplot(122)
        plt.plot(hist, colors[i])
 

    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")

    plt.show()

if __name__ == "__main__":
    GrayHistogram()
    RGBHistogram()
    RGBRegionHistogram()