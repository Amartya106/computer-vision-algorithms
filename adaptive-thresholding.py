import cv2
import numpy as np
import matplotlib.pyplot as plt

def AdaptiveThres():
    img = cv2.imread('pictures/tiger.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    

    maxValue = 255
    blockSize = 7
    offsetC = 3

    imgMean = cv2.adaptiveThreshold(imgGray, maxValue, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, offsetC)
    imgGauss = cv2.adaptiveThreshold(imgGray, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, offsetC)

    plt.subplot(131)
    plt.imshow(imgGray, cmap='gray')
    plt.title("Original Grayscale")


    plt.subplot(132)
    plt.imshow(imgMean, cmap='gray')
    plt.title("Adaptive Mean")

    plt.subplot(133)
    plt.imshow(imgGauss, cmap='gray')
    plt.title("Adaptive Gaussian")

    plt.savefig('results/AdaptiveThreshResult.png')
    plt.show()


if __name__ == "__main__":
    AdaptiveThres()