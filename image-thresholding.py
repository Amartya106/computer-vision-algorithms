import cv2
import numpy as np
import matplotlib.pyplot as plt

def imageThres():
    img = cv2.imread('pictures/tiger.png')
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imgGray], [0], None, [256], [0, 256]) #to get value of threshold
    
    thresholds = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV, cv2.THRESH_TRUNC]
    thresnames = ["Binary", "BinaryInv", "ToZero", "ToZeroInv", "Trunc"]

    plt.subplot(231)
    plt.imshow(imgGray, cmap='gray')
    plt.title("Original Grayscale")

    for i in range(len(thresholds)):
        plt.subplot(2, 3, i+2)
        _, imageThres = cv2.threshold(imgGray, 112, 255, thresholds[i])
        plt.imshow(imageThres, cmap='gray')
        plt.title(thresnames[i])
    
    plt.savefig("results/TigerThreshResult.png")
    plt.show()
    


if __name__  ==  "__main__":
    imageThres()