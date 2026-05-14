import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistogramEqual():

    img = cv2.imread("pictures/tiger.png")
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([imggray], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max())/ cdf.max()

    plt.subplot(231)
    plt.imshow(imggray, cmap='gray')
    plt.title("Original Grayscale Image")


    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, 'b')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")
    

    equiimg = cv2.equalizeHist(imggray)
    equihist = cv2.calcHist([equiimg], [0], None, [256], [0, 256])
    equicdf = equihist.cumsum()
    equicdfNorm = equicdf * float(equihist.max())/ equicdf.max()

    plt.subplot(232)
    plt.imshow(equiimg, cmap='gray')
    plt.title("Histogram Equalized Grayscale Image")

    plt.subplot(235)
    plt.plot(equihist)
    plt.plot(equicdfNorm, 'b')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")
    

    claheObj = cv2.createCLAHE(clipLimit=5, tileGridSize=(8, 8))
    claheimg = claheObj.apply(imggray)
    clahehist = cv2.calcHist([claheimg], [0], None, [256], [0, 256])
    clahecdf = clahehist.cumsum()
    clahecdfNorm = equicdf * float(clahehist.max())/ clahecdf.max()

    plt.subplot(233)
    plt.imshow(claheimg, cmap='gray')
    plt.title("CLAHE Grayscale Image")
    
    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm, 'b')
    plt.xlabel("Pixel Intensity")
    plt.ylabel("# of Pixels")
    

    plt.show()

    cv2.imwrite('results/HistrogramEqualisedResult.png', equiimg)
    cv2.imwrite('results/CLAHEResult.png', claheimg)

if __name__ == '__main__':
    HistogramEqual()