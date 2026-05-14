import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

def roi_mask(img):
    h, w = img.shape
    
    pts = np.array([[
        (0, h), #bottom left
        (w, h), #bottom right
        (int(w*0.6), int(h*0.4)), #top right
        (int(w*0.4), int(h*0.4)) #top left
    ]], dtype=np.int32)

    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(img, mask)


def HoughTransform():
    imgcolor = cv2.imread('pictures/roadline.png')
    imgcolor = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgcolor, cv2.COLOR_RGB2GRAY)
    imgblur = cv2.GaussianBlur(img, (11, 11), 0)
    imgcanny = cv2.Canny(imgblur, 100, 150)
    imgroi = roi_mask(imgcanny)
    
    plt.subplot(151)
    plt.imshow(img, cmap='gray')

    plt.subplot(152)
    plt.imshow(imgblur, cmap='gray')

    plt.subplot(153)
    plt.imshow(imgcanny, cmap='gray')

    plt.subplot(154)
    plt.imshow(imgroi, cmap='gray')

    distResol = 1
    angleResol = np.pi/180

    lines = cv2.HoughLines(imgroi, distResol, angleResol, 125)

    k = 3000

    if lines is not None:

        for curLine in lines:
            rho, theta = curLine[0]
            dhat = np.array([[np.cos(theta)], [np.sin(theta)]])
            d = rho*dhat
            lhat = np.array([[-np.sin(theta)], [np.cos(theta)]])

            p1 = d + k*lhat
            p2 = d - k*lhat
            p1 = p1.astype(int)
            p2 = p2.astype(int)

            cv2.line(imgcolor, (p1[0][0], p1[1][0]), (p2[0][0], p2[1][0]), (0, 255, 0), 10)
    
    else:
        print("Lower threshold")
    
    plt.subplot(155)
    plt.imshow(imgcolor)

    plt.show()
    cv2.imwrite("results/HoughTransformResult.png", cv2.cvtColor(imgcolor, cv2.COLOR_RGB2BGR))
        

if __name__ == '__main__':
    HoughTransform()