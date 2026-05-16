import cv2
import numpy as np
import matplotlib.pyplot as plt

def perspectiveTransform():
    img = cv2.imread('pictures/test1.png')
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = imgRGB.shape

    p1 = np.array([
    [int(w*0.3), int(h*0.4)],  # top-left
    [int(w*0.7), int(h*0.4)],  # top-right
    [w, h], # bottom-right
    [0, h]  # bottom-left
], dtype=np.float32) #order is important

    p2 = np.array([
        [0, 0],
        [400, 0],
        [400, 600],
        [0, 600]
    ], dtype=np.float32)

    T = cv2.getPerspectiveTransform(p1, p2) # Transformation Matrix
    imgTrans = cv2.warpPerspective(imgRGB, T, (400, 600))

    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title("Original")

    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.title("Birds Eye View")

    plt.savefig("results/PerspectiveTransformResult.png")
    plt.show()

if __name__ == "__main__":
    perspectiveTransform()