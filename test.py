import cv2
import numpy as np

img=cv2.imread('img1.jpg')
dimensions = img.shape
blank_image = np.zeros((dimensions[0],dimensions[1],3), np.uint8)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 70, 255)
contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1) 
cv2.imshow('Image',blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()