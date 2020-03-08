from cv2 import cv2
import numpy as np

img=cv2.imread('11.jpg')
dimensions = img.shape
blank_image = np.zeros((dimensions[0],dimensions[1],3), np.uint8)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
bin_img = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,5)

binary_img = cv2.bitwise_not(bin_img)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilated_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel,iterations=1)

contours, hierarchy = cv2.findContours(dilated_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(blank_image, contours, -1, (0, 255, 0), 1) 
cv2.imshow('Image',blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('img1122.jpg',blank_image)

# sq_cnts = []
# for cnt in contours:
#     sq_cnts.append(cnt)

# count=0
# lower_red = np.array([5,91,176])
# upper_red = np.array([131,183,235])
# for i in range(len(sq_cnts)):
#     (x, y, w, h) = cv2.boundingRect(sq_cnts[i])
#     newimg = img[y:y+h,x:x+w]
#     mask = cv2.inRange(newimg,lower_red, upper_red)
#     res = cv2.bitwise_and(newimg,newimg, mask= mask)
#     res[np.where(res<100)] = 0

#     if (res.any()!=0) and w*h>70000:
#         cv2.imshow('Image',newimg)
#         cv2.imwrite(str(count)+'.jpg',newimg)
#         cv2.waitKey(0)
#         count=count+1
# cv2.destroyAllWindows()