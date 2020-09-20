import cv2
import numpy as np

img=cv2.imread('traffic.jpg')

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#read a template to detect
temp=cv2.imread('stop.jpg',0) #read it in grayscale
#set the width and height of template
w, h=temp.shape[::-1]
#now match the image with template
res=cv2.matchTemplate(gray,temp, cv2.TM_CCOEFF_NORMED)

thresh= 0.99
loc=np.where(res>=thresh)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h),(0,255,255),2)
    cv2.putText(img, "Stop", (pt[0],pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),2)

cv2.imshow("org",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
