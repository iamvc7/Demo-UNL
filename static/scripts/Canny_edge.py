import cv2
import sys

image = cv2.imread(sys.argv[1])
image = cv2.Canny(image,100,200)
cv2.imwrite(sys.argv[1],image)