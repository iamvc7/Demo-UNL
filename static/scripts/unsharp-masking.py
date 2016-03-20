import cv2
import sys

image = cv2.imread(sys.argv[1])
gaussian_3 = cv2.GaussianBlur(image, (11,11), 10.0)
unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
cv2.imwrite(sys.argv[1],unsharp_image)