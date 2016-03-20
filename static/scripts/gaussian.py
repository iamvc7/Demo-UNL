#!/usr/bin/python
import cv2
import numpy as np
import sys

img = cv2.imread(sys.argv[1])
blur = cv2.GaussianBlur(img,(9,9),0)
cv2.imwrite(sys.argv[1],blur)

# cv2.imshow('Blur',blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()        # Closes displayed windows
