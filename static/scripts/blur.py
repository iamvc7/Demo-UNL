#!/usr/bin/python
import cv2
import numpy as np
import sys


img = cv2.imread(sys.argv[1])
blur = cv2.blur(img,(15,15))
cv2.imwrite(sys.argv[1],blur)

