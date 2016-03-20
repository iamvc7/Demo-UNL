#!/usr/bin/python
import cv2
import sys

img = cv2.imread(sys.argv[1])
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(sys.argv[1],gray_image)



'''
from PIL import Image
import numpy as np

im = Image.open('img.jpeg')
gray = im.convert('L')

bw = np.asarray(gray).copy()
bw[bw <= 50] = 255

imfile = Image.fromarray(bw)
imfile.save("result.jpeg")


#fig = plt.figure("Superpixels - %d segments" % (numSegments))
#ax = fig.add_subplot(1, 1, 1)
#ax.imshow(mark_boundaries(image, segments,color=(0, 1, 0),mode = "Inner"))
#plt.axis("off")
#plt.show()

'''
