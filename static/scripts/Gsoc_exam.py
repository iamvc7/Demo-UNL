import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import arange,array,ones,linalg
from pylab import plot,show
import sys
from skimage import data
from skimage import feature
from skimage.feature import blob_dog, blob_log
from math import sqrt
from skimage.color import rgb2gray
from decimal import *

img = cv2.imread('img.jpeg')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([0,70,70])
upper_red = np.array([200,255,255])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img,img, mask= mask)

blur = cv2.GaussianBlur(res,(3,3),0)

cv2.imwrite("saturated_image.jpeg",blur)

### Blob Detection

image_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

blobs_dog = blob_dog(image_gray)
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

blobs_log = blob_log(image_gray)
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

res = feature.blob_dog(image_gray)

blobs_list = [blobs_dog, blobs_log]
colors = ['yellow', 'lime']
titles = ['Method_1', 'Method_2']
sequence = zip(blobs_list, colors, titles)

fig,axes = plt.subplots(1, 2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
axes = axes.ravel()
for blobs, color, title in sequence:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title(title)
    ax.imshow(img, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

plt.savefig("Segmented_Image.jpeg")

plt.close()

print "Plot Saved"

### Analysing Concentrations

print "Method 1 segmentation based on Difference of Gaussian seems better - Going for it to analyse Concentrations"

v1 = ['c1','c2','c3','c4','c5','c6']
v2 = [0,62.5,125,250,500,1000]

# res array denotes the centers of all the blobs obtained

v3 = [] # in the order of C6, QC2, C5, sample, C2, C4, C3, QC1, C1 based on the co-ordinates in res

r = int(1.414 * res[0,2]) # radius of each blob is 1.414 * standard deviation

'''
# Considering RED colour intensities of a circumscribed square over the circular blob obtained

centre = x,y
r = radius

c1 = x-r,y-r
c2 = x-r,y+r
c3 = x+r,y-r
c4 = x+r,y+r

All four co-ordinates of the Circumscribed Rectangle

'''

# for C1 as the blob is not detected

temp = []
a = res[0,0]
b = (res[4,1]+res[6,1])/2
c = res[0,2]
temp.append(a)
temp.append(b)
temp.append(c)
temp = np.array(temp)
res = np.vstack((res,temp))

# Calculating intensities of all of them

for i in range(0,9):
    s = 0
    count = 0
    mean_sum = 0
    x = res[i,0]
    y = res[i,1]
    for j in range(x-r,x+r):
        for k in range(y-r,y+r):
            s = s + img[j,k,0] + img[j,k,1] + img[j,k,2]
            count+=1
    mean_sum = s/(3*count)
    v3.append(mean_sum)

print "Intensity of C1 is ", v3[8]
print "Intensity of C2 is ", v3[4]
print "Intensity of C3 is ", v3[6]
print "Intensity of C4 is ", v3[5]
print "Intensity of C5 is ", v3[2]
print "Intensity of C6 is ", v3[0]


Y = []

Y.append(v3[8])
Y.append(v3[4])
Y.append(v3[6])
Y.append(v3[5])
Y.append(v3[2])
Y.append(v3[0])

plt.plot(v2,Y)
plt.ylabel('intensity')
plt.xlabel('ng/ML')

plt.savefig("Standard_Curve.jpeg")

### Linear Regression

X = array([v2,ones(6)])
w = linalg.lstsq(X.T,Y)[0]
q1 = [156,750]

print "Regression done"

print "Regression Coefficient and Noise respectively", w

print "Observed Intensity of QC1 is ", v3[7]
print "Observed Intensity of QC2 is ", v3[1]
print "Calculated intensities for QC1 and QC2", q1[0]*w[0]+w[1] ,"and", q1[1]*w[0]+w[1], "respectively"

print "Calculated Concentrations of QC1 and QC2", (v3[7]-w[1])/w[0],"and", (v3[1]-w[1])/w[0] , "ng/Ml respectively"
print "Observed Intensity of sample is ", v3[3]

p = ((v3[1]-w[1])/w[0]) - q1[1]
q = ((v3[7]-w[1])/w[0]) - q1[0]

ans1 = (Decimal(p)/Decimal(q1[1]))*100
#ans2 = (Decimal(q)/Decimal(q1[0]))*100

print "ERROR in Concentrations of QC2 is ", ans1, " Percentage"
#print "ERROR in Concentrations of QC1 is ", ans2, " Percentage"

print "Calculated Concentrations of Sample is", (v3[3]-w[1])/w[0], "ng/Ml"
