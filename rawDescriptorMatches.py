import cv2
from PIL import Image
import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
import pylab as pl 
import pdb
#load the image from .mat file
fname='twoFrameData.mat'
mat = scipy.io.loadmat(fname)
imname=mat['im1']
mat1=mat['descriptors1']
print(mat1.shape)
im=Image.fromarray(imname,'RGB')
print(type(imname))
#to get the required region from image
print ('use the mouse to draw a polygon, right click to end it')
pl.imshow(im)
MyROI = roipoly(roicolor='r')
#getting descriptors
Ind = MyROI.getIdx(im, mat['positions1'])
fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im)
coners = displaySIFTPatches(mat['positions1'][Ind,:], mat['scales1'][Ind,:], mat['orients1'][Ind,:])
#showing descriptors
for j in range(len(coners)):
        bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
        bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
s=im.size
bx.set_xlim(0, s[0])
bx.set_ylim(0, s[1])
plt.gca().invert_yaxis()
plt.show() 

#accesing second image data
imname2=mat['im2']
mat2=mat['descriptors2']
im2=Image.fromarray(imname2,'RGB')
#using the Euclidean distance
matches = []
for i in Ind:
    desc1 = mat1[i]
    distances = np.sqrt(np.sum((mat2 - desc1) ** 2, axis=1))
    min_idx = np.argmin(distances)
    if distances[min_idx] < 0.6:  # Set a threshold for matching
        matches.append(min_idx)
#now showing the matched data
fig1=plt.figure()
bx1=fig1.add_subplot(111)
bx1.imshow(im2)
print(type(matches))
coners1 = displaySIFTPatches(mat['positions2'][matches,:], mat['scales2'][matches,:], mat['orients2'][matches,:])

for j in range(len(coners1)):
        bx1.plot([coners1[j][0][1], coners1[j][1][1]], [coners1[j][0][0], coners1[j][1][0]], color='g', linestyle='-', linewidth=1)
        bx1.plot([coners1[j][1][1], coners1[j][2][1]], [coners1[j][1][0], coners1[j][2][0]], color='g', linestyle='-', linewidth=1)
        bx1.plot([coners1[j][2][1], coners1[j][3][1]], [coners1[j][2][0], coners1[j][3][0]], color='g', linestyle='-', linewidth=1)
        bx1.plot([coners1[j][3][1], coners1[j][0][1]], [coners1[j][3][0], coners1[j][0][0]], color='g', linestyle='-', linewidth=1)
s2=im2.size
bx1.set_xlim(0, s2[0])
bx1.set_ylim(0, s2[1])
plt.gca().invert_yaxis()
plt.show()    
    