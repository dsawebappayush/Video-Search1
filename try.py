
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
data = scipy.io.loadmat('twoFrameData.mat')

# Extract the necessary information
im1 = data['im1']
im2 = data['im2']
sift1 = data['descriptors1']
sift2 = data['descriptors2']
positions1 = sift1[:, :2]
descriptors1 = sift1[:, 4:]
positions2 = sift2[:, :2]
descriptors2 = sift2[:, 4:]

# Display the first image with SIFT features
fig, ax = plt.subplots()
ax.imshow(im1)
displaySIFTPatches(positions1, ax=ax)

# Allow the user to select a region of interest in the first image
print('Use the mouse to draw a polygon in the first image. Right-click to end it.')
plt.show(block=False)
roi = roipoly(roicolor='r')
indices = roi.getIdx(im1, positions1)