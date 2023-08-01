import cv2
from PIL import Image
import numpy as np
import scipy.io
import pickle as pkl
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
def normalized_scalar_product(hist1, hist2):
    dot_product = np.dot(hist1, hist2)
    norm_hist1 = np.linalg.norm(hist1)
    norm_hist2 = np.linalg.norm(hist2)
    if norm_hist1 == 0 or norm_hist2 == 0:
        return 0
    else:
        return dot_product / (norm_hist1 * norm_hist2)
#function to get descriptor and Ind of sift found and of image in region
def get_region_sift(i):
    fname = siftdir + fnames[i]
    mat = scipy.io.loadmat(fname)
    imname = framesdir + fnames[i][:-4]
    im =cv2.imread(imname)
    print(imname)
    #to get the required region from image
    print ('use the mouse to draw a polygon, right click to end it')
    pl.imshow(im)
    MyROI = roipoly(roicolor='r')
    #getting descriptors
    Ind = MyROI.getIdx(im, mat['positions'])
    print(Ind)
    return Ind,mat


#give us the historgam from Ind found earlier

def give_hist(mat,Ind):
    val=Kmeans.predict(mat['descriptors'])
    all_centres=[]
    for i in Ind:
       if val[i] not in all_centres:  
        all_centres.append(val[i])
    hist=np.zeros(len(all_centres))
    for i in val:
        if i in all_centres:
         j=all_centres.index(i)
         hist[j]+=1
    print(hist)
    plt.show()
    return hist,all_centres
    

def give_all_descriptos(i):
    framesdir = 'frames/'
    siftdir = 'sift/'
    fnames = glob.glob(siftdir + '*.mat')
    fnames = [i[-27:] for i in fnames]
    fname = siftdir + fnames[i]
    mat = scipy.io.loadmat(fname)
    return mat

def print_image(i):
        imname = framesdir + fnames[i][:-4]
        im1 =cv2.imread(imname)
        print(imname)
        fig1=plt.figure()
        ax1=fig1.add_subplot(111)
        ax1.imshow(im1)
        plt.show()

framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]
Kmeans=pkl.load(open('Kmeans_model.pkl','rb'))#getting the model
j=460
Ind,mat1=get_region_sift(j)
dict1={}
hist1,found_centres=give_hist(mat1,Ind)
print(found_centres)
#searching all files to find nearest match
for i in range(0,1000):
  mat=give_all_descriptos(i)
  if mat['descriptors'].shape!=(0,128):    
   val=Kmeans.predict(mat['descriptors'])
   hist2=np.zeros(hist1.shape[0])
   for p in val:
       if p in found_centres:
           k=found_centres.index(p)
           hist2[k]+=1
   c=normalized_scalar_product(hist1,hist2)
  # print(hist2)
   dict1[1-c]=i
   #print(i)
  # print_image(i)
dict1=dict(sorted(dict1.items()))#sort the dictonary
ct=0
for i in dict1:
        #print(i)
        print(1-i)
        min_ind=dict1[i]
        ct=ct+1
        if ct==10: 
            break
       # print(min_ind)
        print_image(min_ind)

