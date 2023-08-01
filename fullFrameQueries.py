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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
#dot product
def get_hist(num):
    all_hist=[]
    for j in range(0,num):#looping to get dot from each mat file
        fname1 = siftdir + fnames[j]
        mat1 = scipy.io.loadmat(fname1)#getting mat of each file
        if mat1['descriptors'].shape!=(0,128):
         val1=Kmeans.predict(mat1['descriptors'])
         hist1=np.zeros(1000)
         if j%100==0:
          print(j)
         for p in val1:
            hist1[p]+=1
         all_hist.append(hist1)
        else:
            all_hist.append(np.zeros(1000))
    return all_hist
        
        
    
def normalized_scalar_product(hist1, hist2):
    dot_product = np.dot(hist1, hist2)
    norm_hist1 = np.linalg.norm(hist1)
    norm_hist2 = np.linalg.norm(hist2)
    if norm_hist1 == 0 or norm_hist2 == 0:
        return 0
    else:
        return dot_product / (norm_hist1 * norm_hist2)
#taking all the mat file
framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]
#looping
Kmeans=pkl.load(open('Kmeans_model.pkl','rb'))#getting the model
num=[180,200,110]
all_hist=get_hist(700)
for j in range(0,2):
    i=num[j]
    fname = siftdir + fnames[i]
    imname = framesdir + fnames[i][:-4]#getting the image
    print(imname)
    mat = scipy.io.loadmat(fname)#getting mat of each file
    val=Kmeans.predict(mat['descriptors'])
    hist=np.zeros(1000)
    for p in val:
        hist[p]+=1
    flag=0
    min_val=0
    min_ind=0
    dict1={}
    for l in range(0,700):
        if l!=num[j]:
         hist1=all_hist[l]
         cal=normalized_scalar_product(hist,hist1)
         dict1[1-cal]=l
    dict1=dict(sorted(dict1.items()))#sort the dictonary
    im =cv2.imread(imname)#showing image 1
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.imshow(im)
    plt.show()
    ct=0
    for i in dict1:
        print(i)
        min_ind=dict1[i]
        ct=ct+1
        if ct==6: 
            break
        print(min_ind)
        fname1 = siftdir + fnames[min_ind]
        imname1 = framesdir + fnames[min_ind][:-4]
        print(imname1)
        im1 =cv2.imread(imname1)#showing image 2
        fig1=plt.figure()
        ax1=fig1.add_subplot(111)
        ax1.imshow(im1)
        plt.show()
        
    
 