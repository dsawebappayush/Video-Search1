import cv2
from PIL import Image
import pickle as pkl #using pickle to save the model of Kmeans
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
#taking all the mat file

framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]

#function to get all descriptors from si number of files

def add_all_mat(si):
 all_desc=np.empty((1,128))
 for i in range(si):
  fname = siftdir + fnames[i]
  mat = scipy.io.loadmat(fname)#getting mat of each file
  if mat['descriptors'].shape!=(0,128):
   all_desc=np.concatenate((mat['descriptors'],all_desc),axis=0)#adding all the descriptors
 return all_desc

#to create the model and create K centre

def create_model(K,all_desc):
 K=1000#value of number of clusters
 Kmeans=KMeans(n_clusters=K)
 Kmeans.fit(all_desc)#fitting the data
 pkl.dump(Kmeans,open('Kmeans_model.pkl','wb'))#saving my Kmeans model
 Kmeans=pkl.load('Kmeans_model.pkl')
 v_w=Kmeans.cluster_centers_
 a_v_w=Kmeans.predict(all_desc)
 return a_v_w,Kmeans

#to load model one's created in system
def load_model(all_desc):
     Kmeans=pkl.load(open('Kmeans_model.pkl','rb'))
     v_w=Kmeans.cluster_centers_
     a_v_w=Kmeans.predict(all_desc)
     return a_v_w,Kmeans

    

#to show a sift value in image with centre_index as index and num as nun of image
def show_image(centre_index,num):
    for i in range(200,num+200):
     fname = siftdir + fnames[i]
     mat = scipy.io.loadmat(fname)#getting mat of each file
     if mat['descriptors'].shape!=(0,128):
      val=Kmeans.predict(mat['descriptors'])
      matches=[]
      s=val.size
      for j in range(s):
         if val[j]==centre_index:
               matches.append(j)
      fig1=plt.figure()
      bx1=fig1.add_subplot(111)
      imname = framesdir + fnames[i][:-4]
      im2 = v=cv2.imread(imname)
      bx1.imshow(im2)   
      print(type(matches))
      coners1 = displaySIFTPatches(mat['positions'][matches,:], mat['scales'][matches,:], mat['orients'][matches,:])

      for j in range(len(coners1)):
         bx1.plot([coners1[j][0][1], coners1[j][1][1]], [coners1[j][0][0], coners1[j][1][0]], color='g', linestyle='-', linewidth=1)
         bx1.plot([coners1[j][1][1], coners1[j][2][1]], [coners1[j][1][0], coners1[j][2][0]], color='g', linestyle='-', linewidth=1)
         bx1.plot([coners1[j][2][1], coners1[j][3][1]], [coners1[j][2][0], coners1[j][3][0]], color='g', linestyle='-', linewidth=1)
         bx1.plot([coners1[j][3][1], coners1[j][0][1]], [coners1[j][3][0], coners1[j][0][0]], color='g', linestyle='-', linewidth=1)
      bx1.set_xlim(0, im2.shape[1])
      bx1.set_ylim(0, im2.shape[0])
      plt.gca().invert_yaxis()
      plt.show()    
   
def look_up(num):
    hist=np.zeros(1000)
    for i in range(200,num+200):
  
         fname = siftdir + fnames[i]
         mat = scipy.io.loadmat(fname)#getting mat of each file
         if mat['descriptors'].shape!=(0,128):
                  val=Kmeans.predict(mat['descriptors'])
                  for j in val:
                     hist[j]+=1
    for p in range(0,1000):
       if hist[p]>100:
          print(p)
          show_image(p,15)
   
    

all_desc=add_all_mat(700)
#a_v_w,Kmeans=create_model(1000,all_desc)
a_v_w,Kmeans=load_model(all_desc)
look_up(20)
#show_image(140,20)
#show_image(33,20)
