# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import sklearn
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image
from sklearn.cluster import KMeans

im = Image.open("hawkeye.jpg")
im.show()
px = im.load()
print (px)
height,width = im.size
print(width)
print(height)
Y = []
for i in range(height):
    for j in range(width):
        Y.append([i,j,px[i,j][0],px[i,j][1],px[i,j][2]])
Y_norm= sklearn.preprocessing.normalize(Y,norm='l1')
for k in range(2,6):
    kmeans = KMeans(n_clusters=k, init='k-means++',max_iter=300, tol=0.0001)
    kmeans = kmeans.fit(Y_norm)
    labels = kmeans.predict(Y_norm)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    pix = im.load()
    for i in range(height):
        for j in range(width):
            pix[i,j]=colors[labels[width*i+j]]
    im.show()
        
        


        
        
# =============================================================================
# for i in range(height):
#     for j in range(width):
#         Y.append([i,j,px[i,j]])
# =============================================================================
        

        
        

        
   


        

