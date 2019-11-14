# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:27:07 2019

@author: hp
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
    gmm = GaussianMixture(n_components=k,tol=0.001,max_iter=100,init_params='kmeans')
    gmm.fit(Y_norm)
    prediction_gmm = gmm.predict(Y_norm)
    probs = gmm.predict_proba(Y_norm)
    colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255)]
    pix = im.load()
    for i in range(height):
        for j in range(width):
            pix[i,j]=colors[prediction_gmm[width*i+j]]
    im.show()
# =============================================================================
# GaussianMixture.predict_proba(self,Y_norm)
# GaussianMixture.predict(self, X)
# =============================================================================
