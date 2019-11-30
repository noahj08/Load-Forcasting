#!/usr/bin/env python

import sys
import pickle
import algorithms
import visualizer
import numpy as np

from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA

def doICA(data):
    ica = FastICA(n_components=12)
    ica_data = ica.fit_transform(data)
    return ica_data

def doPCA(data):
    pca = PCA(n_components=12)
    pca_data = pca.fit_transform(data)
    return pca_data

def doLLE(data):
    lle = LLE(n_components=12)
    lle_data = lle.fit_transform(data)
    return lle_data

def doTSNE(data):
    tsne_data = TSNE(n_components=3).fit_transform(data)
    return tsne_data

def doTog(data):
    ret = np.zeros((data.shape[0], data.shape[1]+2))
    for i in range(len(data)):  
        var = data[i]
        var = np.append(data[i], data[i][6]*data[i][7]) #DA Energy, Congestion coefficients
        var2 = np.append(var, data[i][10]*data[i][11]) #RT Energy, Congestion coefficients
        ret[i] = var2
    return ret

def doSep(data):
    ret = np.zeros((data.shape[0], data.shape[1]+4))
    for i in range(len(data)):  
        var = data[i]
        var = np.append(data[i], data[i][6]*data[i][10]) #DA Energy, RT Energy coefficients
        var2 = np.append(var, data[i][7]*data[i][11]) #DA Congestion, RT Congestion coefficients
        var3 = np.append(var2, data[i][8]*data[i][12]) #marginal loss component
        var4 = np.append(var3, data[i][5]*data[i][9]) #locational marginal price
        ret[i] = var4
    return ret
