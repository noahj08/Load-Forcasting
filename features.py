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
