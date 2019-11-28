#!/usr/bin/env python

import sys
import pickle
import algorithms
import visualizer
import nn_models

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

def autoEncode(data, reduced_dim=28, batch_size=100, epochs=20, filepath="nn_models/best_AutoEncoder.ae"):
    X_train, X_test = data
    ae = nn_models.AutoEncoder()
    ae.build_model(X_train, tuple(np.asarray(X_train).shape[1:]), reduced_dim)
    ae.train(X_train, X_train, X_train, X_train, batch_size, epochs, filepath)
    X_train_enc = ae.encode(X_train)
    X_test_enc = ae.encode(X_test)
    print(X_train)
    print(X_train_enc)
    return (X_train_enc, X_test_enc)
