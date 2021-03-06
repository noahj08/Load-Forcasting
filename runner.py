#!/usr/bin/env python
import sys
import pickle
import algorithms
import visualizer
import features

class Runner():

    def __init__(self, algorithm, featureExtractor, visualizer):
        self.algo = algorithm
        self.featureExt = featureExtractor
        self.visualize = visualizer

    def run(self):
        # Load data
        (X_train, Y_train) = pickle.load(open('train.pickle','rb'))
        (X_test, Y_test) = pickle.load(open("test.pickle", 'rb'))
        # Extract features
        if self.featureExt:
            if "tog" in self.featureExt.lower():
                (X_train, X_test) = [features.doTog(X) for X in (X_train, X_test)]
            if "sep" in self.featureExt.lower():
                (X_train, X_test) = [features.doSep(X) for X in (X_train, X_test)]
            if "pca" in self.featureExt.lower():
                (X_train, X_test) = [features.doPCA(X) for X in (X_train, X_test)]
            if "ica" in self.featureExt.lower():
                (X_train, X_test) = [features.doICA(X) for X in (X_train, X_test)]
            if "lle" in self.featureExt.lower():
                (X_train, X_test) = [features.doLLE(X) for X in (X_train, X_test)]
            if "tsne" in self.featureExt.lower():
                (X_train, X_test) = [features.doTSNE(X) for X in (X_train, X_test)]
        # Run algorithm
        output = self.algo(X_train,Y_train, X_test, Y_test)
        # Visualize
        if self.visualize:
           self.visualize(output, Y_test)
        #return output

if __name__ == '__main__':
    assert(len(sys.argv) <= 3)
    mode = sys.argv[1]
    feature = None
    if len(sys.argv) == 3:
        feature = sys.argv[2]

    linear_names = ["linear", "linear regression", "linear_regression, baseline"]
    nn_names = ["nn", "nn_regression", "nn regression", "neural net", "neural_net"]
    svr_names = ["svr", "svm", "support vector regression"]


    if mode.lower() in linear_names:
        runner = Runner(algorithms.linear_regression, feature, visualizer.visualize_linear_regression)
    elif mode.lower() in nn_names:
        runner = Runner(algorithms.NN_regression, feature, visualizer.visualize_NN_regression)
    elif mode.lower() in svr_names:
        runner = Runner(algorithms.svr, None, visualizer.visualize_svr)
    else:
        raise(f"IDK what the mode {mode} is. Pls change ur input parameter")
    runner.run()
