#!/usr/bin/env python
import sys
import pickle
import algorithms
import visualizer

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
            (X_train, X_test) = [self.featureExt(X) for X in (X_train, X_test)]
        # Run algorithm
        output = self.algo(X_train,Y_train, X_test, Y_test)
        # Visualize
        if self.visualize:
           self.visualize(output, Y_test)

if __name__ == '__main__':
    assert(len(sys.argv) == 2)
    mode = sys.argv[1]
    linear_names = ["linear", "linear regression", "linear_regression, baseline"]
    nn_names = ["nn", "nn_regression", "nn regression", "neural net", "neural_net"]
    if mode.lower() in linear_names:
        runner = Runner(algorithms.linear_regression, None, visualizer.visualize_linear_regression)
    elif mode.lower() in nn_names:
        runner = Runner(algorithms.NN_regression, None, visualizer.visualize_NN_regression)
    else:
        raise(f"IDK what the mode {mode} is. Pls change ur input parameter")
    runner.run()
