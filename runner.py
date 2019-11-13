#!/usr/bin/env python
import pickle
import algorithms
import visualizer

class Runner():

    def __init__(self, algorithm, featureExtractor, visualizer):
        self.algo = algorithm
        self.featureExt = featureExtractor
        self.visualizer = visualizer

    def run(self):
        (X_train, Y_train) = pickle.load(open('train.pickle','rb'))
        (X_test, Y_test) = pickle.load(open("test.pickle", 'rb'))
        if self.featureExt:
            (X_train, X_test) = [self.featureExt(X) for X in (X_train, X_test)]
        output = self.algo(X_train,Y_train, X_test, Y_test)
        self.visualize(output, Y_test)

if __name__ == '__main__':
    runner = Runner(algorithms.linear_regression, None, visualizer.visualize_regression)
    runner.run()
