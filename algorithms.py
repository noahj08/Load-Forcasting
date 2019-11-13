from sklearn.linear_model import LinearRegression
import numpy as np
import nn_models

# NOTE: All algorithms must follow this function header to work in the runner
# def fun(X_train, Y_train, X_test,Y_test)
def linear_regression(X_train, Y_train, X_test, Y_test):
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)
    coefficients = lm.coef_
    return (predictions, coefficients)

def NN_regression(X_train, Y_train, X_test, Y_test):
    nn = nn_models.BasicE2ENN()
    nn.build_model(np.asarray(X_train).shape)
    nn.train(X_train, Y_train)
    predictions = nn.evaluate(X_test,Y_test)
    return predictions
