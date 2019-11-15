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

# Note that NN regression reshapes the input arrays to make them easier to batch
def NN_regression(X_train, Y_train, X_test, Y_test, batch_size=100, filepath="nn_models/best_basicE2E.nn"):
    nn = nn_models.BasicE2ENN()
    nn.build_model(X_train, Y_train, tuple(np.asarray(X_train).shape[1:]))
    nn.train(X_train, Y_train, X_test, Y_test, batch_size, filepath)
    predictions = nn.predict(X_test)
    loss = nn.evaluate(X_test,Y_test)
    return (predictions,loss)

def svr(X_train, Y_train, X_test, Y_test):
    clf = SVR(gamma='scale', C=1.0, epsilon=0.2,verbose=True)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    score = sklearn.metrics.mean_squared_error(Y_test, predictions)
    return (predictions, score)

####################################################################################
# HELPER FUNCTIONS
