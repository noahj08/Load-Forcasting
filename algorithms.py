from sklearn.linear_model import LinearRegression
import nn_models

# NOTE: All algorithms must follow this function header to work in the runner
# def fun(X_train, Y_train, X_test,Y_test)
def linear_regression(X_train, Y_train, X_test, Y_test):
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)
    coefficients = lm.coef_
    return (predictions, coefficients)
