from sklearn.linear_model import LinearRegression
from sklearn import metrics
def linear_regression(X_train, Y_train, X_test, Y_test):
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)
    r2_score = metrics.r2_score(Y_test, predictions)
    mse_loss = metrics.mean_squared_error(Y_test, predictions)
    coefficients = lm.coef_
    return (predictions, r2_score, mse_loss, coefficients)

