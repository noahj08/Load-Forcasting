#!/usr/bin/env python
import pickle
import algorithms
import visualizer

(X_train, Y_train) = pickle.load(open('train.pickle','rb'))
(X_test, Y_test) = pickle.load(open("test.pickle", 'rb'))
(predictions, r2_score, mse_loss) = algorithms.linear_regression(X_train,\
        Y_train, X_test, Y_test)
visualizer.scatter(Y_test, predictions, "Actual vs. Predicted Demand (kW)",\
        "Actual (kW)", "Predicted (kW)", "baseline.jpg")
print(f"R2 Score = {r2_score}")
print(f"Mean Squared Error Loss = {mse_loss}")
