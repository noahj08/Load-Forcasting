from sklearn.linear_model import LinearRegression
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from abc import ABC

def linear_regression(X_train, Y_train, X_test, Y_test):
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    predictions = lm.predict(X_test)
    r2_score = metrics.r2_score(Y_test, predictions)
    mse_loss = metrics.mean_squared_error(Y_test, predictions)
    coefficients = lm.coef_
    return (predictions, r2_score, mse_loss, coefficients)


# Neural Networks base class
class NN(ABC):
    
    def __init__(self):
        self.model = Sequential()
        self.build_model()

    @abstractmethod
    def build_model(self,input_shape):
        pass

    # Trains the NN and saves model parameters to be loaded in regress
    def train(self,X_train, Y_train, X_test, Y_test):
        self.model.fit(X_train,Y_train, epochs=150, batch_size = 10)

    # Returns Y_pred
    def evaluate(self,X,Y):
        return model.evaluate(X,Y)

class BasicE2ENN(NN):

    def __init__(self):
        super().__init__()

    def build_model(self, input_shape):
       self.model.add(Dense(50, input_shape=input_shape))
       self.model.add(Activation('tanh'))
       self.model.add(Dense(40))
       self.model.add(Activation('tanh'))
       self.model.add(Dense(30))
       self.model.add(Activation('tanh'))
       self.model.add(Dense(20))
       self.model.add(Activation('tanh'))
       self.model.add(Dense(10))
       self.model.add(Activation('tanh'))
       self.model.add(Dense(1))

