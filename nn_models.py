from keras.models import Sequential
from keras.layers import Dense, Activation
from abc import ABC,abstractmethod


# Neural Networks base class
class NN(ABC):
    
    def __init__(self):
        self.model = Sequential()

    @abstractmethod
    def build_model(self,input_shape):
        pass

    # Trains the NN and saves model parameters to be loaded in regress
    def train(self, X_train, Y_train):
        if not self.model:
            raise "You must call build_model first!"
        self.model.fit(X_train,Y_train, epochs=150)

    # Returns Y_pred
    def evaluate(self, X, Y):
        if not self.model:
            raise "You must call build_model first!"
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
       self.model.compile(loss='mean_squared_error',optimizer='adam')
