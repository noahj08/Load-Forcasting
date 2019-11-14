from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from abc import ABC,abstractmethod


# Neural Networks base class
class NN(ABC):
    
    def __init__(self):
        self.model = Sequential()
        self.xscaler = MinMaxScaler()
        self.yscaler = MinMaxScaler()

    @abstractmethod
    def build_model(self,input_shape):
        pass

    # Trains the NN and saves model parameters to be loaded in regress
    def train(self, X_train, Y_train, X_test, Y_test, batch_size, filepath):
        #X_train = X_train.reshape(-1,1)
        #Y_train = Y_train.reshape(-1,1)
        #X_test = X_test.reshape(-1,1)
        #Y_test = Y_test.reshape(-1,1)
        if not self.model:
            raise "You must call build_model first!"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=10)
        callbacks_list = [checkpoint]
        self.model.fit(X_train,Y_train, epochs=200, batch_size=batch_size,callbacks=callbacks_list, validation_data=(X_test, Y_test))
        #self.model.fit(self.xscaler.transform(X_train),self.yscaler.transform(Y_train), epochs=30, batch_size=batch_size,callbacks=callbacks_list, validation_data=(self.xscaler.transform(X_test), self.yscaler.transform(Y_test)))

    def predict(self, X):
        if not self.model:
            raise "You must call build_model first!"
        return self.model.predict(X)
        #return self.model.evaluate(self.xscaler.transform(X),self.yscaler.transform(Y))

    # Returns loss
    def evaluate(self, X, Y):
        if not self.model:
            raise "You must call build_model first!"
        return self.model.evaluate(X,Y)
        #return self.model.evaluate(self.xscaler.transform(X),self.yscaler.transform(Y))

class BasicE2ENN(NN):

    def __init__(self):
        super().__init__()

    def build_model(self, X_train, Y_train, input_shape):
        X_train = X_train.reshape(-1,1)
        Y_train = Y_train.reshape(-1,1)
        self.xscaler.fit(X_train)
        self.yscaler.fit(Y_train)
        self.model.add(Dense(50, input_shape=input_shape))
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(40))
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(30))
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(20))
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(10))
        #self.model.add(Activation('tanh'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error',optimizer='adam')
