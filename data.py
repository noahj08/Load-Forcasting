import xlrd
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle

class DataGrabber():

    def __init__(self, data_location="../data/"):
        self.data_location = data_location
    
    # Year: Years since 2000 (1, for 2001, 10 for 2010, etc)
    # data_location: Location of all excell files
    def getFilename(self, year):
        if year < 10:
            yearstr = "0" + str(year)
        else:
            yearstr = str(year)
        if year != 17: # 2017 is in xlsx format for some reason
            return self.data_location + "20" + yearstr + "_smd_hourly.xls"
        return self.data_location + "20" + yearstr + "_smd_hourly.xlsx"

    # Date: mm/dd/yyyy
    def splitDate(self, date):
        split = date.split('/')
        month = int(split[0])
        day = int(split[1])
        year = int(split[2]) - 2000
        return (day, month, year)

    # Returns an array containing all the data in a particular spreadsheet
    def readFile(self, filename, sheetname):
        data = pd.read_excel(filename, sheetname)
        data = list(data.to_numpy())
        X = []
        Y = []
        for row in data:
            split_data = []
            for idx in range(len(row)):
                item = row[idx]
                if idx == 0:
                    dt = item
                    split_data.append(int(dt.dayofyear))
                    split_data.append(int(dt.dayofweek))
                    split_data.append(int(dt.year))
                elif idx == 3:
                    Y.append(int(item))
                else:
                    split_data.append(int(item))
            X.append(split_data)
        return (X, Y)

    # Combines data from all of the different spreadsheets
    def concatinateData(self, arrays):
        arrays = tuple(arrays)
        return list(np.concatenate(arrays, axis=0))

    def train_test_split(self,X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1)
        return ((X_train, Y_train), (X_test, Y_test))

    # Main function (The only function that a user would need to call)
    def getData(self,sheetname='ME'):
        X_data = []
        Y_data = []
        for i in range(3, 18):
            filename = self.getFilename(i)
            X,Y = self.readFile(filename, sheetname)
            X_data.append(X)
            Y_data.append(Y)
        X = self.concatinateData(X_data)
        Y = self.concatinateData(Y_data)
        ((X_train, Y_train), (X_test, Y_test)) = self.train_test_split(X,Y)
        return ((X_train, Y_train), (X_test, Y_test))

# Example usage
if __name__ == '__main__':
    dg = DataGrabber()
    ((X_train, Y_train), (X_test, Y_test)) = dg.getData()
    #print(X_train)
    #print(Y_train)
    #print(X_test)
    #print(Y_test)
    pickle.dump((X_train,Y_train), open("train.pickle", "wb"))
    pickle.dump((X_test,Y_test), open("test.pickle", "wb"))
