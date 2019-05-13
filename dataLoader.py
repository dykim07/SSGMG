import os
import numpy as np
from numba import jit
from sklearn.preprocessing import StandardScaler


@jit
def generateWindow(X:np.ndarray, window_size:int) -> np.ndarray:
    x_data = []
    n_data = X.shape[0]

    for idx in range(n_data - window_size):
        x_data.append(
                X[idx:idx+window_size].reshape(1, window_size, -1)
                )

    x_data = np.concatenate(x_data)
    return x_data

class dataLoader():
    def __init__(self):

        self.sensor_train = dict()
        self.motion_train = dict()
    
        self.motion_pre_train = dict()

        self.sensor_test = dict()
        self.motion_test = dict()

        self.window_size = 120
        self.tags = ['W' + str(i) for i in range(2, 7)]
        self.base_path = os.path.join(os.getcwd(), 'dataset')
        self.sensor_scaler = StandardScaler()

        self.loadDataSet()
        self.normalizeSensorData()

    def normalizeSensorData(self):
        base_data = [data for _, data in self.sensor_train.items()]
        base_data = np.concatenate(base_data)
        self.sensor_scaler.fit(base_data)

    def loadDataSet(self):
        # sensor
        file_name = "train_{}_S.csv"
        for tag in self.tags :
            self.sensor_train[tag] = np.loadtxt(os.path.join(self.base_path,file_name.format(tag)), delimiter=',')

        
        file_name = "test_{}_S.csv"
        for tag in self.tags :
            self.sensor_test[tag] = np.loadtxt(os.path.join(self.base_path,file_name.format(tag)), delimiter=',')

        # motion

        file_name = "train_{}_M.csv"
        for tag in self.tags :
           self.motion_train[tag] = np.loadtxt(os.path.join(self.base_path,file_name.format(tag)), delimiter=',')

        
        file_name = "test_{}_M.csv"
        for tag in self.tags :
            self.motion_test[tag] = np.loadtxt(os.path.join(self.base_path,file_name.format(tag)), delimiter=',')


        file_name = "pre_{}_M.csv"
        for tag in self.tags :
            self.motion_pre_train[tag] = np.loadtxt(os.path.join(self.base_path,file_name.format(tag)), delimiter=',')
        
    def getPretrainDataSet(self) -> np.ndarray:
        data = [data for _, data in self.motion_pre_train.items()]
        return np.concatenate(data)

    
    def getTrainDataSet(self) -> (np.ndarray, np.ndarray)  :
        
        sensor = []
        for _ , data in self.sensor_train.items():
            data = self.sensor_scaler.transform(data)
            data = generateWindow(data, self.window_size)
            sensor.append(data)

        sensor = np.concatenate(sensor)
        motion = np.concatenate([data for _, data in self.motion_train.items()])
         
        return sensor, motion

    def getTestDataSet(self) -> (np.ndarray, np.ndarray):
        sensor = []
        for _, data in self.sensor_test.items():
            data = self.sensor_scaler.transform(data)
            data = generateWindow(data, self.window_size)
            sensor.append(data)

        sensor = np.concatenate(sensor)
        motion = np.concatenate([data for _, data in self.motion_test.items()])

        return sensor, motion
