from pygame.time import Clock
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.wearable import WearableSensorSet
import numpy as np

class KalmanFilter:
    def __init__(self, k, b):
        
        # 初始状态估计
        x0 = np.array([[0],     # 初始高度
                       [0]])    # 初始速度
        P0 = np.eye(2) * 500    # 初始状态协方差矩阵
        self.x = x0
        self.P = P0
        
        # 定义Kalman滤波器的参数
        self.dt = 1.0 / 60.0    # 采样间隔
        self.k = k              # pressure to height slope
        self.b = b              # pressure to height bias
        
        # 定义状态转移矩阵
        self.F = np.array([     # 状态转移矩阵
            [1, self.dt], 
            [0, 1]
        ])
        
        # 定义观测矩阵
        self.H = np.array([     # 观测矩阵
            [1 / self.k, 0]
        ])
        
        # 定义过程噪声协方差矩阵
        self.Q = np.array([     # 过程噪声协方差矩阵
            [1.0, 0],
            [0, 1.0]
        ])
        
        # 定义观测噪声协方差矩阵
        self.R = np.array([[0.015]])  # 观测噪声协方差矩阵
        
        self.I = np.eye(self.F.shape[0])  

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = (z + self.b / self.k) - self.H @ self.x 
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
    
    def get_height(self):
        return self.x[0, 0]
    
def test_wearable_pressure():
    clock = Clock()
    sviewer = StreamingDataViewer(2, y_range=(0, 10), window_length=500, names=['phone', 'watch']); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    k = - 800
    
    # # calculate sensor pressure bias
    # pressures = {"sensor0": [], "sensor1": []}
    # while True:
    #     data = sensor_set.get()
    #     if 0 in data.keys():
    #         pressures["sensor0"].append(data[0].pressure)
    #         pressures["sensor1"].append(data[1].pressure)
    #         if len(pressures["sensor0"]) > 100:
    #             p_bias = np.mean(pressures["sensor0"]) - np.mean(pressures["sensor1"])
    #             print('bias:', p_bias)
    #             break
    
    # kf = KalmanFilter(1, 0)
    # while True:
    #     data = sensor_set.get()
    #     if 0 in data.keys():
    #         pressure1 = data[0].pressure
    #         pressure2 = data[1].pressure + p_bias
    #         sviewer.plot([pressure1, pressure2])
    #         clock.tick(60)
    #         print('\r', clock.get_fps(), end='')
    
    b_window = 100
    bs = []
    
    # calculate height bias
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            bs.append(k * pressure)
            if len(bs) > b_window:
                b = - np.mean(bs)
                print('b:', b)
                break
    
    kf = KalmanFilter(k, b)
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            
            h_raw = k * pressure + b

            z = np.array([[pressure]])
            kf.predict()
            kf.update(z)
            h_filtered = kf.get_height()
            
            sviewer.plot([h_raw * 0.01, h_filtered * 0.01])
            clock.tick(60)
            print('\r', clock.get_fps(), end='')
            
if __name__ == '__main__':
    test_wearable_pressure()