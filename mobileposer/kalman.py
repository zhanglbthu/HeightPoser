from pygame.time import Clock
from articulate.utils.pygame import StreamingDataViewer
from articulate.utils.wearable import WearableSensorSet
from articulate.utils.bullet.view_rotation_np import RotationViewer
import torch
import numpy as np
import articulate as art
from articulate.utils.noitom import *
import time
from auxiliary import calibrate_q, quaternion_inverse

class IMUSet:
    g = 9.8

    def __init__(self, udp_port=7777):
        app = MCPApplication()
        settings = MCPSettings()
        settings.set_udp(udp_port)
        settings.set_calc_data()
        app.set_settings(settings)
        app.open()
        time.sleep(0.5)

        sensors = [None for _ in range(6)]
        evts = []
        print('Waiting for sensors...')
        while len(evts) == 0:
            evts = app.poll_next_event()
            for evt in evts:  
                assert evt.event_type == MCPEventType.SensorModulesUpdated
                sensor_module_handle = evt.event_data.sensor_module_data.sensor_module_handle
                sensor_module = MCPSensorModule(sensor_module_handle)
                sensors[sensor_module.get_id() - 1] = sensor_module

        print('find %d sensors' % len([_ for _ in sensors if _ is not None]))
        self.app = app
        self.sensors = sensors
        self.t = 0

    def get(self):
        evts = self.app.poll_next_event()
        if len(evts) > 0:
            self.t = evts[0].timestamp
        q, a = [], []
        for sensor in self.sensors:
            q.append(sensor.get_posture())
            a.append(sensor.get_accelerated_velocity())

        # assuming g is positive (= 9.8), we need to change left-handed system to right-handed by reversing axis x, y, z
        R = art.math.quaternion_to_rotation_matrix(torch.tensor(q))  # rotation is not changed
        a = -torch.tensor(a) / 1000 * self.g                         # acceleration is reversed
        a = R.bmm(a.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., self.g])   # calculate global free acceleration
        return self.t, R, a

    def clear(self):
        pass

class KalmanFilterBasic:
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
            [1e-1, 0],
            [0, 1e-1]
        ])
        
        # 定义观测噪声协方差矩阵
        self.R = np.array([[1e-4]])  # 观测噪声协方差矩阵
        
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
        
        # 定义控制输入矩阵
        self.B = np.array([     # 控制输入矩阵
            [0.5 * self.dt**2],
            [self.dt]
        ])
        
        # 定义观测矩阵
        self.H = np.array([     # 观测矩阵
            [1 / self.k, 0]
        ])
        
        # 定义过程噪声协方差矩阵
        self.Q = np.array([     # 过程噪声协方差矩阵
            [1e-1, 0],
            [0, 1e-1]
        ])
        
        # 定义观测噪声协方差矩阵
        self.R = np.array([[1e-4]])  # 观测噪声协方差矩阵
        
        self.I = np.eye(self.F.shape[0])  

    def predict(self, az):
        self.x = self.F @ self.x + self.B * az
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
    sviewer = StreamingDataViewer(3, y_range=(0, 10), window_length=500, names=['raw', 'basic', 'kalman filter']); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    k = - 800
    
    # # calculate sensor pressure bias
    # pressures = {"sensor0": [], "sensor1": []}
    # while True:
    #     data = sensor_set.get()
    #     if 0 in data.keys() and 1 in data.keys():
    #         pressures["sensor0"].append(data[0].pressure)
    #         pressures["sensor1"].append(data[1].pressure)
    #         if len(pressures["sensor0"]) > 100:
    #             p_bias = np.mean(pressures["sensor0"]) - np.mean(pressures["sensor1"])
    #             print('p_bias:', p_bias)
    #             break
    
    
    b_window = 200
    bs = []
    # calculate height bias
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            bs.append(k * pressure)
            if len(bs) > b_window:
                h_bias = - np.mean(bs)
                print('h_bias:', h_bias)
                break
    
    # kfs = [KalmanFilter(k, h_bias) for _ in range(2)]
    # while True:
    #     clock.tick(60)
    #     data = sensor_set.get()

    #     heights = [] 
        
        
    #     for i in range(2):
    #         pressure = data[i].pressure
    #         if i == 1:
    #             pressure += p_bias
                
    #         z = np.array([[pressure]])
    #         kfs[i].predict()
    #         kfs[i].update(z)
    #         heights.append(kfs[i].get_height())
            
    #     sviewer.plot(heights)
    
    kfb = KalmanFilterBasic(k, h_bias)
    kf = KalmanFilter(k, h_bias)
    while True:
        data = sensor_set.get()
        if 0 in data.keys():
            
            pressure = data[0].pressure
            aSS = torch.tensor(data[0].raw_acceleration).float()
            qIS = torch.tensor(data[0].orientation).float()
            RIS = art.math.quaternion_to_rotation_matrix(qIS)
            
            aIS = RIS.squeeze(0).mm( - aSS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., - 9.8])
            
            h_raw = k * pressure + h_bias

            z = np.array([[pressure]])
            
            az = aIS.numpy()[2]
            
            kf.predict(az)
            kfb.predict()
            kf.update(z)
            kfb.update(z)
            h_filtered = kf.get_height()
            h_filtered_b = kfb.get_height()
                        
            sviewer.plot([h_raw, h_filtered_b, h_filtered])
            clock.tick(60)
            print('\r', clock.get_fps(), end='')

def test_acceleration():
    clock = Clock()
    sviewer = StreamingDataViewer(3, y_range=(-10, 10), window_length=500, names=['x', 'y', 'z']); sviewer.connect()
    sensor_set = WearableSensorSet()
    
    imu_set = IMUSet()
    
    while True:
        clock.tick(60)
        data = sensor_set.get()
        if 0 in data.keys():
            aSS = torch.tensor(data[0].raw_acceleration).float()
            qIS = torch.tensor(data[0].orientation).float()
            RIS = art.math.quaternion_to_rotation_matrix(qIS)
            
            aIS = RIS.squeeze(0).mm( - aSS.unsqueeze(-1)).squeeze(-1) + torch.tensor([0., 0., - 9.8])
            
            sviewer.plot(aIS)
            
            print('\r', clock.get_fps(), end='')
            
        # tframe, RIS, aI = imu_set.get()

        # sviewer.plot(aI[0])

def test_wearable_noitom(n_calibration=1):   # use imu 0 and sensor 0
    clock = Clock() 
    rviewer = RotationViewer(2, order='wxyz'); rviewer.connect()
    sviewer = StreamingDataViewer(2, (-20, 20), 200, names=['noitom', 'sensor']); sviewer.connect()
    sensor_set = WearableSensorSet()
    imu_set = IMUSet()

    r"""calibration"""
    qIS, qCO = [], []
    print('Rotate the sensor & imu together.')
    while len(qIS) < n_calibration:
        imu_set.app.poll_next_event()
        qIS.append(torch.tensor(imu_set.sensors[0].get_posture()).float()) # noitom
        qCO.append(torch.tensor(sensor_set.get()[0].orientation).float()) # wearable sensor
        print('\rCalibrating... (%d/%d)' % (len(qIS), n_calibration), end='')
    qCI, qSO = calibrate_q(torch.stack(qIS), torch.stack(qCO))
    print('\tfinished\nqCI:', qCI, '\tqSO:', qSO)

    r"""comparison"""
    while True:
        clock.tick(60)
        imu_set.app.poll_next_event()
        qIS_noitom = torch.tensor(imu_set.sensors[0].get_posture()).float()
        qCO_sensor = torch.tensor(sensor_set.get()[0].orientation).float()
        aSS_noitom = torch.tensor(imu_set.sensors[0].get_accelerated_velocity()).float() / 1000 * 9.8
        aSS_sensor = torch.tensor(sensor_set.get()[0].raw_acceleration).float()
        
        qCO_noitom = art.math.quaternion_product(art.math.quaternion_product(qCI, qIS_noitom), qSO)
        # convert CO to IS
        qIC = quaternion_inverse(qCI)
        qOS = quaternion_inverse(qSO)
        
        qIS_sensor = art.math.quaternion_product(art.math.quaternion_product(qIC, qCO_sensor), qOS)
        
        # rviewer.update_all([qCO_noitom, qCO_sensor])
        rviewer.update_all([qIS_noitom, qIS_sensor])
        sviewer.plot([aSS_noitom[0], aSS_sensor[0]])
        print('\r', clock.get_fps(), end='')

def test_R():
    clock = Clock()
    sensor_set = WearableSensorSet()
    
    pressures = []
    
    while True:
        clock.tick(60)
        data = sensor_set.get()
        if 0 in data.keys():
            pressure = data[0].pressure
            pressures.append(pressure)
            if len(pressures) > 1000:
                break
    
    # 计算pressure的方差
    mean_val = sum(pressures) / len(pressures)
    print(mean_val)
    variance = sum((p - mean_val) ** 2 for p in pressures) / len(pressures)

    print("{:.20f}".format(variance))

if __name__ == '__main__':
    test_wearable_pressure()
    # test_acceleration()
    # test_R()