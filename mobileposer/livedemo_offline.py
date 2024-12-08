import time
import socket
import torch
from pygame.time import Clock

import articulate as art
import win32api
import os
from config import *
import keyboard
import datetime
from articulate.utils.noitom import *
from articulate.utils.unity import MotionViewer

from articulate.utils.wearable import WearableSensorSet
from auxiliary import calibrate_q, quaternion_inverse

from mobileposer.utils.model_utils import load_model
from mobileposer.models import MobilePoserNet
from mobileposer.data import PoseDataset
import sys

if __name__ == '__main__':
    os.makedirs(paths.temp_dir, exist_ok=True)
    os.makedirs(paths.live_record_dir, exist_ok=True)
    
    device = torch.device("cuda")
    
    clock = Clock()
    
    # set network
    ckpt_path = "data/checkpoints/mobileposer_finetuneddip/model_finetuned.pth"
    net = load_model(ckpt_path)
    print('Model loaded.')
    
    test_name = 'test20241127162314'
    test_dir = os.path.join(paths.live_record_dir, test_name)
    accs = torch.load(os.path.join(test_dir, 'accs.pt'))
    oris = torch.load(os.path.join(test_dir, 'oris.pt'))
    frame_num = len(accs)
    poses = torch.load(os.path.join(test_dir, 'poses.pt'))
    trans = torch.load(os.path.join(test_dir, 'trans.pt'))
    
    net.eval()
    with torch.no_grad(), MotionViewer(1, names=['Wearable Sensors']) as viewer:
        for i in range(frame_num):
            clock.tick(60)
            
            # aM = accs[i]
            # RMB = oris[i]
            
            # # select combo
            # combo = [0, 3, 4]
            
            # aM = aM[combo] / amass.acc_scale
            # RMB = RMB[combo]
            # # endregion
            
            # input = torch.cat([aM.flatten(), RMB.flatten()], dim=0).to("cuda")  
            
            # pose, _, tran, _ = net.forward_online(input)
            
            # poses.append(pose)
            # trans.append(tran)
            
            # convert tensor to numpy
            pose = poses[i]
            tran = trans[i]
            pose = pose.cpu().numpy()
            tran = tran.cpu().numpy()
            
            viewer.update_all([pose], [tran], render=False)
            viewer.render()
            
            print('\r', clock.get_fps(), end='')
            
            print(f'\rfps: {clock.get_fps():.2f}', end='')

    # oris = torch.stack(oris)
    # accs = torch.stack(accs)
    
    # # print frames num
    # print('Frames: %d' % accs.shape[0])
    
    # torch.save({'acc': accs, 'ori': oris}, os.path.join(paths.live_record_dir, 'test' + datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.pt'))
    # save poses and trans in the same directory
    # torch.save(poses, os.path.join(test_dir, 'poses.pt'))
    # torch.save(trans, os.path.join(test_dir, 'trans.pt'))
    
    print('\rFinish.')