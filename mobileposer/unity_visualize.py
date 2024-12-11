from pygame.time import Clock
import articulate as art
from articulate.utils.unity import MotionViewer
from utils import *
from config import paths
import torch
import os
import open3d as o3d
import numpy as np

body_model = art.ParametricModel(paths.smpl_file)
vi_mask = torch.tensor([1961, 5424, 876, 4362, 411, 3021])

deta_dir = 'data/debug/amass'

def vis_height():
    pose_path = os.path.join(deta_dir, 'pose.pt')
    tran_path = os.path.join(deta_dir, 'tran.pt')
    height_path = os.path.join(deta_dir, 'height.pt')
    
    poses = torch.load(pose_path)
    trans = torch.load(tran_path)
    heights = torch.load(height_path)
    
    index = 1
    pose = poses[index]
    tran = trans[index]
    height = heights[index]
    frame_num = pose.shape[0]
    
    clock = Clock()
    
    with MotionViewer(1, overlap=True, names=".") as viewer:
        for i in range(frame_num):
            clock.tick(60)
            viewer.clear_line(render=False)
            viewer.clear_point(render=False)
            
            pose_matrix = art.math.axis_angle_to_rotation_matrix(pose[i]).unsqueeze(0)
            
            _, _, glb_verts = body_model.forward_kinematics(pose_matrix, tran=tran[i], calc_mesh=True)
            
            viewer.update_all([pose_matrix.squeeze(0)], [tran[i]], render=False)
            
            pos1 = glb_verts[0, vi_mask[0]]
            pos2 = glb_verts[0, vi_mask[3]]
            h1 = height[i][0]
            h2 = height[i][1]
            viewer.draw_point(pos1, color=[255, 0, 0], radius=0.05, render=False)
            viewer.draw_point(pos2, color=[255, 0, 0], radius=0.05, render=False)
            viewer.draw_line(pos1, [pos1[0], pos1[1]-h1, pos1[2]], color=[255, 0, 0], render=False)
            viewer.draw_line(pos2, [pos2[0], pos2[1]-h2, pos2[2]], color=[255, 0, 0], render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

def value2color(value):
    value = torch.clamp(value, 0, 1).cpu().numpy()
    color = np.array([1, value, value])
    return color

def vis_contact(viewer: MotionViewer, pose_list, tran_list, contact_list):
    for i in range(len(pose_list)):
        pose = torch.tensor(pose_list[i]).unsqueeze(0)
        tran = torch.tensor(tran_list[i]).unsqueeze(0)
        
        _, glb_joint = body_model.forward_kinematics(pose, tran=tran, calc_mesh=False)
        
        viewer.draw_point(glb_joint[0][10], color=value2color(contact_list[i][0]), radius=0.2, render=False)
        viewer.draw_point(glb_joint[0][11], color=value2color(contact_list[i][1]), radius=0.2, render=False)

def vis_rheight(viewer: MotionViewer, pose_list, tran_list, rheight_list, offsets=None):
    for i in range(len(pose_list)):
        pose = torch.tensor(pose_list[i]).unsqueeze(0)
        tran = torch.tensor(tran_list[i]).unsqueeze(0)
        
        _, _, glb_verts = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        
        pos1 = glb_verts[0, vi_mask[0]]
        pos2 = glb_verts[0, vi_mask[3]]
        
        if offsets:

            pos1 = pos1 + torch.tensor(offsets[i])
            pos2 = pos2 + torch.tensor(offsets[i])
        
        h = rheight_list[i].cpu()
        viewer.draw_point(pos1, color=[255, 0, 0], radius=0.05, render=False)
        viewer.draw_point(pos2, color=[255, 0, 0], radius=0.05, render=False)
        viewer.draw_line(pos1, [pos1[0], pos1[1]-h, pos1[2]], color=[255, 0, 0], width=0.02, render=False)

def visualize(pose, tran=None, rheight=None):
    '''
    pose: [pose_sub1, pose_sub2, ...]
    tran: [tran_sub1, tran_sub2, ...]
    '''
    clock = Clock()
    sub_num = len(pose)
    with MotionViewer(sub_num, overlap=True, names=["gt", "MobilePoser", "MobilePoser_editH"]) as viewer:
        for i in range(len(pose[0])):
            clock.tick(60)
            viewer.clear_line(render=False)
            viewer.clear_point(render=False)
            
            pose_list = [pose[sub_idx][i] for sub_idx in range(sub_num)]
            if tran:
                tran_list = [tran[sub_idx][i] for sub_idx in range(sub_num)]
            else:
                # generate zero tran on cpu
                tran_list = [torch.zeros(1, 3).cpu() for _ in range(sub_num)]

            if rheight:
                rheight_list = [rheight[sub_idx][i] for sub_idx in range(sub_num)]
                vis_rheight(viewer, pose_list, tran_list, rheight_list, offsets=viewer.offsets)
            
            viewer.update_all(pose_list, tran_list, render=False)
            
            viewer.render()
            
            print('\r', clock.get_fps(), end='')

def process_mobileposer_data(data, relative_height = False):
    pose_t = data['pose_t']
    pose_p = data['pose_p_online']
    
    tran_t = data['tran_t']
    tran_p = data['tran_p_online']
    
    pose_t = pose_t.view(-1, 24, 3, 3)
    pose_p = pose_p.view(-1, 24, 3, 3)
    
    # convert to cpu
    pose_t = pose_t.cpu()
    pose_p = pose_p.cpu()
    tran_t = tran_t.cpu()
    tran_p = tran_p.cpu()
    
    # 将tran_t和tran_p的第一帧align到一起
    tran_t = tran_t - tran_t[0] + tran_p[0]
    
    if relative_height:
        rheight_t = data['rheight']

        _, _, vert = body_model.forward_kinematics(pose_p, calc_mesh=True)
        
        rheight_p = vert[:, vi_mask[0], 1] - vert[:, vi_mask[3], 1]
        
        rheight_p = rheight_p.view(-1, 1)
        
        return pose_t, pose_p, tran_t, tran_p, rheight_t, rheight_p
    
    return pose_t, pose_p, tran_t, tran_p

def edit_height(pose_t, tran_t):
    
    _, _, vert = body_model.forward_kinematics(pose_t, tran=tran_t, calc_mesh=True)
    
    tran_t_y = vert[:, vi_mask[3], 1] - vert[0, vi_mask[3], 1] + tran_t[0, 1]
    
    return tran_t_y

if __name__ == '__main__':
    data_dir = 'data/eval/mobileposer_wphys/totalcapture'
    
    data_path = os.path.join(data_dir, '30.pt')
    data = torch.load(data_path)
    
    pose_t, pose_p, tran_t, tran_p = process_mobileposer_data(data)
    
    visualize([pose_t, pose_p], [tran_t, tran_p])