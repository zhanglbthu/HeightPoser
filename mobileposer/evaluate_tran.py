import os
import numpy as np
import torch
from argparse import ArgumentParser
import tqdm 

from mobileposer.config import *
from mobileposer.helpers import * 
import mobileposer.articulate as art
from mobileposer.constants import MODULES
from mobileposer.utils.model_utils import load_model
from mobileposer.data import PoseDataset
from mobileposer.models import MobilePoserNet
from mobileposer.models import Velocity_new
from pathlib import Path
from mobileposer.utils.file_utils import (
    get_best_checkpoint
)


class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps)

    def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        tran_p = tran_p.clone().view(-1, 3)
        tran_t = tran_t.clone().view(-1, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                                  'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)', 
                                  'Jitter Error (100m/s^3)', 'Distance Error (cm)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
            
    @staticmethod
    def print_single(errors, file=None):
        metric_names = [
            'SIP Error (deg)', 
            'Angular Error (deg)', 
            'Masked Angular Error (deg)',
            'Positional Error (cm)', 
            'Masked Positional Error (cm)', 
            'Mesh Error (cm)', 
            'Jitter Error (100m/s^3)', 
            'Distance Error (cm)'
        ]

        # 找出最长的指标名，以便统一对齐
        max_len = max(len(name) for name in metric_names)

        # 将每个指标的输出字符串保存到列表中，最后 join 成一行输出
        output_parts = []
        for i, name in enumerate(metric_names):
            if name in ['Angular Error (deg)', 'Mesh Error (cm)']:
                # 对这类指标使用“均值 ± 方差”的格式
                output_str = f"{name:<{max_len}}: {errors[i,0]:.2f}"
            else:
                continue
            
            output_parts.append(output_str)

        # 最终打印为一行
        print(" | ".join(output_parts), file=file)
        # 如果需要在末尾换行，print 本身就会换行，无需额外操作

@torch.no_grad()
def evaluate_tran(model, dataset, num_past_frame=20, num_future_frame=5, evaluate_tran=False,
                  save_dir=None):
    # specify device
    device = model_config.device
    
    xs, ys = zip(*[(imu.to(device), (pose.to(device), tran)) for imu, pose, joint, tran in dataset])

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    offline_errs, online_errs = [], []
    tran_errors = {window_size: [] for window_size in list(range(1, 8))}
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm.tqdm(zip(xs, ys), total=len(xs))):
            # x: [N, 60], y: ([N, 144], [N, 3])
            model.reset()
            pose_t, tran_t = y
            pose_t = art.math.r6d_to_rotation_matrix(pose_t)

            if getenv("ONLINE"):
                online_results = [model.forward_online_tran(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
                tran_p_online = torch.stack(online_results)[num_future_frame:]

            if evaluate_tran:
                # compute gt move distance at every frame 
                move_distance_t = torch.zeros(tran_t.shape[0])
                v = (tran_t[1:] - tran_t[:-1]).norm(dim=1)
                for j in range(len(v)):
                    move_distance_t[j + 1] = move_distance_t[j] + v[j] # distance travelled

                for window_size in tran_errors.keys():
                    # find all pairs of start/end frames where gt moves `window_size` meters
                    frame_pairs = []
                    start, end = 0, 1
                    while end < len(move_distance_t):
                        if move_distance_t[end] - move_distance_t[start] < window_size: # if not less than the window_size (in meters)
                            end += 1
                        else:
                            if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                frame_pairs.append((start, end))
                            start += 1

                    # calculate mean distance error 
                    errs = []
                    for start, end in frame_pairs:
                        vel_p = tran_p_online[end] - tran_p_online[start]
                        vel_t = (tran_t[end] - tran_t[start]).to(device)
                        errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                    if len(errs) > 0:
                        tran_errors[window_size].append(sum(errs) / len(errs))

            if getenv("ONLINE"):
                online_errs.append(evaluator.eval(pose_t, pose_t, tran_p=tran_p_online, tran_t=tran_t))
    
            if save_dir:
                torch.save({'tran_t': tran_t, 
                            'tran_p': tran_p_online,},
                            save_dir / f"{idx}.pt")
    
    log_path = save_dir / 'log.txt'
    
    with open(log_path, 'a') as f:
        # print translation errors
        if evaluate_tran:
            # print([0] + [torch.tensor(_).mean() for _ in tran_errors.values()])
            print([0] + ["{:.4f}".format(torch.tensor(_).mean().item()) for _ in tran_errors.values()], file=f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--name', type=str, default='default')
    args = parser.parse_args()

    model_dir = Path(paths.checkpoint) / 'translation' / 'velocity_new' / args.model
    
    checkpoint_path = get_best_checkpoint(model_dir)
    checkpoint = torch.load(model_dir / checkpoint_path, map_location=model_config.device)

    print(f"Loading model from {checkpoint_path}")
    
    # load model 
    model = Velocity_new(combo_id=args.model).to(model_config.device)
    
    model.load_state_dict(checkpoint['state_dict'])

    fold = 'test'
    
    dataset = PoseDataset(fold=fold, evaluate=args.dataset, combo_id=args.model)
    
    save_dir = Path('data') / 'eval_tran' / args.name / args.model / args.dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    evaluate_tran(model, dataset, evaluate_tran=True, save_dir=save_dir)