import os
import oss2
import yaml
import argparse
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--config_file", "-c", type=str, default="config/infer_habitat.yaml")
parser.add_argument('-s', '--scene', type=str, required=False)
args, _ = parser.parse_known_args()
config_file = args.config_file

class DictNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, DictNamespace(**v) if isinstance(v, dict) else v)
    def get(self, key, default=None):
        return getattr(self, key, default)

current_datetime = datetime.now()
date = current_datetime.strftime("%m%d")

with open(config_file) as f:
    cfg = DictNamespace(**yaml.safe_load(f))

model_path = cfg.get('model_path')
map_dir = cfg.get('map_dir')
scene_path = cfg.get('scene_path')
oss_dir = cfg.get('oss_dir')
result_dir = os.path.join(cfg.get('result_dir'), f'{date}')
traj_dir = cfg.get('traj_dir')
graph_dir = cfg.get('graph_dir')
graph_path = cfg.get('graph_path')

render_url = cfg.get('render_url')
bucket_name = cfg.get('bucket_name')
ossKeyId = cfg.get('ossKeyId')
ossKeySecret = cfg.get('ossKeySecret')
endPoint = cfg.get('endPoint')
if bucket_name is not None:
    bucket = oss2.Bucket(oss2.Auth(ossKeyId, ossKeySecret), endPoint, bucket_name)

use_back = cfg.get('use_back')
mode = cfg.get('mode')
decision_mode = cfg.get('decision_mode')
prompt_mode = cfg.get('prompt_mode')
visualize = cfg.get('visualize')
max_step = cfg.get('max_step')
repeat_num = cfg.get('repeat_num')
num_workers = cfg.get('num_workers')

camera_height = cfg.get('camera_height')
camera_intrinsics = cfg.get('camera_intrinsics')
cut_image_height = cfg.get('cut_image_height')
step_dist = cfg.get('step_dist')
step_angle = cfg.get('step_angle')
if camera_intrinsics is not None:
    camera_params = camera_intrinsics.split()
    camera_K = np.array([
        [float(camera_params[4]), 0, float(camera_params[6])],
        [0, float(camera_params[5]), float(camera_params[7])],
        [0, 0, 1]
    ])
    image_height = float(camera_params[3])
    image_width = float(camera_params[2])

resolution = cfg.get('resolution')
node_dist = cfg.get('node_dist')
pix_dist = cfg.get('pix_dist')
vis_dist = cfg.get('vis_dist')
merge_dist = cfg.get('merge_dist')
merge_dir = cfg.get('merge_dir')
metric_dist = cfg.get('metric_dist')