import io
import requests
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation
from utils.params import *

W_U_TRANSFORM = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
U_W_TRANSFORM = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
ISAAC_SIM_TO_GS_CONVENTION = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def as_type(data, dtype):
    if dtype == "float32":
        return data.astype(np.float32)
    elif dtype == "bool":
        return data.astype(bool)
    elif dtype == "int32":
        return data.astype(np.int32)
    elif dtype == "int64":
        return data.astype(np.int64)
    elif dtype == "long":
        return data.astype(np.long)
    elif dtype == "uint8":
        return data.astype(np.uint8)
    
def convert(data, dtype="float32"):
    return as_type(np.asarray(data), dtype)

def create_tensor_from_list(data, dtype):
    return as_type(np.array(data), dtype)

def rot_to_quat2(rot):
    rot = Rotation.from_matrix(rot)
    quat = rot.as_quat()
    if len(quat.shape) == 1:
        quat = quat[[3, 0, 1, 2]]
    else:
        quat = quat[:, [3, 0, 1, 2]]
    return quat

def quat_to_rot2(quat):
    if len(quat.shape) == 1:
        q = quat[[1, 2, 3, 0]]
    else:
        q = quat[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q)
    result = rot.as_matrix()
    return result

def rot_to_quat(R):
    qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    if qw == 0.0:
        qx = 1.0
        qy = 0.0
        qz = 0.0
    else:
        qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-6)
        qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-6)
        qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-6)
    return np.array([qw, qx, qy, qz])

def quat_to_rot(q): 
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2), 0],
        [0, 0, 0, 1]
    ])

def rot_to_angle(mat):
    sy = np.sqrt(mat[0, 0] * mat[0, 0] +  mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(mat[2, 1] , mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0
    return np.array([x, y, z])

def pos_to_pose(pos):
    pose = np.eye(4)
    pose[:3, 3] = pos
    return pose

def convert_quat(quat):
    quat = convert(quat)
    rot = quat_to_rot2(quat)
    w_u_R = create_tensor_from_list(W_U_TRANSFORM[:3, :3].tolist(), dtype="float32")
    quat = rot_to_quat2(np.matmul(rot, w_u_R))
    return quat

def get_render_image(cam_extrinsics, cam_intrinsics, save_path):
    payload = {
        'cam_extrinsics': cam_extrinsics,
        'cam_intrinsics': cam_intrinsics,
        'save_image': False
    }
    response = requests.post(render_url, json=payload)
    if response.status_code == 200:
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image_bytes = io.BytesIO(response.content)
        render_img = Image.open(image_bytes)
        if save_path is not None:
            render_img.save(save_path)
        return save_path,render_img
    return None, None

def render_gs(cur_pos, cur_yaw, cur_height, save_dir):
    render_image = []
    render_image_path = []
    pos = np.array([cur_pos[0], cur_pos[1], cur_height])

    dirs = [0, 90, -90, 180]
    views = ['front', 'left', 'right', 'back']
    for idx, dir in enumerate(dirs):
        view_yaw = np.degrees(cur_yaw + np.radians(dir))
        angle = np.array([0, 0, view_yaw])
        global_rot = Rotation.from_euler('xyz', angle, degrees=True).as_matrix()
        global_quat = rot_to_quat(global_rot)
        local_quat = convert_quat(global_quat)
        local_quat = np.array([float(local_quat[0]), float(local_quat[1]), float(local_quat[2]), float(local_quat[3])])
        local_rot = quat_to_rot(local_quat).T
        local_trans = pos_to_pose(pos).T
        T_c_w = local_rot @ local_trans
        trans_c_w = T_c_w[-1, :3]
        rot_c_w = T_c_w[:3, :3].T
        angle_c_w = rot_to_angle(rot_c_w)
        T_c_w = np.eye(4)
        T_c_w[:3, :3] = Rotation.from_euler('xyz', angle_c_w).as_matrix()
        T_c_w[:3, 3] = trans_c_w
        T_w_c = np.linalg.inv(T_c_w)
        T_w_c = ISAAC_SIM_TO_GS_CONVENTION @ T_w_c
        
        quat = rot_to_quat(T_w_c[:3, :3])
        camera_id = 1
        image_name = f"{views[idx]}.jpg"
        colmap_image = f'{idx} ' + " ".join(map(str, quat)) + " " + " ".join(map(str, T_w_c[:3, 3])) + " " + f'{camera_id} {image_name}'
        save_path = save_dir.replace('.png', f'_{views[idx]}.jpg')
        image_path, image = get_render_image(colmap_image, camera_intrinsics, save_path)
        render_image.append(image)
        render_image_path.append(image_path)
    return render_image
