import math
import random
import numpy as np
from scipy.spatial.transform import Rotation
from utils.params import *

def get_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def get_dir(p1, p2, yaw):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.atan2(dy, dx) - yaw

def get_cos(p1, p2, p3):
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.dot(v1, v2)

def get_pose(cur_info):
    pitch = cur_info['pitch']
    roll = cur_info['roll']
    yaw = cur_info['yaw']
    x = cur_info['x']
    y = cur_info['y']
    z = cur_info['z']
    theta = np.array([roll, pitch, yaw])
    rot = Rotation.from_euler('xyz', theta, degrees=False).as_matrix()
    trans = np.array([x, y, z])
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose

def pose2d_to_3d(pos, dir):
    pitch = 0
    roll = 0
    yaw = dir
    x = pos[0]
    y = pos[1]
    z = -camera_height
    theta = np.array([roll, pitch, yaw])
    rot = Rotation.from_euler('xyz', theta, degrees=False).as_matrix()
    trans = np.array([x, y, z])
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose

def pose_habitat_to_3d(pos, quat):
    rot = Rotation.from_quat([quat[0], quat[1], quat[2], quat[3]]).as_matrix()
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos 
    return pose

def pose3d_to_2d(pose):
    pos = pose[:2, 3]
    rot = pose[:3, :3]
    _, _, yaw = Rotation.from_matrix(rot).as_euler('xyz', degrees=False)
    return pos, yaw

def world_to_pixel(pos, top_left_x, top_left_y, resolution, rotate = 0):
    x, y = pos[0], pos[1]
    u = (top_left_x - x) / resolution
    v = (y - top_left_y) / resolution
    if rotate == -90:
        u, v = -v, u
    return [int(round(u)), int(round(v))]

def world_to_local(cur_pose, pos):
    R = cur_pose[:3, :3]
    T = cur_pose[:3, 3]
    pos = np.array(pos)
    local_pos = np.dot(R.T, (pos - T))
    return local_pos.tolist()

def map_to_local(cur_pos, yaw, pos):
    rot = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    local_pos = rot @ (np.array(pos) - np.array(cur_pos))
    return local_pos.tolist()

def local_to_map(cur_pos, yaw, pos):
    rot_T = np.array([[np.cos(yaw), -np.sin(yaw)], 
                      [np.sin(yaw),  np.cos(yaw)]])
    map_pos = rot_T @ np.array(pos) + np.array(cur_pos)
    return map_pos.tolist()

def local_to_view(pos, yaw):
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    rot = np.linalg.inv(rot)
    local_node = np.array(pos)
    view_node = rot @ local_node
    node_dir = np.arctan2(view_node[1], view_node[0])
    if -np.pi / 4 < node_dir < np.pi / 4:
        view_node = [-view_node[1], -view_node[2], view_node[0]]
        pix_node = camera_K @ view_node
        u, v = pix_node[:2] / (pix_node[2] + 1e-6)
        if 0 < u < float(camera_params[2]) and 0 < v < float(camera_params[3]):
            return [int(u), int(v)]
    return None

def view_to_local(pix, yaw):
    u, v = pix
    pix_node = np.array([u, v, 1])
    camera_K_inv = np.linalg.inv(camera_K)
    view_node = camera_K_inv @ pix_node
    view_node = np.array([view_node[2], -view_node[0], -view_node[1]])
    view_node = view_node * (-abs(camera_height) / view_node[2])
    rot = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0, 0, 1]])
    local_node = rot @ view_node
    pos = [local_node[0], local_node[1]]
    return pos

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
def check_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    A, B, C, D = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def calc_intersection(line1, line2):
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0: 
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    return [px, py]

def get_random_start(edges):
    edge = random.choice(edges)
    start_pos = random.choice(edge)
    start_dir = random.uniform(-math.pi, math.pi)
    return start_pos, start_dir, edges

def get_near_node(edges, pos):
    near_node = None
    for edge in edges:
        for node in edge:
            dist = math.dist(pos, node)
            if dist <= step_dist:
                near_node = node
    return near_node

def get_line_pose(start, end, dir):
    pose_list = []
    start = np.array(start)
    end = np.array(end)
    vec = end - start
    total_dist = np.linalg.norm(vec)
    pose = pose2d_to_3d(start, dir)
    pose_list.append(pose)
    num_steps = int(np.ceil(total_dist / step_dist))
    for alpha in np.linspace(0, 1, num_steps + 1):
        pos = start + alpha * vec
        pose = pose2d_to_3d(pos, dir)
        pose_list.append(pose)
    pose = pose2d_to_3d(end, dir)
    pose_list.append(pose)
    return pose_list

def point_in_polygon(point, poly):
    x, y = point
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points