import os
import re
import cv2
import json
import math
import numpy as np
from PIL import Image
import matplotlib.image as mpimg
from utils.geometry import *
from utils.params import *

def parse_point(s):
    match = re.match(r"POINT\(([-\d\.]+),\s*([-\d\.]+)\)", s)
    return [float(match.group(1)), float(match.group(2))]

def parse_line(s):
    match = re.match(r"LINESTRING\((.+)\)", s)
    points = match.group(1)
    return [list(map(float, point.strip().split())) for point in points.split(",")]

def parse_graph(graph_data):
    cross_nodes = []
    room_nodes = []
    all_nodes = []
    edges = []
    for node in graph_data:
        if node['data_type'] == 'POINT':
            point = parse_point(node['coordinate'])
            all_nodes.append(point)
            if node['type'] == 'multistage':
                cross_nodes.append(point)
            if node['room'] == 'roomnode':
                room_nodes.append(point)
        elif node['data_type'] == 'LINE':
            line = parse_line(node['coordinate'])
            edges.append(line)
    return cross_nodes, room_nodes, all_nodes, edges

def parse_room(room_data):
    rooms = room_data['rooms']
    door_lines = []
    hole_data = room_data['holes']
    for data in hole_data:
        if (data['type'] == 'DOOR' or data['type'] == 'OPENING'):
            door_data = data['profile']
            temp_x = [pt[0] for pt in door_data]
            temp_y = [pt[1] for pt in door_data]
            min_x, max_x = min(temp_x), max(temp_x)
            min_y, max_y = min(temp_y), max(temp_y)
            door_lines.append([[min_x, min_y], [max_x, max_y]])
    polygons = extract_2d_polygons(room_data)
    return rooms, door_lines, polygons

def parse_meta_data(meta_data):
    pattern = re.compile(r'Top Left: \(([^,]+),\s*([^)]+)\)\s*'
                            r'Top Right: \(([^,]+),\s*([^)]+)\)\s*'
                            r'Bottom Left: \(([^,]+),\s*([^)]+)\)\s*'
                            r'Bottom Right: \(([^,]+),\s*([^)]+)\)', re.MULTILINE)
    m = pattern.search(meta_data)
    top_left_x, top_left_y = (float(m.group(1)), float(m.group(2)))
    return top_left_x, top_left_y


def wall_to_polygon(p1, p2, thickness):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return []
    nx = -dy / length
    ny = dx / length
    half_t = thickness / 2.0
    p1_left  = (x1 + nx * half_t, y1 + ny * half_t)
    p1_right = (x1 - nx * half_t, y1 - ny * half_t)
    p2_right = (x2 - nx * half_t, y2 - ny * half_t)
    p2_left  = (x2 + nx * half_t, y2 + ny * half_t)
    return [p1_left, p1_right, p2_right, p2_left]

def hole_to_wall_polygon(hole):
    profile = hole.get("profile", [])
    if len(profile) < 2:
        return []
    xs = [p[0] for p in profile if len(p) >= 2]
    ys = [p[1] for p in profile if len(p) >= 2]
    if max(ys) - min(ys) < 1e-3:
        y_wall = ys[0]
        x_min, x_max = min(xs), max(xs)
        p1 = [x_min, y_wall]
        p2 = [x_max, y_wall]
    elif max(xs) - min(xs) < 1e-3:
        x_wall = xs[0]
        y_min, y_max = min(ys), max(ys)
        p1 = [x_wall, y_min]
        p2 = [x_wall, y_max]
    else:
        return []
    thickness = hole.get("thickness", 0.1)
    return wall_to_polygon(p1, p2, thickness)

def extract_2d_polygons(data):
    polygons = []
    for wall in data.get("walls", []):
        loc = wall.get("location", [])
        if len(loc) != 2:
            continue
        thickness = wall.get("thickness", 0.1)
        poly = wall_to_polygon(loc[0], loc[1], thickness)
        if poly:
            polygons.append(poly)
    for hole in data.get("holes", []):
        poly = hole_to_wall_polygon(hole)
        if poly:
            polygons.append(poly)  
    for room in data.get("rooms", []):
        profile = room.get("profile", [])
        room_type = room.get("room_type", "unknown")
        if not profile or room_type == 'outdoor':
            continue
        poly_2d = [(pt[0], pt[1]) for pt in profile if len(pt) >= 2]
        if len(poly_2d) >= 3:
            polygons.append(poly_2d)
    return polygons

def get_local_node(nodes):
    local_nodes = []
    dir_offsets = [0, np.pi / 2, -np.pi / 2, np.pi]
    for view_node, offset in zip(nodes.values(), dir_offsets):
        for pixel_node in view_node:
            node = view_to_local(pixel_node, offset)
            local_nodes.append(node)
    return local_nodes

def load_map_data(map_dir):
    link_map_file = os.path.join(map_dir,  'link_map.json')
    structure_file = os.path.join(map_dir,  'structure.json')
    meta_file = os.path.join(map_dir,  'occ_map_meta.txt')
    occ_file = os.path.join(map_dir, 'occ_map.png')
    height_file = os.path.join(map_dir, 'occ_map_height.tiff')
    
    with open(link_map_file, 'r') as f:
        graph_data = json.load(f)
        cross_nodes, room_nodes, all_nodes, edges = parse_graph(graph_data)
    with open(structure_file, 'r') as f:
        room_data = json.load(f)
        rooms, door_lines, polygons = parse_room(room_data)
    with open(meta_file, 'r') as f:
        meta_data = f.read()
        top_left_x, top_left_y = parse_meta_data(meta_data)   

    height_map = np.array(Image.open(height_file))
    occ_map = mpimg.imread(occ_file)
    if occ_map.dtype == np.uint8:
        occ_map = occ_map.astype(np.float32) / 255.0
    else:
        occ_map = np.clip(occ_map.astype(np.float32), 0.0, 1.0)
    if occ_map.ndim == 3:
        occ_map = occ_map[:, :, 0]
    h, w = occ_map.shape
    
    explore_map = np.zeros((h, w), dtype=np.uint8)
    for poly_world in polygons:
        if len(poly_world) < 3:
            continue
        poly_pixel = []
        for (wx, wy) in poly_world:
            px, py = world_to_pixel([wx, wy], top_left_x, top_left_y, resolution)
            if 0 <= px < w and 0 <= py < h:
                poly_pixel.append([px, py])
        if len(poly_pixel) < 3:
            continue
        pts = np.array(poly_pixel, dtype=np.int32)
        cv2.fillPoly(explore_map, [pts], color=1)
    explore_map = explore_map & (occ_map > 0.5)
    
    map_dict = {
        'polygons': polygons,
        'resolution': resolution,
        'top_left_x': top_left_x,
        'top_left_y': top_left_y,
        'occ_map': occ_map,
        'height_map': height_map,
        'cross_nodes': cross_nodes,
        'room_nodes': room_nodes,
        'all_nodes': all_nodes,
        'edges': edges,
        'rooms': rooms,
        'door_lines': door_lines,
        'explore_map': explore_map
    }
    return map_dict
