from collections import defaultdict
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from utils.geometry import *
from utils.params import *

def get_explore_info(metric_dict, frame_id, path_length, graph, metric_info, gt_graph):
    top_left_x = -metric_info["top_down_map_vlnce"]['bounds']['lower'][2]
    top_left_y = -metric_info["top_down_map_vlnce"]['bounds']['lower'][0]
    resolution = metric_info["top_down_map_vlnce"]['meters_per_px']
    map_arr = metric_info["top_down_map_vlnce"]["map"]
    fog_arr = metric_info["top_down_map_vlnce"]["fog_of_war_mask"]
    gt_nodes = [tuple(node['position']) for node in gt_graph.nodes._nodes.values()]
    gt_nodes = [[-node[2], -node[0]] for node in gt_nodes]
    traj_pos = [tuple(p) for p in graph.traj_pos]

    node_covered = []
    node_uncovered = []
    for gt in gt_nodes:
        px, py = world_to_pixel(gt, top_left_x, top_left_y, resolution, -90)
        if any(get_dist(gt, tp) <= metric_dist for tp in traj_pos) and fog_arr[py][px] == 1:
            node_covered.append(gt)
        else:
            node_uncovered.append(gt)
    node_coverage_rate = len(node_covered) / len(gt_nodes) if gt_nodes else 0.0
    map_ids = np.argwhere((map_arr != 0) & (map_arr != 2)) 
    total_area = map_ids.shape[0]           
    fog_ids = np.argwhere(fog_arr == 1)
    covered_area = fog_ids.shape[0]
    covered_volume = covered_area * metric_info['top_down_map_vlnce']['meters_per_px']
    occ_coverage_rate = covered_area / total_area if total_area > 0 else 0.0
    metric_dict['graph_info'] = {
        'coverage_rate': node_coverage_rate,
        'covered_nodes': node_covered,
        'uncovered_nodes': node_uncovered
    }
    metric_dict['occ_info'] = {
        'coverage_rate': occ_coverage_rate,
        'covered_area': covered_area,
        'total_area': total_area
    }

    if 'frame_id' in metric_dict:
        metric_dict['frame_id'].append(frame_id)
        metric_dict['occ_coverage_rate'].append(occ_coverage_rate)
        metric_dict['node_coverage_rate'].append(node_coverage_rate)
        metric_dict['path_length'].append(path_length)
        metric_dict['covered_volume'].append(covered_volume)
        metric_dict['covered_nodes'].append(node_covered)
    else:
        metric_dict['frame_id'] = [frame_id]
        metric_dict['occ_coverage_rate'] = [occ_coverage_rate]
        metric_dict['node_coverage_rate'] = [node_coverage_rate]
        metric_dict['path_length'] = [path_length]
        metric_dict['covered_volume'] = [covered_volume]
        metric_dict['covered_nodes'] = [node_covered]
        
    return metric_dict
