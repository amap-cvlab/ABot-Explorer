import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from utils.scene import *
from utils.geometry import * 

def draw_dataset(cur_image, scene_dict, save_path):
    pixel_nodes = {}
    if scene_dict:
        pixel_nodes = scene_dict.get('pixel nodes', {})
    fig = plt.figure(figsize=(16, 5))
    grid_spec = fig.add_gridspec(1, 4, height_ratios=[1], width_ratios=[1, 1, 1, 1])
    view_num = 4 if use_back else 3
    view_titles = ['Front', 'Left', 'Right', 'Back']
    for idx in range(view_num):
        ax = fig.add_subplot(grid_spec[0, idx])
        image = np.array(cur_image[idx])
        if pixel_nodes is not None and view_titles[idx].lower() in pixel_nodes:
            for point in pixel_nodes[view_titles[idx].lower()]:
                if point:
                    cv2.circle(image, tuple(point), 20, (255, 0, 0), -1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(view_titles[idx], fontsize=14)
    plt.tight_layout(pad=2.5)
    fig.subplots_adjust(wspace=0.05)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def draw_habitat_metric(graph, metric_info, metric_dict, save_path):
    cur_pos = graph.cur_pos
    traj_pos = [tuple(p) for p in graph.traj_pos]
    top_left_x = -metric_info["top_down_map_vlnce"]['bounds']['lower'][2]
    top_left_y = -metric_info["top_down_map_vlnce"]['bounds']['lower'][0]
    resolution = metric_info["top_down_map_vlnce"]['meters_per_px']
    map_arr = metric_info["top_down_map_vlnce"]["map"]
    fog_arr = metric_info["top_down_map_vlnce"]["fog_of_war_mask"]
    node_covered = metric_dict['graph_info']['covered_nodes']
    node_uncovered = metric_dict['graph_info']['uncovered_nodes']
    vis_map = np.zeros_like(map_arr, dtype=np.uint8)
    vis_map[(map_arr == 0) | (map_arr == 2)] = 0
    vis_map[(map_arr != 0) & (map_arr != 2)] = 1
    cmap = ListedColormap(["gray", "white"])
    plt.figure(figsize=(10, 10))
    plt.imshow(vis_map, cmap=cmap, origin='upper')
    plt.imshow(fog_arr, cmap='Reds', alpha=0.3, origin='upper')
    for node in node_covered:
        px, py = world_to_pixel(node, top_left_x, top_left_y, resolution, -90)
        plt.scatter(px, py, marker='^', s=30, color='cyan',
                    label='node covered' if 'node covered' not in plt.gca().get_legend_handles_labels()[1] else "")
    for node in node_uncovered:
        px, py = world_to_pixel(node, top_left_x, top_left_y, resolution, -90)
        plt.scatter(px, py, marker='v', s=30, color='magenta',
                    label='node uncovered' if 'node uncovered' not in plt.gca().get_legend_handles_labels()[1] else "")
    px, py = world_to_pixel(cur_pos, top_left_x, top_left_y, resolution, -90)
    plt.scatter(px, py, marker='o', s=30, color='red',
                label='currnet node' if 'current node' not in plt.gca().get_legend_handles_labels()[1] else "")
    traj_px = []
    traj_py = []
    for node in traj_pos:
        px, py = world_to_pixel(node, top_left_x, top_left_y, resolution, -90)
        traj_px.append(px)
        traj_py.append(py)
    plt.plot(traj_px, traj_py, color='yellow', linewidth=2, label='trajectory')
    plt.axis('off')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    occmap_image = Image.open(save_path).convert("RGB")
    return occmap_image

def metric_interiorgs(map_dict, graph, save_path=None):
    traj_pos = graph.traj_pos          
    cur_pos = graph.cur_pos   
    edges = map_dict['edges']    
    explore_map = map_dict['explore_map']
    top_left_x = map_dict['top_left_x']
    top_left_y = map_dict['top_left_y']
    explored_mask = map_dict.get('explored_mask')
    H, W = explore_map.shape
    occ_binary = (explore_map > 0.5).astype(np.uint8)
    if explored_mask is None:
        explored_mask = np.zeros((H, W), dtype=bool)
    total_free_area = np.sum(occ_binary)
    u_cur, v_cur = world_to_pixel(cur_pos, top_left_x, top_left_y, resolution)
    if not (0 <= u_cur < W and 0 <= v_cur < H):
        visible_mask = np.zeros_like(explored_mask)
    else:
        visible_mask = np.zeros_like(explored_mask, dtype=bool)
        max_dist_m = 5.0
        num_rays = 720  
        cur_u, cur_v = world_to_pixel(cur_pos, top_left_x, top_left_y, resolution)
        for angle_deg in range(num_rays):
            theta = np.radians(angle_deg)
            end_x = cur_pos[0] + max_dist_m * np.cos(theta)
            end_y = cur_pos[1] + max_dist_m * np.sin(theta)
            end_u, end_v = world_to_pixel([end_x, end_y], top_left_x, top_left_y, resolution)
            line_pixels = bresenham_line(cur_u, cur_v, end_u, end_v)
            for u, v in line_pixels:
                if not (0 <= u < W and 0 <= v < H):
                    break
                visible_mask[v, u] = True
                if occ_binary[v, u] == 0:
                    break
        explored_mask = explored_mask | visible_mask
    explored_free_area = np.sum(explored_mask & (occ_binary == 1))
    coverage_rate = explored_free_area / total_free_area if total_free_area > 0 else 0.0
    explore_area_m2 = explored_free_area * (resolution ** 2)

    path_length = 0.0
    if len(traj_pos) > 1:
        for i in range(1, len(traj_pos)):
            dx = traj_pos[i][0] - traj_pos[i-1][0]
            dy = traj_pos[i][1] - traj_pos[i-1][1]
            path_length += np.sqrt(dx*dx + dy*dy)
    gt_nodes = set()
    for edge in edges:
        p1, p2 = edge
        x1, y1 = p1
        x2, y2 = p2
        gt_nodes.add((x1, y1))
        gt_nodes.add((x2, y2))
    gt_nodes = list(gt_nodes)
    pred_nodes = []
    for node_id, node in graph.node_info.items():
        if hasattr(node, 'pos'):
            pred_nodes.append(node.pos)

    matched_gt_count = 0
    node_match_radius = 2.0
    matched_gt_pixels = []   
    unmatched_gt_pixels = []
    for gx, gy in gt_nodes:
        gu, gv = world_to_pixel([gx, gy], top_left_x, top_left_y, resolution)
        if not (0 <= gu < W and 0 <= gv < H):
            continue
        is_explored = explored_mask[gv, gu]
        is_matched = False
        if is_explored:
            for ex, ey in pred_nodes:
                dist = np.sqrt((ex - gx)**2 + (ey - gy)**2)
                if dist <= node_match_radius:
                    is_matched = True
                    break
        if is_matched:
            matched_gt_count += 1
            matched_gt_pixels.append((gu, gv))
        else:
            unmatched_gt_pixels.append((gu, gv))
    node_coverage_rate = matched_gt_count / len(gt_nodes) if gt_nodes else 0.0

    if save_path is not None:
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
        ax.imshow(explore_map, cmap='gray', origin='upper')
        explored_overlay = np.zeros((H, W, 4))
        explored_overlay[explored_mask, 0] = 1.0
        explored_overlay[explored_mask, 3] = 0.3
        ax.imshow(explored_overlay, origin='upper', zorder=3)
        if matched_gt_pixels:
            mx, my = zip(*matched_gt_pixels)
            ax.scatter(mx, my, c='lime', s=40, marker='o', edgecolors='black', linewidth=0.5, zorder=8, label='Matched GT')
        if unmatched_gt_pixels:
            ux, uy = zip(*unmatched_gt_pixels)
            ax.scatter(ux, uy, c='red', s=40, marker='x', linewidths=2, zorder=8, label='Unmatched GT')
        if len(traj_pos) >= 2:
            pixel_coords = []
            for pos in traj_pos:
                u, v = world_to_pixel(pos, top_left_x, top_left_y, resolution)
                if 0 <= u < W and 0 <= v < H:
                    pixel_coords.append((u, v))
            if len(pixel_coords) >= 2:
                xs, ys = zip(*pixel_coords)
                ax.plot(xs, ys, color='yellow', linewidth=1.0, zorder=5)
                ax.scatter([xs[0]], [ys[0]], c='#87CEFA', s=60, zorder=6, edgecolors='black', linewidth=0.5)
                ax.scatter([xs[-1]], [ys[-1]], c='red', s=60, zorder=7, edgecolors='black', linewidth=0.5)
        if len(traj_pos) < 2:
            u_cur, v_cur = world_to_pixel(cur_pos, top_left_x, top_left_y, resolution)
            if 0 <= u_cur < W and 0 <= v_cur < H:
                ax.scatter([u_cur], [v_cur], c='red', s=60, zorder=7, edgecolors='black', linewidth=0.5)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
        
    map_dict['explored_mask'] = explored_mask
    map_dict['coverage_rate'] = coverage_rate
    map_dict['explore_area_m2'] = explore_area_m2
    map_dict['path_length'] = path_length
    map_dict['node_coverage_rate'] = node_coverage_rate
    map_dict['total_gt_nodes'] = len(gt_nodes)
    map_dict['matched_gt_nodes'] = matched_gt_count
