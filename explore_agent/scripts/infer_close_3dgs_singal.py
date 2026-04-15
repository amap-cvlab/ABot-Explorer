import os
import shutil 
import json
from tqdm import tqdm
from modules.topograph import TopoGraph
from modules.model_qwen2_5 import QwenModel2_5
from utils.scene import *
from utils.prompt import *
from utils.params import *
from utils.render import *
from utils.visualize import *

# CUDA_VISIBLE_DEVICES=0 python scripts/infer_close_3dgs_singal.py -c config/infer_interiorgs.yaml -s 0876_841803

def process_graph_data(model: QwenModel2_5, graph: TopoGraph, map_dict, scene_name, traj_id):
    render_vis_dir = os.path.join(result_dir, scene_name, traj_id, 'render_vis')
    graph_vis_dir = os.path.join(result_dir, scene_name, traj_id, 'graph_vis')
    map_vis_dir = os.path.join(result_dir, scene_name, traj_id, 'map_vis')
    os.makedirs(render_vis_dir, exist_ok = True)
    os.makedirs(graph_vis_dir, exist_ok = True)
    os.makedirs(map_vis_dir, exist_ok = True)

    frame_id = 0
    metrics_log = []
    graph.reset()
    edges = map_dict['edges']
    cur_pos, cur_dir, edges = get_random_start(edges)
    # with open(os.path.join('YOUR_START_POSE_DIR', scene_name, f'start_pose_{scene_name}.json'), 'r') as f:
    #     start_data = json.load(f)
    #     cur_pos = np.array(start_data['pos'])
    #     cur_dir = start_data['dir']
    cur_pose = pose2d_to_3d(cur_pos, cur_dir)
    cur_height = camera_height
    cur_image = render_gs(cur_pos, cur_dir, cur_height, os.path.join(render_vis_dir, f'{frame_id}.jpg'))
    ordered_image = [cur_image[1], cur_image[0], cur_image[2], cur_image[3]]
    scene_question = generate_scene_question()
    scene_message, resized_height, resized_width = model.qwen_data_pack(ordered_image, scene_question)
    scene_text = model.qwen_infer(scene_message)[0]
    scale_height, scale_width = resized_height / image_height,  resized_width / image_width
    scene_dict = parse_scene_answer(scene_text, cur_pos, cur_dir, scale_height, scale_width)
    graph.add_pose(frame_id, cur_pose)
    graph.add_scene(frame_id, scene_dict)
    if scene_dict is not None:
        graph.add_scene(frame_id, scene_dict)
    if graph.check_all_explored():
        return 
    cur_id = graph.cur_id
    
    if decision_mode == 'normal':
        target_id = graph.get_target_node()
    elif decision_mode == 'infer':
        local_graph = graph.get_local_graph()
        graph_question = generate_graph_question(local_graph)
        graph_message = model.qwen_data_pack([], graph_question)
        graph_text = model.qwen_infer(graph_message)[0]
        target_id = int(graph_text.get("target_id", None))
    
    path = graph.get_path(cur_id, target_id)
    next_id = path[1]
    next_node = graph.node_info[next_id].pos
    last_next_id = next_id

    if visualize:
        draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
        metric_interiorgs(map_dict, graph, os.path.join(map_vis_dir, f'{frame_id}.jpg'))
    else:
        metric_interiorgs(map_dict, graph)        
    metrics_log.append({
        'frame': frame_id,
        'path_length': map_dict['path_length'],
        'explore_area_m2': map_dict['explore_area_m2'],
        'coverage_rate': map_dict['coverage_rate'],
        'node_coverage_rate': map_dict['node_coverage_rate'],
        'matched_gt_nodes': map_dict['matched_gt_nodes'],
        'total_gt_nodes': map_dict['total_gt_nodes']
    })

    while not graph.check_all_explored() and frame_id < max_step:
        while get_dist(cur_pos, next_node) > step_dist and frame_id < max_step:
            frame_id += 1
            next_dir = math.atan2(next_node[1] - cur_pos[1], next_node[0] - cur_pos[0])
            dir_diff = next_dir - cur_dir
            dir_diff = (dir_diff + math.pi) % (2 * math.pi) - math.pi
            if abs(dir_diff) < np.radians(step_angle):
                cur_dir = math.atan2(next_node[1] - cur_pos[1], next_node[0] - cur_pos[0])
                dx = math.cos(cur_dir)
                dy = math.sin(cur_dir)
                cur_pos = [cur_pos[0] + dx * step_dist, cur_pos[1] + dy * step_dist]
            else:
                cur_dir += np.radians(step_angle) * np.sign(dir_diff)
            cur_pose = pose2d_to_3d(cur_pos, cur_dir)
            cur_image = render_gs(cur_pos, cur_dir, cur_height, os.path.join(render_vis_dir, f'{frame_id}.png'))
            ordered_image = [cur_image[1], cur_image[0], cur_image[2], cur_image[3]]
            graph.add_pose(frame_id, cur_pose)
            # if not graph.check_indoor(cur_pos, map_dict['polygons']):
            #     graph.node_info[target_id].explored = True
            #     if graph.node_info[target_id].type == 'frontier':
            #         graph.node_info[target_id].type = 'normal'
            #     target_id = -1
            #     next_id = -1
            #     break

        if frame_id >= max_step:
            return
        
        if target_id != -1:
            near_id = graph.get_near_node()
            cur_id = near_id[0]
            for id in near_id:
                if id == next_id:
                    cur_id = id
            graph.cur_id = cur_id
        else:
            cur_id = graph.reset_cur_node()
            graph.cur_id = cur_id
        
        if cur_id == target_id:
            scene_question = generate_scene_question()
            scene_message, resized_height, resized_width = model.qwen_data_pack(ordered_image, scene_question)
            scene_text = model.qwen_infer(scene_message)[0]
            scale_height, scale_width = resized_height / image_height,  resized_width / image_width
            scene_dict = parse_scene_answer(scene_text, cur_pos, cur_dir, scale_height, scale_width)
            if scene_dict is not None:
                graph.add_scene(frame_id, scene_dict)

            if visualize:
                draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
                metric_interiorgs(map_dict, graph, os.path.join(map_vis_dir, f'{frame_id}.jpg'))
            else:
                metric_interiorgs(map_dict, graph)  
            metrics_log.append({
                'frame': frame_id,
                'path_length': map_dict['path_length'],
                'explore_area_m2': map_dict['explore_area_m2'],
                'coverage_rate': map_dict['coverage_rate'],
                'node_coverage_rate': map_dict['node_coverage_rate'],
                'matched_gt_nodes': map_dict['matched_gt_nodes'],
                'total_gt_nodes': map_dict['total_gt_nodes']
            })

        elif target_id == -1:
            if visualize:
                draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
                metric_interiorgs(map_dict, graph, os.path.join(map_vis_dir, f'{frame_id}.jpg'))
            else:
                metric_interiorgs(map_dict, graph)  
            metrics_log.append({
                'frame': frame_id,
                'path_length': map_dict['path_length'],
                'explore_area_m2': map_dict['explore_area_m2'],
                'coverage_rate': map_dict['coverage_rate'],
                'node_coverage_rate': map_dict['node_coverage_rate'],
                'matched_gt_nodes': map_dict['matched_gt_nodes'],
                'total_gt_nodes': map_dict['total_gt_nodes']
            })

        if target_id == -1 or cur_id == target_id:
            if not graph.check_all_explored():
                if decision_mode == 'normal':
                    target_id = graph.get_target_node()
                elif decision_mode == 'infer':
                    local_graph = graph.get_local_graph()
                    graph_question = generate_graph_question(local_graph)
                    graph_message = model.qwen_data_pack([], graph_question)
                    graph_text = model.qwen_infer(graph_message)[0]
                    target_id = int(graph_text.get("target_id", None))
                path = graph.get_path(cur_id, target_id)
                next_id = path[1]
                next_node = graph.node_info[next_id].pos
                if next_id == last_next_id:
                    cur_pos = next_node
                    cur_pose = pose2d_to_3d(cur_pos, cur_dir)
                    graph.add_pose(frame_id, cur_pose)
                last_next_id = next_id
        else:
            path = graph.get_path(cur_id, target_id)
            next_id = path[1]
            next_node = graph.node_info[next_id].pos
            if next_id == last_next_id:
                cur_pos = next_node
                cur_pose = pose2d_to_3d(cur_pos, cur_dir)
                graph.add_pose(frame_id, cur_pose)
            last_next_id = next_id

    clean_nodes, clean_edges, clean_types, clean_rooms, clean_frame_ids = graph.build_clean_graph()
    final_graph = {
        "nodes": [[round(float(coord), 2) for coord in node] for node in clean_nodes],
        "edges": clean_edges,
        "types": clean_types,
        "rooms": clean_rooms,
        "frame_ids": clean_frame_ids
    }
    json_path = os.path.join(result_dir, scene_name, traj_id, "graph.json")
    with open(json_path, "w") as f:
        json.dump(final_graph, f, indent=4)
    if visualize:
        image_path = os.path.join(result_dir, scene_name, traj_id, "graph.png")
        graph.visualize_graph(clean_nodes, clean_edges, clean_types, clean_rooms, image_path)
    
    final_result = {
        "scene_name": scene_name,
        "occ_cr": metrics_log[-1]["coverage_rate"],
        "node_cr": metrics_log[-1]["node_coverage_rate"],
        "pl": metrics_log[-1]["path_length"],
        "occ_area": metrics_log[-1]["explore_area_m2"],
        "node_count": metrics_log[-1]["matched_gt_nodes"],
        'gt_node_count': metrics_log[-1]['total_gt_nodes']
    }
    json_path = os.path.join(result_dir, scene_name, traj_id, "result.json")
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=4)
    
def main():
    scene_name = args.scene
    print(f'Processing Scene {scene_name}')
    model = QwenModel2_5(model_path)
    os.makedirs(result_dir, exist_ok = True)
    os.makedirs(os.path.join(result_dir, scene_name), exist_ok = True)
    shutil.rmtree(os.path.join(result_dir, scene_name))
    graph = TopoGraph(merge_dist, vis_dist, merge_dir, step_dist)
    for i in tqdm(range(repeat_num), desc='Repeating Traj'):
        map_dict = load_map_data(os.path.join(map_dir, scene_name))
        process_graph_data(model, graph, map_dict, scene_name, str(i))

if __name__ == '__main__':
    main()