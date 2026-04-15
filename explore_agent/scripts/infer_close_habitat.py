import os
import shutil 
import pickle
from PIL import Image
from modules.topograph import TopoGraph
from modules.model_qwen2_5 import QwenModel2_5
from utils.scene import *
from utils.prompt import *
from utils.params import *
from utils.params_habitat import *
from utils.render import *
from utils.visualize import *
from utils.metric import *
from habitat import Env
from habitat.datasets import make_dataset

def process_graph_data(env: Env, model: QwenModel2_5, graph: TopoGraph, gt_graphs):
    obs = env.reset()
    scene_name = env.current_episode.scene_id
    scene_id = os.path.splitext(os.path.basename(scene_name))[0]
    save_dir = os.path.join(result_dir, 'habitat', str(scene_id))
    render_vis_dir = os.path.join(save_dir, 'render_vis')
    graph_vis_dir = os.path.join(save_dir, 'graph_vis')
    map_vis_dir = os.path.join(save_dir, 'map_vis')
    os.makedirs(os.path.join(result_dir, 'habitat', str(scene_id)), exist_ok = True)
    os.makedirs(render_vis_dir, exist_ok = True)
    os.makedirs(graph_vis_dir, exist_ok = True)
    os.makedirs(map_vis_dir, exist_ok = True)
    
    frame_id = 0
    path_length = 0
    metric_dict = {}
    print('frame id', frame_id)
    pos = [-env._sim.get_agent_state().position[2],
        -env._sim.get_agent_state().position[0],
        env._sim.get_agent_state().position[1]]
    quat = [env._sim.get_agent_state().rotation.x, 
        env._sim.get_agent_state().rotation.z,
        env._sim.get_agent_state().rotation.y,
        env._sim.get_agent_state().rotation.w]
    cur_pose = pose_habitat_to_3d(pos, quat)
    cur_pos, cur_dir = pose3d_to_2d(cur_pose)
    views = ['front', 'left', 'right', 'back']
    cur_image = [Image.fromarray(obs[view]) for view in views]
    ordered_image = [cur_image[1], cur_image[0], cur_image[2], cur_image[3]]
    scene_question = generate_scene_question()
    scene_message, resized_height, resized_width = model.qwen_data_pack(ordered_image, scene_question)
    scene_text = model.qwen_infer(scene_message)[0]
    scale_height, scale_width = resized_height / image_height,  resized_width / image_width
    scene_dict = parse_scene_answer(scene_text, cur_pos, cur_dir, scale_height, scale_width)
    graph.add_pose(frame_id, cur_pose)
    if scene_dict is not None:
        graph.add_scene(frame_id, scene_dict)
    metric_info = env.get_metrics()
    metric_dict = get_explore_info(metric_dict, frame_id, path_length, graph, metric_info, gt_graphs[scene_id])
    if visualize:
        draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
    if graph.check_all_explored():
        return 
    
    cur_id = graph.cur_id
    target_id = graph.get_target_node()
    path = graph.get_path(cur_id, target_id)
    next_id = path[1]
    next_node = graph.node_info[next_id].pos
    last_id = frame_id
    while not graph.check_all_explored() and frame_id < max_step:
        while get_dist(cur_pos, next_node) >= step_dist and frame_id < max_step:
            next_dir = math.atan2(next_node[1] - cur_pos[1], next_node[0] - cur_pos[0])
            dir_diff = next_dir - cur_dir
            dir_diff = (dir_diff + math.pi) % (2 * math.pi) - math.pi
            dist = get_dist(next_node, cur_pos)
            action = {"action": "GO_TOWARD_POINT", "action_args": {"theta": dir_diff, "r": dist}}
            obs = env.step(action)
            pos = [-env._sim.get_agent_state().position[2],
                -env._sim.get_agent_state().position[0],
                env._sim.get_agent_state().position[1]]
            quat = [env._sim.get_agent_state().rotation.x,
                env._sim.get_agent_state().rotation.z,
                env._sim.get_agent_state().rotation.y,
                env._sim.get_agent_state().rotation.w]
            cur_pose = pose_habitat_to_3d(pos, quat)
            cur_pos, cur_dir = pose3d_to_2d(cur_pose)
            views = ['front', 'left', 'right', 'back']
            cur_image = [Image.fromarray(obs[view]) for view in views]
            ordered_image = [cur_image[1], cur_image[0], cur_image[2], cur_image[3]]
            pose_list = get_line_pose(graph.cur_pos, cur_pos, cur_dir)
            path_length += get_dist(graph.cur_pos, cur_pos)
            for pose in pose_list:
                frame_id += 1
                graph.add_pose(frame_id, pose)
            print('frame id', frame_id)
            if graph.check_collision():
                graph.node_info[target_id].explored = True
                if graph.node_info[target_id].type == 'frontier':
                    graph.node_info[target_id].type = 'normal'
                target_id = -1
                next_id = -1
                break

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
        
        if last_id == frame_id:
            graph.node_info[target_id].explored = True
            if graph.node_info[target_id].type == 'frontier':
                graph.node_info[target_id].type = 'normal'
        last_id = frame_id
        
        if cur_id == target_id:
            scene_question = generate_scene_question()
            scene_message, resized_height, resized_width = model.qwen_data_pack(ordered_image, scene_question)
            scene_text = model.qwen_infer(scene_message)[0]
            scale_height, scale_width = resized_height / image_height,  resized_width / image_width
            scene_dict = parse_scene_answer(scene_text, cur_pos, cur_dir, scale_height, scale_width)
            if scene_dict is not None:
                graph.add_scene(frame_id, scene_dict)
            metric_info = env.get_metrics()
            metric_dict = get_explore_info(metric_dict, frame_id, path_length, graph, metric_info, gt_graphs[scene_id])
            if visualize:
                draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
                draw_habitat_metric(graph, metric_info, metric_dict, os.path.join(map_vis_dir, f'{frame_id}.jpg'))
                
        elif target_id == -1:
            metric_info = env.get_metrics()
            metric_dict = get_explore_info(metric_dict, frame_id, path_length, graph, metric_info, gt_graphs[scene_id])
            if visualize:
                draw_dataset(cur_image, scene_dict, os.path.join(graph_vis_dir, f'{frame_id}.jpg'))
                draw_habitat_metric(graph, metric_info, metric_dict, os.path.join(map_vis_dir, f'{frame_id}.jpg'))   

        if target_id == -1 or cur_id == target_id:
            if not graph.check_all_explored():
                target_id = graph.get_target_node()
                path = graph.get_path(cur_id, target_id)
                next_id = path[1]
                next_node = graph.node_info[next_id].pos
        else:
            path = graph.get_path(cur_id, target_id)
            next_id = path[1]
            next_node = graph.node_info[next_id].pos

    final_result = {
        'scene_name': scene_id,
        'pl': metric_dict['path_length'][-1],
        'node_cr': metric_dict['node_coverage_rate'][-1],
        'occ_cr': metric_dict['occ_coverage_rate'][-1],
        'occ_area': metric_dict['covered_volume'][-1],
        'node_count': metric_dict['covered_nodes'][-1],
        'gt_node_count': metric_dict['gt_nodes'][-1]
    }
    json_path = os.path.join(save_dir, "result.json")
    with open(json_path, "w") as f:
        json.dump(final_result, f, indent=4)
        
    clean_nodes, clean_edges, clean_types, clean_rooms, clean_frame_ids = graph.build_clean_graph()
    final_graph = {
        "nodes": [[round(float(coord), 2) for coord in node] for node in clean_nodes],
        "edges": clean_edges,
        "types": clean_types,
        "rooms": clean_rooms,
        "frame_ids": clean_frame_ids
    }
    json_path = os.path.join(save_dir, "graph.json")
    with open(json_path, "w") as f:
        json.dump(final_graph, f, indent=4)
    if visualize:
        image_path = os.path.join(save_dir, "graph.png")
        graph.visualize_graph(clean_nodes, clean_edges, clean_types, clean_rooms, image_path)  

def main():
    model = QwenModel2_5(model_path)
    os.makedirs(result_dir, exist_ok = True)
    dataset = make_dataset(id_dataset=habitat_cfg.TASK_CONFIG.DATASET.TYPE, config=habitat_cfg.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    env = Env(habitat_cfg.TASK_CONFIG, dataset)
    with open(graph_path, "rb") as f:
        gt_graphs = pickle.load(f)
    for i in range(len(env.episodes)):
        graph = TopoGraph(merge_dist, vis_dist, merge_dir, step_dist)
        process_graph_data(env, model, graph, gt_graphs)

if __name__ == '__main__':
    main()