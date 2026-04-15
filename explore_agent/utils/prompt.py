import json
from utils.geometry import *
from utils.params import *
from utils.scene import *

def generate_graph_question(local_graph):
    return (
        "You are an autonomous exploration robot. "
        "You will get the current local graph, and you need to choose the next target node id.\n"
        "# Output format:\n"
        "{\"target_id\": <the id of the next target unexplored node>}\n"
        "# Local graph:\n"
        f"{json.dumps(local_graph, ensure_ascii=False)}"
    )

def generate_scene_question():
    pv_node_template = [
        {'id': 'node id of current observation',
         'view': 'the best view of the node',
         'pix': 'pixel position of the node in the view'}
    ]
    pv_map_template = {
        'type': 'the node type of your current position(intersection, room entry, or a normal position)',
        'room type': 'the room type of your current position(e.g. bedroom, living room, etc.)',
        'node': pv_node_template,
        'edge': 'the id list of connected nodes'
    }
    pv_map_template_string = json.dumps(pv_map_template, indent=2).replace('"[','[').replace(']"',']')
    image_string = 'left:<image>, front:<image>, right:<image>, back:<image>' if use_back else 'left:<image>, front:<image>, right:<image>'
    question = f"""You are an autonomous navigation robot. You will get current view images, please output the scene graph.
# Output format: 
{pv_map_template_string}
# Your current observation of each view: {image_string}<|NAV|>"""
    return question

def generate_scene_answer(scene_dict):
    idx = 0
    pv_nodes = []
    pv_nodes_ids = {}
    pix_nodes = scene_dict['pixel nodes']
    for view in pix_nodes.keys():
        for i in range(len(pix_nodes[view])):
            pixel = pix_nodes[view][i]
            pv_nodes.append({
                'id': f'{idx}',
                'view': view,
                'pix': f'[{pixel[0]},{pixel[1]}]'})
            pv_nodes_ids[scene_dict['pixel ids'][view][i]] = idx
            idx += 1
    pv_connection = [] 
    connections = scene_dict['connections']
    for idx in range(len(connections)):
        id1 = pv_nodes_ids[connections[idx][0]]
        id2 = pv_nodes_ids[connections[idx][1]]
        if id1 > id2:
            id1, id2 = id2, id1
        pv_connection.append(f'[{id1},{id2}]')
    if prompt_mode == 'roadmap':
        pv_map = {
            'type': scene_dict['node type'], 
            'room type': scene_dict['room type'], 
            'node': pv_nodes,
            'edge': pv_connection
        }
    else:
        pv_map = {
            'type': scene_dict['node type'], 
            'room type': scene_dict['room type'], 
            'node': pv_nodes
        }
    answer = json.dumps(pv_map, separators=(',', ':')).replace('"[','[').replace(']"',']')
    return answer

def generate_scene_prompt(scene_dict, cur_image):
    question = generate_scene_question()
    answer = generate_scene_answer(scene_dict)
    scene_prompt = {
        "messages": [
        {
            "content": question,
            "role": "user"
        },
        {
            "content": answer,
            "role": "assistant"
        }
        ],
        "images": cur_image,    
    }
    return scene_prompt

def parse_scene_answer(scene_text, cur_pos, cur_yaw, scale_height, scale_width):
    try:
        scene_dict = json.loads(scene_text)
    except:
        scene_dict = None
    if scene_dict is not None:
        pixel_nodes = {'front': [], 'left': [], 'right': [], 'back': []}
        raw_ids = {'front': [], 'left': [], 'right': [], 'back': []}
        edges = []
        for node in scene_dict['node']:
            pixel_nodes[node['view']].append([int(node['pix'][0] / scale_height), int(node['pix'][1] / scale_width)])
            raw_ids[node['view']].append(node['id'])
        for edge in scene_dict['edge']:
            edges.append([edge[0], edge[1]])
        local_nodes = get_local_node(pixel_nodes)
        adj_nodes = [local_to_map(cur_pos, cur_yaw, node) for node in local_nodes]
        scene_dict['raw ids'] = raw_ids
        scene_dict['edges'] = edges
        scene_dict['pixel nodes'] = pixel_nodes
        scene_dict['local nodes'] = local_nodes
        scene_dict['map nodes'] = adj_nodes
    return scene_dict
