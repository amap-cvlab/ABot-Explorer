import math
import numpy as np
import cv2
from collections import defaultdict, deque
from scipy.spatial import KDTree
from matplotlib.path import Path
import matplotlib.pyplot as plt

class NodeInfo:
    def __init__(self, node_id, frame_id, type, pos, room = 'unknown', label = 'normal'):
        self.merge_cnt = 1
        self.node_id = node_id
        self.frame_id = frame_id
        self.type = type
        self.label = label
        self.pos = pos
        self.room = room
        self.explored = False
        
class EdgeInfo:
    def __init__(self, id1, id2):
        self.node_id = sorted([id1, id2])

    def __eq__(self, other):
        if isinstance(other, EdgeInfo):
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash(tuple(self.node_id))

class TopoGraph:
    def __init__(self, merge_dist, vis_dist, merge_dir, step_dist):
        self.merge_dist = merge_dist
        self.vis_dist = vis_dist
        self.merge_dir = merge_dir
        self.step_dist = step_dist
        self.map_size = 512
        self.grid_size = 5
        self.background_color = (255, 255, 255)
        self.traj_color = (46, 134, 171)
        self.edge_color = (130, 0, 75)
        self.subgoal_color = (255, 0, 0)
        self.robot_color = (241, 143, 1)
        self.grid_color = (200, 200, 200)
        self.frontier_color = (139, 0, 0)
        self.cross_color = (0, 165, 255)
        self.normal_color = (150, 150, 150)
        self.room_color = (0, 0, 255)

        self.map_image = np.full((self.map_size, self.map_size, 3), self.background_color, dtype=np.uint8)
        self.reset()

    def reset(self):
        self.cur_id = -1
        self.next_id = 0
        self.cur_pose = None
        self.cur_pos = None
        self.traj_kdtree = None
        self.traj_pos = []
        self.edge_info = []
        self.cur_edge_info = []
        self.traj_pose = {}
        self.node_info = {}
        self.cur_raw_info = {}
        self.cur_frontier_info = {}
        self.id_map = {}

    def get_dist(self, p1, p2):
       return float(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    def get_dir_vec(self, pos1, pos2):
        direction = [p1 - p2 for p1, p2 in zip(pos1, pos2)]
        norm = math.sqrt(sum(d ** 2 for d in direction))
        dir = [d / (norm + 1e-6) for d in direction]
        return tuple(dir)
    
    def get_dir_diff(self, p1, p2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(p1, p2))
        norm1 = math.sqrt(sum(v**2 for v in p1)) 
        norm2 = math.sqrt(sum(v**2 for v in p2))
        return np.degrees(math.acos(dot_product / (norm1 * norm2 + 1e-8)))

    def get_node_diff(self, p1, p2, p3):
        v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        v2 = np.array([p3[0] - p1[0], p3[1] - p1[1]])
        v1 = v1 / (np.linalg.norm(v1))
        v2 = v2 / (np.linalg.norm(v2))
        cos_val = np.dot(v1, v2)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return np.degrees(math.acos(cos_val))
        
    def reset_cur_node(self):
        min_dist = float('inf')
        cur_node_id = 0
        for node_id, node_info in self.node_info.items():
            dist = self.get_dist(self.cur_pos, node_info.pos)
            if dist <= min_dist:
                cur_node_id = node_id
                min_dist = dist
        return cur_node_id
        
    def get_near_node(self, frontier = None):
        near_node_id = []
        node = frontier if frontier is not None else self.cur_pos
        for node_id, node_info in self.node_info.items():
            if self.get_dist(node, node_info.pos) <= self.merge_dist:
                near_node_id.append(node_id)
        near_node_id = sorted(near_node_id, key=lambda node_id: self.get_dist(node, self.node_info[node_id].pos))
        return near_node_id

    def get_near_traj(self, frontier):
        near_traj_id = None
        for i, pos in enumerate(self.traj_pos):
            if self.get_dist(frontier, pos) <= self.merge_dist:
                near_traj_id = i
                break
        return near_traj_id
    
    def add_node(self, frame_id, type, room):
        node_id = self.next_id
        self.node_info[node_id] = NodeInfo(node_id, frame_id, type, list(self.cur_pos), room)
        self.cur_id = node_id
        self.next_id += 1

    def add_frontier(self, adj_node, pos, room, label, raw_id):
        node_id = self.next_id
        self.id_map[raw_id] = node_id                
        self.cur_frontier_info[node_id] = NodeInfo(node_id, -1, 'frontier', pos, room, label)
        self.cur_raw_info[node_id] = NodeInfo(node_id, -1, 'frontier', adj_node, room, label)
        self.next_id += 1
    
    def merge_frontier(self, node_id, pos):
        node = self.node_info[node_id]
        edge = EdgeInfo(self.cur_id, node_id)
        if edge not in self.edge_info:
            self.edge_info.append(edge)
        node.pos[0] = (node.pos[0] * node.merge_cnt + pos[0]) / (node.merge_cnt + 1)
        node.pos[1] = (node.pos[1] * node.merge_cnt + pos[1]) / (node.merge_cnt + 1)
        node.merge_cnt += 1

    def add_edge(self, node_id):
        edge = EdgeInfo(self.cur_id, node_id)
        if edge not in self.edge_info:
            self.edge_info.append(edge)
    
    def merge_node(self, node_id, type, room):
        self.cur_id = node_id
        if type != 'normal':
            self.node_info[node_id].type = type
        self.node_info[node_id].room = room
        
    def fix_node(self, frame_id, node_id, type, room):
        self.cur_id = node_id
        node = self.node_info[node_id]
        node.frame_id = frame_id
        node.type = type
        node.room = room
        
    def update_node(self, frame_id, type, room):
        near_node_id = self.get_near_node()
        if len(near_node_id) > 0:
            node_id = near_node_id[0]
            if self.node_info[node_id].type == 'frontier':
                self.fix_node(frame_id, node_id, type, room)
            else:
                self.merge_node(node_id, type, room)
        else:
            self.add_node(frame_id, type, room)
    
    def check_frontier(self, pos, edge):
        tol = 0.5 * self.merge_dist
        p = pos
        a = self.node_info[edge.node_id[0]].pos
        b = self.node_info[edge.node_id[1]].pos
        px, py = p
        ax, ay = a
        bx, by = b
        if not (min(ax, bx) - tol <= px <= max(ax, bx) + tol and
                min(ay, by) - tol <= py <= max(ay, by) + tol):
            return False
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        if abs(cross) > tol:
            return False
        dot_product = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
        squared_length = (bx - ax)**2 + (by - ay)**2
        if squared_length < tol:
            return abs(px - ax) < tol and abs(py - ay) < tol
        t = dot_product / squared_length
        return -tol <= t <= 1.0 + tol

    def update_frontier(self, adj_nodes, raw_ids, frame_id):
        label = 'frontier'
        room = 'unknown'
        for adj_node, raw_id in zip(adj_nodes, raw_ids):
            dist = self.get_dist(adj_node, self.cur_pos)
            if dist < self.vis_dist:
                frontier = adj_node
            else:
                node_diff = np.array(adj_node) - np.array(self.cur_pos)
                frontier = np.array(self.cur_pos) + self.vis_dist * node_diff / np.linalg.norm(node_diff)
                label = 'normal'
            near_traj_id = self.get_near_traj(frontier)
            near_frontier_id = self.get_near_node(frontier)
            # if near_frontier_id == [] and near_traj_id is not None:
            #     min_dist = float('inf')
            #     near_id = None
            #     for node_id, node_info in self.node_info.items():
            #         dist = self.get_dist(self.traj_pos[near_traj_id], node_info.pos)
            #         if dist <= min_dist:
            #             min_dist = dist
            #             near_id = node_id
            #     near_frontier_id = [near_id]

            if len(near_frontier_id) > 0 and self.node_info[near_frontier_id[0]].type == 'frontier':
                self.merge_frontier(near_frontier_id[0], frontier)
                # self.add_edge(near_frontier_id[0])
            if len(near_frontier_id) > 0 and self.node_info[near_frontier_id[0]].type != 'frontier':
                if self.get_dist(frontier, self.cur_pos) > self.merge_dist:
                    self.add_edge(near_frontier_id[0])
                # if self.get_dist(frontier, self.cur_pos) < self.merge_dist:
                add_flag = True
                for near_id in near_frontier_id:
                    if frame_id - self.node_info[near_id].frame_id > 10:
                        add_flag = False
                if add_flag:
                    cur_dir = math.atan2(self.cur_pose[1, 0], self.cur_pose[0, 0])
                    vec = frontier - self.cur_pos 
                    node_dir = math.atan2(vec[1], vec[0]) 
                    theta_diff = abs((node_dir - cur_dir + math.pi) % (2 * math.pi) - math.pi)
                    if theta_diff < 5 * math.pi / 6 or frame_id == 0:
                        self.add_frontier(adj_node, frontier, room, label, raw_id)  

            if near_frontier_id == []:
                add_flag = True
                for edge in self.edge_info:
                    if self.check_frontier(frontier, edge):
                        add_flag = False
                        break
                if add_flag:
                    self.add_frontier(adj_node, frontier, room, label, raw_id)
                    frontier = 0

    def get_adj_edge(self):
        graph = defaultdict(list)
        for e in self.cur_edge_info:
            u, v = e.node_id
            graph[u].append(v)
            graph[v].append(u)
            
        visited = set()
        components = []
        for node_id in self.cur_frontier_info.keys():
            if node_id in visited:
                continue
            comp = set()
            queue = deque([node_id])
            visited.add(node_id)
            comp.add(node_id)
            while queue:
                u = queue.popleft()
                for v in graph[u]:
                    if v not in visited and v in self.cur_raw_info:
                        visited.add(v)
                        queue.append(v)
                        comp.add(v)
            components.append(comp)

        candidates = []
        for comp in components:
            best_id = None
            best_dist = float('inf')
            for node_id in comp:
                node = self.cur_raw_info[node_id]
                d = self.get_dist(self.cur_pos, node.pos)
                if d < best_dist:
                    best_dist = d
                    best_id = node_id
            if best_id is not None:
                node = self.cur_raw_info[best_id]
                dx = node.pos[0] - self.cur_pos[0]
                dy = node.pos[1] - self.cur_pos[1]
                angle = math.atan2(dy, dx)
                candidates.append((best_id, best_dist, angle))

        selected = []
        for node_id, dist, angle in candidates:
            norm_angle = angle % (2 * math.pi)
            too_close_in_direction = False
            replace_idx = -1
            for i, (sel_id, sel_dist, sel_angle) in enumerate(selected):
                ang_diff = abs(norm_angle - sel_angle)
                ang_diff = min(ang_diff, 2 * math.pi - ang_diff)
                if ang_diff <= math.radians(self.merge_dir):
                    too_close_in_direction = True
                    if dist < sel_dist:
                        replace_idx = i
                    break
            # too_close_in_direction = False
            if not too_close_in_direction:
                selected.append((node_id, dist, norm_angle))
            elif replace_idx != -1:
                selected[replace_idx] = (node_id, dist, norm_angle)
                
        for node_id, _, _ in selected:
            edge = EdgeInfo(self.cur_id, node_id)
            if edge not in self.edge_info:
                self.edge_info.append(edge)
                self.node_info[node_id] = self.cur_frontier_info[node_id]

    def update_edge_info(self, edges):
        for edge in edges:
            raw_id1, raw_id2 = edge[0], edge[1]
            if raw_id1 in self.id_map.keys() and raw_id2 in self.id_map.keys():
                node_id1, node_id2 = self.id_map[raw_id1], self.id_map[raw_id2]
                p1, p2 = self.cur_raw_info[node_id1].pos, self.cur_raw_info[node_id2].pos
                if self.get_node_diff(self.cur_pos, p1, p2) < self.merge_dir:
                    edge = EdgeInfo(node_id1, node_id2)
                    if edge not in self.cur_edge_info:
                        self.cur_edge_info.append(edge)
        self.get_adj_edge()
        # for edge in self.cur_edge_info:
        #     if edge not in self.edge_info:
        #         self.edge_info.append(edge)
        # for node_id, node in self.cur_frontier_info.items():
        #     self.node_info[node_id] = node

    def update_explore_info(self):
        for id, node in self.node_info.items():
            if id == self.cur_id:
                node.explored = True
            # if not node.explored:
            #     pos = self.cur_pos
            #     if self.get_dist(pos, node.pos) < self.merge_dist and self.get_dist(pos, node.pos) > 0.5 * self.merge_dist:
            #         cur_dir = math.atan2(self.cur_pose[1, 0], self.cur_pose[0, 0])
            #         vec = node.pos - self.cur_pos 
            #         node_dir = math.atan2(vec[1], vec[0]) 
            #         theta_diff = abs((node_dir - cur_dir + math.pi) % (2 * math.pi) - math.pi)
            #         if theta_diff > 2 * math.pi / 3:  
            #             node.explored = True
            #             if node.type == 'frontier':
            #                 node.type = 'normal'
            #         break

    def add_pose(self, frame_id, pose_info):
        self.cur_pose = pose_info
        self.cur_pos = pose_info[:2, 3]
        self.traj_pose[frame_id] = pose_info
        self.traj_pos.append(self.cur_pos)
        self.traj_kdtree = KDTree(np.array(self.traj_pos))
        self.update_explore_info()

    def add_scene(self, frame_id, scene_info):
        self.id_map = {}
        self.cur_frontier_info = {}
        self.cur_raw_info = {}
        self.cur_edge_info = []
        type = scene_info.get('type', 'normal')
        room = scene_info.get('room type', 'unknown')
        raw_ids = scene_info.get('raw ids', {})
        edges = scene_info.get('edges', {})
        adj_nodes = scene_info.get('map nodes', [])
        raw_ids = [item for sublist in raw_ids.values() for item in sublist]
        self.update_node(frame_id, type, room)
        self.update_frontier(adj_nodes, raw_ids, frame_id)
        self.update_edge_info(edges)
        self.update_explore_info()

    def check_all_explored(self):
        for node in self.node_info.values():
            if not node.explored:
                return False
        return True

    def check_collision(self):
        history_pose = [pose for pose in list(self.traj_pose.values())[-5:]]
        history_dir = [math.atan2(pose[1, 0], pose[0, 0]) for pose in history_pose]
        if len(self.traj_pos) < 5:
            return False
        history_pos = self.traj_pos[-5:]
        first_pos = history_pos[0]
        first_dir = history_dir[0]
        for pos in history_pos:
            if self.get_dist(first_pos, pos) > self.step_dist:
                return False
        for dir in history_dir:
            diff = abs((first_dir - dir + math.pi) % (2 * math.pi) - math.pi)
            if diff > math.radians(self.merge_dir):
                return False
        return True

    def check_indoor(self, pos, polygons):
        x, y = pos
        point = (x, y)
        for poly in polygons:
            if len(poly) < 3:
                continue
            path = Path(poly)
            if path.contains_point(point):
                return True
        return False

    def check_occ(self, occ_map, top_left_x, top_left_y, resolution):
        return occ_map[self.world_to_pixel(self.cur_pos, top_left_x, top_left_y, resolution)] < 0.5
        
    def get_node_info(self, node_id):
        node = self.node_info[node_id]
        cur_node_data = {
            'frame_id': node.frame_id,
            'pos': node.pos.tolist(),
        }
        return cur_node_data

    def get_all_node_info(self):
        node_data = []
        for node_id, node in self.node_info.items():
            node_data.append({'id': node_id, 'pos': node.pos})
        return node_data

    def get_all_edge_info(self):
        edge_data = []
        for edge in self.edge_info:
            edge_data.append({'id1': edge.node_id[0], 'id2': edge.node_id[1], 'explored': edge.explored})
        return edge_data

    def get_target_node(self):
        candidate_id = [node_id for node_id, node in self.node_info.items() if not node.explored]
        cur_dir = math.atan2(self.cur_pose[1, 0], self.cur_pose[0, 0])
        forward_id = []
        for node_id in candidate_id:
            node_pos = self.node_info[node_id].pos
            vec = node_pos - self.cur_pos
            node_dir = math.atan2(vec[1], vec[0])
            theta_diff = abs((node_dir - cur_dir + math.pi) % (2 * math.pi) - math.pi)
            if theta_diff < math.radians(self.merge_dir) and self.get_path_length(self.cur_id, node_id) < self.vis_dist:
                forward_id.append(node_id)
        if forward_id:
            target_id = min(forward_id, key=lambda nid: self.get_path_length(self.cur_id, nid))
        else:
            target_id = min(candidate_id, key=lambda nid: self.get_path_length(self.cur_id, nid))
        return target_id

    def get_local_graph(self):
        local_graph_info = {}
        cur_dir = math.atan2(self.cur_pose[1, 0], self.cur_pose[0, 0])
        for node_id, node in self.node_info.items():
            dx = node.pos[0] - self.cur_pos[0]
            dy = node.pos[1] - self.cur_pos[1]
            local_x = math.cos(cur_dir) * dx + math.sin(cur_dir) * dy
            local_y = -math.sin(cur_dir) * dx + math.cos(cur_dir) * dy
            local_x = int(round(local_x / self.res))
            local_y = int(round(local_y / self.res))
            local_graph_info[node_id] = {
                "pos": [float(local_x), float(local_y)],
                "explored": node.explored,
                "neighbors": []
            }
        for edge in self.edge_info:
            u, v = edge.node_id
            if u in local_graph_info and v in local_graph_info:
                local_graph_info[u]["neighbors"].append(v)
                local_graph_info[v]["neighbors"].append(u)
        return local_graph_info

    def get_adj_nodes(self, node_id):
        adj_nodes = set()
        for edge in self.edge_info:
            if node_id in edge.node_id:
                adj_nodes.update(edge.node_id)
        adj_nodes.discard(node_id)
        return list(adj_nodes)

    def get_path(self, start_id, target_id):
        if start_id == target_id:
            return [start_id, target_id]
        graph = defaultdict(list)
        for edge in self.edge_info:
            id1, id2 = edge.node_id
            graph[id1].append(id2)
            graph[id2].append(id1)
        queue = deque([[start_id]])
        visited = set()
        while queue:
            path = queue.popleft()
            cur_node = path[-1]
            if cur_node == target_id:
                return path
            if cur_node not in visited:
                visited.add(cur_node)
                for neighbor in graph[cur_node]:
                    if neighbor not in visited:
                        new_path = list(path)
                        new_path.append(neighbor)
                        queue.append(new_path)
        return None

    def get_path_length(self, start_id, target_id):
        path = self.get_path(start_id, target_id)
        if not path:
            return float('inf')
        length = 0.0
        for i in range(len(path) - 1):
            n1 = self.node_info[path[i]].pos
            n2 = self.node_info[path[i + 1]].pos
            length += self.get_dist(n1, n2)
        return length
    
    # transform point to pixel
    def world_to_vis(self, x, y, world_size = None, offset = None):
        offset_x, offset_y = offset
        x = x + offset_x
        y = y + offset_y
        resolution = self.map_size / world_size
        u = int((x + world_size / 2) * resolution)
        v = self.map_size - int((y + world_size / 2) * resolution)
        u = max(0, min(self.map_size - 1, u))
        v = max(0, min(self.map_size - 1, v))
        return u, v

    def world_to_pixel(pos, top_left_x, top_left_y, resolution, rotate = 0):
        x, y = pos[0], pos[1]
        u = (top_left_x - x) / resolution
        v = (y - top_left_y) / resolution
        if rotate == -90:
            u, v = -v, u
        return [int(round(u)), int(round(v))]

    # set visualize map scale
    def set_map_scale(self):
        if not self.traj_pos:
            return 20, (0, 0)
        px = [p[0] for p in self.traj_pos]
        py = [p[1] for p in self.traj_pos]
        for info in self.node_info.values():
            px.append(info.pos[0])
            py.append(info.pos[1])
        x_min, x_max = min(px), max(px)
        y_min, y_max = min(py), max(py)
        x_range = x_max - x_min
        y_range = y_max - y_min
        world_size = max(max(x_range, y_range) * 1.2, 10)
        x_offset = -(x_min + x_max) / 2
        y_offset = -(y_min + y_max) / 2
        return world_size, (x_offset, y_offset)

    # draw grid
    def draw_grid(self, map_image, resolution, grid_size):
        grid_size = int(grid_size * resolution)
        for x in range(0, self.map_size, grid_size):
            cv2.line(map_image, (x, 0), (x, self.map_size), self.grid_color, 1)
        for y in range(0, self.map_size, grid_size):
            cv2.line(map_image, (0, y), (self.map_size, y), self.grid_color, 1)

    # draw grid
    def draw_axes(self):
        orin_x, orin_y = 20 , self.map_size - 20
        axis_length = 30
        cv2.arrowedLine(self.map_image, (orin_x, orin_y), (orin_x + axis_length, orin_y), (0, 0, 255), 2, tipLength=0.1)
        cv2.arrowedLine(self.map_image, (orin_x, orin_y), (orin_x, orin_y - axis_length), (0, 255, 0), 2, tipLength=0.1)
        cv2.putText(self.map_image, 'X', (orin_x + axis_length + 10, orin_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(self.map_image, 'Y', (orin_x - 10, orin_y - axis_length - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # draw nodes
    def draw_nodes(self, world_size, offset):
        for node_id, node in self.node_info.items():
            if node.type == 'frontier':
                color = self.frontier_color
            elif node.type == 'room':
                color = self.room_color
            elif node.type == 'cross':
                color = self.cross_color
            else:
                color = self.normal_color
            u, v = self.world_to_vis(node.pos[0], node.pos[1], world_size, offset)
            cv2.circle(self.map_image, (u, v), 12, color, -1)
            cv2.circle(self.map_image, (u, v), 12, (0, 0, 0), 2)
            cv2.putText(self.map_image, f"{node_id}", (u - 20, v - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def draw_edges(self, world_size, offset):
        for edge in self.edge_info:
            id1, id2 = edge.node_id
            node1, node2 = self.node_info[id1].pos, self.node_info[id2].pos
            u1, v1 = self.world_to_vis(node1[0], node1[1], world_size, offset)
            u2, v2 = self.world_to_vis(node2[0], node2[1], world_size, offset)
            cv2.line(self.map_image, (u1, v1), (u2, v2), self.edge_color, 1)

    def draw_current_map(self):
        cur_dir = math.atan2(-self.cur_pose[0, 0], self.cur_pose[1, 0])
        world_size, offset = self.set_map_scale()
        resolution = self.map_size / world_size
        self.map_image = np.full((self.map_size, self.map_size, 3), self.background_color, dtype=np.uint8)
        self.draw_grid(self.map_image, resolution, self.grid_size)
        if len(self.traj_pos) > 1:
            traj_pix = []
            for pos in self.traj_pos:
                pix = self.world_to_vis(pos[0], pos[1], world_size, offset)
                traj_pix.append(pix)
            for i in range(1, len(traj_pix)):
                cv2.line(self.map_image, traj_pix[i - 1], traj_pix[i], self.traj_color, 3)
        self.draw_axes()
        self.draw_nodes(world_size, offset)
        self.draw_edges(world_size, offset)
        cur_u, cur_v = self.world_to_vis(self.cur_pos[0], self.cur_pos[1], world_size, offset)
        cv2.circle(self.map_image, (cur_u, cur_v), 8, self.robot_color, -1)
        cv2.circle(self.map_image, (cur_u, cur_v), 8, (0, 0, 0), 2)
        end_u = cur_u - int(30 * math.sin(cur_dir))
        end_v = cur_v - int(30 * math.cos(cur_dir))
        cv2.arrowedLine(self.map_image, (cur_u, cur_v), (end_u, end_v), (0, 0, 0), 3, tipLength = 0.2)
        return self.map_image

    def save_map(self, filename):
        cv2.imwrite(filename, self.map_image)

    def build_clean_graph(self, r=2.0, tol=1.2):
        ids = list(self.node_info.keys())
        pos = np.array([self.node_info[i].pos for i in ids])
        types = [self.node_info[i].type for i in ids]
        fids = [self.node_info[i].frame_id for i in ids]
        rooms = [self.node_info[i].room for i in ids]
        m_pos, m_types, m_fids, m_rooms = [], [], [], []
        cluster_map = {}
        used = np.zeros(len(pos), bool)
        prio = {'room': 1, 'cross': 2}

        def room_priority(rt):
            rt = rt.lower().strip()
            if rt in ('living room', 'dining room', 'hallway', 'corridor'):
                if rt == 'living room':
                    return 2
                elif rt == 'dining room':
                    return 3
                else: 
                    return 4
            else:
                return 1

        for i in range(len(pos)):
            if used[i]:
                continue
            cur_type = types[i]
            dists = np.linalg.norm(pos - pos[i], axis=1)
            close = (dists <= r) & (~used)
            if cur_type == 'room':
                mask = np.array([t == 'room' for t in types])
            else:
                mask = np.array([t != 'room' for t in types])
            close &= mask
            idxs = np.where(close)[0]
            if len(idxs) == 0:
                idxs = np.array([i])
            cluster_fids = [fids[j] for j in idxs]
            cluster_rooms = [rooms[j] for j in idxs] 
            if cur_type == 'room':
                chosen_room = 'room entry'
            else:
                from collections import Counter
                room_counts = Counter(cluster_rooms)
                max_count = max(room_counts.values())
                candidates = [rt for rt, cnt in room_counts.items() if cnt == max_count]
                if len(candidates) == 1:
                    chosen_room = candidates[0]
                else:
                    candidates.sort(key=room_priority)
                    chosen_room = candidates[0]
            imp = [(j, prio.get(types[j], 0)) for j in idxs if types[j] in prio]
            if imp:
                imp.sort(key=lambda x: -x[1])
                best = imp[0][0]
                c_pos = pos[best]
                c_type = types[best]
            else:
                c_pos = np.mean(pos[idxs], axis=0)
                c_type = 'normal'
            cid = len(m_pos)
            m_pos.append(c_pos)
            m_types.append(c_type)
            m_fids.append(cluster_fids)
            m_rooms.append(chosen_room)
            for j in idxs:
                used[j] = True
                cluster_map[ids[j]] = cid

        init_edges = set()
        for e in self.edge_info:
            a, b = e.node_id
            if a in cluster_map and b in cluster_map:
                ca, cb = cluster_map[a], cluster_map[b]
                if ca != cb:
                    init_edges.add(tuple(sorted((ca, cb))))
        init_edges = list(init_edges)
        pts = np.array(m_pos)
        N = len(pts)
        sub_edges = set()
        for i, j in init_edges:
            A, C = pts[i], pts[j]
            vec = C - A
            d = np.linalg.norm(vec)
            if d < 1e-6:
                continue
            cands = []
            for k in range(N):
                if k in (i, j):
                    continue
                B = pts[k]
                t = np.dot(B - A, vec) / (d * d)
                if t <= 0 or t >= 1:
                    continue
                proj = A + t * vec
                if np.linalg.norm(B - proj) <= tol:
                    cands.append((t, k))
            if not cands:
                sub_edges.add((i, j))
            else:
                cands.sort()
                seq = [i] + [k for _, k in cands] + [j]
                for u, v in zip(seq, seq[1:]):
                    sub_edges.add(tuple(sorted((u, v))))
                    
        adj = defaultdict(set)
        for u, v in sub_edges:
            adj[u].add(v)
            adj[v].add(u)
        to_remove = set()
        for i in range(N):
            nb = sorted(adj[i])
            for p in range(len(nb)):
                for q in range(p + 1, len(nb)):
                    j, k = nb[p], nb[q]
                    if k in adj[j]:
                        dij = np.linalg.norm(pts[i] - pts[j])
                        dik = np.linalg.norm(pts[i] - pts[k])
                        djk = np.linalg.norm(pts[j] - pts[k])
                        tri = sorted([(dij, (i, j)), (dik, (i, k)), (djk, (j, k))], reverse=True)
                        to_remove.add(tuple(sorted(tri[0][1])))
        final_edges = [e for e in sub_edges if tuple(sorted(e)) not in to_remove]
        N = len(m_pos)
        adj_list = defaultdict(list)
        for u, v in final_edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        degree = [len(adj_list[i]) for i in range(N)]
        visited = set()

        def get_priority(rt):
            if not isinstance(rt, str):
                return 0
            rt = rt.lower().strip()
            if rt in ('hallway', 'corridor', 'living room', 'dining room'):
                return {'living room': 4, 'dining room': 3, 'corridor': 2, 'hallway': 1}.get(rt, 0)
            else:
                return 5
            
        room_nodes = [i for i in range(N) if m_types[i] == 'room']
        for start in room_nodes:
            if start in visited:
                continue
            path = [start]
            visited.add(start)
            curr = start
            prev = -1
            valid_path = True
            while True:
                next_candidates = [
                    nxt for nxt in adj_list[curr]
                    if nxt != prev and m_types[nxt] == 'normal' and nxt not in visited
                ]
                if len(next_candidates) != 1:
                    break
                nxt = next_candidates[0]
                path.append(nxt)
                visited.add(nxt)
                if degree[nxt] == 1: 
                    break
                if degree[nxt] != 2:  
                    valid_path = False
                    break
                prev, curr = curr, nxt
            if not valid_path or len(path) < 2 or degree[path[-1]] != 1:
                continue
            normal_indices = path[1:]
            rts = []
            for i in normal_indices:
                rt = m_rooms[i]
                if isinstance(rt, str):
                    rt_clean = rt.strip()
                    if rt_clean and rt_clean.lower() != 'unknown':
                        rts.append(rt_clean)
            if not rts:
                continue
            cnt = Counter(rts)
            max_count = max(cnt.values())
            candidates = [rt for rt, c in cnt.items() if c == max_count]
            candidates.sort(key=lambda x: (-get_priority(x), x.lower()))
            best_room_type = candidates[0].title()
            for i in normal_indices:
                m_rooms[i] = best_room_type
        return m_pos, final_edges, m_types, m_rooms, m_fids 

    def visualize_graph(self, m_pos, clean_edges, m_types, m_rooms, save_path, grid_step=3.0):
        plt.figure(figsize=(12, 9))
        RED    = (32/255,  56/255, 102/255)
        CROSS  = (255/255, 0/255,   0/255)
        NORMAL = (134/255, 151/255, 177/255)
        ROOM   = (255/255, 165/255, 0/255)
        for i, j in clean_edges:
            xi, yi = m_pos[i][0], m_pos[i][1]
            xj, yj = m_pos[j][0], m_pos[j][1]
            plt.plot([xi, xj], [yi, yj], 'k-', linewidth=1.5, alpha=0.7, zorder=1)
        type_to_color = {'cross': CROSS, 'room': CROSS, 'normal': NORMAL}
        xs = np.array([p[0] for p in m_pos], float)
        ys = np.array([p[1] for p in m_pos], float)
        colors = [type_to_color.get(t, NORMAL) for t in m_types]
        plt.scatter(xs, ys, c=colors, s=500, edgecolors='k', zorder=5)
        plt.scatter([self.cur_pos[0]], [self.cur_pos[1]], c=[RED], s=800, edgecolors='k', zorder=6)
        cur_dir = math.atan2(-self.cur_pose[0, 0], self.cur_pose[1, 0]) + math.pi / 2
        x0, y0 = self.cur_pos[0], self.cur_pos[1]
        L = 3.0
        dx = L * math.cos(cur_dir)
        dy = L * math.sin(cur_dir)
        plt.arrow(
            x0, y0, dx, dy,
            color=RED,
            width=0.15,
            head_width=0.6,
            head_length=0.6,
            length_includes_head=True,
            zorder=7
        )
        all_x = np.append(xs, self.cur_pos[0])
        all_y = np.append(ys, self.cur_pos[1])
        xmin, xmax = all_x.min(), all_x.max()
        ymin, ymax = all_y.min(), all_y.max()
        pad = grid_step
        xmin, xmax = xmin - pad, xmax + pad
        ymin, ymax = ymin - pad, ymax + pad
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.gca().set_aspect('equal', adjustable='box')
        gx0 = np.floor(xmin / grid_step) * grid_step
        gx1 = np.ceil (xmax / grid_step) * grid_step
        gy0 = np.floor(ymin / grid_step) * grid_step
        gy1 = np.ceil (ymax / grid_step) * grid_step
        plt.xticks(np.arange(gx0, gx1 + 1e-6, grid_step))
        plt.yticks(np.arange(gy0, gy1 + 1e-6, grid_step))
        plt.grid(True, color='0.85', linewidth=1.0)
        plt.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        plt.axis('on')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()