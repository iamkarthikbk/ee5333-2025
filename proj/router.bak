# Detailed router
#
# Author: ee24s053 Karthik B K
# Date: 23 April 2025

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
from rtree import index
import LEFDEFParser
from LEFDEFParser import Point, Rect

class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.count = 0
    
    def empty(self):
        return len(self.elements) == 0
    
    def push(self, item, priority):
        # Count is used as a tie-breaker for items with the same priority
        heapq.heappush(self.elements, (priority, self.count, item))
        self.count += 1
    
    def pop(self):
        return heapq.heappop(self.elements)[2]

def custom_astar_path(G, source, target, heuristic=None, weight='weight'):
    if source not in G or target not in G:
        raise nx.NodeNotFound(f"Either source {source} or target {target} is not in G")
    
    open_set = PriorityQueue()
    open_set.push(source, 0)
    came_from = {source: None}
    g_score = {source: 0}
    f_score = {source: heuristic(source, target)}
    closed_set = set()
    
    while not open_set.empty():
        current = open_set.pop()
        
        if current == target:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        closed_set.add(current)
        
        for neighbor in G.neighbors(current):
            if neighbor in closed_set:
                continue
                
            edge_weight = G[current][neighbor].get(weight, 1)
            tentative_g_score = g_score[current] + edge_weight
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score_val = tentative_g_score + heuristic(neighbor, target)
                
                if neighbor not in f_score or f_score_val < f_score[neighbor]:
                    f_score[neighbor] = f_score_val
                    open_set.push(neighbor, f_score_val)
    
    raise nx.NetworkXNoPath(f"No path between {source} and {target}.")


def custom_astar_path_length(G, source, target, heuristic=None, weight='weight'):
    path = custom_astar_path(G, source, target, heuristic, weight)
    return sum(G[u][v].get(weight, 1) for u, v in zip(path[:-1], path[1:]))

skipCells = {"sky130_fd_sc_hd__decap_3", "sky130_fd_sc_hd__decap_4", "sky130_fd_sc_hd__decap_6", "sky130_fd_sc_hd__decap_8",
            "sky130_fd_sc_hd__decap_12", "sky130_fd_sc_hd__fill_1", "sky130_fd_sc_hd__fill_2", "sky130_fd_sc_hd__fill_4",
            "sky130_fd_sc_hd__fill_8", "sky130_fd_sc_hd__tap_1", "sky130_fd_sc_hd__tap_2", "FILL", "DECAP", "TAPCELL"}

layerColors = { 'li1': 'red', 'met1': 'blue', 'met2': 'magenta', 'met3': 'orange', 'met4': 'green', 'met5': 'cyan' }
layerOrient = { 'li1': 'VERTICAL', 'met1': 'HORIZONTAL', 'met2': 'VERTICAL', 'met3': 'HORIZONTAL', 'met4': 'VERTICAL', 'met5': 'HORIZONTAL' }

skipNets = {'clk', 'VPWR', 'VGND'}

adjLayer = {
  'li1':  ['met1'],
  'met1': ['li1',  'met2'],
  'met2': ['met1', 'met3'],
  'met3': ['met2', 'met4'],
  'met4': ['met3', 'met5'],
  'met5': ['met4']
}

layerWidth = {'li1': 170, 'met1': 140, 'met2': 140, 'met3': 300, 'met4': 300, 'met5': 1600}
layerSpacing = {'li1': 170, 'met1': 140, 'met2': 140, 'met3': 300, 'met4': 300, 'met5': 1600}

class Rect:
    def __init__(self, x1, y1, x2, y2):
        self.ll = LEFDEFParser.Point(x1, y1)
        self.ur = LEFDEFParser.Point(x2, y2)
    
    def width(self):
        return self.ur.x - self.ll.x
        
    def height(self):
        return self.ur.y - self.ll.y
        
    def xcenter(self):
        return (self.ll.x + self.ur.x) // 2
        
    def ycenter(self):
        return (self.ll.y + self.ur.y) // 2
    
    def transform(self, orient, origin, xdim, ydim):
        if orient == 'N':
            pass
        elif orient == 'S':
            self.ll.x, self.ur.x = xdim - self.ur.x, xdim - self.ll.x
            self.ll.y, self.ur.y = ydim - self.ur.y, ydim - self.ll.y
        elif orient == 'W':
            self.ll.x, self.ll.y = self.ll.y, xdim - self.ll.x
            self.ur.x, self.ur.y = self.ur.y, xdim - self.ur.x
        elif orient == 'E':
            self.ll.x, self.ll.y = ydim - self.ll.y, self.ll.x
            self.ur.x, self.ur.y = ydim - self.ur.y, self.ur.x
        elif orient == 'FN':
            self.ll.x, self.ur.x = xdim - self.ur.x, xdim - self.ll.x
        elif orient == 'FS':
            self.ll.y, self.ur.y = ydim - self.ur.y, ydim - self.ll.y
        elif orient == 'FW':
            self.ll.x, self.ll.y = ydim - self.ll.y, xdim - self.ll.x
            self.ur.x, self.ur.y = ydim - self.ur.y, xdim - self.ur.x
        elif orient == 'FE':
            self.ll.x, self.ll.y = self.ll.y, self.ll.x
            self.ur.x, self.ur.y = self.ur.y, self.ur.x
        
        self.ll.x += origin.x
        self.ll.y += origin.y
        self.ur.x += origin.x
        self.ur.y += origin.y

class Inst:
    def __init__(self, inst, macro):
        self._inst = inst
        self._macro = macro
        origin = inst.location()
        self._bbox = Rect(origin.x, origin.y, origin.x + macro.xdim(), origin.y + macro.ydim())
        self._pins = dict()
        self._obsts = dict()
        
        for p in macro.pins():
            shapes = dict()
            for port in p.ports():
                for layer, rects in port.items():
                    if layer not in layerColors: continue
                    if layer not in shapes: shapes[layer] = list()
                    for v in rects:
                        r = Rect(v.ll.x, v.ll.y, v.ur.x, v.ur.y)
                        r.transform(inst.orient(), origin, macro.xdim(), macro.ydim())
                        shapes[layer].append(r)
            self._pins[p.name()] = shapes
        
        if hasattr(macro, 'obstructions'):
            for layer, rects in macro.obstructions().items():
                if layer not in layerColors: continue
                if layer not in self._obsts: self._obsts[layer] = list()
                for v in rects:
                    r = Rect(v.ll.x, v.ll.y, v.ur.x, v.ur.y)
                    r.transform(inst.orient(), origin, macro.xdim(), macro.ydim())
                    self._obsts[layer].append(r)
    
    def getObstacles(self):
        return self._obsts

class Logger:
    DEBUG, INFO, WARNING, ERROR = 10, 20, 30, 40
    LEVEL_NAMES = {DEBUG: "DEBUG", INFO: "INFO", WARNING: "WARNING", ERROR: "ERROR"}
    
    def __init__(self, name, level=INFO): 
        self.name, self.level = name, level
    
    def _log(self, level, msg):
        if level >= self.level: 
            level_name = self.LEVEL_NAMES.get(level, "INFO")
            print(f"[{level_name}] {msg}")
    
    def debug(self, msg): self._log(self.DEBUG, msg)
    def info(self, msg): self._log(self.INFO, msg)
    def warning(self, msg): self._log(self.WARNING, msg)
    def error(self, msg): self._log(self.ERROR, msg)
def setup_logger(name, level=Logger.INFO): return Logger(name, level)
logger = None

# Function to visualize the NetworkX graph. This function was written using AI.
def visualize(G, guide_shapes, pin_shapes, bbox, path=None, rectangles=None, bloated_obstacles=None, net_name=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    ax = plt.figure(figsize=(14, 10)).add_subplot(111)
    plt.subplots_adjust(left=0.05, right=0.85)
    plt.xlim(bbox.ll.x, bbox.ur.x)
    plt.ylim(bbox.ll.y, bbox.ur.y)
    ax.set_aspect('equal')
    
    plt.title(f'Routing for Net: {net_name}')

    # Draw nodes by layer
    node_collections = {}
    for layer in set(n[2] for n in G.nodes()):
        nodes = [n for n in G.nodes() if n[2] == layer]
        pos = {n: (n[0], n[1]) for n in nodes}
        node_collections[f'nodes({layer})'] = nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_size=20, node_color=layerColors[layer], alpha=0.7, label=f'Nodes {layer}')
    # Draw edges
    edge_collections = {}
    wire_edges_by_layer = {}
    via_edges = []
    for u, v, d in G.edges(data=True):
        (via_edges if d.get('type', 'wire') == 'via' else wire_edges_by_layer.setdefault(u[2], [])).append((u, v))
    for layer, edges in wire_edges_by_layer.items():
        pos = {n: (n[0], n[1]) for e in edges for n in e}
        edge_collections[f'wires({layer})'] = nx.draw_networkx_edges(
            G, pos, edgelist=edges, width=1.5, edge_color=layerColors[layer], alpha=0.5, label=f'Wires {layer}')
    if via_edges:
        pos = {n: (n[0], n[1]) for e in via_edges for n in e}
        edge_collections['vias'] = nx.draw_networkx_edges(
            G, pos, edgelist=via_edges, width=2.0, edge_color='black', alpha=0.7,
            style='dotted', label='Vias')
    # Draw path if provided
    if path:
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        pos = {n: (n[0], n[1]) for n in path}
        edge_collections['path'] = nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3.0, edge_color='red', alpha=1.0, label='Routing Path')
        # Highlight pins
        pin_nodes, pin_positions = [], {}
        for pin in pin_shapes:
            px, py, pl = pin['x'], pin['y'], pin['layer']
            closest = min((n for n in G.nodes() if n[2]==pl), key=lambda n: abs(n[0]-px)+abs(n[1]-py), default=None)
            if closest:
                pin_positions[closest] = (closest[0], closest[1])
                pin_nodes.append(closest)
        pin_collections = {}
        if pin_nodes:
            pin_node_collection = nx.draw_networkx_nodes(G, pin_positions, nodelist=pin_nodes, node_size=10, node_color='yellow', edgecolors='black', label='Pins')
            pin_collections['pin_nodes'] = pin_node_collection
            
            # Add pin labels
            pin_labels = nx.draw_networkx_labels(G, pin_positions, labels={n: f'Pin {i+1}' for i, n in enumerate(pin_nodes)}, font_size=8, font_color='black')
            pin_collections['pin_labels'] = pin_labels
    # Guides
    guide_collections = {}
    for layer in set(g['layer'] for g in guide_shapes):
        guides = [g for g in guide_shapes if g['layer']==layer]
        guide_collections[f'guides({layer})'] = [ax.add_patch(plt.Rectangle((g['rect'].ll.x, g['rect'].ll.y), g['rect'].width(), g['rect'].height(), fill=True, alpha=0.1, color=layerColors[layer])) for g in guides]
    # Pins
    pin_collections = {}
    for pin in pin_shapes:
        try:
            layer, rect = pin['layer'], pin['rect']
            if hasattr(rect, 'width') and hasattr(rect, 'height') and hasattr(rect, 'll'):
                pin_collections.setdefault(f'pins({layer})', []).append(
                    ax.add_patch(plt.Rectangle((rect.ll.x, rect.ll.y), rect.width(), rect.height(), fill=True, alpha=0.7, color='red')))
        except Exception as e:
            logger.warning(f"Error drawing pin: {e}")
    # Rectangles
    rect_collections = {}
    if rectangles:
        for layer in set(r[0] for r in rectangles):
            rects = [(x1, y1, x2, y2) for l, x1, y1, x2, y2 in rectangles if l==layer]
            color, alpha = ('black', 0.8) if 'via' in layer else (layerColors.get(layer, 'gray'), 0.6)
            rect_collections[f'rect({layer})'] = [ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, alpha=alpha, color=color, linewidth=1.5)) for x1, y1, x2, y2 in rects]
    # Obstacles
    obstacle_collections = {}
    if bloated_obstacles:
        for layer, obstacles in bloated_obstacles.items():
            if not obstacles:
                continue
            color = layerColors.get(layer, 'darkred')
            patches = []
            for x1, y1, x2, y2 in obstacles:
                patch = ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, alpha=0.4, color=color, linewidth=0))
                patches.append(patch)
            if patches:
                obstacle_collections[f'obstacles({layer})'] = patches
    
    # Checkbox controls
    all_collections = {}
    for d in (node_collections, edge_collections, guide_collections, pin_collections, rect_collections, obstacle_collections):
        all_collections.update(d)
    
    # Set initial visibility - only show obstacles and final rectangles
    for label, collection in all_collections.items():
        is_visible = 'obstacles' in label or 'rect' in label
        if isinstance(collection, list):
            for patch in collection:
                patch.set_visible(is_visible)
        else:
            collection.set_visible(is_visible)
    
    rax = plt.axes([0.87, 0.1, 0.12, 0.8])
    labels = list(all_collections.keys())
    initial_states = [('obstacles' in label or 'rect' in label) for label in labels]
    check = CheckButtons(rax, labels, initial_states)
    
    def func(label):
        col = all_collections[label]
        if isinstance(col, list):
            for patch in col: patch.set_visible(not patch.get_visible())
        else:
            col.set_visible(not col.get_visible())
        plt.draw()
    check.on_clicked(func)
    plt.show()

class GuideRect:
    def __init__(self, x1, y1, x2, y2, layer):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.layer = layer
    def __repr__(self):
        return f"GuideRect({self.x1}, {self.y1}, {self.x2}, {self.y2}, '{self.layer}')"

class GuideParser:
    def __init__(self, guide_file):
        self.guide_file = guide_file
        self.guides = defaultdict(list)

    def parse(self):
        cur, blk = None, False
        with open(self.guide_file) as f:
            for l in map(str.strip, f):
                if not l or l.startswith('#'): continue
                if not blk and l not in ('(',')'): cur = l
                elif l == '(': blk = True
                elif l == ')': blk = False
                elif blk and cur:
                    p = l.split()
                    if len(p)==5:
                        self.guides[cur].append(GuideRect(*p))
        return self.guides

def bloat_rectangle(rect, spacing):
    x1, y1, x2, y2 = rect
    return (x1 - spacing, y1 - spacing, x2 + spacing, y2 + spacing)

def net_to_rects(path, layer_tracks):
    # Special case for single-point paths - create a small square
    if not path:
        return []
    elif len(path) == 1:
        x, y, layer = path[0]
        width = layerWidth.get(layer, 170)
        half_width = width // 2
        return [(layer, x - half_width, y - half_width, x + half_width, y + half_width)]
    
    via_rectangles = []
    for i in range(len(path) - 1):
        p1, p2 = path[i], path[i+1]
        # If same x,y coordinates but different layers, add via connection
        if p1[0] == p2[0] and p1[1] == p2[1] and p1[2] != p2[2]:
            x, y = p1[0], p1[1]
            # Add rectangles in both layers
            for layer in [p1[2], p2[2]]:
                width = layerWidth.get(layer, 170)
                half_width = width // 2
                via_rect = (layer, x - half_width, y - half_width, x + half_width, y + half_width)
                via_rectangles.append(via_rect)
    
    def get_width(layer): return layerWidth[layer]
    
    def process_segments(segments, is_horizontal, width):
        # Group segments by constant coordinate (y for horizontal, x for vertical)
        idx = 1 if is_horizontal else 0
        groups = {}
        for seg in segments:
            key = seg[0][idx]
            groups.setdefault(key, []).append(seg)
            
        results = []
        for coord, segs in groups.items():
            var_idx = 0 if is_horizontal else 1
            segs.sort(key=lambda s: min(s[0][var_idx], s[1][var_idx]))
            
            # Merge adjacent segments
            merged = []
            if not segs: continue
            start = min(segs[0][0][var_idx], segs[0][1][var_idx])
            end = max(segs[0][0][var_idx], segs[0][1][var_idx])
            
            for i in range(1, len(segs)):
                s_start = min(segs[i][0][var_idx], segs[i][1][var_idx])
                s_end = max(segs[i][0][var_idx], segs[i][1][var_idx])
                
                merged.append((start, end, coord))
                start, end = s_start, s_end
            
            merged.append((start, end, coord))
            
            # Create rectangles
            for a, b, c in merged:
                if is_horizontal:  # Horizontal segment: constant y
                    results.append((a, c-width//2, b, c+width//2))
                else:  # Vertical segment: constant x
                    results.append((c-width//2, a, c+width//2, b))
                    
        return results
    
    rectangles = []
    i = 0
    while i < len(path) - 1:
        # Find segment on current layer
        layer = path[i][2]
        j = i + 1
        while j < len(path) and path[j][2] == layer: j += 1
        segment = path[i:j]
        width = get_width(layer)
        
        # Split into horizontal and vertical segments
        h_segs = [(p1, p2) for p1, p2 in zip(segment, segment[1:]) if p1[0] != p2[0]]
        v_segs = [(p1, p2) for p1, p2 in zip(segment, segment[1:]) if p1[0] == p2[0]]
        
        # Process horizontal segments
        for x1, y1, x2, y2 in process_segments(h_segs, True, width):
            rectangles.append((layer, x1, y1, x2, y2))
    
        # Process vertical segments
        for x1, y1, x2, y2 in process_segments(v_segs, False, width):
            rectangles.append((layer, x1, y1, x2, y2))
        
        # Fallback for any unprocessed segments
        if not h_segs and not v_segs and len(segment) > 1:
            for p1, p2 in zip(segment, segment[1:]):
                if p1[0] == p2[0]:  # Vertical
                    x, y1, y2 = p1[0], min(p1[1], p2[1]), max(p1[1], p2[1])
                    rectangles.append((layer, x-width//2, y1, x+width//2, y2))
                else:  # Horizontal
                    y, x1, x2 = p1[1], min(p1[0], p2[0]), max(p1[0], p2[0])
                    rectangles.append((layer, x1, y-width//2, x2, y+width//2))
        i = j
    
    # Combine via rectangles with path rectangles
    return via_rectangles + rectangles

def is_blocked(layer_obs, node):
    x, y, _ = node
    for obs in layer_obs:
        x1, y1, x2, y2 = obs
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False

def is_crossing(layer_obs, edge):
    node1, node2 = edge
    x1, y1, layer = node1
    x2, y2, _ = node2
    
    # If either endpoint is blocked, the edge crosses an obstacle
    if is_blocked(layer_obs, node1) or is_blocked(layer_obs, node2):
        return True

    if x1 == x2:
        edge_x1, edge_y1 = x1, min(y1, y2)  # bottom-left
        edge_x2, edge_y2 = x1, max(y1, y2)  # top-right
    elif y1 == y2:
        edge_x1, edge_y1 = min(x1, x2), y1  # bottom-left
        edge_x2, edge_y2 = max(x1, x2), y1  # top-right
    else:
        logger.error('Diagonal edge!')
        raise SystemExit(1)
    
    for obs in layer_obs:
        ox1, oy1, ox2, oy2 = obs
        obs_x1 = min(ox1, ox2)  # left
        obs_y1 = min(oy1, oy2)  # bottom
        obs_x2 = max(ox1, ox2)  # right
        obs_y2 = max(oy1, oy2)  # top
    
        if not ((edge_x2 <= obs_x1) or (edge_x1 >= obs_x2) or (edge_y2 <= obs_y1) or (edge_y1 >= obs_y2)):
            logger.debug(f"Edge ({x1},{y1})-({x2},{y2}) crosses obstacle ({obs_x1},{obs_y1})-({obs_x2},{obs_y2})")
            return True
    
    return False

# Global counter for R-tree insertions
rtree_count = 0

def create_grid_graph(defr, guides_by_layer, check_obstacles=True, bloated_obstacles=None, net_idx=None, layer_trees=None, ignore_orientation=False):
    G = nx.Graph()
    
    # Track information for each layer
    layer_tracks = {}
    
    # Create tracks within guide rectangles or use entire DEF bounding box
    for layer, orientation in layerOrient.items():
        if layer not in guides_by_layer:
            logger.debug(f"Layer {layer} has no guide rectangles, using entire DEF bounding box")
            guide_rects = [defr.bbox()]
        else:
            guide_rects = guides_by_layer[layer]
        ltracks = defr.tracks()[layer]
        
        # Process tracks based on orientation or create both if ignore_orientation is True
        orientations_to_process = ['VERTICAL', 'HORIZONTAL'] if ignore_orientation else [orientation]
        
        for current_orientation in orientations_to_process:
            if current_orientation == 'VERTICAL':
                for t in ltracks:
                    if t.orient == 'X':  # X-oriented tracks create vertical lines
                        grid_lines = []
                        y_coords = sorted(list({y for rect in guide_rects for y in (rect.ll.y, rect.ur.y)}))
                        
                        for i in range(t.num):
                            x_pos = t.x + i * t.step
                            for rect in guide_rects:
                                if rect.ll.x <= x_pos <= rect.ur.x:
                                    rect_y_coords = [y for y in y_coords if rect.ll.y <= y <= rect.ur.y]
                                    if len(rect_y_coords) >= 2:
                                        min_y, max_y = min(rect_y_coords), max(rect_y_coords)
                                        if max_y - min_y >= t.step:
                                            grid_lines.append((x_pos, min_y, x_pos, max_y))
                        
                        if grid_lines:
                            if layer not in layer_tracks:
                                layer_tracks[layer] = {'orientation': 'VERTICAL', 'lines': grid_lines}
                            else:
                                # If we're processing both orientations, append to existing lines
                                layer_tracks[layer]['lines'].extend(grid_lines)
                                layer_tracks[layer]['orientation'] = 'BOTH' if ignore_orientation else 'VERTICAL'
                            logger.debug(f"Created {len(grid_lines)} vertical grid lines for layer {layer}")
                        break
            elif current_orientation == 'HORIZONTAL':
                for t in ltracks:
                    if t.orient == 'Y':  # Y-oriented tracks create horizontal lines
                        grid_lines = []
                        x_coords = sorted(list({x for rect in guide_rects for x in (rect.ll.x, rect.ur.x)}))
                        
                        for i in range(t.num):
                            y_pos = t.x + i * t.step
                            for rect in guide_rects:
                                if rect.ll.y <= y_pos <= rect.ur.y:
                                    rect_x_coords = [x for x in x_coords if rect.ll.x <= x <= rect.ur.x]
                                    if len(rect_x_coords) >= 2:
                                        min_x, max_x = min(rect_x_coords), max(rect_x_coords)
                                        if max_x - min_x >= t.step:
                                            grid_lines.append((min_x, y_pos, max_x, y_pos))
                        
                        if grid_lines:
                            if layer not in layer_tracks:
                                layer_tracks[layer] = {'orientation': 'HORIZONTAL', 'lines': grid_lines}
                            else:
                                # If we're processing both orientations, append to existing lines
                                layer_tracks[layer]['lines'].extend(grid_lines)
                                layer_tracks[layer]['orientation'] = 'BOTH' if ignore_orientation else 'HORIZONTAL'
                            logger.debug(f"Created {len(grid_lines)} horizontal grid lines for layer {layer}")
                        break
    
    # Find via locations (intersections between adjacent layers)
    via_locations = {}
    valid_via_nodes = {}
    
    for layer1 in layerOrient.keys():
        if layer1 in layer_tracks:
            for layer2 in adjLayer[layer1]:
                # Check if layers can connect via vias
                # Either orientations are different or one/both have 'BOTH' orientation
                can_connect = False
                if layer2 in layer_tracks:
                    orient1 = layer_tracks[layer1]['orientation']
                    orient2 = layer_tracks[layer2]['orientation']
                    
                    # Can connect if orientations are different or if either has 'BOTH' orientation
                    if orient1 == 'BOTH' or orient2 == 'BOTH' or orient1 != orient2:
                        can_connect = True
                
                if layer2 in layer_tracks and can_connect:
                    via_key = f"{layer1}_{layer2}"
                    via_locations[via_key] = []
                    valid_via_nodes[via_key] = []
                    
                    # Find all intersection points
                    for line1 in layer_tracks[layer1]['lines']:
                        for line2 in layer_tracks[layer2]['lines']:
                            # Determine intersection point based on orientation
                            orient1 = layer_tracks[layer1]['orientation']
                            orient2 = layer_tracks[layer2]['orientation']
                            
                            # Handle all orientation combinations
                            intersection = None
                            
                            # Case 1: One vertical, one horizontal (traditional case)
                            if (orient1 == 'VERTICAL' and orient2 == 'HORIZONTAL'):
                                x, y = line1[0], line2[1]
                                intersection = (x, y)
                            elif (orient1 == 'HORIZONTAL' and orient2 == 'VERTICAL'):
                                x, y = line2[0], line1[1]
                                intersection = (x, y)
                            
                            # Case 2: One or both are 'BOTH' orientation
                            elif orient1 == 'BOTH' or orient2 == 'BOTH':
                                # Determine if line1 is vertical (x1=x2) or horizontal (y1=y2)
                                line1_is_vertical = line1[0] == line1[2]
                                line2_is_vertical = line2[0] == line2[2]
                                
                                # Only create intersection if lines have different orientations
                                if line1_is_vertical != line2_is_vertical:
                                    if line1_is_vertical:  # line1 is vertical, line2 is horizontal
                                        x, y = line1[0], line2[1]
                                    else:  # line1 is horizontal, line2 is vertical
                                        x, y = line2[0], line1[1]
                                    intersection = (x, y)
                            
                            # Skip if no valid intersection found
                            if intersection is None:
                                continue
                            
                            # Add via nodes and connection only if they don't cross obstacles
                            x, y = intersection
                            node1, node2 = (x, y, layer1), (x, y, layer2)
                            
                            # Check if nodes are blocked by obstacles in current or adjacent layers
                            node1_blocked = False
                            node2_blocked = False
                            
                            if check_obstacles:
                                # Create a small rectangle around the node for checking
                                x, y = node1[0], node1[1]
                                width1 = layerWidth.get(layer1, 170)
                                width2 = layerWidth.get(layer2, 170)
                                
                                # Rectangle for node1
                                rect1 = (x - width1//2, y - width1//2, x + width1//2, y + width1//2)
                                # Rectangle for node2
                                rect2 = (x - width2//2, y - width2//2, x + width2//2, y + width2//2)
                                
                                # Check current layer using bloated obstacles for DRC checking
                                if layer1 in bloated_obstacles and is_blocked(bloated_obstacles[layer1], node1):
                                    node1_blocked = True
                                
                                # Check adjacent layers using layer trees directly (no bloating)
                                for adj_layer in adjLayer.get(layer1, []):
                                    if adj_layer in layer_trees:
                                        hits = list(layer_trees[adj_layer].intersection(rect1, objects=True))
                                        if any(hit.object != net_idx for hit in hits):
                                            node1_blocked = True
                                            break
                                
                                # Check current layer using bloated obstacles for DRC checking
                                if layer2 in bloated_obstacles and is_blocked(bloated_obstacles[layer2], node2):
                                    node2_blocked = True
                                
                                # Check adjacent layers using layer trees directly (no bloating)
                                for adj_layer in adjLayer.get(layer2, []):
                                    if adj_layer in layer_trees:
                                        hits = list(layer_trees[adj_layer].intersection(rect2, objects=True))
                                        if any(hit.object != net_idx for hit in hits):
                                            node2_blocked = True
                                            break
                            
                            # Only add valid via locations
                            if not node1_blocked and not node2_blocked:
                                via_locations[via_key].append(intersection)
                                valid_via_nodes[via_key].append((node1, node2))
                                
                                # Add nodes to graph
                                G.add_node(node1, pos=(x, y), layer=layer1)
                                G.add_node(node2, pos=(x, y), layer=layer2)
                                
                                # Add via edge
                                G.add_edge(node1, node2, weight=500, type='via')  # Higher cost for layer transitions
                    
                    logger.debug(f"Found {len(via_locations[via_key])} valid via locations between {layer1} and {layer2}")
    
    # Process each layer for additional connections between via locations
    for layer in {node[2] for node in G.nodes()}:
        # Group nodes by coordinates
        points_by_x = defaultdict(list)
        points_by_y = defaultdict(list)
        
        for point in [n for n in G.nodes() if n[2] == layer]:
            x, y, _ = point
            points_by_x[x].append(point)
            points_by_y[y].append(point)
        
        # Make vertical connections on VERTICAL layers
        if layerOrient.get(layer) == 'VERTICAL':
            vertical_count = 0
            for x, points in points_by_x.items():
                points.sort(key=lambda p: p[1])  # Sort by y-coordinate
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i+1]
                    # Check if the edge crosses any obstacle in current or adjacent layers
                    edge_is_valid = True
                    
                    if check_obstacles:
                        # Create wire rectangle for checking
                        if p1[0] == p2[0]:  # Vertical wire
                            wire_x1 = p1[0] - layerWidth[layer]//2
                            wire_y1 = min(p1[1], p2[1])
                            wire_x2 = p1[0] + layerWidth[layer]//2
                            wire_y2 = max(p1[1], p2[1])
                        else:  # Horizontal wire
                            wire_x1 = min(p1[0], p2[0])
                            wire_y1 = p1[1] - layerWidth[layer]//2
                            wire_x2 = max(p1[0], p2[0])
                            wire_y2 = p1[1] + layerWidth[layer]//2
                        
                        # Check current layer using bloated obstacles for DRC checking
                        if layer in bloated_obstacles and is_crossing(bloated_obstacles[layer], (p1, p2)):
                            edge_is_valid = False
                        
                        # Check adjacent layers using layer trees directly (no bloating)
                        for adj_layer in adjLayer.get(layer, []):
                            if adj_layer in layer_trees:
                                # Query the adjacent layer tree
                                hits = list(layer_trees[adj_layer].intersection((wire_x1, wire_y1, wire_x2, wire_y2), objects=True))
                                if any(hit.object != net_idx for hit in hits):
                                    edge_is_valid = False
                                    break
                    
                    if edge_is_valid:
                        weight = abs(p1[1] - p2[1])
                        G.add_edge(p1, p2, weight=weight, type='wire')
                        vertical_count += 1
            logger.debug(f"Made {vertical_count} vertical connections on layer {layer}")
        else:
            logger.debug(f"Skipping vertical connections on layer {layer} (not a VERTICAL layer)")

        # Make horizontal connections on HORIZONTAL layers
        if layerOrient.get(layer) == 'HORIZONTAL':
            horizontal_count = 0
            for y, points in points_by_y.items():
                points.sort(key=lambda p: p[0])  # Sort by x-coordinate
                for i in range(len(points) - 1):
                    p1, p2 = points[i], points[i+1]
                    # Check if the edge crosses any obstacle in current or adjacent layers
                    edge_is_valid = True
                    
                    if check_obstacles:
                        # Create wire rectangle for checking
                        if p1[0] == p2[0]:  # Vertical wire
                            wire_x1 = p1[0] - layerWidth[layer]//2
                            wire_y1 = min(p1[1], p2[1])
                            wire_x2 = p1[0] + layerWidth[layer]//2
                            wire_y2 = max(p1[1], p2[1])
                        else:  # Horizontal wire
                            wire_x1 = min(p1[0], p2[0])
                            wire_y1 = p1[1] - layerWidth[layer]//2
                            wire_x2 = max(p1[0], p2[0])
                            wire_y2 = p1[1] + layerWidth[layer]//2
                        
                        # Check current layer using bloated obstacles for DRC checking
                        if layer in bloated_obstacles and is_crossing(bloated_obstacles[layer], (p1, p2)):
                            edge_is_valid = False
                        
                        # Check adjacent layers using layer trees directly (no bloating)
                        for adj_layer in adjLayer.get(layer, []):
                            if adj_layer in layer_trees:
                                # Query the adjacent layer tree
                                hits = list(layer_trees[adj_layer].intersection((wire_x1, wire_y1, wire_x2, wire_y2), objects=True))
                                if any(hit.object != net_idx for hit in hits):
                                    edge_is_valid = False
                                    break
                    
                    if edge_is_valid:
                        weight = abs(p1[0] - p2[0])
                        G.add_edge(p1, p2, weight=weight, type='wire')
                        horizontal_count += 1
            logger.debug(f"Made {horizontal_count} horizontal connections on layer {layer}")
        else:
            logger.debug(f"Skipping horizontal connections on layer {layer} (not a HORIZONTAL layer)")
    
    return G, layer_tracks


def create_direct_path(source_node, target_node):
    # Create a simple path that will always work
    direct_path = []
    
    # Start with source node
    direct_path.append(source_node)
    
    # If source and target are on different layers, create a simple via stack
    if source_node[2] != target_node[2]:
        # First move horizontally to target's x-coordinate on source layer
        if source_node[0] != target_node[0]:
            direct_path.append((target_node[0], source_node[1], source_node[2]))
        
        # Then move vertically to target's y-coordinate on source layer
        if source_node[1] != target_node[1]:
            direct_path.append((target_node[0], target_node[1], source_node[2]))
        
        # Get all layers in order
        all_layers = list(layerOrient.keys())
        if source_node[2] in all_layers and target_node[2] in all_layers:
            source_idx = all_layers.index(source_node[2])
            target_idx = all_layers.index(target_node[2])
            
            # Create a direct via stack at target's x,y coordinates
            if source_idx < target_idx:
                for i in range(source_idx + 1, target_idx + 1):
                    direct_path.append((target_node[0], target_node[1], all_layers[i]))
            else:
                for i in range(source_idx - 1, target_idx - 1, -1):
                    direct_path.append((target_node[0], target_node[1], all_layers[i]))
    else:
        # If on the same layer, just move horizontally then vertically
        if source_node[0] != target_node[0]:
            direct_path.append((target_node[0], source_node[1], source_node[2]))
        if source_node[1] != target_node[1]:
            direct_path.append((target_node[0], target_node[1], source_node[2]))
    
    # End with target node if not already there
    if direct_path[-1] != target_node:
        direct_path.append(target_node)
    
    return direct_path


def detailed_route(input_DEF, input_LEF, input_GUIDE, output_DEF, show_visualization=False):
    global rtree_count
    logger.info(f"Starting detailed routing")
    logger.info(f"Input DEF: {input_DEF}")
    logger.info(f"Input LEF: {input_LEF}")
    logger.info(f"Input GUIDE: {input_GUIDE}")
    logger.info(f"Output DEF: {output_DEF}")
    
    # Parse lef, def, guide inputs
    lefr = LEFDEFParser.LEFReader()
    lefr.readLEF(input_LEF)
    logger.info("LEF file parsed successfully")
    
    defr = LEFDEFParser.DEFReader()
    defr.readDEF(input_DEF)
    logger.info("DEF file parsed successfully")
    
    guide_parser = GuideParser(input_GUIDE)
    guides = guide_parser.parse()
    logger.info(f"GUIDE file parsed successfully, found {len(guides)} nets")

    # Step 1:   Collect all nets from the DEF reader using the nets() method.
    #           Sort them by Half Perimeter Wire Length
    nets = defr.nets()
    logger.info(f"Found {len(nets)} nets in DEF file")
    
    net_hpwls = []
    for net in nets:
        # Get all pin locations for this net
        pins = []
        for pin_pair in net.pins():
            comp_name, pin_name = pin_pair
            
            # Handle boundary pins
            if comp_name == "PIN":
                # Find the pin in the DEF file's pins
                for p in defr.pins():
                    if p.name() == pin_name:
                        # Process pin ports and rectangles
                        for port in p.ports():
                            for layer, rects in port.items():
                                for r in rects:
                                    # Use rectangle center for pin location
                                    pins.append((
                                        (r.ll.x + r.ur.x) // 2,
                                        (r.ll.y + r.ur.y) // 2
                                    ))
            else:
                for comp in defr.components():
                    if comp.name() == comp_name:
                        # Get component location and orientation
                        comp_x, comp_y = comp.location().x, comp.location().y
                        comp_orient = comp.orient()
                        
                        # Get the macro for this component
                        macro_name = comp.macro()
                        macro = next((m for m in lefr.macros() if m.name() == macro_name), None)
                        
                        if macro:
                            # Find the pin in the macro
                            for pin in macro.pins():
                                if pin.name() == pin_name:
                                    # Process pin ports and rectangles
                                    for port in pin.ports():
                                        for layer, rects in port.items():
                                            for r in rects:
                                                # Transform rectangle based on component orientation
                                                rect = Rect(r.ll.x, r.ll.y, r.ur.x, r.ur.y)
                                                rect.transform(comp_orient, Point(comp_x, comp_y), macro.xdim(), macro.ydim())
                                                
                                                # Use rectangle center for pin location
                                                pins.append((
                                                    (rect.ll.x + rect.ur.x) // 2,
                                                    (rect.ll.y + rect.ur.y) // 2
                                                ))
        
        # Calculate HPWL if we have at least 2 pins
        hpwl = 0
        min_x = min(p[0] for p in pins)
        max_x = max(p[0] for p in pins)
        min_y = min(p[1] for p in pins)
        max_y = max(p[1] for p in pins)
        hpwl = (max_x - min_x) + (max_y - min_y)
        
        net_hpwls.append((net, hpwl))
    
    # Sort nets by HPWL
    net_hpwls.sort(key=lambda x: x[1])
    nets = [n[0] for n in net_hpwls]
    logger.debug(f"Sorted {len(nets)} nets by HPWL")
    
    # Step 2: Build R-tree that can be queried later for obstacles
    logger.debug("Building spatial indices for pins and obstacles...")
    
    for layer in lefr.layers():
        layerWidth[layer.name()] = layer.width()
        layerSpacing[layer.name()] = layer.pitch() - layer.width()
    pins = {}
    for p in defr.pins():
        pin_name = p.name()
        pins[pin_name] = {}
        for port in p.ports():
            for layer, rects in port.items():
                if layer not in layerColors: continue
                if layer not in pins[pin_name]: pins[pin_name][layer] = []
                for r in rects:
                    pins[pin_name][layer].append(Rect(r.ll.x, r.ll.y, r.ur.x, r.ur.y))
    lefDict = {m.name() : m for m in lefr.macros()}
    insts = {inst.name(): Inst(inst, lefDict[inst.macro()]) 
             for inst in defr.components() 
             if inst.macro() not in skipCells}  # Skip filler cells
    
    class Net:
        def __init__(self, net, insts, pins, idx):
            self._name = net.name()
            self._pins = {}
            self._id = idx  # unique ID used to identify nets from rtree
            self._sol = net.rects()  # solution shapes read from the output def file
            for p in net.pins():
                if p[0] in insts:
                    self._pins[p] = insts[p[0]]._pins[p[1]]  # copy shapes from the transformed instance pins
                elif p[0] == 'PIN' and p[1] in pins:
                    self._pins[p] = pins[p[1]]  # copy shapes from the boundary pins
                    
        def has_pin(self, comp_name, pin_name):
            return (comp_name, pin_name) in self._pins
    
    # Build nets list using Net class
    logger.debug("Building nets list...")
    net_list = []
    idx = 0
    for net in nets:
        if net.name() not in skipNets:
            net_list.append(Net(net, insts, pins, idx))
            idx += 1
    
    def buildTree(nets, insts):
        import rtree.index
        global rtree_count
        lT = {layer: rtree.index.Index() for layer in layerColors}
        obstid = len(nets)
        
        for inst in insts.values():
            for layer, rects in inst._obsts.items():
                for r in rects:
                    lT[layer].insert(rtree_count, (r.ll.x, r.ll.y, r.ur.x, r.ur.y), obj=obstid)
                    rtree_count += 1
        
        for net in nets:
            for layer, rects in net._sol.items():
                for r in rects:
                    lT[layer].insert(rtree_count, (r.ll.x, r.ll.y, r.ur.x, r.ur.y), obj=net._id)
                    rtree_count += 1
            
            for p, lr in net._pins.items():
                for layer, rects in lr.items():
                    for r in rects:
                        lT[layer].insert(rtree_count, (r.ll.x, r.ll.y, r.ur.x, r.ur.y), obj=net._id)
                        rtree_count += 1
        
        return lT
    layer_trees = buildTree(net_list, insts)
    
    # Step 3:   Process each net for routing
    #           First we create a grid graph that is obstacle aware
    #           Then we route nets using astar, sequentially
    #           Then convert those nets into rectangles and add them to r-tree as obs for subsequent nets
    logger.debug("Processing nets for routing...")
    for net_idx, net in enumerate(net_list):
        net_name = net._name
        logger.debug(f"Net {net_idx+1}/{len(net_list)}: {net_name}")

        # Skip nets that contain any name from the skipNets list
        should_skip = False
        for skip_name in skipNets:
            if skip_name in net_name.lower():
                should_skip = True
                logger.debug(f'Skipping net {net_name} (contains "{skip_name}")')
                break
        if should_skip:
            continue
            
        # Process pins for this net using the Net class structure
        logger.debug(f"Processing pins for net {net_name}")
        
        # Collect all pin shapes for this net
        pin_shapes = []
        for pin_pair, shapes in net._pins.items():
            comp_name, pin_name = pin_pair
            
            # Process each layer's shapes
            for layer, rects in shapes.items():
                for rect in rects:
                    pin_shapes.append({
                        'comp_name': comp_name,
                        'pin_name': pin_name,
                        'layer': layer,
                        'rect': rect,
                        'x': rect.xcenter(),
                        'y': rect.ycenter()
                    })
                    logger.debug(f"Found {comp_name}/{pin_name} pin shape at ({rect.xcenter()}, {rect.ycenter()}) on layer {layer}")
        
        # If we don't have at least 2 pins, we're in trouble. Parsing is messed up.
        if len(pin_shapes) < 2:
            logger.error(f"Net {net_name} has fewer than 2 pins!")
            raise SystemExit(1)

        # Extract guide information for this net
        guide_shapes = []
        if net._name in guide_parser.guides:
            net_guides = guide_parser.guides[net._name]
            logger.debug(f"Found {len(net_guides)} guide rectangles for net {net._name}")
            
            for guide in net_guides:
                layer = guide.layer
                rect = Rect(guide.x1, guide.y1, guide.x2, guide.y2)
                guide_shapes.append({
                    'layer': layer,
                    'rect': rect
                })
                logger.debug(f"Guide rectangle on layer {layer}: ({guide.x1}, {guide.y1}) - ({guide.x2}, {guide.y2})")
        else:
            logger.error(f"No guide information found for net {net._name}")
            raise SystemExit(1)
            
        logger.debug(f"Creating grid graph for net {net._name}")
        
        # Extract track information for each layer
        layer_tracks = {}
        bbox = defr.bbox()
        
        # Filter guide shapes by layer
        guides_by_layer = {}
        for guide in guide_shapes:
            layer = guide['layer']
            if layer not in guides_by_layer:
                guides_by_layer[layer] = []
            guides_by_layer[layer].append(guide['rect'])

        G = nx.Graph()
        fallback_G = nx.Graph()
        using_fallback = False
        fallback_source = None
        fallback_target = None
        
        # Create bloated obstacle rectangles for each layer
        bloated_obstacles = {}
        for layer in layerOrient.keys():
            if layer not in layer_trees:
                continue
                
            # Get the layer-specific spacing for bloating
            spacing = layerSpacing[layer]
                
            # Get all obstacles for this layer
            bloated_obstacles[layer] = []
            
            # Convert the bbox to a tuple for rtree intersection
            bbox = defr.bbox()
            bbox_tuple = (bbox.ll.x, bbox.ll.y, bbox.ur.x, bbox.ur.y)
            
            for item in layer_trees[layer].intersection(bbox_tuple, objects=True):
                if item.object == net_idx:
                    continue  # Skip obstacles from current net
                    
                # # Get original coordinates from the R-tree
                ox1, oy1, ox2, oy2 = item.bbox
                
                # Bloat the normalized obstacle rectangle by the layer spacing
                bloated_rect = bloat_rectangle((ox1, oy1, ox2, oy2), spacing)
                bloated_obstacles[layer].append(bloated_rect)
        
        # Create the routing grid graph with obstacle checking
        G, layer_tracks = create_grid_graph(defr, guides_by_layer, check_obstacles=True, 
                                           bloated_obstacles=bloated_obstacles, net_idx=net_idx, 
                                           layer_trees=layer_trees, ignore_orientation=False)
        
        # A* routing between pins
        logger.debug(f"Finding path between pins for net {net_name}")
        
        # Manhattan distance + layer transition + spacing violation penalty heuristic
        def heuristic(u, v):
            # Base Manhattan distance
            manhattan_dist = abs(u[0]-v[0]) + abs(u[1]-v[1])
            
            # Layer transition penalty
            layer_penalty = 500 if u[2]!=v[2] else 0
            
            # Spacing violation penalty
            spacing_penalty = 0
            if u[2] == v[2]:  # Only check spacing on the same layer
                layer = u[2]
                if layer in layer_trees and layer in layerSpacing:
                    # Create a rectangle representing the path segment
                    if u[0] == v[0]:  # Vertical segment
                        x1, y1 = u[0], min(u[1], v[1])
                        x2, y2 = u[0], max(u[1], v[1])
                    else:  # Horizontal segment
                        x1, y1 = min(u[0], v[0]), u[1]
                        x2, y2 = max(u[0], v[0]), u[1]
                    
                    # Bloat the rectangle by the spacing requirement
                    spacing = layerSpacing[layer] // 2
                    rect = (x1, y1, x2, y2)
                    bloated_rect = bloat_rectangle(rect, spacing)
                    
                    # Check for potential spacing violations
                    hits = list(layer_trees[layer].intersection(bloated_rect, objects=True))
                    for hit in hits:
                        if hit.object != net_idx:  # Skip current net's objects
                            spacing_penalty += 20000  # High penalty for potential spacing violations
            
            return manhattan_dist + layer_penalty + spacing_penalty
        all_paths, all_rectangles = [], []
        source_pin = pin_shapes[0]
        routed_pins = {(source_pin['comp_name'], source_pin['pin_name'])}
        
        # Sort pins by nearest-neighbor for multi-pin nets
        if len(pin_shapes) > 2:
            logger.debug(f"Net {net_name} has {len(pin_shapes)} pins, using sequential routing")
            # Greedy nearest-neighbor sorting
            sorted_pins, current_pin = [], source_pin
            remaining_pins = pin_shapes[1:]
            while remaining_pins:
                closest_pin = min(remaining_pins, key=lambda p: abs(p['x']-current_pin['x'])+abs(p['y']-current_pin['y']))
                sorted_pins.append(closest_pin)
                remaining_pins.remove(closest_pin)
                current_pin = closest_pin
            pin_shapes = [source_pin] + sorted_pins
        
        # Route to each remaining pin sequentially
        for i in range(1, len(pin_shapes)):
            target_pin = pin_shapes[i]
            source_node = target_node = None
            min_source_dist = min_target_dist = float('inf')
            
            # Find source node - either from existing path or closest grid point
            if all_paths:
                # Use existing path points as potential sources
                all_path_points = set(point for path in all_paths for point in path)
                target_layer = target_pin['layer']
                
                for node in all_path_points:
                    dist = abs(node[0]-target_pin['x']) + abs(node[1]-target_pin['y'])
                    if node[2] == target_layer: dist -= 500  # Same layer preference
                    if dist < min_source_dist:
                        min_source_dist, source_node = dist, node
            else:
                # First pin - find closest grid point
                source_layer = source_pin['layer']
                source_node = min([n for n in G.nodes() if n[2] == source_layer], 
                                  key=lambda n: abs(n[0]-source_pin['x'])+abs(n[1]-source_pin['y']), default=None)
            
            # Find closest grid point to target pin
            target_layer = target_pin['layer']
            target_node = min([n for n in G.nodes() if n[2] == target_layer], 
                              key=lambda n: abs(n[0]-target_pin['x'])+abs(n[1]-target_pin['y']), default=None)
            
            # Route between source and target nodes
            if source_node and target_node:
                logger.debug(f"		Routing from {source_node} to {target_pin['comp_name']}/{target_pin['pin_name']} at {target_node}")
                try:
                    path = custom_astar_path(G, source_node, target_node, heuristic, weight='weight')
                    path_length = custom_astar_path_length(G, source_node, target_node, heuristic, weight='weight')
                    logger.debug(f"Found path with length {path_length}")
                    all_paths.append(path)
                    rectangles = net_to_rects(path, defr.tracks())
                    all_rectangles.extend(rectangles)
                    routed_pins.add((target_pin['comp_name'], target_pin['pin_name']))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    logger.error(f"No path found between {source_node} and {target_node} for net {net_name}")
                    
                    # Fallback: Create an obstacle-unaware grid graph
                    logger.debug(f"Attempting fallback routing for net {net_name} (may violate DRCs)")
                    using_fallback = True
                    
                    # Set fallback source and target to the same nodes as the main routing
                    fallback_source = source_node
                    fallback_target = target_node
                    
                    # Create the fallback routing grid graph without obstacle checking
                    # This allows for more direct paths when normal routing fails
                    fallback_G, _ = create_grid_graph(defr, guides_by_layer, check_obstacles=True, 
                                                     bloated_obstacles=bloated_obstacles, net_idx=net_idx, 
                                                     layer_trees=layer_trees, ignore_orientation=True)
                    
                    try:
                        # Try to find a path using the fallback graph
                        fallback_path = custom_astar_path(fallback_G, fallback_source, fallback_target, heuristic, weight='weight')
                        path_length = custom_astar_path_length(fallback_G, fallback_source, fallback_target, heuristic, weight='weight')
                        logger.info(f"Fallback path found with length {path_length}")
                        all_paths.append(fallback_path)
                        all_rectangles.extend(net_to_rects(fallback_path, defr.tracks()))
                        routed_pins.add((target_pin['comp_name'], target_pin['pin_name']))
                    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                        logger.warning(f"A* fallback routing failed for net {net_name}: {str(e)}")
                        
                        # Create a direct via path through layers if needed
                        logger.warning(f"Creating direct via path between {source_node} and {target_node}")
                        
                        # Create a direct path that will always work
                        direct_path = create_direct_path(source_node, target_node)
                        
                        logger.warning(f"Created direct path with {len(direct_path)} nodes: {direct_path}")
                        
                        try:
                            # Try to create rectangles from the direct path
                            path_rects = net_to_rects(direct_path, defr.tracks())
                            all_paths.append(direct_path)
                            all_rectangles.extend(path_rects)
                            routed_pins.add((target_pin['comp_name'], target_pin['pin_name']))
                            logger.info(f"Created {len(path_rects)} rectangles from direct path")
                        except Exception as rect_error:
                            # If rectangle creation fails, create a single rectangle connecting source and target
                            logger.warning(f"Failed to create rectangles from direct path: {str(rect_error)}")
                            
                            # Create a single rectangle from source to target on the source layer
                            x1, y1, layer1 = source_node
                            x2, y2, _ = target_node
                            
                            # Create a rectangle with the appropriate width for the layer
                            width = layerWidth.get(layer1, 170)
                            
                            # Create a simple rectangle
                            if x1 == x2:  # Vertical
                                rect = LEFDEFParser.Rect(LEFDEFParser.Point(x1 - width//2, min(y1, y2)),
                                                        LEFDEFParser.Point(x1 + width//2, max(y1, y2)))
                            else:  # Horizontal
                                rect = LEFDEFParser.Rect(LEFDEFParser.Point(min(x1, x2), y1 - width//2),
                                                        LEFDEFParser.Point(max(x1, x2), y1 + width//2))
                            
                            all_rectangles.append((layer1, rect))
                            
                        routed_pins.add((target_pin['comp_name'], target_pin['pin_name']))
            else:
                logger.error(f"Could not find grid points close to pins for net {net_name}")
        
        # Process routing results
        if all_rectangles:
            logger.info(f'Done routing net {net_name} with {len(all_rectangles)} rectangles')
            
            # Find the corresponding net in the DEF file to add rectangles
            def_net = None
            for n in defr.nets():
                if n.name() == net_name:
                    def_net = n
                    break
            
            if def_net:
                # Process all rectangles by layer
                for rect in all_rectangles:
                    layer, x1, y1, x2, y2 = rect
                    if 'via_' in layer: continue  # Skip via segments
                    
                    min_width = layerWidth.get(layer, 170)
                    is_vertical = x1 == x2 or abs(x2 - x1) < abs(y2 - y1)
                    
                    # Create rectangle based on orientation
                    if is_vertical:
                        rect_coords = (x1, y1, x1 + min_width, y2)  # Vertical
                    else:
                        rect_coords = (x1, y1, x2, y1 + min_width)  # Horizontal
                    
                    # Add to DEF file
                    def_net.addRect(layer, *rect_coords)
                    
                    # Add to layer tree for obstacle avoidance in subsequent nets
                    if layer in layer_trees:
                        layer_trees[layer].insert(rtree_count, rect_coords, obj=net_idx)
                        rtree_count += 1
            else:
                logger.error(f"Could not find net {net_name} in DEF file")
            
            # Visualize all paths and rectangles
            # Flatten all paths into a single list for visualization
            all_points = [point for path in all_paths for point in path]
            logger.debug(f"Displaying visualization for net {net_name}. Close the window to continue...")
            try:
                # Debug the layer_trees structure
                logger.debug(f"layer_trees keys: {list(layer_trees.keys()) if layer_trees else 'None'}")
                if layer_trees:
                    for layer, rtree in layer_trees.items():
                        try:
                            # Check if rtree is empty
                            count = sum(1 for _ in rtree.intersection((-float('inf'), -float('inf'), float('inf'), float('inf'))))
                            logger.debug(f"Layer {layer} has {count} obstacles")

                        except Exception as e:
                            logger.error(f"Error inspecting rtree for layer {layer}: {str(e)}")
                
                # Only show visualization if requested
                if show_visualization:
                    visualize(G if not using_fallback else fallback_G, guide_shapes, pin_shapes, defr.bbox(), all_points, all_rectangles, bloated_obstacles, net_name)
                    logger.debug(f"Visualization window closed for net {net_name}")
            except Exception as e:
                logger.error(f"Error displaying visualization: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
        else:
            logger.error(f"No routing solution found for net {net_name}")

        # Print pin information
        if pin_shapes:
            logger.debug(f"Net {net_name} has {len(pin_shapes)} pins:")
            for pin in pin_shapes:
                logger.debug(f"  {pin['comp_name']}/{pin['pin_name']} at ({pin['x']}, {pin['y']}) on layer {pin['layer']}")
        else:
            logger.error(f"No pins found for net {net_name}, skipping")
            raise SystemExit(1)
    
    # Write the updated DEF file with routing information
    logger.info(f"Writing output DEF file: {output_DEF}")
    defr.writeDEF(output_DEF)
    
    logger.info(f"Detailed routing completed")
    logger.info(f"Output written to: {output_DEF}")

def init_logger(verbose=False):
    global logger
    logger = setup_logger('detailed_router', Logger.DEBUG if verbose else Logger.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detailed Router CLI")
    parser.add_argument('-l', '--leffile', type=str, required=True, help='Input LEF file')
    parser.add_argument('-d', '--deffile', type=str, required=True, help='Input DEF file')
    parser.add_argument('-g', '--guide', type=str, required=True, help='Input GUIDE file')
    parser.add_argument('-o', '--output', type=str, help='Output DEF file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose debug output')
    parser.add_argument('--show', action='store_true', help='Show visualization for each net')
    args = parser.parse_args()
    
    # Setup logger with appropriate level based on verbose flag
    init_logger(args.verbose)
    
    # Set default output file name if not provided
    output_file = args.output
    if not output_file:
        # Generate default output name based on input DEF file
        input_def_base = args.deffile.rsplit('.', 1)[0]  # Remove extension
        output_file = f"{input_def_base}_routed.def"
        logger.info(f"No output file specified, using default: {output_file}")
    
    # Run detailed routing
    detailed_route(args.deffile, args.leffile, args.guide, output_file, args.show)
