# Global router using A*
#
# Author: ee24s053 Karthik B K
# Date created: 07 April 2025
#
# A* is basically a search algorithm that finds the shortest path between two nodes in a graph.
# we use the magic structure provided by the prof (the priority queue). we define the g_score and f_score
# as the cost from the starting node to the current node and the estimated cost using the manhattan distance.
#
# we always pop the front of the priority queue and process it. this is the node with the lowest f_score.
# if the current node is the target, we return the path. else, we update the cost of the neighbors and push
# them into the priority queue. for each neighbor of the current node, we calculate a tentative g_score (i.e.
# the cosr from start to neighbor through current). if this path turns out to be bettwe than any previously found
# paths, we update the neighbor;s parent to the current node, the g_score for this neigbor, and the f_Score.
# along with this, we also upfate the neighbor in the open set, or simply add it if not present.

import math
import heapq as hq
import time
import random

class Vertex:
  def __init__(self, x, y, cost=math.inf, parent=None, nbrs=None):
    self._xy = (x, y)
    self._cost = cost
    self._parent = parent
    self._nbrs = nbrs
  def __lt__(self, r):
    return self._cost < r._cost
  def __eq__(self, r):
    return self._xy == r._xy
  def __repr__(self):
    return f'(xy:{self._xy}, cost:{self._cost})'

class priority_queue:
  def __init__(self, vertices = []):
    self._vertices = vertices[:]
    self._q = vertices[:]
    hq.heapify(self._q)
  def push(self, v):
    hq.heappush(self._q, v)
  def pop(self):
    return(hq.heappop(self._q))
  def update(self, v, cost):
    try: i = self._q.index(v)
    except ValueError: i = None
    if i is not None:
      self._q[i]._cost = cost
      hq.heapify(self._q)
  def updateIndex(self, i, cost):
    assert i < len(self._q)
    self._vertices[i]._cost = cost
    hq.heapify(self._q)
  def empty(self):
    return len(self._q) == 0
  def __contains__(self, v):
    return v in self._q
  def __repr__(self):
    return str(self._q)

def dist(u, v):
  return abs(u._xy[0] - v._xy[0]) + abs(u._xy[1] - v._xy[1])

# # Shortest path between a pair of vertices
# The cost of edge between $(u,v)$ as the Manhattan distance between $u$ and $v$.

# ## Dijkstra's algorithm
# * Input graph $G \equiv (V, E)$ and each edge $(u, v) \in E$ has a cost $w_{u,v} \in \mathbb{R}^+$
# * Find shortest path beween vertices $s\in V$ and $t\in V$
#   1. $(s.dist, s.parent) := (0, NULL)$
#   2. $(v.dist, v.parent) := (\infty, NULL)$, $\forall v \in V \setminus \{s\}$
#   3. Priority $Q$; prioritizes on least distance from $s$
#   4. Repeat until $Q$ is empty:
#     1. $u := Q.pop()$
#     2. if $u = t$: break
#     3. Repeat for each $(u, v) \in E$ and $v\in Q$:

#       if $v.dist > u.dist + w_{u,v}$:
#           $(v.dist, v.parent) = (u.dist + w_{u,v}, u)$
#   5. $path = [t]$
#   6. while $path.last().parent~!=~NULL$:

#     $path.append(path.last().parent)$
#   7. Return $path$

def dijkstra(V, s, t):
  for v in V:
    v._cost, v._parent = math.inf, None
  s._cost = 0
  Q = priority_queue(V)
  while not Q.empty():
    u = Q.pop()
    if u == t: break
    for v in u._nbrs:
      if v in Q:
        newcost = u._cost + dist(u, v)
        if newcost < v._cost:
          Q.update(v, newcost)
          v._parent = u
  path = [t]
  while path[-1]._parent is not None:
    path.append(path[-1]._parent)
  return path

# ## A* algorithm
# * Input graph $G \equiv (V, E)$ and each edge $(u, v) \in E$ has a cost $w_{u,v} \in \mathbb{R}^+$
# * Find shortest path beween vertices $s\in V$ and $t\in V$
# * dist(u, v) returns an estimate of minimum distance between u and v
#   * The estimate has to be a lower bound on the true distance
#   1. $(s.g, s.h, s.parent) := (0, dist(s, t), NULL)$
#   2. $(v.g, v.h, v.parent) := (\infty, dist(v, t), NULL)$, $\forall v \in V \setminus \{s\}$
#   3. Priority $Q = \{s\}$; prioritizes on least $v.g + v.h$
#     * If $u, v \in Q$ have $u.g + u.h = v.g + v.h$, $u$ gets higher priority if $u.g > v.g$ and vice-versa
#   4. Repeat until $Q$ is empty:
#     1. $u := Q.pop()$
#     2. if $u = t$: break
#     3. Repeat for each $(u, v) \in E$:

#       * if $v.g > u.g + w_{u,v}$:
#           * $(v.g, v.parent) = (u.g + w_{u,v}, u)$
#           * if $v \in Q$: $update(Q)$
#           * else:$Q.push(v)$
#   5. $path = [t]$
#   6. while $path.last().parent~!=~NULL$:

#     $path.append(path.last().parent)$
#   7. Return $path$

# [karthik] so basically this uses a guess of how far a certain node is, adds that to the actual distance till the
## current node and uses that as the priority. this is a better way of finding the shortest path as it uses a guess of
## the distance to the target. an early out definitely helps :D

def astar(V, s, t):
  # Early termination if source and target are the same
  if s == t:
    return [s]
    
  # Precompute heuristic for target to avoid repeated calculations
  target_xy = t._xy
  
  # Initialize costs
  for v in V:
    v._cost, v._parent = math.inf, None
  
  # Create closed set to track explored nodes - use a dictionary for faster lookups
  closed_set = {}
  
  # Set initial node cost to f_score (g_score + heuristic)
  h_score = abs(s._xy[0] - target_xy[0]) + abs(s._xy[1] - target_xy[1])  # Manhattan distance
  s._cost = h_score  # f_score = 0 + heuristic
  
  # Create open set with start node
  open_set = priority_queue([s])
  
  # Track g scores (cost from start to current node) - use a dictionary for O(1) lookups
  g_score = {}
  for v in V:
    g_score[v._xy] = math.inf
  g_score[s._xy] = 0
  
  # Use a separate dictionary to track nodes in the open set for O(1) membership checks
  in_open_set = {s._xy: True}
  
  while not open_set.empty():
    # Get node with lowest f_score
    current = open_set.pop()
    current_xy = current._xy
    
    # Remove from open set tracking
    in_open_set.pop(current_xy, None)
    
    # Skip if we've already processed this node
    if current_xy in closed_set:
      continue
    
    # If we reached the target, reconstruct and return the path
    if current == t:
      path = [t]
      while path[-1]._parent is not None:
        path.append(path[-1]._parent)
      return path
    
    # Add current node to closed set
    closed_set[current_xy] = True
    
    # Get current g_score once to avoid repeated dictionary lookups
    current_g = g_score[current_xy]
    
    # Explore neighbors
    for neighbor in current._nbrs:
      neighbor_xy = neighbor._xy
      
      # Skip if neighbor is already processed
      if neighbor_xy in closed_set:
        continue
        
      # Calculate tentative g_score - use direct calculation for speed
      edge_cost = abs(current_xy[0] - neighbor_xy[0]) + abs(current_xy[1] - neighbor_xy[1])
      tentative_g_score = current_g + edge_cost
      
      # If we found a better path to this neighbor
      if tentative_g_score < g_score.get(neighbor_xy, math.inf):
        # Update path and scores
        neighbor._parent = current
        g_score[neighbor_xy] = tentative_g_score
        
        # Calculate f_score = g_score + heuristic
        h_score = abs(neighbor_xy[0] - target_xy[0]) + abs(neighbor_xy[1] - target_xy[1])
        f_score_val = tentative_g_score + h_score
        
        # Always update the vertex cost to match its f_score
        neighbor._cost = f_score_val
        
        # Update neighbor in open set or add it if not present
        if neighbor_xy in in_open_set:
          open_set.update(neighbor, f_score_val)
        else:
          open_set.push(neighbor)
          in_open_set[neighbor_xy] = True
  
  # No path found
  return []

Vertices = [Vertex(0, 0, -1), Vertex(0,10,-1), Vertex(5,5,-1), Vertex(5,10,-1), Vertex(10,10,-1)]
Vertices[0]._nbrs = [Vertices[1], Vertices[2]]
Vertices[1]._nbrs = [Vertices[0], Vertices[4]]
Vertices[2]._nbrs = [Vertices[1], Vertices[3]]
Vertices[3]._nbrs = [Vertices[2], Vertices[4]]
Vertices[4]._nbrs = [Vertices[1], Vertices[3]]
for alg in [dijkstra, astar]:
  src = Vertices[0]
  tgt = Vertices[-1]
  t = time.time()
  path = alg(Vertices, src, tgt)
  print('src :', src, ' tgt :', tgt, 'path :', path, time.time() - t)

Vertices = [Vertex(random.randint(0,1000), random.randint(0,1000), -1) for i in range(10000)]
for v in Vertices:
  if v._nbrs is None: v._nbrs = list()
  for i in range(random.randint(1, 2)):
    nbr = Vertices[random.randint(0, len(Vertices)-1)]
    if nbr._nbrs is None: nbr._nbrs = list()
    v._nbrs.append(nbr)
    nbr._nbrs.append(v)
for alg in [dijkstra, astar]:
  src = Vertices[0]
  tgt = Vertices[-1]
  t = time.time()
  path = alg(Vertices, src, tgt)
  print('src :', src, ' tgt :', tgt, 'path :', path, time.time() - t)

# --- Larger random graph: 100,000 vertices ---
Vertices_large = [Vertex(random.randint(0,1000), random.randint(0,1000), -1) for _ in range(100000)]
for v in Vertices_large:
  if v._nbrs is None:
    v._nbrs = list()
  for _ in range(random.randint(1, 2)):
    nbr = Vertices_large[random.randint(0, len(Vertices_large)-1)]
    if nbr._nbrs is None:
      nbr._nbrs = list()
    v._nbrs.append(nbr)
    nbr._nbrs.append(v)
for alg in [dijkstra, astar]:
  src = Vertices_large[0]
  tgt = Vertices_large[-1]
  t = time.time()
  path = alg(Vertices_large, src, tgt)
  print('LARGE GRAPH:', 'src :', src, ' tgt :', tgt, 'path :', path, time.time() - t)