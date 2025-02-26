###############################################################################
# 
# Hierarchical Graph Partitioning
#
# Author: ee24s053 Karthik B K
# Date: 21 February 2025
#
# Design Specification
# --------------------
#
# We need to implement, in a 10k feet perspective, a graph partitioner.
# The input will take the number of vertices, and integer, and the edges,
# a list of tuples indicating the edge between the vertices as the
# primary arguments. The function will also take the number of max levels
# of coarsening that may be performed on the provided graph and also a
# skewness parameter to give the partitioner some wiggle room when trying
# to balance the graph during partitioning.
#
# The base algirithm for graph partitioining is going to be fiduccia-matheysses
#
# Implementation Details
# ----------------------
#
# The main graph partitioning loop itself is going to be quite straughtforward.
# We just need to ensure that we don't blow up executino time by adding stuff
# before/after the loop to transition from one iteration to the other.
# 
# Things to keep in mind while implementing the partitioner:
# 1. the initial partitioning should be computer outside the partitioner itself
#    because wew ant to reuse the partitioner function
# 2. the area of the input graph for the partitioner need not necessarily mean
#    anything for the vertices itself.
# 3. the uncoarsening step will become trivial if we simply store the
#    intermediate results from the coarsening function in an easily accessible
#    fashion. this is also a place where i can easily mess up the executino
#    time, if i use poorly thought out data structures.
###############################################################################

class Vertex:
    def __init__(self, name, area, partition=None):
        self.name = name
        self.area = area
        self.partition = partition
        self.partition_to_be = partition

    def update_partition(self, new_partition):
        # print(f'Partition for {self.name} updated from {self.partition} to {new_partition}')
        self.partition = int(new_partition)
        self.partition_to_be = int(new_partition)

    def stage_partition_update(self, new_partition):
        # print(f'Partition for {self.name} staged to change from {self.partition} to {new_partition}')
        self.partition_to_be = int(new_partition)

class Edge:
    def __init__(self, f, t, w):
        # print(f'Made an edge from {f} to {t}')
        self.f = f
        self.t = t
        self.weight = w

class Graph:
    def __init__(self, level, nodes, edges, areas, weights, partitions, coarsening_map, area_skew):
        # print(f'Made a level {level} graph with {nodes} nodes and {len(edges)} edges') # pr_info
        # blindly save everything that is thrown at you
        self.level = level
        self.nodes = nodes
        self.areas = areas
        self.partitions = partitions
        self.coarsening_map = coarsening_map
        self.area_tollerance = (sum(self.areas)) * area_skew * (level+1)

        # initialize the vertixes
        self.vertices = [Vertex(i, areas[i], partitions[i]) for i in range(nodes)]
        
        # initialize an empty list for edges
        self.edges = []
        for edge, weight in zip(edges, weights):
            # unpack this edge
            (edge_from, edge_to) = edge

            # digest this edge
            self.edges.append(Edge(f=edge_from, t=edge_to, w=weight))
        
        # compute total area
        self.total_area = sum(areas)

    # just go and throw half the guys into part 0 and the other half into part 1
    def dirty_partition(self):
        # print(f'Making dirty partitions for {self.nodes} nodes') # pr_info
        for vnum in range(self.nodes):
            self.vertices[vnum].update_partition(0 if (vnum < self.nodes // 2) else 1)

    # update partitions only
    def update_partitions(self, partitions):
        # print(f'Updated partitions for self to [{"".join(partitions)}]') # pr_info
        self.partitions = partitions
    
    # FM partitioner
    def partition(self, begin_with, coarsest):
        if coarsest: self.dirty_partition()
        else:
            for v in self.vertices:
                v.update_partition(begin_with[int(v.name)])
        
        # track moves being tried. keys are vertex names, values are gains
        staged_moves = dict()

        # try until no more moves are possible
        while len(staged_moves) < len(self.vertices):

            # Get best move excluding frozen (staged) vertices
            potential_move = self.get_best_move(frozen_vertices=staged_moves.keys())
            if potential_move is None:  # No valid moves available
                # print('No more valid moves possible (area balance constraint)')
                break
            
            candidate_vertex, gain = potential_move
            candidate_vertex.stage_partition_update(candidate_vertex.partition_to_be ^ 1)
            staged_moves[candidate_vertex] = gain

        # take a running sum of all staged moves, and take as few moves as possible to maximise the running sum
        moves = list(staged_moves.items())
        running_sums = [(moves[i][0], sum(m[1] for m in moves[:i+1])) for i in range(len(moves))]
        best_idx = max(range(len(running_sums)), key=lambda i: running_sums[i][1])
        
        # print('Staged Moves:', {v.name: gain for v, gain in staged_moves.items()}) #pr_info
        # print('Running Sums:', [sum for _, sum in running_sums]) #pr_info
        # print(f'Making the first {best_idx + 1} moves (best sum at index {best_idx})') #pr_info
        staged_moves = dict(moves[:best_idx + 1])

        # commit the moves
        for candidate_vertex in staged_moves.keys():
            candidate_vertex.update_partition(candidate_vertex.partition ^ 1)

        # update the partitions attribute
        self.update_partitions(''.join([str(v.partition) for v in self.vertices]))

        return self.partitions 

    # find the best move i.e. vertex with maximum gain difference (stay_gain - move_gain)
    def get_best_move(self, frozen_vertices=set()):
        part0_area = sum(v.area for v in self.vertices if v.partition_to_be == 0)
        total_area = sum(v.area for v in self.vertices)
        valid_moves = [(v, self.get_stay_gain(v) - self.get_move_gain(v))
                      for v in self.vertices if v not in frozen_vertices
                      and (0 <= part0_area + (v.area if v.partition_to_be == 1 else -v.area) <= total_area + self.area_tollerance)]
        return max(valid_moves, key=lambda x: x[1]) if valid_moves else None

    # get the stay gain (FS) for a vertex
    def get_stay_gain(self, vertex):
        return sum(self.get_edge_weight(vertex.name, n.name) 
                 for n in self.get_neighbors(vertex) 
                 if n.partition_to_be == vertex.partition_to_be)

    # get the move gain (TD) for a vertex
    def get_move_gain(self, vertex):
        return sum(self.get_edge_weight(vertex.name, n.name) 
                 for n in self.get_neighbors(vertex) 
                 if n.partition_to_be != vertex.partition_to_be)

    # get edge weight by edge description
    def get_edge_weight(self, from_vertex, to_vertex):
        return next((edge.weight for edge in self.edges 
                    if (from_vertex, to_vertex) in [(edge.f, edge.t), (edge.t, edge.f)]), 0)
    
    # coarsen once
    def coarsen(self):
        # max vertices that can be in a super-vertex
        max_collapse = 10

        # sort vertices by degree
        v_by_degree = self.get_vertices_by_degree()

        # track frozen vertices in a set. these can not be picked again
        # since they're already part of a super vertex.
        v_frozen = set()

        # make a temporary dict that contains what vertices go into what super-vertex
        sv_count = 0
        super_vertices = dict()

        # print(f'Vertices sorted by degree: {[v.name for v in v_by_degree]}')

        # come up with a vertex-grouping
        for v in v_by_degree:
            if v not in v_frozen:
                # this vertex has a large degree and has not been frozen yet
                # we can make a super vertex out of this guy and his neighbors
                super_vertices[sv_count] = [v]

                # freeze this initial vertex to avoid duplicates
                v_frozen.add(v)

                # add as many neighbors of this vertex into the same super vertex
                for neighbor in self.get_neighbors(v):
                    if neighbor not in v_frozen:

                        # ensure that we don't coarsen destructively.
                        if len(super_vertices[sv_count]) < max_collapse and \
                            (len(v_frozen) < self.nodes - 1):
                            # this neighbor has not been frozen and can be pinched.
                            super_vertices[sv_count].append(neighbor)

                            # freeze this neighbor
                            v_frozen.add(neighbor)

                # increment the sv ocunter
                sv_count += 1

        # accumulate areas for the same vertex grouping
        sv_areas = {sv: sum(member.area for member in members) for sv, members in super_vertices.items()}

        # create vertex to super-vertex mapping for quick lookups
        vertex_to_sv = {}
        for sv_id, members in super_vertices.items():
            for member in members:
                vertex_to_sv[member.name] = sv_id

        # connect super-vertices by edges. new edge weight would be sum of all edge weights between super vertices
        new_edges = {}
        new_weights = {}
        
        # Iterate through original edges and accumulate weights between super-vertices
        for edge in self.edges:
            sv1 = vertex_to_sv[edge.f]
            sv2 = vertex_to_sv[edge.t]
            
            # Skip self-loops in super-vertices
            if sv1 != sv2:
                # Ensure consistent edge naming by ordering super-vertex IDs
                edge_key = tuple(sorted([sv1, sv2]))
                
                # Add edge weight to existing or create new
                if edge_key in new_weights:
                    new_weights[edge_key] += edge.weight
                else:
                    new_edges[edge_key] = (edge_key[0], edge_key[1])  # Create tuple of (from, to)
                    new_weights[edge_key] = edge.weight

        # print(f'Making {sv_count} super-vertices with {new_edges} edges and weights {new_weights}')

        # Get weights in matching order
        weight_list = [new_weights[tuple(sorted([e[0], e[1]]))] for e in list(new_edges.values())]

        return sv_count, list(new_edges.values()), list(sv_areas.values()), weight_list, ([None] * sv_count), vertex_to_sv

    # return the edges as a list
    def get_edges(self):
        return [(edge.f, edge.t) for edge in self.edges]

    # return list of vertices sorted by their degree
    def get_vertices_by_degree(self):
        # Create a dictionary to count degrees
        degrees = {i: 0 for i in range(self.nodes)}
        
        # Count degree for each vertex
        for edge in self.edges:
            degrees[edge.f] += 1
            degrees[edge.t] += 1
        
        # Sort vertices by their degree in descending order
        sorted_vertices = sorted(range(self.nodes), key=lambda x: degrees[x], reverse=True)
        
        # Return the list of Vertex objects in sorted order
        return [self.vertices[i] for i in sorted_vertices]

    # return a list of all neighboring vertices for the given vertex
    def get_neighbors(self, vertex):
        # Get vertex name
        v_name = vertex.name
        
        # Find all edges connected to this vertex
        neighbors = []
        for edge in self.edges:
            if edge.f == v_name:
                neighbors.append(self.vertices[edge.t])
            elif edge.t == v_name:
                neighbors.append(self.vertices[edge.f])
        
        return neighbors


# this is the main entry point and the function signature here can not change.
def hier_part(N, E, c, s):
    # this function will deal with multiple graphs. keep track of them all.
    # this graph should only be appended to, and never modified otherwise.
    graphs = []

    # create the graph object with unit areas to begin with
    # this will be the 0th level, have all unit areas, and
    # partitions are basically a dont-care for now, hence an none-list.
    # initial coarsening map is also None.
    graph = Graph(level=0, nodes=N, edges=E,
        areas=[1 for _ in range(N)],
        weights=[1 for _ in range(len(E))],
        partitions=[None for _ in range(N)],
        coarsening_map=None, area_skew=s)

    # # create the initial partition
    # graph.dirty_partition()

    # initial partitioning should be done. we need to coarsen now.
    # this is essentially our starting point for hmetis.
    graphs.append(graph)

    # start coarsening c times. we go from 1 -> c+1 because we want to reuse
    # this iter variable for indicating the graph level. makes debugging easier
    for c_iter in range(1, c+1):

        # the graph can coarsen itself and tell what the new graph should be
        # while coarsening, the graph will NOT modify itself.
        new_nodes, new_edges, new_areas, new_weights, new_partitions, coarsening_map = graph.coarsen()

        # replace the graph handle
        graph = Graph(
            level=c_iter,
            nodes=new_nodes,
            edges=new_edges,
            areas=new_areas,
            weights=new_weights,
            partitions=new_partitions,
            coarsening_map=coarsening_map,
            area_skew=s
        )

        # push it inot the intermediate results list.
        graphs.append(graph)

    # we save these grpahs and start modifying just the partitions in each level
    # this variable will never be written to ever again.
    saved_graphs = graphs.copy()

    # initialize a variable that will be used as a temporary partition holder
    # between iterations while uncoarsening. initialize with Nones to make sure
    # erroneous uncoarsening can easily be seen.
    new_partitions = [None for _ in range(new_nodes)]

    # we should now have (c + 1) graphs
    # now we partition the coarsest graph using FM
    # in this step, the graph will modify itself.
    for graph in reversed(graphs):
        # print(f'Processing level {graph.level} graph with {graph.nodes} nodes') # pr_info

        # we don't try to partition the original graph for a last-level optimization to prevent
        # astronimical runtimes. this takes minutes to run for thousands of nodes.
        if graph.level > 0:
            new_partitions = graph.partition(
                begin_with=new_partitions,
                coarsest = graph.level == c
            )
            new_partitions = ''.join(new_partitions[sv] for _, sv in sorted(graph.coarsening_map.items()))

    # we are done with the parititiotiong and should exit
    sol = new_partitions
    return sol
