## Global Placement
##
## Author: ee24s053 Karthik B K
## Last Modified Date: 02 April 2025
##
## Basically this is a global placer. there is a place function that takes a def file, lef file and gives an out file.
## there is a num iters but i have set ot to 10 by default. although, 6 should be sufficient for the s=given sample.
##
## Declaration: I have searched the web and consulted the prof a lot for the methods used in this placer. a lot of the
## design decisionsare heavily influenced by them. some tiny sections of code have also been borrowed from the internet,
## andit goes without saying that some of the functions are norrowed and used without any modificaion from the prof's placer.

from numbers import Number
import LEFDEFParser
import numpy as np
import networkx as nx

class Vertex:
    def __init__(self, c, dim):
        self._comp = c # component info directly from the parser
        self._nbrs = []
        self._pinNbrs = []
        self._bin = None
        self._index = -1
        self._dim = dim
        
    def __repr__(self):
        return self._comp.name() + f' {self._nbrs} {self._pinNbrs}'

# this stuff is borrowed from srampr's global_place that uses a quad approximation. unmodified.
class Bin:
    def __init__(self, index, bb):
        self._index = index
        self._vertices = []
        self._bbox = bb
        hw = (bb[1][0] - bb[0][0])/2.
        hh = (bb[1][1] - bb[0][1])/2.
        self._vpins = [LEFDEFParser.Point(int(bb[0][0] + hw), int(bb[0][1])),      #0 South pin
                      LEFDEFParser.Point(int(bb[0][0] + hw), int(bb[1][1])),      #1 North pin
                      LEFDEFParser.Point(int(bb[1][0]),      int(bb[0][1])),      #2 South East pin
                      LEFDEFParser.Point(int(bb[0][0]),      int(bb[0][1])),      #3 South West pin
                      LEFDEFParser.Point(int(bb[0][0]),      int(bb[0][1] + hh)), #4 East pin
                      LEFDEFParser.Point(int(bb[1][0]),      int(bb[0][1] + hh)), #5 West pin
                      LEFDEFParser.Point(int(bb[1][0]),      int(bb[1][1])),      #6 North East pin
                      LEFDEFParser.Point(int(bb[0][0]),      int(bb[1][1]))       #7 North West pin
                      ]
    
    # replace connections to neighbours that do not belong to this tile with virtual pin connections at appropriate boundaries
    def build(self):
        for i in range(len(self._vertices)):
            self._vertices[i]._index = i
            totalNbrs = len(self._vertices[i]._pinNbrs) + len(self._vertices[i]._nbrs)
            pnbrs = list()
            for pi in range(len(self._vertices[i]._pinNbrs)):
                p = self._vertices[i]._pinNbrs[pi]
                if p.x >= self._bbox[0][0] and p.x <= self._bbox[1][0] and p.y >= self._bbox[0][1] and p.y <= self._bbox[1][1]:
                    pnbrs.append(p)
                else:
                    if p.x < self._bbox[0][0]:
                        if p.y < self._bbox[0][1]: # south west
                            pnbrs.append(self._vpins[3])
                        elif p.y > self._bbox[0][1]: # north west
                            pnbrs.append(self._vpins[7])
                        else: #west
                            pnbrs.append(self._vpins[5])
                    elif p.x > self._bbox[1][0]:
                        if p.y < self._bbox[0][1]: # south east
                            pnbrs.append(self._vpins[2])
                        elif p.y > self._bbox[0][1]: # north east
                            pnbrs.append(self._vpins[6])
                        else: #east
                            pnbrs.append(self._vpins[4])
                    elif p.y < self._bbox[0][1]:#south
                        pnbrs.append(self._vpins[0])
                    elif p.y > self._bbox[1][1]:#north
                        pnbrs.append(self._vpins[1])
            self._vertices[i]._pinNbrs = pnbrs
            actNbrs = list() # remove neighbours not in this bin and add a connection to the corresponding pin at the boundary
            for nbr in self._vertices[i]._nbrs:
                index = self._index
                nBinIndex = nbr._bin._index
                if nBinIndex != index:
                    if nBinIndex[1] == index[1]: # east or west
                        if nBinIndex[0] < index[0]: #west
                            self._vertices[i]._pinNbrs.append(self._vpins[5])
                        else:
                            self._vertices[i]._pinNbrs.append(self._vpins[4])
                    elif nBinIndex[0] == index[0]: # south or north
                        if nBinIndex[1] < index[1]: #south
                            self._vertices[i]._pinNbrs.append(self._vpins[0])
                        else:
                            self._vertices[i]._pinNbrs.append(self._vpins[1])
                    elif nBinIndex[0] < index[0]: # south west or north west
                        if nBinIndex[1] < index[1]: #south
                            self._vertices[i]._pinNbrs.append(self._vpins[3])
                        else:
                            self._vertices[i]._pinNbrs.append(self._vpins[7])
                    elif nBinIndex[0] > index[0]: # south east or north east
                        if nBinIndex[1] < index[1]: #south
                            self._vertices[i]._pinNbrs.append(self._vpins[2])
                        else:
                            self._vertices[i]._pinNbrs.append(self._vpins[6])
                else:
                    actNbrs.append(nbr)
            self._vertices[i]._nbrs = actNbrs
            assert(totalNbrs == len(self._vertices[i]._pinNbrs) + len(self._vertices[i]._nbrs))
        return

def min_cut_placement(G):
    # tried using my own KL implementation, also the SA one we saw in class. seemed much slower.
    # the prof has allowed me to make a submission with this library function.
    partition = nx.algorithms.community.kernighan_lin_bisection(G)
    partition1 = set(partition[0])
    partition2 = set(partition[1])
    
    cut_size = 0
    for u in partition1:
        for v in partition2:
            if G.has_edge(u, v):
                cut_size += 1
    
    return partition1, partition2, cut_size

# Build graph from vertices and solve using min-cut
def solve(Vertices, iter_num):
    # parameter to control the strength of pin influence (higher = stronger pull toward pins)
    pin_influence_weight = 0.7
    # parameter to control the strength of cut attraction (higher = stronger pull toward cut edges)
    # Use a lower weight for cut attraction to avoid overwhelming pin influence
    # Also decrease it with iterations to allow fine-tuning in later stages
    cut_attraction_weight = 0.3 * max(0.1, (10 - iter_num) / 10)
    
    for i in range(len(Vertices)):
        if Vertices[i]._comp.location().x == 0 and Vertices[i]._comp.location().y == 0:
            Vertices[i]._comp.setLocation(0, 0)
    
    G = nx.Graph()
    
    for i, v in enumerate(Vertices):
        G.add_node(i, vertex=v)
    
    # track connection counts between vertices
    connection_counts = {}
    
    for i, v in enumerate(Vertices):
        for nbr in v._nbrs:
            edge = tuple(sorted([i, nbr._index]))
            if edge in connection_counts:
                connection_counts[edge] += 1
            else:
                connection_counts[edge] = 1
    
    for edge, count in connection_counts.items():
        G.add_edge(edge[0], edge[1], weight=count) # use connection count as weight
    
    part1, part2, cut_size = min_cut_placement(G)
    cut_attraction = {}

    # cut attraction. this is where we add a virtual pin to the cut edge, on both sides of the bupartition.
    # this is a way to pull the cells closer to the cut edge. this change was suggested by the prof on Wed Apr 2.
    if iter_num >= 2:  # Only start applying cut attraction after iteration 2. one h and one v cut. before that their positions are somewhat random.
        cut_edges = []
        for u in part1:
            for v in part2:
                if G.has_edge(u, v):
                    weight = G[u][v]['weight']
                    cut_edges.append((u, v, weight))
        
        # For each vertex in part1, calculate attraction from its connections in part2
        for u in part1:
            attraction_x, attraction_y = 0, 0
            total_weight = 0
            
            # Look at all cut edges involving this vertex
            for edge_u, edge_v, weight in cut_edges:
                if edge_u == u:
                    # This vertex has a connection to part2
                    # Use current position of the other vertex as attraction point
                    v_pos_x = Vertices[edge_v]._comp.location().x
                    v_pos_y = Vertices[edge_v]._comp.location().y
                    
                    # Only consider if the other vertex has a valid position
                    if v_pos_x != 0 or v_pos_y != 0:
                        attraction_x += v_pos_x * weight
                        attraction_y += v_pos_y * weight
                        total_weight += weight
            
            if total_weight > 0:
                cut_attraction[u] = (attraction_x / total_weight, attraction_y / total_weight)
        
        # For each vertex in part2, calculate attraction from its connections in part1
        for v in part2:
            attraction_x, attraction_y = 0, 0
            total_weight = 0
            
            # Look at all cut edges involving this vertex
            for edge_u, edge_v, weight in cut_edges:
                if edge_v == v:
                    # This vertex has a connection to part1
                    # Use current position of the other vertex as attraction point
                    u_pos_x = Vertices[edge_u]._comp.location().x
                    u_pos_y = Vertices[edge_u]._comp.location().y
                    
                    # Only consider if the other vertex has a valid position
                    if u_pos_x != 0 or u_pos_y != 0:
                        attraction_x += u_pos_x * weight
                        attraction_y += u_pos_y * weight
                        total_weight += weight
            
            if total_weight > 0:
                cut_attraction[v] = (attraction_x / total_weight, attraction_y / total_weight)
    
    # a gravitational pull effect where vertices are attracted toward their connected pins to minimize wirelength by placing cells closer to their connected pins
    pin_influence = {}
    for i, v in enumerate(Vertices):
        pin_sum_x, pin_sum_y = 0, 0
        pin_count = 0
        
        for p in v._pinNbrs:
            pin_sum_x += p.x
            pin_sum_y += p.y
            pin_count += 1
            
        if pin_count > 0:
            pin_influence[i] = (pin_sum_x / pin_count, pin_sum_y / pin_count)
    
    # place vertices based on partition and bin boundaries
    if v._bin is not None:
        bbox = v._bin._bbox
        mid_x = (bbox[0][0] + bbox[1][0]) / 2
        mid_y = (bbox[0][1] + bbox[1][1]) / 2
        
        # alternate between top/bottom and left/right partitioning based on iteration number
        if iter_num % 2 == 0:  # even iterations: top and bottom partitioning
            # place partition 1 on bottom
            for i in part1:
                v = Vertices[i]
                # Determine placement based on pin influence and cut attraction
                has_influence = False
                influence_x, influence_y = mid_x, mid_y
                
                # Apply pin influence if available
                if i in pin_influence:
                    has_influence = True
                    pin_x, pin_y = pin_influence[i]
                    influence_x = pin_x
                    influence_y = pin_y
                
                # Apply cut attraction if available
                if i in cut_attraction:
                    has_influence = True
                    cut_x, cut_y = cut_attraction[i]
                    # Blend pin influence and cut attraction if both exist
                    if i in pin_influence:
                        # Weight the blend - pins have higher priority than cut edges
                        influence_x = (pin_influence_weight * influence_x + cut_attraction_weight * cut_x) / (pin_influence_weight + cut_attraction_weight)
                        influence_y = (pin_influence_weight * influence_y + cut_attraction_weight * cut_y) / (pin_influence_weight + cut_attraction_weight)
                    else:
                        influence_x = cut_x
                        influence_y = cut_y
                
                if has_influence:
                    # bias placement toward influence point (pins and/or cut connections)
                    x = int(bbox[0][0] + (bbox[1][0] - bbox[0][0]) * (0.5 + pin_influence_weight * (influence_x - mid_x) / (bbox[1][0] - bbox[0][0])))
                    y = int(bbox[0][1] + (mid_y - bbox[0][1]) * (0.5 + pin_influence_weight * (influence_y - mid_y) / (bbox[1][1] - bbox[0][1])))
                else:
                    # random placement within bottom region
                    # no reason to use np.random.random - just doing it because np is already imported. don't overthink xD
                    x = int(bbox[0][0] + (bbox[1][0] - bbox[0][0]) * np.random.random())
                    y = int(bbox[0][1] + (mid_y - bbox[0][1]) * np.random.random())
                v._comp.setLocation(x, y)
            
            # place partition 2 on top
            for i in part2:
                v = Vertices[i]
                # Determine placement based on pin influence and cut attraction
                has_influence = False
                influence_x, influence_y = mid_x, mid_y
                
                # Apply pin influence if available
                if i in pin_influence:
                    has_influence = True
                    pin_x, pin_y = pin_influence[i]
                    influence_x = pin_x
                    influence_y = pin_y
                
                # Apply cut attraction if available
                if i in cut_attraction:
                    has_influence = True
                    cut_x, cut_y = cut_attraction[i]
                    # Blend pin influence and cut attraction if both exist
                    if i in pin_influence:
                        # Weight the blend - pins have higher priority than cut edges
                        influence_x = (pin_influence_weight * influence_x + cut_attraction_weight * cut_x) / (pin_influence_weight + cut_attraction_weight)
                        influence_y = (pin_influence_weight * influence_y + cut_attraction_weight * cut_y) / (pin_influence_weight + cut_attraction_weight)
                    else:
                        influence_x = cut_x
                        influence_y = cut_y
                
                if has_influence:
                    # bias placement toward influence point (pins and/or cut connections)
                    x = int(bbox[0][0] + (bbox[1][0] - bbox[0][0]) * (0.5 + pin_influence_weight * (influence_x - mid_x) / (bbox[1][0] - bbox[0][0])))
                    y = int(mid_y + (bbox[1][1] - mid_y) * (0.5 + pin_influence_weight * (influence_y - mid_y) / (bbox[1][1] - bbox[0][1])))
                else:
                    # random placement within top region
                    x = int(bbox[0][0] + (bbox[1][0] - bbox[0][0]) * np.random.random())
                    y = int(mid_y + (bbox[1][1] - mid_y) * np.random.random())
                v._comp.setLocation(x, y)
        else:  # odd iterations: left and right partitioning
            # place partition 1 on left
            for i in part1:
                v = Vertices[i]
                # Determine placement based on pin influence and cut attraction
                has_influence = False
                influence_x, influence_y = mid_x, mid_y
                
                # Apply pin influence if available
                if i in pin_influence:
                    has_influence = True
                    pin_x, pin_y = pin_influence[i]
                    influence_x = pin_x
                    influence_y = pin_y
                
                # Apply cut attraction if available
                if i in cut_attraction:
                    has_influence = True
                    cut_x, cut_y = cut_attraction[i]
                    # Blend pin influence and cut attraction if both exist
                    if i in pin_influence:
                        # Weight the blend - pins have higher priority than cut edges
                        influence_x = (pin_influence_weight * influence_x + cut_attraction_weight * cut_x) / (pin_influence_weight + cut_attraction_weight)
                        influence_y = (pin_influence_weight * influence_y + cut_attraction_weight * cut_y) / (pin_influence_weight + cut_attraction_weight)
                    else:
                        influence_x = cut_x
                        influence_y = cut_y
                
                if has_influence:
                    # bias placement toward influence point (pins and/or cut connections)
                    x = int(bbox[0][0] + (mid_x - bbox[0][0]) * (0.5 + pin_influence_weight * (influence_x - mid_x) / (bbox[1][0] - bbox[0][0])))
                    y = int(bbox[0][1] + (bbox[1][1] - bbox[0][1]) * (0.5 + pin_influence_weight * (influence_y - mid_y) / (bbox[1][1] - bbox[0][1])))
                else:
                    # random placement within left region
                    x = int(bbox[0][0] + (mid_x - bbox[0][0]) * np.random.random())
                    y = int(bbox[0][1] + (bbox[1][1] - bbox[0][1]) * np.random.random())
                v._comp.setLocation(x, y)
            
            # place partition 2 on right
            for i in part2:
                v = Vertices[i]
                # Determine placement based on pin influence and cut attraction
                has_influence = False
                influence_x, influence_y = mid_x, mid_y
                
                # Apply pin influence if available
                if i in pin_influence:
                    has_influence = True
                    pin_x, pin_y = pin_influence[i]
                    influence_x = pin_x
                    influence_y = pin_y
                
                # Apply cut attraction if available
                if i in cut_attraction:
                    has_influence = True
                    cut_x, cut_y = cut_attraction[i]
                    # Blend pin influence and cut attraction if both exist
                    if i in pin_influence:
                        # Weight the blend - pins have higher priority than cut edges
                        influence_x = (pin_influence_weight * influence_x + cut_attraction_weight * cut_x) / (pin_influence_weight + cut_attraction_weight)
                        influence_y = (pin_influence_weight * influence_y + cut_attraction_weight * cut_y) / (pin_influence_weight + cut_attraction_weight)
                    else:
                        influence_x = cut_x
                        influence_y = cut_y
                
                if has_influence:
                    # bias placement toward influence point (pins and/or cut connections)
                    x = int(mid_x + (bbox[1][0] - mid_x) * (0.5 + pin_influence_weight * (influence_x - mid_x) / (bbox[1][0] - bbox[0][0])))
                    y = int(bbox[0][1] + (bbox[1][1] - bbox[0][1]) * (0.5 + pin_influence_weight * (influence_y - mid_y) / (bbox[1][1] - bbox[0][1])))
                else:
                    # random placement within right region
                    x = int(mid_x + (bbox[1][0] - mid_x) * np.random.random())
                    y = int(bbox[0][1] + (bbox[1][1] - bbox[0][1]) * np.random.random())
                v._comp.setLocation(x, y)
    else:
        # If no bin information, use simple left/right partitioning
        for i in part1:
            Vertices[i]._comp.setLocation(0, 0)
        for i in part2:
            Vertices[i]._comp.setLocation(100, 100)



# bins : quadrisection
# create bins from the previous iterations solution
# reusing this function from the prof's code because this gives me 4 bins. i am merging the top two and bottom two or the left two and right two
# based on which iteration we are in, within the placement loop. 
def createBins(Vertices, bbox):
    w = (bbox[1][0] - bbox[0][0])/2.
    h = (bbox[1][1] - bbox[0][1])/2.
    bins = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            bins[i][j] = Bin((i, j), ((bbox[0][0] + i * w, bbox[0][1] + j * h), (bbox[0][0] + (i + 1) * w, bbox[0][1] + (j + 1)*h)))

    Vertices.sort(key=lambda v: v._comp.location().x)
    for xi in range(2):
        if 0 == xi: vec = Vertices[0:int(len(Vertices)/2)]
        else:       vec = Vertices[int(len(Vertices)/2):]
        vec.sort(key=lambda v:v._comp.location().y)
        for yi in range(2):
            if 0 == yi: bins[xi][yi]._vertices = vec[0:int(len(vec)/2)]
            else:       bins[xi][yi]._vertices = vec[int(len(vec)/2):]
            for v in bins[xi][yi]._vertices:
                v._bin = bins[xi][yi]
    for i in range(len(bins)):
        for j in range(len(bins[i])):
            bins[i][j].build()
    return bins

# Iteratively partition using min-cut and solve
def solveIter(V, bbox, outfile, d, Numiter):
    # Initial placement using min-cut
    for v in V:
        v._bin = None
    
    # Create a virtual bin for the entire chip
    root_bin = Bin((0,0), bbox)
    root_bin._vertices = V
    for v in V:
        v._bin = root_bin
    
    # Solve the initial placement
    savedSol = [(np.array([v._comp.location().x for v in V]), np.array([v._comp.location().y for v in V]))]
    solve(V, 0)
    savedSol.append((np.array([v._comp.location().x for v in V]), np.array([v._comp.location().y for v in V])))
    d.writeDEF(f'{outfile}_iter0.def')
    
    # Calculate and print costs for initial placement
    Vdict = {V[i]._comp.name():i for i in range(len(V))}
    P = [p.origin() for p in d.pins()]
    Pdict = {d.pins()[i].name():P[i] for i in range(len(P))}
    # oCost = overlapCost(V)
    # hCost = hpwlCost(d.nets(), Vdict, Pdict, V)
    # print(f'iter0,{oCost:.2f},{hCost:.2f},{0.1 * oCost + hCost:.2f}') # pr_info
    
    # Create initial bins
    bins = [createBins(V, bbox)]
    
    # Recursive min-cut partitioning
    for niter in range(1, Numiter):
        binstmp = []
        
        # Process each bin from the previous iteration
        for i in range(len(bins)):
            for j in range(len(bins[i])):
                for k in range(len(bins[i][j])):
                    try:
                        # Apply min-cut to place cells within this bin
                        solve(bins[i][j][k]._vertices, niter)
                    except:
                        # this stuff fails when there is only one vertex in the bin. instead of complex error handling i just throuw it out of here.
                        pass
                    
                    # Further partition this bin
                    if len(bins[i][j][k]._vertices) > 1:
                        binstmp.append(createBins(bins[i][j][k]._vertices, bins[i][j][k]._bbox))
                    else:
                        # If only one vertex, keep the bin as is
                        binstmp.append([[bins[i][j][k]]])
        
        # Save the solution for this iteration
        savedSol.append((np.array([v._comp.location().x for v in V]), np.array([v._comp.location().y for v in V])))
        d.writeDEF(f'{outfile}_iter{niter}.def')
        
        # # Calculate and print costs for this iteration
        # oCost = overlapCost(V)
        # hCost = hpwlCost(d.nets(), Vdict, Pdict, V)
        # print(f'iter{niter},{oCost:.2f},{hCost:.2f},{0.1 * oCost + hCost:.2f}') # pr_info
        
        # Update bins for next iteration
        bins = binstmp
    
    return savedSol

# Calculate overlap between two cells
def overlap(p1, p2):
    xd = max(min(p1[2] - p2[0], p2[2] - p1[0]), 0)
    yd = max(min(p1[3] - p2[1], p2[3] - p1[1]), 0)
    return xd * yd * 1.e-6

def overlapCost(V):
    overlapArea = 0.
    P = [(v._comp.location().x, v._comp.location().y, v._comp.location().x + v._dim[0], v._comp.location().y + v._dim[1]) for v in V]
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            overlapArea += overlap(P[i], P[j])
    return overlapArea

def hpwlCost(nets, Vdict, Pdict, V):
    hpwl = 0.
    for n in nets:
        minx, miny = 1.e20, 1.e20
        maxx, maxy = -1.e20, -1.e20
        for p in n.pins():
            if p[0] != 'PIN':
                assert(p[0] in Vdict)
                v = V[Vdict[p[0]]]
                pos = (v._comp.location().x + v._dim[0]//2, v._comp.location().y + v._dim[1])
            else:
                assert(p[1] in Pdict)
                pos = (Pdict[p[1]].x, Pdict[p[1]].y)
            minx = min(minx, pos[0])
            maxx = max(maxx, pos[0])
            miny = min(miny, pos[1])
            maxy = max(maxy, pos[1])
        hpwl += ((maxx - minx) + (maxy - miny)) * 1.e-3
    return hpwl

# load the DEF file and build the connectivity graph using the Vertex class
# Boundary pins have the name PIN followed by the pinName
def place(deffile, leffile, outfile, Numiter=4):
    # load LEF file for cell dimensions
    l = LEFDEFParser.LEFReader()
    dimLookup = dict()
    if leffile:
      l.readLEF(leffile)
      dimLookup = {m.name():(m.xdim(), m.ydim()) for m in l.macros()}

    d = LEFDEFParser.DEFReader()
    d.readDEF(deffile)
    chip_bbox = d.bbox()
    V = [Vertex(c, dimLookup[c.macro()]) for c in d.components()]
    for i in range(len(V)):
        V[i]._index = i
    Vdict = {V[i]._comp.name():i for i in range(len(V))}
    
    P = [p.origin() for p in d.pins()]
    Pdict = {d.pins()[i].name():P[i] for i in range(len(P))}
    
    for n in d.nets():
        u = list()
        pins = list()
        for p in n.pins():
            if p[0] != 'PIN':
                assert(p[0] in Vdict)
                u.append(Vdict[p[0]])
            else:
                assert(p[1] in Pdict)
                pins.append(Pdict[p[1]])
        for i in range(len(u)):
            for p in pins:
              V[u[i]]._pinNbrs.append(p)
            for j in range(i + 1, len(u)):
                V[u[i]]._nbrs.append(V[u[j]])
                V[u[j]]._nbrs.append(V[u[i]])
    
    bb = ((chip_bbox.ll.x, chip_bbox.ll.y),(chip_bbox.ur.x, chip_bbox.ur.y))
    # plot(solveIter(V, bb, outfile, d, Numiter=Numiter), bb)
    solveIter(V, bb, outfile, d, Numiter=Numiter)
    d.writeDEF(outfile)
    # oCost = overlapCost(V)
    # hCost = hpwlCost(d.nets(), Vdict, Pdict, V)
    # print("final: overlap cost :", oCost, " hpwlCost :", hCost, "total cost :", 0.1 * oCost + hCost) # pr_info

# Visualize the placement using a slider for the iteration
# Move the slide to see the cell migration
def plot(sol, bb):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    fig, ax = plt.subplots()
    ax.plot(sol[0][0], sol[0][1], 'o')
    ax.set_xlim([bb[0][0], bb[1][0]])
    ax.set_ylim([bb[0][1], bb[1][1]])
    fig.subplots_adjust(bottom=0.25)

    iterax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    iter_slider = Slider(ax=iterax, label='Iter', valmin=0, valmax=(len(sol)-1), valstep=1, valinit=0)

    def update(val):
        ax.clear()
        ax.plot(sol[val][0], sol[val][1], 'o')
        ax.set_xlim([bb[0][0], bb[1][0]])
        ax.set_ylim([bb[0][1], bb[1][1]])
        fig.canvas.draw_idle()

    iter_slider.on_changed(update)

    plt.show()


################ UNCOMMENT BELOW TO ENABLE MANUAL VERIFICATION #################
def main():
    print('Starting global placement using min-cut approach...')
    # place('sample/sample.def', 'sample/sample.lef', 'sample/sample_out.def',10)
    place('sample/dma.def', 'sample/dma.lef', 'sample/dma_out.def')
    print('Min-cut placement completed!')
if __name__ == '__main__':
    main()
