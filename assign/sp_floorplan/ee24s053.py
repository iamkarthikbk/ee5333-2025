###############################################################################
#
# Sequence Pair Floorplanner
#
# Author: ee24s053 Karthik B K
# Date: 26 February 2025
#
# For now, this is simply the code provided with the problem statement.
# The solution will be not be pushed here until 24 hours after the submission
# deadline as it appears on Moodle.
#
###############################################################################

import math
import random

class Module:
    def __init__(self, name, area, aspect_ratios):

        # blindly save the name and area
        self._name = name
        self._area = area

        # create a list of width,height tuples for all aspect ratios.
        self._wh = [(math.sqrt(area*r), math.sqrt(area/r)) for r in aspect_ratios]

class SeqPair:
    def __init__(self, modules):

        # initialize the g+, g- sequences
        self._pos = [i for i in range(len(modules))] # positive sequence
        self._neg = [i for i in range(len(modules))] # negative sequence

        # initialise aspect ratios and current locations
        self._ap = [0 for i in range(len(modules))] # aspect ratio choice
        self._coords = [(0,0) for i in range(len(modules))]

        # initialize the dimensions of our bounding box
        self._w = 0
        self._h = 0

    def perturb(self, modules):
        # Create a new sequence pair for the neighbor
        new_sp = SeqPair(modules)
        new_sp._pos = self._pos[:]
        new_sp._neg = self._neg[:]
        new_sp._ap = self._ap[:]
        
        # Choose a random move
        move = random.randint(0, 2)
        
        if move == 0:  # M0 from slides: Swap two blocks in positive sequence only
            if len(modules) > 1:

                # pick two random indices
                i = random.randint(0, len(modules)-1)
                j = random.randint(0, len(modules)-1)

                # ensure that the indices are unique
                while i == j:
                    j = random.randint(0, len(modules)-1)

                # swap the values in those indices in g+
                new_sp._pos[i], new_sp._pos[j] = new_sp._pos[j], new_sp._pos[i]
        
        elif move == 1:  # M1 from slides: Swap two blocks in both sequences
            if len(modules) > 1:
                # again, pick two random indices
                i = random.randint(0, len(modules)-1)
                j = random.randint(0, len(modules)-1)

                # ensure they're unique
                while i == j:
                    j = random.randint(0, len(modules)-1)

                # swap the values in those indices in both g+ and g-
                new_sp._pos[i], new_sp._pos[j] = new_sp._pos[j], new_sp._pos[i]
                new_sp._neg[i], new_sp._neg[j] = new_sp._neg[j], new_sp._neg[i]
        
        else:  # Change aspect ratio of a random module
            # pick a random module
            i = random.randint(0, len(modules)-1)
            old_ap = new_sp._ap[i]

            # Choose a different aspect ratio if more than 1 option is available
            if len(modules[i]._wh) > 1:
                new_ap = random.randint(0, len(modules[i]._wh)-1)

                # ensure that the new aspect ratio is a different one
                while new_ap == old_ap:
                    new_ap = random.randint(0, len(modules[i]._wh)-1)

                # set the aspect ratio to be the new one.
                new_sp._ap[i] = new_ap
        
        return new_sp

    def costEval(self, modules):
        n = len(modules)
        
        # Initialize coordinates to 0
        self._coords = [(0,0) for _ in range(n)]
        
        # Create horizontal constraints
        for i in range(1, n):  # For each block in positive sequence except first
            pos_block = self._pos[i]  # Current block in positive sequence
            j = 0
            neg_block = self._neg[j]  # Start with first block in negative sequence
            
            # Keep scanning negative sequence until we find pos_block
            while pos_block != neg_block:
                # If neg_block appears before pos_block in positive sequence
                if self._pos.index(neg_block) < self._pos.index(pos_block):
                    # neg_block should be to the left of pos_block
                    min_x = self._coords[neg_block][0] + modules[neg_block]._wh[self._ap[neg_block]][0]
                    old_x = self._coords[pos_block][0]
                    self._coords[pos_block] = (max(old_x, min_x), self._coords[pos_block][1])
                    
                j += 1
                neg_block = self._neg[j]
        
        # Create vertical constraints
        # Scan positive sequence from right to left
        for i in range(n-2, -1, -1):  # For each block in positive sequence except last
            pos_block = self._pos[i]  # Current block in positive sequence
            j = 0
            neg_block = self._neg[j]  # Start with first block in negative sequence
            
            # Keep scanning negative sequence until we find pos_block
            while self._pos.index(pos_block) != self._pos.index(neg_block):
                # If neg_block appears after pos_block in positive sequence
                if self._pos.index(neg_block) > self._pos.index(pos_block):
                    # pos_block should be below neg_block
                    min_y = self._coords[neg_block][1] + modules[neg_block]._wh[self._ap[neg_block]][1]
                    old_y = self._coords[pos_block][1]
                    self._coords[pos_block] = (self._coords[pos_block][0], max(old_y, min_y))
                j += 1
                if j >= n: break  # Safety check to avoid index out of bounds
                neg_block = self._neg[j]
            j = 0  # Reset j for next iteration
        
        # Calculate bounding box
        self._w = max([self._coords[i][0] + modules[i]._wh[self._ap[i]][0] for i in range(n)])
        self._h = max([self._coords[i][1] + modules[i]._wh[self._ap[i]][1] for i in range(n)])
        
        return self._w * self._h

def accept(delC, T):
    if delC <= 0: return True
    return random.random() < math.exp(-delC/T)

# S = Initial sequence pair, choice of aspect ratio
# ARmin, ARmax: minimum/maximum allowed aspect ratio of solution
def simulated_annealing(Tmin, Tmax, N, alpha, S, modules, ARmin, ARmax, plot):
    assert(alpha < 1. and Tmin < Tmax)
    T = Tmax
    C = S.costEval(modules)
    minC = C
    minS = S
    Clist = []
    Temp = []
    while T > Tmin:
        for i in range(N):
            Snew = S.perturb(modules)
            Cnew = Snew.costEval(modules)
            if accept(Cnew - C, T):
                C, S = Cnew, Snew
                if minC >= Cnew:
                    minC, minS = Cnew, Snew
                Clist.append(Cnew)
                Temp.append(T)
        T = T * alpha
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Temp, Clist)
        plt.xlim(max(Temp), min(Temp))
        plt.xscale('log')
    return minS, minC

def sp_floorplan(modules, ARmin, ARmax):
    S = SeqPair(modules)
    Tmax = sum([i._area for i in modules])
    Smin, Cmin = simulated_annealing(1, Tmax, 100, 0.9, S, modules, ARmin, ARmax, False)
    assert(len(Smin._coords) == len(Smin._ap) and (len(Smin._coords) == len(modules)))
    sol = [(Smin._coords[i], m[i]._wh[Smin._ap[i]], m[i]._name) for i in range(len(modules)) ]
    return (sol, Cmin)

def plot(coords):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    ax.plot([0, 0])
    ax.set_aspect('equal')
    ax.set_xlim(0,max([r[0][0] + r[1][0] for r in coords]))
    ax.set_ylim(0,max([r[0][1] + r[1][1] for r in coords]))
    for i,r in enumerate(coords):
        if i%4 == 3:
            hatch, color = '/+', 'red'
        elif i%4 == 2:
            hatch, color = '///', 'green'
        elif i%4 == 1:
            hatch, color = '/\//\//\/', 'blue'
        else:
            hatch, color = '\\', 'gray'
        ax.add_patch(Rectangle(r[0], r[1][0], r[1][1],
            edgecolor = color, facecolor=color,
            hatch=hatch, fill = False,
            lw=2))
        ax.text(r[0][0] + r[1][0]//2, r[0][1] + r[1][1]//2, r[2], fontsize=8)
    plt.show()

m = [Module('a', 16, [0.25, 4]), Module('b', 32, [2.0, 0.5]), Module('c', 27, [1./3, 3.]), Module('d', 6, [6])]
sol, area = sp_floorplan(m, 0.75, 1.33)
plot(sol)

m = [Module(str(i), random.randint(10,100), [1.]) for i in range(10)]
sol, area = sp_floorplan(m, 0.5, 2)
plot(sol)