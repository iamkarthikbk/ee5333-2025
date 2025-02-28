###############################################################################
#
# Sequence Pair Floorplanner
#
# Author: ee24s053 Karthik B K
# Date: 26 February 2025
#
# For now, this is simply the code provided with the problem statement.
# The solution will not be pushed here until 24 hours after the submission
# deadline as it appears on Moodle.
#
###############################################################################

import math
import random

class Module:
    def __init__(self, module_name, module_area, aspect_ratios):

        # blindly save the name and area
        self._module_name = module_name
        self._module_area = module_area

        # create a list of width,height tuples for all aspect ratios.
        self._width_height = [(math.sqrt(module_area*ratio), math.sqrt(module_area/ratio)) for ratio in aspect_ratios]

class SeqPair:
    def __init__(self, modules):

        # initialize the g+, g- sequences
        self._positive_sequence = [module_idx for module_idx in range(len(modules))] # positive sequence
        self._negative_sequence = [module_idx for module_idx in range(len(modules))] # negative sequence

        # initialise aspect ratios and current locations
        self._aspect_ratio_choice = [0 for module_idx in range(len(modules))] # aspect ratio choice
        self._coordinates = [(0,0) for module_idx in range(len(modules))]

        # initialize the dimensions of our bounding box
        self._bounding_box_width = 0
        self._bounding_box_height = 0

    def perturb(self, modules):
        # Create a new sequence pair for the neighbor
        new_seq_pair = SeqPair(modules)
        new_seq_pair._positive_sequence = self._positive_sequence[:]
        new_seq_pair._negative_sequence = self._negative_sequence[:]
        new_seq_pair._aspect_ratio_choice = self._aspect_ratio_choice[:]
        
        # Choose a random move
        move_type = random.randint(0, 2)
        
        if move_type == 0:  # M0 from slides: Swap two blocks in positive sequence only
            if len(modules) > 1:
                # pick two random, distinct indices
                idx1, idx2 = random.sample(range(len(modules)), 2)
                # swap in positive sequence only
                new_seq_pair._positive_sequence[idx1], new_seq_pair._positive_sequence[idx2] = new_seq_pair._positive_sequence[idx2], new_seq_pair._positive_sequence[idx1]
        
        elif move_type == 1:  # M1 from slides: Swap two blocks in both sequences
            if len(modules) > 1:
                # pick two random, distinct indices
                idx1, idx2 = random.sample(range(len(modules)), 2)
                # swap in both sequences
                new_seq_pair._positive_sequence[idx1], new_seq_pair._positive_sequence[idx2] = new_seq_pair._positive_sequence[idx2], new_seq_pair._positive_sequence[idx1]
                new_seq_pair._negative_sequence[idx1], new_seq_pair._negative_sequence[idx2] = new_seq_pair._negative_sequence[idx2], new_seq_pair._negative_sequence[idx1]
        
        else:  # Change aspect ratio of a random module
            module_idx = random.randint(0, len(modules)-1)
            # Only change if multiple aspect ratios are available
            if len(modules[module_idx]._width_height) > 1:
                # Choose a new aspect ratio different from current
                aspect_options = [ratio_idx for ratio_idx in range(len(modules[module_idx]._width_height)) if ratio_idx != new_seq_pair._aspect_ratio_choice[module_idx]]
                new_seq_pair._aspect_ratio_choice[module_idx] = random.choice(aspect_options)
        
        return new_seq_pair

    def costEval(self, modules):
        num_modules = len(modules)
        self._coordinates = [(0,0) for _ in range(num_modules)]
        
        # Create horizontal constraints (left-to-right)
        for pos_idx in range(1, num_modules):
            pos_block = self._positive_sequence[pos_idx]
            pos_seq_idx = pos_idx
            
            for neg_block in self._negative_sequence:
                neg_seq_idx = self._positive_sequence.index(neg_block)
                if neg_seq_idx < pos_seq_idx and neg_block != pos_block:
                    # neg_block should be to the left of pos_block
                    min_x_pos = self._coordinates[neg_block][0] + modules[neg_block]._width_height[self._aspect_ratio_choice[neg_block]][0]
                    self._coordinates[pos_block] = (max(self._coordinates[pos_block][0], min_x_pos), self._coordinates[pos_block][1])
                
                if neg_block == pos_block:
                    break
        
        # Create vertical constraints (bottom-to-top)
        for pos_idx in range(num_modules-2, -1, -1):
            pos_block = self._positive_sequence[pos_idx]
            pos_seq_idx = pos_idx
            
            for neg_block in self._negative_sequence:
                neg_seq_idx = self._positive_sequence.index(neg_block)
                if neg_seq_idx > pos_seq_idx:
                    # pos_block should be below neg_block
                    min_y_pos = self._coordinates[neg_block][1] + modules[neg_block]._width_height[self._aspect_ratio_choice[neg_block]][1]
                    self._coordinates[pos_block] = (self._coordinates[pos_block][0], max(self._coordinates[pos_block][1], min_y_pos))
                
                if neg_block == pos_block:
                    break
        
        # Calculate bounding box dimensions
        self._bounding_box_width = max(x_coord + modules[mod_idx]._width_height[self._aspect_ratio_choice[mod_idx]][0] for mod_idx, (x_coord, _) in enumerate(self._coordinates))
        self._bounding_box_height = max(y_coord + modules[mod_idx]._width_height[self._aspect_ratio_choice[mod_idx]][1] for mod_idx, (_, y_coord) in enumerate(self._coordinates))
        
        return self._bounding_box_width * self._bounding_box_height

def accept(delta_cost, temperature):
    if delta_cost <= 0: return True
    return random.random() < math.exp(-delta_cost/temperature)

# S = Initial sequence pair, choice of aspect ratio
# ARmin, ARmax: minimum/maximum allowed aspect ratio of solution
def simulated_annealing(Tmin, Tmax, N, alpha, S, modules, ARmin, ARmax, plot):
    assert(alpha < 1. and Tmin < Tmax)
    temperature, current_cost = Tmax, S.costEval(modules)
    min_cost, min_solution = current_cost, S
    cost_history, temp_history = [], []
    
    while temperature > Tmin:
        for _ in range(N):
            new_solution = S.perturb(modules)
            new_cost = new_solution.costEval(modules)
            
            if accept(new_cost - current_cost, temperature):
                current_cost, S = new_cost, new_solution
                if min_cost >= new_cost:
                    min_cost, min_solution = new_cost, new_solution
                cost_history.append(new_cost)
                temp_history.append(temperature)
        temperature *= alpha
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(temp_history, cost_history)
        plt.xlim(max(temp_history), min(temp_history))
        plt.xscale('log')
        
    return min_solution, min_cost

def sp_floorplan(modules, ARmin, ARmax):
    initial_seq_pair = SeqPair(modules)
    max_temp = sum([module._module_area for module in modules])
    best_solution, min_area = simulated_annealing(1, max_temp, 100, 0.9, initial_seq_pair, modules, ARmin, ARmax, False)
    assert(len(best_solution._coordinates) == len(best_solution._aspect_ratio_choice) and (len(best_solution._coordinates) == len(modules)))
    solution = [(best_solution._coordinates[mod_idx], modules[mod_idx]._width_height[best_solution._aspect_ratio_choice[mod_idx]], modules[mod_idx]._module_name) 
               for mod_idx in range(len(modules))]
    return (solution, min_area)

def plot(coords):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    ax.plot([0, 0])
    ax.set_aspect('equal')
    ax.set_xlim(0,max([rect[0][0] + rect[1][0] for rect in coords]))
    ax.set_ylim(0,max([rect[0][1] + rect[1][1] for rect in coords]))
    for idx, rect in enumerate(coords):
        if idx%4 == 3:
            hatch_pattern, color = '/+', 'red'
        elif idx%4 == 2:
            hatch_pattern, color = '///', 'green'
        elif idx%4 == 1:
            hatch_pattern, color = '\\\\\\', 'blue'
        else:
            hatch_pattern, color = 'o', 'black'
        ax.add_patch(Rectangle(rect[0], rect[1][0], rect[1][1], hatch=hatch_pattern, facecolor='white', fill=True, 
                            edgecolor=color, linewidth=3, label=rect[2]))
        ax.text(rect[0][0]+rect[1][0]/2, rect[0][1]+rect[1][1]/2, rect[2])
    plt.show()
    return (fig, ax)

modules = [Module('a', 16, [0.25, 4]), Module('b', 32, [2.0, 0.5]), Module('c', 27, [1./3, 3.]), Module('d', 6, [6])]
solution, area = sp_floorplan(modules, 0.75, 1.33)
plot(solution)

modules = [Module(str(i), random.randint(10,100), [1.]) for i in range(10)]
solution, area = sp_floorplan(modules, 0.5, 2)
plot(solution)