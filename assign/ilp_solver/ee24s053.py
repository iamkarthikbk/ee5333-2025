# LP solver using two-phase simplex + branch & bound to obtain integer solutions
#
# author: ee24s053 Karhtik B K
# date: 26 april 2025
#
# i referred to the following resources for understanding the simplex procedure.
# video: https://youtu.be/E72DWgKP_1Y
# code: https://github.com/rsizem2/simplex-algorithm
# also, perplexity is incredibly good at solving the entire thing step by step.
# referred to solutions from there also, to understand the procedure step by step.


import numpy as np
from mip import Model, maximize, INTEGER

# retained from the prof's code
def print_tableau(a, b, c, f):
    for i in range(len(a)):
        print("{0:30} | {1}".format(str(a[i]), b[i]))
    print('_______________________________________')
    print("{0:30} | {1}\n".format(str(c), f))

# simplex lp solver. this constains both phases.
def simplex(m, obj, verbose=False):
    constraints_str = '\n  '.join([str(c) for c in m.constrs])
    print(f'Starting the simplex solver with:\nobj:\n  {obj}\n\nconstrs:\n  {constraints_str}')
    
    var_map = {v.name: i for i, v in enumerate(m.vars)}
    
    # Setup variables and determine if Phase 1 is needed
    num_orig_vars = len(m.vars)
    num_slack_vars = len(m.constrs)
    num_art_vars = sum(1 for e in m.constrs if e.expr.sense in ['>', '='])

    # we need to go through phase1 only if we have any artificial variables.
    # otherwise just phase 2 will do. if this variable is unset, then it simply skips pgase1.
    need_phase1 = num_art_vars > 0
    
    # artificaial variables are only needed in phase 1
    num_vars = num_orig_vars + num_slack_vars + (num_art_vars if need_phase1 else 0)
    
    a = np.zeros(shape=(len(m.constrs), num_vars))
    c = np.zeros(num_vars)
    b = np.zeros(len(m.constrs))
    for i, v in obj.expr.items():
        c[var_map[i.name]] = v
    
    # Track which rows have artificial variables
    art_rows = []
    art_var_idx = num_orig_vars + num_slack_vars
    
    for i, e in enumerate(m.constrs):
        sens = e.expr.sense
        
        # Set RHS and fill constraint row with correct sign
        sign = -1 if e.rhs < 0 else 1
        b[i] = abs(e.rhs)
        for ind, val in e.expr.expr.items():
            a[i][var_map[ind.name]] = sign * val
        # Flip the sense if we multiplied by -1
        if sign == -1:
            if sens == '<': sens = '>'
            elif sens == '>': sens = '<'
        
        if sens == '<':
            a[i][num_orig_vars + i] = 1
        elif sens == '>':
            # these are surplus variables
            a[i][num_orig_vars + i] = -1
            
            if need_phase1:
                a[i][art_var_idx] = 1
                art_rows.append(i)
                art_var_idx += 1
        elif sens == '=':
            # only art variables for equality constraints
            if need_phase1:
                a[i][art_var_idx] = 1
                art_rows.append(i)
                art_var_idx += 1
    
    sol = np.zeros(num_vars)
    
    # we need to go through phase1 simplex if we have any artificial variables
    if need_phase1 and art_rows:
        # this is phase 1. goal is to find a basic feasible soluton
        
        w = np.zeros(num_vars)
        w[num_orig_vars + num_slack_vars : num_vars] = 1
        
        # objective value for phase1.
        phase1_f = 0
        
        # Make artificial variables basic by eliminating them from objective
        for i, row in enumerate(art_rows):
            art_idx = num_orig_vars + num_slack_vars + i
            w -= w[art_idx] * a[row]
            phase1_f -= w[art_idx] * b[row]
        
        if verbose:
            # print the tableua using the profs print function.
            print("Phase 1 Initial tableau:")
            print_tableau(a, b, w, phase1_f)
        
        for i in range(len(m.constrs)):
            if i in art_rows:
                art_idx = num_orig_vars + num_slack_vars + art_rows.index(i)
                sol[art_idx] = b[i]
            else:
                sol[num_orig_vars + i] = b[i]
        
        # phase 1 main loop
        while np.any(w > 0):
            if not np.any(w > 0): break
            
            # Select entering variable (col with rhe largest pos coeff)
            # this is the dantzig's rule from the video.
            pivot = np.argmax(w > 0)
            
            # Select leaving variable (row with minimum ratio)
            ratios = [b[i]/a[i][pivot] if a[i][pivot] > 0 else np.inf for i in range(len(b))]

            # check if we can pivot any further
            if all(np.isinf(ratios)):
                print("terminating..")
                return None, float('-inf')
                
            # pivot at the row with least ratio. this is the leaving variable from the vidoe
            pr = np.argmin(ratios)
            scale = a[pr][pivot]
            a[pr] /= scale
            b[pr] /= scale
            
            # these are the operations on the pivot row
            for i in range(len(b)):
                if i == pr: continue
                scale = a[i][pivot]
                a[i] -= (scale * a[pr])
                b[i] -= (scale * b[pr])
            
            scale = w[pivot]
            w -= scale * a[pr]
            phase1_f -= scale * b[pr]
            
            sol = np.zeros(num_vars)
            for j in range(num_vars):
                if np.sum(a, axis=0)[j] == 1.0:
                    sol[j] = b[np.argmax(a[:, j] > 0)]
        
        if verbose:
            print("Phase 1 Final tableau:")
            print_tableau(a, b, w, phase1_f)
        
        # Check if Phase 1 found a feasible solution
        if abs(phase1_f) > 1e-10:
            print(f"Problem is infeasible - Phase 1 objective is {phase1_f}")
            return None, float('-inf')
        
        # Prepare for Phase 2
        # Set artificial variable coefficients to 0 in objective
        for i in range(num_orig_vars + num_slack_vars, num_vars):
            c[i] = 0
    
    # Initialize objective value
    f = 0
    
    # Print initial tableau for Phase 2
    print("Initial tableau:")
    print_tableau(a, b, c, f)
    
    # phase2 simplex. this is more or less what the prof provided through moodle.
    while np.any(c > 0):
        if not np.any(c > 0): break
        
        pivot = np.argmax(c > 0)
        ratios = [b[i]/a[i][pivot] if a[i][pivot] > 0 else np.inf for i in range(len(b))]
        if all(np.isinf(ratios)):
            print("Problem is unbounded")
            return sol, float('inf')
            
        pr = np.argmin(ratios)
        scale = a[pr][pivot]
        a[pr] /= scale
        b[pr] /= scale
        
        for i in range(len(b)):
            if i == pr: continue
            scale = a[i][pivot]
            a[i] -= (scale * a[pr])
            b[i] -= (scale * b[pr])
        
        scale = c[pivot]
        c -= scale * a[pr]
        f -= scale * b[pr]

        sol = np.zeros(num_vars)
        for j in range(num_vars):
            if np.sum(a, axis=0)[j] == 1.0:
                i = np.argmax(a[:, j] > 0)
                sol[j] = b[i]
        
        if verbose: 
            print(f'after : a = {a} b = {b}, c = {c}, f = {f}, sol = {sol}')
    
    print("Final tableau:")
    print_tableau(a, b, c, f)
    
    return sol[:num_orig_vars], -f

# this is a tiny value epsilon. fcmp can end up suffereing from precision issues.
# we use this as a threshold for such precisoin issues.
eps = 1e-6

def is_sol_integer(sol, Nvar):
    for i in range(Nvar):
        if abs(round(sol[i]) - sol[i]) > eps: return False
    return True

def solve_ilp(m, obj, verbose=False):

    Nvar = len(m.vars)

    # first we simply solve the simplex and see if the solution is alreafy int or infeasibel.
    sol, f = simplex(m, obj)
    if sol is None:
        print("Problem is infeasible")
        return None, float('-inf')
    if is_sol_integer(sol, Nvar):
        return sol, f

    # sol is not integers and we need to snap them using b&b.
    print('Solution contains fractional values...')
    print('Starting branch and bound...\n\n')
    
    # best solution so far is the worst possoble solution.
    best_sol = None
    best_obj = float('-inf')
    nodes_explored = 0
    
    # this is a recursive function.
    def branch_and_bound(model, lower_bounds=None, upper_bounds=None, depth=0):

        # this variable should be taken from teh larger scope.
        nonlocal best_sol, best_obj, nodes_explored
        nodes_explored += 1
        
        lower_bounds = lower_bounds or [0] * Nvar  # assume non-negative vars
        upper_bounds = upper_bounds or [float('inf')] * Nvar
        
        new_model = Model()
        x_new = [new_model.add_var() for _ in range(Nvar)]
        
        # noww e copy the objective function and constraints from the original problem
        coeffs = {}
        for var, coef in obj.expr.items():
            var_idx = next(i for i, v in enumerate(model.vars) if v.name == var.name)
            coeffs[x_new[var_idx]] = coef
        new_model.objective = maximize(sum(coef * var for var, coef in coeffs.items()))
        
        for constr in model.constrs:
            expr_terms = {}
            for var, coef in constr.expr.expr.items():
                var_idx = next(i for i, v in enumerate(model.vars) if v.name == var.name)
                expr_terms[x_new[var_idx]] = coef
            
            expr = sum(coef * var for var, coef in expr_terms.items())
            if constr.expr.sense == '<':
                new_model += expr <= constr.rhs
            elif constr.expr.sense == '>':
                new_model += expr >= constr.rhs
            else:
                new_model += expr == constr.rhs
        
        # bounds are additional constrs
        for i in range(Nvar):
            if lower_bounds[i] > 0:
                new_model += x_new[i] >= lower_bounds[i]
            if upper_bounds[i] < float('inf'):
                new_model += x_new[i] <= upper_bounds[i]
        
        # invoke simplex on the new lp
        current_sol, current_obj = simplex(new_model, new_model.objective)
        
        if verbose:
            print(f"Node {nodes_explored} at depth {depth}: obj = {current_obj}")
        
        # Prune if infeasible or objective is not better than best so far
        if current_sol is None or (best_sol is not None and current_obj <= best_obj):
            return None, float('-inf')
        
        # If integer solution, update best if better than the best known sol so far
        if is_sol_integer(current_sol, Nvar):
            if best_sol is None or current_obj > best_obj:
                best_sol = current_sol.copy()
                best_obj = current_obj
                if verbose:
                    print(f"New best solution: {best_sol[:Nvar]} with objective {best_obj}")
            return current_sol, current_obj
        
        # Find a variable with fractional value to branch on
        branch_var = -1
        max_frac = 0
        for i in range(Nvar):
            frac_part = abs(current_sol[i] - round(current_sol[i]))
            if frac_part > eps and frac_part > max_frac:
                max_frac = frac_part
                branch_var = i
        
        # left branch
        left_upper_bounds = upper_bounds.copy()
        left_upper_bounds[branch_var] = int(current_sol[branch_var])
        
        # right branch
        right_lower_bounds = lower_bounds.copy()
        right_lower_bounds[branch_var] = int(current_sol[branch_var]) + 1
        
        if verbose:
            print(f"Branching on x_{branch_var} = {current_sol[branch_var]}")
            print(f"Left branch: x_{branch_var} <= {left_upper_bounds[branch_var]}")
            print(f"Right branch: x_{branch_var} >= {right_lower_bounds[branch_var]}")
        
        # invoke the solver on both branches
        left_sol, left_obj = branch_and_bound(model, lower_bounds, left_upper_bounds, depth + 1)
        right_sol, right_obj = branch_and_bound(model, right_lower_bounds, upper_bounds, depth + 1)
        
        # Return the better solution
        if left_sol is None and right_sol is None:
            return None, float('-inf')
        elif left_sol is None:
            return right_sol, right_obj
        elif right_sol is None:
            return left_sol, left_obj
        else:
            if left_obj > right_obj:
                return left_sol, left_obj
            else:
                return right_sol, right_obj
    
    # start the b&b process
    result_sol, result_obj = branch_and_bound(m)
    
    # something is not right if this happens
    if result_sol is None:
        return sol, f
    
    if verbose:
        print(f"Branch and bound explored {nodes_explored} nodes")
        print(f"Best integer solution: {result_sol[:Nvar]} with objective {result_obj}")
    
    return result_sol[:Nvar], result_obj

### Please uncomment all code below to enable manual testing. ###
# # sample problem 1
# m = Model()
# x = [m.add_var(var_type=INTEGER) for i in range(2)]
# obj = m.objective = maximize(x[0] + x[1])
# m += x[0] + 3 * x[1] <= 9.2
# m += 2 * x[0] + x[1] <= 8.4
# sol, f = solve_ilp(m, obj)
# m.optimize()
# print('\n\n----------------')
# if sol is not None:
#     print('[OUR SOLUTION] solution :', sol[0:len(m.vars)], f'objective : {f}')
# else:
#     print('[OUR SOLUTION] Problem is infeasible')
# print('----------------')
# print('[MIP SOLUTION] mip sol :', [v.x for v in m.vars], 'objective :', m.objective.x)
# print('----------------\n\n')


# # sample problem 2
# m = Model()
# x = [m.add_var(var_type=INTEGER) for i in range(3)]
# obj = m.objective = maximize(6 * x[0] + x[1])
# m += 9 * x[0] + x[1] + x[2] <= 18.4
# m += 24 * x[0] + x[1] + 4 * x[2] <= 42.3
# m += 12 * x[0] + 3 * x[1] + 4 * x[2] <= 96.5
# sol, f = solve_ilp(m, obj)
# m.optimize()
# print('\n\n----------------')
# if sol is not None:
#     print('[OUR SOLUTION] solution :', sol[0:len(m.vars)], f'objective : {f}')
# else:
#     print('[OUR SOLUTION] Problem is infeasible')
# print('----------------')
# print('[MIP SOLUTION] mip sol :', [v.x for v in m.vars], 'objective :', m.objective.x)
# print('----------------\n\n')