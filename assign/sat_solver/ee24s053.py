# SAT Solver using DPLL
#
# Author: ee24s053 Karthik B K
# Date: 2025-04-07
#
# Usage: python ee24s053.py -c <cnf_file>
#
# we basically have an initial propagation step where we check if any clauses are already SAT or UNSAT.
# some sort of an early out. then we repeatedly apply unit propagation and pure literal elimination.
# here, we find all clauses that have exactly one unassigned variable (unit clause) andget the unassigned value.
# we set that variable to SAT that particular unit clause (for each clause), i.e. true for non-inverted variables
# and false for inverted variables. then we find all literals with the same polarity and assign values st these
# literals are all SAT. while doing this, we check for conflicts with existing variable assignments. if there
# is a conflict encountered, then we declare the problem UNSAT. if not, we propagate these assignments through all
# clauses and check for conflicts. if a conflict is encountered, then we declare the problem UNSAT. if not, we
# continue with the process.
#
# the above loop basically achieves some sort of a simplification. if there exist variables after the above loop
# that are still unassigned, we _branch_ on one of the variables. that is, we make an assignment for the selected
# (branching) variable and recursively call the dpll algo on the new assignment. for every branching variable,
# we try both assignments. if both fail, we declare the problem UNSAT.
#
# after this, simply make a final check if all clauses are SAT and exit.

import random

class Clause:
  def __init__(self, vl):
    self._vars = [v for v in vl]
    self._vact = [True for v in vl]
    self._nact = len(self._vars)
    self._val  = None # None for not decided; False/True for evaled to False/True

  def eval(self, m):
    assigned_variable_count = 0
    for variable in self._vars:
      # If any literal is satisfied, the clause is satisfied
      if (m[abs(variable)] == True and variable > 0) or (m[abs(variable)] == False and variable < 0):
        return True
      # Count assigned variables
      if m[abs(variable)] is not None:
        assigned_variable_count += 1
    # If all variables are assigned but none satisfied the clause, it's unsatisfied
    if len(self._vars) == assigned_variable_count:
      return False

    # If some variables are unassigned, the clause is undetermined
    return None

  def getUnitVal(self):
    if self._nact == 1:
      for i in range(len(self._vars)):
        if self._vact[i]:
          return self._vars[i]
    return None
  
  def propagate(self, m):
    # Reset active variables
    self._vact = [True for v in self._vars]
    # Evaluate the clause with current assignments
    self._val = self.eval(m)
    
    # If clause is satisfied, mark all variables as inactive
    if self._val == True:
      self._vact = [False for v in self._vars]
      self._nact = 0
      return True
    # If clause is unsatisfied, return False
    elif self._val == False:
      return False
      
    # Mark assigned variables as inactive
    for index in range(len(self._vact)):
      if self._vact[index] and m[abs(self._vars[index])] is not None:
        self._vact[index] = False
    
    # Update count of active variables
    self._nact = self._vact.count(True)
    return self._val

  def __repr__(self):
    return '[' + str(self._vars) + ' ' + str(self._vact) + ' ' + str(self._nact) + ' ' + str(self._val) + ']'


def unitClauses(f):
  return [c for c in f if 1 == c._nact]
        

def pureLiterals(f, m):
  # Get all unassigned variables
  unassigned_variables = [i for i in range(1, len(m)) if None == m[i]]
  
  pure_literals = []
  for variable in unassigned_variables:
    polarity = None  # None = undetermined, True = positive, False = negative
    
    for clause in f:
      # Check if positive literal appears
      if variable in clause._vars:
        if polarity is None or polarity == True:
          polarity = True
        else:
          # Both polarities found, not pure
          polarity = None
          break
      # Check if negative literal appears
      if -variable in clause._vars:
        if polarity is None or polarity == False:
          polarity = False
        else:
          # Both polarities found, not pure
          polarity = None
          break
          
    # If variable appears with only one polarity, add it to pure literals
    if polarity is not None:
      pure_literals.append(variable if polarity else -variable)
      
  return pure_literals


def pickBranchingLiteral(m):
  l = [i for i in range(1, len(m)) if None == m[i]]
  return l[0] if len(l) else None
#return random.choice(l)

def dpll(f, m):
  # Create a copy of the assignment map to avoid modifying the original
  current_assignment = [i for i in m]
  
  # Initial propagation of all clauses
  for clause in f:
    if clause.propagate(current_assignment) == False:
      return False, None  # Immediate conflict detected
  
  # Check if all clauses are satisfied
  if all([clause._val == True for clause in f]):
    return True, current_assignment
  
  # Unit propagation and pure literal elimination loop
  changes_made = True
  while changes_made:
    changes_made = False
    
    # Unit propagation - assign values to variables in unit clauses
    unit_clauses_list = unitClauses(f)
    for clause in unit_clauses_list:
      unit_variable = clause.getUnitVal()
      if unit_variable is not None:
        variable_value = True if unit_variable > 0 else False
        variable_index = abs(unit_variable)
        
        # Check for conflicts with existing assignments
        if current_assignment[variable_index] is not None and variable_value != current_assignment[variable_index]:
          return False, None  # Conflict detected
          
        # Assign value if not already assigned
        if current_assignment[variable_index] is None:
          current_assignment[variable_index] = variable_value
          changes_made = True
    
    # Pure literal elimination - assign values to pure literals
    pure_literals_list = pureLiterals(f, current_assignment)
    for literal in pure_literals_list:
      variable_value = True if literal > 0 else False
      variable_index = abs(literal)
      
      # Check for conflicts with existing assignments
      if current_assignment[variable_index] is not None and variable_value != current_assignment[variable_index]:
        return False, None  # Conflict detected
        
      # Assign value if not already assigned
      if current_assignment[variable_index] is None:
        current_assignment[variable_index] = variable_value
        changes_made = True
    
    # Propagate changes if any were made
    if changes_made:
      for clause in f:
        if clause.propagate(current_assignment) == False:
          return False, None  # Conflict after propagation
      
      # Check if all clauses are satisfied after propagation
      if all([clause._val == True for clause in f]):
        return True, current_assignment
  
  # Choose a variable for branching
  branching_variable = pickBranchingLiteral(current_assignment)
  if branching_variable is not None:
    # Try both possible assignments (True and False)
    for assignment_value in [True, False]:
      # Create a copy of the current assignment for this branch
      branch_assignment = [value for value in current_assignment]
      branch_assignment[branching_variable] = assignment_value
      
      # Recursive DPLL call
      is_satisfiable, solution = dpll(f, branch_assignment)
      if is_satisfiable:
        return True, solution  # Return the solution found
    
    # If both assignments fail, this branch has no solution
    return False, None
  else:
    # No more variables to assign, check if all clauses are satisfied
    if all([clause._val == True for clause in f]):
      return True, current_assignment
    else:
      return False, None  # No solution found


def loadCNFFile(fn):
  numvars = 0
  numclauses = 0
  clauses = []
  with open(fn, 'r') as fs:
    for line in fs:
      if line[0] == '%': break
      # p is the description line
      if line[0] == 'p':
        numvars = int(line.split()[2])
        numclauses = int(line.split()[3])
        continue
      # c is a comment
      if line[0] == 'c': continue
      if numvars > 0:
        tmp = line.split()
        tmp = [int(tmp[i]) for i in range(len(tmp) - 1)]
        clauses.append(Clause(tmp))
        assert abs(tmp[0]) <= numvars and abs(tmp[1]) <= numvars and abs(tmp[2]) <= numvars
  assert len(clauses) == numclauses
  return numvars, clauses



if __name__ == '__main__':
  import argparse

  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--cnf", type=str, default="", help='<cnf file>')
  args = ap.parse_args()
  if args.cnf != "":
    print(f"CNF file  : {args.cnf}")
    numvars, clauses = loadCNFFile(args.cnf)
    m = [None for i in range(numvars + 1)]
    ret, m = (dpll(clauses, m))
    if m is None:
        print("UNSAT")
    else:
        print([(i if m[i] == True else -i) for i in range(1, len(m))])