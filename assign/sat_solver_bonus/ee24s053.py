# SAT Solver using CDCL

# Author: ee24s053 Karthik B K
# Date: 2025-04-12

# The following online source was used as a reference: https://github.com/sgomber/CDCL-SAT
# The basic idea in CDCL is the same as that in DPLL, but here it just learns clauses whenever
# a conflict occurs. use the -v option to see a cool little description of how the solver actually
# solved the problem. i have also limited the number of clauses because in UUF250 i saw that the
# clauses keep piling up. so we remove old clauses when we exceed the limit.

import time
import random
import argparse


class Expression:
    # A CNF expression is a conjunction of clauses, each clause being a disjunction of literals
    
    def __init__(self, list_clause):
        # Initialize the expression with a list of clauses
        # Each clause is converted to a Clause with watched literals
        self.expression = [Clause(c) for c in list_clause if len(c) > 0]  # Skip empty clauses
        self.value = self.get_value()  # Initial expression value (0=unknown, 1=sat, -1=unsat)
        self.given_clauses = len(list_clause)  # Track original clause count
        self.clause_limit = 2*self.given_clauses  # Limit on total clauses (original + learned)

    def get_value(self):
        # Compute the value of the expression:
        # -1 if any clause is unsatisfiable, 0 if undetermined, 1 if all satisfied
        values = [c.value for c in self.expression]
        return -1 if -1 in values else 0 if 0 in values else 1

    def get_counter(self):
        # Count how many times each watched literal is referenced in the expression
        refs = [clause.watched_lit_1 for clause in self.expression if clause.size >= 1] + \
               [clause.watched_lit_2 for clause in self.expression if clause.size > 1]
        counter = {x: refs.count(x) for x in set(refs)}  # Frequency count
        return dict(sorted(counter.items(), key=lambda item: item[1], reverse=True))  # Sort by count

    def is_sat(self):
        # Return the current SAT status of the expression
        return self.value
    
    def propagate_constr(self, literal, current_decision_level, graph):
        # Propagate a given literal through all clauses
        # If any clause becomes unsatisfiable, return -1 and the conflicting clause
        for clause in self.expression:
            if clause.value == -1 or (clause.value == 0 and clause.propagate_constr(literal, current_decision_level, graph) == -1):
                self.value = -1
                return self.value, clause
        self.value = self.get_value()
        return self.value, None
            
    def unit_propagate(self, current_decision_level, graph=None):
        # Perform unit propagation: assign literals that are forced by unit clauses
        for clause in self.expression:
            if self.value != 0:
                break  # Stop if expression is already SAT or UNSAT
            if clause.is_unit():
                unit_literal = clause.watched_lit_1  # The single unassigned literal
                if graph is not None and (unit_literal not in graph.assigned_vars):
                    graph.add_node(unit_literal, clause, current_decision_level)  # Assign unit literal
                is_sat, conflict_clause = self.propagate_constr(unit_literal, current_decision_level, graph)
                if is_sat == -1:
                    self.value = -1  # Conflict found during propagation
                    return self.value, conflict_clause
                else:
                    return self.unit_propagate(current_decision_level, graph)  # Continue propagating
        return self.value, None

    def backjump(self, backjump_level, graph):
        # Restore all clauses and variable assignments to the state at backjump_level
        for clause in self.expression:
            clause.restore(backjump_level, graph)
        self.value = 0  # Reset expression value to undetermined

    def add_clause(self, clause):
        # Add a new clause to the expression, respecting the clause limit
        if len(self.expression) > self.clause_limit:
            self.expression.pop(self.given_clauses)  # Remove the first learned clause if over limit
        self.expression += [clause]  # Add the new clause

class Clause:
    # Clause implementation using watched literals optimization. Basically, a clause is a disjunction of literals
    # and the solver only watches two literals in a clause. As long as at least one of these two literals
    # is unassigned or True, the clause can not become UNSAT, so we don't need to watch all literals.
    # when the watched literal is assigned, say False, the solver starts watching a different literal.
    # if only one literal is left, the clause is a unit clause, and the solver will fix that literal.
    
    def __init__(self, list_literal):
        # Initialize clause with given literals
        
        self.clause = list(list_literal)     # List of literals in the clause
        self.current_decision_level = [-1] * len(self.clause)  # Decision level for each literal
        self.value = 0  # Clause value: 0=unassigned, 1=true, -1=false
        self.size = len(self.clause)  # Current effective size (unassigned literals)
        
        # Initialize watched literals for 2-watched-literal scheme
        # These two literals are monitored for changes during propagation
        if self.size > 1:
            # For clauses with multiple literals, select two random ones to watch
            self.watched_lit_1, self.watched_lit_2 = random.sample(self.clause, 2)
            self.indexA, self.indexB = self.clause.index(self.watched_lit_1), self.clause.index(self.watched_lit_2)
        elif self.size == 1:
            # For unit clauses, watch the single literal
            self.watched_lit_1 = self.watched_lit_2 = self.clause[0]
            self.indexA = self.indexB = 0
        else:
            # For empty clauses, initialize watched literals to None
            self.watched_lit_1 = self.watched_lit_2 = self.indexA = self.indexB = None

    def is_unit(self):
        return int(self.size == 1)

    def update(self, graph):
        # Iterate through each literal in the clause and update its decision level
        assigned_vars = graph.assigned_vars
        sat_dl = []
        for i, lit in enumerate(self.clause):
            if lit in assigned_vars:
                # If the literal is assigned, update its decision level
                self.current_decision_level[i] = graph.graph[lit][1]
                sat_dl.append(graph.graph[lit][1])
            elif -lit in assigned_vars and self.current_decision_level[i] == -1:
                # If the negation of the literal is assigned, update its decision level
                self.current_decision_level[i] = graph.graph[-lit][1]
        # If any literals are satisfied, update the clause value and decision levels
        if sat_dl:
            min_dl = min(sat_dl)
            self.value = 1
            self.current_decision_level = [min_dl if dl == -1 or dl > min_dl else dl for dl in self.current_decision_level]
        # Update the effective size of the clause (unassigned literals)
        self.size = self.current_decision_level.count(-1)
        # If the clause is empty and not satisfied, it is unsatisfiable
        if self.size == 0 and not sat_dl:
            self.value = -1
        # Sort the clause literals and decision levels by decision level
        zipped = sorted(zip(self.current_decision_level, self.clause), reverse=True)
        if zipped:
            self.current_decision_level, self.clause = zip(*zipped)
            self.current_decision_level = list(self.current_decision_level[-self.size:] + self.current_decision_level[:-self.size])
            self.clause = list(self.clause[-self.size:] + self.clause[:-self.size])
        else:
            self.current_decision_level, self.clause = [], []

    def check_and_update(self, graph):
        # Check if the clause is satisfied or unsatisfiable based on assigned variables
        assigned_vars = graph.assigned_vars
        sat_dl = []
        for i, lit in enumerate(self.clause):
            if lit in assigned_vars:
                self.current_decision_level[i] = graph.graph[lit][1]
                sat_dl.append(graph.graph[lit][1])
            elif -lit in assigned_vars and self.current_decision_level[i] == -1:
                self.current_decision_level[i] = graph.graph[-lit][1]
        # If any literals are satisfied, update the clause value and decision levels
        if sat_dl:
            min_dl = min(sat_dl)
            self.current_decision_level = [min_dl if dl == -1 or dl > min_dl else dl for dl in self.current_decision_level]
            self.value = 1
            self.size = 0
            self.remove_refs()
        # Update the effective size of the clause (unassigned literals)
        self.size = self.current_decision_level.count(-1)
        # Sort the clause literals and decision levels by decision level
        zipped = sorted(zip(self.current_decision_level, self.clause), reverse=True)
        if zipped:
            self.current_decision_level, self.clause = zip(*zipped)
            self.current_decision_level = list(self.current_decision_level)
            self.clause = list(self.clause)
        # If the clause is not satisfied and has unassigned literals, pick new watched literals
        if self.size > 0:
            self.clause = self.clause[-self.size:] + self.clause[:-self.size]
            self.current_decision_level = self.current_decision_level[-self.size:] + self.current_decision_level[:-self.size]
            self.indexA = self.clause.index(self.watched_lit_1)
            self.indexB = self.clause.index(self.watched_lit_2)
            self.pick_new_ref()
        # If the clause is empty and not satisfied, it is unsatisfiable
        elif self.value != 1:
            self.value = -1
            self.remove_refs()

    def pick_new_ref(self):
        # Pick new watched literals for the clause
        if self.size == 1:
            # If the clause is a unit clause, watch the single literal
            self.watched_lit_1 = self.watched_lit_2 = self.clause[0]
        else:
            # If the clause has multiple literals, select two random ones to watch
            pool = list(self.clause[:self.size])  # Pool of unassigned literals
            refs = set([self.watched_lit_1, self.watched_lit_2])  # Current watched literals
            # Find which watched literals are still in the pool
            valid_refs = [r for r in refs if r in pool]
            # Remove valid watched literals from the pool
            for r in valid_refs:
                pool.remove(r)
            # Randomly select new watched literals until we have two
            while len(valid_refs) < 2:
                new_ref = random.choice(pool)
                valid_refs.append(new_ref)
                pool.remove(new_ref)
            # Assign the two watched literals
            self.watched_lit_1, self.watched_lit_2 = valid_refs[:2]
        # Update indices of watched literals in the clause
        self.indexA = self.clause.index(self.watched_lit_1)
        self.indexB = self.clause.index(self.watched_lit_2)

    def __repr__(self) -> str:
        # Return a string representation of the clause
        return f'{self.clause}'

    def remove_refs(self):
        # Remove the watched literals and their indices
        self.watched_lit_1, self.watched_lit_2 = None, None
        self.indexA, self.indexB = None, None

    def propagate_constr(self, literal, current_decision_level, graph):
        # Propagate the given literal through the clause
        if self.size == 1:
            # If the clause is a unit clause, update its value and decision level
            if literal == self.watched_lit_1 or -literal == self.watched_lit_1:
                self.current_decision_level[self.indexA] = current_decision_level
                self.size = 0
                self.value = 1 if literal == self.watched_lit_1 else -1
                self.remove_refs()
            return self.value
        # If the clause has multiple literals, check if the given literal is watched
        if self.size > 1 and (literal == self.watched_lit_1 or literal == self.watched_lit_2 or -literal == self.watched_lit_1 or -literal == self.watched_lit_2):
            # If the given literal is watched, update the clause state
            self.check_and_update(graph)
        return self.value

    def restore(self, level, graph):
        # Restore the clause state to the given decision level
        self.update(graph)
        self.size = sum(lvl == -1 or lvl > level for lvl in self.current_decision_level)
        if self.size > 0:
            # If the clause has unassigned literals, update its value and decision levels
            self.value = 0
            self.current_decision_level[:self.size] = [-1] * self.size
            self.pick_new_ref()
        else:
            # If the clause is empty, remove its watched literals
            self.remove_refs()

    def literal_at_level(self, lvl):
        # Return the literals at the given decision level
        return [lit for lit, dl in zip(self.clause, self.current_decision_level) if dl == lvl]

    def get_backjump_level(self):
        # Return the decision level to backjump to
        levels = sorted({dl for dl in self.current_decision_level if dl != -1}, reverse=True)
        if not levels:
            return -1
        return levels[0] - 1 if len(levels) == 1 else levels[1]

    def resolve(self, other, literal):
        # Resolve this clause with another clause on the given literal
        idx_self = self.clause.index(literal)  # Find the index of the resolving literal in self
        # Remove the resolving literal from self's literals and decision levels
        new_literals = self.clause[:idx_self] + self.clause[idx_self+1:]
        new_decision_levels = self.current_decision_level[:idx_self] + self.current_decision_level[idx_self+1:]
        tautology = False
        # Loop through the literals in the other clause
        for idx_other, lit in enumerate(other.clause):
            if abs(lit) == abs(literal):
                continue  # Skip the resolved literal (and its negation)
            if -lit in new_literals:
                tautology = True  # If both a literal and its negation appear, clause is a tautology
                break
            if lit not in new_literals:
                new_literals.append(lit)  # Add new literal from other clause
                new_decision_levels.append(other.current_decision_level[idx_other])  # Track its decision level
        if tautology:
            return Clause([])  # Return empty clause if tautology detected
        resolved_clause = Clause(new_literals)  # Create the new resolved clause
        resolved_clause.set_decision_levels(new_decision_levels)  # Set its decision levels
        resolved_clause.size = resolved_clause.current_decision_level.count(-1)  # Update size
        return resolved_clause  # Return the resolved clause

    def set_decision_levels(self, current_decision_level):
        # Set the decision levels for the clause literals
        self.current_decision_level = current_decision_level
        self.clause = [x for _,x in sorted(zip(self.current_decision_level,self.clause), reverse=True)]
        self.current_decision_level.sort(reverse=True)

class ImplicationGraph:
    def __init__(self):
        self.graph = dict()
        self.assigned_vars = []

    def add_node(self, literal, blame, current_decision_level):
        self.graph[literal] = [blame, current_decision_level]
        self.assigned_vars = list(self.graph.keys())

    def remove_node(self, literal):
        # Remove the node for literal or its negation, if present
        self.graph.pop(literal, None)
        self.graph.pop(-literal, None)
        self.assigned_vars = list(self.graph.keys())

    def backjump(self, backjump_level):
        # Remove all nodes with decision level greater than backjump_level
        [self.remove_node(node) for node in list(self.graph.keys()) if self.graph[node][1] > backjump_level]

    def get_blame(self, literal):
        # Return the blame for the literal, or None if not present
        return self.graph.get(literal, [None])[0]

class Solver: 
    def __init__(self, input_cnf_file, verbose):
        self.verbose_output = verbose
        self.original_clauses, self.num_variables = parse(input_cnf_file, self.verbose_output)
        self.expression = Expression(self.original_clauses)
        self.graph = ImplicationGraph()
        self.current_decision_level = 0
        self.restart_conflict_threshold = 100
        self.sat_status = 0
        self.current_conflict = None
        self.num_original_clauses = len(self.original_clauses)
        self.num_learnt_clauses = 0
        self.num_decisions = 0
        self.num_restarts = 0
        self.num_conflicts = 0
        self.num_conflict_analyses = 0
    
    def restart(self):
        # Restart the solver by reinitializing key data structures
        # Keeps learned clauses but resets the search process
        self.expression = Expression(self.original_clauses)
        self.graph = ImplicationGraph()
        self.current_decision_level = 0
        self.num_restarts += 1
        self.num_conflicts = 0
        self.sat_status = 0
        self.current_conflict = None
    
    def conflict_analysis(self, conflict_clause):
        # Increment the number of conflict analyses performed
        self.num_conflict_analyses += 1
        # If too many analyses, signal a restart
        if self.num_conflict_analyses >= 100:
            self.num_conflict_analyses = 0
            return None, -100
        # Get literals in the conflict clause assigned at the current decision level
        literals = conflict_clause.literal_at_level(self.current_decision_level)
        # If no such literals, return backjump level (or -1 at root)
        if not literals:
            return None, -1 if self.current_decision_level == 0 else conflict_clause.get_backjump_level()
        # If only one such literal, return the clause and backjump level (learned clause found)
        if len(literals) == 1:
            self.num_conflict_analyses = 0
            return conflict_clause, conflict_clause.get_backjump_level()
        # Otherwise, resolve on each literal and recursively analyze
        for conflict_literal in literals:
            blame = self.graph.get_blame(-conflict_literal)  # Get antecedent clause for the negated literal
            if blame:
                # Resolve current conflict clause with the antecedent and recurse
                return self.conflict_analysis(conflict_clause.resolve(blame, conflict_literal))
        # If no resolution possible, return failure
        return None, -1

    def pick_branching_variable(self):
        # Pick the next branching variable using a frequency-based heuristic:
        # Choose the most frequent unassigned literal (or 1 if none found)
        literal_frequency = self.expression.get_counter()
        return next((lit for lit in literal_frequency if lit not in self.graph.assigned_vars and -lit not in self.graph.assigned_vars), 1 if literal_frequency else None)

    def is_all_assigned(self):
        # Teue if all variables are assigned.
        return self.num_variables == len(self.graph.assigned_vars)

    def solve(self):
        # Main CDCL SAT solving loop
        solved = False
        start_time = time.time()
        # Initial unit propagation
        self.is_sat, self.conflict = self.expression.unit_propagate(self.current_decision_level, self.graph)
        while self.sat_status == 0 and not solved:
            # If all variables are assigned, the formula is SAT
            if self.is_all_assigned():
                self.sat_status = 1
                break
            # Pick a new decision literal and increment decision level
            decision_lit = self.pick_branching_variable()
            self.num_decisions += 1
            self.current_decision_level += 1
            self.graph.add_node(decision_lit, None, self.current_decision_level)
            if self.verbose_output:
                print(f'@{self.current_decision_level:>4}  [D] {decision_lit}')
            # Propagate the decision literal
            self.sat_status, self.current_conflict = self.expression.propagate_constr(decision_lit, self.current_decision_level, self.graph)
            # Perform unit propagation if still undecided
            if self.sat_status == 0:
                self.sat_status, self.current_conflict = self.expression.unit_propagate(self.current_decision_level, self.graph)
            # If SAT, break
            if self.sat_status == 1:
                break
            # Conflict analysis and learning loop
            while self.sat_status == -1 and not solved:
                learnt_clause, backjump_level = self.conflict_analysis(self.current_conflict)
                if self.verbose_output:
                    print(f'@{self.current_decision_level:>4}  [C] {self.current_conflict}')
                # If too many conflicts, trigger a restart
                if backjump_level == -100:
                    if self.verbose_output:
                        print(f'@{self.current_decision_level:>4} [R]\n')
                    self.restart()
                    break
                # If root conflict, UNSAT
                if backjump_level == -1:
                    self.sat_status = -1
                    solved = True
                    break
                # If a learned clause is found, add it
                if learnt_clause is not None:
                    if self.verbose_output:
                        print(f'@{self.current_decision_level:>4}  [L] {learnt_clause}')
                    self.expression.add_clause(learnt_clause)
                    self.num_learnt_clauses += 1
                # Backjump in the implication graph and formula
                self.graph.backjump(backjump_level)
                if self.verbose_output:
                    print(f'@{self.current_decision_level:>4} ~~> {backjump_level}\n')
                self.expression.backjump(backjump_level, self.graph)
                self.current_decision_level = backjump_level
                # Resume propagation after backjump
                self.sat_status, self.current_conflict = self.expression.unit_propagate(backjump_level, self.graph)
                self.num_conflicts += 1
                # If too many conflicts, trigger a restart
                if self.num_conflicts > self.restart_conflict_threshold:
                    if self.verbose_output:
                        print(f'@{self.current_decision_level:>4} [R]\n')
                    self.restart()
                    break
        # If SAT, print satisfying assignment
        if self.sat_status == 1:
            assignment = {abs(lit): lit > 0 for lit in self.graph.assigned_vars}
            print("\nSolution: ", end="")
            print(", ".join(str(var if assignment[var] else -var) for var in sorted(assignment)))
        # Print statistics and result
        print(f"[R]: {self.num_restarts}, [L]: {self.num_learnt_clauses}, [D]: {self.num_decisions}, [T]: {time.time()-start_time:.6f}s")
        print("SAT" if self.sat_status == 1 else "UNSAT")
        return self.sat_status

# This is more or less what the prof did.
def parse(filename, verbose):
    clauses = []
    for line in open(filename):
        # Skip lines after %
        if line[0] == '%': break
        # Skip comment lines
        if line.startswith('c'):
            continue
        # Parse the problem line to get the number of variables
        if line.startswith('p'):
            nvars = int(line.split()[2])
            print(f'File: {filename}, Variables: {line.split()[2]}, Clauses: {line.split()[3]}')
            continue
        # Parse each clause: remove trailing '0' and split into integers
        clause = [int(x) for x in line[:-2].split()]
        clauses.append(clause)
    # Return the list of clauses and the number of variables
    return clauses, nvars


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CDCL SAT Solver")
    ap.add_argument('-c', '--cnf', required=True, help='DIMACS CNF file')
    ap.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = ap.parse_args()
    Solver(args.cnf, verbose=args.verbose).solve()