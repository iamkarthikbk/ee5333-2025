class SumExpression:
    def __init__(self, variables):

        # a list of Variable type objects
        self.expression = list()
        for _ in variables:
            self.expression.append(Variable(id=variables.index(_), truth=False if _ > 0 else False))
    
    # return a boolean indicating the state of this expression
    def evaluates_to(self):

        # initialize the result to False
        result = False

        # iterate through the expression and find the state of each variable
        # accumulate the state of the variable into 'result'. since this is a
        # SumExpression, break the moment our 'result' evaluates to True
        for variable in self.expression:
            result = result or variable.evaluates_to()
            if result == True:
                break
        
        return result

class Variable:
    def __init__(self, id, truth):
        # a number ot represent the variable
        self.id = id

        # boolean to indicate the sign. True is POSITIVE.
        self.truth = truth

        # capture if this state is set by the user
        self.set_by_user = False

        # capture if this state is frozen
        self.frozen = False

    # simply return a boolean indicating the state of this variable
    def evaluates_to(self):
        return self.truth


def dpll(clauses, variables):
    print(f'Clauses: {clauses},\nVariables: {variables}')
    my_variables = [Variable(id=variables.index(_), truth=False) for _ in variables]
    my_clauses = [SumExpression(clause, my_variables) for clause in clauses]
    return False, None
###############################################################################
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
        clauses.append(SumExpression(tmp))
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
    exit()
    print([(i if m[i] == True else -i) for i in range(1, len(m))])