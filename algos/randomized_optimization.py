from pulp import LpProblem, LpMaximize, LpVariable, LpInteger, lpSum, LpStatus, value


def lp_pack(inventory_plan):

    # define the problem
    prob = LpProblem("The Santa Uncertain Bags Problem", LpMaximize)

    coefficients = inventory_plan.toy_types.values()
    A = (inventory_plan.candidate_bags.loc[:, coefficients].as_matrix()).T
    b = [inventory_plan.avialable_inventory[toy_type] for toy_type in coefficients]

    # define variables
    vals_name = [str(i) for i in range(A.shape[1])]
    variables = LpVariable.dicts("x", vals_name, 0, None, LpInteger)

    # Objective (we want to maximize) c*x
    c = inventory_plan.candidate_bags['expected_score'].as_matrix()
    prob += lpSum([c[i] * variables[vals_name[i]] for i in range(A.shape[1])]), "objective"

    # Constraints (9 toy constraints (restricted by stocks) + 1 global constraint (restricted by 1000 bags))
    for i in range(A.shape[0]):
        prob += lpSum([A[i][j] * variables[vals_name[j]] for j in range(A.shape[1])]) <= b[i], ""
    prob += lpSum([variables[vals_name[i]] for i in range(A.shape[1])]) <= 1000, ""

    # Solve it
    prob.solve()
    print ("Status:", LpStatus[prob.status])

    # get variable values and reordered them properly
    vals = {}
    for v in prob.variables():
        vals[v.name] = v.varValue
    vals = [vals['x_' + str(i)] for i in range(A.shape[1])]

    # Return a dict with the score and the coeffs
    return ({'Score': value(prob.objective), 'Bags': vals})