import pulp
from pulp.apis.core import PulpSolverError
import logging
import time
# try:
from baseline.shared_utils import *
# except:
#     from shared_utils import *


"""
Returns: Lpproblem, status, clusters
"""


def frequency_distributor_lp(C, S, k, groups, alpha, beta, lamb, d, solver=None, reps=5):
    name_map = {}

    def _remap(signature):
        """Remaps variable names so that they are not too long"""
        if signature not in name_map:
            name_map[signature] = f"x{len(name_map)}"
        return name_map[signature]

    t_setup = time.time()
    # d = distance_matrix(C, S)
    # logging.info("distance computation %f s", time.time() - t_setup)

    l = len(groups)
    points = {i: [] for i in range(len(C))}
    for g_i in range(l):
        for point in groups[g_i]:
            points[point].append(g_i)

    """
    joiners is a dictionary from a tuple of indices (representing S' \subset S) to list of points indices from C
    Example: (0,2,3):{ (0,1) : [2,4]} means that the J_{S'} with S'={0,2,3} has the points 2,4 from C with signatue (0,1)
    """
    joiners = {}
    for i in range(len(C)):
        connect_to = []
        for j in range(len(S)):
            d_i_j = d[i, j]
            if d_i_j <= lamb:
                connect_to.append(j)
        if len(connect_to) == 0:
            print("lp distributor has no feasible solution.")
            return None, 0, None, None

        Sprime = tuple(connect_to)

        signature = tuple(points[i])

        if Sprime not in joiners:
            joiners[Sprime] = {}
        if signature not in joiners[Sprime]:
            joiners[Sprime][signature] = []

        joiners[Sprime][signature].append(i)

    """
    For each signature c, S',  and j in S' such that L(c, S')>0, create a variable
    """
    variables = {}
    for Sprime in joiners.keys():
        for c in joiners[Sprime].keys():
            for j in Sprime:
                variable_sig = _remap(
                    tuple([tuple(Sprime), tuple([c]), tuple([j])]))
                # print(ast.literal_eval(str(variable_sig)))
                variables[variable_sig] = p.LpVariable(
                    str(variable_sig).replace(' ', ''), lowBound=0)

    logging.info("There are %d variables", len(variables))

    obj = 1
    Lp_prob = p.LpProblem('Problem', p.LpMaximize)
    Lp_prob += obj

    """
    First set of constraints, fairness constraints
    """
    l = len(groups)
    for j in range(len(S)):
        # First, get all variables that point to cluster j
        all_vars = []
        for Sprime in joiners.keys():
            if j in Sprime:
                for c in joiners[Sprime].keys():
                    var_sig = _remap(
                        tuple([tuple(Sprime), tuple([c]), tuple([j])]))
                    all_vars.append(variables[var_sig])
        all_vars_sum = p.lpSum(all_vars)

        for a in range(l):
            # Next get variables that point to cluster j AND has belongs to group a
            vars_in_group = []
            for Sprime in joiners.keys():
                if j in Sprime:
                    for c in joiners[Sprime].keys():
                        if a in c:
                            var_sig = _remap(tuple(
                                [tuple(Sprime), tuple([c]), tuple([j])]))
                            vars_in_group.append(variables[var_sig])

            # Finally, add the alpha and beta constraints
            vars_in_group_sum = p.lpSum(vars_in_group)

            Lp_prob += vars_in_group_sum <= alpha[a]*all_vars_sum
            Lp_prob += vars_in_group_sum >= beta[a]*all_vars_sum

    """
    Second set of constraints, conservation of points
    """
    for Sprime in joiners.keys():
        for c in joiners[Sprime].keys():
            L_c_Sprime = len(joiners[Sprime][c])

            points_sprime_c = []
            for j in Sprime:
                var_sig = _remap(
                    tuple([tuple(Sprime), tuple([c]), tuple([j])]))
                points_sprime_c.append(variables[var_sig])

            Lp_prob += p.lpSum(points_sprime_c) == L_c_Sprime
    elapsed_setup = time.time() - t_setup
    logging.info("Setup time: %f", elapsed_setup)

    t_solve = time.time()
    solver_cmd = solver if solver is not None else solver_cmd
    try:
        status = Lp_prob.solve(solver_cmd)
    except PulpSolverError:
        status = 0
    elapsed_solve = time.time() - t_solve
    logging.info("Solve time: %f", elapsed_solve)

    if p.LpStatus[status] != 'Optimal':
        return None, status, None, None
    logging.info("The problem was feasible")

    best_cluster = None
    lowest_violation = 2e9
    for _ in range(reps):
        clusters = {i: [] for i in range(k)}

        for Sprime in joiners.keys():
            for c in joiners[Sprime].keys():
                L_c_Sprime = len(joiners[Sprime][c])
                probs = [variables[_remap(tuple([tuple(Sprime), tuple([c]), tuple([j])]))].value(
                )/L_c_Sprime for j in Sprime]
                assert abs(sum(probs)-1) < 1e-4
                probs = np.array(probs)
                probs[probs < 0] = 0
                probs = probs / np.sum(probs)
                draw = np.random.choice(Sprime, L_c_Sprime, p=probs)
                for i, idx in enumerate(joiners[Sprime][c]):
                    to = draw[i]
                    clusters[to].append(idx)
        violations = max_add_violation(
            C, S, groups, clusters, alpha, beta, points)
        if violations < lowest_violation:
            lowest_violation = violations
            best_cluster = clusters
    clusters = best_cluster
    tot_points = 0
    for i in clusters.keys():
        tot_points += len(clusters[i])
    assert tot_points == len(C)
    return Lp_prob, status, clusters, points


def fair_k_clustering(C, F, k, groups, alpha=None, beta=None, delta=None, epsilon=1e-3):
    if delta is None:
        if alpha is None or beta is None:
            raise Exception("alpha, beta, and delta cannot be all None")
    else:
        alpha, beta = calculate_alpha_beta(C, F, k, groups, delta)
    print(alpha.shape)

    t = time.time()
    S_idx = greedy_helper(F, k)
    S = F[S_idx]
    d = distance_matrix(C, S)
    l, r = 0, 2*np.max(d)
    feasible = False
    print("Centers identified and distance matrix computed")

    while r-l > epsilon or not feasible:
        if r == l:
            raise Exception("Boom")
        lamb = (l+r)/2
        print("Binary search:", lamb,
              f"epsilon {epsilon}, r-l={r-l}, feasible: {feasible}")
        skip = False
        for i in range(len(C)):
            if d[i].min() > lamb:
                l = lamb
                feasible = False
                skip = True
                continue
        if skip:
            continue

        LP, status, clusters, points = frequency_distributor_lp(
            C, S, k, groups, alpha, beta, lamb)
        if p.LpStatus[status] == 'Optimal':
            r, feasible = lamb, True
        else:
            l, feasible = lamb, False

    time_takes = time.time()-t
    violations = max_add_violation(C, S, groups, clusters, alpha, beta, points)

    cost = max([distance_matrix([S[j]], C[clusters[j]]).max()
               if len(clusters[j]) > 0 else 0 for j in range(len(S))])
    return violations, time_takes, cost
