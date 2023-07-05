import logging

from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpConstraint, LpConstraintEQ, LpConstraintGE, LpConstraintLE
from pulp.apis import COIN_CMD, CPLEX_CMD
from pulp.apis.core import PulpSolverError
import numpy as np
import time


def inner_fair_assignment(R, costs, colors, fairness_constraints, solver, integer=False):
    t_setup_start = time.time()
    vartype = "Integer" if integer else "Continuous"
    n, k = costs.shape
    ncolors = np.max(colors) + 1
    prob = LpProblem()
    vars = {}
    # Set up the variables
    for x in range(n):
        for c in range(k):
            if costs[x, c] <= R:
                vars[x, c] = LpVariable(f"x_{x}_{c}", 0, 1, cat=vartype)

    # The point should be assigned to one cluster
    for x in range(n):
        prob += (
            lpSum([vars[x, c]
                   for c in range(k) if (x, c) in vars]) == 1
        )

    # The fairness constraints should be respected
    for c in range(k):
        cluster_size = lpSum([vars[x, c]
                              for x in range(n)
                              if (x, c) in vars])
        for color in range(ncolors):
            beta, alpha = fairness_constraints[color]
            color_cluster_size = lpSum([vars[x, c]
                                        for x in range(n)
                                        if (x, c) in vars and colors[x] == color])
            prob += (
                beta * cluster_size <= color_cluster_size
            )
            prob += (
                color_cluster_size <= alpha * cluster_size
            )
    t_setup_end = time.time()
    logging.info("  Setup LP in %f s", t_setup_end - t_setup_start)

    t_solve_start = time.time()
    try:
        prob.solve(solver)
    except PulpSolverError:
        return None
    # prob.solve(COIN_CMD(mip=integer, msg=False))
    t_solve_end = time.time()
    logging.info("  Solve LP in %f s", t_solve_end - t_solve_start)

    if LpStatus[prob.status] == "Optimal":
        assignment = {}
        for x in range(n):
            for c in range(k):
                if (x, c) in vars:
                    v = vars[x, c]
                    if v.value() > 1e-10:
                        assignment[x, c] = v.value()
        return assignment
    else:
        return None


def inner_weighted_fair_assignment(R, costs, weights, fairness_constraints, solver, integer=False):
    t_setup_start = time.time()
    vartype = "Integer" if integer else "Continuous"
    n, k = costs.shape
    ncolors = weights.shape[1]
    logging.info("n=%d k=%d ncolors=%d upper bound=%d",
                 n, k, ncolors, n*k*ncolors)
    prob = LpProblem()
    vars = {}
    # Set up the variables
    for x in range(n):
        for c in range(k):
            if costs[x, c] <= R:
                for color in range(ncolors):
                    if weights[x, color] > 0:
                        vars[x, c, color] = LpVariable(
                            f"x_{x}_{c}_{color}", 0, cat=vartype)

    logging.info("There are %d variables in the problem", len(vars))

    # All the weight should be assigned
    for x in range(n):
        for color in range(ncolors):
            prob += (
                lpSum([vars[x, c, color] for c in range(k) if (
                    x, c, color) in vars]) == np.sum(weights[x, color])
            )

    # The fairness constraints should be respected
    for c in range(k):
        cluster_size = lpSum([vars[x, c, color] for x in range(n)
                             for color in range(ncolors) if (x, c, color) in vars])
        for color in range(ncolors):
            beta, alpha = fairness_constraints[color]
            color_cluster_size = lpSum(
                [vars[x, c, color] for x in range(n) if (x, c, color) in vars])
            prob += (
                beta * cluster_size <= color_cluster_size
            )
            prob += (
                color_cluster_size <= alpha * cluster_size
            )
    t_setup_end = time.time()
    logging.info("  Setup LP in %f s", t_setup_end - t_setup_start)

    t_solve_start = time.time()
    # prob.solve(COIN_CMD(mip=integer, msg=False))
    try:
        prob.solve(solver)
    except PulpSolverError:
        return None
    t_solve_end = time.time()
    logging.info("  Solve LP in %f s", t_solve_end - t_solve_start)
    if LpStatus[prob.status] == "Optimal":
        total_weight = 0
        wassignment = {}  # np.zeros((n,k,ncolors))
        for x in range(n):
            for c in range(k):
                for color in range(ncolors):
                    if (x, c, color) in vars:
                        v = vars[x, c, color]
                        total_weight += v.value()
                        if v.value() > 1e-10:
                            wassignment[x, c, color] = v.value()
                        # if v.value() > 0:
                        #     wassignment[x, c, color] = v.value()
        logging.info(f"total weight {total_weight}")
        return wassignment
    else:
        return None


def round_assignment(n, k, ncolors, input_assignment, colors, solver):
    VARIABLE = 0
    CLUSTER_CONSTRAINT = 1
    COLORED_CLUSTER_CONSTRAINT = 2

    def _setup_lp(point_ids, cluster_sizes, colored_cluster_sizes, blacklist):
        # Set up the rounding problem
        lp = LpProblem()
        # 1. set up the variables
        vars = {}
        for x in point_ids:
            for c in range(k):
                if (x, c) in input_assignment and input_assignment[x, c] > 1e-10 and not (VARIABLE, x, c) in blacklist:
                    vars[x, c] = LpVariable(f"x_{x}_{c}", 0, 1)
        # 2. add constraints on the assignment to clusters
        for x in point_ids:
            lp += LpConstraint(lpSum([vars[x, c]
                                      for c in range(k)
                                      if (x, c) in vars]),
                               LpConstraintEQ,
                               f"assign_{x}",
                               1)
        # 3. add constraints on cluster size
        for c in range(k):
            if (CLUSTER_CONSTRAINT, c) in blacklist:
                continue
            csize = lpSum([vars[x, c] for x in point_ids if (x, c) in vars])
            lp += LpConstraint(csize, LpConstraintGE, f"lower_csize_{c}",
                               np.floor(cluster_sizes[c]))
            lp += LpConstraint(csize, LpConstraintLE, f"upper_csize_{c}",
                               np.ceil(cluster_sizes[c]))
        # 4. add constraints on colored cluster size
        for c in range(k):
            for color in range(ncolors):
                if (COLORED_CLUSTER_CONSTRAINT, c) in blacklist:
                    continue
                csize = lpSum([vars[x, c]
                               for x in point_ids
                               if (x, c) in vars and colors[x] == color])
                lp += LpConstraint(csize,
                                   LpConstraintGE,
                                   f"lower_csize_{c}_{color}",
                                   np.floor(colored_cluster_sizes[c, color]))
                lp += LpConstraint(csize,
                                   LpConstraintLE,
                                   f"upper_csize_{c}_{color}",
                                   np.ceil(colored_cluster_sizes[c, color]))
        return lp, vars

    output_assignment = np.ones(n, dtype="int") * 999999

    # Compute cluster sizes
    cluster_sizes = np.zeros(k)
    colored_cluster_sizes = np.zeros((k, ncolors))
    for (x, c) in input_assignment:
        z = input_assignment[x, c]
        if z == 1:
            output_assignment[x] = c
        else:
            cluster_sizes[c] += z
            colored_cluster_sizes[c, colors[x]] += z

    point_ids = set(x for x in range(n) if output_assignment[x] > k)
    blacklist = set()

    while len(point_ids) > 0:
        lp, vars = _setup_lp(point_ids,
                             cluster_sizes,
                             colored_cluster_sizes,
                             blacklist)
        try:
            lp.solve(solver)
        except PulpSolverError:
            return None
        # lp.solve(COIN_CMD(mip=False, msg=False))
        assert LpStatus[lp.status] == "Optimal"

        residual = np.zeros(k)
        color_residual = np.zeros((k, ncolors))
        for (x, c), v in vars.items():
            if v.value() == 1:
                output_assignment[x] = int(c)
                cluster_sizes[c] -= 1
                colored_cluster_sizes[c, colors[x]] -= 1
                point_ids.remove(x)
            elif v.value() == 0:
                blacklist.add((VARIABLE, x, c))
            else:
                residual[c] += 1
                color_residual[c, colors[x]] += 1
        for c in range(k):
            if residual[c] <= 3:
                blacklist.add((CLUSTER_CONSTRAINT, c))
            for color in range(ncolors):
                if color_residual[c, color] <= 3:
                    blacklist.add((COLORED_CLUSTER_CONSTRAINT, c, color))

    return output_assignment


def weighted_round_assignment(R, n, k, ncolors, costs, input_assignment, weights, solver):
    VARIABLE = 0
    CLUSTER_CONSTRAINT = 1
    COLORED_CLUSTER_CONSTRAINT = 2

    logging.info("Rounding with radius %f", R)

    def _setup_lp(point_ids, weights, cluster_sizes, colored_cluster_sizes, blacklist):
        print("total weight to assign ", np.sum(weights))
        lp = LpProblem()

        # 1. set up the variables
        vars = {}
        for (x, color) in point_ids:
            for c in range(k):
                if costs[x, c] <= R and weights[x, color] > 0 and (VARIABLE, x, c, color) not in blacklist:
                    vars[x, c, color] = LpVariable(
                        f"z_{x}_{c}_{color}", 0, 1)
        # 2. add constraints on the assignment to clusters
        for (x, color) in point_ids:
            pvars = [vars[x, c, color]
                     for c in range(k)
                     if (x, c, color) in vars]
            lp += LpConstraint(lpSum(pvars),
                               LpConstraintEQ,
                               f"assign_{x}_{color}",
                               weights[x, color])
        # 3. add the constraints on the cluster sizes
        for c in range(k):
            if (CLUSTER_CONSTRAINT, c) in blacklist:
                continue
            cvars = [vars[x, cluster, color]
                     for (x, cluster, color) in vars if cluster == c]
            csize = lpSum(cvars)
            lp += LpConstraint(csize, LpConstraintGE, f"lower_csize_{c}",
                               np.floor(cluster_sizes[c]))
            lp += LpConstraint(csize, LpConstraintLE, f"upper_csize_{c}",
                               np.ceil(cluster_sizes[c]))
        # 4. add constraints on colored cluster size
        for c in range(k):
            for color in range(ncolors):
                if (COLORED_CLUSTER_CONSTRAINT, c, color) in blacklist:
                    continue
                cvars = [vars[x, cluster, ccolor]
                         for (x, cluster, ccolor) in vars
                         if cluster == c and ccolor == color]
                logging.debug("c=%d, color=%d, len(cvars) = %d, colored_cluster_size = %d", c, color, len(
                    cvars), colored_cluster_sizes[c, color])
                csize = lpSum(cvars)
                lp += LpConstraint(csize,
                                   LpConstraintGE,
                                   f"lower_csize_{c}_{color}",
                                   np.floor(colored_cluster_sizes[c, color]))
                lp += LpConstraint(csize,
                                   LpConstraintLE,
                                   f"upper_csize_{c}_{color}",
                                   np.ceil(colored_cluster_sizes[c, color]))

        return lp, vars

    output_assignment = np.zeros((n, k, ncolors))

    orig_weights = weights.copy()
    weights = weights.copy().astype(np.float64)
    check = np.zeros_like(weights, dtype=np.float64)
    cluster_sizes = np.zeros(k)
    colored_cluster_sizes = np.zeros((k, ncolors))
    # Build the partial assignment with the floor of each weight
    for (x, c, color) in input_assignment:
        z = input_assignment[x, c, color]
        check[x, color] += z
        floor = np.floor(z)
        output_assignment[x, c, color] = int(floor)
        weights[x, color] -= floor
        if z != floor:
            cluster_sizes[c] += z - floor
            colored_cluster_sizes[c, color] += z - floor

    point_ids = set((x, color)
                    for x in range(n)
                    for color in range(ncolors)
                    if weights[x, color] > 0)
    blacklist = set()

    while np.sum(weights) > 0:
        lp, vars = _setup_lp(point_ids,
                             weights,
                             cluster_sizes,
                             colored_cluster_sizes,
                             blacklist)
        try:
            lp.solve(solver)
        except PulpSolverError:
            return None
        # lp.solve(COIN_CMD(mip=False, msg=False))
        logging.info("status: %s", LpStatus[lp.status])
        assert LpStatus[lp.status] == "Optimal"

        residual = np.zeros(k)
        color_residual = np.zeros((k, ncolors))
        for (x, c, color), v in vars.items():
            if v.value() == 1:
                logging.debug("assign %d (color %d) to center %d", x, color, c)
                output_assignment[x, c, color] += 1
                cluster_sizes[c] -= 1
                colored_cluster_sizes[c, color] -= 1
                weights[x, color] -= 1
            elif v.value() == 0:
                blacklist.add((VARIABLE, x, c, color))
            else:
                residual[c] += 1
                color_residual[c, color] += 1
        for c in range(k):
            if residual[c] <= 3:
                blacklist.add((CLUSTER_CONSTRAINT, c))
            for color in range(ncolors):
                if color_residual[c, color] <= 3:
                    blacklist.add((COLORED_CLUSTER_CONSTRAINT, c, color))

    logging.info("total output weight %f", np.sum(output_assignment))
    assert np.sum(output_assignment) == np.sum(orig_weights)

    return output_assignment


def fair_assignment(centers, costs, colors, fairness_contraints, solver):
    k = centers.shape[0]
    n = colors.shape[0]
    ncolors = np.max(colors) + 1
    allcosts = np.sort(np.unique(costs))

    def binary_search():
        def relative_difference(high, low):
            return (allcosts[high] - allcosts[low]) / allcosts[high]

        last_valid = None
        low, high = 0, allcosts.shape[0] - 1
        # Run while the relative difference is more than 1%
        while last_valid is None or relative_difference(high, low) >= 0.01:
            logging.info("Relative difference %f",
                         relative_difference(high, low))
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = inner_fair_assignment(
                R, costs, colors, fairness_contraints, solver)
            if low == high:
                assert assignment is not None
                return assignment
            if assignment is None:
                low = mid + 1
            else:
                last_valid = assignment
                if low == mid:
                    return assignment
                else:
                    high = mid
        assert last_valid is not None
        return last_valid

    assignment = binary_search()
    assignment = round_assignment(
        n, k, ncolors, assignment, colors, solver)
    return centers, assignment


def _weighted_assignment_radius(costs, centers, weighted_assignment):
    max_cost = 0
    if hasattr(weighted_assignment, "items"):
        for (x, c, color), w in weighted_assignment.items():
            cost = costs[x, c]
            if w > 0 and cost > max_cost:
                max_cost = cost
    else:
        # we have a numpy array
        n, k, ncolors = weighted_assignment.shape
        for x in range(n):
            for c in range(k):
                for color in range(ncolors):
                    w = weighted_assignment[x, c, color]
                    if w > 0:
                        cost = costs[x, c]
                        if cost > max_cost:
                            max_cost = cost
    return max_cost


def weighted_fair_assignment(centers, costs, weights, fairness_contraints, solver):
    k = centers.shape[0]
    n, ncolors = weights.shape
    allcosts = np.sort(np.unique(costs))

    def binary_search():
        def relative_difference(high, low):
            return (allcosts[high] - allcosts[low]) / allcosts[high]

        last_valid = None
        low, high = 0, allcosts.shape[0] - 1
        # while low < high:
        while last_valid is None or relative_difference(high, low) >= 0.1:
            logging.info("Relative difference %f",
                         relative_difference(high, low))
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = inner_weighted_fair_assignment(
                R, costs, weights, fairness_contraints, solver)
            if low == high:
                return assignment, R
            if assignment is None:
                low = mid + 1
            else:
                last_valid = assignment
                last_R = R
                if low == mid:
                    return assignment, R
                else:
                    high = mid
        assert last_valid is not None
        logging.info("returning last_valid with total weight %.10f",
                     sum(last_valid.values()))
        return last_valid, last_R

    wassignment, R = binary_search()
    fradius = _weighted_assignment_radius(costs, centers, wassignment)
    logging.info("Radius of the fractional weight assignment %f", fradius)

    assignment = weighted_round_assignment(
        R, n, k, ncolors, costs, wassignment, weights, solver)
    radius = _weighted_assignment_radius(costs, centers, assignment)
    logging.info("Radius of the weight assignment after rounding %f", radius)
    assert radius <= R
    return centers, assignment


def inner_freq_distributor(R, costs, weights, fairness_constraints, solver):
    logging.info(R)
    joiners = {}
    k = costs.shape[1]
    n, ncolors = weights.shape

    joiner_names = {}

    def var_name(joiner_centers, center, color):
        if joiner_centers not in joiner_names:
            joiner_names[joiner_centers] = len(joiner_names)
        jid = joiner_names[joiner_centers]
        return f"x_{jid},{center},{color}"

    for x in range(n):
        reaching_centers = frozenset((costs[x] <= R).nonzero()[0])
        if reaching_centers not in joiners:
            joiners[reaching_centers] = []
        joiners[reaching_centers].append(x)

    logging.info("there are %d joiners", len(joiners))

    lp = LpProblem()
    vars = {}

    # Variables and assignment constaints
    for joiner_centers, xs in joiners.items():
        for color in range(ncolors):
            joiner_vars = []
            joiner_weight = np.sum(weights[xs, color])
            if joiner_weight > 0:
                for c in joiner_centers:
                    vname = var_name(joiner_centers, c, color)
                    v = LpVariable(vname, 0)
                    vars[joiner_centers, c, color] = v
                    joiner_vars.append(v)
                lp += lpSum(joiner_vars) == joiner_weight
    logging.info("there are %d variables", len(vars))

    # Fairness constraints
    for c in range(k):
        cluster_vars = [var for (joiner_centers, cc, ccolor), var in vars.items()
                        if c in joiner_centers and cc == c]
        cluster_size = lpSum(cluster_vars)
        assert len(cluster_vars) > 0
        for color in range(ncolors):
            beta, alpha = fairness_constraints[color]
            assert alpha >= beta
            colored_cluster_vars = [
                var for (joiner_centers, cc, ccolor), var in vars.items()
                if c in joiner_centers and color == ccolor and cc == c
            ]
            assert len(colored_cluster_vars) <= len(cluster_vars)
            colored_cluster_size = lpSum(colored_cluster_vars)
            lp += LpConstraint(beta * cluster_size - colored_cluster_size,
                               LpConstraintLE,
                               f"lower_{c}_{color}",
                               0)
            lp += LpConstraint(alpha * cluster_size - colored_cluster_size,
                               LpConstraintGE,
                               f"upper_{c}_{color}",
                               0)

    logging.info("there are %d constraints", len(lp.constraints))

    # solve the problem
    try:
        lp.solve(solver)
        status = lp.status
    except PulpSolverError:
        status = -1

    if LpStatus[status] == "Optimal":
        logging.info("LP is feasible")
        toassign = weights.copy().astype(np.float64)
        wassignment = {}  # np.zeros((n,k,ncolors))
        for (joiner_id, c, color), var in vars.items():
            budget = var.value()
            points = joiners[joiner_id]
            for x in points:
                if budget == 0:
                    break
                w = toassign[x, color]
                if w > 0:
                    if budget >= w:
                        wassignment[x, c, color] = w
                        toassign[x, color] = 0
                        budget -= w
                    else:
                        # insufficient budget, partial assignment
                        ww = budget
                        wassignment[x, c, color] = ww
                        toassign[x, color] -= ww
                        budget = 0
            assert budget <= 1e-7, f"Leftover budget {budget} > 0"

        logging.info("leftover weight %f", np.sum(toassign))
        logging.info("assigned weight %f", sum(wassignment.values()))
        logging.info("target weight %f", np.sum(weights))
        assert np.isclose(sum(wassignment.values()), np.sum(weights))

        return wassignment
    else:
        logging.info("Infeasible")
        return None


def freq_distributor(centers, costs, weights, fairness_constraints, solver):
    k = costs.shape[1]
    n, ncolors = weights.shape
    print(n, ncolors)
    print(fairness_constraints)
    allcosts = np.sort(np.unique(costs))
    print("max distance", np.max(allcosts))

    def binary_search():
        def relative_difference(high, low):
            return (allcosts[high] - allcosts[low]) / allcosts[high]

        last_valid = None
        last_radius = None
        low, high = 0, allcosts.shape[0] - 1
        # Run while the relative difference is more than 1%
        while last_valid is None or relative_difference(high, low) >= 0.1:
            logging.info("Relative difference %f",
                         relative_difference(high, low))
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = inner_freq_distributor(
                R, costs, weights, fairness_constraints, solver)
            if low == high:
                assert assignment is not None
                return assignment, R
            if assignment is None:
                low = mid + 1
            else:
                last_valid = assignment
                last_radius = R
                if low == mid:
                    return assignment, R
                else:
                    high = mid

        assert last_valid is not None
        return last_valid, last_radius

    wassignment, R = binary_search()
    logging.info("Radius returned by binary search %f", R)
    # inner_freq_distributor(2.41, costs, weights, fairness_constraints, solver)
    fradius = _weighted_assignment_radius(costs, centers, wassignment)
    logging.info("Radius of the fractional weight assignment %f", fradius)

    assignment = weighted_round_assignment(
        R, n, k, ncolors, costs, wassignment, weights, solver)
    radius = _weighted_assignment_radius(costs, centers, assignment)
    logging.info("Radius of the weight assignment after rounding %f", radius)
    assert radius <= R
    return centers, assignment
