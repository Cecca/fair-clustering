import logging

from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpConstraint, LpConstraintEQ, LpConstraintGE, LpConstraintLE
from pulp.apis import COIN_CMD
import numpy as np
import time


def inner_fair_assignment(R, costs, colors, fairness_constraints, integer=False):
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
    prob.solve(COIN_CMD(mip=integer, msg=False))
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


def inner_weighted_fair_assignment(R, costs, weights, fairness_constraints, integer=False):
    t_setup_start = time.time()
    vartype = "Integer" if integer else "Continuous"
    n, k = costs.shape
    ncolors = weights.shape[1]
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
    prob.solve(COIN_CMD(mip=integer, msg=False))
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


def round_assignment(n, k, ncolors, input_assignment, colors):
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
        lp.solve(COIN_CMD(mip=False, msg=False))
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


def weighted_round_assignment(n, k, ncolors, input_assignment, weights):
    VARIABLE = 0
    CLUSTER_CONSTRAINT = 1
    COLORED_CLUSTER_CONSTRAINT = 2

    def _setup_lp(point_ids, weights, cluster_sizes, colored_cluster_sizes, blacklist):
        lp = LpProblem()

        # 1. set up the variables
        vars = {}
        for (x, color) in point_ids:
            for c in range(k):
                if (x,c,color) in input_assignment and (VARIABLE, x, c, color) not in blacklist:
                    vars[x, c, color] = LpVariable(
                        f"z_{x}_{c}_{color}", 0, 1)
        # 2. add constraints on the assignment to clusters
        for (x, color) in point_ids:
            lp += LpConstraint(lpSum([vars[x, c, color]
                                      for c in range(k)
                                      if (x, c, color) in vars]),
                               LpConstraintEQ,
                               f"assign_{x}_{color}",
                               weights[x, color])
        # 3. add the constraints on the cluster sizes
        for c in range(k):
            if (CLUSTER_CONSTRAINT, c) in blacklist:
                continue
            if cluster_sizes[c] == 0:
                continue
            cvars = [vars[x, cluster, color]
                     for (x, cluster, color) in vars if cluster == c]
            assert len(cvars) > 0
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
                if colored_cluster_sizes[c,color] == 0:
                    continue
                cvars = [vars[x, cluster, ccolor]
                         for (x, cluster, ccolor) in vars
                         if cluster == c and ccolor == color]
                assert len(cvars) > 0
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

    point_ids = dict(((x, color), weights[x, color])
                     for x in range(n)
                     for color in range(ncolors)
                     if weights[x, color] > 0)
    blacklist = set()

    while len(point_ids) > 0:
        logging.info("still to assign %d", len(point_ids))
        lp, vars = _setup_lp(point_ids,
                             weights,
                             cluster_sizes,
                             colored_cluster_sizes,
                             blacklist)
        lp.solve(COIN_CMD(mip=False, msg=False))
        logging.info("status: %s", LpStatus[lp.status])
        assert LpStatus[lp.status] == "Optimal"

        residual = np.zeros(k)
        color_residual = np.zeros((k, ncolors))
        for (x, c, color), v in vars.items():
            if v.value() == 1:
                output_assignment[x, c, color] += 1
                cluster_sizes[c] -= 1
                colored_cluster_sizes[c, color] -= 1
                point_ids[x, color] -= 1
                if point_ids[x, color] == 0:
                    del point_ids[x, color]
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


def fair_assignment(centers, costs, colors, fairness_contraints):
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
                R, costs, colors, fairness_contraints)
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
        n, k, ncolors, assignment, colors)
    return centers, assignment


def _weighted_assignment_radius(costs, centers, weighted_assignment):
    max_cost = 0
    if hasattr(weighted_assignment, "items"):
        for (x,c,color), w in weighted_assignment.items():
            cost = costs[x,c]
            if w > 0 and cost > max_cost:
                max_cost = cost
    else:
        # we have a numpy array
        n, k, ncolors = weighted_assignment.shape
        for x in range(n):
            for c in range(k):
                for color in range(ncolors):
                    w = weighted_assignment[x,c,color]
                    if w > 0:
                        cost = costs[x,c]
                        if cost > max_cost:
                            max_cost = cost
    return max_cost

    


def weighted_fair_assignment(centers, costs, weights, fairness_contraints):
    k = centers.shape[0]
    n, ncolors = weights.shape
    allcosts = np.sort(np.unique(costs))

    def binary_search():
        def relative_difference(high, low):
            return (allcosts[high] - allcosts[low]) / allcosts[high]

        last_valid = None
        low, high = 0, allcosts.shape[0] - 1
        while low < high:
        # while last_valid is None or relative_difference(high, low) >= 0.01:
            logging.info("Relative difference %f",
                         relative_difference(high, low))
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = inner_weighted_fair_assignment(
                R, costs, weights, fairness_contraints)
            if low == high:
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
        logging.info("returning last_valid with total weight %.10f",
                     sum(last_valid.values()))
        return last_valid

    wassignment = binary_search()
    fradius = _weighted_assignment_radius(costs, centers, wassignment)
    logging.info("Radius of the fractional weight assignment %f", fradius)

    assignment = weighted_round_assignment(
        n, k, ncolors, wassignment, weights)
    radius = _weighted_assignment_radius(costs, centers, assignment)
    logging.info("Radius of the weight assignment after rounding %f", radius)
    assert radius <= fradius
    return centers, assignment
