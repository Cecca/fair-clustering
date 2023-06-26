import logging

from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpConstraint, LpConstraintEQ, LpConstraintGE, LpConstraintLE
from pulp.apis import COIN_CMD
import numpy as np


def inner_fair_assignment(R, costs, colors, fairness_constraints, integer=False):
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

    prob.solve(COIN_CMD(mip=integer, msg=False))
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

    prob.solve(COIN_CMD(mip=integer, msg=False))
    if LpStatus[prob.status] == "Optimal":
        wassignment = {}  # np.zeros((n,k,ncolors))
        for x in range(n):
            for c in range(k):
                for color in range(ncolors):
                    if (x, c, color) in vars:
                        v = vars[x, c, color]
                        if v.value() > 1e-10:
                            wassignment[x, c, color] = v.value()
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
                colored_cluster_sizes[c] -= 1
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
    output_assignment = {}
    # Dictionaries for residual variables
    vars = {}
    weights = weights.copy()
    cluster_sizes = np.zeros(k)
    colored_cluster_sizes = np.zeros((k, ncolors))
    for (x, c, color) in input_assignment:
        z = input_assignment[x, c, color]
        floor = np.floor(z)
        output_assignment[x, c, color] = int(floor)
        weights[x, color] -= floor
        cluster_sizes[c] += z - floor
        colored_cluster_sizes[c, color] += z - floor
        if z != floor:
            vars[x, c, color] = LpVariable(f"z_{x}_{c}_{color}", 0, 1)

    blacklist = set()
    blacklist_colored = set()

    logging.info("variables to round: %d", len(vars))
    while len(vars) > 0:
        logging.info("There are still %d variables to round", len(vars))
        lp = LpProblem()
        for x in range(n):
            for color in range(ncolors):
                lp += (
                    lpSum([vars[x, c, color] for c in range(k) if (
                        x, c, color) in vars]) == weights[x, color]
                )
        for c in range(k):
            if c in blacklist:
                continue
            csize = cluster_sizes[c]
            zsum = lpSum([
                vars[x, c, color]
                for x in range(n)
                for color in range(ncolors)
                if (x, c, color) in vars
            ])
            lp += (zsum >= np.floor(csize))
            lp += zsum <= np.ceil(csize)
        for c in range(k):
            for color in range(ncolors):
                if (c, color) in blacklist_colored:
                    continue
                ccsize = colored_cluster_sizes[c, color]
                zsum = lpSum([
                    vars[x, c, color]
                    for x in range(n)
                    if (x, c, color) in vars
                ])
                lp += (zsum >= np.floor(ccsize))
                lp += zsum <= np.ceil(ccsize)

        lp.solve(COIN_CMD(mip=False, msg=False))
        assert LpStatus[lp.status] == "Optimal"
        blacklist_counts = {}
        blacklist_counts_colored = {}
        for (x, c, color) in vars.copy():
            if vars[x, c, color].value() == 0:
                del vars[x, c, color]
            elif vars[x, c, color].value() == 1:
                output_assignment[x, c, color] += 1
                cluster_sizes[c] -= 1
                colored_cluster_sizes[c, color] -= 1
                del vars[x, c, color]
            else:
                # Count how many point may go in a cluster
                if c not in blacklist_counts:
                    blacklist_counts[c] = 0
                blacklist_counts[c] += 1
                if (c, color) not in blacklist_counts_colored:
                    blacklist_counts_colored[c, color] = 0
                blacklist_counts_colored[c, color] += 1
        for c in blacklist_counts:
            if blacklist_counts[c] <= 3:
                blacklist.add(c)
        for c, color in blacklist_counts_colored:
            if blacklist_counts_colored[c, color] <= 3:
                blacklist_colored.add((c, color))

    return output_assignment


def fair_assignment(centers, costs, colors, fairness_contraints):
    k = centers.shape[0]
    n = colors.shape[0]
    ncolors = np.max(colors) + 1
    allcosts = np.sort(np.unique(costs))

    def binary_search():
        low, high = 0, allcosts.shape[0]
        while low <= high:
            mid = low + (high - low) // 2
            R = allcosts[mid]
            logging.info("R %f", R)
            assignment = inner_fair_assignment(
                R, costs, colors, fairness_contraints)
            if low == high:
                return assignment
            if assignment is None:
                low = mid + 1
            else:
                if low == mid:
                    return assignment
                else:
                    high = mid

    assignment = binary_search()
    assignment = round_assignment(
        n, k, ncolors, assignment, colors)
    return centers, assignment


def weighted_fair_assignment(centers, costs, weights, fairness_contraints):
    k = centers.shape[0]
    n, ncolors = weights.shape
    allcosts = np.sort(np.unique(costs))

    def binary_search():
        low, high = 0, allcosts.shape[0]
        while low <= high:
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
                if low == mid:
                    return assignment
                else:
                    high = mid

    wassignment = binary_search()
    assignment = weighted_round_assignment(
        n, k, ncolors, wassignment, weights)
    return centers, assignment
