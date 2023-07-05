from .lp_freq_distributor import *
from .shared_utils import *
import logging


class KFC(object):
    def __init__(self, k, cplex_path=None, seed=123, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.seed = seed
        self.solver_cmd = pulp.CPLEX_CMD(
            path=cplex_path, msg=0) if cplex_path is not None else pulp.COIN_CMD(msg=False)
        logging.info("solver: %s", self.solver_cmd)

    def attrs(self):
        return {
            "seed": self.seed,
            "epsilon": self.epsilon
        }

    def additional_metrics(self):
        return {}

    def name(self):
        return "kfc-k-center"

    def time(self):
        return self.elapsed

    def fit_predict(self, C, colors, fairness_constraints):
        np.random.seed(self.seed)

        # Preprocessing
        groups = {}
        for color in range(colors.max()+1):
            groups[color] = (colors == color).nonzero()[0].tolist()

        alpha = np.array([tup[1] for tup in fairness_constraints])
        beta = np.array([tup[0] for tup in fairness_constraints])

        # Start the algorithm

        start = time.time()

        k = self.k
        epsilon = self.epsilon
        F = C
        S_idx = greedy_helper(F, k)
        S = F[S_idx]
        d = distance_matrix(C, S)
        l, r = 0, 2*np.max(d)
        feasible = False

        while r-l > epsilon or not feasible:
            bs_iteration_start = time.time()
            if r == l:
                raise Exception("Boom")
            lamb = (l+r)/2
            logging.info(
                f"Binary search {lamb}: epsilon {epsilon}, r-l={r-l}")
            skip = False
            for i in range(len(C)):
                if d[i].min() > lamb:
                    l = lamb
                    feasible = False
                    skip = True
                    continue
            if skip:
                continue

            logging.info("Start frequency distributor with radius %f", lamb)
            fd_start = time.time()
            LP, status, clusters, points = frequency_distributor_lp(
                C, S, k, groups, alpha, beta, lamb, solver=self.solver_cmd)
            logging.info("completed frequency distributor in time %f",
                         time.time() - fd_start)
            if p.LpStatus[status] == 'Optimal':

                r, feasible = lamb, True
            else:
                l, feasible = lamb, False

            logging.info("Binary search iteration %f",
                         time.time() - bs_iteration_start)

        end = time.time()
        self.elapsed = end - start

        # Build the assignment
        self.centers = np.array(S_idx)
        self.assignment = np.ones(C.shape[0], np.int32) * 9999
        for c, xs in clusters.items():
            for x in xs:
                self.assignment[x] = c

        assert self.assignment.max() <= k, "there are some unassigned points!"
        return self.assignment
