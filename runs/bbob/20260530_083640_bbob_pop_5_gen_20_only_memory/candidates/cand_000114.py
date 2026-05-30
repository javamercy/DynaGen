import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        pop_size = max(10, min(2 * dim, (budget - evals) // 2))
        if pop_size < 2:
            pop_size = 2

        # Latin Hypercube sampling for initial swarm
        def lhs(n, d):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = (perm[i] + rng.uniform()) / n
            return lb + (ub - lb) * samples

        swarm = lhs(pop_size, dim)
        velocity = rng.uniform(-0.5, 0.5, size=(pop_size, dim)) * (ub - lb)

        pbest = swarm.copy()
        pbest_val = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(swarm[i])
            evals += 1
            pbest_val[i] = val
            if val < best_val:
                best_val = val
                best_x = swarm[i].copy()
                report_best(best_val, best_x)

        gbest = best_x.copy()
        gbest_val = best_val

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        v_max = 0.5 * (ub - lb)

        # Determine number of generations
        remaining = budget - evals
        max_gen = remaining // pop_size
        for gen in range(max_gen):
            w = w_start - (w_start - w_end) * (gen / max_gen)
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocity[i] = w * velocity[i] + c1 * r1 * (pbest[i] - swarm[i]) + c2 * r2 * (gbest - swarm[i])
                velocity[i] = np.clip(velocity[i], -v_max, v_max)
                swarm[i] = swarm[i] + velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                val = func(swarm[i])
                evals += 1
                if val < pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = swarm[i].copy()
                if val < best_val:
                    best_val = val
                    best_x = swarm[i].copy()
                    gbest = best_x.copy()
                    gbest_val = best_val
                    report_best(best_val, best_x)
            if evals >= budget:
                break

        # Use any remaining budget for random perturbations around best
        while evals < budget:
            candidate = best_x + rng.normal(0, 0.1 * (ub - lb), dim)
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)

        return best_val, best_x