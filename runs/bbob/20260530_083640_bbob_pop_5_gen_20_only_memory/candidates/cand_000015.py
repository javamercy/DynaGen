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

        # Population size: at least 10, at most budget/3
        pop_size = max(10, min(2 * dim, budget // 3))
        pop_size = min(pop_size, budget)
        if pop_size < 2:
            pop_size = 2

        # Initialize particles uniformly
        particles = rng.uniform(lb, ub, size=(pop_size, dim))
        velocities = np.zeros_like(particles)

        # Evaluate initial population
        fitness = np.full(pop_size, np.inf)
        best_global_val = np.inf
        best_global_x = None
        for i in range(pop_size):
            x = particles[i]
            val = func(x)
            fitness[i] = val
            if val < best_global_val:
                best_global_val = val
                best_global_x = x.copy()
                report_best(best_global_val, best_global_x)
        evals = pop_size

        # Personal bests
        pbest_x = particles.copy()
        pbest_val = fitness.copy()

        # PSO parameters
        c1 = 2.0
        c2 = 2.0
        w_start = 0.9
        w_end = 0.4
        max_iter = budget - evals
        if max_iter <= 0:
            return best_global_val, best_global_x
        # We'll evaluate each particle per iteration, so iterations count
        iter_count = 0
        while evals < budget and iter_count < max_iter:
            w = w_start - (w_start - w_end) * (iter_count / max_iter)
            for i in range(pop_size):
                if evals >= budget:
                    break
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (pbest_x[i] - particles[i]) +
                                 c2 * r2 * (best_global_x - particles[i]))
                # Update position
                particles[i] = particles[i] + velocities[i]
                # Clip to bounds
                particles[i] = np.clip(particles[i], lb, ub)
                # Evaluate
                val = func(particles[i])
                evals += 1
                if val < fitness[i]:
                    fitness[i] = val
                    pbest_x[i] = particles[i].copy()
                    pbest_val[i] = val
                    if val < best_global_val:
                        best_global_val = val
                        best_global_x = particles[i].copy()
                        report_best(best_global_val, best_global_x)
            iter_count += 1

        return best_global_val, best_global_x