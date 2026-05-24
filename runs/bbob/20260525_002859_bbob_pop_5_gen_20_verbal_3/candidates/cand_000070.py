import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Swarm size: proportional to dimension, but not too large
        popsize = min(50, max(10, 4 * dim))
        if popsize > budget // 2:
            popsize = max(4, budget // 2)

        # Initialize positions and velocities
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        vel = (ub - lb) * (2 * rng.rand(popsize, dim) - 1) * 0.5

        # Personal bests
        pbest = pop.copy()
        pbest_fit = np.full(popsize, np.inf)

        evals = 0
        # Evaluate initial population
        for i in range(popsize):
            if evals >= budget:
                break
            val = func(pop[i])
            pbest_fit[i] = val
            evals += 1
            if val < self.best_value:
                self.best_value = val
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        # Global best index
        gbest_idx = np.argmin(pbest_fit)
        gbest = pbest[gbest_idx].copy()
        gbest_fit = pbest_fit[gbest_idx]

        # PSO parameters
        w_start = 0.9
        w_end = 0.4
        c1 = 2.0
        c2 = 2.0
        max_iter = int((budget - evals) / popsize) + 1
        stagnation_counter = 0
        stagnation_limit = max(dim, int(0.1 * budget / popsize))

        for gen in range(max_iter):
            if evals >= budget:
                break
            # Update inertia weight
            w = w_start - (w_start - w_end) * gen / max_iter
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # Update velocity
                r1 = rng.rand(dim)
                r2 = rng.rand(dim)
                cognitive = c1 * r1 * (pbest[i] - pop[i])
                social = c2 * r2 * (gbest - pop[i])
                vel[i] = w * vel[i] + cognitive + social
                # Update position
                pop[i] = pop[i] + vel[i]
                # Clip to bounds
                pop[i] = np.clip(pop[i], lb, ub)
                # Evaluate
                val = func(pop[i])
                evals += 1
                if val < pbest_fit[i]:
                    pbest_fit[i] = val
                    pbest[i] = pop[i].copy()
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = pop[i].copy()
                        report_best(self.best_value, self.best_x)
                        improved = True
                        gbest = self.best_x.copy()
                        gbest_fit = self.best_value
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            # Restart if stagnation and enough budget left
            if stagnation_counter >= stagnation_limit and evals < budget - popsize:
                stagnation_counter = 0
                # Keep best
                new_pop = [self.best_x.copy()]
                # Reinitialize worst half
                sorted_idx = np.argsort(pbest_fit)
                worst_indices = sorted_idx[1:]  # exclude best
                rng.shuffle(worst_indices)
                for idx in worst_indices:
                    if len(new_pop) >= popsize:
                        break
                    # Perturb best with scaled random direction
                    sigma = (ub - lb) * 0.1 * (2 * rng.rand(dim) - 1)
                    new_x = self.best_x + sigma
                    new_x = np.clip(new_x, lb, ub)
                    new_pop.append(new_x)
                # Evaluate new individuals (except best already evaluated)
                for j, x in enumerate(new_pop[1:], start=1):
                    if evals >= budget:
                        break
                    val = func(x)
                    evals += 1
                    pop[j] = x
                    pbest[j] = x
                    pbest_fit[j] = val
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                # Reset velocities for new individuals
                for j in range(popsize):
                    vel[j] = (ub - lb) * (2 * rng.rand(dim) - 1) * 0.5
                # Update global best
                gbest = self.best_x.copy()
                gbest_fit = self.best_value
        return self.best_value, self.best_x