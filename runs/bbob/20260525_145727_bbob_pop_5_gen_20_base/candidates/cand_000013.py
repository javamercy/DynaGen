import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget // 2))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = self.pop_size
        rng = self.rng
        budget = self.budget

        # LHS initialization
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            pop[:, j] = lb[j] + (perm + rng.rand(pop_size)) / pop_size * (ub[j] - lb[j])
        pop = np.clip(pop, lb, ub)

        # Evaluate initial population
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Per-individual adaptive step sizes (as fraction of bounds width)
        step_size = 0.1 * (ub - lb)  # shape (dim,) but will be broadcast
        step_sizes = np.tile(step_size, (pop_size, 1))  # each individual has own step size vector
        stagnation = np.zeros(pop_size, dtype=int)
        max_stagnation = max(15, 5 * dim)

        # Main loop
        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate random direction
                direction = rng.normal(0, 1, dim)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    direction = rng.uniform(-1, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm == 0:
                        direction = np.ones(dim)
                        norm = np.sqrt(dim)
                direction = direction / norm

                # Perturb
                candidate = pop[i] + direction * step_sizes[i]
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1

                if val < fitness[i]:
                    # Improvement
                    pop[i] = candidate
                    fitness[i] = val
                    step_sizes[i] *= 1.5
                    stagnation[i] = 0
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)

                    # Extra local perturbations along same direction
                    for _ in range(3):
                        if evals >= budget:
                            break
                        local_candidate = pop[i] + direction * step_sizes[i]
                        local_candidate = np.clip(local_candidate, lb, ub)
                        local_val = func(local_candidate)
                        evals += 1
                        if local_val < fitness[i]:
                            pop[i] = local_candidate
                            fitness[i] = local_val
                            step_sizes[i] *= 1.5
                            if local_val < best_val:
                                best_val = local_val
                                best_x = local_candidate.copy()
                                report_best(best_val, best_x)
                        else:
                            break
                else:
                    # Failure
                    step_sizes[i] *= 0.6
                    stagnation[i] += 1

                # Restart if stagnated
                if stagnation[i] >= max_stagnation and evals < budget:
                    new_x = rng.uniform(lb, ub, dim)
                    new_val = func(new_x)
                    evals += 1
                    if new_val < fitness[i]:
                        pop[i] = new_x
                        fitness[i] = new_val
                        if new_val < best_val:
                            best_val = new_val
                            best_x = new_x.copy()
                            report_best(best_val, best_x)
                    step_sizes[i] = 0.1 * (ub - lb)
                    stagnation[i] = 0

        return best_val, best_x