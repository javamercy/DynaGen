import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # Population size: larger for exploration
        pop_size = min(30, max(5, budget // 2))
        if budget < pop_size:
            pop_size = max(1, budget)

        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1
            if budget <= 0:
                best_idx = np.argmin(pop_f[:i+1])
                best_x = pop[best_idx].copy()
                best_f = pop_f[best_idx]
                report_best(best_f, best_x)
                return best_f, best_x

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        CR = 0.9
        stagnation_limit = max(1, budget // (3 * pop_size))
        stagnation_counter = 0

        while budget > 0:
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Choose two distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                r1, r2 = rng.choice(indices, 2, replace=False)
                # Sample F uniformly per individual
                F = rng.uniform(0.5, 1.0)
                # Mutation: rand-to-best, but with random F
                mut = pop[i] + F * (pop[r1] - pop[r2]) + F * (best_x - pop[i])
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mut[j]
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        improved = True
            if not improved:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit and budget >= pop_size:
                    # Restart: keep best, add perturbed best, and random
                    pert_std = 0.05 * (ub - lb)
                    best_perturbed = np.clip(best_x + rng.randn(dim) * pert_std, lb, ub)
                    new_size = pop_size - 2
                    new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                    # Evaluate perturbed best
                    if budget > 0:
                        f_pert = func(best_perturbed)
                        budget -= 1
                    else:
                        f_pert = np.inf
                    # Evaluate random individuals
                    new_pop_f = np.full(new_size, np.inf)
                    for k in range(new_size):
                        if budget <= 0:
                            break
                        new_pop_f[k] = func(new_pop[k])
                        budget -= 1
                    # Combine: best, perturbed, random
                    pop = np.vstack((best_x.reshape(1, -1), best_perturbed.reshape(1, -1), new_pop))
                    pop_f = np.concatenate(([best_f], [f_pert], new_pop_f))
                    stagnation_counter = 0
            else:
                stagnation_counter = 0

        return best_f, best_x