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

        pop_size = min(20, max(4, budget // 2))
        if budget < pop_size:
            # fallback: random search
            best_x = rng.uniform(lb, ub, dim)
            best_f = func(best_x)
            budget -= 1
            report_best(best_f, best_x)
            while budget > 0:
                x = rng.uniform(lb, ub, dim)
                f = func(x)
                budget -= 1
                if f < best_f:
                    best_x = x.copy()
                    best_f = f
                    report_best(best_f, best_x)
            return best_f, best_x

        # initialization
        pop = rng.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.8
        CR = 0.9
        stagnation_limit = max(1, budget // (2 * pop_size))
        stagnation_counter = 0

        while budget > 0:
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # select r1, r2 distinct from i
                indices = [j for j in range(pop_size) if j != i]
                if len(indices) < 2:
                    continue
                r1, r2 = rng.choice(indices, 2, replace=False)
                mutant = pop[i] + F * (pop[r1] - pop[r2]) + F * (best_x - pop[i])
                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
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

            if not improved and budget >= pop_size:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit:
                    # restart: keep best, replace others with random
                    new_pop = rng.uniform(lb, ub, (pop_size - 1, dim))
                    new_pop_f = np.full(pop_size - 1, np.inf)
                    for k in range(pop_size - 1):
                        if budget <= 0:
                            break
                        new_pop_f[k] = func(new_pop[k])
                        budget -= 1
                    pop = np.vstack((best_x.reshape(1, -1), new_pop))
                    pop_f = np.concatenate(([best_f], new_pop_f))
                    stagnation_counter = 0
            else:
                stagnation_counter = 0

        return best_f, best_x