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

        def clip(x):
            return np.clip(x, lb, ub)

        # Population size: larger for exploration
        pop_size = max(10, min(50, dim * 2, budget // 2))
        if pop_size < 2:
            pop_size = 2

        # Initial population
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # DE parameters for exploration
        F = 1.0
        CR = 1.0

        stag_count = 0
        stag_limit = 3  # generations without improvement before restart

        # Main DE loop
        while budget > 0:
            # Generate one generation
            new_pop = np.empty_like(pop)
            new_f = np.empty(pop_size)
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select three distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover: CR=1 means all dimensions from mutant
                trial = mutant
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                new_pop[i] = trial
                new_f[i] = trial_f
                if trial_f < pop_f[i]:
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        improved = True
                else:
                    new_pop[i] = pop[i]
                    new_f[i] = pop_f[i]

            if budget <= 0:
                break

            pop = new_pop
            pop_f = new_f

            if improved:
                stag_count = 0
            else:
                stag_count += 1
                if stag_count >= stag_limit:
                    # Restart: reinitialize all points except best
                    # Keep best in population
                    new_pop = np.empty_like(pop)
                    new_f = np.empty(pop_size)
                    # Keep best
                    new_pop[0] = best_x
                    new_f[0] = best_f
                    # Random rest for others
                    for i in range(1, pop_size):
                        new_pop[i] = rng.uniform(lb, ub, size=dim)
                        if budget <= 0:
                            break
                        new_f[i] = func(new_pop[i])
                        budget -= 1
                        if new_f[i] < best_f:
                            best_x = new_pop[i].copy()
                            best_f = new_f[i]
                            report_best(best_f, best_x)
                    pop = new_pop
                    pop_f = new_f
                    stag_count = 0

        # Final random search to explore more
        while budget > 0:
            trial = rng.uniform(lb, ub, size=dim)
            trial_f = func(trial)
            budget -= 1
            if trial_f < best_f:
                best_x = trial.copy()
                best_f = trial_f
                report_best(best_f, best_x)

        return best_f, best_x