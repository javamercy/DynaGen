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

        # Population size: at least 5, at most min(40, budget//2)
        pop_size = min(40, budget // 2)
        if pop_size < 5:
            pop_size = 5

        # Initialize population uniformly
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f[:pop_size])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # Main loop
        while budget > 0:
            # Check population diversity (mean variance across dimensions)
            mean_pop = np.mean(pop, axis=0)
            var_pop = np.mean((pop - mean_pop)**2)
            if var_pop < 1e-6 * np.mean((ub - lb)**2):
                # Reinitialize half of population (worst individuals)
                num_reinit = max(1, pop_size // 2)
                worst_indices = np.argsort(pop_f)[-num_reinit:]
                for idx in worst_indices:
                    if idx == best_idx:
                        continue
                    pop[idx] = rng.uniform(lb, ub)
                    pop_f[idx] = func(pop[idx])
                    budget -= 1
                    if budget <= 0:
                        break
                # Update best if necessary
                best_idx = np.argmin(pop_f[:pop_size])
                if pop_f[best_idx] < best_f:
                    best_x = pop[best_idx].copy()
                    best_f = pop_f[best_idx]
                    report_best(best_f, best_x)
                continue

            # DE/rand/1/bin with per-individual F and CR
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select three distinct parents (excluding i)
                candidates = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(candidates, 3, replace=False)
                F = rng.uniform(0.5, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                CR = rng.uniform(0.1, 0.9)
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                # Greedy selection
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)

        return best_f, best_x