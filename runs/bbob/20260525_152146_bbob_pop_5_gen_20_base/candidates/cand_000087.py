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

        pop_size = max(4, min(20, budget // 4))
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        num_eval = 0

        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1
            num_eval += 1

        best_idx = np.argmin(pop_f[:num_eval])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F_lo, F_hi = 0.5, 1.0
        CR = 0.9
        random_prob = 0.05

        while budget > 0:
            for i in range(num_eval):
                if budget <= 0:
                    break
                # Generate trial
                if rng.rand() < random_prob:
                    # Random injection
                    trial = rng.uniform(low=lb, high=ub, size=dim)
                else:
                    # Standard DE/rand/1
                    candidates = [j for j in range(num_eval) if j != i]
                    if len(candidates) < 3:
                        continue
                    a, b, c = rng.choice(candidates, 3, replace=False)
                    F = rng.uniform(F_lo, F_hi)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    # Binomial crossover
                    j0 = rng.randint(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.rand() < CR or j == j0:
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

        return best_f, best_x