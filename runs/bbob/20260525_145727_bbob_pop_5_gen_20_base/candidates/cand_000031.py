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

        # Population size: at least 5 for rand/2, max 20, also limited by budget
        pop_size = min(4 * dim, 20, budget // 3)
        pop_size = max(pop_size, 5)
        if pop_size > budget:
            pop_size = budget

        pop_x = rng.uniform(lb, ub, (pop_size, dim))
        pop_y = np.full(pop_size, np.inf)
        age = np.zeros(pop_size, dtype=int)
        best_x = None
        best_y = np.inf

        for i in range(pop_size):
            pop_y[i] = func(pop_x[i])
            if pop_y[i] < best_y:
                best_y = pop_y[i]
                best_x = pop_x[i].copy()
                report_best(best_y, best_x)

        evals = pop_size
        # Max age before forced exploration: half of average generations budget
        max_age = max(1, budget // pop_size // 2)

        while evals < budget:
            # Adaptive F and CR per generation
            F = 0.6 + 0.4 * rng.rand()  # [0.6, 1.0]
            CR = 0.8 + 0.2 * rng.rand()  # [0.8, 1.0]
            for i in range(pop_size):
                if evals >= budget:
                    break
                target = pop_x[i]
                if age[i] > max_age:
                    # Exploration: random point
                    trial = rng.uniform(lb, ub)
                else:
                    # DE/rand/2 mutation: need 5 distinct indices different from i
                    indices = list(range(pop_size))
                    indices.remove(i)
                    if len(indices) < 5:
                        continue
                    chosen = rng.choice(indices, 5, replace=False)
                    a, b, c, d, e = pop_x[chosen]
                    mutant = a + F * (b - c + d - e)
                    trial = target.copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_y = func(trial)
                evals += 1
                if trial_y < pop_y[i]:
                    pop_x[i] = trial
                    pop_y[i] = trial_y
                    age[i] = 0
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                        report_best(best_y, best_x)
                else:
                    age[i] += 1
        return best_y, best_x