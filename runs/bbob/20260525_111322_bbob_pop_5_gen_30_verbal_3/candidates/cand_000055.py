class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = max(4, min(5 * self.dim, self.budget // 2))
        pop_size = min(pop_size, self.budget)
        points = self.rng.uniform(lb, ub, size=(pop_size, self.dim))
        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        F = 0.5
        CR = 0.9
        while evals < self.budget:
            target = self.rng.randint(pop_size)
            indices = [i for i in range(pop_size) if i != target]
            if len(indices) < 3:
                break
            a, b, c = self.rng.choice(indices, 3, replace=False)
            mutant = points[a] + F * (points[b] - points[c])
            trial = points[target].copy()
            j_rand = self.rng.randint(self.dim)
            for j in range(self.dim):
                if self.rng.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lb, ub)
            f_trial = func(trial)
            evals += 1
            if f_trial < best_f:
                best_f = f_trial
                best_x = trial.copy()
                report_best(best_f, best_x)
            if f_trial < pop_fitness[target]:
                points[target] = trial
                pop_fitness[target] = f_trial
        return best_f, best_x