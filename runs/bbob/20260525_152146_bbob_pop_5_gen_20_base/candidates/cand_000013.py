import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub
        clip = lambda x: np.clip(x, lb, ub)

        # Phase 1: Differential Evolution with adaptive parameters and pbest mutation
        pop_size = max(4, min(int(np.sqrt(dim) * 4), budget // 10))
        pop_size = min(pop_size, budget - 5)
        if pop_size < 4:
            pop_size = 4
        pop = rng.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # Adaptive parameters
        F_mean = 0.7
        F_std = 0.2
        CR_mean = 0.8
        CR_std = 0.15
        # Main DE loop
        while budget >= pop_size:
            F = np.clip(rng.normal(F_mean, F_std), 0, 2)
            CR = np.clip(rng.normal(CR_mean, CR_std), 0, 1)
            # Determine pbest indices (top 20% of population)
            sorted_idx = np.argsort(pop_f)
            pbest_num = max(1, int(0.2 * pop_size))
            pbest_indices = sorted_idx[:pbest_num]
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Choose pbest randomly from top set
                pbest = rng.choice(pbest_indices)
                # Choose two distinct random indices different from i
                pool = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(pool, 2, replace=False)
                # current-to-pbest/1 mutation
                mut = pop[i] + F * (pop[pbest] - pop[i]) + F * (pop[a] - pop[b])
                # Binomial crossover
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
            if budget <= 0:
                break

        # Phase 2: Local refinement with adaptive diagonal steps
        sigma0 = 0.1 * (ub - lb)  # initial step per dimension
        scale = np.ones(dim)  # scaling factor
        decay = 0.95
        expansion = 1.05
        stagnation = 0
        while budget > 0:
            pert = rng.normal(0, sigma0 * scale, dim)
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)
                scale = scale * expansion
                stagnation = 0
            else:
                scale = scale * decay
                stagnation += 1
            # Restart with a new random point if stuck
            if stagnation >= 10:
                if budget > 0:
                    candidate = rng.uniform(lb, ub)
                    cand_f = func(candidate)
                    budget -= 1
                    if cand_f < best_f:
                        best_x = candidate.copy()
                        best_f = cand_f
                        report_best(best_f, best_x)
                    scale = np.ones(dim)
                    stagnation = 0
        return best_f, best_x