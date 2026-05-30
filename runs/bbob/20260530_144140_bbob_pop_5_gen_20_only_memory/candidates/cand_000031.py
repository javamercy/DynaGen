class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Bigger population for exploration
        self.NP = max(10, min(15*dim, budget // 2))
        self.CR = 0.9
        self.F = 0.8

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        # Initialization
        pop = rng.uniform(bounds_lb, bounds_ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_idx = -1
        best_val = np.inf
        best_x = None

        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], bounds_lb, bounds_ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # DE/rand/1/bin main loop
        while budget > 0 and NP > 1:
            for i in range(NP):
                if budget <= 0:
                    break
                # Choose three distinct indices different from i
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 3:
                    break
                r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                # Mutation: DE/rand/1
                mutant = pop[r1] + self.F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, bounds_lb, bounds_ub)
                # Crossover binomial
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluate
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Restart: if budget remains, reinitialize around best with larger step
        if budget > 0 and best_x is not None:
            # Create a small new population around best
            restart_NP = min(5, budget)
            restart_pop = rng.uniform(low=-1, high=1, size=(restart_NP, dim)) * (bounds_ub - bounds_lb) * 0.3 + best_x
            restart_pop = np.clip(restart_pop, bounds_lb, bounds_ub)
            for i in range(restart_NP):
                if budget <= 0:
                    break
                val = func(restart_pop[i])
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = restart_pop[i].copy()
                    report_best(best_val, best_x)
            # Run a few DE generations on this small population
            while budget > 0 and restart_NP > 1:
                for i in range(restart_NP):
                    if budget <= 0:
                        break
                    indices = [j for j in range(restart_NP) if j != i]
                    if len(indices) < 3:
                        break
                    r1, r2, r3 = rng.choice(indices, size=3, replace=False)
                    mutant = restart_pop[r1] + self.F * (restart_pop[r2] - restart_pop[r3])
                    mutant = np.clip(mutant, bounds_lb, bounds_ub)
                    j_rand = rng.randint(dim)
                    trial = np.where(rng.rand(dim) < self.CR, mutant, restart_pop[i])
                    trial[j_rand] = mutant[j_rand]
                    val = func(trial)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        if best_x is None:
            x = rng.uniform(bounds_lb, bounds_ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x