class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(2*dim, budget//4))
        if self.popsize > budget:
            self.popsize = budget
        self.F = 0.5
        self.CR = 0.9
        self.local_steps = max(1, dim//2)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        popsize = self.popsize
        F = self.F
        CR = self.CR
        rng = self.rng
        local_steps = self.local_steps

        best_x = None
        best_val = np.inf
        evaluations = 0

        if popsize < 4:
            for i in range(budget):
                x = rng.uniform(lb, ub, dim)
                val = func(x)
                evaluations += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        while evaluations < budget:
            for i in range(popsize):
                if evaluations >= budget:
                    break
                idx_best = np.argmin(fitness)
                possible = [j for j in range(popsize) if j != i]
                if len(possible) < 2:
                    continue
                r1, r2 = rng.choice(possible, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if evaluations < budget:
                progress = evaluations / budget
                step_size = 0.1 * (1 - progress)
                remaining = budget - evaluations
                local_iters = min(local_steps, remaining)
                for _ in range(local_iters):
                    if evaluations >= budget:
                        break
                    direction = rng.normal(0, 1, dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    new_x = best_x + step_size * direction
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    evaluations += 1
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)

        return best_val, best_x