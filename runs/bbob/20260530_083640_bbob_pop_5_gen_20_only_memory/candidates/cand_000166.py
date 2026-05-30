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

        best_val = np.inf
        best_x = None
        evals = 0

        if budget <= 0:
            return None, None

        # Initial random point
        x = lb + rng.rand(dim) * (ub - lb)
        val = func(x)
        evals += 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)

        if budget == 1:
            return best_val, best_x

        # Small budget: simple random search then local search
        if budget < 5 * dim:
            for _ in range(min(budget // 2, dim * 2)):
                if evals >= budget:
                    break
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            self._local_search(func, lb, ub, dim, budget, rng, best_val, best_x, evals)
            return best_val, best_x

        # Larger budget: use small population DE then intense local search
        pop_size = min(max(5, dim), budget // 5, budget // 2)
        pop_size = max(pop_size, 3)
        lhs = self._latin_hypercube(pop_size, dim, rng)
        pop = lb + (ub - lb) * lhs
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Reserve budget for local search
        local_reserve = max(2 * dim, 20)
        local_reserve = min(local_reserve, budget - evals)
        budget_for_de = budget - evals - local_reserve
        max_gen = budget_for_de // pop_size if budget_for_de > 0 else 0
        max_gen = max(0, min(max_gen, 10))  # limit DE generations

        for gen in range(max_gen):
            if evals >= budget - local_reserve:
                break
            F = 0.8
            CR = 0.9
            for i in range(pop_size):
                if evals >= budget - local_reserve:
                    break
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # Restart worst 20% every 3 generations
            if gen % 3 == 2:
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 5):]
                for idx in worst_idx:
                    if evals >= budget - local_reserve:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)

        # Local search phase
        self._local_search(func, lb, ub, dim, budget, rng, best_val, best_x, evals)

        return best_val, best_x

    def _local_search(self, func, lb, ub, dim, budget, rng, best_val, best_x, evals):
        step = 0.1 * (ub - lb)
        while evals < budget:
            improved = False
            for i in rng.permutation(dim):
                if evals >= budget:
                    break
                # Positive direction
                trial = best_x.copy()
                trial[i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 1.5, ub[i] - lb[i])
                    improved = True
                    break
                # Negative direction
                trial[i] = np.clip(best_x[i] - step[i], lb[i], ub[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step[i] = min(step[i] * 1.5, ub[i] - lb[i])
                    improved = True
                    break
                else:
                    step[i] = max(step[i] * 0.5, 1e-10 * (ub[i] - lb[i]))
            if not improved and evals < budget:
                # Random perturbation
                direction = rng.randn(dim)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                trial = np.clip(best_x + step * direction, lb, ub)
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial
                    report_best(best_val, best_x)
                    step = np.minimum(step * 1.5, ub - lb)
                    improved = True
            if not improved and evals < budget:
                # Random restart
                new_x = lb + rng.rand(dim) * (ub - lb)
                new_val = func(new_x)
                evals += 1
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                    step = 0.1 * (ub - lb)
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs