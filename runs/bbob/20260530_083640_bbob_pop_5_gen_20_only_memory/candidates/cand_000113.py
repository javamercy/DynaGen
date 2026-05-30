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

        # Initial population size (larger for exploration)
        pop_size = min(budget // 2, max(5 * dim, 20))
        pop_size = min(pop_size, budget)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial points
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                r_best(val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Reserve for local search
        reserve = min(10 * dim + 20, budget - evals)
        max_gen = (budget - evals - reserve) // pop_size
        max_gen = max(0, max_gen)

        # Stagnation tracking
        no_improve = 0

        # Adaptive DE with restarts
        for gen in range(max_gen):
            # Random F and CR each generation
            F = rng.uniform(0.5, 1.0)
            CR = rng.uniform(0.1, 0.9)
            improved = False
            for i in range(pop_size):
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
                        r_best(val, best_x)
                        improved = True
                if evals >= budget - reserve:
                    break
            if evals >= budget - reserve:
                break

            # Stagnation restart
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 2:
                    # Reinitialize 30% of population (excluding best)
                    n_restart = max(1, int(0.3 * pop_size))
                    # Pick worst n_restart individuals to replace
                    worst_indices = np.argsort(pop_fitness)[-n_restart:]
                    lhs_restart = self._latin_hypercube(n_restart, dim, rng)
                    new_pts = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs_restart
                    for idx, new_x in zip(worst_indices, new_pts):
                        pop[idx] = new_x
                        val = func(new_x)
                        evals += 1
                        pop_fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_x.copy()
                            r_best(val, best_x)
                        if evals >= budget - reserve:
                            break
                    no_improve = 0
                    if evals >= budget - reserve:
                        break

        # Local random search around best
        remaining = budget - evals
        if remaining > 0:
            x_curr = best_x.copy()
            f_curr = best_val
            # Initial step as fraction of domain
            step = 0.1 * (ub - lb)
            min_step = 1e-5 * (ub - lb)
            for _ in range(remaining):
                if np.all(step < min_step):
                    break
                # Random perturbation in each dimension
                perturb = rng.normal(0, step / 2, size=dim)
                x_new = np.clip(x_curr + perturb, lb, ub)
                val = func(x_new)
                evals += 1
                if val < f_curr:
                    x_curr = x_new
                    f_curr = val
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        r_best(val, best_x)
                else:
                    # Reduce step on failure
                    step *= 0.9
                if evals >= budget:
                    break
        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs