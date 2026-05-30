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

        # Reserve a significant portion for local search
        reserve_local = max(20, int(0.4 * budget))
        # Initial population size: at least 4*dim, at most budget/4
        init_pop_size = max(4*dim, min(40, budget // 4))
        # Ensure we have enough for DE + initial
        max_for_initial = budget - reserve_local
        if init_pop_size > max_for_initial:
            init_pop_size = max_for_initial
            if init_pop_size < 2*dim:
                init_pop_size = max(2*dim, 1)
        eval_init = init_pop_size
        # Budget for DE generations
        de_budget = budget - eval_init - reserve_local
        if de_budget < 0:
            de_budget = 0
        # Number of generations, at least 1 if de_budget allows
        gens = max(1, de_budget // init_pop_size) if de_budget >= init_pop_size else 0
        if gens == 0:
            gens = 0  # skip DE if not enough budget

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(init_pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial points
        pop_fitness = np.full(init_pop_size, np.inf)
        for i in range(init_pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Differential Evolution with adaptive parameters
        for gen in range(gens):
            if evals >= budget - reserve_local:
                break
            progress = gen / gens
            F = 0.8 - 0.6 * progress  # from 0.8 to 0.2
            CR = 0.9 - 0.4 * progress  # from 0.9 to 0.5

            for i in range(init_pop_size):
                if evals >= budget - reserve_local:
                    break
                # Mutation
                indices = [j for j in range(init_pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                # Evaluation
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if evals >= budget:
                    return best_val, best_x
            if evals >= budget - reserve_local:
                break

        # Local search: randomized coordinate descent with step size adaptation
        remaining = budget - evals
        if remaining > 0:
            x_current = best_x.copy()
            f_current = best_val
            step_sizes = 0.1 * (ub - lb)  # initial step sizes per dimension
            # We'll do cycles of random permutations
            while evals < budget:
                perm = rng.permutation(dim)
                improved = False
                for d in perm:
                    if evals >= budget:
                        break
                    # Try positive step
                    step = step_sizes[d]
                    x_new = x_current.copy()
                    x_new[d] = min(ub[d], x_current[d] + step)
                    val = func(x_new)
                    evals += 1
                    if val < f_current:
                        f_current = val
                        x_current = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x_current.copy()
                            report_best(best_val, best_x)
                        step_sizes[d] *= 1.2  # increase step
                        improved = True
                        continue
                    # Try negative step
                    x_new = x_current.copy()
                    x_new[d] = max(lb[d], x_current[d] - step)
                    val = func(x_new)
                    evals += 1
                    if val < f_current:
                        f_current = val
                        x_current = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x_current.copy()
                            report_best(best_val, best_x)
                        step_sizes[d] *= 1.2
                        improved = True
                    else:
                        step_sizes[d] *= 0.5  # decrease step
                if not improved:
                    # If no improvement in full cycle, break to avoid stagnation
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