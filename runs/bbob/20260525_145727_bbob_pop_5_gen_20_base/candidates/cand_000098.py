import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        if budget == 0:
            return np.inf, np.zeros(dim)

        pop_size = max(4, min(5*dim, budget // 3))
        pop_size = min(pop_size, budget)
        if pop_size < 2:
            pop_size = budget

        budget_de = max(1, int(0.6 * budget))
        if budget_de < pop_size:
            budget_de = pop_size

        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        if evals == 0:
            return best_val, best_x

        F = 0.8
        CR = 0.9

        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break

                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]

                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            step = 0.1 * (ub - lb)
            while remaining > 0:
                improved = False
                for coord in range(dim):
                    if remaining <= 0:
                        break
                    old_val = best_val
                    # try positive direction
                    candidate = best_x.copy()
                    candidate[coord] += step[coord]
                    candidate[coord] = np.clip(candidate[coord], lb[coord], ub[coord])
                    val = func(candidate)
                    remaining -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        improved = True
                        continue
                    # try negative direction
                    candidate = best_x.copy()
                    candidate[coord] -= step[coord]
                    candidate[coord] = np.clip(candidate[coord], lb[coord], ub[coord])
                    val = func(candidate)
                    remaining -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        improved = True
                if not improved:
                    step *= 0.5  # shrink step size
                # prevent infinite loop if step becomes too small
                if np.max(step) < 1e-15 or remaining <= 0:
                    break

        return best_val, best_x