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

        pop_size = max(4, min(5*dim, budget // 5))
        pop_size = min(pop_size, budget)
        if pop_size < 2:
            pop_size = max(1, budget)

        budget_de = max(1, int(0.7 * budget))
        if budget_de < pop_size:
            budget_de = budget

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

        # DE/rand/1 with adaptive F and CR
        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                # Adaptive parameters
                frac = evals / budget_de
                F = 0.5 - 0.3 * frac  # from 0.5 to 0.2
                CR = 0.9 - 0.4 * frac  # from 0.9 to 0.5

                # Select three distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]

                # DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
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

        # Local coordinate-wise refinement
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            step_start = 0.1 * (ub - lb)
            step_end = 0.01 * (ub - lb)
            for k in range(remaining):
                frac = k / remaining if remaining > 1 else 0.0
                step_size = step_start - (step_start - step_end) * frac
                coord = rng.randint(dim)
                perturb = np.zeros(dim)
                # Try both directions and pick better
                candidate_plus = best_x.copy()
                candidate_plus[coord] = np.clip(best_x[coord] + step_size[coord], lb[coord], ub[coord])
                candidate_minus = best_x.copy()
                candidate_minus[coord] = np.clip(best_x[coord] - step_size[coord], lb[coord], ub[coord])
                val_plus = func(candidate_plus)
                evals += 1
                if val_plus < best_val:
                    best_val = val_plus
                    best_x = candidate_plus
                    report_best(best_val, best_x)
                    continue
                if evals >= budget:
                    break
                val_minus = func(candidate_minus)
                evals += 1
                if val_minus < best_val:
                    best_val = val_minus
                    best_x = candidate_minus
                    report_best(best_val, best_x)
                # If neither improved, we still consumed evaluations

        return best_val, best_x