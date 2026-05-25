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

        pop_size = max(5, min(15*dim, budget // 3))
        if pop_size > budget:
            pop_size = budget

        budget_de = int(0.8 * budget)
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

        no_improve_counter = 0
        max_no_improve = 50

        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                F = np.clip(rng.normal(0.5, 0.3), 0.1, 1.0)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                CR = np.clip(rng.normal(0.8, 0.2), 0.0, 1.0)
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
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1

            # Restart if too long without improvement
            if no_improve_counter >= max_no_improve and evals < budget_de:
                no_improve_counter = 0
                # Replace worst half of population
                sorted_indices = np.argsort(fitness)
                n_replace = pop_size // 2
                worst_indices = sorted_indices[-n_replace:]
                for idx in worst_indices:
                    if evals >= budget_de:
                        break
                    if rng.rand() < 0.5:
                        # Around best with perturbation
                        sigma = 0.2 * (ub - lb)
                        candidate = best_x + rng.normal(0, sigma, dim)
                        candidate = np.clip(candidate, lb, ub)
                    else:
                        # Random
                        candidate = lb + rng.rand(dim) * (ub - lb)
                    val = func(candidate)
                    evals += 1
                    fitness[idx] = val
                    pop[idx] = candidate
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)

        # Local refinement
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.1 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.normal(0, sigma, dim)
                candidate = best_x + perturb
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x