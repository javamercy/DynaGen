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

        pop_size = max(4, min(10 * dim, budget // 4) )
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

        F = 0.8
        CR = 0.7
        stagnation_counter = 0
        max_stagnation = 50

        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
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
                improved = False
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if improved:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if stagnation_counter >= max_stagnation:
                    # Restart: replace worst half with random points
                    num_replace = pop_size // 2
                    if num_replace == 0:
                        num_replace = 1
                    order = np.argsort(fitness)[::-1]  # worst first
                    for idx in order[:num_replace]:
                        if evals >= budget_de or evals >= budget:
                            break
                        new_point = lb + rng.rand(dim) * (ub - lb)
                        val_new = func(new_point)
                        evals += 1
                        pop[idx] = new_point
                        fitness[idx] = val_new
                        if val_new < best_val:
                            best_val = val_new
                            best_x = new_point.copy()
                            report_best(best_val, best_x)
                    stagnation_counter = 0

        remaining = budget - evals
        if remaining > 0:
            for _ in range(remaining):
                candidate = lb + rng.rand(dim) * (ub - lb)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x