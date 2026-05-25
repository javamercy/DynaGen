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

        pop_size = max(4, min(8*dim, budget // 4))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 2:
            pop_size = budget

        budget_de = int(0.6 * budget)
        if budget_de < pop_size:
            budget_de = max(pop_size, 1)

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

        CR = 0.9
        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                fraction = evals / budget_de
                F = 0.5 + 0.4 * np.sin(np.pi * fraction)

                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]

                mutant = best_x + F * (pop[a] - pop[b])
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
            # Local refinement: multiple scales and coordinate descent
            scale = 0.5 * (ub - lb)
            while remaining > 0 and np.any(scale > 1e-10):
                # Random perturbations around best
                n_perturb = min(remaining, max(1, remaining // 2))
                for _ in range(n_perturb):
                    if remaining <= 0:
                        break
                    perturb = rng.normal(0, scale, dim)
                    candidate = best_x + perturb
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    remaining -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                # Coordinate descent-like pattern search
                if remaining >= 2 * dim:
                    # Try positive and negative steps for each dimension
                    step = 0.1 * scale
                    for j in range(dim):
                        if remaining < 2:
                            break
                        # positive step
                        candidate = best_x.copy()
                        candidate[j] = np.clip(candidate[j] + step[j], lb[j], ub[j])
                        val = func(candidate)
                        evals += 1
                        remaining -= 1
                        if val < best_val:
                            best_val = val
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                        # negative step
                        candidate = best_x.copy()
                        candidate[j] = np.clip(candidate[j] - step[j], lb[j], ub[j])
                        val = func(candidate)
                        evals += 1
                        remaining -= 1
                        if val < best_val:
                            best_val = val
                            best_x = candidate.copy()
                            report_best(best_val, best_x)
                scale *= 0.5
        return best_val, best_x