import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Larger population for diversity
        self.NP = max(6, min(7 * dim, budget // 3))
        self.CR = 0.9
        self.F = 0.5

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Initial population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        # Evaluate initial points
        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                # report_best called here, but we'll handle after loop? Actually call inside
            pop[i] = x
        if best_x is None:
            # fallback
            best_x = pop[0].copy()
            best_val = fitness[0]

        report_best(best_val, best_x)

        stagnation_counter = 0
        max_stagnation = 2 * dim

        # Main DE loop with restart
        while budget > 0 and NP > 1:
            # DE/rand/1/bin
            for i in range(NP):
                if budget <= 0:
                    break
                candidates = [j for j in range(NP) if j != i]
                if len(candidates) < 3:
                    break
                chosen = rng.choice(candidates, size=3, replace=False)
                a, b, c = chosen
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                else:
                    stagnation_counter += 1

            # Restart if stagnation
            if stagnation_counter >= max_stagnation:
                # Reinitialize worst half of population
                num_replace = max(1, NP // 2)
                worst_idx = np.argsort(fitness)[-num_replace:]
                for idx in worst_idx:
                    pop[idx] = rng.uniform(lb, ub, size=dim)
                    # Evaluate new point
                    if budget <= 0:
                        break
                    x = np.clip(pop[idx], lb, ub)
                    val = func(x)
                    budget -= 1
                    fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Reset stagnation counter (but keep best)
                stagnation_counter = 0

        # Local search after DE (if budget remains)
        if budget > 0 and best_x is not None:
            step = (ub - lb) * 0.1  # larger initial step for exploration
            while budget > 0:
                dims = rng.permutation(dim)
                for d in dims:
                    if budget <= 0:
                        break
                    direction = 1 if rng.rand() < 0.5 else -1
                    candidate = best_x.copy()
                    candidate[d] += direction * step[d]
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        # Keep step large; if improvement, maybe increase? We keep as is.
                    else:
                        step[d] *= 0.95  # shrink gradually
                # If steps become too small, break
                if np.max(step) < 1e-10:
                    break

        if best_x is None:
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x

        return best_val, best_x