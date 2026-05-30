import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Small population to reserve budget for local search
        NP = max(3, min(10 * dim, budget // 2))
        NP = min(NP, budget)

        # Initialize population
        pop = rng.uniform(lb, ub, (NP, dim))
        fit = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            budget -= 1
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # DE/best/1/bin with decreasing F
        gen = 0
        while budget > 0 and NP >= 3:
            F = 0.7 - 0.6 * gen / max(1, budget // NP)
            gen += 1
            for i in range(NP):
                if budget <= 0:
                    break
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 2:
                    break
                r1, r2 = rng.choice(indices, 2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < 0.9, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                budget -= 1
                if val < fit[i]:
                    pop[i] = trial
                    fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search intensification
        if budget > 0 and best_x is not None:
            step = 0.1 * (ub - lb)
            decay = 0.9
            while budget > 0:
                num_candidates = min(budget, 5)
                perturbations = rng.normal(0, step, (num_candidates, dim))
                candidates = best_x + perturbations
                candidates = np.clip(candidates, lb, ub)
                best_candidate_val = np.inf
                best_candidate = None
                for k in range(num_candidates):
                    if budget <= 0:
                        break
                    val = func(candidates[k])
                    budget -= 1
                    if val < best_candidate_val:
                        best_candidate_val = val
                        best_candidate = candidates[k].copy()
                if best_candidate_val < best_val:
                    best_val = best_candidate_val
                    best_x = best_candidate
                    report_best(best_val, best_x)
                else:
                    step = step * decay
                if np.max(step) < 1e-12:
                    break

        if best_x is None:
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x

        return best_val, best_x