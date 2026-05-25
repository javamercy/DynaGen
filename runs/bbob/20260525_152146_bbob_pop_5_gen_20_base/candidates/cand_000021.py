import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        if budget <= 0:
            return None, None

        # Evaluate initial point(s)
        best_x = rng.uniform(lb, ub)
        best_f = func(best_x)
        budget -= 1
        report_best(best_f, best_x)

        # Adaptive population size, but at least 4 for DE, and not exceeding budget
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget)  # ensure we have enough budget for initial pop
        # If insufficient budget for DE, just do local refinement
        if pop_size < 4 or budget < pop_size:
            # Fallback to simple random + local search
            while budget > 0:
                x = rng.uniform(lb, ub)
                f = func(x)
                budget -= 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        # Initialize population
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1
            if pop_f[i] < best_f:
                best_f = pop_f[i]
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # Linear schedules
        F_start, F_end = 0.9, 0.3
        CR_start, CR_end = 0.9, 0.1

        # Main DE loop
        max_gen = budget // pop_size if budget > 0 else 0
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0
            F = F_start + (F_end - F_start) * frac
            CR = CR_start + (CR_end - CR_start) * frac
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select three distinct random indices != i
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                crossover = rng.rand(dim) < CR
                if not crossover.any():
                    crossover[rng.randint(dim)] = True
                trial = np.where(crossover, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_f = trial_f
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            if budget <= 0:
                break

        # Local refinement via Gaussian perturbations
        sigma = 0.1 * (ub - lb)
        while budget > 0:
            pert = rng.normal(0, sigma)
            candidate = best_x + pert
            candidate = np.clip(candidate, lb, ub)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_f = cand_f
                best_x = candidate.copy()
                report_best(best_f, best_x)
                sigma *= 0.9

        return best_f, best_x