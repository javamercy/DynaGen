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

        def clip(x):
            return np.clip(x, lb, ub)

        # Handle trivial budget
        if budget == 0:
            raise ValueError("Budget must be at least 1")
        if budget == 1:
            x = rng.uniform(lb, ub, size=dim)
            f = func(x)
            report_best(f, x)
            return f, x

        # Population size: ensure at least 2 for DE
        pop_size = max(2, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget - 1)

        # Initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # Initialize individual F and CR
        F = np.full(pop_size, 0.5)
        CR = np.full(pop_size, 0.9)
        tau1 = 0.1
        tau2 = 0.1

        # Main DE loop
        max_gen = budget // pop_size if budget > 0 else 0
        for gen in range(max_gen):
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Update F and CR with probability
                if rng.rand() < tau1:
                    F[i] = rng.uniform(0, 1)
                else:
                    F[i] += rng.uniform(-0.1, 0.1)
                    F[i] = np.clip(F[i], 0, 1)
                if rng.rand() < tau2:
                    CR[i] = rng.uniform(0, 1)
                else:
                    CR[i] += rng.uniform(-0.1, 0.1)
                    CR[i] = np.clip(CR[i], 0, 1)

                # Mutation: DE/rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mut = pop[a] + F[i] * (pop[b] - pop[c])

                # Crossover: binomial
                cross = rng.rand(dim) < CR[i]
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)

                trial_f = func(trial)
                budget -= 1

                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
            if budget <= 0:
                break

        # Local refinement with success-based step size adaptation
        sigma = 0.1 * (ub - lb)
        while budget > 0:
            pert = rng.normal(0, sigma, size=dim)
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)
                # reset sigma on success
                sigma = 0.1 * (ub - lb)
            else:
                sigma *= 0.9

        return best_f, best_x