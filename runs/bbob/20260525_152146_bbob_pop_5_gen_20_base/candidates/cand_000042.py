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

        # Population size adaptive to dimension, leave room for local search
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, max(4, budget // 2))  # ensure at least 4 and not exceed half budget

        # Initialize population
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # Initialize self-adaptive parameters for each individual
        F = rng.uniform(0.1, 1.0, size=pop_size)
        CR = rng.uniform(0.0, 1.0, size=pop_size)

        # Differential Evolution with self-adaptive F and CR (jDE)
        # Use DE/rand/1/bin
        max_gen = budget // pop_size
        for gen in range(max_gen):
            if budget <= 0:
                break
            # Generate new F and CR for this generation with probability tau
            tau_F = 0.1
            tau_CR = 0.1
            new_F = np.where(rng.rand(pop_size) < tau_F,
                             rng.uniform(0.1, 1.0, size=pop_size),
                             F)
            new_CR = np.where(rng.rand(pop_size) < tau_CR,
                              rng.uniform(0.0, 1.0, size=pop_size),
                              CR)
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select three distinct random indices
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + new_F[i] * (pop[b] - pop[c])
                cross = rng.rand(dim) < new_CR[i]
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
                    # Update parameters if trial was successful
                    F[i] = new_F[i]
                    CR[i] = new_CR[i]
                # else parameters unchanged (stay old)
            # Optional: reduce population size if stuck? Not implemented here.
            if budget <= 0:
                break

        # Local refinement via Gaussian perturbations with adaptive step size
        sigma = 0.1 * (ub - lb)  # initial step size per dimension
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
                sigma *= 0.9  # exploit: shrink step
            else:
                sigma *= 1.1  # explore: enlarge step
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 1e-3 * (ub - lb), 1.0 * (ub - lb))

        return best_f, best_x