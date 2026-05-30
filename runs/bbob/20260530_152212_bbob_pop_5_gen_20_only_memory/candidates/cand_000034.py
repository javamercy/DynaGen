import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim

        # Initial population size: small for exploitation
        pop_size = min(3, budget // 2)
        if pop_size < 1:
            pop_size = 1

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Parameters
        F = 0.8
        CR = 0.9
        sigma = 0.2 * (ub - lb).mean()
        success_hist = []
        cov = np.eye(dim)

        # Main loop: use DE/best/1/bin for global exploration sparingly
        de_phase_evals = min(budget // 4, 5 * dim)  # limit DE to 1/4 of budget
        while evals < min(budget, de_phase_evals):
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                if len(idxs) < 2:
                    continue
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local refinement phase: use covariance from best points and adaptive step size
        # Store best points to estimate covariance
        best_points = [best_x.copy()]
        while evals < budget:
            # Estimate covariance from collected best points
            if len(best_points) > 1:
                cov = np.cov(np.array(best_points).T) + 1e-8 * np.eye(dim)
            else:
                cov = np.eye(dim)

            # Generate candidate
            delta = np.random.multivariate_normal(np.zeros(dim), sigma ** 2 * cov)
            trial = best_x + delta
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            if val < best_val:
                # success: accept, update best, increase step size
                sigma *= 1.1
                best_val = val
                best_x = trial.copy()
                best_points.append(best_x.copy())
                # Keep only last 10 best points for covariance
                if len(best_points) > 10:
                    best_points.pop(0)
                report_best(best_val, best_x)
            else:
                # failure: decrease step size
                sigma *= 0.9

            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-10, (ub - lb).max())

        return best_val, best_x