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

        # Initial point
        x0 = rng.uniform(lb, ub)
        best_val = func(x0)
        best_x = x0.copy()
        budget -= 1
        report_best(best_val, best_x)

        # DE phase: small population
        NP = max(3, min(10 * dim, budget // 2))
        NP = min(NP, budget)
        pop = rng.uniform(lb, ub, (NP, dim))
        fit = np.full(NP, np.inf)
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

        # Intensified local search: (1+lambda)-ES with step-size adaptation
        if best_x is not None and budget > 0:
            step_size = 0.1 * (ub - lb)
            success_count = 0
            total_attempts = 0
            while budget > 0:
                lambda_ = min(budget, 8)
                # Generate offspring
                offspring = best_x + rng.normal(0, step_size, (lambda_, dim))
                offspring = np.clip(offspring, lb, ub)
                best_off_val = np.inf
                best_off_x = None
                for k in range(lambda_):
                    if budget <= 0:
                        break
                    val = func(offspring[k])
                    budget -= 1
                    if val < best_off_val:
                        best_off_val = val
                        best_off_x = offspring[k].copy()
                # Check improvement
                if best_off_val < best_val:
                    best_val = best_off_val
                    best_x = best_off_x
                    report_best(best_val, best_x)
                    success_count += 1
                total_attempts += 1
                # Adapt step size based on success rate (1/5 rule)
                if total_attempts > 0:
                    success_rate = success_count / total_attempts
                    if success_rate > 0.2:
                        step_size *= 1.5
                    elif success_rate < 0.2:
                        step_size *= 0.85
                    # Reset counters
                    success_count = 0
                    total_attempts = 0
                if np.max(step_size) < 1e-12:
                    break

        return best_val, best_x