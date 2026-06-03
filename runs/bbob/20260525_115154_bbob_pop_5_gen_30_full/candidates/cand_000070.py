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
        span = ub - lb

        # initial point
        best_x = rng.uniform(lb, ub, dim).astype(float)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # step size
        sigma = 0.2 * np.linalg.norm(span) / np.sqrt(dim)
        sigma = max(sigma, 1e-12)

        # adaptation parameters
        window = max(1, 10 * dim)
        success_counter = 0
        gen_counter = 0
        no_improve_gen = 0

        while evals < budget:
            # generate candidate
            candidate = best_x + sigma * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            cand_val = func(candidate)
            evals += 1
            if cand_val < best_val:
                best_val = cand_val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                success_counter += 1
                no_improve_gen = 0
            else:
                no_improve_gen += 1

            gen_counter += 1

            # update sigma every window generations
            if gen_counter >= window:
                success_rate = success_counter / window
                if success_rate > 0.2:
                    sigma *= np.exp(1.0 / dim)
                else:
                    sigma *= np.exp(-1.0 / (4.0 * dim))
                sigma = max(sigma, 1e-12)
                # reset counters
                success_counter = 0
                gen_counter = 0

            # restart if stagnated or sigma too small
            if no_improve_gen >= 10 * dim or sigma < 1e-12:
                if evals >= budget:
                    break
                best_x = rng.uniform(lb, ub, dim).astype(float)
                best_val = func(best_x)
                evals += 1
                report_best(best_val, best_x)
                sigma = 0.2 * np.linalg.norm(span) / np.sqrt(dim)
                sigma = max(sigma, 1e-12)
                success_counter = 0
                gen_counter = 0
                no_improve_gen = 0

        return best_val, best_x