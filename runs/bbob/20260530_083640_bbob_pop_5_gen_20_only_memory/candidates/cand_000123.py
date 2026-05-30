import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_val = np.inf
        best_x = None
        evals = 0

        # Initial point
        x = lb + rng.rand(dim) * (ub - lb)
        val = func(x)
        evals += 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        # Parameters
        sigma = 0.2 * (ub - lb).mean()  # initial step size
        T = 1.0  # initial temperature
        T_min = 1e-5
        cooling_rate = 0.95
        success_counter = 0
        fail_counter = 0
        target_success_rate = 0.2
        restart_threshold = max(1, budget // 10)
        no_improve_iters = 0
        max_no_improve = max(1, budget // 20)

        while evals < budget:
            # Generate neighbor
            direction = rng.randn(dim)
            direction = direction / np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else np.ones(dim)/np.sqrt(dim)
            step = sigma * direction
            x_new = x + step
            # Reflect to stay within bounds
            x_new = np.clip(x_new, lb, ub)
            # Also ensure we don't stay exactly at boundary by random perturbation if stuck? no
            val_new = func(x_new)
            evals += 1

            # Acceptance
            delta = val_new - val
            if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
                x = x_new
                val = val_new
                success_counter += 1
                fail_counter = 0
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                    no_improve_iters = 0
                else:
                    no_improve_iters += 1
            else:
                fail_counter += 1
                success_counter = 0
                no_improve_iters += 1

            # Step size adaptation (Rechemberg 1/5 rule)
            if evals % (2*dim) == 0:
                actual_success_rate = success_counter / (success_counter + fail_counter + 1e-10)
                if actual_success_rate > target_success_rate:
                    sigma = sigma * 1.2
                else:
                    sigma = sigma * 0.85
                success_counter = 0
                fail_counter = 0

            # Cooling
            T = max(T * cooling_rate, T_min)

            # Restart if no improvement for a long time
            if no_improve_iters >= max_no_improve and evals < budget:
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                sigma = 0.2 * (ub - lb).mean()
                T = 1.0
                no_improve_iters = 0
                success_counter = 0
                fail_counter = 0

        return best_val, best_x