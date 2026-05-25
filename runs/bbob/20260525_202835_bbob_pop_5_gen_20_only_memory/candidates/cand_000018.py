import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        total_budget = self.budget

        # Determine number of restarts
        min_budget_per_restart = max(2 * dim + 1, 10)
        R = max(1, min(10, total_budget // min_budget_per_restart))
        if R < 1:
            R = 1

        best_val = np.inf
        best_x = None

        # Allocate budgets per restart
        base_budget = total_budget // R
        left_budget = total_budget - base_budget * (R - 1)
        budgets = [base_budget] * (R - 1) + [left_budget]

        for i_restart in range(R):
            restart_budget = budgets[i_restart]
            if restart_budget <= 0:
                continue

            # Initialize random point
            x = lb + self.rng.random(dim) * (ub - lb)
            val = func(x)
            used = 1

            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

            # Pattern search parameters
            step = 0.1 * (ub - lb)  # per-dim step
            improvement = True

            while used < restart_budget:
                if not improvement:
                    # Shrink all steps (minimum step 1e-8 relative)
                    step *= 0.5
                    improvement = True
                    if np.max(step) < 1e-8 * (ub - lb):
                        # Random restart within this run
                        x = lb + self.rng.random(dim) * (ub - lb)
                        val = func(x)
                        used += 1
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        step = 0.1 * (ub - lb)
                        improvement = True

                improved = False
                perm = self.rng.permutation(dim)

                for i in perm:
                    if used >= restart_budget:
                        break

                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    used += 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        step[i] *= 1.5
                        improved = True
                        continue

                    # Try negative direction
                    if used >= restart_budget:
                        break
                    x_new = x.copy()
                    x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                    val_new = func(x_new)
                    used += 1
                    if val_new < val:
                        val = val_new
                        x = x_new.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        step[i] *= 1.5
                        improved = True

                if not improved and used < restart_budget:
                    # Random perturbation
                    x_pert = lb + self.rng.random(dim) * (ub - lb)
                    val_pert = func(x_pert)
                    used += 1
                    if val_pert < val:
                        val = val_pert
                        x = x_pert.copy()
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                        improvement = True
                    else:
                        improvement = False
                else:
                    improvement = improved

        return best_val, best_x