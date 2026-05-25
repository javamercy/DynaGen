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
        R = min(5, total_budget // (2 * dim + 1))
        if R < 1:
            R = 1
        budget_per_restart = total_budget // R
        best_val = np.inf
        best_x = None

        for _ in range(R):
            used = 0
            x = lb + self.rng.random(dim) * (ub - lb)
            val = func(x)
            used += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

            step = 0.1 * (ub - lb)
            improvement = True

            while used < budget_per_restart:
                if not improvement:
                    step *= 0.5
                    improvement = True
                improved = False
                perm = self.rng.permutation(dim)

                for i in perm:
                    if used >= budget_per_restart:
                        break
                    # positive direction
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
                    # negative direction
                    if used >= budget_per_restart:
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

                if not improved and used < budget_per_restart:
                    x_rand = lb + self.rng.random(dim) * (ub - lb)
                    val_rand = func(x_rand)
                    used += 1
                    if val_rand < val:
                        val = val_rand
                        x = x_rand.copy()
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