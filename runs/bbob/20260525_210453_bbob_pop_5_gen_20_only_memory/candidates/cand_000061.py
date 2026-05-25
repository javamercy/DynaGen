import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        range_ = ub - lb

        best_x = None
        best_f = np.inf
        fcalls = 0

        # Phase 1: Simulated annealing with aggressive cooling
        sa_budget = max(1, budget // 2)
        T0 = 10.0
        Tf = 1e-3
        cooling = (Tf / T0) ** (1.0 / max(1, sa_budget - 1))
        T = T0

        # Initial point
        x = np.random.uniform(lb, ub, size=dim)
        f = func(x)
        fcalls += 1
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)

        current_x = x.copy()
        current_f = f

        for i in range(sa_budget - 1):
            if fcalls >= budget:
                break
            step = 0.2 * range_ * np.random.randn(dim)
            new_x = np.clip(current_x + step, lb, ub)
            fnew = func(new_x)
            fcalls += 1
            if fnew < current_f or np.random.rand() < np.exp((current_f - fnew) / T):
                current_x = new_x.copy()
                current_f = fnew
                if fnew < best_f:
                    best_f = fnew
                    best_x = new_x.copy()
                    report_best(best_f, best_x)
            T *= cooling

        # Phase 2: Local random walk refinement around best
        radius = 0.1 * np.linalg.norm(range_)
        while fcalls < budget:
            dir_vec = np.random.randn(dim)
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir == 0:
                continue
            dir_vec = dir_vec / norm_dir
            step = radius * dir_vec
            new_x = np.clip(best_x + step, lb, ub)
            fnew = func(new_x)
            fcalls += 1
            if fnew < best_f:
                best_f = fnew
                best_x = new_x.copy()
                report_best(best_f, best_x)
                radius *= 0.9  # shrink after improvement
            else:
                radius *= 0.95  # gradual reduction
            # keep radius from getting too small
            radius = max(radius, 1e-8)

        return best_f, best_x