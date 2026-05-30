import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self.best_value = np.inf
        self.best_x = None
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        n_restarts = 3
        per_restart = self.budget // n_restarts

        for restart in range(n_restarts):
            if self.calls >= self.budget:
                break
            if restart == 0:
                # first restart: use initial sampling
                n_init = max(2, min(per_restart // 5, 50))
                for _ in range(n_init):
                    if self.calls >= self.budget:
                        break
                    x = self.rng.uniform(lb, ub, size=self.dim)
                    val = func(x)
                    self.calls += 1
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                if self.best_x is None:
                    x0 = self.rng.uniform(lb, ub, size=self.dim)
                    val0 = func(x0)
                    self.calls += 1
                    if val0 < self.best_value:
                        self.best_value = val0
                        self.best_x = x0.copy()
                        report_best(self.best_value, self.best_x)
                else:
                    x0 = self.best_x.copy()
                    val0 = self.best_value
                local_best_x = x0
                local_best_val = val0
                remaining = per_restart - n_init
                if remaining < 1:
                    continue
            else:
                # subsequent restarts: random start
                if self.calls >= self.budget:
                    break
                x0 = self.rng.uniform(lb, ub, size=self.dim)
                val0 = func(x0)
                self.calls += 1
                if val0 < self.best_value:
                    self.best_value = val0
                    self.best_x = x0.copy()
                    report_best(self.best_value, self.best_x)
                local_best_x = x0
                local_best_val = val0
                remaining = per_restart - 1
                if remaining < 1:
                    continue

            sigma = np.mean(ub - lb) * 0.2
            update_freq = 10
            success_counter = 0
            local_calls = 0

            while local_calls < remaining and self.calls < self.budget:
                candidate = local_best_x + self.rng.normal(0, sigma, size=self.dim)
                candidate = np.clip(candidate, lb, ub)
                cand_val = func(candidate)
                self.calls += 1
                local_calls += 1

                if cand_val < local_best_val:
                    # line search along direction
                    direction = candidate - local_best_x
                    best_cand_x = candidate.copy()
                    best_cand_val = cand_val
                    for step in [0.5, 1.0, 1.5, 2.0]:
                        if local_calls >= remaining or self.calls >= self.budget:
                            break
                        x_step = local_best_x + step * direction
                        x_step = np.clip(x_step, lb, ub)
                        val_step = func(x_step)
                        self.calls += 1
                        local_calls += 1
                        if val_step < best_cand_val:
                            best_cand_val = val_step
                            best_cand_x = x_step.copy()
                    # update local best
                    local_best_x = best_cand_x
                    local_best_val = best_cand_val
                    # update global best
                    if local_best_val < self.best_value:
                        self.best_value = local_best_val
                        self.best_x = local_best_x.copy()
                        report_best(self.best_value, self.best_x)
                    success_counter += 1

                # step-size adaptation
                if local_calls % update_freq == 0:
                    success_rate = success_counter / update_freq
                    if success_rate > 0.2:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.85
                    success_counter = 0

        return (self.best_value, self.best_x)