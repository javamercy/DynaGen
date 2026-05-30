import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        low = func.bounds.lb
        high = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point
        x = np.random.uniform(low, high, size=dim)
        f = func(x)
        best_x = x.copy()
        best_f = f
        report_best(best_f, best_x)
        evals = 1

        if budget <= 1:
            return best_f, best_x

        # Initial step sizes: 20% of range
        step = (high - low) * 0.2
        step = np.maximum(step, 1e-6)

        stagnation = 0
        restart_counter = 0
        restart_interval = max(1, budget // 3)

        while evals < budget:
            improved = False

            # Periodic full restart
            if evals >= restart_counter * restart_interval + restart_interval:
                restart_counter += 1
                x = np.random.uniform(low, high, size=dim)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                step = (high - low) * 0.2
                step = np.maximum(step, 1e-6)
                stagnation = 0
                continue

            # Restart if stagnation too long
            if stagnation > 3 * dim:
                x = np.random.uniform(low, high, size=dim)
                f = func(x)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)
                step = (high - low) * 0.2
                step = np.maximum(step, 1e-6)
                stagnation = 0
                continue

            # Shuffle coordinate order
            for i in np.random.permutation(dim):
                if evals >= budget:
                    break

                # With 20% probability, try random direction
                if np.random.rand() < 0.2:
                    direction = np.random.randn(dim)
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        direction = direction / norm
                    else:
                        direction = np.zeros(dim)
                    step_scaled = step * direction
                    x_new = np.clip(x + step_scaled, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        # pattern step in same direction (if budget allows)
                        if evals < budget:
                            x_pat = np.clip(x + step_scaled, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                        break
                else:
                    # Try positive direction
                    x_new = x.copy()
                    x_new[i] += step[i]
                    x_new = np.clip(x_new, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        if evals < budget:
                            x_pat = x_new.copy()
                            x_pat[i] += step[i]
                            x_pat = np.clip(x_pat, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                        break
                    # Try negative direction
                    x_new = x.copy()
                    x_new[i] -= step[i]
                    x_new = np.clip(x_new, low, high)
                    f_new = func(x_new)
                    evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        report_best(best_f, best_x)
                        x = x_new.copy()
                        improved = True
                        if evals < budget:
                            x_pat = x_new.copy()
                            x_pat[i] -= step[i]
                            x_pat = np.clip(x_pat, low, high)
                            f_pat = func(x_pat)
                            evals += 1
                            if f_pat < best_f:
                                best_f = f_pat
                                best_x = x_pat.copy()
                                report_best(best_f, best_x)
                                x = x_pat.copy()
                        break

            if improved:
                stagnation = 0
            else:
                # Random wide jump with 50% probability
                if np.random.rand() < 0.5:
                    x_jump = np.random.uniform(low, high, size=dim)
                    f_jump = func(x_jump)
                    evals += 1
                    if f_jump < best_f:
                        best_f = f_jump
                        best_x = x_jump.copy()
                        report_best(best_f, best_x)
                        x = x_jump.copy()
                        improved = True
                if not improved:
                    step *= 0.5
                    stagnation += 1

        return best_f, best_x